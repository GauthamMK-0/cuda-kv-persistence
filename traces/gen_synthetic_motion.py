"""Synthetic trace generator (Phase 1).

Generates per-tile motion scores and the matching Q/K/V token matrices for a
configurable number of frames. Coherence guarantee: each tile's K/V rows drift
across frames in proportion to that tile's motion score, so "static" tiles are
genuinely redundant across frames (high persistence value) and "moving" tiles
genuinely churn. Without this coupling the gating policy would have no real
signal to exploit.

Outputs (in --out-dir, default traces/):
  qkv_{F}f.npz            Q/K/V [F, 1560, 1536] float32 + tile bookkeeping
  motion_trace_{F}f.csv   per-frame per-tile motion scores, human-readable
  trace_meta.json         config echo + validation stats
"""

import argparse
import csv
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import BackboneConfig

STATIC_BASE_MEAN = 0.05
STATIC_BASE_STD = 0.02
STATIC_JITTER_STD = 0.01
MOVING_START_MEAN = 1.5
MOVING_START_STD = 0.4
MOVING_WALK_STD = 0.25
MOVING_CLIP = (0.8, 4.0)
KV_DRIFT_SCALE = 0.15


def build_tiles(cfg: BackboneConfig):
    full = cfg.tokens_per_frame // cfg.tile_size_tokens
    remainder = cfg.tokens_per_frame % cfg.tile_size_tokens
    counts = torch.full((full,), cfg.tile_size_tokens, dtype=torch.long)
    if remainder:
        counts = torch.cat([counts, torch.tensor([remainder])])
    ends = counts.cumsum(0)
    starts = ends - counts
    return starts, counts


def generate_motion_scores(gen, num_tiles, num_frames, frac_static):
    is_static = torch.rand(num_tiles, generator=gen) < frac_static
    static_base = (
        STATIC_BASE_MEAN + STATIC_BASE_STD * torch.randn(num_tiles, generator=gen)
    ).clamp(min=0.005)
    moving_start = (
        MOVING_START_MEAN + MOVING_START_STD * torch.randn(num_tiles, generator=gen)
    ).clamp(*MOVING_CLIP)
    base = torch.where(is_static, static_base, moving_start)

    scores = torch.empty(num_frames, num_tiles)
    scores[0] = base
    for f in range(1, num_frames):
        jitter = STATIC_JITTER_STD * torch.randn(num_tiles, generator=gen)
        walk = MOVING_WALK_STD * torch.randn(num_tiles, generator=gen)
        scores[f] = torch.where(
            is_static,
            (scores[f - 1] + jitter).clamp(min=0.0),
            (scores[f - 1] + walk).clamp(*MOVING_CLIP),
        )
    return scores, is_static


def generate_qkv(gen, cfg: BackboneConfig, scores, counts):
    num_frames, num_tiles = scores.shape
    t, d = cfg.tokens_per_frame, cfg.hidden_dim
    m_tok = torch.repeat_interleave(scores, counts, dim=1)  # [F, T]

    q = torch.randn(num_frames, t, d, generator=gen)
    k = torch.empty(num_frames, t, d)
    v = torch.empty(num_frames, t, d)
    k[0] = torch.randn(t, d, generator=gen)
    v[0] = torch.randn(t, d, generator=gen)
    for f in range(1, num_frames):
        scale = KV_DRIFT_SCALE * m_tok[f].unsqueeze(1)  # [T, 1]
        k[f] = k[f - 1] + scale * torch.randn(t, d, generator=gen)
        v[f] = v[f - 1] + scale * torch.randn(t, d, generator=gen)
    return q, k, v


def similarity_report(x, static_tok_mask):
    cos = F.cosine_similarity(x[0], x[-1], dim=1)
    s = cos[static_tok_mask]
    m = cos[~static_tok_mask]
    return {
        "static": {"mean": round(s.mean().item(), 6), "min": round(s.min().item(), 6)},
        "moving": {"mean": round(m.mean().item(), 6), "max": round(m.max().item(), 6)},
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--frames", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--frac-static", type=float, default=0.65)
    p.add_argument("--out-dir", default="traces")
    p.add_argument("--config", default="configs/grounding_config.json")
    p.add_argument("--motion-csv", default=None,
                   help="real motion trace (gen_motion_from_video.py output):"
                        " drives per-tile drift; reuse rows emit bit-identical"
                        " K/V to the previous frame (streaming skip semantics)")
    p.add_argument("--stem", default=None, help="output stem (default {frames}f)")
    args = p.parse_args()

    cfg = BackboneConfig.from_grounding(args.config)
    gen = torch.Generator().manual_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    starts, counts = build_tiles(cfg)
    num_tiles = int(counts.numel())
    assert int(counts.sum()) == cfg.tokens_per_frame

    if args.motion_csv:
        raw = np.genfromtxt(args.motion_csv, delimiter=",", names=True,
                            dtype=None, encoding="utf-8")
        fids = raw["frame_id"].astype(int)
        tids = raw["tile_id"].astype(int)
        chg = raw["motion_score"].astype(np.float64)
        rls = raw["reuse"].astype(bool)
        frames = int(fids.max()) + 1
        assert frames >= 3 and int(tids.max()) + 1 <= num_tiles
        change = np.zeros((frames, num_tiles))
        reuse = np.zeros((frames, num_tiles), dtype=bool)
        change[fids, tids] = chg
        reuse[fids, tids] = rls
        med = float(np.median(change[1:]))
        lo = float(np.percentile(change[1:], 10))
        hi = float(np.percentile(change[1:], 90))
        # Map measured change rates onto drift scale with real contrast:
        # bottom-decile regions ~0 (genuinely stable), top-decile ~3x.
        scale_tok = np.clip((change - lo) / max(hi - lo, 1e-9), 0.0, 3.0)
        is_static_np = (change < med).mean(axis=0) >= 0.5        # majority-static
        is_static = torch.from_numpy(is_static_np)
        scores = torch.from_numpy(change.astype(np.float32))

        t, d = cfg.tokens_per_frame, cfg.hidden_dim
        drift_rows = torch.from_numpy(scale_tok.astype(np.float32))
        drift_rows = torch.repeat_interleave(drift_rows, counts, dim=1)
        reuse_tok = torch.repeat_interleave(
            torch.from_numpy(reuse), counts, dim=1)

        q = torch.randn(frames, t, d, generator=gen)
        k = torch.empty(frames, t, d)
        v = torch.empty(frames, t, d)
        k[0] = torch.randn(t, d, generator=gen)
        v[0] = torch.randn(t, d, generator=gen)
        for f in range(1, frames):
            step_k = KV_DRIFT_SCALE * drift_rows[f].unsqueeze(1) * \
                torch.randn(t, d, generator=gen)
            step_v = KV_DRIFT_SCALE * drift_rows[f].unsqueeze(1) * \
                torch.randn(t, d, generator=gen)
            frozen = reuse_tok[f].unsqueeze(1)
            k[f] = torch.where(frozen, k[f - 1], k[f - 1] + step_k)
            v[f] = torch.where(frozen, v[f - 1], v[f - 1] + step_v)
    else:
        frames = args.frames
        scores, is_static = generate_motion_scores(gen, num_tiles, frames,
                                                   args.frac_static)
        q, k, v = generate_qkv(gen, cfg, scores, counts)

    assert torch.isfinite(q).all() and torch.isfinite(k).all() \
        and torch.isfinite(v).all()

    is_static_tok = torch.repeat_interleave(is_static, counts)
    report_k = similarity_report(k, is_static_tok)
    report_v = similarity_report(v, is_static_tok)
    if not args.motion_csv:
        # Phase-1 coherence contract is a synthetic-mode guarantee; real
        # footage has no region as quiet as the synthetic static regime.
        assert report_k["static"]["min"] > report_k["moving"]["max"], (
            f"K coherence failed: static {report_k['static']} vs moving {report_k['moving']}"
        )

    stem = args.stem or f"{frames}f"
    npz_path = out_dir / f"qkv_{stem}.npz"
    np.savez(
        npz_path,
        Q=q.numpy().astype(np.float32),
        K=k.numpy().astype(np.float32),
        V=v.numpy().astype(np.float32),
        motion_scores=scores.numpy().astype(np.float32),
        tile_token_starts=starts.numpy().astype(np.int64),
        tile_token_counts=counts.numpy().astype(np.int64),
        is_static=is_static.numpy(),
    )

    csv_path = out_dir / f"motion_trace_{stem}.csv"
    with open(csv_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["frame_id", "tile_id", "motion_score", "is_static",
                    "token_start", "token_count"])
        for f in range(frames):
            for i in range(num_tiles):
                w.writerow([f, i, f"{scores[f, i].item():.6f}",
                            int(is_static[i].item()),
                            int(starts[i].item()), int(counts[i].item())])

    meta = {
        "generated_on": date.today().isoformat(),
        "generator": "traces/gen_synthetic_motion.py",
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "seed": args.seed,
        "frames": frames,
        "frac_static_requested": args.frac_static,
        "real_motion_source": args.motion_csv,
        "regime_params": {
            "static_base_mean": STATIC_BASE_MEAN,
            "static_base_std": STATIC_BASE_STD,
            "static_jitter_std": STATIC_JITTER_STD,
            "moving_start_mean": MOVING_START_MEAN,
            "moving_start_std": MOVING_START_STD,
            "moving_walk_std": MOVING_WALK_STD,
            "moving_clip": list(MOVING_CLIP),
            "kv_drift_scale": KV_DRIFT_SCALE,
        },
        "model": {
            "tokens_per_frame": cfg.tokens_per_frame,
            "latent_grid": [cfg.latent_grid_rows, cfg.latent_grid_cols],
            "tile_size_tokens": cfg.tile_size_tokens,
            "hidden_dim": cfg.hidden_dim,
            "heads": cfg.num_heads,
            "head_dim": cfg.head_dim,
        },
        "tiles": {
            "count": num_tiles,
            "static": int(is_static.sum()),
            "moving": int((~is_static).sum()),
        },
        "files": {
            "qkv_npz": {"path": str(npz_path),
                        "qkv_shape": [frames, cfg.tokens_per_frame, cfg.hidden_dim]},
            "motion_csv": {"path": str(csv_path), "rows": frames * num_tiles},
        },
        "validation_cosine_first_vs_last": {"K": report_k, "V": report_v},
    }
    meta_path = out_dir / f"trace_meta_{stem}.json"
    if stem == "8f":
        # legacy canonical name consumed by the 8-frame regression tests
        (out_dir / "trace_meta.json").write_text(json.dumps(meta, indent=2))
    meta_path.write_text(json.dumps(meta, indent=2))

    print(f"tiles: {num_tiles} ({int(is_static.sum())} static / "
          f"{int((~is_static).sum())} moving)")
    print(f"Q/K/V shape: {tuple(q.shape)} dtype float32 -> {npz_path}")
    print(f"K cross-frame cosine (frame 0 vs last): "
          f"static mean {report_k['static']['mean']}, "
          f"moving mean {report_k['moving']['mean']}")
    print(f"V cross-frame cosine (frame 0 vs last): "
          f"static mean {report_v['static']['mean']}, "
          f"moving mean {report_v['moving']['mean']}")
    print(f"CSV rows: {args.frames * num_tiles} -> {csv_path}")
    print(f"meta -> {meta_path}")


if __name__ == "__main__":
    main()
