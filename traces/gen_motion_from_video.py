"""Real-motion trace generator (Part 5).

Farneback dense optical flow between consecutive video frames -> per-pixel
magnitude -> aggregated over the exact 16x16-px latent-grid cells (30x52 grid
on 480x832 frames) -> per-tile scores over row-major 16-token tiles.

Row semantics (frame f >= 1):
  change_rate(f,t) = mean flow magnitude of tile t between frames f-1 -> f,
                     i.e. how much content CHANGED to produce this frame.
  motion_score     = change_rate (also used by policies for ranking).
  reuse            = 1 if change_rate < tau (content ~unchanged vs previous
                     frame -> streaming writer can skip rewriting it).
Frame 0: written fresh by definition (reuse=0); its score borrows step-1
change rates so ranking sees a real signal everywhere.

Outputs (in --out-dir, default traces/):
  motion_trace_real.csv    driver-compatible columns + reuse flag
  trace_real_meta.json     provenance, calibration, validation stats
  docs/results/motion_heatmap_<stem>.png  median cell-magnitude map
"""

import argparse
import csv
import json
import sys
from datetime import date
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import BackboneConfig


def load_gray_frames(source: Path, cap):
    if source.is_dir():
        paths = sorted(list(source.glob("*.jpg")) + list(source.glob("*.png")))
        raws = []
        for p in paths[:cap]:
            img = cv2.imread(str(p))
            assert img is not None, f"unreadable frame {p}"
            raws.append(img)
    else:
        capv = cv2.VideoCapture(str(source))
        raws = []
        while len(raws) < cap:
            ok, img = capv.read()
            if not ok:
                break
            raws.append(img)
        capv.release()
    return [cv2.cvtColor(cv2.resize(im, (0, 0), fx=1, fy=1),
                         cv2.COLOR_BGR2GRAY) for im in raws]


def resize_gray(gray, width, height):
    r = cv2.resize(gray, (width, height), interpolation=cv2.INTER_AREA)
    return r


def cell_map(mag, rows, cols):
    h, w = mag.shape
    ch, cw = h // rows, w // cols
    assert ch * rows == h and cw * cols == w, \
        f"grid {rows}x{cols} does not divide {w}x{h}"
    return mag[:rows * ch, :cols * cw].reshape(rows, ch, cols, cw)\
                                         .mean(axis=(1, 3))  # [rows, cols]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", default=None,
                   help="video dir/file; default auto-detect under traces/video")
    p.add_argument("--frames", type=int, default=128, help="max frames to use")
    p.add_argument("--reuse-percentile", type=float, default=50.0,
                   help="percentile of change-rate used as reuse threshold")
    p.add_argument("--out-dir", default="traces")
    p.add_argument("--config", default="configs/grounding_config.json")
    args = p.parse_args()

    cfg = BackboneConfig.from_grounding(args.config)
    rows_g, cols_g = cfg.latent_grid_rows, cfg.latent_grid_cols
    tt = cfg.tile_size_tokens
    num_tiles = cfg.tokens_per_frame // tt + bool(cfg.tokens_per_frame % tt)
    width, height = 832, 480

    video_root = Path(__file__).resolve().parent / "video"
    source = Path(args.source) if args.source else (
        video_root / "input.mp4" if (video_root / "input.mp4").exists()
        else video_root / "DAVIS" / "JPEGImages" / "480p" / "blackswan")
    assert source.exists(), f"no clip at {source}"

    grays = [resize_gray(g, width, height)
             for g in load_gray_frames(source, args.frames)]
    n = len(grays)
    assert n >= 3, f"need >=3 frames, got {n}"
    print(f"source: {source} -> {n} frames @ {width}x{height}")

    # change_rate[f][tile] for f>=1 from Farneback; index 0 mirrors step 1
    change = np.zeros((n, num_tiles), dtype=np.float64)
    cell_maps = np.zeros((n, rows_g, cols_g))
    farneback = dict(pyr_scale=0.5, levels=3, winsize=15, iterations=3,
                     poly_n=5, poly_sigma=1.2, flags=0)
    prev = grays[0]
    for f in range(1, n):
        flow = cv2.calcOpticalFlowFarneback(prev, grays[f], None, **farneback)
        mag = np.linalg.norm(flow, axis=2)
        cm = cell_map(mag, rows_g, cols_g)
        cell_maps[f] = cm
        flat = cm.reshape(-1)
        counts = np.full(num_tiles, tt)
        counts[-1] = cfg.tokens_per_frame - (num_tiles - 1) * tt
        sums = np.add.reduceat(flat, np.arange(0, num_tiles * tt, tt))
        change[f] = sums / counts
        prev = grays[f]
    change[0] = change[1]  # frame 0 ranking signal (documented approximation)

    tau = float(np.percentile(change[1:], args.reuse_percentile))
    reuse = (change < tau).astype(np.int64)
    reuse[0] = 0  # initial frame is always written fresh

    # spatial coherence: adjacent-cell correlation vs shuffled control
    med = np.median(cell_maps[1:], axis=0)
    horiz = med[:, :-1].ravel(), med[:, 1:].ravel()
    vert = med[:-1, :].ravel(), med[1:, :].ravel()
    corr_h = float(np.corrcoef(horiz[0], horiz[1])[0, 1])
    corr_v = float(np.corrcoef(vert[0], vert[1])[0, 1])
    rng = np.random.default_rng(7)
    sh = med.reshape(-1).copy()
    rng.shuffle(sh)
    sh = sh.reshape(rows_g, cols_g)
    corr_ctrl = float(np.corrcoef(sh[:, :-1].ravel(), sh[:, 1:].ravel())[0, 1])

    out_dir = Path(args.out_dir)
    stem = source.name
    csv_path = out_dir / "motion_trace_real.csv"
    starts = np.arange(num_tiles) * tt
    with open(csv_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["frame_id", "tile_id", "motion_score", "is_static",
                    "token_start", "token_count", "reuse"])
        for f in range(n):
            for t in range(num_tiles):
                w.writerow([f, t, f"{change[f, t]:.6f}",
                            int(change[f, t] < tau),
                            int(starts[t]),
                            int(counts[t]), int(reuse[f, t])])

    heat_dir = Path(__file__).resolve().parent.parent / "docs" / "results"
    heat_dir.mkdir(parents=True, exist_ok=True)
    norm = np.clip(med / max(med.max(), 1e-9) * 255, 0, 255).astype(np.uint8)
    big = cv2.resize(norm, (cols_g * 16, rows_g * 16),
                     interpolation=cv2.INTER_NEAREST)
    colored = cv2.applyColorMap(big, cv2.COLORMAP_JET)
    heat_path = heat_dir / f"motion_heatmap_{stem}.png"
    cv2.imwrite(str(heat_path), colored)

    meta = {
        "generated_on": date.today().isoformat(),
        "generator": "traces/gen_motion_from_video.py",
        "source": str(source), "frames": n, "size": [width, height],
        "cell_px": [height // rows_g, width // cols_g],
        "grid": [rows_g, cols_g], "tile_size_tokens": tt,
        "num_tiles": int(num_tiles),
        "farneback": farneback,
        "tau_reuse": round(tau, 6),
        "reuse_percentile_target": args.reuse_percentile,
        "reuse_fraction_achieved": round(float(reuse[1:].mean()), 4),
        "change_rate_percentiles": {
            q: round(float(np.percentile(change[1:], q)), 4)
            for q in (10, 25, 50, 75, 90)},
        "spatial_coherence_adjacent_cell_corr": {
            "horizontal": round(corr_h, 4), "vertical": round(corr_v, 4),
            "shuffled_control": round(corr_ctrl, 4)},
        "files": {"csv": str(csv_path), "heatmap": str(heat_path)},
    }
    meta_path = out_dir / "trace_real_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2))

    print(f"tiles/frame: {num_tiles}  rows written: {n * num_tiles}")
    print(f"change-rate percentiles (px/frame): "
          f"{meta['change_rate_percentiles']}")
    print(f"tau_reuse={tau:.4f} -> reuse fraction "
          f"{meta['reuse_fraction_achieved']} (target "
          f"{args.reuse_percentile}%)")
    print(f"adjacent-cell corr: h={corr_h:.3f} v={corr_v:.3f} "
          f"(shuffled control {corr_ctrl:.3f})")
    print(f"heatmap -> {heat_path}")
    print(f"csv -> {csv_path}  meta -> {meta_path}")


if __name__ == "__main__":
    main()
