"""Golden reference attention (Phase 1).

Plain scaled dot-product multi-head attention in FP32 on CPU, written with
explicit ops (matmul + softmax) rather than F.scaled_dot_product_attention or
any fused kernel. Deliberately independent of every GPU code path that later
phases must validate against this output.

Modes:
  intra (default): frame f attends only to frame f (original Phase-1 math).
  cross:           causal temporal prefix — frame f attends to frames 0..f,
                   i.e. streaming video-DiT inference semantics.

Input : traces/qkv_{F}f.npz   (from gen_synthetic_motion.py)
Output: traces/golden_attention_{F}f[_cross].npz + matching meta json.
"""

import argparse
import json
import math
import sys
from datetime import date
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def attention_reference(q, k, v, num_heads, head_dim):
    """Intra-frame: q,k,v [T, D] -> out [T, D]."""
    t = q.shape[0]
    qh = q.view(t, num_heads, head_dim).transpose(0, 1)  # [H, T, dh]
    kh = k.view(t, num_heads, head_dim).transpose(0, 1)
    vh = v.view(t, num_heads, head_dim).transpose(0, 1)

    scores = torch.matmul(qh, kh.transpose(-1, -2)) / math.sqrt(head_dim)  # [H,T,T]
    attn = torch.softmax(scores, dim=-1)
    ctx = torch.matmul(attn, vh)  # [H, T, dh]

    row_sums = attn.sum(dim=-1)
    max_err = float((row_sums - 1.0).abs().max())
    return ctx.transpose(0, 1).reshape(t, num_heads * head_dim), max_err


def attention_reference_cross(q, ks, vs, num_heads, head_dim, chunk=128):
    """Causal prefix: q [T,D], ks/vs list of [(f+1) x [T,D]] -> [T, D].

    Softmax rows are independent, so processing query-row chunks is exact
    while capping the score matrix at [chunk, S] (~40 MB at S=78k).
    """
    t = q.shape[0]
    k_all = torch.cat(ks, dim=0)  # [(f+1)*T, D]
    v_all = torch.cat(vs, dim=0)
    kh = k_all.view(-1, num_heads, head_dim).transpose(0, 1)  # [H, S, dh] view
    vh = v_all.view(-1, num_heads, head_dim).transpose(0, 1)

    out = torch.empty(t, num_heads * head_dim)
    max_err = 0.0
    scale = 1.0 / math.sqrt(head_dim)
    with torch.no_grad():
        for h in range(num_heads):
            q_h = q.view(t, num_heads, head_dim)[:, h]
            k_h = kh[h]
            v_h = vh[h]
            for c0 in range(0, t, chunk):
                c1 = min(c0 + chunk, t)
                scores = q_h[c0:c1] @ k_h.transpose(0, 1) * scale  # [c, S]
                attn = torch.softmax(scores, dim=-1)
                err = float((attn.sum(dim=-1) - 1.0).abs().max())
                max_err = max(max_err, err)
                out[c0:c1, h * head_dim:(h + 1) * head_dim] = attn @ v_h
                del scores, attn
    return out, max_err


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--qkv", default="traces/qkv_8f.npz")
    p.add_argument("--out-dir", default="traces")
    p.add_argument("--mode", choices=("intra", "cross"), default="intra")
    args = p.parse_args()

    data = np.load(args.qkv)
    q_np, k_np, v_np = data["Q"], data["K"], data["V"]
    num_frames, t, d = q_np.shape

    meta_in = json.loads((Path(args.qkv).parent / "trace_meta.json").read_text())
    heads = meta_in["model"]["heads"]
    head_dim = meta_in["model"]["head_dim"]
    assert heads * head_dim == d

    out = torch.empty(num_frames, t, d)
    max_row_err = 0.0
    frame_stats = []
    with torch.no_grad():
        for f in range(num_frames):
            if args.mode == "cross":
                o_f, err = attention_reference_cross(
                    torch.from_numpy(q_np[f]),
                    [torch.from_numpy(k_np[g]) for g in range(f + 1)],
                    [torch.from_numpy(v_np[g]) for g in range(f + 1)],
                    heads, head_dim)
            else:
                o_f, err = attention_reference(
                    torch.from_numpy(q_np[f]),
                    torch.from_numpy(k_np[f]),
                    torch.from_numpy(v_np[f]),
                    heads, head_dim)
            assert torch.isfinite(o_f).all()
            out[f] = o_f
            max_row_err = max(max_row_err, err)
            frame_stats.append({"frame": f,
                                "mean": round(float(o_f.mean()), 6),
                                "std": round(float(o_f.std()), 6)})
    assert max_row_err < 1e-5, f"softmax row sums deviate by {max_row_err}"

    stem = f"{num_frames}f{'' if args.mode == 'intra' else '_cross'}"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    npz_path = out_dir / f"golden_attention_{stem}.npz"
    np.savez(npz_path, output=out.numpy().astype(np.float32))

    meta = {
        "generated_on": date.today().isoformat(),
        "generator": "traces/gen_golden.py",
        "torch_version": torch.__version__,
        "source_qkv": args.qkv,
        "source_seed": meta_in["seed"],
        "precision": "float32",
        "device": "cpu",
        "mode": args.mode,
        "output_shape": [num_frames, t, d],
        "max_softmax_rowsum_error": max_row_err,
        "per_frame_stats": frame_stats,
    }
    meta_path = out_dir / f"golden_meta{'' if args.mode == 'intra' else '_cross'}.json"
    meta_path.write_text(json.dumps(meta, indent=2))

    print(f"[{args.mode}] golden output shape: {tuple(out.shape)} "
          f"dtype float32 -> {npz_path}")
    print(f"max softmax row-sum error: {max_row_err:.2e}")
    print(f"sample values, output[frame {num_frames - 1}][token 779][:8]: "
          f"{np.array2string(out[-1][779, :8].numpy(), precision=6)}")
    print(f"output stats last frame: mean {out[-1].mean():.6f} "
          f"std {out[-1].std():.6f}")
    print(f"meta -> {meta_path}")


if __name__ == "__main__":
    main()
