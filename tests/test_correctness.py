"""Phase 2 scoreboard: baseline CUDA attention vs golden PyTorch reference.

Pipeline: ensure raw bins exist (export_bins.py) -> build the CUDA binary if
missing -> run it -> compare its output against traces/golden_attention_8f.npz.

Exit criteria (plan Phase 2): scoreboard passes against golden reference.
No performance claims are made here — correctness only.

Runnable standalone or via pytest.
"""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BINS = [ROOT / "traces" / f"{n}_8f.bin" for n in ("q", "k", "v")]
BINARY = ROOT / "bin" / "gated_attention"
OUT_BIN = ROOT / "build" / "attn_baseline_out.bin"
GOLDEN = ROOT / "traces" / "golden_attention_8f.npz"

MAX_ABS_TOL = 1e-3
MEAN_ABS_TOL = 1e-4


def nvcc():
    return shutil.which("nvcc") or "/usr/local/cuda/bin/nvcc"


def ensure_inputs():
    if not all(b.exists() for b in BINS):
        subprocess.run([sys.executable, str(ROOT / "traces" / "export_bins.py")],
                       check=True, cwd=ROOT)


def build_if_needed():
    src = ROOT / "src" / "gated_attention_kernel.cu"
    stale = (not BINARY.exists())
    if BINARY.exists():
        for dep in (src,):
            if dep.stat().st_mtime > BINARY.stat().st_mtime:
                stale = True
    if stale:
        BINARY.parent.mkdir(parents=True, exist_ok=True)
        cmd = [nvcc(), "-O3", "-arch=native", "-std=c++17",
               "-o", str(BINARY), str(src)]
        print("building:", " ".join(cmd))
        subprocess.run(cmd, check=True, cwd=ROOT)


def run_binary(meta):
    cmd = [
        str(BINARY),
        "--q", str(BINS[0]), "--k", str(BINS[1]), "--v", str(BINS[2]),
        "--out", str(OUT_BIN),
        "--frames", str(meta["frames"]),
        "--tokens", str(meta["model"]["tokens_per_frame"]),
        "--hidden", str(meta["model"]["hidden_dim"]),
        "--heads", str(meta["model"]["heads"]),
    ]
    print("running:", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=ROOT)


def score():
    meta = json.loads((ROOT / "traces" / "trace_meta.json").read_text())
    ensure_inputs()
    build_if_needed()
    run_binary(meta)

    frames = meta["frames"]
    t = meta["model"]["tokens_per_frame"]
    d = meta["model"]["hidden_dim"]
    got = np.fromfile(OUT_BIN, dtype=np.float32)
    assert got.size == frames * t * d, (
        f"output size {got.size} != {frames}x{t}x{d}")
    got = got.reshape(frames, t, d)

    want = np.load(GOLDEN)["output"]
    assert want.shape == got.shape, f"golden shape {want.shape} != {got.shape}"

    rows = []
    for f in range(frames):
        diff = np.abs(got[f] - want[f])
        rows.append((f, float(diff.max()), float(diff.mean())))
    overall_max = max(r[1] for r in rows)
    overall_mean = max(r[2] for r in rows)

    print("\n=== Phase 2 scoreboard: baseline kernel vs golden ===")
    print(f"{'frame':>5} {'max_abs_err':>12} {'mean_abs_err':>13}")
    for f, mx, mn in rows:
        print(f"{f:>5} {mx:>12.3e} {mn:>13.3e}")
    passed = overall_max < MAX_ABS_TOL and overall_mean < MEAN_ABS_TOL
    print(f"\noverall: max_abs_err={overall_max:.3e} "
          f"(tol {MAX_ABS_TOL:.0e}), mean_abs_err={overall_mean:.3e} "
          f"(tol {MEAN_ABS_TOL:.0e})")
    print("VERDICT:", "PASS" if passed else "FAIL")
    return passed, rows, overall_max, overall_mean


if __name__ == "__main__":
    ok, *_ = score()
    sys.exit(0 if ok else 1)


def test_scoreboard():
    ok, _, _, _ = score()
    assert ok, "baseline kernel failed golden-reference comparison"
