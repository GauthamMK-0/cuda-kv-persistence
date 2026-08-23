"""Phase 4/4b/6 policy A/B harness.

Configs:
  intra8 / cross8    — Phase-4 originals on the synthetic 8-frame trace.
  dense50 / stream50 — Part-6 ablations on the REAL-motion 50-frame trace
                       (blackswan flow), causal prefix attention; streaming
                       adds sticky arena slots so restage traffic becomes
                       policy-sensitive.

Gates before any metric is reported:
  - outputs bitwise-equal across policies within a config
  - streaming outputs bitwise-equal to dense (immutable rows => same values)
  - max-abs error vs the config's golden reference under tolerance

Metrics: Nsight Compute L2 hit rate + DRAM read bytes over a steady-state
launch window, plus analytic restage bytes (driver-reported).
"""

import csv
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BIN = ROOT / "bin" / "policy_experiment"
SRCS = [ROOT / "src" / "gated_attention_kernel.cu",
        ROOT / "src" / "tile_gating_policy.cpp",
        ROOT / "src" / "kv_tile_manager.cu",
        ROOT / "src" / "run_policy_experiment.cu"]
METRICS = ["lts__t_sector_hit_rate.pct", "dram__bytes_read.sum"]
MAX_ABS_TOL = 1e-3


def nvcc():
    return shutil.which("nvcc") or "/usr/local/cuda/bin/nvcc"


def ncu():
    return shutil.which("ncu") or "/usr/local/cuda/bin/ncu"


def build():
    BIN.parent.mkdir(parents=True, exist_ok=True)
    newest_src = max(s.stat().st_mtime for s in SRCS)
    if BIN.exists() and BIN.stat().st_mtime > newest_src:
        return
    cmd = [nvcc(), "-O3", "-arch=native", "-std=c++17", "-o", str(BIN)] + \
        [str(s) for s in SRCS]
    print("building:", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=ROOT)


def make_cfg(name, mode, attn_mode, policies, meta_name, golden, stem,
             skip=None, count=None):
    meta = json.loads((ROOT / "traces" / meta_name).read_text())
    cfg = json.loads((ROOT / "configs" / "grounding_config.json").read_text())
    return {
        "name": name, "mode": mode, "attn_mode": attn_mode,
        "policies": policies,
        "meta": meta, "golden": golden,
        "q": ROOT / "traces" / f"q_{stem}.bin",
        "k": ROOT / "traces" / f"k_{stem}.bin",
        "v": ROOT / "traces" / f"v_{stem}.bin",
        "motion_csv": ROOT / "traces" / (
            "motion_trace_real.csv" if stem == "50f"
            else f"motion_trace_{stem}.csv"),
        "set_aside": str(cfg["persistence_budget"]["max_persisting_l2_bytes"]),
        "skip": skip, "count": count,
    }


def base_cmd(c, policy, out):
    m = c["meta"]
    cmd = [str(BIN), "--policy", policy, "--attn-mode", c["attn_mode"],
           "--q", str(c["q"]), "--k", str(c["k"]), "--v", str(c["v"]),
           "--motion-csv", str(c["motion_csv"]), "--out", str(out),
           "--frames", str(m["frames"]),
           "--tokens", str(m["model"]["tokens_per_frame"]),
           "--hidden", str(m["model"]["hidden_dim"]),
           "--heads", str(m["model"]["heads"]),
           "--tile-tokens", str(m["model"]["tile_size_tokens"])]
    if policy != "none":
        cmd += ["--set-aside-bytes", c["set_aside"]]
    if c["mode"] == "streaming":
        cmd += ["--streaming"]
    return cmd


def run_plain(c, policy):
    out = ROOT / "build" / f"pol_{c['name']}_{policy}_out.bin"
    proc = subprocess.run(base_cmd(c, policy, out), check=True, cwd=ROOT,
                          capture_output=True, text=True)
    staged = None
    m = re.search(r"staged_bytes=(\d+)", proc.stdout)
    if m:
        staged = int(m.group(1))
    return out, staged


def profile(c, policy):
    out = ROOT / "build" / f"pol_{c['name']}_{policy}_ncu.bin"
    cmd = [ncu(), "--csv", "--target-processes", "all",
           "--kernel-name", "regex:attn_forward_kernel",
           "--metrics", ",".join(METRICS)]
    if c["skip"] is not None:
        cmd += ["--launch-skip", str(c["skip"])]
    if c["count"] is not None:
        cmd += ["--launch-count", str(c["count"])]
    cmd += base_cmd(c, policy, out)
    proc = subprocess.run(cmd, check=True, cwd=ROOT, capture_output=True,
                          text=True)
    lines = [ln for ln in proc.stdout.splitlines()
             if not ln.startswith("==") and ln.strip()]
    start = next((i for i, ln in enumerate(lines)
                  if ln.startswith('"ID"') or ln.startswith("ID,")), None)
    rows = list(csv.DictReader(lines[start:] if start is not None else []))
    unit_mult = {"byte": 1, "Kbyte": 1e3, "Mbyte": 1e6, "Gbyte": 1e9, "%": 1}
    hits, dram = [], []
    for r in rows:
        name = r.get("Metric Name", "").strip()
        val = r.get("Metric Value", "").replace(",", "")
        unit = r.get("Metric Unit", "").strip()
        if not val:
            continue
        v = float(val) * unit_mult.get(unit, 1)
        if name == METRICS[0]:
            hits.append(v)
        elif name == METRICS[1]:
            dram.append(v)
    assert hits and dram, f"no metrics parsed for {policy}/{c['name']}"
    return {"hit_pct_mean": float(np.mean(hits)), "launches": len(hits),
            "dram_read_bytes_sum": float(np.sum(dram))}


def run_config(c):
    pols = c["policies"]
    outs, staged = {}, {}
    for p in pols:
        outs[p], staged[p] = run_plain(c, p)
    arrs = {p: np.fromfile(outs[p], dtype=np.float32).reshape(
        c["meta"]["frames"], 1560, 1536) for p in pols}

    for p in pols:
        assert np.array_equal(arrs[p], arrs["none"]), \
            f"[{c['name']}] {p} differs from none — remap corruption!"
    want = np.load(ROOT / "traces" / c["golden"])["output"]
    worst = max(float(np.abs(arrs[p] - want).max()) for p in pols)
    assert worst < MAX_ABS_TOL, f"[{c['name']}] golden check failed {worst:.3e}"
    print(f"[{c['name']}] gates PASS (bitwise across policies; "
          f"max_abs vs golden {worst:.2e})")

    results = {}
    for p in pols:
        results[p] = profile(c, p)

    base = results["none"]
    print(f"\n=== scoreboard [{c['name']}] ===")
    hdr = f"{'policy':>14} {'L2 hit %':>9} {'DRAM read':>12} {'delta':>8} {'launches':>8}"
    if c["mode"] == "streaming":
        hdr += f" {'restaged MB':>12}"
    print(hdr)
    for p in pols:
        r = results[p]
        delta = ((r["dram_read_bytes_sum"] - base["dram_read_bytes_sum"])
                 / base["dram_read_bytes_sum"] * 100)
        line = (f"{p:>14} {r['hit_pct_mean']:>9.2f} "
                f"{r['dram_read_bytes_sum']:>12.0f} {delta:>7.1f}% "
                f"{r['launches']:>8}")
        if c["mode"] == "streaming":
            line += f" {(staged[p] or 0) / 1e6:>12.2f}"
        print(line)
    return True


def main():
    build()
    eight = dict(mode="dense", attn_mode=None, meta_name="trace_meta_8f.json")
    configs = [
        make_cfg("intra8", "dense", "intra", ("none", "uniform", "motion"),
                 "trace_meta_8f.json", "golden_attention_8f.npz", "8f"),
        make_cfg("cross8", "dense", "cross", ("none", "recency", "motion"),
                 "trace_meta_8f.json", "golden_attention_8f_cross.npz", "8f"),
        make_cfg("dense50", "dense", "cross",
                 ("none", "recency", "motion", "reuse_weighted"),
                 "trace_meta_50f.json", "golden_attention_50f_cross.npz",
                 "50f", skip=34, count=16),
        make_cfg("stream50", "streaming", "cross",
                 ("none", "recency", "motion", "reuse_weighted"),
                 "trace_meta_50f.json", "golden_attention_50f_cross.npz",
                 "50f", skip=34, count=16),
    ]
    # sanity: streaming must produce identical values to dense
    d50 = next(c for c in configs if c["name"] == "dense50")
    s50 = next(c for c in configs if c["name"] == "stream50")

    ok = True
    for c in configs[:2] + configs[2:]:
        ok = run_config(c) and ok

    a = np.fromfile(ROOT / "build" / "pol_dense50_none_out.bin",
                    dtype=np.float32)
    b = np.fromfile(ROOT / "build" / "pol_stream50_none_out.bin",
                    dtype=np.float32)
    assert np.array_equal(a, b), "streaming != dense values!"
    print("\ncross-mode invariant: streaming outputs == dense outputs "
          "(bitwise)")
    return ok


def test_policy_variants():
    assert main(), "policy experiment failed"


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
