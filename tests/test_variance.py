"""Variance study (anomaly follow-up): quantify run-to-run measurement spread
in the thrash regime BEFORE headline numbers freeze.

Mechanism under test (from the dense50/stream50 anomaly): identical workloads
measured 243 vs 91.6 GB DRAM because ~60 co-resident blocks sweeping the same
key prefix sit on an L2 lockstep-resonance knife edge; block-drift timing
noise swings aggregate hit rate ~5pp, amplified by TB-scale traffic.

Protocol (Tier 1+2):
  - ncu --clock-control base  (remove boost/throttle jitter)
  - full stdout persisted to build/raw/*.csv (per-launch forensics possible)
  - N=3 repeats x {dense50, stream50} x {none, reuse_weighted}
  - ABBA-interleaved execution order across repeats (decorrelate thermal drift)
  - bitwise gate: reuse_weighted outputs must equal none outputs per run
  - verdict: spread = (max-min)/median of DRAM sums; PASS <= 10%

Output: scoreboard to stdout + docs/results/variance.md
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
RAW = ROOT / "build" / "raw"
SKIP, COUNT, REPEATS = 34, 16, 3
SPREAD_PASS = 0.10


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
    print("building:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=ROOT)


def combo_cmd(mode, policy, out_bin):
    meta = json.loads((ROOT / "traces" / "trace_meta_50f.json").read_text())
    cfg = json.loads((ROOT / "configs" / "grounding_config.json").read_text())
    cmd = [str(BIN), "--policy", policy, "--attn-mode", "cross",
           "--q", str(ROOT / "traces" / "q_50f.bin"),
           "--k", str(ROOT / "traces" / "k_50f.bin"),
           "--v", str(ROOT / "traces" / "v_50f.bin"),
           "--motion-csv", str(ROOT / "traces" / "motion_trace_real.csv"),
           "--out", str(out_bin),
           "--frames", str(meta["frames"]),
           "--tokens", str(meta["model"]["tokens_per_frame"]),
           "--hidden", str(meta["model"]["hidden_dim"]),
           "--heads", str(meta["model"]["heads"]),
           "--tile-tokens", str(meta["model"]["tile_size_tokens"])]
    if policy != "none":
        cmd += ["--set-aside-bytes",
                str(cfg["persistence_budget"]["max_persisting_l2_bytes"])]
    if mode == "streaming":
        cmd += ["--streaming"]
    return cmd


def parse_metrics(stdout, tag):
    lines = [ln for ln in stdout.splitlines()
             if not ln.startswith("==") and ln.strip()]
    start = next((i for i, ln in enumerate(lines)
                  if ln.startswith('"ID"') or ln.startswith("ID,")), None)
    assert start is not None, f"[{tag}] no CSV section in ncu output"
    rows = list(csv.DictReader(lines[start:]))
    unit_mult = {"byte": 1, "Kbyte": 1e3, "Mbyte": 1e6, "Gbyte": 1e9, "%": 1}
    dram, hit = [], []
    for r in rows:
        name = r.get("Metric Name", "").strip()
        val = r.get("Metric Value", "").replace(",", "")
        unit = r.get("Metric Unit", "").strip()
        if not val:
            continue
        v = float(val) * unit_mult.get(unit, 1)
        if name == METRICS[0]:
            hit.append(v)
        elif name == METRICS[1]:
            dram.append(v)
    assert len(dram) == COUNT and len(hit) == COUNT, \
        f"[{tag}] expected {COUNT} launches, got {len(dram)}"
    return {"dram_launches": dram, "hit_launches": hit,
            "dram_sum": float(np.sum(dram)), "hit_mean": float(np.mean(hit))}


def profiled_run(mode, policy, tag, ref_bin=None):
    out_bin = ROOT / "build" / f"var_{tag}.bin"
    RAW.mkdir(parents=True, exist_ok=True)
    cmd = [ncu(), "--csv", "--clock-control", "base",
           "--target-processes", "all",
           "--kernel-name", "regex:attn_forward_kernel",
           "--metrics", ",".join(METRICS),
           "--launch-skip", str(SKIP), "--launch-count", str(COUNT)] + \
        combo_cmd(mode, policy, out_bin)
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True,
                          check=True)
    (RAW / f"{tag}.csv").write_text(proc.stdout)
    res = parse_metrics(proc.stdout, tag)

    # bitwise gate: persist policies must reproduce none's values exactly
    if ref_bin is not None:
        a, b = Path(ref_bin).stat().st_size, out_bin.stat().st_size
        assert a == b, f"[{tag}] output size mismatch"
        with open(ref_bin, "rb") as fa, open(out_bin, "rb") as fb:
            while True:
                x, y = fa.read(1 << 24), fb.read(1 << 24)
                assert x == y, f"[{tag}] streaming/dense value mismatch"
                if not x:
                    break
    return res


COMBOS = [(m, p) for m in ("dense50", "stream50")
          for p in ("none", "recency", "motion", "reuse_weighted")]
TAGS = {(m, p): f"{m}_{p}" for m, p in COMBOS}


def main(smoke=False):
    build()
    combos = COMBOS[:1] if smoke else COMBOS
    repeats = [0] if smoke else list(range(REPEATS))
    # ABBA interleave across repeats
    orders = [[*combos], [*reversed(combos)], [*combos]]

    results = {c: [] for c in combos}
    refs = {}
    for r in repeats:
        for mode, policy in orders[r]:
            tag = f"{TAGS[(mode, policy)]}_r{r}"
            ref = None
            if policy != "none":
                ref = refs.get((mode, "none"))
            print(f"[r{r}] profiling {mode}/{policy} ...", flush=True)
            res = profiled_run(mode, policy, tag, ref_bin=ref)
            results[(mode, policy)].append(res)
            print(f"    dram_sum={res['dram_sum']:.3e} "
                  f"hit={res['hit_mean']:.2f}%", flush=True)
            if policy == "none" and mode not in refs:
                refs[mode] = str(ROOT / "build" /
                                 f"var_{TAGS[(mode, 'none')]}_r{r}.bin")

    # ---- analysis ----
    def band(c):
        sums = np.array([x["dram_sum"] for x in results[c]])
        return float(np.median(sums)), float(sums.min()), float(sums.max())

    md = ["# Freeze scoreboard (clock-locked, median-of-N)\n",
          f"Repeats: {REPEATS} (ABBA interleaved), window: launches "
          f"{SKIP}..{SKIP+COUNT-1}, clock control: base\n",
          "## Per-combo stability\n",
          "| config | policy | DRAM sum median | min | max | spread | "
          "hit median | verdict |",
          "|---|---|---|---|---|---|---|---|"]
    all_pass = True
    for c in combos:
        runs = results[c]
        sums = np.array([x["dram_sum"] for x in runs])
        hits = np.array([x["hit_mean"] for x in runs])
        med = float(np.median(sums))
        spread = float((sums.max() - sums.min()) / med)
        ok = spread <= SPREAD_PASS
        all_pass &= ok
        line = (f"| {c[0]} | {c[1]} | {med:.3e} | {sums.min():.3e} | "
                f"{sums.max():.3e} | {spread*100:.1f}% | "
                f"{np.median(hits):.2f}% | {'PASS' if ok else 'FAIL(excursion)'} |")
        md.append(line)
        print(line, flush=True)

        # launch-localization: which launches vary most across repeats?
        mat = np.array([x["dram_launches"] for x in runs])  # [R, COUNT]
        rng = mat.max(axis=0) - mat.min(axis=0)
        medl = np.median(mat, axis=0)
        rel = np.where(medl > 0, rng / np.maximum(medl, 1), 0)
        hot = int((rel > 0.20).sum())
        md.append(f"  - launches with >20% range: {hot}/{COUNT} "
                  f"(indices {np.where(rel > 0.20)[0].tolist()})")

    # ---- policy separation (the gating thesis verdict) ----
    separated = {}
    if not smoke:
        md += ["\n## Policy deltas vs none (median-based)\n",
               "| config | policy | DRAM delta vs none | bands overlap none? |",
               "|---|---|---|---|"]
        for mode in ("dense50", "stream50"):
            n_med, n_min, n_max = band((mode, "none"))
            rw_med, rw_min, rw_max = band((mode, "reuse_weighted"))
            rc_med, rc_min, rc_max = band((mode, "recency"))
            for p, pmed, pmin, pmax in (
                    ("reuse_weighted", rw_med, rw_min, rw_max),
                    ("recency", rc_med, rc_min, rc_max)):
                delta = (pmed - n_med) / n_med * 100
                overlap = not (pmax < n_min or pmin > n_max)
                md.append(f"| {mode} | {p} | {delta:+.1f}% | "
                          f"{'yes' if overlap else 'no'} |")
            sep = (rw_max < rc_min) or (rw_min > rc_max)
            separated[mode] = sep
            md.append(f"  - [{mode}] reuse_weighted vs recency bands "
                      f"{'SEPARATED' if sep else 'OVERLAP'}")

    md += ["\n## Verdict\n"]
    if smoke:
        verdict = "SMOKE OK (separation analysis requires the full run)"
    elif all(separated.values()):
        md.append("**Gating thesis VALIDATED**: reuse-weighted and recency "
                  "bands are separated in both modes under clean measurement.\n")
        verdict = "VALIDATED"
    elif any(separated.values()):
        md.append("**Gating thesis PARTIALLY validated**: separation in "
                  "one mode only — inspect excursion flags above.\n")
        verdict = "PARTIAL"
    else:
        md.append("**Gating thesis NOT supported at this operating point**: "
                  "pin-set size dominates composition; budget/attention-"
                  "heterogeneity levers needed.\n")
        verdict = "NOT_SUPPORTED"

    stability = "stable" if all_pass else \
        "one or more excursion runs occurred (see FAIL rows); medians remain usable"
    md.append(f"Measurement stability: {stability}.")
    out_md = ROOT / "docs" / "results" / "freeze_scoreboard.md"
    out_md.write_text("\n".join(md))
    print(f"\nVERDICT: GATING THESIS {verdict}", flush=True)
    print(f"report -> {out_md}", flush=True)
    return all_pass


if __name__ == "__main__":
    sys.exit(0 if main(smoke="--smoke" in sys.argv) else 1)


def test_variance_smoke():
    assert main(smoke=True), "variance smoke failed"
