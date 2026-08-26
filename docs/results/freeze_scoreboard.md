# Freeze scoreboard (clock-locked, median-of-N)

Repeats: 3 (ABBA interleaved), window: launches 34..49, clock control: base

## Per-combo stability

| config | policy | DRAM sum median | min | max | spread | hit median | verdict |
|---|---|---|---|---|---|---|---|
| dense50 | none | 9.112e+10 | 9.110e+10 | 9.115e+10 | 0.1% | 96.44% | PASS |
  - launches with >20% range: 0/16 (indices [])
| dense50 | recency | 5.324e+10 | 5.322e+10 | 5.324e+10 | 0.0% | 96.39% | PASS |
  - launches with >20% range: 0/16 (indices [])
| dense50 | motion | 5.325e+10 | 5.323e+10 | 5.325e+10 | 0.0% | 96.04% | PASS |
  - launches with >20% range: 0/16 (indices [])
| dense50 | reuse_weighted | 5.324e+10 | 5.324e+10 | 5.325e+10 | 0.0% | 96.07% | PASS |
  - launches with >20% range: 0/16 (indices [])
| stream50 | none | 9.104e+10 | 9.101e+10 | 9.110e+10 | 0.1% | 96.44% | PASS |
  - launches with >20% range: 0/16 (indices [])
| stream50 | recency | 5.324e+10 | 5.323e+10 | 5.324e+10 | 0.0% | 96.31% | PASS |
  - launches with >20% range: 0/16 (indices [])
| stream50 | motion | 5.324e+10 | 5.324e+10 | 5.325e+10 | 0.0% | 95.90% | PASS |
  - launches with >20% range: 0/16 (indices [])
| stream50 | reuse_weighted | 5.325e+10 | 5.324e+10 | 5.325e+10 | 0.0% | 95.96% | PASS |
  - launches with >20% range: 0/16 (indices [])

## Policy deltas vs none (median-based)

| config | policy | DRAM delta vs none | bands overlap none? |
|---|---|---|---|
| dense50 | reuse_weighted | -41.6% | no |
| dense50 | recency | -41.6% | no |
  - [dense50] reuse_weighted vs recency bands OVERLAP
| stream50 | reuse_weighted | -41.5% | no |
| stream50 | recency | -41.5% | no |
  - [stream50] reuse_weighted vs recency bands OVERLAP

## Verdict

**Gating thesis NOT supported at this operating point**: pin-set size dominates composition; budget/attention-heterogeneity levers needed.

Measurement stability: stable.