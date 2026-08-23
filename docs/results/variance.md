# Variance study

Repeats: 3 (ABBA interleaved), window: launches 34..49, clock control: base

| config | policy | DRAM sum median | min | max | spread | hit median | verdict |
|---|---|---|---|---|---|---|---|
| dense50 | none | 9.105e+10 | 8.929e+10 | 9.114e+10 | 2.0% | 96.45% | PASS |
  - launches with >20% range: 0/16 (indices [])
  - per-launch relative range: [0.02 0.04 0.   0.01 0.   0.   0.   0.06 0.01 0.19 0.01 0.01 0.01 0.
 0.   0.  ]
| stream50 | none | 9.111e+10 | 9.103e+10 | 2.338e+11 | 156.8% | 96.44% | FAIL |
  - launches with >20% range: 13/16 (indices [1, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
  - per-launch relative range: [0.02 0.63 0.02 0.22 0.78 0.02 0.66 1.79 1.62 3.28 1.77 3.04 2.59 1.6
 2.92 2.5 ]
| dense50 | reuse_weighted | 5.324e+10 | 5.324e+10 | 5.324e+10 | 0.0% | 96.31% | PASS |
  - launches with >20% range: 0/16 (indices [])
  - per-launch relative range: [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
| stream50 | reuse_weighted | 5.324e+10 | 5.323e+10 | 5.324e+10 | 0.0% | 96.47% | PASS |
  - launches with >20% range: 0/16 (indices [])
  - per-launch relative range: [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]

## Verdict

locked-clock spread across repeats: **at least one FAIL** (threshold 10%). FAIL -> Tier-3 deterministic-sweep kernel becomes justified
