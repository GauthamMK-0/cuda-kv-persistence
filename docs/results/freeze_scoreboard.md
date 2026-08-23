# Freeze scoreboard (clock-locked, median-of-N)

Repeats: 3 (ABBA interleaved), window: launches 34..49, clock control: base

## Per-combo stability

| config | policy | DRAM sum median | min | max | spread | hit median | verdict |
|---|---|---|---|---|---|---|---|
| dense50 | none | 9.112e+10 | 9.112e+10 | 9.112e+10 | 0.0% | 96.44% | PASS |
  - launches with >20% range: 0/16 (indices [])

## Verdict

Measurement stability: stable.