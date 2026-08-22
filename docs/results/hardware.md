# Target Hardware Inventory

Machine: WSL2 Linux, driver 610.57.01 / CUDA UMD 13.3.
Recorded 2026-08-22 via `nvidia-smi` queries + CUDA runtime attribute calls (`cudaDeviceGetAttribute`).
All numbers below are measured on this machine, not assumed.

## GPU identity

| Property | Value |
|---|---|
| Model | NVIDIA GeForce RTX 3060 Laptop GPU |
| Compute capability | 8.6 (Ampere, GA106) |
| Streaming multiprocessors | 30 |
| Max memory clock | 7001 MHz |

## Memory hierarchy

| Property | Value |
|---|---|
| Total VRAM | 6144 MiB (6,441,926,656 B) |
| **L2 cache (total)** | **3,145,728 B (3.00 MiB)** — `cudaDevAttrL2CacheSize` |
| **Max persisting L2** | **2,162,688 B (~2.06 MiB)** — `cudaDevAttrMaxPersistingL2CacheSize` (68.75% of L2) |
| `accessPolicyMaxWindowSize` | 134,213,632 B (128 MiB) |
| Shared mem / block (opt-in) | 101,376 B |

## Power / TDP (`nvidia-smi -q -d POWER`, driver 610.57.01)

| Property | Value |
|---|---|
| Default power limit | 80.00 W |
| Max power limit | 120.00 W |
| Min power limit | 1.00 W |
| Current ceiling at query time | 116.52 W (dynamic, laptop boost behavior) |
| Idle draw at query time | ~28 W |

Note: laptop dynamic boost raises the effective limit above the 80 W default under load;
the project's perf numbers must be read against this variability (wall-clock is noisy → hence
the Nsight-confirmed L2 hit-rate as headline metric).

## Project-relevant derivations (preview of grounding_config.json)

- K/V bytes per frame per layer: 560 tokens × 1536 dim × 2 B (fp16) × 2 (K,V) = 3,481,600 B ≈ 3.32 MiB
- Per 16-token tile: 16 × 1536 × 2 B × 2 = 98,304 B ≈ 96 KiB
- Tiles fully persistable in max-persisting budget (2,162,688 B): ⌊2,162,688 / 98,304⌋ = **22 tiles**
- Tokens/frame = 560 → tiles/frame at 16 tokens = 35 → a single frame's full KV does NOT fit the
  persisting window; admission control is genuinely required from Phase 3 onward.

Toolchain versions are recorded in Task 0.2 (appended to this file).
