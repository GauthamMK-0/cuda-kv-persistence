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

## Toolchain (Task 0.2)

| Component | Version | Verified runnable |
|---|---|---|
| CUDA Toolkit (nvcc) | release 12.9, V12.9.86 (Build cuda_12.9.r12.9) | ✓ compiled sm_86 probe kernel; ran on device, exit 0 |
| Nsight Compute (ncu) | 2025.2.1.0 (build 35987062), public-release | ⚠ launches target app but **cannot read counters** (see below) |
| Nsight Systems (nsys) | 2025.1.3.140-251335620677v0 | installed |
| Driver / CUDA UMD | 610.57.01 / 13.3 | ✓ |

### Known limitation discovered during verification

`ncu --metrics l2_hit_rate <probe>` fails with **`ERR_NVGPUCTRPERM`** — the user does not have
permission to access NVIDIA GPU Performance Counters. Environment is WSL2; on WSL this permission
is controlled on the **Windows host** (NVIDIA Control Panel → Developer menu → *Manage GPU
Performance Counter Permissions* → allow access), not from inside the Linux guest.

**Impact:** does NOT affect Phase 0 L2-persistence validation (`cudaLimitPersistingL2CacheSize`
needs no perf counters) or any correctness work through Phase 4. DOES gate the Phase 5 headline
metric (L2 hit rate / HBM bandwidth via Nsight Compute) until enabled. Tracked as an open issue;
must be resolved before Phase 5 measurements are accepted.

