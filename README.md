# Motion-Gated Tile-Level KV Cache Persistence (CUDA)

A high-performance CUDA systems-level implementation of motion-gated, tile-level Key-Value (KV) cache persistence for Video Diffusion Transformers (DiT), evaluated against the **Wan2.1-1.3B** backbone architecture.

Unlike prior high-level PyTorch masking approaches, this project leverages hardware-level L2 cache pinning via NVIDIA's L2 persistence API (`cudaAccessPropertyPersisting` / `cudaStreamSetAttribute`) combined with dynamic arena compaction and admission control to minimize HBM/DRAM memory traffic during video generation.

---

## Table of Contents

- [Key Results & Findings](#key-results--findings)
- [Architecture & Core Mechanisms](#architecture--core-mechanisms)
  - [Hardware L2 Cache Persistence](#1-hardware-l2-cache-persistence)
  - [Compacted Arena Management (`KVTileManager`)](#2-compacted-arena-management-kvtilemanager)
  - [Indirection-Aware Tiled MHA Kernel](#3-indirection-aware-tiled-mha-kernel)
  - [Tile Gating Policies](#4-tile-gating-policies)
- [Gating Thesis & Determinism Analysis](#gating-thesis--determinism-analysis)
- [Repository Structure](#repository-structure)
- [Hardware & Environment Requirements](#hardware--environment-requirements)
- [Quickstart & Reproduction](#quickstart--reproduction)
  - [1. Environment Setup](#1-environment-setup)
  - [2. Standalone L2 Persistence Smoke Test](#2-standalone-l2-persistence-smoke-test)
  - [3. Running the Test Suite](#3-running-the-test-suite)
  - [4. Running Full Variance & Policy Matrix Benchmarks](#4-running-full-variance--policy-matrix-benchmarks)
  - [5. CLI Experiment Driver](#5-cli-experiment-driver)
- [Configuration & Derivations](#configuration--derivations)
- [Future Work](#future-work)
- [Documentation & Reports](#documentation--reports)

---

## Key Results & Findings

| Metric | Measured Value | Significance / Notes |
|---|---|---|
| **DRAM Read Traffic (Dense Attention)** | **−41.5% to −41.6%** reduction | Hardware L2 cache persistence mechanism validated |
| **DRAM Read Traffic (Cross-Frame Causal)** | **−34.0%** reduction | Confirmed under multi-frame streaming workload |
| **Execution Determinism** | **0.0% variance** (pinned) vs **±156.8%** (unpinned) | Cache pinning eliminates L2 lockstep-resonance thrashing |
| **L2 Persistence Budget** | 2.16 MB (RTX 3060 Laptop, CC 8.6) | 68.75% of 3.15 MB physical L2 (`cudaDevAttrMaxPersistingL2CacheSize`) |
| **Tile Resolution** | 98 tiles / frame (16 tokens / tile) | Matches 30×52 latent grid (1560 tokens / frame) |
| **Max Persisted Capacity** | 22 tiles / layer (~22.4% of frame) | Strictly budget-enforced by `KVTileManager` |
| **Numerical Accuracy** | Bitwise equal across policies, $< 10^{-3}$ max-abs err vs golden | Verified against PyTorch reference outputs |

```
                                    DRAM Read Traffic Reduction
Baseline (None)          [████████████████████████████████████████] 9.11e10 B (100%)
Recency Policy           [███████████████████████                 ] 5.32e10 B (-41.6%)
Motion-Gated Policy      [███████████████████████                 ] 5.32e10 B (-41.6%)
Reuse-Weighted Policy    [███████████████████████                 ] 5.32e10 B (-41.6%)
```

---

## Architecture & Core Mechanisms

```
+-----------------------------------------------------------------------------------------+
|                                    Video Frame Input                                    |
|                      (Latent Grid: 30x52 = 1560 tokens = 98 tiles)                      |
+-----------------------------------------------------------------------------------------+
                                             |
                                             v
+-----------------------------------------------------------------------------------------+
|                              Optical Flow / Motion Scoring                              |
|           (Farneback dense flow -> tile-level motion scores -> TileGatingPolicy)        |
+-----------------------------------------------------------------------------------------+
                                             |
                   +-------------------------+-------------------------+
                   | (Propose candidate tile IDs)                      |
                   v                                                   v
+--------------------------------------+             +----------------------------------+
|          TileGatingPolicy            |             |          KVTileManager           |
|  - None (streaming control)          |             |  - Enforces 2.16 MB L2 budget    |
|  - Uniform Window (spatial prefix)   |             |  - Compacts into contiguous arena|
|  - Recency Window (temporal newest)  |             |  - Applies persisting L2 window  |
|  - Motion-Gated (low-motion first)   |             |  - Manages sticky slots / restage|
|  - Reuse-Weighted (lifespan x motion)|             |  - Emits KVMap row indirection   |
+--------------------------------------+             +----------------------------------+
                   |                                                   |
                   +-------------------------+-------------------------+
                                             |
                                             v
+-----------------------------------------------------------------------------------------+
|                           Remapped Tiled MHA CUDA Kernel                                |
|                               (`attn_forward_kernel`)                                   |
|  - Queries unpinned KV from DRAM or pinned KV from compacted L2 persistence Arena       |
|  - Seamlessly resolves row pointers via KVMap indirection table                         |
|  - Supports intra-frame and causal temporal cross-frame attention                       |
+-----------------------------------------------------------------------------------------+
```

### 1. Hardware L2 Cache Persistence
NVIDIA Ampere and newer architectures allow reserving a portion of the L2 cache for persistent lines (`cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, budget)`). When memory addresses within an active `cudaAccessPolicyWindow` are accessed with `cudaAccessPropertyPersisting` on a CUDA stream, the cache controller biases eviction against these lines, keeping them resident across kernel launches.

### 2. Compacted Arena Management (`KVTileManager`)
- **Single-Window Constraint Solution**: A CUDA stream supports only **one** active access policy window at any time. Applying windows to scattered tile pointers causes subsequent windows to overwrite preceding ones. `KVTileManager` solves this by compacting admitted K and V tiles into a single contiguous device staging buffer (`Arena`) and applying a single persistence window over the allocated span.
- **Strict Admission Control**: Hard budget checks ensure the arena footprint never exceeds `max_persisting_l2_bytes` (2,162,688 bytes = 22 tiles), avoiding cache thrash-demote cycles.
- **Sticky Slot Streaming**: In streaming mode, tiles that remain valid across frames persist in the arena without re-copying, turning static tile hits into zero-copy metadata lookups.
- **Per-Layer Lifecycle**: The persistence window and arena are re-established per transformer layer, matching sequential feedforward execution.

### 3. Indirection-Aware Tiled MHA Kernel
The CUDA attention kernel (`attn_forward_kernel` in `src/gated_attention_kernel.cu`) evaluates multi-head attention with online softmax:
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V$$
- Reads queries ($Q$) from normal device memory.
- Reads keys ($K$) and values ($V$) either from base DRAM buffers or the persisting `Arena` via an indirection lookup array (`KVMap`).
- Transparently supports intra-frame attention ($T$ tokens) and cross-frame causal attention with expanding temporal prefix ($(f + 1) \times T$ tokens).
- Guaranteed bitwise output equivalence with unremapped baselines.

### 4. Tile Gating Policies
Located in `src/tile_gating_policy.h` / `src/tile_gating_policy.cpp`:
1. **`NoPersistencePolicy` (`none`)**: Baseline unpinned execution with identity mapping.
2. **`UniformWindowPolicy` (`uniform`)**: Static leading spatial tile IDs.
3. **`RecencyWindowPolicy` (`recency`)**: Selects the most recent frame tokens first in temporal causal attention.
4. **`MotionGatedPolicy` (`motion`)**: Ranks tiles by motion score ascending (low optical flow $\rightarrow$ high redundancy across frames $\rightarrow$ highest persistence priority).
5. **`ReuseWeightedPolicy` (`reuse_weighted`)**: Priority scored by $\text{remaining sweeps} \times (1 - \text{normalized motion score})$, accounting for temporal query lifespans.

---

## Gating Thesis & Determinism Analysis

### The Determinism Discovery
When profiling unpinned multi-frame attention under dense traffic, the memory subsystem exhibited massive run-to-run DRAM traffic jitter (**±156.8% variance**, swinging between 91.1 GB and 233.8 GB).
- **Root Cause**: ~60 co-resident threadblocks across 30 SMs simultaneously sweeping identical key prefixes sit on an L2 lockstep-resonance cliff. Slight phase shifts in block execution create chaotic thrashing.
- **The Persistence Solution**: Pinning admitted tiles via `cudaAccessPropertyPersisting` locks the resident lines into L2, collapsing run-to-run DRAM variance to **0.0%**. Persistence provides **execution determinism**, not just average bandwidth reduction.

### Why Pin-Set Size Dominates Composition in Dense Attention
In benchmark experiments at a 22.4% budget under full dense attention:
- All persistent policies (`recency`, `motion`, `reuse_weighted`) achieve an identical **−41.5% DRAM reduction**.
- **Explanation**: In dense self-attention, every query token attends to every key token with equal frequency. When access frequency is uniform across all spatial tiles, the choice of *which* tiles to persist does not change total memory traffic—only the *volume* of persisted data matters.
- **Avenues for Policy Differentiation**:
  1. **Banded / Local Attention**: Restricting receptive fields introduces spatial access heterogeneity.
  2. **Real Model Attention Density**: Sparse attention weight distributions in trained diffusion models.
  3. **Layer-wise Skip-Recompute**: Skipping QKV projections entirely for persistent static tiles.

---

## Repository Structure

```
.
├── configs/
│   └── grounding_config.json        # Grounding hardware attributes & budget derivations
├── docs/
│   └── results/
│       ├── final_report.md          # Comprehensive experimental report & conclusions
│       ├── freeze_scoreboard.md     # Clock-locked median-of-N benchmark scoreboard
│       ├── hardware.md              # Target GPU profile (RTX 3060 Laptop, CC 8.6)
│       └── variance.md              # Run-to-run variance & L2 thrash analysis
├── scripts/
│   ├── run_l2_smoke.sh              # Standalone L2 persistence smoke test runner
│   └── setup_env.sh                 # Environment & PyTorch/CUDA setup script
├── src/
│   ├── attn_kernel.h                # Attention kernel declarations & constants
│   ├── backbone.py                  # PyTorch reference DiT backbone (Wan2.1-1.3B)
│   ├── config.py                    # Python backbone configuration dataclasses
│   ├── gated_attention_kernel.cu    # CUDA tiled MHA kernel + KVMap indirection
│   ├── kv_tile_manager.cu/.h        # L2 persistence manager, arena, admission control
│   ├── l2_persist_smoke.cu          # Minimal device L2 persistence verification
│   ├── run_policy_experiment.cu     # End-to-end benchmark & profiling binary
│   ├── tile_gating_policy.cpp/.h    # Swappable tile gating policy implementations
│   └── tile_layout.h                # Spatial tile layout arithmetic (30x52 -> 98 tiles)
├── tests/
│   ├── conftest.py                  # Pytest configuration
│   ├── test_capacity_admission.py   # Budget enforcement & slot lifecycle unit tests
│   ├── test_correctness.py          # CUDA MHA vs golden PyTorch numerical validation
│   ├── test_policy_variants.py      # Multi-policy matrix benchmark harness
│   ├── test_skeleton.py             # DiT model topology & shape invariant tests
│   └── test_variance.py             # Clock-locked variance & stability study (NCU)
└── traces/
    ├── export_bins.py               # Raw binary tensor exporter (Q, K, V)
    ├── fetch_clip.sh                # Video clip download helper for optical flow
    ├── gen_golden.py                # CPU/PyTorch golden reference generator
    ├── gen_motion_from_video.py     # OpenCV Farneback dense optical flow generator
    └── gen_synthetic_motion.py      # Synthetic motion trace generator
```

---

## Hardware & Environment Requirements

### Target Hardware
- **NVIDIA GPU**: Compute Capability $\ge 8.0$ (Ampere, Ada Lovelace, Hopper, or Blackwell).
- *Tested On*: NVIDIA GeForce RTX 3060 Laptop GPU (Ampere GA106, CC 8.6, 30 SMs, 3.15 MB L2 cache, 2.16 MB max persisting L2).

### Software Requirements
- **OS**: Linux or WSL2 (Ubuntu 22.04+ recommended)
- **CUDA Toolkit**: 12.0+ / 13.x (with `nvcc`)
- **Python**: 3.10+
- **Profiler (Optional for profiling benchmarks)**: NVIDIA Nsight Compute (`ncu`)

> **Note for WSL2 users**: To allow `ncu` to read hardware performance counters, enable GPU performance counter permissions in the Windows NVIDIA Control Panel (*Developer* $\rightarrow$ *Manage GPU Performance Counter Permissions* $\rightarrow$ *Allow access*).

---

## Quickstart & Reproduction

### 1. Environment Setup
Clone the repository and run the setup script to create a virtual environment and install PyTorch with CUDA support:

```bash
bash scripts/setup_env.sh
source .venv/bin/activate
```

### 2. Standalone L2 Persistence Smoke Test
Verify that your GPU supports the CUDA L2 persistence API:

```bash
bash scripts/run_l2_smoke.sh
```

Expected output:
```text
cudaLimitPersistingL2CacheSize set to 2162688 B (readback: 2162688 B)
allocating 16.00 MiB array
persisting-window launch done
demote-to-normal launch done
PASS: L2 persistence API smoke test succeeded
```

### 3. Running the Test Suite
Execute the full pytest suite (covering PyTorch model invariants, CUDA kernel correctness vs golden references, admission control, and smoke profiling):

```bash
pytest tests/ -v
```

To run individual test targets:
```bash
# Verify bitwise and numerical accuracy vs PyTorch golden references
pytest tests/test_correctness.py -v

# Verify exact budget capacity, refusal upon overflow, and slot lifecycle
pytest tests/test_capacity_admission.py -v

# Verify DiT model structure and parameter shapes
pytest tests/test_skeleton.py -v
```

### 4. Running Full Variance & Policy Matrix Benchmarks
To run the clock-locked variance study (interleaved ABBA repetitions with Nsight Compute):

```bash
# Quick smoke profiling (single repetition)
python tests/test_variance.py --smoke

# Full variance study (N=3 repeats, clock-controlled)
python tests/test_variance.py
```

To run the complete policy matrix across dense and streaming modes:

```bash
python tests/test_policy_variants.py
```

### 5. CLI Experiment Driver
You can build and run the C++/CUDA experiment driver directly:

```bash
# Build binary
nvcc -O3 -arch=native -std=c++17 -o bin/policy_experiment \
  src/gated_attention_kernel.cu \
  src/tile_gating_policy.cpp \
  src/kv_tile_manager.cu \
  src/run_policy_experiment.cu

# Run with motion policy on 50-frame cross-attention trace
./bin/policy_experiment \
  --policy motion \
  --attn-mode cross \
  --q traces/q_50f.bin \
  --k traces/k_50f.bin \
  --v traces/v_50f.bin \
  --motion-csv traces/motion_trace_real.csv \
  --out build/pol_motion_out.bin \
  --frames 50 \
  --tokens 1560 \
  --hidden 1536 \
  --heads 12 \
  --tile-tokens 16 \
  --set-aside-bytes 2162688 \
  --streaming
```

---

## Configuration & Derivations

All model parameters and GPU hardware constants are consolidated in `configs/grounding_config.json`:

```json
{
  "gpu": {
    "compute_capability": "8.6",
    "l2_cache_bytes": 3145728,
    "max_persisting_l2_bytes": 2162688
  },
  "model": {
    "class": "Wan2.1-1.3B-shaped backbone",
    "layers": 30,
    "hidden_dim": 1536,
    "heads": 12,
    "head_dim": 128,
    "tokens_per_frame": 1560,
    "latent_grid": [30, 52],
    "tile_size_tokens": 16
  },
  "kv_derivation": {
    "kv_bytes_per_token_per_layer": 6144,
    "kv_bytes_per_tile_per_layer": 98304,
    "tiles_per_frame": 98,
    "tiles_fully_persistable": 22
  }
}
```

### Key Mathematical Derivations
- **KV Bytes per Token per Layer**:
  $$\text{Dim} \times 2 \,(\text{FP16}) \times 2 \,(K \text{ and } V) = 1536 \times 2 \times 2 = 6,144 \text{ Bytes}$$
- **KV Bytes per 16-Token Tile**:
  $$16 \times 6,144 \text{ B} = 98,304 \text{ Bytes} \ (96 \text{ KiB})$$
- **Total Tiles per Frame**:
  $$\frac{1560 \text{ tokens}}{16 \text{ tokens/tile}} = 97.5 \longrightarrow 97 \text{ full tiles} + 1 \text{ partial tile (8 tokens)} = 98 \text{ tiles}$$
- **Persistable Tile Budget**:
  $$\left\lfloor \frac{2,162,688 \text{ Bytes}}{98,304 \text{ Bytes/tile}} \right\rfloor = 22 \text{ tiles per layer } (\approx 22.4\% \text{ of frame})$$

---

## Future Work

1. **Banded / Local Attention Kernels**: Implement sliding-window and block-sparse attention patterns to create access heterogeneity where motion gating delivers differentiated bandwidth savings.
2. **Pretrained Weight Integration**: Profile attention density maps using real weights from Wan2.1-1.3B / Wan2.1-14B checkpoints.
3. **Layer-Wise Skip-Recompute**: Bypass QKV projection math and memory writes for tiles identified as static across successive video frames.
4. **Anti-Pinning Volatile Tiles**: Direct stream property (`cudaAccessPropertyStreaming`) for high-churn tiles to accelerate eviction and protect static cache lines.

---

## Documentation & Reports

Detailed technical documentation and profiling reports are available in the `docs/` directory:
- [Final Engineering Report](docs/results/final_report.md) — In-depth executive summary, methodology, and conclusions.
- [Freeze Scoreboard](docs/results/freeze_scoreboard.md) — Full median-of-N benchmark records across configs and policies.
- [Hardware Inventory](docs/results/hardware.md) — GPU specifications, cache hierarchy measurements, and power constraints.
- [Variance Study](docs/results/variance.md) — In-depth analysis of L2 lockstep resonance and cache thrash dynamics.
