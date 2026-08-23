"""Phase 3 capacity/admission test: KVTileManager refuses to exceed budget.

Plan Section 3.5: deliberately overflow the budget and verify the manager
refuses/evicts correctly rather than silently overflowing. Exit criteria
(plan Phase 3): accounting is exact — 22 tiles of 98,304 B fill the
2,162,688 B set-aside, tile #23 is refused, release frees room, duplicate
admits never double-count, clear() resets everything.

All budget numbers come from configs/grounding_config.json (single source of
truth). Windows are aimed at real device memory via a scratch allocation.
"""

import ctypes
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
LIB = ROOT / "bin" / "libkv_tile_manager.so"
SRC = [ROOT / "src" / "kv_tile_manager.cu"]

OK, REFUSED, INVALID = 1, 0, -1


def nvcc():
    return __import__("shutil").which("nvcc") or "/usr/local/cuda/bin/nvcc"


def build_if_needed():
    src = SRC[0]
    if LIB.exists() and LIB.stat().st_mtime > src.stat().st_mtime:
        return
    LIB.parent.mkdir(parents=True, exist_ok=True)
    cmd = [nvcc(), "-O3", "-arch=native", "-std=c++17", "-shared",
           "-Xcompiler", "-fPIC", "-o", str(LIB), str(src)]
    print("building:", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=ROOT)


class Kvtm:
    def __init__(self):
        build_if_needed()
        self.lib = ctypes.CDLL(str(LIB))
        self.lib.kvtm_create.argtypes = [ctypes.c_long, ctypes.c_long]
        self.lib.kvtm_create.restype = ctypes.c_void_p
        self.lib.kvtm_admit.argtypes = [
            ctypes.c_void_p, ctypes.c_long, ctypes.c_void_p, ctypes.c_long]
        for name in ("kvtm_init", "kvtm_release", "kvtm_clear",
                     "kvtm_destroy", "kvtm_dev_free"):
            getattr(self.lib, name).argtypes = [ctypes.c_void_p]
        self.lib.kvtm_query_limit.argtypes = [ctypes.c_void_p]
        self.lib.kvtm_persisted_bytes.argtypes = [ctypes.c_void_p]
        self.lib.kvtm_persisted_tiles.argtypes = [ctypes.c_void_p]

    def create(self, budget, max_window):
        h = self.lib.kvtm_create(budget, max_window)
        assert h, "kvtm_create failed"
        return h


def scenario(cfg):
    lib_holder = Kvtm()
    lib = lib_holder.lib
    budget = cfg["persistence_budget"]["max_persisting_l2_bytes"]
    tile_bytes = cfg["persistence_budget"]["kv_tile_bytes"]
    max_window = cfg["gpu"]["access_policy_max_window_bytes"]
    expected_tiles = cfg["persistence_budget"]["tiles_fully_persistable"]
    tpf = cfg["model"]["tokens_per_frame"]
    tt = cfg["model"]["tile_size_tokens"]

    print(f"config: budget={budget} B, tile={tile_bytes} B, "
          f"expected full tiles={expected_tiles}")
    assert budget % tile_bytes == 0, "budget not an exact tile multiple?!"
    assert budget // tile_bytes == expected_tiles

    m = lib_holder.create(budget, max_window)
    try:
        # S1: init + device read-back
        assert lib.kvtm_init(m) == 1
        readback = lib.kvtm_query_limit(m)
        print(f"S1 init: SetLimit ok, device read-back={readback} B")
        assert readback == budget, (
            f"device carve-out {readback} != config budget {budget}")

        # scratch device buffer to aim windows at real addresses
        scratch = lib.kvtm_dev_alloc(budget + 4 * tile_bytes)
        assert scratch, "scratch alloc failed"
        ptr_int = ctypes.cast(scratch, ctypes.c_void_p).value

        def admit(tile_id):
            return lib.kvtm_admit(m, tile_id,
                                  ctypes.c_void_p(ptr_int + tile_id * tile_bytes),
                                  tile_bytes)

        # S2: exact fill
        results = [admit(i) for i in range(expected_tiles)]
        assert all(r == OK for r in results), f"admit sequence: {results}"
        assert lib.kvtm_persisted_bytes(m) == expected_tiles * tile_bytes == budget
        assert lib.kvtm_persisted_tiles(m) == expected_tiles
        print(f"S2 exact fill: {expected_tiles} tiles admitted, "
              f"used={lib.kvtm_persisted_bytes(m)} == budget")

        # S3: overflow refusal — the core Phase 3 criterion
        assert admit(expected_tiles) == REFUSED
        assert lib.kvtm_persisted_bytes(m) == budget
        print(f"S3 overflow: tile #{expected_tiles} refused, "
              f"used still {lib.kvtm_persisted_bytes(m)}")

        # S4: duplicate admit must never double-count
        assert admit(5) == INVALID
        assert lib.kvtm_persisted_tiles(m) == expected_tiles
        assert lib.kvtm_persisted_bytes(m) == budget

        # S5: release frees exactly one tile's worth
        assert lib.kvtm_release(m, 7) == 1
        assert lib.kvtm_persisted_bytes(m) == budget - tile_bytes
        assert lib.kvtm_release(m, 7) == 0  # double release rejected
        print(f"S5 release: freed one tile -> used="
              f"{lib.kvtm_persisted_bytes(m)}; double-release rejected")

        # S6: re-admission into freed space succeeds
        assert admit(expected_tiles) == OK
        assert lib.kvtm_persisted_tiles(m) == expected_tiles
        assert admit(expected_tiles + 1) == REFUSED
        print("S6 re-admit: freed slot reused, next candidate refused again")

        # S7: clear resets everything (per-layer teardown path)
        lib.kvtm_clear(m)
        assert lib.kvtm_persisted_tiles(m) == 0
        assert lib.kvtm_persisted_bytes(m) == 0
        assert admit(90) == OK
        print(f"S7 clear+readmit: used={lib.kvtm_persisted_bytes(m)} after reset")

        # S8: C++ tile layout must match Python-side arithmetic
        tiles_cpp = lib.kvtm_layout_tiles_per_frame(tpf, tt)
        assert tiles_cpp == (tpf + tt - 1) // tt == 98
        starts, counts = [], []
        for i in range(tiles_cpp):
            s = lib.kvtm_layout_tile_start(tpf, tt, i)
            c = lib.kvtm_layout_tile_count(tpf, tt, i)
            assert s == i * tt and c == min(tt, tpf - s)
            starts.append(s)
            counts.append(c)
        assert sum(counts) == tpf == 1560 and counts[-1] == 8
        print(f"S8 layout cross-check: {tiles_cpp} tiles, last count="
              f"{counts[-1]}, sum={sum(counts)} tokens")

        lib.kvtm_dev_free(ctypes.c_void_p(scratch))
        print("\nVERDICT: PASS")
        return True
    finally:
        lib.kvtm_destroy(m)


if __name__ == "__main__":
    cfg = json.loads((ROOT / "configs" / "grounding_config.json").read_text())
    sys.exit(0 if scenario(cfg) else 1)


def test_capacity_admission():
    cfg = json.loads((ROOT / "configs" / "grounding_config.json").read_text())
    assert scenario(cfg), "capacity admission scenario failed"
