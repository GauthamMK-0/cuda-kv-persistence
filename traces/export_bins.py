"""Export trace tensors to raw float32 binaries for the CUDA harness.

The CUDA side reads plain flat binaries (no zip/npy parsing). Shapes live in
traces/trace_meta_{stem}.json; the binary gets them via CLI flags.

Outputs: traces/{q,k,v}_{stem}.bin
"""

import argparse
from pathlib import Path

import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--npz", default="traces/qkv_8f.npz")
    p.add_argument("--stem", default="8f")
    args = p.parse_args()

    traces = Path(__file__).resolve().parent
    data = np.load(traces.parent / args.npz if not args.npz.startswith("/")
                   else args.npz)
    for name in ("Q", "K", "V"):
        arr = np.ascontiguousarray(data[name], dtype=np.float32)
        out = traces / f"{name.lower()}_{args.stem}.bin"
        arr.tofile(out)
        print(f"{name} shape={arr.shape} dtype=float32 -> {out} "
              f"({out.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
