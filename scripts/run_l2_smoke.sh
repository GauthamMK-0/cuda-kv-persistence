#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$ROOT/build"
nvcc -arch=sm_86 -o "$ROOT/build/l2_persist_smoke" "$ROOT/src/l2_persist_smoke.cu"
"$ROOT/build/l2_persist_smoke"
