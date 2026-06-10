#!/usr/bin/env bash
# Build cake for a 2014-era Intel Mac (CPU-only inference).
#
# No acceleration features: the pure-Rust gemm path keeps F16 weights in F16,
# which is the fastest option on bandwidth-limited LPDDR3. target-cpu=native
# enables AVX2/FMA (Haswell), which the default x86_64 baseline does not.
#
# Optional experiment: `--features accelerate` compiles on Intel macOS but
# only accelerates F32 matmul (F16 weights get converted, doubling memory
# bandwidth). Benchmark both on one machine and keep whichever wins.
set -euo pipefail

cd "$(dirname "$0")/.."

if ! command -v cargo >/dev/null 2>&1; then
    echo "error: cargo not found — install Rust via https://rustup.rs and Xcode CLT (xcode-select --install)" >&2
    exit 1
fi

export RUSTFLAGS="${RUSTFLAGS:--C target-cpu=native}"
exec cargo build --release "$@"
