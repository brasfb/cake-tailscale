#!/usr/bin/env bash
# Start cake workers on the Mac Minis over SSH, then print the master command.
#
# Usage:
#   CAKE_CLUSTER_KEY=secret MODEL=Qwen/Qwen3-1.7B MINIS="mini2 mini3 mini4" \
#     ./scripts/start-minis.sh
#
# Workers stay attached to this script's SSH sessions (logs stream here);
# Ctrl-C stops them. The cluster key enables mutual authentication and is
# required for worker mode.
set -euo pipefail

MODEL="${MODEL:?set MODEL, e.g. MODEL=Qwen/Qwen3-1.7B}"
CAKE_CLUSTER_KEY="${CAKE_CLUSTER_KEY:?set CAKE_CLUSTER_KEY (shared secret for mutual auth)}"
MINIS="${MINIS:-mini2 mini3 mini4}"
CAKE_DIR="${CAKE_DIR:-~/cake}"
TOPOLOGY="${TOPOLOGY:-topology-minis.yml}"

pids=()
cleanup() { kill "${pids[@]}" 2>/dev/null || true; }
trap cleanup EXIT

# Worker mode = no positional model: the model is passed via --model so each
# worker loads its assigned layers from its local copy/cache.
for m in $MINIS; do
    echo "==> starting worker on $m"
    ssh "$m" "cd $CAKE_DIR && \
        ./target/release/cake run --cluster-key '$CAKE_CLUSTER_KEY' \
        --model '$MODEL' --name '$m' \
        --topology '$TOPOLOGY' --address 0.0.0.0:10128" \
        2>&1 | sed "s/^/[$m] /" &
    pids+=($!)
done

echo
echo "Workers starting. Run the master with:"
echo "  ./target/release/cake serve '$MODEL' --cluster-key '<same key>' --topology $TOPOLOGY --api 0.0.0.0:8080"
echo
wait
