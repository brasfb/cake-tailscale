#!/usr/bin/env bash
# Start cake workers on the Mac Minis over SSH, then print the master command.
#
# Usage:
#   MODEL=Qwen/Qwen3-1.7B MINIS="mini2 mini3 mini4" ./scripts/start-minis.sh
#
# Workers stay attached to this script's SSH sessions (logs stream here);
# Ctrl-C stops them. Set CAKE_CLUSTER_KEY to enable mutual authentication —
# strongly recommended even inside a tailnet.
set -euo pipefail

MODEL="${MODEL:?set MODEL, e.g. MODEL=Qwen/Qwen3-1.7B}"
MINIS="${MINIS:-mini2 mini3 mini4}"
CAKE_DIR="${CAKE_DIR:-~/cake}"
TOPOLOGY="${TOPOLOGY:-topology-minis.yml}"

pids=()
cleanup() { kill "${pids[@]}" 2>/dev/null || true; }
trap cleanup EXIT

for m in $MINIS; do
    echo "==> starting worker on $m"
    ssh "$m" "cd $CAKE_DIR && CAKE_CLUSTER_KEY=${CAKE_CLUSTER_KEY:-} \
        ./target/release/cake run '$MODEL' --name '$m' \
        --topology '$TOPOLOGY' --address 0.0.0.0:10128" \
        2>&1 | sed "s/^/[$m] /" &
    pids+=($!)
done

echo
echo "Workers starting. Run the master with:"
echo "  ./target/release/cake serve '$MODEL' --topology $TOPOLOGY --api 0.0.0.0:8080"
echo
wait
