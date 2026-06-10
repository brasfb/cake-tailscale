#!/usr/bin/env bash
# Pull + rebuild cake on every worker Mac Mini over SSH (via Tailscale).
#
# Usage:
#   MINIS="mini2 mini3 mini4" ./scripts/deploy-minis.sh
#
# Hosts can be Tailscale MagicDNS names or 100.x.y.z addresses. Assumes the
# repo is cloned at $CAKE_DIR (default ~/cake) on each mini.
set -euo pipefail

MINIS="${MINIS:-mini2 mini3 mini4}"
CAKE_DIR="${CAKE_DIR:-~/cake}"

for m in $MINIS; do
    echo "==> deploying to $m"
    ssh "$m" "cd $CAKE_DIR && git pull && ./scripts/build-intel-mac.sh"
done

echo "==> all workers up to date"
