#!/usr/bin/env bash
# Build the algorithm container image.
set -euo pipefail
cd "$(dirname "$0")"
IMAGE="${IMAGE:-pengwin-task1}"
docker build -t "$IMAGE" .
echo "Built image: $IMAGE"
