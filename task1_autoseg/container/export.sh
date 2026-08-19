#!/usr/bin/env bash
# Export the built image as a .tar.gz for direct upload to Grand Challenge.
set -euo pipefail
cd "$(dirname "$0")"
IMAGE="${IMAGE:-pengwin-task1}"
OUT="${1:-pengwin-task1.tar.gz}"
echo "Saving $IMAGE -> $OUT (this can take a few minutes)..."
docker save "$IMAGE" | gzip -c > "$OUT"
echo "Wrote $OUT  ($(du -h "$OUT" | cut -f1))"
