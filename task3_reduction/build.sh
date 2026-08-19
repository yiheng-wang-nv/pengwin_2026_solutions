#!/usr/bin/env bash
# Build the Task 3 algorithm container.
#
#   bash build.sh                 -> pengwin-task3:latest
#   IMAGE=foo TAG=bar bash build.sh
#
# The AssemblyNet baseline is cloned into the build context rather than vendored, so this
# repository holds only our own code. Set BASELINE to an existing checkout to skip the clone.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE="${IMAGE:-pengwin-task3}"
TAG="${TAG:-latest}"
BASELINE_URL="https://github.com/Sutuk/PENGWIN2026_Task3_Reduction_Baseline"

STAGE="$(mktemp -d /tmp/pengwin-task3-build.XXXXXX)"
trap 'rm -rf "$STAGE"' EXIT
cp "$HERE/container/Dockerfile" "$HERE/container/requirements.txt" \
   "$HERE/container/process.py" "$STAGE/"
cp -r "$HERE/scripts" "$STAGE/scripts"

if [ -n "${BASELINE:-}" ]; then
    cp -r "$BASELINE" "$STAGE/PENGWIN2026_Task3_Reduction_Baseline"
else
    git clone --depth 1 "$BASELINE_URL" "$STAGE/PENGWIN2026_Task3_Reduction_Baseline"
fi
rm -rf "$STAGE/PENGWIN2026_Task3_Reduction_Baseline/.git"

docker build -t "$IMAGE:$TAG" "$STAGE"
echo "Built $IMAGE:$TAG"
