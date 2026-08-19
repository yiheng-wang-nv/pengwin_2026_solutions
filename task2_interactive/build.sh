#!/usr/bin/env bash
# Build the Task 2 algorithm container.
#
#   bash build.sh                 -> pengwin-task2:latest
#   IMAGE=foo TAG=bar bash build.sh
#
# The build context is this directory: the Dockerfile needs container/requirements.txt,
# container/process.py and inference/, so they are staged together first.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE="${IMAGE:-pengwin-task2}"
TAG="${TAG:-latest}"

STAGE="$(mktemp -d /tmp/pengwin-task2-build.XXXXXX)"
trap 'rm -rf "$STAGE"' EXIT
cp "$HERE/container/Dockerfile" "$HERE/container/requirements.txt" \
   "$HERE/container/process.py" "$STAGE/"
cp -r "$HERE/inference" "$STAGE/inference"

docker build -t "$IMAGE:$TAG" "$STAGE"
echo "Built $IMAGE:$TAG"
