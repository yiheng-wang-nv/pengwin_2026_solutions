#!/usr/bin/env bash
# Phase 3 (single split): train one ResEncXL model per dataset, in parallel.
# Assumes scripts/02d_single_split.sh has been run (single-fold splits_final.json).
#
# Uses 2 GPUs in parallel (one per dataset). Override with GPUS env var:
#   GPUS="2 3" bash scripts/04_train_xl.sh        # use GPU 2 for ds001, GPU 3 for ds002
#
# Plans default to nnUNetResEncUNetXLPlans; override with PLANS env var.
# Config default to 3d_fullres; override with CONFIG env var.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/env.sh"

PLANS="${PLANS:-nnUNetResEncUNetXLPlans}"
CONFIG="${CONFIG:-3d_fullres}"
read -ra GPU_LIST <<< "${GPUS:-0 1}"
[ "${#GPU_LIST[@]}" -ge 2 ] || die "Need 2 GPUs in GPUS env var (default '0 1')"

log "=========================================="
log "Training with PLANS=$PLANS  CONFIG=$CONFIG"
log "  Dataset001 fold 0  on GPU ${GPU_LIST[0]}"
log "  Dataset002 fold 0  on GPU ${GPU_LIST[1]}"
log "  (parallel — expect ~12-24 h on H100 80GB)"
log "=========================================="

mkdir -p "$SCRIPT_DIR/../logs"
LOG_DIR="$SCRIPT_DIR/../logs"

CUDA_VISIBLE_DEVICES="${GPU_LIST[0]}" PLANS="$PLANS" \
    bash "$SCRIPT_DIR/03_train.sh" 001 0 "$CONFIG" \
    > "$LOG_DIR/train_001_f0_xl.log" 2>&1 &
PID_001=$!
log "  ds001 PID=$PID_001  log=$LOG_DIR/train_001_f0_xl.log"

CUDA_VISIBLE_DEVICES="${GPU_LIST[1]}" PLANS="$PLANS" \
    bash "$SCRIPT_DIR/03_train.sh" 002 0 "$CONFIG" \
    > "$LOG_DIR/train_002_f0_xl.log" 2>&1 &
PID_002=$!
log "  ds002 PID=$PID_002  log=$LOG_DIR/train_002_f0_xl.log"

log ""
log "Watch progress:"
log "  tail -f $LOG_DIR/train_00{1,2}_f0_xl.log"
log ""
log "Waiting for both to finish..."
FAIL=0
wait $PID_001 || { log "  ds001 training FAILED (exit $?)"; FAIL=1; }
wait $PID_002 || { log "  ds002 training FAILED (exit $?)"; FAIL=1; }

[ "$FAIL" -eq 0 ] && log "Both trainings finished successfully." || die "One or both trainings failed."

log ""
log "Checkpoints:"
log "  \$nnUNet_results/Dataset001_*/nnUNetTrainer__${PLANS}__${CONFIG}/fold_0/checkpoint_best.pth"
log "  \$nnUNet_results/Dataset002_*/nnUNetTrainer__${PLANS}__${CONFIG}/fold_0/checkpoint_best.pth"
