#!/usr/bin/env bash
# Phase 2c: re-plan + preprocess with a ResEnc planner for large GPUs.
#
# nnUNet's default planner targets 12 GB GPUs (batch=2, small patch). On
# H100 / A100 80GB cards that wastes >90% of memory. The ResEnc presets
# auto-pick larger patches/batches AND switch the architecture to
# Residual Encoder UNet (typically +1-3 Dice).
#
# This script ONLY re-runs plan + preprocess (fingerprint is reused, raw
# data conversion is not touched). New plans coexist with the default
# under nnUNet_preprocessed/Dataset00X/ as a separate plans.json file.
#
# Usage:
#   bash scripts/02c_replan_resenc.sh                  # XL (recommended for 80GB)
#   PLANNER=nnUNetPlannerResEncL  bash scripts/02c_replan_resenc.sh
#   PLANNER=nnUNetPlannerResEncM  bash scripts/02c_replan_resenc.sh
#
# After this, train with the new plans via:
#   bash scripts/03_train.sh 001 0 3d_fullres -- -p nnUNetResEncUNetXLPlans
#   (or set PLANS=nnUNetResEncUNetXLPlans in env.sh — see 03_train.sh patch)

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/env.sh"

PLANNER="${PLANNER:-nnUNetPlannerResEncXL}"

# Planner -> plans-name convention (nnUNet v2)
case "$PLANNER" in
    nnUNetPlannerResEncM)   PLANS_NAME="nnUNetResEncUNetMPlans"  ;;
    nnUNetPlannerResEncL)   PLANS_NAME="nnUNetResEncUNetLPlans"  ;;
    nnUNetPlannerResEncXL)  PLANS_NAME="nnUNetResEncUNetXLPlans" ;;
    *) die "Unknown planner '$PLANNER'. Choose one of nnUNetPlannerResEnc{M,L,XL}" ;;
esac

activate_env

for DS in 001 002; do
    log "=========================================="
    log "Re-plan + preprocess Dataset$DS with $PLANNER"
    log "  -> writes $nnUNet_preprocessed/Dataset${DS}_.../$PLANS_NAME.json"
    log "=========================================="
    nnUNetv2_plan_and_preprocess -d "$DS" -pl "$PLANNER" -c 3d_fullres
done

log "=========================================="
log "Done. Inspect new patch / batch sizes:"
log "  for d in 001 002; do"
log "    python -c \"import json; p=json.load(open('\$nnUNet_preprocessed/Dataset\${d}_*/${PLANS_NAME}.json')); cfg=p['configurations']['3d_fullres']; print(d, 'patch=', cfg['patch_size'], 'batch=', cfg['batch_size'])\""
log "  done"
log ""
log "Train with these plans:"
log "  bash scripts/03_train.sh 001 0 3d_fullres -p $PLANS_NAME"
log "=========================================="
