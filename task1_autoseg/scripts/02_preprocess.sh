#!/usr/bin/env bash
# Phase 2: raw .mha -> two nnUNetv2 datasets, then nnUNet plan_and_preprocess.
#
# Steps:
#   a. gen_nnunet_dataset.py  -> data/{Dataset001_PENGWIN_Anatomical, Dataset002_PENGWIN_Frac}/
#   b. gen_CSM_dataset.py     -> data/Dataset002_PENGWIN_Frac/labelsTr/  (3-class CSM labels)
#   c. Mirror both datasets into $nnUNet_raw and run plan_and_preprocess.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/env.sh"

[ -d "$RAW_DATA" ] || die "RAW_DATA='$RAW_DATA' does not exist"
[ "$(ls "$RAW_DATA" 2>/dev/null | wc -l)" -ge 340 ] || \
    die "Expected 340 case dirs under $RAW_DATA, found $(ls "$RAW_DATA" 2>/dev/null | wc -l)"

activate_env
mkdir -p "$BASELINE_DATA" "$nnUNet_raw" "$nnUNet_preprocessed" "$nnUNet_results"

# --- a. Raw -> two datasets ---
log "Step a: gen_nnunet_dataset.py  (src=$RAW_DATA  out=$BASELINE_DATA)"
python "$BASELINE_DIR/preprocessing/gen_nnunet_dataset.py" \
    --src "$RAW_DATA" \
    --out "$BASELINE_DATA"

# --- b. CSM labels (3-class) for Dataset002 ---
log "Step b: gen_CSM_dataset.py  (kernel=7)"
python "$BASELINE_DIR/preprocessing/gen_CSM_dataset.py" \
    --input  "$BASELINE_DATA/Dataset002_PENGWIN_Frac/labelsTr_instance" \
    --output "$BASELINE_DATA/Dataset002_PENGWIN_Frac/labelsTr" \
    --kernel 7

# --- c. Copy + nnUNet plan_and_preprocess for each dataset ---
for ds in Dataset001_PENGWIN_Anatomical Dataset002_PENGWIN_Frac; do
    if [ -d "$nnUNet_raw/$ds" ]; then
        log "Step c: $ds already present in nnUNet_raw, skipping cp"
    else
        log "Step c: copying $ds -> nnUNet_raw"
        cp -r "$BASELINE_DATA/$ds" "$nnUNet_raw/$ds"
    fi
done

log "Step c: nnUNetv2_plan_and_preprocess for 001 (3d_fullres only)"
nnUNetv2_plan_and_preprocess -d 001 --verify_dataset_integrity -c 3d_fullres

log "Step c: nnUNetv2_plan_and_preprocess for 002 (3d_fullres only)"
nnUNetv2_plan_and_preprocess -d 002 --verify_dataset_integrity -c 3d_fullres

log "Done. Next: bash scripts/03_train.sh 001 0    (and again for 002, all 5 folds)"
