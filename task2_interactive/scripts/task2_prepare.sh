#!/usr/bin/env bash
# Task2 pre-training pipeline: dataset.json -> copy to nnUNet_raw ->
# plan_and_preprocess -> XL plans -> patient split (same 20 val as Task1).
# Run AFTER Dataset457 generation finishes.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/env.sh"
activate_env

T2="$BASELINE_DIR/.."   # not used; explicit path below
T2BASE="$REPO_ROOT/external/PENGWIN2026_Task2_InteractiveSeg_Baseline"
SRC456="$REPO_ROOT/task2_data/Dataset456_PENGWIN"
SRC457="$REPO_ROOT/task2_data/Dataset457_PENGWIN_frag"

step() { log "=========== $* ==========="; }

# --- 1. dataset.json (copy from baseline, fix numTraining to actual count) ---
step "1/5 dataset.json (numTraining = actual case count)"
fix_json() {
    local src_json="$1" dst="$2" n="$3"
    python - "$src_json" "$dst/dataset.json" "$n" <<'PY'
import json, sys
src, dst, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
d = json.load(open(src)); d["numTraining"] = n
json.dump(d, open(dst, "w"), indent=2)
print(f"  wrote {dst}  (numTraining={n})")
PY
}
n456=$(ls "$SRC456/labelsTr"/*.mha | wc -l)
n457=$(ls "$SRC457/labelsTr"/*.mha | wc -l)
fix_json "$T2BASE/training/anatomy/dataset.json"   "$SRC456" "$n456"
fix_json "$T2BASE/training/fragments/dataset.json" "$SRC457" "$n457"

# --- 2. copy to nnUNet_raw ---
step "2/5 copy to nnUNet_raw"
for D in Dataset456_PENGWIN Dataset457_PENGWIN_frag; do
    if [ ! -d "$nnUNet_raw/$D" ]; then
        cp -r "$REPO_ROOT/task2_data/$D" "$nnUNet_raw/$D"
        log "  copied $D"
    else
        log "  $D already in nnUNet_raw, skip"
    fi
done

# --- 3. fingerprint + plan + FAULT-TOLERANT preprocess (only 3d_fullres) ---
# Split from plan_and_preprocess so we can use the robust per-case wrapper
# (skip+log+retry) for the heavy preprocessing step. NEVER run two preprocess
# jobs on the same dataset at once -> they rmtree each other (blosc2 crash).
NP="${NP:-8}"
prep_one() {
    local d="$1"
    step "3/5 dataset $d: fingerprint -> plan -> robust preprocess (np=$NP)"
    nnUNetv2_extract_fingerprint -d "$d" --verify_dataset_integrity
    nnUNetv2_plan_experiment -d "$d"                              # default nnUNetPlans
    python "$SCRIPT_DIR/task2/preprocess_robust.py" -d "$d" -c 3d_fullres -p nnUNetPlans -np "$NP"
}
prep_one 456
prep_one 457

# --- 4. XL plans (reuse nnUNetPlans_3d_fullres preprocessed data; no re-preprocess) ---
step "4/5 ResEncXL plans"
nnUNetv2_plan_experiment -d 456 -pl nnUNetPlannerResEncXL
nnUNetv2_plan_experiment -d 457 -pl nnUNetPlannerResEncXL

# --- 5. patient split (same 20 val as Task1) ---
step "5/5 patient split (reuse Task1's 20 val patients)"
python "$SCRIPT_DIR/task2_split.py" --dataset Dataset456_PENGWIN
python "$SCRIPT_DIR/task2_split.py" --dataset Dataset457_PENGWIN_frag

log "Done. Ready to train:"
log "  ds456 (warm-start ds001): PLANS=nnUNetResEncUNetXLPlans bash scripts/03_train.sh 456 0 3d_fullres -pretrained_weights <ds001_best>"
log "  ds457 (from scratch):     PLANS=nnUNetResEncUNetXLPlans bash scripts/03_train.sh 457 0 3d_fullres"
