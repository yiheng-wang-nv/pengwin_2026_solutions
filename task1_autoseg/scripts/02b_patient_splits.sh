#!/usr/bin/env bash
# Phase 2b: write patient-level 5-fold splits_final.json into both preprocessed datasets.
# Fixes the patient-level leak in Dataset002 (per-bone cases of the same patient
# would otherwise land in different folds under nnUNet's default random split).
# Run AFTER scripts/02_preprocess.sh and BEFORE scripts/03_train.sh.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/env.sh"

activate_env

log "Writing patient-level splits_final.json into both datasets"
python "$SCRIPT_DIR/02b_patient_splits.py" "$@"

log "Done. Next: bash scripts/03_train.sh 001 0"
