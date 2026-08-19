#!/usr/bin/env bash
# Shared environment for all baseline scripts. Source me; do not execute.
# Override any of these by exporting before sourcing:
#   ANACONDA_BASE=/opt/conda CONDA_ENV_NAME=myenv source scripts/env.sh

# --- Paths ---
export REPO_ROOT="${REPO_ROOT:?set REPO_ROOT to the working tree}"
export BASELINE_DIR="${BASELINE_DIR:-$REPO_ROOT/external/PENGWIN2026_Task1_AutoSeg_Baseline}"

# 340 symlinks: <id>/{image.mha,label.mha}
export RAW_DATA="${RAW_DATA:-$REPO_ROOT/raw_data/PENGWIN_train}"

# Output of preprocessing/gen_nnunet_dataset.py + gen_CSM_dataset.py
export BASELINE_DATA="${BASELINE_DATA:-$BASELINE_DIR/data}"

# nnUNetv2 workdir (raw / preprocessed / results). Keep all three on a big disk.
export NNUNET_WORKDIR="${NNUNET_WORKDIR:-$REPO_ROOT/nnUNet_workdir}"
export nnUNet_raw="${nnUNet_raw:-$NNUNET_WORKDIR/raw}"
export nnUNet_preprocessed="${nnUNet_preprocessed:-$NNUNET_WORKDIR/preprocessed}"
export nnUNet_results="${nnUNet_results:-$NNUNET_WORKDIR/results}"

# --- Conda ---
export ANACONDA_BASE="${ANACONDA_BASE:?set ANACONDA_BASE to your conda install}"
export CONDA_ENV_NAME="${CONDA_ENV_NAME:-nnunet}"
export PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
export TORCH_CUDA="${TORCH_CUDA:-cu121}"   # cu118 / cu121 / cu124 / cpu

# --- Helpers ---
activate_env() {
    # shellcheck disable=SC1091
    source "$ANACONDA_BASE/etc/profile.d/conda.sh"
    conda activate "$ANACONDA_BASE/envs/$CONDA_ENV_NAME"
}

log() { printf '\033[1;34m[%s]\033[0m %s\n' "$(date +%H:%M:%S)" "$*"; }
die() { printf '\033[1;31m[ERROR]\033[0m %s\n' "$*" >&2; exit 1; }
