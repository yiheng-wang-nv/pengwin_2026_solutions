#!/usr/bin/env python3
"""Generate patient-level 5-fold splits for both PENGWIN datasets.

Why this exists
---------------
nnUNetv2 normally generates a random 5-fold split over its training-case list.
For Dataset002_PENGWIN_Frac that leaks at the patient level: each source patient
produces up to 3 per-bone cases (e.g. PENGWIN_001_sacrum, _leftHip, _rightHip),
which can land in different folds. The model would then "validate" on a patient
it already trained on.

This script writes splits_final.json into both preprocessed dataset directories,
using the *same* patient-level stratified split for both. Doing so also keeps
folds aligned between the anatomical and CSM models, which is useful for
downstream ensembling.

When to run
-----------
After `nnUNetv2_plan_and_preprocess` and before `nnUNetv2_train`.
Idempotent unless you pass --overwrite.

Outputs
-------
$nnUNet_preprocessed/Dataset001_PENGWIN_Anatomical/splits_final.json
$nnUNet_preprocessed/Dataset002_PENGWIN_Frac/splits_final.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

from sklearn.model_selection import StratifiedKFold


# Case naming: Dataset001 -> "PENGWIN_<3digit>"
#              Dataset002 -> "PENGWIN_<3digit>_<bone>"
CASE_RE = re.compile(r"^PENGWIN_(\d{3})(?:_(sacrum|leftHip|rightHip|femur))?$")


def patient_id(case: str) -> str:
    m = CASE_RE.match(case)
    if not m:
        raise ValueError(f"Cannot parse case name '{case}' (expected PENGWIN_<id>[_<bone>])")
    return m.group(1)


def patient_type(pid: str) -> str:
    """Pelvic vs femur per the Zenodo README subject-ID layout."""
    n = int(pid)
    if 1 <= n <= 250:
        return "pelvic"
    if 251 <= n <= 500:
        return "femur"
    raise ValueError(f"Patient ID {pid} outside [001, 500]")


def list_cases(preprocessed_dataset_dir: Path) -> list[str]:
    """Read case identifiers from gt_segmentations/ (canonical post-preprocess)."""
    gt = preprocessed_dataset_dir / "gt_segmentations"
    if not gt.is_dir():
        raise FileNotFoundError(
            f"{gt} not found. Run `nnUNetv2_plan_and_preprocess -d <id>` first."
        )
    cases = sorted(
        p.name.removesuffix(".nii.gz") for p in gt.glob("*.nii.gz")
    )
    if not cases:
        raise RuntimeError(f"No .nii.gz cases found under {gt}")
    return cases


def build_splits(cases: list[str], n_splits: int, seed: int) -> list[dict]:
    """Patient-stratified KFold. Every case of a patient lands in the same fold."""
    by_patient: dict[str, list[str]] = {}
    for case in cases:
        by_patient.setdefault(patient_id(case), []).append(case)

    patients = sorted(by_patient.keys())
    strata = [patient_type(p) for p in patients]

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = []
    for fi, (train_idx, val_idx) in enumerate(skf.split(patients, strata)):
        train_patients = [patients[i] for i in train_idx]
        val_patients = [patients[i] for i in val_idx]
        train_cases = sorted(c for p in train_patients for c in by_patient[p])
        val_cases = sorted(c for p in val_patients for c in by_patient[p])
        folds.append({"train": train_cases, "val": val_cases})

        pelvic = sum(1 for p in val_patients if patient_type(p) == "pelvic")
        femur = len(val_patients) - pelvic
        print(
            f"  fold {fi}: train={len(train_cases):4d} cases / {len(train_patients):3d} patients"
            f"   val={len(val_cases):4d} cases / {len(val_patients):3d} patients"
            f"  (pelvic={pelvic}, femur={femur})"
        )
    return folds


def write_splits(out_path: Path, folds: list[dict], overwrite: bool):
    if out_path.exists() and not overwrite:
        print(f"  EXISTS: {out_path} (use --overwrite to replace)")
        return
    out_path.write_text(json.dumps(folds, indent=2))
    print(f"  WROTE : {out_path}")


def process_dataset(name: str, nnunet_preprocessed: Path, n_splits: int, seed: int,
                    overwrite: bool) -> None:
    ds_dir = nnunet_preprocessed / name
    if not ds_dir.is_dir():
        print(f"\nSKIP {name}: {ds_dir} does not exist")
        return

    print(f"\n[{name}]")
    cases = list_cases(ds_dir)
    print(f"  found {len(cases)} cases across "
          f"{len({patient_id(c) for c in cases})} patients")
    folds = build_splits(cases, n_splits=n_splits, seed=seed)
    write_splits(ds_dir / "splits_final.json", folds, overwrite=overwrite)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--nnunet-preprocessed", type=Path,
                    default=os.environ.get("nnUNet_preprocessed"),
                    help="Path to nnUNet_preprocessed (default: $nnUNet_preprocessed)")
    ap.add_argument("--n-splits", type=int, default=5, help="number of folds (default 5)")
    ap.add_argument("--seed", type=int, default=12345, help="KFold seed (default 12345)")
    ap.add_argument("--overwrite", action="store_true",
                    help="overwrite existing splits_final.json")
    ap.add_argument("--datasets", nargs="+",
                    default=["Dataset001_PENGWIN_Anatomical", "Dataset002_PENGWIN_Frac"],
                    help="dataset folder names under nnUNet_preprocessed")
    args = ap.parse_args()

    if not args.nnunet_preprocessed:
        sys.exit("ERROR: --nnunet-preprocessed not given and $nnUNet_preprocessed unset")
    args.nnunet_preprocessed = Path(args.nnunet_preprocessed)
    if not args.nnunet_preprocessed.is_dir():
        sys.exit(f"ERROR: {args.nnunet_preprocessed} is not a directory")

    print(f"nnUNet_preprocessed = {args.nnunet_preprocessed}")
    print(f"n_splits={args.n_splits}  seed={args.seed}  overwrite={args.overwrite}")

    for ds in args.datasets:
        process_dataset(ds, args.nnunet_preprocessed,
                        n_splits=args.n_splits, seed=args.seed,
                        overwrite=args.overwrite)


if __name__ == "__main__":
    main()
