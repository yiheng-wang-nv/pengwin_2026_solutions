#!/usr/bin/env python3
"""
Script to copy single-fragment anatomy predictions to the fragment output structure.
For anatomies with only one fragment, we copy the Phase 1 prediction directly 
instead of running Phase 2 inference.

Usage:
    python copy_single_fragments.py \
        --anatomy_pred temp/predictions/anatomical/ \
        --fragment_input temp/nnunet_input/fragments/ \
        --fragment_output temp/predictions/fragments/ \
        --pid case_001
"""

import argparse
import shutil
from pathlib import Path
import SimpleITK as sitk
import numpy as np

ANATOMY_NAMES = {
    1: "sacrum",
    2: "left_hip", 
    3: "right_hip",
    4: "femur",
}

# Mapping from anatomy name to label value in Phase 1 prediction
ANATOMY_LABELS = {
    "sacrum": 1,
    "left_hip": 2,
    "right_hip": 3,
    "femur": 4,
}


def find_single_fragment_anatomies(fragment_input_base):
    """
    Find which anatomies have only one fragment (or no fragments).
    Returns a list of anatomy names that should be copied from Phase 1.
    """
    single_fragment_anatomies = []
    
    for anatomy_name in ANATOMY_NAMES.values():
        anatomy_dir = fragment_input_base / anatomy_name
        
        if not anatomy_dir.exists():
            print(f"  {anatomy_name}: directory missing - will copy from Phase 1")
            single_fragment_anatomies.append(anatomy_name)
            continue
        
        # Count fragment directories (directories with numeric names)
        fragment_dirs = [d for d in anatomy_dir.iterdir() if d.is_dir() and d.name.isdigit()]
        n_fragments = len(fragment_dirs)
        
        if n_fragments <= 1:
            print(f"  {anatomy_name}: {n_fragments} fragment(s) - will copy from Phase 1")
            single_fragment_anatomies.append(anatomy_name)
        else:
            print(f"  {anatomy_name}: {n_fragments} fragments - will use Phase 2 predictions")
    
    return single_fragment_anatomies


def copy_anatomy_prediction_to_fragments(anatomy_pred_path, fragment_output_base, 
                                         anatomy_name, pid, force=False):
    """
    Copy the Phase 1 anatomy prediction to the fragment output structure.
    
    Args:
        anatomy_pred_path: Path to the Phase 1 prediction .mha file
        fragment_output_base: Base directory for fragment outputs
        anatomy_name: Name of the anatomy (sacrum, left_hip, right_hip, femur)
        pid: Patient ID/basename
        force: Overwrite existing files if True
    """
    # Read the anatomy prediction
    if not anatomy_pred_path.exists():
        print(f"  WARNING: Anatomy prediction not found: {anatomy_pred_path}")
        return False
    
    img = sitk.ReadImage(str(anatomy_pred_path))
    data = sitk.GetArrayFromImage(img)
    
    # Get the label value for this anatomy
    label_value = ANATOMY_LABELS[anatomy_name]
    
    # Create binary mask for this anatomy
    binary_mask = (data == label_value).astype(np.uint8)
    
    # Create output directory for this anatomy (single_fragment marker)
    output_dir = fragment_output_base / anatomy_name / str((label_value - 1) * 50 + 1)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save the prediction as a binary mask (Phase 2 format)
    output_path = output_dir / f"{pid}.mha"
    
    if output_path.exists() and not force:
        print(f"  Skipping {anatomy_name} (already exists: {output_path})")
        return True
    
    # Convert to SimpleITK and save
    out_img = sitk.GetImageFromArray(binary_mask)
    out_img.CopyInformation(img)
    sitk.WriteImage(out_img, str(output_path))
    
    print(f"  ✓ Copied {anatomy_name} prediction to {output_path}")
    
    # Also create an info file
    info_path = output_dir / "info.txt"
    with open(info_path, 'w') as f:
        f.write(f"Anatomy: {anatomy_name}\n")
        f.write(f"Patient: {pid}\n")
        f.write(f"Source: Phase 1 anatomical prediction\n")
        f.write(f"Label value in Phase 1: {label_value}\n")
        f.write(f"Binary mask saved for Phase 2 compatibility\n")
        f.write(f"Reason: Single fragment detected (skipping Phase 2 inference)\n")
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Copy single-fragment anatomy predictions to fragment output structure",
        epilog="For anatomies with only one fragment, copy Phase 1 predictions instead of running Phase 2"
    )
    
    parser.add_argument(
        "--anatomy_pred",
        type=str,
        required=True,
        help="Directory containing Phase 1 anatomy predictions (output of nnUNetv2_predict)"
    )
    
    parser.add_argument(
        "--fragment_input",
        type=str,
        required=True,
        help="Directory containing fragment inputs (output of convert_fragments_to_nnunet_input.py)"
    )
    
    parser.add_argument(
        "--fragment_output",
        type=str,
        required=True,
        help="Directory where fragment predictions will be stored"
    )
    
    parser.add_argument(
        "--pid",
        type=str,
        required=True,
        help="Patient ID/basename (e.g., 'case_001')"
    )
    
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force overwrite existing files"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed information"
    )
    
    args = parser.parse_args()
    
    # Convert paths
    anatomy_pred_dir = Path(args.anatomy_pred)
    fragment_input_base = Path(args.fragment_input)
    fragment_output_base = Path(args.fragment_output)
    
    # Validate paths
    if not anatomy_pred_dir.exists():
        print(f"ERROR: Anatomy prediction directory not found: {anatomy_pred_dir}")
        return 1
    
    if not fragment_input_base.exists():
        print(f"ERROR: Fragment input directory not found: {fragment_input_base}")
        return 1
    
    # Create fragment output base directory
    fragment_output_base.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Copying Single-Fragment Anatomies")
    print(f"{'='*60}")
    print(f"Anatomy predictions: {anatomy_pred_dir}")
    print(f"Fragment input base: {fragment_input_base}")
    print(f"Fragment output base: {fragment_output_base}")
    print(f"Patient ID: {args.pid}")
    
    # Find which anatomies have single fragments
    print(f"\nAnalyzing fragment counts...")
    single_fragment_anatomies = find_single_fragment_anatomies(fragment_input_base)
    
    if not single_fragment_anatomies:
        print(f"\nNo single-fragment anatomies found. Nothing to copy.")
        return 0
    
    # Path to the Phase 1 prediction for this patient
    # Assuming the prediction file is named {pid}.mha or similar
    anatomy_pred_path = anatomy_pred_dir / f"{args.pid}.mha"
    
    # Also try with _0000 suffix if not found
    if not anatomy_pred_path.exists():
        anatomy_pred_path = anatomy_pred_dir / f"{args.pid}_0000.mha"
    
    if not anatomy_pred_path.exists():
        print(f"\nERROR: Could not find anatomy prediction for {args.pid}")
        print(f"  Tried: {anatomy_pred_dir / f'{args.pid}.mha'}")
        print(f"  Tried: {anatomy_pred_dir / f'{args.pid}_0000.mha'}")
        return 1
    
    print(f"\nUsing anatomy prediction: {anatomy_pred_path}")
    
    # Copy predictions for each single-fragment anatomy
    print(f"\nCopying predictions...")
    success_count = 0
    
    for anatomy_name in single_fragment_anatomies:
        if args.verbose:
            print(f"\nProcessing {anatomy_name}...")
        
        success = copy_anatomy_prediction_to_fragments(
            anatomy_pred_path,
            fragment_output_base,
            anatomy_name,
            args.pid,
            force=args.force
        )
        
        if success:
            success_count += 1
    
    # Create a summary file
    summary_path = fragment_output_base / "single_fragment_summary.txt"
    with open(summary_path, 'w') as f:
        f.write(f"Single-Fragment Anatomy Copy Summary\n")
        f.write(f"{'='*60}\n")
        f.write(f"Patient: {args.pid}\n")
        f.write(f"Source prediction: {anatomy_pred_path}\n")
        f.write(f"Date: {__import__('datetime').datetime.now()}\n\n")
        f.write(f"Anatomies copied (single fragment):\n")
        for anatomy_name in single_fragment_anatomies:
            f.write(f"  - {anatomy_name}\n")
        f.write(f"\nOutput location: {fragment_output_base}\n")
        f.write(f"Each anatomy has a 'single_fragment' subdirectory with:\n")
        f.write(f"  - {args.pid}.mha: Binary mask of the anatomy\n")
        f.write(f"  - info.txt: Metadata about the copy operation\n")
    
    print(f"\n{'='*60}")
    print(f"✓ Done!")
    print(f"  Copied {success_count}/{len(single_fragment_anatomies)} single-fragment anatomies")
    print(f"  Output directory: {fragment_output_base}")
    print(f"  Summary: {summary_path}")
    print(f"{'='*60}")
    
    return 0


if __name__ == "__main__":
    exit(main())
