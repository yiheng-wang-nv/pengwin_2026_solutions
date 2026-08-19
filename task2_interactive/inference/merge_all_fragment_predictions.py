#!/usr/bin/env python3
"""
Script to merge all fragment predictions into a single file with fragment_ids as labels.

Input structure:
    temp/predictions/fragments/{anatomy}/{fragment_id}/{pid}.mha
    Each file is a binary mask (0=background, 1=fragment)

Output:
    Single MHA file where each fragment gets its fragment_id as label value

Label ranges:
    Sacrum:     1-50
    Left Hip:   51-100
    Right Hip:  101-150
    Femur:      151-200

Usage:
    python merge_fragment_predictions.py \
        --input temp/predictions/fragments/ \
        --output final_prediction.mha \
        --pid case_001
"""

import argparse
from pathlib import Path
import numpy as np
import SimpleITK as sitk
from collections import defaultdict

# Label ranges for each anatomy
ANATOMY_RANGES = {
    "sacrum": (1, 50),
    "left_hip": (51, 100),
    "right_hip": (101, 150),
    "femur": (151, 200),
}

# Order of processing (to handle overlaps - later overwrites earlier)
ANATOMY_ORDER = ["left_hip", "right_hip", "femur", "sacrum"]


def find_predictions(input_base, pid):
    """
    Find all fragment predictions for a given patient.
    
    Returns:
        dict: {anatomy: {fragment_id: path_to_prediction}}
    """
    predictions = defaultdict(dict)
    input_path = Path(input_base)
    
    for anatomy_dir in input_path.iterdir():
        if not anatomy_dir.is_dir():
            continue
        
        anatomy = anatomy_dir.name
        if anatomy not in ANATOMY_RANGES:
            print(f"  Warning: Unknown anatomy directory: {anatomy}, skipping...")
            continue
        
        # Check for single_fragment directory (copied from Phase 1)
        single_fragment_dir = anatomy_dir / "single_fragment"
        if single_fragment_dir.exists():
            pred_file = single_fragment_dir / f"{pid}.mha"
            if pred_file.exists():
                # Use a special fragment_id for single fragments (e.g., 0 or None)
                # We'll handle this separately
                predictions[anatomy]["single"] = pred_file
                print(f"  Found single fragment for {anatomy}")
        
        # Check for numbered fragment directories
        for fragment_dir in anatomy_dir.iterdir():
            if not fragment_dir.is_dir():
                continue
            
            # Check if directory name is a number
            if not fragment_dir.name.isdigit():
                continue
            
            fragment_id = int(fragment_dir.name)
            pred_file = fragment_dir / f"{pid}.mha"
            
            if pred_file.exists():
                predictions[anatomy][fragment_id] = pred_file
    
    return predictions


def load_prediction(pred_path, reference_img=None):
    """Load a prediction MHA file and return as numpy array."""
    img = sitk.ReadImage(str(pred_path))
    data = sitk.GetArrayFromImage(img)
    
    # Ensure binary (0/1)
    data = (data > 0.5).astype(np.uint16)
    
    return data


def merge_predictions(predictions, reference_shape):
    """
    Merge all fragment predictions into a single volume.
    
    Args:
        predictions: dict from find_predictions()
        reference_shape: shape of the reference volume
    
    Returns:
        merged: numpy array with fragment_ids as labels
    """
    # Initialize with zeros
    merged = np.zeros(reference_shape, dtype=np.uint16)
    
    # Track conflicts (voxels that get multiple labels)
    conflicts = np.zeros(reference_shape, dtype=np.uint16)
    
    # Process anatomies in order (later anatomies will overwrite if there are overlaps)
    for anatomy in ANATOMY_ORDER:
        if anatomy not in predictions:
            continue
        
        anatomy_predictions = predictions[anatomy]
        range_start, range_end = ANATOMY_RANGES[anatomy]
        
        # Handle single fragment (copied from Phase 1)
        if "single" in anatomy_predictions:
            pred_path = anatomy_predictions["single"]
            print(f"  Processing {anatomy} (single fragment, using range {range_start}-{range_end})")
            
            # Load prediction
            pred_data = load_prediction(pred_path)
            
            # For single fragment, we need to find connected components
            # Each connected component gets a unique ID within the anatomy range
            import cc3d
            components = cc3d.connected_components(pred_data, connectivity=26)
            num_components = components.max()
            
            print(f"    Found {num_components} connected component(s)")
            
            # Assign fragment IDs
            for comp_id in range(1, num_components + 1):
                fragment_label = range_start + comp_id - 1
                if fragment_label > range_end:
                    print(f"    Warning: Too many components for {anatomy} (max {range_end - range_start + 1})")
                    fragment_label = range_end  # Cap at max
                
                # Create mask for this component
                component_mask = (components == comp_id)
                
                # Check for conflicts
                conflict_mask = (merged > 0) & component_mask
                if np.any(conflict_mask):
                    conflicts[conflict_mask] = fragment_label
                    print(f"    Warning: Conflict detected for {anatomy} component {comp_id}")
                
                # Assign label
                merged[component_mask] = fragment_label
        
        # Handle multiple fragments
        for fragment_id, pred_path in anatomy_predictions.items():
            if fragment_id == "single":
                continue
            
            # Calculate label value
            label_value = int(fragment_id)
            
            # Check if within range
            if label_value > range_end:
                print(f"  Warning: Fragment {fragment_id} for {anatomy} exceeds range {range_start}-{range_end}, capping at {range_end}. Label values is {label_value}")
                label_value = range_end
            
            print(f"  Processing {anatomy} fragment {fragment_id} -> label {label_value}")
            
            # Load prediction
            pred_data = load_prediction(pred_path)
            
            # Check for conflicts
            conflict_mask = (merged > 0) & (pred_data > 0)
            if np.any(conflict_mask):
                conflicts[conflict_mask] = label_value
                print(f"    Warning: Conflict detected with existing labels")
            
            # Assign label
            merged[pred_data > 0] = label_value
    
    # Report conflicts
    if np.any(conflicts > 0):
        unique_conflicts = np.unique(conflicts[conflicts > 0])
        print(f"\n⚠️ Conflicts detected: {len(unique_conflicts)} voxel(s) had multiple labels")
        print(f"   Conflicting labels: {unique_conflicts.tolist()}")
        print(f"   (Later anatomies overwrote earlier ones)")
    
    return merged


def main():
    parser = argparse.ArgumentParser(
        description="Merge fragment predictions into a single file with fragment_ids as labels",
        epilog="Output is a single MHA with labels 1-200 representing each fragment"
    )
    
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input directory containing fragment predictions (temp/predictions/fragments/)"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output MHA file path for merged prediction"
    )
    
    parser.add_argument(
        "--pid",
        type=str,
        required=True,
        help="Patient ID/basename (e.g., 'case_001')"
    )
    
    parser.add_argument(
        "--reference",
        type=str,
        default=None,
        help="Reference image to copy metadata from (optional, uses first prediction if not provided)"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed information"
    )
    
    args = parser.parse_args()
    
    # Convert paths
    input_base = Path(args.input)
    output_path = Path(args.output)
    
    if not input_base.exists():
        print(f"ERROR: Input directory not found: {input_base}")
        return 1
    
    print(f"\n{'='*60}")
    print(f"Merging Fragment Predictions")
    print(f"{'='*60}")
    print(f"Input: {input_base}")
    print(f"Output: {output_path}")
    print(f"Patient ID: {args.pid}")
    
    # Find all predictions
    print(f"\nScanning for predictions...")
    predictions = find_predictions(input_base, args.pid)
    
    # Check if any predictions found
    total_fragments = sum(len(v) for v in predictions.values())
    if total_fragments == 0:
        print(f"ERROR: No predictions found for patient {args.pid}")
        print(f"  Expected structure: {input_base}/{{anatomy}}/{{fragment_id}}/{args.pid}.mha")
        return 1
    
    print(f"\nFound predictions:")
    for anatomy, fragments in predictions.items():
        n_fragments = len(fragments)
        range_start, range_end = ANATOMY_RANGES[anatomy]
        print(f"  {anatomy}: {n_fragments} fragment(s) (range {range_start}-{range_end})")
        if args.verbose:
            for frag_id in sorted(fragments.keys()):
                print(f"    - Fragment {frag_id}")
    
    # Determine reference shape and metadata
    if args.reference and Path(args.reference).exists():
        print(f"\nUsing reference image: {args.reference}")
        reference_img = sitk.ReadImage(args.reference)
        reference_shape = sitk.GetArrayFromImage(reference_img).shape
    else:
        # Use first prediction to get shape
        print(f"\nNo reference provided, using first prediction for shape")
        first_anatomy = next(iter(predictions.values()))
        first_fragment = next(iter(first_anatomy.values()))
        reference_img = sitk.ReadImage(str(first_fragment))
        reference_shape = sitk.GetArrayFromImage(reference_img).shape
    
    print(f"Reference shape (z,y,x): {reference_shape}")
    
    # Merge predictions
    print(f"\nMerging predictions...")
    merged_data = merge_predictions(predictions, reference_shape)
    
    # Statistics
    unique_labels = np.unique(merged_data)
    n_labels = len(unique_labels[unique_labels > 0])
    
    print(f"\nMerge complete:")
    print(f"  Total voxels: {merged_data.size}")
    print(f"  Non-zero voxels: {np.sum(merged_data > 0)}")
    print(f"  Unique fragment labels: {n_labels}")
    print(f"  Label range: {merged_data.min()} - {merged_data.max()}")
    
    if args.verbose and n_labels > 0:
        print(f"\nLabel distribution:")
        for label in unique_labels[unique_labels > 0][:20]:  # Show first 20
            count = np.sum(merged_data == label)
            print(f"    Label {label}: {count} voxels")
        if n_labels > 20:
            print(f"    ... and {n_labels - 20} more labels")
    
    # Save output
    print(f"\nSaving merged prediction...")
    output_img = sitk.GetImageFromArray(merged_data)
    output_img.CopyInformation(reference_img)
    sitk.WriteImage(output_img, str(output_path))
    
    print(f"✓ Saved: {output_path}")
    
    # Also save a label mapping file
    mapping_path = output_path.with_suffix('.txt') if output_path.suffix == '.mha' else Path(str(output_path) + '_labels.txt')
    
    with open(mapping_path, 'w') as f:
        f.write(f"Fragment Label Mapping for {args.pid}\n")
        f.write(f"{'='*60}\n")
        f.write(f"Input directory: {input_base}\n\n")
        f.write("Anatomy Ranges:\n")
        for anatomy, (start, end) in ANATOMY_RANGES.items():
            f.write(f"  {anatomy}: {start}-{end}\n")
        f.write(f"\nFound fragments:\n")
        for anatomy, fragments in predictions.items():
            f.write(f"  {anatomy}:\n")
            for frag_id in sorted(fragments.keys()):
                if frag_id == "single":
                    f.write(f"    - Single fragment (from Phase 1)\n")
                else:
                    label_value = ANATOMY_RANGES[anatomy][0] + frag_id - 1
                    if label_value > ANATOMY_RANGES[anatomy][1]:
                        label_value = ANATOMY_RANGES[anatomy][1]
                    f.write(f"    - Fragment {frag_id} -> Label {label_value}\n")
    
    print(f"✓ Label mapping saved: {mapping_path}")
    print(f"\n{'='*60}")
    print("Done!")
    
    return 0


if __name__ == "__main__":
    exit(main())
