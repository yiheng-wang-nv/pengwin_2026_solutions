#!/usr/bin/env python3
"""
Map anatomy labels to fragment labels based on maximum overlap.
For each anatomy label, find the fragment label that has the most overlap,
then fill ALL voxels within that anatomy region (including background in fragments)
with that fragment label.
"""

import numpy as np
import SimpleITK as sitk
import argparse
from pathlib import Path

def read_image(filepath):
    """Read image file and return data and metadata."""
    img = sitk.ReadImage(filepath)
    data = sitk.GetArrayFromImage(img)
    return data, img

def write_image(data, reference_img, output_path):
    """Write the FINAL PENGWIN label map as uint8 + compressed. Grand Challenge's
    segmentation validator requires an integer (uint8) label image with <=201 segments;
    a uint16 output fails with 'segments could not be determined'. Labels are 0-200 so
    uint8 is exact. (Matches the Task 1 container, which passed GC as uint8.)"""
    out_img = sitk.GetImageFromArray(np.asarray(data).astype(np.uint8))
    out_img.CopyInformation(reference_img)
    sitk.WriteImage(out_img, output_path, True)
    print(f"  Saved: {output_path}")

def fill_anatomy_with_best_fragment(anat_data, frag_data, verbose=True):
    """
    For each anatomy label, find best matching fragment label and fill the entire
    anatomy region with that fragment label (overwriting zeros only).
    
    Args:
        anat_data: Array with anatomy labels (1,2,3,4)
        frag_data: Array with fragment labels (1-200)
    
    Returns:
        filled_data: Updated fragment data with anatomy regions filled
        mapping: Dictionary mapping anatomy_label -> fragment_label
    """
    # Make a copy of fragment data to modify
    filled_data = frag_data.copy()
    
    # Get unique anatomy labels (excluding 0/background)
    anat_labels = np.unique(anat_data)
    anat_labels = anat_labels[anat_labels > 0]
    
    if verbose:
        print(f"  Found anatomy labels: {anat_labels}")
    
    mapping = {}
    
    for anat_label in anat_labels:
        # Create mask for current anatomy region
        anat_mask = (anat_data == anat_label)
        
        # Get fragment labels within this anatomy region
        frag_labels_in_region = frag_data[anat_mask]
        
        # Exclude background (0) when finding best match
        non_zero_labels = frag_labels_in_region[frag_labels_in_region > 0]
        
        if len(non_zero_labels) == 0:
            if verbose:
                print(f"    Label {anat_label}: No non-zero fragments in region, skipping")
            mapping[anat_label] = None
            continue
        
        # Count occurrences of each fragment label
        unique_frags, counts = np.unique(non_zero_labels, return_counts=True)
        
        # Find fragment with maximum overlap
        max_idx = np.argmax(counts)
        best_frag = unique_frags[max_idx]
        best_count = counts[max_idx]
        total_voxels = np.sum(anat_mask)
        
        if verbose:
            percentage = (best_count / total_voxels) * 100
            print(f"    Label {anat_label} -> Fragment {best_frag} "
                  f"(overlap: {best_count}/{total_voxels} voxels, {percentage:.1f}%)")
        
        mapping[anat_label] = best_frag
        
        # Fill ALL voxels in anatomy region (including zeros) with best fragment
        # BUT only where frag_data is zero (don't overwrite existing non-zero fragments)
        fill_mask = anat_mask & (frag_data == 0)
        num_filled = np.sum(fill_mask)
        
        if num_filled > 0:
            filled_data[fill_mask] = best_frag
            if verbose:
                print(f"      Filled {num_filled} background voxels with fragment {best_frag}")
    
    return filled_data, mapping

def process_file(anat_path, frag_path, output_path, verbose=True):
    """Process a pair of files."""
    try:
        if verbose:
            print(f"\nProcessing:")
            print(f"  Anatomy: {Path(anat_path).name}")
            print(f"  Fragments: {Path(frag_path).name}")
        
        # Read images
        anat_data, anat_img = read_image(anat_path)
        frag_data, _ = read_image(frag_path)
        
        # Check dimensions match
        if anat_data.shape != frag_data.shape:
            print(f"  ERROR: Shape mismatch!")
            print(f"    Anatomy: {anat_data.shape}")
            print(f"    Fragments: {frag_data.shape}")
            return False
        
        # Fill anatomy regions with best matching fragments
        filled_data, mapping = fill_anatomy_with_best_fragment(anat_data, frag_data, verbose)
        
        # Save output
        write_image(filled_data, anat_img, output_path)
        
        if verbose:
            print(f"\n  Summary of changes:")
            for anat_label, frag_label in mapping.items():
                if frag_label is not None:
                    # Count how many voxels were changed for this anatomy region
                    anat_mask = (anat_data == anat_label)
                    original_nonzero = np.sum((anat_mask) & (frag_data > 0))
                    final_filled = np.sum((anat_mask) & (filled_data == frag_label))
                    print(f"    Anatomy {anat_label} -> Fragment {frag_label}: "
                          f"{original_nonzero} original, {final_filled} total")
        
        return True
        
    except Exception as e:
        print(f"  ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    parser = argparse.ArgumentParser(
        description='Fill anatomy regions with best matching fragment labels.',
        epilog='For each anatomy label, finds the most overlapping fragment and fills all '
               'background voxels in that region with that fragment label.'
    )
    
    parser.add_argument(
        '--anat_pred',
        type=str,
        required=True,
        help='Anatomy prediction file (.mha or .nii.gz) with labels 1,2,3,4'
    )
    
    parser.add_argument(
        '--frag_pred',
        type=str,
        required=True,
        help='Fragment prediction file (.mha or .nii.gz) with labels 1-200'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output file path (.mha or .nii.gz)'
    )
    
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress verbose output'
    )
    
    args = parser.parse_args()
    
    # Check input files exist
    if not Path(args.anat_pred).exists():
        print(f"ERROR: Anatomy file not found: {args.anat_pred}")
        return 1
    
    if not Path(args.frag_pred).exists():
        print(f"ERROR: Fragment file not found: {args.frag_pred}")
        return 1
    
    # Process
    success = process_file(
        args.anat_pred,
        args.frag_pred,
        args.output,
        not args.quiet
    )
    
    return 0 if success else 1

if __name__ == "__main__":
    exit(main())