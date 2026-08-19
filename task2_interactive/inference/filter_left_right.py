#!/usr/bin/env python3
"""
Script to correct left/right hip labels in nnUNet segmentations based on sacrum centroid.
Labels: 1=sacrum, 2=left hip, 3=right hip, 4=femur (unchanged)
Processes all .mha and .nii.gz files in a directory and overwrites them.
"""

import numpy as np
import cc3d
import SimpleITK as sitk
import argparse
from pathlib import Path
from tqdm import tqdm

def read_image(filepath):
    """Read image file (MHA or NIfTI) and return image data and metadata."""
    img = sitk.ReadImage(filepath)
    data = sitk.GetArrayFromImage(img)
    return data, img

def write_image(data, reference_img, output_path):
    """Write data to file using reference image metadata."""
    out_img = sitk.GetImageFromArray(data)
    out_img.CopyInformation(reference_img)
    
    # Determine output format based on file extension
    if str(output_path).endswith('.nii.gz'):
        sitk.WriteImage(out_img, output_path, True)  # True for compression
    else:
        sitk.WriteImage(out_img, output_path)
    print(f"  Overwritten: {output_path}")

def process_file_fast(filepath, connectivity=6, verbose=False):
    """Process a single file using vectorized operations."""
    try:
        if verbose:
            print(f"\nProcessing: {filepath.name}")
        
        # Read image
        data, reference_img = read_image(filepath)
        
        # Check for required labels
        unique_labels = np.unique(data)
        required = {1, 2, 3}
        if not required.issubset(set(unique_labels)):
            missing = required - set(unique_labels)
            if verbose:
                print(f"  Warning: Missing labels {missing}. Skipping...")
            return False
        
        # Compute sacrum centroid (fast)
        sacrum_mask = (data == 1)
        sacrum_centroid = np.array(np.where(sacrum_mask)).mean(axis=1)
        
        if verbose:
            print(f"  Sacrum centroid (z,y,x): {sacrum_centroid}")
        
        # Get connected components for both hips at once
        # Create labeled image where each component has unique ID
        hip_mask = (data == 2) | (data == 3)
        labeled_components, num_components = cc3d.connected_components(
            hip_mask, connectivity=connectivity, return_N=True
        )
        
        if num_components == 0:
            if verbose:
                print(f"  No hip components found")
            return False
        
        # Calculate centroids for all components at once using bincount
        # Get coordinates of all component voxels
        coords = np.array(np.where(labeled_components > 0)).T
        comp_ids = labeled_components[coords[:, 0], coords[:, 1], coords[:, 2]]
        
        # Calculate centroids using vectorized operations
        centroids = {}
        unique_comps = np.unique(comp_ids)
        
        for comp_id in unique_comps:
            mask = comp_ids == comp_id
            comp_coords = coords[mask]
            centroids[comp_id] = comp_coords.mean(axis=0)
        
        # Determine original labels for each component (majority vote)
        comp_original_labels = {}
        for comp_id in unique_comps:
            mask = comp_ids == comp_id
            # Get original labels for these positions
            orig_labels = data[coords[mask, 0], coords[mask, 1], coords[mask, 2]]
            # Use mode (most frequent) as the component's original label
            comp_original_labels[comp_id] = np.bincount(orig_labels).argmax()
        
        # Determine new labels based on centroid position
        needs_flip = False
        comp_new_labels = {}
        
        for comp_id in unique_comps:
            orig_label = comp_original_labels[comp_id]
            centroid = centroids[comp_id]
            
            # Compare x-coordinate (index 2) with sacrum centroid
            if orig_label == 2:  # Was labeled as left
                if centroid[2] > sacrum_centroid[2]:
                    new_label = 2  # Keep left
                else:
                    new_label = 3  # Flip to right
                    needs_flip = True
            else:  # orig_label == 3, right hip
                if centroid[2] < sacrum_centroid[2]:
                    new_label = 3  # Keep right
                else:
                    new_label = 2  # Flip to left
                    needs_flip = True
            
            comp_new_labels[comp_id] = new_label
            
            if verbose:
                side = "left" if orig_label == 2 else "right"
                new_side = "left" if new_label == 2 else "right"
                print(f"    Component {comp_id} ({side}): x={centroid[2]:.1f} -> {new_side}")
        
        # If no flips needed, skip
        if not needs_flip:
            if verbose:
                print(f"  No correction needed, skipping write")
            return True
        
        # Create new segmentation efficiently
        mapped_data = data.copy()
        
        # Reset hip labels
        mapped_data[(mapped_data == 2) | (mapped_data == 3)] = 0
        
        # Apply new labels to all components at once using vectorized assignment
        for comp_id in unique_comps:
            new_label = comp_new_labels[comp_id]
            comp_mask = (labeled_components == comp_id)
            mapped_data[comp_mask] = new_label
        
        # Verify sacrum and femur remain unchanged
        if np.any((data == 1) != (mapped_data == 1)) or np.any((data == 4) != (mapped_data == 4)):
            print(f"  WARNING: Sacrum or femur was modified! This should not happen.")
            return False
        
        # Overwrite file
        write_image(mapped_data, reference_img, filepath)
        
        if verbose:
            print(f"  Left hip voxels: {np.sum(mapped_data == 2)}")
            print(f"  Right hip voxels: {np.sum(mapped_data == 3)}")
        
        return True
        
    except Exception as e:
        print(f"  ERROR processing {filepath.name}: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    parser = argparse.ArgumentParser(
        description='Correct left/right hip labels in all .mha and .nii.gz files in a directory.',
        epilog='Labels: 1=sacrum, 2=left hip, 3=right hip, 4=femur (unchanged)\nFiles are overwritten in place!'
    )
    
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Input directory path containing .mha and/or .nii.gz files'
    )
    
    parser.add_argument(
        '--connectivity',
        type=int,
        default=6,
        choices=[6, 18, 26],
        help='Connectivity for connected components (default: 6)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print detailed information for each file'
    )
    
    parser.add_argument(
        '--no-progress',
        action='store_true',
        help='Disable progress bar'
    )
    
    args = parser.parse_args()
    
    # Check if input is a directory
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"ERROR: Path does not exist: {args.input}")
        return 1
    
    if not input_path.is_dir():
        print(f"ERROR: Input must be a directory, not a file: {args.input}")
        return 1
    
    # Find all .mha and .nii.gz files
    mha_files = list(input_path.glob("*.mha"))
    nii_files = list(input_path.glob("*.nii.gz"))
    all_files = mha_files + nii_files
    
    if not all_files:
        print(f"No .mha or .nii.gz files found in {args.input}")
        return 1
    
    print(f"Found {len(all_files)} file(s) to process:")
    for f in all_files:
        print(f"  - {f.name}")
    print()
    
    # Process files
    success_count = 0
    fail_count = 0
    
    iterator = all_files
    if not args.no_progress and not args.verbose:
        iterator = tqdm(all_files, desc="Processing files", unit="file")
    
    for filepath in iterator:
        result = process_file_fast(filepath, args.connectivity, args.verbose)
        if result:
            success_count += 1
        else:
            fail_count += 1
    
    # Print summary
    print(f"\n" + "="*50)
    print(f"SUMMARY:")
    print(f"  Total files: {len(all_files)}")
    print(f"  Successfully processed: {success_count}")
    print(f"  Failed/Skipped: {fail_count}")
    print("="*50)
    
    return 0 if fail_count == 0 else 1

if __name__ == "__main__":
    exit(main())