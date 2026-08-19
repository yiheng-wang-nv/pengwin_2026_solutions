#!/usr/bin/env python3
"""
Keep the connected component from binary prediction that overlaps with a Gaussian heatmap.
Useful for selecting the correct component when predictions are fragmented.
"""

import numpy as np
import cc3d
import SimpleITK as sitk
import argparse
from pathlib import Path

def read_image(filepath):
    """Read image file and return data and metadata."""
    img = sitk.ReadImage(filepath)
    data = sitk.GetArrayFromImage(img)
    return data, img

def write_image(data, reference_img, output_path):
    """Write data to file using reference image metadata."""
    out_img = sitk.GetImageFromArray(data)
    out_img.CopyInformation(reference_img)
    sitk.WriteImage(out_img, output_path)
    print(f"  Saved: {output_path}")

def keep_component_by_heatmap(pred_data, heatmap_data, threshold=0.5):
    """
    Keep only the connected component from prediction that overlaps with heatmap.
    
    Args:
        pred_data: Binary prediction (0/1 or 0/other values)
        heatmap_data: Gaussian heatmap (any range)
        threshold: Minimum heatmap value to consider as overlap (default: 0.5)
    
    Returns:
        Filtered prediction with only the selected component kept
    """
    # Ensure binary prediction (treat any non-zero as foreground)
    binary_pred = (pred_data > 0)
    
    # Get connected components
    labeled_components, num_components = cc3d.connected_components(
        binary_pred, connectivity=26, return_N=True
    )
    
    if num_components == 0:
        print("  No components found in prediction")
        return np.zeros_like(pred_data)
    
    print(f"  Found {num_components} connected component(s)")
    
    # Find which component has maximum overlap with heatmap
    best_component = None
    best_overlap = -1
    
    for comp_id in range(1, num_components + 1):
        # Get mask for this component
        comp_mask = (labeled_components == comp_id)
        
        # Calculate overlap score (sum of heatmap values within component)
        overlap_score = np.sum(heatmap_data[comp_mask])
        
        # Alternative: use number of voxels above threshold
        # overlap_score = np.sum((heatmap_data > threshold) & comp_mask)
        
        print(f"    Component {comp_id}: overlap score = {overlap_score:.2f}")
        
        if overlap_score > best_overlap:
            best_overlap = overlap_score
            best_component = comp_id
    
    # Create output with only the best component
    output = np.zeros_like(pred_data)
    output[labeled_components == best_component] = 1
    
    print(f"  Kept component {best_component} (score: {best_overlap:.2f})")
    
    return output

def process_file(pred_path, heatmap_path, output_path, threshold=0.5, verbose=True):
    """Process a single pair of files."""
    try:
        if verbose:
            print(f"\nProcessing prediction: {Path(pred_path).name}")
            print(f"  Heatmap: {Path(heatmap_path).name}")
        
        # Read images
        pred_data, pred_img = read_image(pred_path)
        heatmap_data, _ = read_image(heatmap_path)
        
        # Check dimensions match
        if pred_data.shape != heatmap_data.shape:
            print(f"  ERROR: Shape mismatch!")
            print(f"    Prediction: {pred_data.shape}")
            print(f"    Heatmap: {heatmap_data.shape}")
            return False
        
        # Keep component that overlaps with heatmap
        output_data = keep_component_by_heatmap(pred_data, heatmap_data, threshold)
        
        # Save output
        write_image(output_data, pred_img, output_path)
        
        # Print statistics
        original_voxels = np.sum(pred_data > 0)
        kept_voxels = np.sum(output_data > 0)
        
        if verbose:
            print(f"  Original voxels: {original_voxels}")
            print(f"  Kept voxels: {kept_voxels}")
            print(f"  Reduction: {(1 - kept_voxels/original_voxels)*100:.1f}%")
        
        return True
        
    except Exception as e:
        print(f"  ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    parser = argparse.ArgumentParser(
        description='Keep the connected component from binary prediction that overlaps with Gaussian heatmap.',
        epilog='Outputs a new binary prediction with only the selected component.'
    )
    
    parser.add_argument(
        '--input_pred',
        type=str,
        required=True,
        help='Input binary prediction file (.mha or .nii.gz)'
    )
    
    parser.add_argument(
        '--input_heatmap',
        type=str,
        required=True,
        help='Input Gaussian heatmap file (.mha or .nii.gz)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output file path (.mha or .nii.gz)'
    )
    
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.5,
        help='Heatmap threshold for overlap (default: 0.5)'
    )
    
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress verbose output'
    )
    
    args = parser.parse_args()
    
    # Check input files exist
    if not Path(args.input_pred).exists():
        print(f"ERROR: Prediction file not found: {args.input_pred}")
        return 1
    
    if not Path(args.input_heatmap).exists():
        print(f"ERROR: Heatmap file not found: {args.input_heatmap}")
        return 1
    
    # Process
    success = process_file(
        args.input_pred,
        args.input_heatmap,
        args.output,
        args.threshold,
        not args.quiet
    )
    
    return 0 if success else 1

if __name__ == "__main__":
    exit(main())
