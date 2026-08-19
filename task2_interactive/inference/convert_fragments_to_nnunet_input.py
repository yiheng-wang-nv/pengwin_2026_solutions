#!/usr/bin/env python3
"""
Script to create fragment-based Gaussian heatmaps from JSON clicks file.
For each fragment point:
  - Creates foreground heatmap (_0001.mha) for that specific point
  - Creates background heatmap (_0002.mha) using all other points
  - Copies original image as _0000.mha
  - Sets up directories for nnUNet predictions

Output structure:
  temp/predictions/fragments/{anatomy}/{index}/
    - case_0000.mha (original image)
    - case_0001.mha (foreground heatmap for this fragment)
    - case_0002.mha (background heatmap from other fragments)
"""

import argparse
import json
from pathlib import Path
import numpy as np
import SimpleITK as sitk
from scipy.ndimage import gaussian_filter
import shutil

CLASS_MAP = {
    "Sacrum": 1,
    "Left Hip": 2,
    "Right Hip": 3,
    "Femur": 4,
}

ANATOMY_NAMES = {
    1: "sacrum",
    2: "left_hip",
    3: "right_hip",
    4: "femur",
}

# Label ranges for each anatomy
LABEL_RANGES = {
    1: (1, 50),     # Sacrum: 1-50
    2: (51, 100),   # Left Hip: 51-100
    3: (101, 150),  # Right Hip: 101-150
    4: (151, 200),  # Femur: 151-200
}


def gaussian_3d_fast(shape, center, sigma=1.0):
    """Create a 3D Gaussian heatmap, computing np.exp ONLY within a small window
    (radius 7*sigma) around the center instead of over the whole volume. For sigma=1
    the blob is a few voxels; beyond ~7 sigma the value is < 3e-11 and was already ~0,
    so the full-shape result is identical to float precision but ~100x cheaper."""
    out = np.zeros(shape, dtype=np.float32)
    cz, cy, cx = center
    R = int(np.ceil(7.0 * sigma)) + 1
    z0, z1 = max(0, int(np.floor(cz)) - R), min(shape[0], int(np.ceil(cz)) + R + 1)
    y0, y1 = max(0, int(np.floor(cy)) - R), min(shape[1], int(np.ceil(cy)) + R + 1)
    x0, x1 = max(0, int(np.floor(cx)) - R), min(shape[2], int(np.ceil(cx)) + R + 1)
    if z0 >= z1 or y0 >= y1 or x0 >= x1:
        return out  # center outside volume

    z = np.arange(z0, z1, dtype=np.float32) - cz
    y = np.arange(y0, y1, dtype=np.float32) - cy
    x = np.arange(x0, x1, dtype=np.float32) - cx
    dist_sq = (z ** 2)[:, np.newaxis, np.newaxis] + (y ** 2)[np.newaxis, :, np.newaxis] + (x ** 2)[np.newaxis, np.newaxis, :]
    out[z0:z1, y0:y1, x0:x1] = np.exp(-dist_sq / (2 * sigma ** 2))
    return out


def parse_points_by_fragment(json_path):
    """
    Parse JSON and return points organized by fragment.
    Each point is treated as a separate fragment.
    Returns: dict with anatomy as key, list of points (each point is a dict with index and coordinates)
    """
    with open(json_path) as f:
        data = json.load(f)

    fragments = {1: [], 2: [], 3: [], 4: []}  # anatomy_id -> list of points
    
    for idx, p in enumerate(data["points"]):
        name = p["name"]
        coord = p["point"]  # [x, y, z]
        
        # Convert to z, y, x (numpy indexing)
        coord = [coord[0], coord[1], coord[2]]
        
        # Check which class this point belongs to
        assigned = False
        for keyword, cls_id in KEYWORD_MAP.items():
            if keyword in name:
                fragments[cls_id].append({
                    'index': idx,
                    'coord': coord,
                    'name': name
                })
                assigned = True
                break
        
        if not assigned:
            print(f"  WARNING: Could not classify point: {name}")
    
    return fragments


def create_foreground_background_heatmaps(image, all_points, current_point_idx, sigma=1.0):
    """
    Create foreground heatmap for a specific point and background heatmap for all other points.
    
    Args:
        image: SimpleITK image
        all_points: List of all point coordinates for this anatomy
        current_point_idx: Index of the current point to use as foreground
        sigma: Gaussian sigma
    
    Returns:
        foreground_heatmap, background_heatmap
    """
    shape = sitk.GetArrayFromImage(image).shape
    
    # Create foreground (single point)
    foreground = np.zeros(shape, dtype=np.float32)
    point = all_points[current_point_idx]
    foreground += gaussian_3d_fast(shape, point, sigma=sigma)
    
    # Normalize foreground
    if np.max(foreground) > 0:
        foreground = foreground / np.max(foreground)
    
    # Create background (all other points)
    background = np.zeros(shape, dtype=np.float32)
    for idx, pt in enumerate(all_points):
        if idx != current_point_idx:
            background += gaussian_3d_fast(shape, pt, sigma=sigma)
    
    # Normalize background
    if np.max(background) > 0:
        background = background / np.max(background)
    
    return foreground, background


def save_image(data, reference_img, out_path):
    """Save image data using reference image metadata."""
    img = sitk.GetImageFromArray(data.astype(np.float32))
    img.CopyInformation(reference_img)
    
    if str(out_path).endswith('.nii.gz'):
        sitk.WriteImage(img, str(out_path), True)
    else:
        sitk.WriteImage(img, str(out_path))


def main():
    parser = argparse.ArgumentParser(
        description="Create fragment-based heatmaps for nnUNet inference",
        epilog="For each fragment, creates foreground/background heatmaps and sets up prediction directories"
    )

    parser.add_argument(
        "--json",
        type=str,
        required=True,
        help="Path to JSON file containing click points"
    )

    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to input image (.mha or .nii.gz)"
    )

    parser.add_argument(
        "--output_base",
        type=str,
        required=True,
        help="Base output directory (e.g., temp/predictions/fragments/)"
    )

    parser.add_argument(
        "--basename",
        type=str,
        default=None,
        help="Basename for output files (default: stem of image filename)"
    )

    parser.add_argument(
        "--sigma",
        type=float,
        default=1.0,
        help="Sigma for Gaussian heatmap (default: 1.0)"
    )

    parser.add_argument(
        "--method",
        type=str,
        choices=["direct", "filter"],
        default="direct",
        help="Method for generating heatmaps"
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed information"
    )

    args = parser.parse_args()

    # Validate input files
    json_path = Path(args.json)
    if not json_path.exists():
        print(f"ERROR: JSON file not found: {json_path}")
        return 1

    image_path = Path(args.image)
    if not image_path.exists():
        print(f"ERROR: Image file not found: {image_path}")
        return 1

    # Setup output base directory
    output_base = Path(args.output_base)
    output_base.mkdir(parents=True, exist_ok=True)

    # Determine basename
    if args.basename is None:
        basename = image_path.stem
        if basename.endswith("_0000"):
            basename = basename[:-5]
    else:
        basename = args.basename

    # Get file extension
    extension = image_path.suffix
    if image_path.name.endswith('.nii.gz'):
        extension = '.nii.gz'

    print(f"\n{'='*60}")
    print(f"Fragment-based Heatmap Generation")
    print(f"{'='*60}")
    print(f"JSON: {json_path}")
    print(f"Image: {image_path}")
    print(f"Output base: {output_base}")
    print(f"Basename: {basename}")
    print(f"Sigma: {args.sigma}")
    print(f"Method: {args.method}")

    # Read image
    print(f"\nReading image...")
    img = sitk.ReadImage(str(image_path))
    data_shape = sitk.GetArrayFromImage(img).shape
    print(f"  Image shape (z,y,x): {data_shape}")

    # Parse JSON and group by anatomy
    print(f"\nParsing JSON and grouping by anatomy...")
    
    # Import KEYWORD_MAP inside function to avoid global issues
    global KEYWORD_MAP
    KEYWORD_MAP = {
        "Sacrum": 1,
        "Left Hip": 2,
        "Right Hip": 3,
        "Femur": 4,
    }
    
    fragments = parse_points_by_fragment(json_path)
    
    # Print summary and filter anatomies with only one point
    print(f"\nFragment summary:")
    filtered_fragments = {}
    for anatomy_id, points in fragments.items():
        anatomy_name = ANATOMY_NAMES[anatomy_id]
        n_points = len(points)
        

        filtered_fragments[anatomy_id] = points
    
    # Save original image once (will be copied to each fragment directory)
    print(f"\nSetting up fragment directories...")
    
    total_fragments = 0
    for anatomy_id, points in filtered_fragments.items():
        if points is not None:  # Has multiple fragments
            total_fragments += len(points)
    
    print(f"Total fragments to process: {total_fragments}")
    
    # Process each fragment
    current_fragment_id = 0
    all_coords = []
    for anatomy_id, points in filtered_fragments.items():
        all_coords += [p['coord'] for p in points]
    for anatomy_id, points in filtered_fragments.items():

            
        anatomy_name = ANATOMY_NAMES[anatomy_id]

        if len(points) <= 1:
            print(f"Skipping {anatomy_name} as it has {len(points)} points...")
            current_fragment_id += len(points)
            continue

        
        # Process each point as a separate fragment
        for point_idx, point_info in enumerate(points):
            fragment_index = point_info['index'] 
            point_coord = point_info['coord']
            point_name = point_info['name']
            
            # Create directory for this fragment
            # Structure: {output_base}/{anatomy}/{fragment_index}/
            fragment_dir = output_base / anatomy_name / str(50 * (anatomy_id - 1) + point_idx + 1)
            fragment_dir.mkdir(parents=True, exist_ok=True)
            
            print(f"\n[{current_fragment_id + 1}/{total_fragments}] Processing {anatomy_name} fragment {str(50 * (anatomy_id - 1) + point_idx + 1)}")
            if args.verbose:
                print(f"  Point: {point_name}")
                print(f"  Coordinates (z,y,x): {point_coord}")
                print(f"  Output dir: {fragment_dir}")
            
            # Create foreground and background heatmaps
            foreground, background = create_foreground_background_heatmaps(
                img, all_coords, current_fragment_id, sigma=args.sigma
            )
            
            # Save files
            # _0000: original image
            output_image_path = fragment_dir / f"{basename}_0000{extension}"
            save_image(sitk.GetArrayFromImage(img), img, output_image_path)
            
            # _0001: foreground heatmap
            output_foreground_path = fragment_dir / f"{basename}_0001{extension}"
            save_image(foreground, img, output_foreground_path)
            
            # _0002: background heatmap
            output_background_path = fragment_dir / f"{basename}_0002{extension}"
            save_image(background, img, output_background_path)
            current_fragment_id += 1

            if args.verbose:
                fg_max = foreground.max()
                bg_max = background.max()
                print(f"  Foreground max: {fg_max:.4f}")
                print(f"  Background max: {bg_max:.4f}")
                print(f"  Files saved to: {fragment_dir}")
    
    # Handle single-fragment anatomies (just copy original prediction if it exists)
    print(f"\n{'='*60}")
    print(f"Setting up directories for single-fragment anatomies...")
    
    for anatomy_id, points in filtered_fragments.items():
        if points is None:
            anatomy_name = ANATOMY_NAMES[anatomy_id]
            print(f"  {anatomy_name}: Copying original prediction (single fragment)")
            
            # Create a marker file to indicate that this anatomy should just copy the original
            marker_dir = output_base / anatomy_name / "single_fragment"
            marker_dir.mkdir(parents=True, exist_ok=True)
            
            # Create a info file
            info_file = marker_dir / "info.txt"
            with open(info_file, 'w') as f:
                f.write(f"Anatomy: {anatomy_name}\n")
                f.write(f"Status: Single fragment detected - use original segmentation\n")
                f.write(f"Original image: {image_path}\n")
                f.write(f"JSON: {json_path}\n")
            
            # Also copy the original image to this directory for completeness
            output_image_path = marker_dir / f"{basename}_0000{extension}"
            save_image(sitk.GetArrayFromImage(img), img, output_image_path)
    

    
    print(f"\n✓ Setup complete!")
    print(f"  Total fragments processed: {current_fragment_id}")
    print(f"  Single-fragment anatomies: {sum(1 for v in filtered_fragments.values() if v is None)}")
    print(f"\nNext steps:")
    print(f"  1. Run nnUNet prediction on each fragment directory")
    print(f"  2. Use the prediction merging script to combine results into labels 0-200")
    
    return 0


if __name__ == "__main__":
    exit(main())
