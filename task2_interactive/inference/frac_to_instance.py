import os
import argparse
import numpy as np
import SimpleITK as sitk
from scipy.ndimage import label
from scipy.spatial import cKDTree
import torch
import torch.nn.functional as F
from tqdm import tqdm

def save_nifti(arr, ref_img, out_path):
    """Save a NIfTI file inheriting spacing / origin / direction from ref_img."""
    out_img = sitk.GetImageFromArray(arr.astype(np.int32))  # instance IDs may be large; use int32
    out_img.CopyInformation(ref_img)
    sitk.WriteImage(out_img, out_path)

def assign_to_nearest(seed_mask, query_mask, label_map):
    """Assign every voxel in `query_mask` to the instance ID of its nearest seed voxel via KD-Tree."""
    coords_seed = np.argwhere(seed_mask > 0)
    coords_query = np.argwhere(query_mask)
    assigned_map = np.zeros_like(seed_mask, dtype=np.int32)

    if len(coords_seed) == 0 or len(coords_query) == 0:
        return assigned_map

    tree = cKDTree(coords_seed)
    dists, idxs = tree.query(coords_query)
    nearest_labels = label_map[tuple(coords_seed[idxs].T)]
    assigned_map[tuple(coords_query.T)] = nearest_labels
    return assigned_map

def dilate_mask(mask, kernel_size=3, device='cpu'):
    """3-D binary dilation via Torch conv3d on CPU or GPU."""
    tensor = torch.from_numpy(mask.astype(np.float32)).to(device)
    tensor = tensor.unsqueeze(0).unsqueeze(0)  # [1, 1, D, H, W]
    kernel = torch.ones((1, 1, kernel_size, kernel_size, kernel_size), dtype=torch.float32, device=device)

    with torch.no_grad():
        out = F.conv3d(tensor, kernel, padding=kernel_size//2)

    dilated = (out > 0).squeeze().cpu().numpy().astype(np.bool_)
    return dilated

def merged_mask_to_instance(merged_mask_path, kernel_size=5, ccf_threshold=100, device='cpu'):
    """Core post-processing: turn a CSM semantic mask into instance segmentation.

    Input label semantics:
        1 = fragment core / boundary (foreground)
        2 = CSM (contact surface between two fragments)
    Returns:
        instance_array (np.ndarray, int32) and the reference sitk image for metadata.
    """
    merged_img = sitk.ReadImage(merged_mask_path)
    merged = sitk.GetArrayFromImage(merged_img)

    # 1. Extract connected components from merged==1 as initial cores; drop small noise.
    core_mask_init = (merged == 1)
    initial_cc_mask, initial_num = label(core_mask_init)
    sizes = [(initial_cc_mask == i).sum() for i in range(1, initial_num+1)]

    kept_indices = [i+1 for i, s in enumerate(sizes) if s >= ccf_threshold]
    mask_keep = np.isin(initial_cc_mask, kept_indices)
    initial_cc_mask_filtered = initial_cc_mask * mask_keep
    merged[(core_mask_init) & (~mask_keep)] = 0

    # Sort kept cores by descending volume and renumber 1..K.
    sizes_kept = [sizes[i-1] for i in kept_indices]
    order = np.argsort(sizes_kept)[::-1]
    remap = np.zeros(initial_num+1, dtype=np.int32)
    for new_idx, idx in enumerate(np.array(kept_indices)[order], start=1):
        remap[idx] = new_idx
    initial_core_label = remap[initial_cc_mask_filtered]

    # 2. For each contact (merged==2) component: if dilation touches exactly one core,
    #    that contact ribbon belongs to that core.
    mask2 = (merged == 2)
    cc_mask, num = label(mask2)
    assign_mask = np.zeros_like(merged, dtype=np.bool_)

    for i in range(1, num+1):
        region_mask = (cc_mask == i)
        if region_mask.sum() < 20:
            continue

        region_dilated = dilate_mask(region_mask, kernel_size=kernel_size, device=device)
        overlap_labels = initial_core_label[region_dilated & (initial_core_label > 0)]
        unique_labels = np.unique(overlap_labels)

        # Touches only one core => this CSM region is part of that core's fragment.
        if len(unique_labels) == 1:
            sel = region_dilated & (initial_core_label == unique_labels[0])
            merged[sel] = 2
            assign_mask[sel] = True

    # 3. Re-filter and renumber the cores that remain unassigned.
    core_mask = (merged == 1) & (~assign_mask)
    cc_mask, num = label(core_mask)
    sizes = [(cc_mask == i).sum() for i in range(1, num+1)]

    kept_indices = [i+1 for i, s in enumerate(sizes) if s >= ccf_threshold]
    sizes_kept = [sizes[i-1] for i in kept_indices]
    order = np.argsort(sizes_kept)[::-1]
    remap = np.zeros(num+1, dtype=np.int32)
    for new_idx, idx in enumerate(np.array(kept_indices)[order], start=1):
        remap[idx] = new_idx

    mask_keep = np.isin(cc_mask, kept_indices)
    cc_mask_filtered = cc_mask * mask_keep
    merged[(core_mask) & (~mask_keep)] = 0
    kernel_cc = remap[cc_mask_filtered]

    # 4. KD-Tree nearest-neighbour assignment for the remaining CSM (==2) and boundary (==1).
    assigned_2 = assign_to_nearest(kernel_cc > 0, assign_mask, kernel_cc)
    current_instance = kernel_cc.copy()
    current_instance[assigned_2 > 0] = assigned_2[assigned_2 > 0]

    merged_1_mask = (merged == 1)
    assigned_1 = assign_to_nearest(current_instance > 0, merged_1_mask, current_instance)
    current_instance[assigned_1 > 0] = assigned_1[assigned_1 > 0]

    merged_2_mask = (merged == 2)
    assigned_2_final = assign_to_nearest(current_instance > 0, merged_2_mask, current_instance)
    current_instance[assigned_2_final > 0] = assigned_2_final[assigned_2_final > 0]

    return current_instance, merged_img

def process_dataset(input_dir, output_dir, kernel_size, ccf_threshold, device):
    """Batch-process every prediction file in a folder."""
    os.makedirs(output_dir, exist_ok=True)
    files = [f for f in os.listdir(input_dir) if f.endswith('.nii') or f.endswith('.nii.gz')]

    if not files:
        print(f"Error: No NIfTI files found in {input_dir}")
        return

    print(f"Found {len(files)} files. Starting processing using {device.upper()}...")

    for fname in tqdm(files, desc="Processing Masks"):
        in_path = os.path.join(input_dir, fname)
        # keep the nnUNet naming convention so results can be submitted as-is
        out_path = os.path.join(output_dir, fname)

        try:
            # release fragmented GPU memory between large volumes
            if device == 'cuda':
                torch.cuda.empty_cache()

            instance_array, ref_img = merged_mask_to_instance(
                in_path,
                kernel_size=kernel_size,
                ccf_threshold=ccf_threshold,
                device=device
            )
            save_nifti(instance_array, ref_img, out_path)

        except Exception as e:
            print(f"\nFailed to process {fname}: {str(e)}")

    print(f"All tasks completed. Results saved to: {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert CSM semantic masks (foreground + contact) into instance segmentation masks.")

    # required
    parser.add_argument("-i", "--input_dir", type=str, required=True,
                        help="Directory containing the CSM model's semantic predictions (3-class .nii.gz).")
    parser.add_argument("-o", "--output_dir", type=str, required=True,
                        help="Directory to write per-fragment instance masks to.")

    # hyperparameters
    parser.add_argument("-k", "--kernel_size", type=int, default=5,
                        help="Kernel size for 3D dilation of CSM regions (default: 5).")
    parser.add_argument("-c", "--ccf_threshold", type=int, default=100,
                        help="Minimum volume (voxels) for an instance core to be kept (default: 100).")

    # hardware
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"],
                        help="Device used for the 3D dilation step (default: cuda).")

    args = parser.parse_args()

    # auto fallback: if cuda requested but unavailable, drop to cpu
    active_device = args.device
    if active_device == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available. Falling back to CPU.")
        active_device = "cpu"

    process_dataset(args.input_dir, args.output_dir, args.kernel_size, args.ccf_threshold, active_device)