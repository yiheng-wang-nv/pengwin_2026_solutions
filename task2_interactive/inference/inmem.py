"""In-memory Task 2 input building — no per-fragment .mha write/read round-trip.

The file-based pipeline re-writes the FULL CT (channel _0000) once per fragment and
reads all 3 channels back for every nnUNet call: ~12 GB of disk I/O for a 15-fragment
case, the last GC-environment-dependent bottleneck after GPU resampling. This builds
the (C, Z, Y, X) channel arrays directly and feeds predict_single_npy_array, reusing
the EXACT convert_* parse/heatmap functions so arrays are byte-identical to the files
(SimpleITKIO would have read them back as float32 with the CT's geometry — replicated
here in build_props/_stack).
"""
import numpy as np

import convert_anatomy_to_nnunet_input as _anat
import convert_fragments_to_nnunet_input as _frag
import fastheat
import fastheat as _fh

# parse_points_by_fragment reads the module global KEYWORD_MAP, which the CLI only sets
# inside main(); set it here (identical to CLASS_MAP) so the function works when imported.
_frag.KEYWORD_MAP = {"Sacrum": 1, "Left Hip": 2, "Right Hip": 3, "Femur": 4}

ANATOMY_NAMES = _frag.ANATOMY_NAMES  # {1:'sacrum',2:'left_hip',3:'right_hip',4:'femur'}


def build_props(ref_img):
    """Replicate SimpleITKIO.read_images' properties dict from a reference sitk image."""
    sp = ref_img.GetSpacing()
    return {
        "sitk_stuff": {"spacing": sp, "origin": ref_img.GetOrigin(),
                       "direction": ref_img.GetDirection()},
        "spacing": list(np.abs(np.array(list(sp)[::-1], dtype=float))),
    }


def _stack(arrays):
    # SimpleITKIO returns np.vstack of per-channel (1,Z,Y,X), float32
    return np.vstack([a[None] for a in arrays]).astype(np.float32)


def anatomy_input(ct_img, ct_arr, json_path, sigma=1.0, verify=False):
    """Build the (5, Z, Y, X) anatomy input in a single allocation.

    Bit-identical to the original `_stack([ct_arr, hms[1..4]])` -- asserted, in-container
    and natively, on the largest case, via `verify=True`.

    This is a MEMORY fix, and the memory is the binding constraint: Grand Challenge caps
    DRAM at 16 GiB and a dev-phase run of the flat pipeline died with
    `numpy._core._exceptions._ArrayMemoryError: Unable to allocate 932 MiB` inside
    nnU-Net's `resample_data_or_seg`, which pads each channel to float64. That resample
    is 5x more expensive here than in Task 1 purely because ds456 takes 5 channels where
    ds001 takes 1 -- it is the one place the two pipelines genuinely differ.

    The original path allocated, per case: one full-volume float32 inside gaussian_3d_fast
    for EVERY click, four more for the per-class heatmaps, another inside
    create_heatmaps_fast purely to read `.shape`, and finally the stacked copy. On a
    132M-voxel case that is ~5.3 GiB of churn immediately before nnU-Net asks for a
    contiguous 932 MiB, so it costs both headroom and heap fragmentation.

    This was once reverted on the strength of a cgroup sample that showed the container
    peak unmoved (13291 -> 13387 MiB on case 283). That sampler polled every 2 s and case
    283 fits anyway; the dev-phase failure is the better evidence and it says the
    transient matters.
    """
    grouped = _anat.parse_points(json_path)
    if verify:
        _fh.verify_anatomy_against_original(ct_img, ct_arr, grouped, sigma)
    buf = np.empty((5,) + ct_arr.shape, dtype=np.float32)
    buf[0] = ct_arr
    fastheat.fill_anatomy(buf, grouped, sigma)
    return buf


def fragment_iter(ct_img, ct_arr, json_path, sigma=1.0):
    """Yield (anatomy_name, frag_label, data_array, fg_heatmap) for each multi-fragment
    click — replicating convert_fragments' all_coords / current_fragment_id indexing and
    the 50*(anatomy_id-1)+point_idx+1 label.

    The (3, Z, Y, X) buffer and the two heatmaps are allocated ONCE and rewritten per
    fragment. Only channels 1 and 2 change between fragments; the CT in channel 0 is
    identical every time, so the original per-fragment `_stack([ct_arr, fg, bg])` was
    re-copying 1.1 GB and `create_foreground_background_heatmaps` was allocating a
    fresh 367 MB volume per click (29 of them for the background of a 30-click case).
    Together that was ~3.5 s per fragment; in place it is ~0.2 s, which matters because
    the organisers' test-phase failure is a timeout driven by fragment count.

    CONSEQUENCE FOR CALLERS: the yielded array and heatmap are VIEWS onto reusable
    buffers and are invalidated by the next iteration. That is safe for the existing
    consumer (it predicts and runs keep-clicked before advancing) but anything that
    wants to retain them across iterations must copy.
    """
    fragments = _frag.parse_points_by_fragment(json_path)
    all_coords = []
    for aid in (1, 2, 3, 4):
        all_coords += [p["coord"] for p in fragments[aid]]

    buf = np.empty((3,) + ct_arr.shape, dtype=np.float32)
    buf[0] = ct_arr                      # constant across fragments — write once
    fg, bg = buf[1], buf[2]              # views: filling them fills the buffer

    cur = 0
    for aid in (1, 2, 3, 4):
        pts = fragments[aid]
        if len(pts) <= 1:
            cur += len(pts)
            continue
        name = ANATOMY_NAMES[aid]
        for point_idx in range(len(pts)):
            fastheat.fill_fg_bg(fg, bg, all_coords, cur, sigma=sigma)
            label = 50 * (aid - 1) + point_idx + 1
            # the click itself is yielded so the caller can decide which sliding-window
            # positions this fragment can actually reach (see patchcache)
            yield name, label, buf, fg, tuple(all_coords[cur])
            cur += 1


def single_fragment_anatomy_ids(json_path):
    """Anatomy ids with <=1 click point (their whole anatomy mask is the single fragment)."""
    fragments = _frag.parse_points_by_fragment(json_path)
    return [aid for aid in (1, 2, 3, 4) if len(fragments[aid]) <= 1]
