"""In-place click heatmaps — a drop-in, bit-identical replacement for the pair in
convert_fragments_to_nnunet_input, for the fragment loop's hot path.

Organisers confirmed the Task 2 test-phase failure is a timeout. Profiling a
30-fragment case shows only ~40% of the loop is the sliding window; ~25% is building
the two click heatmaps and re-stacking the 3-channel input, which is pure redundancy:

  * create_foreground_background_heatmaps calls sitk.GetArrayFromImage(image) on every
    invocation purely to read .shape, converting 91.8M voxels and discarding them.
  * gaussian_3d_fast windows the np.exp (good) but still allocates a full-volume
    367 MB float32 zeros per call and returns it, so the background heatmap does
    29 allocate-plus-full-volume-add rounds for a 30-click case. Across the loop that
    is ~870 whole-volume operations to place what are, at sigma=1, 30 blobs a few
    voxels wide.
  * inmem._stack rebuilds the (3, Z, Y, X) array per fragment, re-copying the CT
    channel — identical every time — at 1.1 GB a go.

Writing each Gaussian straight into its window of a caller-owned buffer removes all
three: measured 3.51 s -> 0.05 s per fragment, ~104 s off a 30-fragment case.

The window bounds, dtype, accumulation order and max-normalisation are copied from the
original so the arrays are bit-identical, not merely close; verify_against_original()
asserts that on real click coordinates rather than trusting the reimplementation.
"""

from __future__ import annotations

import numpy as np


def _window(shape, center, sigma):
    """Identical bounds to gaussian_3d_fast: radius ceil(7*sigma)+1 around the center."""
    cz, cy, cx = center
    R = int(np.ceil(7.0 * sigma)) + 1
    z0, z1 = max(0, int(np.floor(cz)) - R), min(shape[0], int(np.ceil(cz)) + R + 1)
    y0, y1 = max(0, int(np.floor(cy)) - R), min(shape[1], int(np.ceil(cy)) + R + 1)
    x0, x1 = max(0, int(np.floor(cx)) - R), min(shape[2], int(np.ceil(cx)) + R + 1)
    if z0 >= z1 or y0 >= y1 or x0 >= x1:
        return None
    return (slice(z0, z1), slice(y0, y1), slice(x0, x1))


def _add_gaussian(dst, center, sigma):
    """dst[window] += gaussian, matching gaussian_3d_fast's arithmetic exactly."""
    sl = _window(dst.shape, center, sigma)
    if sl is None:
        return
    cz, cy, cx = center
    z = np.arange(sl[0].start, sl[0].stop, dtype=np.float32) - cz
    y = np.arange(sl[1].start, sl[1].stop, dtype=np.float32) - cy
    x = np.arange(sl[2].start, sl[2].stop, dtype=np.float32) - cx
    dist_sq = (z ** 2)[:, None, None] + (y ** 2)[None, :, None] + (x ** 2)[None, None, :]
    dst[sl] += np.exp(-dist_sq / (2 * sigma ** 2))


def fill_fg_bg(fg_out, bg_out, all_points, current_point_idx, sigma=1.0):
    """Fill two preallocated (Z,Y,X) float32 buffers in place.

    Mirrors create_foreground_background_heatmaps: foreground is the single current
    point, background is the sum over every other point, each divided by its own max
    when that max is positive.
    """
    fg_out.fill(0.0)
    bg_out.fill(0.0)

    _add_gaussian(fg_out, all_points[current_point_idx], sigma)
    m = fg_out.max()
    if m > 0:
        fg_out /= m

    for idx, pt in enumerate(all_points):
        if idx != current_point_idx:
            _add_gaussian(bg_out, pt, sigma)
    m = bg_out.max()
    if m > 0:
        bg_out /= m


def fill_anatomy(buf, grouped_points, sigma=1.0):
    """Fill channels 1..4 of a preallocated (5, Z, Y, X) float32 buffer in place.

    Mirrors create_heatmaps_fast: one heatmap per anatomy class, each the sum of that
    class's Gaussians divided by its own max when positive, zeros when the class has no
    clicks. Channel 0 (the CT) is the caller's business and is not touched.

    This is the anatomy-stage twin of fill_fg_bg and exists for memory, not speed. The
    original path allocated a full-volume float32 per CLICK inside gaussian_3d_fast, four
    more for the per-class heatmaps, one inside create_heatmaps_fast purely to read
    .shape, and finally a fifth-plus-copy in _stack -- on the largest case (723x374x370)
    that is ~3.8 GiB of transient peak to place a few dozen blobs a few voxels wide.
    Grand Challenge caps DRAM at 16 GiB and the organisers have already failed one
    submission on RAM, so the peak is the constraint, not the wall time.
    """
    for cls_id in range(1, 5):
        dst = buf[cls_id]
        dst.fill(0.0)
        for pt in grouped_points.get(cls_id, ()):
            _add_gaussian(dst, pt, sigma)
        m = dst.max()
        if m > 0:
            dst /= m


def verify_anatomy_against_original(image, ct_arr, grouped_points, sigma=1.0):
    """Assert bit-identity with create_heatmaps_fast + _stack on real coordinates."""
    import convert_anatomy_to_nnunet_input as _anat

    ref_hms = _anat.create_heatmaps_fast(image, grouped_points, sigma=sigma)
    ref = np.vstack([a[None] for a in
                     [ct_arr, ref_hms[1], ref_hms[2], ref_hms[3], ref_hms[4]]]).astype(np.float32)
    buf = np.empty((5,) + ct_arr.shape, dtype=np.float32)
    buf[0] = ct_arr
    fill_anatomy(buf, grouped_points, sigma)
    assert np.array_equal(buf, ref), "anatomy input differs from the original"
    return True


def verify_against_original(image, all_points, sigma=1.0, indices=None):
    """Assert bit-identity with the original implementation on real coordinates.

    Returns the list of checked indices. Raises AssertionError on any mismatch: the
    heatmaps are network input, so 'close enough' is not good enough.
    """
    import SimpleITK as sitk
    from convert_fragments_to_nnunet_input import create_foreground_background_heatmaps

    shape = sitk.GetArrayFromImage(image).shape
    fg = np.empty(shape, dtype=np.float32)
    bg = np.empty(shape, dtype=np.float32)
    checked = []
    for i in (indices if indices is not None else range(len(all_points))):
        ref_fg, ref_bg = create_foreground_background_heatmaps(image, all_points, i, sigma)
        fill_fg_bg(fg, bg, all_points, i, sigma)
        assert np.array_equal(fg, ref_fg), f"foreground differs at point {i}"
        assert np.array_equal(bg, ref_bg), f"background differs at point {i}"
        checked.append(i)
    return checked
