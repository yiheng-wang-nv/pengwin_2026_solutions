"""Skip sliding-window positions that cannot contain the target bone.

Organisers confirmed Task 2's test-phase failure is a timeout, and profiling put ~40%
of the fragment loop in the sliding window itself (5.5 s of 13.9 s per fragment, and
the loop is 91% of a 30-fragment case). The obvious fix -- crop each fragment's input
to its bone -- was measured and REJECTED: cropping changes the array shape, which
changes nnU-Net's resample grid and sliding-window origins, so every patch lands
somewhere different. It is brutally sensitive; trimming just two voxels off one axis
(512 -> 510) already moved 1.74% of the foreground, and a useful crop destroyed the
prediction outright (foreground IoU 0.07-49%). Pre-normalising the ZScore heatmap
channels did not rescue it either. A margin=100000 identity crop scoring exactly
100.00% confirms that was a real effect and not a bug in the crop.

So leave the input completely alone -- same array, same preprocessing, same resample
grid, same window origins -- and only DROP the windows that do not touch the bone.
Every window that is kept sees byte-identical data at a byte-identical position and
contributes exactly the weight it did before, so inside the bone the Gaussian-weighted
accumulation is unchanged and the output is preserved by construction rather than by
hoping the network generalises. Voxels no window covers are background, which they
already were.

The one thing that needs care is nnU-Net's `torch.div(predicted_logits, n_predictions)`
followed by an inf check: uncovered voxels have n_predictions == 0. They are clamped to
1 here, which leaves their logits at 0 and therefore their argmax at the background
class -- the same answer, without tripping the inf guard.
"""

from __future__ import annotations

import numpy as np
import torch


def bbox_of(mask, margin=0):
    """Inclusive-exclusive (lo, hi) index arrays of a boolean mask, padded by margin."""
    idx = np.argwhere(mask)
    if len(idx) == 0:
        return None
    lo = np.maximum(idx.min(0) - margin, 0)
    hi = np.minimum(idx.max(0) + 1 + margin, np.array(mask.shape))
    return lo, hi


def map_bbox_to_preprocessed(lo, hi, orig_shape, prop, prep_shape):
    """Carry an original-space bbox through nnU-Net's crop-to-nonzero and resample.

    `prop` is the properties dict the preprocessor returns; its bbox_used_for_cropping
    is the crop applied before resampling. The resample is a pure scale from the
    cropped shape to the preprocessed shape, so the bbox maps by the same ratio.
    Rounds outward so the mapped box can only ever be too big, never too small -- a
    box that is too small would silently drop windows the bone needs.
    """
    crop = prop.get("bbox_used_for_cropping")
    if crop is not None:
        cl = np.array([c[0] for c in crop])
        ch = np.array([c[1] for c in crop])
        lo = np.maximum(lo - cl, 0)
        hi = np.minimum(hi - cl, ch - cl)
        cropped_shape = ch - cl
    else:
        cropped_shape = np.array(orig_shape)

    scale = np.array(prep_shape, dtype=float) / np.maximum(cropped_shape, 1)
    plo = np.floor(lo * scale).astype(int)
    phi = np.ceil(hi * scale).astype(int)
    plo = np.maximum(plo, 0)
    phi = np.minimum(phi, np.array(prep_shape))
    return plo, phi


def filter_slicers(slicers, plo, phi):
    """Keep only windows whose spatial extent intersects [plo, phi)."""
    kept = []
    for sl in slicers:
        spatial = sl[1:] if len(sl) > 3 else sl
        if all(s.start < h and s.stop > l for s, l, h in zip(spatial, plo, phi)):
            kept.append(sl)
    return kept


class restrict_windows:
    """Context manager: make `predictor` skip windows that miss the ROI.

    Rather than reimplementing the accumulation loop -- which means re-deriving
    nnU-Net's padding, autocast and float16 handling and getting all three exactly
    right -- this swaps out only `_internal_get_sliding_window_slicers` and lets the
    stock code path run untouched. The ROI is given in PREPROCESSED coordinates
    because that is the space the slicers live in.

        with restrict_windows(pred, plo, phi) as r:
            logits = pred.predict_logits_from_preprocessed_data(data)
        seg = convert_predicted_logits_to_segmentation_with_correct_shape(...)
        print(r.kept, r.total)

    Uncovered voxels end up 0/0 = NaN rather than inf, so they slip past nnU-Net's
    inf guard; they are mapped to 0 on the way out, which puts their argmax on the
    background class -- the answer they would have had anyway.
    """

    def __init__(self, predictor, plo, phi):
        self.predictor = predictor
        self.plo = np.asarray(plo)
        self.phi = np.asarray(phi)
        self.kept = self.total = 0
        self._orig = None

    def __enter__(self):
        self._orig = self.predictor._internal_get_sliding_window_slicers

        def patched(image_size):
            all_sl = self._orig(image_size)
            keep = filter_slicers(all_sl, self.plo, self.phi)
            self.total += len(all_sl)
            self.kept += len(keep)
            # never return nothing: an empty window list would make the whole volume
            # NaN. If the ROI mapped outside the padded volume, fall back to full.
            return keep if keep else all_sl

        self.predictor._internal_get_sliding_window_slicers = patched
        return self

    def __exit__(self, *exc):
        self.predictor._internal_get_sliding_window_slicers = self._orig
        return False


def sanitize_logits(logits):
    """Uncovered voxels are NaN (0/0); send them to the background class."""
    return torch.nan_to_num(logits, nan=0.0, posinf=0.0, neginf=0.0)
