"""Reuse sliding-window patches that are identical across fragments.

Task 2 runs one full sliding window per fragment, so a 30-fragment case costs 30x the
windows -- which is what times out on Grand Challenge. But for a given window, the
three input channels are:

    ch0  the CT                                     identical for every fragment
    ch1  Gaussian at THIS click                     zero unless this click reaches here
    ch2  sum of Gaussians at the OTHER clicks       everything that reaches here, minus this one

so for any window that THIS click cannot reach, ch1 is a constant and ch2 is the sum of
whatever else reaches the window -- the same tensor for every such fragment. The network
is deterministic, so those windows only need computing once and can then be reused. This
is exact deduplication of identical work, not an approximation.

Counting it on real cases (clicks that drive per-fragment inference, windows at step 0.5):

    084   30 frags, 18 windows   540 -> 198 forwards   2.73x
    064   23 frags, 27 windows   621 -> 219 forwards   2.84x
    029   11 frags, 27 windows   297 -> 111 forwards   2.68x
    001    5 frags, 36 windows   180 ->  78 forwards   2.31x

The saving grows with fragment count, which is exactly where the timeout is.

THREE conditions have to hold for two fragments to share a window, and two of them are
easy to miss:

  reach   gaussian_3d_fast computes a window of radius ceil(7*sigma)+1 = 8 voxels, so
          the test is whether the click's 8-voxel support intersects the window, NOT
          whether the click itself is inside it.

  stats   the heatmap channels get ZScoreNormalization over the WHOLE volume before the
          sliding window. Every click's Gaussian has the same shape, so the statistics
          normally do not depend on which click is which -- but a click near the volume
          border has its Gaussian truncated, which changes the sum and therefore the
          statistics. Case 196's clicks sit 5 voxels from the border. Fragments are
          therefore grouped by their actual (mean, std) pairs and cache entries are only
          shared inside a group.

  scale   fg and bg are each divided by their own max. Measured across all 340 training
          cases the closest two clicks are 3.0 voxels apart (median 29), where a sigma=1
          Gaussian contributes 0.011, so max is 1.0 to float precision essentially
          always -- but the group key includes it rather than assuming it.

Memory: the cache holds one logits patch per window, 18 x 2 classes x 12.6M x 2 bytes
~= 900 MB, versus the 8.4 GB it would take to keep 30 per-fragment accumulators live.
Fragments are still processed one at a time.
"""

from __future__ import annotations

import numpy as np
import torch

CLICK_RADIUS = 8  # ceil(7 * sigma) + 1 for sigma = 1, matching gaussian_3d_fast


def click_reaches(click_zyx, window_slices, radius=CLICK_RADIUS):
    """Does this click's Gaussian support intersect the window?"""
    for c, sl in zip(click_zyx, window_slices):
        if c + radius < sl.start or c - radius >= sl.stop:
            return False
    return True


def group_key(fg, bg):
    """Fragments sharing this key have identical normalisation, so patches are shareable.

    Uses the raw float bits: these are compared for exact equality on purpose, since any
    difference at all means the normalised patch differs and the cache would be wrong.
    """
    return (float(fg.mean()), float(fg.std()), float(fg.max()),
            float(bg.mean()), float(bg.std()), float(bg.max()))


class WindowCache:
    """LRU cache of per-window logits for the 'this click is out of reach' case.

    The budget is not optional. Each entry is n_classes x patch_size in fp16 -- about
    50 MB for a 2-class [192,256,256] patch -- and an unbounded run on case 084 kept 462
    entries and peaked at 32 GB of VRAM, against the T4's 16 GB. Entries do not collapse
    across fragments as much as one might hope: the heatmap channels are ZScore-normalised
    over the whole volume, and the per-fragment statistics genuinely differ (fg.std varies
    by 250% across fragments on 084), so each fragment tends to form its own group.
    Widening the key to absorb that is NOT safe -- the differences are far above fp16's
    ~1e-3 resolution, so they really do change the input.

    Eviction is least-recently-used, which suits the access pattern: fragments are
    processed in order and a group's entries are reused by nearby fragments.
    """

    def __init__(self, max_bytes=1_000_000_000):
        self._store = {}
        self._order = []        # oldest first
        self.max_bytes = max_bytes
        self.bytes = 0
        self.hits = 0
        self.misses = 0
        self.evictions = 0

    def get(self, key):
        v = self._store.get(key)
        if v is None:
            self.misses += 1
            return None
        self.hits += 1
        try:
            self._order.remove(key)
        except ValueError:
            pass
        self._order.append(key)
        return v

    def put(self, key, value):
        nbytes = value.element_size() * value.nelement()
        if nbytes > self.max_bytes:
            return                       # single entry larger than the whole budget
        while self.bytes + nbytes > self.max_bytes and self._order:
            old = self._order.pop(0)
            v = self._store.pop(old, None)
            if v is not None:
                self.bytes -= v.element_size() * v.nelement()
                self.evictions += 1
            del v
        self._store[key] = value
        self._order.append(key)
        self.bytes += nbytes

    def clear(self):
        self._store.clear()
        self._order.clear()
        self.bytes = 0

    def stats(self):
        tot = self.hits + self.misses
        return (f"hits={self.hits} misses={self.misses} "
                f"reuse={100*self.hits/max(tot,1):.0f}% "
                f"evict={self.evictions} peakMB={self.bytes//1_000_000}")


@torch.inference_mode()
def predict_with_cache(predictor, data, slicers, click_zyx, gkey, cache,
                       revert_padding=None, pad_offset=None):
    """Sliding window for one fragment, reusing cached windows this click cannot reach.

    `data` is the padded, preprocessed (C,Z,Y,X) tensor for THIS fragment. `click_zyx`
    is the current click in the same padded coordinate frame. `gkey` is its
    normalisation group. Windows the click cannot reach are looked up under
    (window index, gkey) and computed once per group.
    """
    from nnunetv2.inference.sliding_window_prediction import compute_gaussian

    # predict_logits_from_preprocessed_data is what loads the weights into the network
    # and moves it to the device; predict_sliding_window_return_logits does the .eval().
    # Calling _internal_maybe_mirror_and_predict without them runs an unprepared network
    # -- which surfaces as "Input type (c10::Half) and bias type (float) should be the
    # same", not as an obviously-wrong result. A standalone test that happens to call
    # nnU-Net's own path first will pass while the real pipeline fails.
    if len(predictor.list_of_parameters) != 1:
        raise RuntimeError(f"window caching assumes a single fold, got "
                           f"{len(predictor.list_of_parameters)}")
    if not getattr(predictor, "_patchcache_ready", False):
        predictor.network.load_state_dict(predictor.list_of_parameters[0])
        predictor.network = predictor.network.to(predictor.device)
        predictor.network.eval()
        predictor._patchcache_ready = True

    device = predictor.device
    data = data.to(device, non_blocking=True)
    n_heads = predictor.label_manager.num_segmentation_heads
    logits = torch.zeros((n_heads, *data.shape[1:]), dtype=torch.half, device=device)
    n_pred = torch.zeros(data.shape[1:], dtype=torch.half, device=device)
    gaussian = (compute_gaussian(tuple(predictor.configuration_manager.patch_size),
                                 sigma_scale=1. / 8, value_scaling_factor=10, device=device)
                if predictor.use_gaussian else 1)

    with torch.autocast(device.type, enabled=True) if device.type == "cuda" else _null():
        for wi, sl in enumerate(slicers):
            spatial = sl[1:]
            reachable = click_reaches(click_zyx, spatial)
            key = None if reachable else (wi, gkey)
            pred = None if key is None else cache.get(key)
            if pred is None:
                pred = predictor._internal_maybe_mirror_and_predict(data[sl][None])[0]
                if key is not None:
                    cache.put(key, pred)
            if predictor.use_gaussian:
                logits[sl] += pred * gaussian
            else:
                logits[sl] += pred
            n_pred[sl[1:]] += gaussian

    torch.div(logits, n_pred, out=logits)
    if torch.any(torch.isinf(logits)):
        raise RuntimeError("inf in cached-window logits")
    if revert_padding is not None:
        logits = logits[(slice(None), *revert_padding[1:])]
    return logits


class _null:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False
