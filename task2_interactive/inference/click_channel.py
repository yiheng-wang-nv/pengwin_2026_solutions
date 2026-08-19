"""The ONE implementation of ds458's click channel, shared by training and inference.

This module exists because it was written twice and the two copies drifted. The generator
(scripts/task2/gen_ds458_click_csm.py) moved to a dense distance field; process_flat.py kept
building the sparse Gaussian its comment claimed matched. A model trained on one and fed the
other sees an input distribution it has never seen, and the resulting score says nothing
about the model. Both callers now import from here, and the parameters travel with the
dataset (see `params_from_dataset_json`) so inference cannot guess wrong.

It lives under docker_task2/inference/ because that directory is what ships inside the Grand
Challenge container and is already on sys.path there (INF_DIR).

Two encodings:

`gaussian` -- one windowed Gaussian per click, summed and max-normalised. This is what the
first ds458 trained on. At the shipped sigma=1.0 a click covers ~33 voxels above 0.1; against
a real per-bone volume of a median 37.6M voxels that is ~0.0001%, most training patches
contain no click at all, and -- the part that matters -- the field reads exactly 0.000 at the
midpoint between two clicks, which is where the contact surface belongs. Kept so the failed
run stays reproducible.

`distance` -- every voxel gets exp(-d / sigma) for d the Euclidean distance to the NEAREST
click. Dense: measured on real data at sigma=15, 46.2% of bone-foreground voxels read above
0.1 and the contact surface reads a median 0.2496. The surface equidistant from two clicks is
where their shared seam should be, so the channel is evidence about the seam rather than a
hint about fragment centres.

(A 64^3 synthetic once suggested "81.5% of voxels above 0.1" for `distance`. That does not
transfer -- exp(-d/15) > 0.1 is a ~172k-voxel ball, which nearly fills a 262k-voxel box and
covers 1.19% of a real volume. Judge this channel on the foreground and seam numbers above.)
"""
from __future__ import annotations

import numpy as np

GAUSSIAN = "gaussian"
DISTANCE = "distance"
ENCODINGS = (GAUSSIAN, DISTANCE)

# What the first (failed) ds458 shipped. Used when a model pack predates the encoding being
# recorded in dataset.json, so old packs keep behaving exactly as they did.
LEGACY_PARAMS = {"encoding": GAUSSIAN, "sigma": 1.0}


def build(shape, clicks, encoding=DISTANCE, sigma=15.0):
    """Channel 1 for one bone. `clicks` is a list of (z, y, x) in voxel coords.

    Returns float32 in [0, 1], peaking at exactly 1.0 on a click voxel and nowhere else --
    both encodings share that property, which is what the peak check in the rebuild script
    relies on to prove the coordinates landed inside their own fragment.
    """
    hm = np.zeros(shape, dtype=np.float32)
    if not clicks:
        return hm

    if encoding == GAUSSIAN:
        import fastheat
        scratch = np.zeros(shape, dtype=np.float32)
        # fill_fg_bg puts click 0 in the fg buffer and the rest in the bg one; the training
        # data took the elementwise max, so all clicks end up in a single channel.
        fastheat.fill_fg_bg(hm, scratch, clicks, 0, sigma=sigma)
        np.maximum(hm, scratch, out=hm)
        mx = hm.max()
        if mx > 0:
            hm /= mx
        return hm

    if encoding != DISTANCE:
        raise ValueError(f"unknown click encoding {encoding!r}, expected one of {ENCODINGS}")

    from scipy.ndimage import distance_transform_edt
    # EDT measures distance to the nearest ZERO, so the clicks are the zeros.
    seeds = np.ones(shape, dtype=bool)
    for c in clicks:
        z, y, x = (int(round(v)) for v in c)
        if 0 <= z < shape[0] and 0 <= y < shape[1] and 0 <= x < shape[2]:
            seeds[z, y, x] = False
    if seeds.all():                       # every click fell outside the volume
        return hm
    d = distance_transform_edt(seeds).astype(np.float32)
    # sigma is a decay length in voxels here, not a Gaussian width, so the field is
    # scale-free with respect to bone size.
    return np.exp(-d / max(sigma, 1e-6)).astype(np.float32)


def params_from_dataset_json(dataset_json):
    """Read the encoding the model was TRAINED with out of its own dataset.json.

    gen_ds458_click_csm.py records it under "click_encoding". Packs built before that key
    existed are the sparse sigma=1 Gaussian, so that is the fallback -- never the current
    default, or an old pack would silently be fed a channel it never saw.
    """
    p = (dataset_json or {}).get("click_encoding") or LEGACY_PARAMS
    return str(p.get("encoding", LEGACY_PARAMS["encoding"])), float(p.get("sigma", LEGACY_PARAMS["sigma"]))
