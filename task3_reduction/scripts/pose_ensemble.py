"""Shared pose-ensemble math for Task 3, used by both the offline sweep and the container.

Kept in one module on purpose: the weights were selected against numbers produced by
`ensemble_poses_centroid.py`, so the submitted image has to combine poses with the exact same
code or the validated score does not describe what ships.

Poses are combined in the PHYSICAL parameterisation -- SO(3) rotation and fragment-centroid
displacement -- not in se(3) and never on the raw 4x4 translation column:

    d = R c + t - c
    t = c + d - R c

Only this decomposition lets rotation and translation carry different weights. Averaging
log(T) forces one weight on both, because its translation part v = V(omega)^-1 t is a function
of the rotation; blending v across members with different omega was measured at TRE 3.14 ->
8.7 and Trans 3.88 -> 15.9.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import expm, logm


def so3_mean(rotations, weights):
    """Weighted rotation mean through the matrix log, projected back onto SO(3)."""
    rotations = list(rotations)
    if len(rotations) == 1:
        return rotations[0]
    algebra = sum(w * np.real(logm(R)) for w, R in zip(weights, rotations))
    R = np.real(expm(algebra))
    U, _, Vt = np.linalg.svd(R)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    return R


def combine_pose(matrices, centroid, w_rot, w_trans):
    """Combine one fragment's 4x4 poses given its source-mesh centroid."""
    matrices = [np.asarray(M, dtype=np.float64) for M in matrices]
    if len(matrices) == 1:
        return matrices[0]
    c = np.asarray(centroid, dtype=np.float64)
    R = so3_mean([M[:3, :3] for M in matrices], w_rot)
    d = sum(w * (M[:3, :3] @ c + M[:3, 3] - c) for w, M in zip(w_trans, matrices))
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = c + d - R @ c
    return T


def normalise(weights, n):
    w = np.asarray(weights if weights is not None else [1.0 / n] * n, dtype=np.float64)
    if w.size != n:
        raise ValueError(f"expected {n} weights, got {w.size}")
    total = w.sum()
    if total <= 0:
        raise ValueError("weights must sum to a positive value")
    return w / total


def combine_predictions(predictions, centroids, w_rot=None, w_trans=None):
    """predictions: list (one per member) of [{fragment_id, transformation}, ...].

    Fragments are matched by id and keep the first member's order. A fragment that is missing
    from some member is passed through from the first member that has it rather than dropped.
    """
    n = len(predictions)
    wr, wt = normalise(w_rot, n), normalise(w_trans, n)
    per_fragment: dict[str, list] = {}
    order: list[str] = []
    for pred in predictions:
        for entry in pred:
            fid = entry["fragment_id"]
            if fid not in per_fragment:
                per_fragment[fid] = []
                order.append(fid)
            per_fragment[fid].append(np.asarray(entry["transformation"], dtype=np.float64))

    out = []
    for fid in order:
        mats = per_fragment[fid]
        c = centroids.get(fid)
        if c is None or len(mats) != n:
            out.append({"fragment_id": fid, "transformation": np.asarray(mats[0]).tolist()})
            continue
        out.append({"fragment_id": fid,
                    "transformation": combine_pose(mats, c, wr, wt).tolist()})
    return out
