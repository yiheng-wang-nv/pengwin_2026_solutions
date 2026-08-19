"""Ensemble Task 3 poses in the PHYSICAL parameterisation: SO(3) rotation + centroid displacement.

`ensemble_poses.py` averages in se(3), which forces one weight on the whole pose: log(T) is
[[omega]x, v] and v = V(omega)^-1 t depends on omega, so taking the rotation from one member
and the translation part from a blend of members with different rotations is not a valid
operation. Measured: TRE 3.14 -> 8.7, Trans 3.88 -> 15.9. Same coupling trap the repo's rule
about the raw 4x4 translation column warns about, one level down.

The decomposition that *does* separate them is the one used by the damping, the shrink
post-processing and the residual head alike (PENGWIN_TASK3_EVAL_PROTOCOL.md):

    d = R c + t - c            fragment-centroid displacement
    t = c + d - R c            reconstruction

R and d are independent, so they can carry different weights. This makes "rotation from the
champion line only, translation from the ensemble" expressible -- which se(3) averaging cannot
do -- at the cost of needing the meshes for the centroids.

Usage:
  python scripts/task3/ensemble_poses_centroid.py --roots A B --obj-dir MESH --out DIR \
      --w-rot 1 0 --w-trans 0.55 0.45
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
from pose_ensemble import combine_predictions          # the container runs this same code

PRED = "reduction-poses-matrices.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="+", required=True)
    ap.add_argument("--obj-dir", required=True, help="<dir>/<case>/*.obj")
    ap.add_argument("--out", required=True)
    ap.add_argument("--w-rot", nargs="+", type=float, default=None)
    ap.add_argument("--w-trans", nargs="+", type=float, default=None)
    args = ap.parse_args()

    import trimesh

    n = len(args.roots)
    wr = np.asarray(args.w_rot if args.w_rot else [1 / n] * n, float)
    wt = np.asarray(args.w_trans if args.w_trans else [1 / n] * n, float)
    wr, wt = wr / wr.sum(), wt / wt.sum()
    roots = [Path(r) for r in args.roots]
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    obj_dir = Path(args.obj_dir)

    cases = sorted({d.name for r in roots for d in r.iterdir()
                    if d.is_dir() and (d / PRED).exists()})
    print(f"{n} members, {len(cases)} cases, w_rot={wr.round(3)}, w_trans={wt.round(3)}")

    for cid in cases:
        objs = sorted((obj_dir / cid).glob("*.obj"))
        scene = trimesh.load(str(objs[0]), split_object=True, process=False)
        cents = {}
        if isinstance(scene, trimesh.Scene):
            for k, g in scene.geometry.items():
                if isinstance(g, trimesh.Trimesh):
                    cents[str(k)] = np.asarray(g.vertices, np.float64).mean(axis=0)
        else:
            cents["1"] = np.asarray(scene.vertices, np.float64).mean(axis=0)

        preds = [json.load(open(r / cid / PRED)) for r in roots if (r / cid / PRED).exists()]
        res = combine_predictions(preds, cents, wr, wt)
        (out / cid).mkdir(exist_ok=True)
        json.dump(res, open(out / cid / PRED, "w"), indent=2)
    print(f"wrote {len(cases)} cases")


if __name__ == "__main__":
    main()
