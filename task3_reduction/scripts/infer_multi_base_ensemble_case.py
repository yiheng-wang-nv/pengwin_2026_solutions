#!/usr/bin/env python3
"""N AssemblyNet bases, each with its own five-fold residual, combined by centroid weighting.

Generalises infer_dual_base_ensemble_case.py from two members to any number. Each base runs the
champion damping recipe and is corrected by the residual ensemble trained on ITS OWN base
predictions -- a residual head is tied to the base it was cached from and cannot be reused.

Members are loaded and released one at a time so peak memory stays at one AssemblyNet plus one
residual ensemble regardless of member count.

Combination is SO(3) rotation + fragment-centroid displacement via pose_ensemble.py, the same
module that produced the offline numbers. Centroids come from trimesh.load(..., process=False)
to match the offline pipeline and the official evaluator.
"""
from __future__ import annotations
import argparse, importlib.util, json, sys, tempfile, time
from pathlib import Path
import numpy as np, torch

SCRIPT_DIR = Path(__file__).resolve().parent
REPO = SCRIPT_DIR.parents[1]
BASELINE_DIR = REPO / "external/PENGWIN2026_Task3_Reduction_Baseline"
for p in (SCRIPT_DIR, BASELINE_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from residual_cv import build_case_arrays, load_model, predict_arrays, write_prediction
from pose_ensemble import combine_predictions


def load_baseline_inference():
    spec = importlib.util.spec_from_file_location("pengwin_t3_baseline_inference",
                                                  BASELINE_DIR / "inference.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def validate_prediction(prediction):
    for item in prediction:
        M = np.asarray(item["transformation"], dtype=np.float64)
        if M.shape != (4, 4) or not np.isfinite(M).all():
            raise ValueError(f"fragment {item['fragment_id']}: invalid 4x4 matrix")
        R = M[:3, :3]
        if np.max(np.abs(R.T @ R - np.eye(3))) > 1e-5:
            raise ValueError(f"fragment {item['fragment_id']}: rotation not orthogonal")
        if abs(np.linalg.det(R) - 1.0) > 1e-5:
            raise ValueError(f"fragment {item['fragment_id']}: det != +1")


def source_centroids(obj_path: Path) -> dict:
    import trimesh
    scene = trimesh.load(str(obj_path), split_object=True, process=False)
    if isinstance(scene, trimesh.Scene):
        return {str(k): np.asarray(g.vertices, np.float64).mean(axis=0)
                for k, g in scene.geometry.items() if isinstance(g, trimesh.Trimesh)}
    return {"1": np.asarray(scene.vertices, np.float64).mean(axis=0)}


def run_one_base(baseline, args, config_path: Path, weights_dir: Path, case: str, device: str):
    residual_paths = sorted(Path(weights_dir).glob("fold*_last.pt"))
    if len(residual_paths) != 5:
        raise SystemExit(f"expected five residual checkpoints in {weights_dir}, got {len(residual_paths)}")
    t0 = time.time()
    model, cfg = baseline.load_config_and_model(str(config_path), device)
    meshdict = baseline.load_obj_fragments(str(args.obj), verbose=False)
    fragment_data = baseline.build_fragment_data_from_meshdict(
        meshdict, npoints=args.npoints, seed=args.seed)
    transforms, iters, converged, _ = baseline.run_iterative_inference(
        model, fragment_data, device, output_type=cfg.output_type,
        max_parts=cfg.transformer_model.max_parts, max_iters=args.max_iters,
        convergence_threshold=args.convergence_threshold, update_scale=args.update_scale,
        convergence_metric=args.convergence_metric)
    names = [n for bone in ("SA", "LI", "RI") for n in sorted(meshdict.get(bone, {}))]
    transforms = baseline.normalize_transforms_by_first_sa(transforms, meshdict)
    base_prediction = baseline.build_final_results(transforms, names)
    t_base = time.time() - t0

    del model
    if device == "cuda":
        torch.cuda.empty_cache()

    t1 = time.time()
    with tempfile.TemporaryDirectory(prefix="task3-residual-") as td:
        bp = Path(td) / "base.json"
        bp.write_text(json.dumps(base_prediction))
        arrays, _ = build_case_arrays(case=case, obj_path=args.obj, pred_path=bp,
                                      n_fragment=args.n_fragment, n_context=args.n_context,
                                      gt_path=None)
    dev = torch.device(device)
    models = [load_model(p, dev) for p in residual_paths]
    prediction = predict_arrays(arrays, models, dev)
    del models
    if device == "cuda":
        torch.cuda.empty_cache()
    print(f"  {config_path.name}: iters={iters} conv={converged} "
          f"base={t_base:.2f}s residual={time.time()-t1:.2f}s", flush=True)
    return prediction


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--obj", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--case-id")
    ap.add_argument("--base-config", type=Path, nargs="+", required=True)
    ap.add_argument("--weights-dir", type=Path, nargs="+", required=True)
    ap.add_argument("--w-rot", type=float, nargs="+", default=None)
    ap.add_argument("--w-trans", type=float, nargs="+", default=None)
    ap.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--npoints", type=int, default=5000)
    ap.add_argument("--max-iters", type=int, default=20)
    ap.add_argument("--convergence-threshold", type=float, default=2.0)
    ap.add_argument("--update-scale", type=float, default=0.3)
    ap.add_argument("--convergence-metric",
                    choices=("matrix_translation", "centroid", "max_point"), default="max_point")
    ap.add_argument("--n-fragment", type=int, default=256)
    ap.add_argument("--n-context", type=int, default=512)
    args = ap.parse_args()

    if len(args.base_config) != len(args.weights_dir):
        raise SystemExit("--base-config and --weights-dir must have the same length")
    if not args.obj.exists():
        raise FileNotFoundError(args.obj)
    case = args.case_id or args.obj.parent.name or args.obj.stem
    device = "cuda" if (args.device == "auto" and torch.cuda.is_available()) else \
             ("cpu" if args.device == "auto" else args.device)
    if device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")
    baseline = load_baseline_inference()

    started = time.time()
    preds = [run_one_base(baseline, args, c, w, case, device)
             for c, w in zip(args.base_config, args.weights_dir)]
    combined = combine_predictions(preds, source_centroids(args.obj),
                                   w_rot=args.w_rot, w_trans=args.w_trans)
    validate_prediction(combined)
    write_prediction(args.out, combined)
    print(f"Wrote {args.out}: case={case}, members={len(preds)}, "
          f"fragments={len(combined)}, total={time.time()-started:.2f}s")


if __name__ == "__main__":
    main()
