#!/usr/bin/env python3
"""Cache, train and evaluate a strict 5-fold Task-3 residual model.

The pipeline is deliberately separate from AssemblyNet. It consumes a fixed
directory of baseline pose predictions, so changing the base inference recipe
cannot silently contaminate a residual comparison.

Typical use (from the repository root)::

  PYTHONPATH=. python scripts/task3/residual_cv.py cache \
    --data-dir datasets/task3_reduction/extracted/PENGWIN26_task3_clinical_fractures_train/mesh \
    --pred-root output_task3_damp030_point20_seed42 \
    --cache-dir output_task3_residual/cache
  PYTHONPATH=. python scripts/task3/residual_cv.py make-splits \
    --cache-dir output_task3_residual/cache \
    --split-file output_task3_residual/folds.json
  PYTHONPATH=. python scripts/task3/residual_cv.py train-cv \
    --cache-dir output_task3_residual/cache \
    --split-file output_task3_residual/folds.json \
    --weights-dir output_task3_residual/weights --device cuda
  PYTHONPATH=. python scripts/task3/residual_cv.py predict-oof \
    --cache-dir output_task3_residual/cache \
    --split-file output_task3_residual/folds.json \
    --weights-dir output_task3_residual/weights \
    --out-dir output_task3_residual/oof_predictions
  PYTHONPATH=. python scripts/task3/residual_cv.py score \
    --data-dir datasets/task3_reduction/extracted/PENGWIN26_task3_clinical_fractures_train/mesh \
    --base-pred-root output_task3_damp030_point20_seed42 \
    --residual-pred-root output_task3_residual/oof_predictions \
    --out-csv output_task3_residual/oof_scores.csv

Five-fold OOF is the only clinical model-selection estimate. For unseen test
cases, use ``predict-ensemble`` to average the five independently trained
residuals. Never score that in-sample ensemble on these 170 training cases.
For a single hidden case with no GT, ``predict-case`` builds the same features
directly from its OBJ and fixed AssemblyNet base-prediction JSON.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from torch.utils.data import DataLoader, Dataset

SCRIPT_DIR = Path(__file__).resolve().parent
REPO = SCRIPT_DIR.parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from residual_model import ResidualPointNet, correct_centered_points


PRED_NAME = "reduction-poses-matrices.json"
POINT_SCALE_MM = 50.0
POSITION_SCALE_MM = 100.0
GLOBAL_DIM = 14


def load_evaluator():
    path = REPO / "external/PENGWIN2026_Task3_Reduction_Baseline/evaluate.py"
    spec = importlib.util.spec_from_file_location("pengwin_task3_evaluate", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_poses(path: Path) -> dict[str, np.ndarray]:
    with path.open() as f:
        raw = json.load(f)
    if isinstance(raw, list):
        return {
            str(item["fragment_id"]): np.asarray(item["transformation"], dtype=np.float64)
            for item in raw
        }
    return {str(k): np.asarray(v, dtype=np.float64) for k, v in raw.items()}


def normalise_poses(poses: dict[str, np.ndarray], anchor: str) -> dict[str, np.ndarray]:
    inverse = np.linalg.inv(poses[anchor])
    return {fid: inverse @ transform for fid, transform in poses.items()}


def bone_index(fid: str) -> int:
    value = int(fid)
    if value <= 100:
        return 0
    if value <= 200:
        return 1
    return 2


def deterministic_seed(*values: str) -> int:
    digest = hashlib.sha256("/".join(values).encode()).digest()
    return int.from_bytes(digest[:4], "little")


def apply_transform(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    return points @ transform[:3, :3].T + transform[:3, 3]


def sample_mesh(mesh, count: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Surface samples and corresponding normals, deterministically."""
    rng = np.random.default_rng(seed)
    if len(mesh.vertices) == 0:
        return np.zeros((count, 3), np.float64), np.zeros((count, 3), np.float64)
    if len(mesh.faces):
        # trimesh supports an explicit integer seed in the installed challenge env.
        import trimesh

        points, face_ids = trimesh.sample.sample_surface(
            mesh, count, seed=int(rng.integers(0, 2**31 - 1))
        )
        normals = np.asarray(mesh.face_normals)[face_ids]
    else:
        indices = rng.integers(0, len(mesh.vertices), size=count)
        points = np.asarray(mesh.vertices)[indices]
        vertex_normals = np.asarray(mesh.vertex_normals)
        normals = vertex_normals[indices] if len(vertex_normals) else np.zeros_like(points)
    return np.asarray(points, np.float64), np.asarray(normals, np.float64)


def fixed_indices(indices: np.ndarray, size: int, fallback_order: np.ndarray) -> np.ndarray:
    """Return exactly size indices, preserving deterministic nearest-first order."""
    seen = set()
    ordered = []
    for value in np.concatenate((indices.reshape(-1), fallback_order.reshape(-1))):
        value = int(value)
        if value not in seen:
            seen.add(value)
            ordered.append(value)
        if len(ordered) == size:
            break
    if not ordered:
        raise ValueError("cannot select context from an empty point set")
    while len(ordered) < size:
        ordered.extend(ordered[: size - len(ordered)])
    return np.asarray(ordered[:size], dtype=np.int64)


def discover_records(data_dir: Path, pred_root: Path) -> list[tuple[str, Path, Path, Path]]:
    records = []
    for case_dir in sorted(data_dir.iterdir()):
        if not case_dir.is_dir():
            continue
        objs = sorted(case_dir.glob("*.obj"))
        gt_path = case_dir / "plan_pl_gt.json"
        pred_path = pred_root / case_dir.name / PRED_NAME
        if objs and gt_path.exists() and pred_path.exists():
            records.append((case_dir.name, objs[0], gt_path, pred_path))
    return records


def cache_case(
    record: tuple[str, Path, Path, Path],
    output: Path,
    n_fragment: int,
    n_context: int,
) -> dict:
    case, obj_path, gt_path, pred_path = record
    arrays, item = build_case_arrays(
        case=case,
        obj_path=obj_path,
        pred_path=pred_path,
        n_fragment=n_fragment,
        n_context=n_context,
        gt_path=gt_path,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **arrays)
    return item


def build_case_arrays(
    case: str,
    obj_path: Path,
    pred_path: Path,
    n_fragment: int,
    n_context: int,
    gt_path: Path | None = None,
) -> tuple[dict[str, np.ndarray], dict]:
    """Build the shared residual inputs, with optional training-only targets.

    Test inference must not depend on ``plan_pl_gt.json``.  All arrays consumed
    by :func:`predict_arrays` are therefore derived only from the input OBJ and
    the fixed AssemblyNet base prediction.  Supplying ``gt_path`` merely adds
    target arrays used by training; it does not change the feature recipe.
    """
    evaluator = load_evaluator()
    meshes = evaluator.load_obj_meshes(str(obj_path))
    pred_raw = load_poses(pred_path)
    gt_raw = load_poses(gt_path) if gt_path is not None else None
    common = sorted(set(meshes) & set(pred_raw), key=int)
    if not common:
        sources = "mesh/GT/base prediction" if gt_raw is not None else "mesh/base prediction"
        raise ValueError(f"{case}: no common fragments in {sources}")
    if gt_raw is not None:
        missing_gt = sorted(set(common) - set(gt_raw), key=int)
        if missing_gt:
            raise ValueError(f"{case}: GT is missing predicted mesh fragments {missing_gt}")
    sa = [fid for fid in common if 1 <= int(fid) <= 100]
    anchor = sa[0] if sa else common[0]
    pred = normalise_poses(pred_raw, anchor)
    gt = normalise_poses(gt_raw, anchor) if gt_raw is not None else None

    source_points = {}
    source_normals = {}
    source_centroids = {}
    pred_points = {}
    pred_normals = {}
    pred_centroids = {}
    gt_points = {}
    for fid in common:
        points, normals = sample_mesh(
            meshes[fid], n_fragment, deterministic_seed(case, fid, "fragment")
        )
        source_points[fid] = points
        source_normals[fid] = normals
        source_centroids[fid] = np.asarray(meshes[fid].vertices, np.float64).mean(axis=0)
        pred_points[fid] = apply_transform(points, pred[fid])
        pred_normals[fid] = normals @ pred[fid][:3, :3].T
        pred_centroids[fid] = apply_transform(source_centroids[fid][None], pred[fid])[0]
        if gt is not None:
            gt_points[fid] = apply_transform(points, gt[fid])

    # Use a point-weighted center so absolute position features have a stable
    # assembly reference without seeing GT geometry.
    assembly_center = np.concatenate([pred_points[fid] for fid in common]).mean(axis=0)
    moving = [fid for fid in common if fid != anchor]
    current_features, context_features, globals_, targets = [], [], [], []
    target_dc, target_dr = [], []
    moving_source_centroids, moving_pred_centroids = [], []

    for fid in moving:
        center = pred_centroids[fid]
        current_xyz = pred_points[fid] - center
        current_features.append(
            np.concatenate((current_xyz / POINT_SCALE_MM, pred_normals[fid]), axis=1)
        )

        other_ids = [other for other in common if other != fid]
        other_xyz = np.concatenate([pred_points[other] for other in other_ids], axis=0)
        other_nrm = np.concatenate([pred_normals[other] for other in other_ids], axis=0)
        other_same_bone = np.concatenate(
            [
                np.full((len(pred_points[other]), 1), bone_index(other) == bone_index(fid))
                for other in other_ids
            ],
            axis=0,
        ).astype(np.float64)
        tree = cKDTree(other_xyz)
        query_points = pred_points[fid][:: max(1, n_fragment // 64)]
        k = min(8, len(other_xyz))
        _, near = tree.query(query_points, k=k)
        centroid_order = np.argsort(np.linalg.norm(other_xyz - center, axis=1))
        selected = fixed_indices(np.asarray(near), n_context, centroid_order)
        context_features.append(
            np.concatenate(
                (
                    (other_xyz[selected] - center) / POINT_SCALE_MM,
                    other_nrm[selected],
                    other_same_bone[selected],
                ),
                axis=1,
            )
        )

        source_center = source_centroids[fid]
        pred_center = pred_centroids[fid]
        if gt is not None:
            gt_center = apply_transform(source_center[None], gt[fid])[0]
            residual_rotation = gt[fid][:3, :3] @ pred[fid][:3, :3].T
            dc = gt_center - pred_center
            dr = Rotation.from_matrix(residual_rotation).as_rotvec()
            target_dc.append(dc)
            target_dr.append(dr)
            targets.append(gt_points[fid] - pred_center)
        moving_source_centroids.append(source_center)
        moving_pred_centroids.append(pred_center)

        onehot = np.eye(3, dtype=np.float64)[bone_index(fid)]
        nearest_distance = float(np.linalg.norm(other_xyz[centroid_order[0]] - center))
        globals_.append(
            np.concatenate(
                (
                    (pred_center - assembly_center) / POSITION_SCALE_MM,
                    (pred_center - source_center) / POINT_SCALE_MM,
                    Rotation.from_matrix(pred[fid][:3, :3]).as_rotvec() / math.pi,
                    onehot,
                    [len(common) / 15.0, nearest_distance / 20.0],
                )
            )
        )

    arrays = {
        "case": np.asarray(case),
        "anchor_id": np.asarray(anchor),
        "all_frag_ids": np.asarray(common),
        "moving_frag_ids": np.asarray(moving),
        "base_transforms": np.stack([pred[fid] for fid in common]).astype(np.float64),
        "source_centroids": np.stack(moving_source_centroids).astype(np.float32),
        "pred_centroids": np.stack(moving_pred_centroids).astype(np.float32),
        "fragment": np.stack(current_features).astype(np.float32),
        "context": np.stack(context_features).astype(np.float32),
        "global_features": np.stack(globals_).astype(np.float32),
    }
    if gt is not None:
        arrays.update(
            {
                "target_points": np.stack(targets).astype(np.float32),
                "target_dc": np.stack(target_dc).astype(np.float32),
                "target_dr": np.stack(target_dr).astype(np.float32),
            }
        )
    item = {"case": case, "n_fragments": len(common), "n_moving": len(moving), "anchor": anchor}
    return arrays, item


def command_cache(args) -> None:
    records = discover_records(args.data_dir, args.pred_root)
    if not records:
        raise SystemExit("No complete OBJ/GT/base-prediction cases found")
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for index, record in enumerate(records, 1):
        output = args.cache_dir / f"{record[0]}.npz"
        if args.resume and output.exists():
            with np.load(output) as existing:
                item = {
                    "case": str(existing["case"]),
                    "n_fragments": len(existing["all_frag_ids"]),
                    "n_moving": len(existing["moving_frag_ids"]),
                    "anchor": str(existing["anchor_id"]),
                }
        else:
            item = cache_case(record, output, args.n_fragment, args.n_context)
        manifest.append(item)
        print(f"[{index:3d}/{len(records)}] {record[0]}: {item['n_fragments']} fragments")
    with (args.cache_dir / "manifest.json").open("w") as f:
        json.dump(
            {
                "base_prediction_root": str(args.pred_root.resolve()),
                "data_dir": str(args.data_dir.resolve()),
                "n_fragment_points": args.n_fragment,
                "n_context_points": args.n_context,
                "cases": manifest,
            },
            f,
            indent=2,
        )
    print(f"Cached {len(manifest)} cases in {args.cache_dir}")


def command_make_splits(args) -> None:
    manifest_path = args.cache_dir / "manifest.json"
    with manifest_path.open() as f:
        manifest = json.load(f)["cases"]
    if len(manifest) < args.folds:
        raise SystemExit("Not enough cases for requested folds")

    # Adjacent groups in fragment-count order are distributed one per fold.
    # This keeps fold size and case difficulty balanced without depending on
    # scikit-learn or treating fragments from one patient as independent.
    ordered = sorted(manifest, key=lambda item: (item["n_fragments"], item["case"]))
    rng = random.Random(args.seed)
    val_sets = [[] for _ in range(args.folds)]
    for start in range(0, len(ordered), args.folds):
        block = ordered[start : start + args.folds]
        destinations = list(range(args.folds))
        rng.shuffle(destinations)
        for item, fold in zip(block, destinations):
            val_sets[fold].append(item["case"])
    all_cases = {item["case"] for item in ordered}
    splits = []
    for fold, val in enumerate(val_sets):
        val = sorted(val)
        train = sorted(all_cases - set(val))
        splits.append({"fold": fold, "train": train, "val": val})
    args.split_file.parent.mkdir(parents=True, exist_ok=True)
    with args.split_file.open("w") as f:
        json.dump({"seed": args.seed, "n_folds": args.folds, "splits": splits}, f, indent=2)
    for split in splits:
        counts = [next(x["n_fragments"] for x in manifest if x["case"] == c) for c in split["val"]]
        print(
            f"fold {split['fold']}: train={len(split['train'])}, val={len(split['val'])}, "
            f"val fragments mean={np.mean(counts):.2f} range={min(counts)}-{max(counts)}"
        )


class CachedResidualDataset(Dataset):
    def __init__(self, cache_dir: Path, cases: list[str], augment: bool):
        self.augment = augment
        fields = {key: [] for key in ("fragment", "context", "global_features", "target_points", "target_dc", "target_dr")}
        self.case_ids = []
        for case in cases:
            with np.load(cache_dir / f"{case}.npz") as data:
                n = len(data["moving_frag_ids"])
                for key in fields:
                    fields[key].append(data[key].astype(np.float32))
                self.case_ids.extend([case] * n)
        self.arrays = {key: np.concatenate(value, axis=0) for key, value in fields.items()}

    def __len__(self) -> int:
        return len(self.case_ids)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        item = {key: torch.from_numpy(value[index].copy()) for key, value in self.arrays.items()}
        if self.augment:
            # Geometry sampling noise only; targets stay in the exact physical frame.
            item["fragment"][:, :3] += torch.randn_like(item["fragment"][:, :3]) * 0.003
            item["context"][:, :3] += torch.randn_like(item["context"][:, :3]) * 0.003
            if torch.rand(()) < 0.25:
                keep = torch.rand(len(item["context"])) > 0.10
                item["context"][~keep] = 0
        return item


def residual_loss(model, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, float]]:
    dc, dr = model(batch["fragment"], batch["context"], batch["global_features"])
    current_mm = batch["fragment"][:, :, :3] * POINT_SCALE_MM
    corrected = correct_centered_points(current_mm, dc, dr)
    point = F.smooth_l1_loss(corrected, batch["target_points"], beta=1.0)
    centroid = F.smooth_l1_loss(dc, batch["target_dc"], beta=1.0)
    rotation = F.smooth_l1_loss(dr, batch["target_dr"], beta=math.radians(1.0))
    identity = (dc.square().mean() / 400.0) + (dr.square().mean() / math.radians(25.0) ** 2)
    loss = point + 0.25 * centroid + 2.0 * rotation + 0.01 * identity
    return loss, {
        "point": float(point.detach()),
        "centroid": float(centroid.detach()),
        "rotation": float(rotation.detach()),
    }


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but torch.cuda.is_available() is false")
    return torch.device(name)


def load_splits(path: Path) -> list[dict]:
    with path.open() as f:
        spec = json.load(f)
    return spec["splits"]


def train_fold(args, split: dict, device: torch.device) -> None:
    fold = int(split["fold"])
    train_cases = split["train"][: args.max_train_cases or None]
    outer_val_cases = split["val"]

    # Select training duration only inside the outer-training partition. The
    # outer 34 cases are not even loaded here, so they cannot select an epoch.
    def fragment_count(case: str) -> int:
        with np.load(args.cache_dir / f"{case}.npz") as data:
            return len(data["all_frag_ids"])

    ordered = sorted(train_cases, key=lambda case: (fragment_count(case), case))
    block_size = max(2, round(1.0 / args.inner_val_fraction))
    inner_rng = random.Random(args.seed + 100 * fold)
    inner_val_cases = []
    for start in range(0, len(ordered), block_size):
        block = ordered[start : start + block_size]
        inner_val_cases.append(block[inner_rng.randrange(len(block))])
    inner_val_set = set(inner_val_cases)
    inner_train_cases = [case for case in train_cases if case not in inner_val_set]

    def set_seed(seed: int) -> None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

    def make_model() -> ResidualPointNet:
        model = ResidualPointNet(
            global_dim=GLOBAL_DIM,
            trans_limit_mm=args.trans_limit_mm,
            rot_limit_deg=args.rot_limit_deg,
        ).to(device)
        if args.pretrained is not None:
            checkpoint = torch.load(args.pretrained, map_location="cpu", weights_only=False)
            model.load_state_dict(checkpoint["state_dict"])
        return model

    def make_loader(cases: list[str], augment: bool, seed: int, shuffle: bool) -> DataLoader:
        dataset = CachedResidualDataset(args.cache_dir, cases, augment=augment)
        return DataLoader(
            dataset,
            batch_size=args.batch_size if augment else args.batch_size * 2,
            shuffle=shuffle,
            num_workers=args.workers,
            generator=torch.Generator().manual_seed(seed),
            pin_memory=device.type == "cuda",
        )

    def train_epoch(model, loader, optimizer) -> float:
        model.train()
        losses = []
        for batch in loader:
            batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            loss, _ = residual_loss(model, batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        return float(np.mean(losses))

    def evaluate_loss(model, loader) -> float:
        model.eval()
        losses = []
        with torch.no_grad():
            for batch in loader:
                batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
                loss, _ = residual_loss(model, batch)
                losses.append(float(loss))
        return float(np.mean(losses))

    # Phase A: inner validation selects only the number of epochs.
    selection_seed = args.seed + fold
    set_seed(selection_seed)
    model = make_model()
    inner_train_loader = make_loader(inner_train_cases, True, selection_seed, True)
    inner_val_loader = make_loader(inner_val_cases, False, selection_seed, False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    best_inner_loss = float("inf")
    best_epoch = 1
    stale = 0
    for epoch in range(args.epochs):
        train_loss = train_epoch(model, inner_train_loader, optimizer)
        scheduler.step()
        inner_loss = evaluate_loss(model, inner_val_loader)
        if inner_loss < best_inner_loss - args.inner_min_delta:
            best_inner_loss = inner_loss
            best_epoch = epoch + 1
            stale = 0
        else:
            stale += 1
        if epoch == 0 or (epoch + 1) % args.report_every == 0:
            print(
                f"fold={fold} select epoch={epoch + 1:03d}/{args.epochs} "
                f"train={train_loss:.5f} inner={inner_loss:.5f} best={best_epoch}"
            )
        if stale >= args.inner_patience:
            print(f"fold={fold} inner early stop at {epoch + 1}; selected epoch={best_epoch}")
            break
    del model, optimizer, scheduler
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # --fixed-epochs overrides what phase A chose, for diagnosing whether a configuration
    # underperforms because of the DURATION or because of the data. It is a diagnostic, not
    # a submission path: a duration picked by looking at scores on the 170 outer cases is
    # exactly the leakage the nested protocol exists to prevent, and the doc already records
    # that a fixed 200-epoch schedule was unsafe. Phase A still runs, so its choice is
    # printed and can be compared against the override.
    if getattr(args, "fixed_epochs", None):
        print(f"fold={fold} OVERRIDE: refit for {args.fixed_epochs} epochs "
              f"instead of the inner-selected {best_epoch}")
        best_epoch = args.fixed_epochs

    # Phase B: restart and use every one of the 136 outer-training cases for
    # exactly the selected duration. No outer-validation loss is computed.
    final_seed = args.seed + 1000 + fold
    set_seed(final_seed)
    model = make_model()
    final_loader = make_loader(train_cases, True, final_seed, True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    for epoch in range(best_epoch):
        train_loss = train_epoch(model, final_loader, optimizer)
        scheduler.step()
        if epoch == 0 or (epoch + 1) % args.report_every == 0 or epoch + 1 == best_epoch:
            print(
                f"fold={fold} refit epoch={epoch + 1:03d}/{best_epoch} "
                f"train={train_loss:.5f} (outer val untouched)"
            )

    args.weights_dir.mkdir(parents=True, exist_ok=True)
    path = args.weights_dir / f"fold{fold}_last.pt"
    torch.save(
        {
            "state_dict": model.cpu().state_dict(),
            "fold": fold,
            "epochs": best_epoch,
            "max_selection_epochs": args.epochs,
            "seed": final_seed,
            "global_dim": GLOBAL_DIM,
            "trans_limit_mm": args.trans_limit_mm,
            "rot_limit_deg": args.rot_limit_deg,
            "train_cases": train_cases,
            "val_cases": outer_val_cases,
            "inner_train_cases": inner_train_cases,
            "inner_val_cases": inner_val_cases,
            "inner_best_loss": best_inner_loss,
            "selection_policy": "inner validation chooses duration; outer validation untouched",
        },
        path,
    )
    print(f"saved {path}")


def command_train_cv(args) -> None:
    device = resolve_device(args.device)
    splits = load_splits(args.split_file)
    requested = set(args.folds if args.folds is not None else range(len(splits)))
    for split in splits:
        if int(split["fold"]) in requested:
            train_fold(args, split, device)


def command_pretrain_cached(args) -> None:
    """Pretrain on cached frozen-AssemblyNet simulation errors."""
    device = resolve_device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    train_cases = sorted(path.stem for path in args.train_cache.glob("*.npz"))
    val_cases = sorted(path.stem for path in args.val_cache.glob("*.npz"))
    if not train_cases or not val_cases:
        raise SystemExit("Both train and validation cache directories need NPZ cases")
    train_set = CachedResidualDataset(args.train_cache, train_cases, augment=True)
    val_set = CachedResidualDataset(args.val_cache, val_cases, augment=False)
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
        generator=torch.Generator().manual_seed(args.seed), pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size * 2, shuffle=False, num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )
    model = ResidualPointNet(
        global_dim=GLOBAL_DIM,
        trans_limit_mm=args.trans_limit_mm,
        rot_limit_deg=args.rot_limit_deg,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    best_loss, best_epoch, stale = float("inf"), 0, 0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(args.epochs):
        model.train()
        train_losses = []
        for batch in train_loader:
            batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            loss, _ = residual_loss(model, batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(float(loss.detach()))
        scheduler.step()
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
                loss, _ = residual_loss(model, batch)
                val_losses.append(float(loss))
        val_loss = float(np.mean(val_losses))
        if val_loss < best_loss - args.min_delta:
            best_loss, best_epoch, stale = val_loss, epoch + 1, 0
            torch.save(
                {
                    "state_dict": model.cpu().state_dict(),
                    "epoch": best_epoch,
                    "sim_val_loss": best_loss,
                    "global_dim": GLOBAL_DIM,
                    "trans_limit_mm": args.trans_limit_mm,
                    "rot_limit_deg": args.rot_limit_deg,
                    "seed": args.seed,
                    "train_cases": train_cases,
                    "val_cases": val_cases,
                    "source": "cached real epoch984 beta0.3 simulation residuals; no clinical labels",
                },
                args.out,
            )
            model.to(device)
        else:
            stale += 1
        if epoch == 0 or (epoch + 1) % args.report_every == 0:
            print(
                f"epoch={epoch + 1:03d}/{args.epochs} train={np.mean(train_losses):.5f} "
                f"sim_val={val_loss:.5f} best={best_epoch} stale={stale}", flush=True,
            )
        if stale >= args.patience:
            print(f"early stop at {epoch + 1}; best={best_epoch} sim_val={best_loss:.5f}")
            break
    print(f"saved {args.out}; fragments train={len(train_set)} val={len(val_set)}")


def load_model(path: Path, device: torch.device) -> ResidualPointNet:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model = ResidualPointNet(
        global_dim=checkpoint.get("global_dim", GLOBAL_DIM),
        trans_limit_mm=checkpoint["trans_limit_mm"],
        rot_limit_deg=checkpoint["rot_limit_deg"],
    )
    model.load_state_dict(checkpoint["state_dict"])
    return model.to(device).eval()


def predict_arrays(
    data: dict[str, np.ndarray],
    models: list[ResidualPointNet],
    device: torch.device,
) -> list[dict]:
    """Apply one or more residual heads to GT-free per-case feature arrays."""
    fragment = torch.from_numpy(np.asarray(data["fragment"])).to(device)
    context = torch.from_numpy(np.asarray(data["context"])).to(device)
    global_features = torch.from_numpy(np.asarray(data["global_features"])).to(device)
    all_ids = [str(x) for x in data["all_frag_ids"]]
    moving_ids = [str(x) for x in data["moving_frag_ids"]]
    transforms = {
        fid: matrix.copy()
        for fid, matrix in zip(all_ids, np.asarray(data["base_transforms"]))
    }
    source_centroids = np.asarray(data["source_centroids"]).astype(np.float64)
    pred_centroids = np.asarray(data["pred_centroids"]).astype(np.float64)
    dc_all, dr_all = [], []
    with torch.no_grad():
        for model in models:
            dc, dr = model(fragment, context, global_features)
            dc_all.append(dc.cpu().numpy())
            dr_all.append(dr.cpu().numpy())
    dc_mean = np.mean(dc_all, axis=0)
    dr_mean = np.mean(dr_all, axis=0)
    for index, fid in enumerate(moving_ids):
        base = transforms[fid]
        rotation = Rotation.from_rotvec(dr_mean[index]).as_matrix() @ base[:3, :3]
        centroid = pred_centroids[index] + dc_mean[index]
        corrected = np.eye(4, dtype=np.float64)
        corrected[:3, :3] = rotation
        corrected[:3, 3] = centroid - rotation @ source_centroids[index]
        transforms[fid] = corrected
    return [
        {"fragment_id": fid, "transformation": transforms[fid].tolist()}
        for fid in sorted(transforms, key=int)
    ]


def predict_case(cache_path: Path, models: list[ResidualPointNet], device: torch.device) -> list[dict]:
    with np.load(cache_path) as cached:
        data = {key: cached[key] for key in cached.files}
    return predict_arrays(data, models, device)


def write_prediction(path: Path, prediction: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(prediction, f, indent=2)


def command_predict_oof(args) -> None:
    device = resolve_device(args.device)
    seen = set()
    for split in load_splits(args.split_file):
        fold = int(split["fold"])
        model = load_model(args.weights_dir / f"fold{fold}_last.pt", device)
        for case in split["val"]:
            if case in seen:
                raise RuntimeError(f"OOF leakage/duplicate: {case} occurs in multiple val folds")
            prediction = predict_case(args.cache_dir / f"{case}.npz", [model], device)
            write_prediction(args.out_dir / case / PRED_NAME, prediction)
            seen.add(case)
        print(f"fold {fold}: wrote {len(split['val'])} held-out predictions")
    all_cases = {path.stem for path in args.cache_dir.glob("*.npz")}
    if seen != all_cases:
        raise RuntimeError(f"OOF coverage mismatch: predicted={len(seen)}, cache={len(all_cases)}")
    print(f"OOF complete: {len(seen)} unique cases")


def command_predict_ensemble(args) -> None:
    device = resolve_device(args.device)
    paths = sorted(args.weights_dir.glob("fold*_last.pt"))
    if len(paths) != 5:
        raise SystemExit(f"Expected exactly 5 fold checkpoints, found {len(paths)}")
    models = [load_model(path, device) for path in paths]
    cases = sorted(path.stem for path in args.cache_dir.glob("*.npz"))
    for case in cases:
        prediction = predict_case(args.cache_dir / f"{case}.npz", models, device)
        write_prediction(args.out_dir / case / PRED_NAME, prediction)
    print(f"Five-fold ensemble wrote {len(cases)} cases to {args.out_dir}")


def command_predict_case(args) -> None:
    """End-to-end residual inference for one hidden case, without any GT."""
    device = resolve_device(args.device)
    paths = sorted(args.weights_dir.glob("fold*_last.pt"))
    if len(paths) != 5:
        raise SystemExit(f"Expected exactly 5 fold checkpoints, found {len(paths)}")
    case = args.case_id or args.obj.parent.name or args.obj.stem
    arrays, item = build_case_arrays(
        case=case,
        obj_path=args.obj,
        pred_path=args.base_pred,
        n_fragment=args.n_fragment,
        n_context=args.n_context,
        gt_path=None,
    )
    models = [load_model(path, device) for path in paths]
    prediction = predict_arrays(arrays, models, device)
    write_prediction(args.out, prediction)
    print(
        f"Five-fold residual ensemble wrote {item['n_fragments']} fragments "
        f"for case {case} to {args.out}"
    )


def command_predict_single(args) -> None:
    device = resolve_device(args.device)
    model = load_model(args.checkpoint, device)
    cases = sorted(path.stem for path in args.cache_dir.glob("*.npz"))
    for case in cases:
        prediction = predict_case(args.cache_dir / f"{case}.npz", [model], device)
        write_prediction(args.out_dir / case / PRED_NAME, prediction)
    print(f"Single residual checkpoint wrote {len(cases)} cases to {args.out_dir}")


def command_score(args) -> None:
    evaluator = load_evaluator()
    rows = []
    for case_dir in sorted(args.data_dir.iterdir()):
        if not case_dir.is_dir():
            continue
        objs = sorted(case_dir.glob("*.obj"))
        gt_path = case_dir / "plan_pl_gt.json"
        base_path = args.base_pred_root / case_dir.name / PRED_NAME
        residual_path = args.residual_pred_root / case_dir.name / PRED_NAME
        if not (objs and gt_path.exists() and base_path.exists() and residual_path.exists()):
            continue
        meshes = evaluator.load_obj_meshes(str(objs[0]))
        gt = load_poses(gt_path)
        row = {"case": case_dir.name}
        for name, path in (("base", base_path), ("residual", residual_path)):
            metrics = evaluator.evaluate_sample(
                meshes, gt, load_poses(path), args.pa_threshold,
                npoints_bone=args.npoints_bone, npoints_frag=args.npoints_frag,
                seed=args.eval_seed,
            )
            row.update(
                {
                    f"{name}_tre_mm": metrics["tre_mm"],
                    f"{name}_trans_mm": metrics["trans_mean_mm"],
                    f"{name}_rot_deg": metrics["rot_mean_deg"],
                    f"{name}_pa": metrics["pa"],
                    f"{name}_cd_mm": metrics["sample_cd_raw_mm"],
                }
            )
        rows.append(row)
        print(f"[{len(rows):3d}] {case_dir.name}")
    if not rows:
        raise SystemExit("No scoreable cases")
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print("\npaired OOF means")
    for metric in ("tre_mm", "trans_mm", "rot_deg", "pa", "cd_mm"):
        base = np.mean([row[f"base_{metric}"] for row in rows])
        residual = np.mean([row[f"residual_{metric}"] for row in rows])
        scale = 100.0 if metric == "pa" else 1.0
        print(f"{metric:9s}: base={base*scale:.4f} residual={residual*scale:.4f} delta={(residual-base)*scale:+.4f}")
    print(f"saved {args.out_csv}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    cache = sub.add_parser("cache")
    cache.add_argument("--data-dir", type=Path, required=True)
    cache.add_argument("--pred-root", type=Path, required=True)
    cache.add_argument("--cache-dir", type=Path, required=True)
    cache.add_argument("--n-fragment", type=int, default=256)
    cache.add_argument("--n-context", type=int, default=512)
    cache.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    cache.set_defaults(func=command_cache)

    split = sub.add_parser("make-splits")
    split.add_argument("--cache-dir", type=Path, required=True)
    split.add_argument("--split-file", type=Path, required=True)
    split.add_argument("--folds", type=int, default=5)
    split.add_argument("--seed", type=int, default=42)
    split.set_defaults(func=command_make_splits)

    train = sub.add_parser("train-cv")
    train.add_argument("--cache-dir", type=Path, required=True)
    train.add_argument("--split-file", type=Path, required=True)
    train.add_argument("--weights-dir", type=Path, required=True)
    train.add_argument("--folds", type=int, nargs="+")
    train.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    train.add_argument("--epochs", type=int, default=200)
    train.add_argument("--batch-size", type=int, default=64)
    train.add_argument("--workers", type=int, default=2)
    train.add_argument("--lr", type=float, default=3e-4)
    train.add_argument("--weight-decay", type=float, default=1e-3)
    train.add_argument("--trans-limit-mm", type=float, default=20.0)
    train.add_argument("--rot-limit-deg", type=float, default=25.0)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--report-every", type=int, default=10)
    train.add_argument("--inner-val-fraction", type=float, default=0.15)
    train.add_argument("--inner-patience", type=int, default=30)
    train.add_argument("--inner-min-delta", type=float, default=1e-3)
    train.add_argument("--pretrained", type=Path)
    train.add_argument("--max-train-cases", type=int, help="smoke-test only; never use for the official OOF")
    train.add_argument("--fixed-epochs", type=int,
                       help="diagnostic only: refit for exactly this many epochs, ignoring the "
                            "inner-validation choice. Never use for a submission -- a duration "
                            "chosen from outer-fold scores defeats the nested protocol.")
    train.set_defaults(func=command_train_cv)

    pretrain = sub.add_parser("pretrain-cached")
    pretrain.add_argument("--train-cache", type=Path, required=True)
    pretrain.add_argument("--val-cache", type=Path, required=True)
    pretrain.add_argument("--out", type=Path, required=True)
    pretrain.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    pretrain.add_argument("--epochs", type=int, default=200)
    pretrain.add_argument("--patience", type=int, default=20)
    pretrain.add_argument("--min-delta", type=float, default=1e-3)
    pretrain.add_argument("--batch-size", type=int, default=64)
    pretrain.add_argument("--workers", type=int, default=2)
    pretrain.add_argument("--lr", type=float, default=3e-4)
    pretrain.add_argument("--weight-decay", type=float, default=1e-3)
    pretrain.add_argument("--trans-limit-mm", type=float, default=20.0)
    pretrain.add_argument("--rot-limit-deg", type=float, default=25.0)
    pretrain.add_argument("--seed", type=int, default=42)
    pretrain.add_argument("--report-every", type=int, default=5)
    pretrain.set_defaults(func=command_pretrain_cached)

    for name, function in (("predict-oof", command_predict_oof), ("predict-ensemble", command_predict_ensemble)):
        predict = sub.add_parser(name)
        predict.add_argument("--cache-dir", type=Path, required=True)
        if name == "predict-oof":
            predict.add_argument("--split-file", type=Path, required=True)
        predict.add_argument("--weights-dir", type=Path, required=True)
        predict.add_argument("--out-dir", type=Path, required=True)
        predict.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
        predict.set_defaults(func=function)

    predict_case_parser = sub.add_parser(
        "predict-case",
        help="apply the five-fold residual ensemble to one OBJ/base prediction without GT",
    )
    predict_case_parser.add_argument("--obj", type=Path, required=True)
    predict_case_parser.add_argument("--base-pred", type=Path, required=True)
    predict_case_parser.add_argument("--weights-dir", type=Path, required=True)
    predict_case_parser.add_argument("--out", type=Path, required=True)
    predict_case_parser.add_argument("--case-id")
    predict_case_parser.add_argument("--n-fragment", type=int, default=256)
    predict_case_parser.add_argument("--n-context", type=int, default=512)
    predict_case_parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    predict_case_parser.set_defaults(func=command_predict_case)

    single = sub.add_parser("predict-single")
    single.add_argument("--cache-dir", type=Path, required=True)
    single.add_argument("--checkpoint", type=Path, required=True)
    single.add_argument("--out-dir", type=Path, required=True)
    single.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    single.set_defaults(func=command_predict_single)

    score = sub.add_parser("score")
    score.add_argument("--data-dir", type=Path, required=True)
    score.add_argument("--base-pred-root", type=Path, required=True)
    score.add_argument("--residual-pred-root", type=Path, required=True)
    score.add_argument("--out-csv", type=Path, required=True)
    score.add_argument("--pa-threshold", type=float, default=0.05)
    score.add_argument("--npoints-bone", type=int, default=5000)
    score.add_argument("--npoints-frag", type=int, default=1000)
    score.add_argument("--eval-seed", type=int, default=42)
    score.set_defaults(func=command_score)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
