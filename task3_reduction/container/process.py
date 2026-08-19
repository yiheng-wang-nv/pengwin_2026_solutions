#!/usr/bin/env python3
"""Grand Challenge entrypoint: N AssemblyNet bases, each with its own five-fold residual,
combined by SO(3) rotation + fragment-centroid displacement.

Member order, weights and per-member checkpoint config all come from the model tar's
manifest.json rather than being hard-coded, so the image does not have to be rebuilt to change
the ensemble -- swapping the model tar is enough. W_ROT / W_TRANS still override the manifest
if the ensemble has to be backed out under time pressure (W_ROT=1,0,0,0,0 with W_TRANS the
same reproduces the champion line alone).
"""
from __future__ import annotations
import json, os, shutil, subprocess, sys, tempfile
from pathlib import Path

# cuda by default; DEVICE=cpu lets the image be verified where no nvidia runtime is available
# (this host's docker has none) and is the fallback if the GC node ever hands us no GPU.
DEVICE = os.environ.get("DEVICE", "cuda")
INPUT_DIR = Path(os.environ.get("INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "/output"))
MODEL_DIR = Path(os.environ.get("MODEL_DIR", "/opt/ml/model"))
ALGORITHM_DIR = Path("/opt/algorithm")
BASELINE = ALGORITHM_DIR / "external/PENGWIN2026_Task3_Reduction_Baseline"
# Per-member configs are generated at run time, so they go to a writable temp dir rather than
# into the image: Grand Challenge runs the container as a non-root user with a read-only
# /opt/algorithm, and writing next to the baseline's own configs fails with EACCES there.
CONFIGS = Path(tempfile.mkdtemp(prefix="gc-multi-cfg-"))

CONFIG_TEMPLATE = """model_config: "model/AssemblyTransformer_coords"
checkpoint: {ckpt}
input_dir: {input_dir}
npoints: 5000
max_iters: 20
convergence_threshold: 2.0
seed: 42
update_scale: 0.3
convergence_metric: max_point
debug_vis: false
output_type: "coords"
experiment_name: "gc_multi_{tag}"
"""


def main() -> None:
    objects = sorted(INPUT_DIR.rglob("*.obj"))
    if len(objects) != 1:
        raise SystemExit(f"expected exactly one OBJ below {INPUT_DIR}, found {len(objects)}")

    manifest_path = MODEL_DIR / "manifest.json"
    if not manifest_path.exists():
        raise SystemExit(f"missing {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    order = manifest["order"]
    w_rot = [str(x) for x in manifest["w_rot"]]
    w_trans = [str(x) for x in manifest["w_trans"]]
    if os.environ.get("W_ROT"):
        w_rot = os.environ["W_ROT"].split(",")
    if os.environ.get("W_TRANS"):
        w_trans = os.environ["W_TRANS"].split(",")
    if len(w_rot) != len(order) or len(w_trans) != len(order):
        raise SystemExit(f"{len(order)} members but {len(w_rot)} rot / {len(w_trans)} trans weights")

    # inference.py resolves model_config as <config dir>/<model_config>.yaml, so the model
    # definition has to sit beside the generated configs, not beside the baseline's own.
    shutil.copytree(BASELINE / "configs" / "model", CONFIGS / "model", dirs_exist_ok=True)

    configs, weight_dirs = [], []
    for tag in order:
        ckpt = MODEL_DIR / f"model_{tag}.ckpt"
        resid = MODEL_DIR / f"residual_{tag}"
        if not ckpt.exists():
            raise SystemExit(f"missing AssemblyNet checkpoint {ckpt}")
        folds = sorted(resid.glob("fold*_last.pt"))
        if len(folds) != 5:
            raise SystemExit(f"expected five residual checkpoints in {resid}, got {len(folds)}")
        cfg = CONFIGS / f"gc_multi_{tag}.yaml"
        cfg.write_text(CONFIG_TEMPLATE.format(ckpt=ckpt, input_dir=objects[0].parent, tag=tag))
        configs.append(str(cfg))
        weight_dirs.append(str(resid))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output = OUTPUT_DIR / "reduction-poses-matrices.json"
    command = [
        sys.executable,
        str(ALGORITHM_DIR / "scripts/task3/infer_multi_base_ensemble_case.py"),
        "--obj", str(objects[0]),
        "--out", str(output),
        "--case-id", objects[0].parent.name,
        "--base-config", *configs,
        "--weights-dir", *weight_dirs,
        "--w-rot", *w_rot,
        "--w-trans", *w_trans,
        "--device", DEVICE,
        "--seed", "42",
        "--npoints", "5000",
        "--max-iters", "20",
        "--convergence-threshold", "2.0",
        "--update-scale", "0.3",
        "--convergence-metric", "max_point",
    ]
    print(f"members={order} w_rot={w_rot} w_trans={w_trans}", flush=True)
    print("running:", " ".join(command), flush=True)
    subprocess.run(command, check=True)
    if not output.exists() or output.stat().st_size == 0:
        raise SystemExit(f"inference did not produce {output}")
    print(f"completed: {output}", flush=True)


if __name__ == "__main__":
    main()
