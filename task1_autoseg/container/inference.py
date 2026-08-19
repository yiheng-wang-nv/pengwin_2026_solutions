"""PENGWIN 2026 Task 1 — Grand Challenge container entrypoint.

Single-image inference: read one CT from /input, write one instance
segmentation to /output. Same two-stage pipeline as scripts/e2e_predict.py
(anatomical ds001 -> global gate -> per-bone CSM ds002 -> frac_to_instance ->
offset + merge), adapted for the GC single-input contract.

Model weights come from /opt/ml/model (nnUNet_results), populated by the
separately-uploaded model tarball. Network is disabled at runtime.

GC I/O (paths are globbed defensively because the interface slug spelling
"pelvic-facture-ct" is easy to mismatch):
  input :  /input/**/*.mha   (the CT)
  output:  /output/images/pelvic-fracture-segmentation/<stem>.mha
"""

from __future__ import annotations

import os
import subprocess
import sys
from glob import glob
from pathlib import Path

import numpy as np
import SimpleITK as sitk

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from frac_to_instance import merged_mask_to_instance

# ---- paths (overridable via env for local testing) ----
# Grand Challenge interface (Task 1 phase):
#   Input  "Peripelvic Fracture CT"              -> /input/images/peripelvic-fracture-ct/
#   Output "Peripelvic Fracture CT Segmentation" -> /output/images/peripelvic-fracture-ct-segmentation/
# Input is globbed (slug-agnostic). Output slug must match the phase exactly;
# override with OUTPUT_DIR or OUTPUT_SLUG env if GC reports a different slug.
INPUT_DIR = Path(os.environ.get("INPUT_DIR", "/input"))
_OUT_SLUG = os.environ.get("OUTPUT_SLUG", "peripelvic-fracture-ct-segmentation")
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", f"/output/images/{_OUT_SLUG}"))
WORK = Path(os.environ.get("WORKDIR", "/tmp/work"))

# Instance-decoding parameters. The official baseline ships kernel_size=5.
#
# k=7 was chosen from a 28-config sweep on held-out cases; ccf_threshold is flat across
# 20..500 and stays at the official 100. Both are env-overridable.
KERNEL_SIZE = int(os.environ.get("KERNEL_SIZE", "7"))
CCF_THRESHOLD = int(os.environ.get("CCF_THRESHOLD", "100"))
DEVICE = os.environ.get("DEVICE", "cuda")

# ---- model config (must match the model tarball layout under /opt/ml/model) ----
# tta=False for both: ds001 mirror TTA swaps leftHip/rightHip; ds002 mirror TTA
# was tested on 20 val and gave NO gain (Dice 0.739->0.736) at 8x inference cost.
DS001 = dict(d="001", tr="nnUNetTrainerNoMirroring", p="nnUNetResEncUNetXLPlans",
             c="3d_fullres", chk="checkpoint_best.pth", tta=False)
DS002 = dict(d="002", tr="nnUNetTrainer", p="nnUNetResEncUNetXLPlans",
             c="3d_fullres", chk="checkpoint_best.pth", tta=False)


def discover_folds(m) -> list:
    """Folds to predict with = whatever the mounted model tarball actually ships.

    The same image is reused across model packs that differ only in folds:
    a single champion fold (fold_0), a cross-validation ensemble (fold_0..4, whose
    softmax nnU-Net averages), or a model trained on all data (fold_all). Hardcoding
    a fold list silently loads the wrong thing — `-f 0` finds nothing in an all-data
    pack — so read it from disk instead. Falls back to ["0"] if the layout is
    unexpected, which is the historical behaviour.
    """
    root = Path(os.environ.get("nnUNet_results", "/opt/ml/model"))
    pat = str(root / f"Dataset{m['d']}_*" / f"{m['tr']}__{m['p']}__{m['c']}" / "fold_*")
    names = sorted(os.path.basename(p) for p in glob(pat) if os.path.isdir(p))
    if not names:
        log(f"WARNING: no fold_* under {pat}; defaulting to fold 0")
        return ["0"]
    if "fold_all" in names:
        return ["all"]
    return [n.split("_", 1)[1] for n in names]

BONES = {1: ("sacrum", 0), 2: ("leftHip", 50), 3: ("rightHip", 100), 4: ("femur", 150)}
PELVIC_CLASSES = (1, 2, 3)
FEMUR_CLASS = 4


# --- OFFICIAL pelvic-vs-femur routing (PENGWIN 2026 Update Notice, Tasks 1&2) ---
# Each case is EITHER a pelvic OR a femur case; the algorithm must output the
# matching fragment set. The organizers require a DETERMINISTIC geometry rule
# (image spacing + physical FOV), NOT a prediction-based heuristic, and verified
# the test sets conform to it. Copied VERBATIM from the notice — note the
# deliberately unconventional axis naming (spacing_z = sitk spacing[0]) that
# guards against the SimpleITK (x,y,z) vs numpy (z,y,x) axis-order confusion;
# do not "fix" it.
def get_image_info(image_nii) -> dict:
    import SimpleITK as sitk
    img = sitk.ReadImage(str(image_nii))
    arr = sitk.GetArrayFromImage(img)  # (z, y, x)
    sp = img.GetSpacing()
    return {
        "dim_z": arr.shape[0], "dim_y": arr.shape[1], "dim_x": arr.shape[2],
        "spacing_z": sp[0], "spacing_y": sp[1], "spacing_x": sp[2],
        "physical_z_mm": sp[0] * arr.shape[0], "physical_x_mm": sp[2] * arr.shape[2],
    }


def classify_pelvic_femur(spacing_x, spacing_y, spacing_z, physical_x_mm, physical_z_mm):
    if physical_x_mm <= 285.35:
        if spacing_x <= 0.71:
            return "pelvic"
        elif spacing_z <= 0.90:
            return "femur"
        else:
            return "pelvic" if spacing_y <= 0.91 else "femur"
    else:
        if spacing_z <= 0.68:
            return "pelvic" if physical_z_mm <= 193.55 else "femur"
        else:
            return "pelvic" if physical_z_mm <= 390.78 else "femur"


def log(*a):
    print(*a, flush=True)


def run(cmd):
    log("  $", " ".join(cmd))
    subprocess.run(cmd, check=True)


def find_input_ct() -> Path:
    # GC mounts the image at /input/images/<slug>/<uuid>.mha. Glob defensively.
    cands = sorted(glob(str(INPUT_DIR / "**" / "*.mha"), recursive=True))
    cands += sorted(glob(str(INPUT_DIR / "**" / "*.mhd"), recursive=True))
    cands += sorted(glob(str(INPUT_DIR / "**" / "*.nii.gz"), recursive=True))
    if not cands:
        sys.exit(f"No input image found under {INPUT_DIR}")
    if len(cands) > 1:
        log(f"WARNING: multiple inputs found, using first: {cands}")
    return Path(cands[0])


def write_like(arr, ref_img, path, dtype):
    out = sitk.GetImageFromArray(arr)
    out.CopyInformation(ref_img)
    out = sitk.Cast(out, dtype)
    sitk.WriteImage(out, str(path), useCompression=True)


def predict(input_dir, output_dir, m):
    folds = discover_folds(m)
    cmd = [
        "nnUNetv2_predict", "-i", str(input_dir), "-o", str(output_dir),
        "-d", m["d"], "-tr", m["tr"], "-p", m["p"], "-c", m["c"],
        "-f", *folds, "-chk", m["chk"], "-step_size", "0.5",
        "-device", DEVICE,
    ]
    # Grand Challenge gives 16GB of VRAM. nnU-Net assembles the full-image logits
    # array (n_classes x whole volume) on the GPU; that array is what scales with case
    # size (patch activations are fixed by patch_size, and fold weights load with
    # map_location='cpu'). --not_on_device keeps it in host memory instead.
    #
    # Enabled automatically only for MULTI-model configs, which is where the 5-fold
    # submission OOMed: more folds do not multiply VRAM, they re-predict the same
    # large case once per fold, multiplying the chances of hitting the ceiling and
    # fragmenting the pool. A single fold keeps the on-device path, which the champion
    # proved on this exact 16GB device over the whole final test set — slowing that
    # one down would only risk trading an OOM for a timeout.
    # Override either way with NNUNET_NOT_ON_DEVICE=1 / =0.
    env = os.environ.get("NNUNET_NOT_ON_DEVICE")
    not_on_device = (len(folds) > 1) if env is None else (env == "1")
    if not_on_device:
        cmd.append("--not_on_device")
        log(f"    [{len(folds)} folds] --not_on_device: logits array kept on host memory")
    if not m.get("tta", False):
        cmd.append("--disable_tta")
    run(cmd)


def global_gate(ana, ct_path):
    # Route by the official deterministic geometry rule on the INPUT CT (not by
    # predicted voxel counts, which fail when the model mispredicts the case type).
    info = get_image_info(ct_path)
    region = classify_pelvic_femur(
        info["spacing_x"], info["spacing_y"], info["spacing_z"],
        info["physical_x_mm"], info["physical_z_mm"],
    )
    g = ana.copy()
    if region == "femur":
        g[np.isin(g, PELVIC_CLASSES)] = 0
    else:  # pelvic
        g[g == FEMUR_CLASS] = 0
    return g, region


def main():
    for d in [WORK / "ana_in", WORK / "ana_out", WORK / "bone_in", WORK / "bone_out"]:
        d.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    ct_path = find_input_ct()
    stem = ct_path.name
    for suf in [".mha", ".mhd", ".nii.gz"]:
        if stem.endswith(suf):
            stem = stem[: -len(suf)]
            break
    log(f"Input CT : {ct_path}  (stem={stem})")
    log(f"Model dir: {os.environ.get('nnUNet_results')}")

    img = sitk.ReadImage(str(ct_path))
    ct = sitk.GetArrayFromImage(img)

    # ---- Stage 1: anatomical ----
    log("[1] anatomical (ds001)")
    write_like(ct, img, WORK / "ana_in" / "case_0000.nii.gz", sitk.sitkFloat32)
    predict(WORK / "ana_in", WORK / "ana_out", DS001)
    ana = sitk.GetArrayFromImage(sitk.ReadImage(str(WORK / "ana_out" / "case.nii.gz")))

    # ---- Stage 2: gate + per-bone masked CT ----
    gated, region = global_gate(ana, ct_path)
    log(f"[2] region={region} (official geometry rule)")
    present = []
    for cls, (bone, _off) in BONES.items():
        mask = gated == cls
        if mask.sum() == 0:
            continue
        present.append(cls)
        masked = np.where(mask, ct, 0).astype(ct.dtype)
        write_like(masked, img, WORK / "bone_in" / f"{bone}_0000.nii.gz", sitk.sitkFloat32)
    log(f"    bones: {[BONES[c][0] for c in present]}")

    # ---- Stage 3: CSM ----
    log("[3] CSM (ds002)")
    final = np.zeros(ct.shape, dtype=np.int32)
    if present:
        predict(WORK / "bone_in", WORK / "bone_out", DS002)
        # ---- Stage 4: instance + offset + merge ----
        log("[4] instance + merge")
        for cls in present:
            bone, offset = BONES[cls]
            csm = WORK / "bone_out" / f"{bone}.nii.gz"
            if not csm.exists():
                log(f"    WARN no CSM for {bone}")
                continue
            inst, _ = merged_mask_to_instance(
                str(csm), kernel_size=KERNEL_SIZE, ccf_threshold=CCF_THRESHOLD,
                device=("cuda" if DEVICE == "cuda" else "cpu"))
            m = inst > 0
            final[m] = inst[m].astype(np.int32) + offset
    else:
        log("    no bones detected; writing empty segmentation")

    out_path = OUTPUT_DIR / f"{stem}.mha"
    write_like(final, img, out_path, sitk.sitkUInt8)
    n = len(np.unique(final)) - (1 if (final == 0).any() else 0)
    log(f"Wrote {out_path}  ({n} fragments)")


if __name__ == "__main__":
    main()
