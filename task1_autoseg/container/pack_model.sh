#!/usr/bin/env bash
# Build the model tarball uploaded separately to Grand Challenge.
# Extracts to /opt/ml/model at runtime (= nnUNet_results).
#
# Packs exactly what the predictor needs:
#   Dataset001_.../nnUNetTrainerNoMirroring__nnUNetResEncUNetXLPlans__3d_fullres/
#     ├── fold_0..fold_4/checkpoint_best.pth   (the retrained 5-fold ENSEMBLE)
#     ├── plans.json
#     └── dataset.json
#   Dataset002_.../nnUNetTrainer__nnUNetResEncUNetXLPlans__3d_fullres/
#     ├── fold_0/checkpoint_best.pth           (ORIGINAL champion single fold)
#     ├── plans.json
#     └── dataset.json
#
# Two subtleties that silently cost ~0.03 dice if you get them wrong:
#   * ds001 folds are staged from their checkpoint_FINAL. Packing nnU-Net's
#     checkpoint_best instead drops the same ensemble to 0.8501 — "best" is picked by
#     a noisy EMA pseudo-dice, and averaging 5 noisy picks compounds the noise. They
#     are still WRITTEN as checkpoint_best.pth because that is the filename
#     inference.py requests.
#   * ds001 ships LRMirror weights but the checkpoint trainer_name is re-labeled to
#     nnUNetTrainerNoMirroring (+ empty mirror axes): the custom trainer class is not
#     in the container and nnU-Net rebuilds the network from trainer_name.
#     Architecture is identical; NoMirroring just means no mirror TTA.
#
set -euo pipefail
# REPO must point at the WORKING TREE that holds nnUNet_workdir/, weights/ and the
# output_* directories -- that tree is not part of this repository, because it contains
# trained weights and multi-GB intermediates. Set it before running:
#   REPO=/path/to/working/tree bash pack_model.sh
REPO="${REPO:?set REPO to the working tree containing nnUNet_workdir/ and weights/}"
PROFILE="${PROFILE:-alldata}"
RESULTS_ALL1="${RESULTS_ALL1:-$REPO/nnUNet_workdir/results_alldata/001_alldata}"
RESULTS_ALL2="${RESULTS_ALL2:-$REPO/nnUNet_workdir/results_alldata/002_alldata}"
OUT="${1:-$REPO/docker/model.tar.gz}"

DS1_DST="Dataset001_PENGWIN_Anatomical/nnUNetTrainerNoMirroring__nnUNetResEncUNetXLPlans__3d_fullres"
DS2_DST="Dataset002_PENGWIN_Frac/nnUNetTrainer__nnUNetResEncUNetXLPlans__3d_fullres"

case "$PROFILE" in
  alldata)
    DS1_ROOT="$RESULTS_ALL1"; DS1_FOLDS="all"; DS1_SRC_CHK="checkpoint_final.pth"
    DS1_SRC="Dataset001_PENGWIN_Anatomical/nnUNetTrainer_LRMirror__nnUNetResEncUNetXLPlans__3d_fullres"
    DS2_ROOT="$RESULTS_ALL2"; DS2_FOLDS="all"; DS2_SRC_CHK="checkpoint_final.pth"
    DS2_SRC="$DS2_DST" ;;
  *) echo "unknown PROFILE=$PROFILE (alldata)"; exit 1 ;;
esac
echo "PROFILE=$PROFILE"

STAGE=$(mktemp -d)
trap 'rm -rf "$STAGE"' EXIT

stage_folds() {   # <results_root> <src_folder> <src_checkpoint> <dst_folder> <folds...>
    local root="$1" src="$2" chk="$3" dst="$4"; shift 4
    [ -d "$root/$src" ] || { echo "ERROR: missing $root/$src"; exit 1; }
    for f in "$@"; do
        [ -f "$root/$src/fold_$f/$chk" ] || { echo "ERROR: missing $root/$src/fold_$f/$chk"; exit 1; }
        mkdir -p "$STAGE/$dst/fold_$f"
        cp "$root/$src/fold_$f/$chk" "$STAGE/$dst/fold_$f/checkpoint_best.pth"
        echo "  staged $dst/fold_$f  (from $chk)"
    done
    cp "$root/$src/plans.json" "$STAGE/$dst/"
    cp "$root/$src/dataset.json" "$STAGE/$dst/"
}

echo "ds001 <- $DS1_ROOT  folds=[$DS1_FOLDS]  $DS1_SRC_CHK"
stage_folds "$DS1_ROOT" "$DS1_SRC" "$DS1_SRC_CHK" "$DS1_DST" $DS1_FOLDS
echo "ds002 <- $DS2_ROOT  folds=[$DS2_FOLDS]  $DS2_SRC_CHK"
stage_folds "$DS2_ROOT" "$DS2_SRC" "$DS2_SRC_CHK" "$DS2_DST" $DS2_FOLDS

echo "scrubbing source_dir / re-labelling trainer_name ..."
python - "$STAGE" <<'PY'
import sys, json, glob, os, torch
stage = sys.argv[1]
for dj in glob.glob(f"{stage}/*/*/dataset.json"):
    d = json.load(open(dj)); d.pop("source_dir", None)
    json.dump(d, open(dj, "w"), indent=2)
for pj in glob.glob(f"{stage}/*/*/plans.json"):
    p = json.load(open(pj))
    if p.get("plans_name") != "nnUNetResEncUNetXLPlans":
        print(f"  re-label plans_name {p.get('plans_name')} -> nnUNetResEncUNetXLPlans")
        p["plans_name"] = "nnUNetResEncUNetXLPlans"
        json.dump(p, open(pj, "w"), indent=2)
BUILTIN = {"Dataset001": "nnUNetTrainerNoMirroring", "Dataset002": "nnUNetTrainer"}
for ck in sorted(glob.glob(f"{stage}/*/*/fold_*/checkpoint_best.pth")):
    c = torch.load(ck, map_location="cpu", weights_only=False)
    ia = c.get("init_args", {})
    dj = ia.get("dataset_json")
    if isinstance(dj, dict):
        dj.pop("source_dir", None)
    top = os.path.relpath(ck, stage).split("/")[0]
    want = next((v for k, v in BUILTIN.items() if top.startswith(k)), None)
    if want and c.get("trainer_name") != want:
        c["trainer_name"] = want
        if "inference_allowed_mirroring_axes" in c:
            c["inference_allowed_mirroring_axes"] = tuple()
    # inference never reads optimizer/scaler state; dropping it roughly halves the
    # tarball (6 full checkpoints would otherwise be ~4.9GB).
    for k in ("optimizer_state", "grad_scaler_state", "logging"):
        c.pop(k, None)
    torch.save(c, ck)
    print(f"  scrubbed {os.path.relpath(ck, stage)}  (epoch {c.get('current_epoch')}, trainer {c.get('trainer_name')})")
PY

# mktemp -d gives the stage root mode 700, and tar preserves it. Grand Challenge
# extracts this tarball and the algorithm runs as a NON-ROOT user, so a 700 root
# directory makes /opt/ml/model unreadable ("Permission denied") at inference time.
chmod -R a+rX "$STAGE"

tar czf "$OUT" -C "$STAGE" .
echo ""
echo "Wrote $OUT  ($(du -h "$OUT" | cut -f1))"
tar tzf "$OUT" | sed 's/^/  /' | head -25
