#!/usr/bin/env bash
# Build the model tarball for the FLAT Task 2 pipeline (process_flat.py).
# Extracts to /opt/ml/model at runtime (= nnUNet_results).
#
#   Dataset456_PENGWIN/nnUNetTrainerNoMirroring__nnUNetResEncUNetXLPlans__3d_fullres/
#     fold_0/checkpoint_final.pth   anatomy, click-conditioned  (unchanged from the
#                                   currently-submitted Task 2 pack)
#   Dataset002_PENGWIN_Frac/nnUNetTrainer__nnUNetResEncUNetXLPlans__3d_fullres/
#     fold_0/checkpoint_best.pth    contact-surface model, borrowed from Task 1
#
# ds457 is deliberately NOT packed: the flat pipeline never calls it, and dropping it
# takes ~800 MB off the upload.
#
# PROFILE picks the ds002 weights (process_flat.py discovers folds and checkpoint name at
# runtime, so switching packs does not require rebuilding the image):
#   champion : Task 1's original 1st-place ds002, single fold, checkpoint_best. Trained on
#              320 patients holding out the 20 the flat pipeline was validated on, so its
#              measured dice 0.8962 / insF1 0.9508 is an honest generalisation number.
#   alldata  : ds002 retrained on ALL 340 patients. Must use checkpoint_final -- with
#              fold=all nnU-Net validates on its own training data so checkpoint_best is
#              meaningless. Expected to be better on the hidden test set (more data, more
#              epochs) but there is no local data left to prove it on, which is the whole
#              tradeoff.
set -euo pipefail
# REPO must point at the WORKING TREE that holds nnUNet_workdir/, weights/ and the
# output_* directories -- that tree is not part of this repository, because it contains
# trained weights and multi-GB intermediates. Set it before running:
#   REPO=/path/to/working/tree bash pack_model_flat.sh
REPO="${REPO:?set REPO to the working tree containing nnUNet_workdir/ and weights/}"
PROFILE="${PROFILE:-champion}"
T2PACK="${T2PACK:-/tmp/t2model}"                                 # existing Task 2 pack
RESULTS_OLD="${RESULTS_OLD:-$REPO/nnUNet_workdir/results}"
RESULTS_ALL2="${RESULTS_ALL2:-$REPO/nnUNet_workdir/results_alldata/002_alldata}"
OUT="${1:-$REPO/docker_task2/model-task2-flat.tar.gz}"

DS456="Dataset456_PENGWIN/nnUNetTrainerNoMirroring__nnUNetResEncUNetXLPlans__3d_fullres"
DS002="Dataset002_PENGWIN_Frac/nnUNetTrainer__nnUNetResEncUNetXLPlans__3d_fullres"

case "$PROFILE" in
  champion) DS2_ROOT="$RESULTS_OLD";  DS2_FOLDS="0";   DS2_CHK="checkpoint_best.pth" ;;
  alldata)  DS2_ROOT="$RESULTS_ALL2"; DS2_FOLDS="all"; DS2_CHK="checkpoint_final.pth" ;;
  *) echo "unknown PROFILE=$PROFILE (champion|alldata)"; exit 1 ;;
esac
echo "PROFILE=$PROFILE  ds002 <- $DS2_ROOT  folds=$DS2_FOLDS  $DS2_CHK"

STAGE=$(mktemp -d)
trap 'rm -rf "$STAGE"' EXIT

# --- anatomy ------------------------------------------------------------------------
# ANATOMY=001 packs Task 1's 1-channel ds001; ANATOMY=456 packs Task 2's 5-channel ds456.
# 001 is the default because ds456's anatomy pass does not fit Grand Challenge's 16 GiB
# DRAM cap: nnU-Net resamples and float64-pads EVERY channel, so 5 channels cost ~5x, and
# under a 15 GiB cap case 048 was OOM-killed in that stage while 389 survived on 306 MiB.
# process_flat.py picks whichever the pack contains, so the image needs no rebuild.
# 001all packs the all-data ds001 (fold_all, 1000ep) instead of fold_0. Note that the 20 local
# evaluation cases sit inside the all-data training set, so a local screen of an all-data model
# can only reject it, never confirm it.
ANATOMY="${ANATOMY:-001}"
if [ "$ANATOMY" = "001all" ]; then
  DS001="Dataset001_PENGWIN_Anatomical/nnUNetTrainerNoMirroring__nnUNetResEncUNetXLPlans__3d_fullres"
  SRC1="${RESULTS_ALL1:-$REPO/nnUNet_workdir/results_alldata/001_alldata}/Dataset001_PENGWIN_Anatomical/nnUNetTrainer_LRMirror__nnUNetResEncUNetXLPlans__3d_fullres"
  [ -d "$SRC1" ] || { echo "missing $SRC1"; exit 1; }
  mkdir -p "$STAGE/$DS001/fold_all"
  cp "$SRC1"/plans.json "$SRC1"/dataset.json "$STAGE/$DS001/"
  cp "$SRC1"/fold_all/checkpoint_final.pth "$STAGE/$DS001/fold_all/"
  echo "  ds001 fold_all <- checkpoint_final.pth (all-data 1000ep)"
elif [ "$ANATOMY" = "456" ]; then
  [ -d "$T2PACK/$DS456" ] || { echo "missing $T2PACK/$DS456"; exit 1; }
  mkdir -p "$STAGE/$DS456"
  cp "$T2PACK/$DS456"/plans.json "$T2PACK/$DS456"/dataset.json "$STAGE/$DS456/"
  for f in $(ls -d "$T2PACK/$DS456"/fold_* 2>/dev/null); do
    n=$(basename "$f"); mkdir -p "$STAGE/$DS456/$n"
    src=$(ls "$f"/checkpoint_*.pth | head -1)
    cp "$src" "$STAGE/$DS456/$n/$(basename "$src")"
    echo "  ds456 $n <- $(basename "$src")"
  done
else
  DS001="Dataset001_PENGWIN_Anatomical/nnUNetTrainerNoMirroring__nnUNetResEncUNetXLPlans__3d_fullres"
  [ -d "$RESULTS_OLD/$DS001" ] || { echo "missing $RESULTS_OLD/$DS001"; exit 1; }
  mkdir -p "$STAGE/$DS001/fold_0"
  cp "$RESULTS_OLD/$DS001"/plans.json "$RESULTS_OLD/$DS001"/dataset.json "$STAGE/$DS001/"
  cp "$RESULTS_OLD/$DS001"/fold_0/checkpoint_best.pth "$STAGE/$DS001/fold_0/"
  echo "  ds001 fold_0 <- checkpoint_best.pth"
fi

# --- ds002 CSM: borrowed from Task 1 -------------------------------------------------
[ -d "$DS2_ROOT/$DS002" ] || { echo "missing $DS2_ROOT/$DS002"; exit 1; }
mkdir -p "$STAGE/$DS002"
cp "$DS2_ROOT/$DS002"/plans.json "$DS2_ROOT/$DS002"/dataset.json "$STAGE/$DS002/"
for k in $DS2_FOLDS; do
  src="$DS2_ROOT/$DS002/fold_$k/$DS2_CHK"
  [ -f "$src" ] || { echo "missing $src"; exit 1; }
  mkdir -p "$STAGE/$DS002/fold_$k"
  # Written under the name process_flat.py probes for first; it accepts either.
  cp "$src" "$STAGE/$DS002/fold_$k/$DS2_CHK"
  echo "  ds002 fold_$k <- $DS2_CHK ($(du -h "$src" | cut -f1))"
done

# --- ds457 per-click fragment model, for the progressive upgrade ----------------------
# Optional (WITH_DS457=0 drops it) because it costs ~800 MB of upload and process_flat.py
# runs without it. With it, bones are re-segmented per click for as long as the wall clock
# allows, which is what closes the merge gap: on the 18 held-out cases merge goes
# 0.3889 -> 0.0370 (identical to running ds457 on everything), recall 0.8801 -> 0.9657 and
# dice 0.8676 -> 0.9124, against a split_error cost of 0.0000 -> 0.0185.
WITH_DS457="${WITH_DS457:-1}"
if [ "$WITH_DS457" = "1" ]; then
  DS457="Dataset457_PENGWIN_frag/nnUNetTrainer__nnUNetResEncUNetXLPlans__3d_fullres"
  SRC457="${T2PACK}/${DS457}"
  if [ -d "$SRC457" ]; then
    mkdir -p "$STAGE/$DS457/fold_0"
    cp "$SRC457"/plans.json "$SRC457"/dataset.json "$STAGE/$DS457/"
    src=$(ls "$SRC457"/fold_0/checkpoint_*.pth | head -1)
    cp "$src" "$STAGE/$DS457/fold_0/$(basename "$src")"
    echo "  ds457 fold_0 <- $(basename "$src") ($(du -h "$src" | cut -f1))"
  else
    echo "  ds457 not found under $T2PACK -- upgrade disabled at run time"
  fi
fi

# The container ships no custom trainer classes and nnU-Net rebuilds the network from
# checkpoint["trainer_name"], so an LRMirror/4000ep checkpoint must be relabelled to a
# built-in name or it dies with "Unable to locate trainer class". Architecture is unchanged;
# NoMirroring only disables mirror TTA, which this pipeline does not use.
python "$REPO/docker_task2/_relabel_trainer.py" "$STAGE"

tar -C "$STAGE" -czf "$OUT" .
echo "wrote $OUT  ($(du -h "$OUT" | cut -f1))"
tar -tzf "$OUT" | grep -E "checkpoint|plans.json" | sed 's/^/    /'
