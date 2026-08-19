#!/usr/bin/env bash
# One command: turn a fresh AssemblyNet checkpoint into a scored, weight-selected, packaged
# Task 3 submission candidate.
#
# Written to be run against the clock. The submission budget is one shot and the ranking takes
# the LAST submission, so this must not require improvising any step at 3am: every stage below
# is the same code that produced the numbers already on record, in the same order, with the
# same evaluator settings (5,000/1,000, eval seed 42) and the same damping recipe.
#
#   1. damping-recipe inference on the 170 clinical cases        (~5 min, 1 GPU)
#   2. score the raw base against the champion base              (~10 min, CPU)
#   3. fit its own 5-fold residual, warm-started from sim        (~20 min, 1 GPU)
#      -- a residual head is tied to the base it was cached from, so it cannot be reused
#   4. out-of-fold residual predictions + score                  (~10 min)
#   5. centroid-parameterisation ensemble with the champion line over a weight grid,
#      selected on cases 001-183 and confirmed on the locked 184-200   (~40 min)
#
# Packaging is deliberately NOT automated: docker_task3_dual/package_model.py takes the two
# (base, residual-dir) pairs and which one wins is a judgement call this script only informs.
#
# Usage:
#   bash scripts/task3/promote_new_base.sh <ckpt-path> <tag> [gpu]
set -euo pipefail

# REPO must point at the WORKING TREE that holds nnUNet_workdir/, weights/ and the
# output_* directories -- that tree is not part of this repository, because it contains
# trained weights and multi-GB intermediates. Set it before running:
#   REPO=/path/to/working/tree bash promote_new_base.sh
REPO="${REPO:?set REPO to the working tree containing nnUNet_workdir/ and weights/}"
PY="${PY:-python}"   # the interpreter with the AssemblyNet baseline deps installed
MESH=$REPO/datasets/task3_reduction/extracted/PENGWIN26_task3_clinical_fractures_train/mesh
CHAMP_BASE=$REPO/output_task3_damp030_point20_seed42
CHAMP_RESID=$REPO/output_task3_residual/oof_predictions_sim_actual_pretrain
SPLIT=$REPO/output_task3_residual/folds.json
PRETRAIN=$REPO/output_task3_residual/sim_actual_pretrain/best.pt

CKPT="${1:?usage: promote_new_base.sh <ckpt-path> <tag> [gpu]}"
TAG="${2:?missing tag}"
GPU="${3:-0}"
OUT=$REPO/output_task3_promote/$TAG
mkdir -p "$OUT"
exec > >(tee -a "$OUT/promote.log") 2>&1
echo "=== promote $TAG from $CKPT (gpu $GPU) ==="; date

# ---- 1. base inference under the champion damping recipe -------------------------------
cp -n "$CKPT" "$REPO/weights/exact_june/$TAG.ckpt" 2>/dev/null || true
if [ ! -f "$REPO/output_task3_exact_june_damp030/$TAG/001/reduction-poses-matrices.json" ]; then
    GPU_START=$GPU TAGS_OVERRIDE="$TAG" bash "$REPO/scripts/task3/run_exact_june_damp030_sweep.sh"
fi
BASE_PRED=$REPO/output_task3_exact_june_damp030/$TAG

# ---- 2. score the raw base -------------------------------------------------------------
echo; echo "--- [2] raw base vs champion base ---"
$PY "$REPO/scripts/task3/residual_cv.py" score --data-dir "$MESH" \
    --base-pred-root "$CHAMP_BASE" --residual-pred-root "$BASE_PRED" \
    --out-csv "$OUT/score_base.csv" | tail -7

# ---- 3+4. its own 5-fold residual, then out-of-fold predictions -------------------------
echo; echo "--- [3] 5-fold residual on this base ---"; date
$PY "$REPO/scripts/task3/residual_cv.py" cache \
    --data-dir "$MESH" --pred-root "$BASE_PRED" --cache-dir "$OUT/cache"
$PY "$REPO/scripts/task3/residual_cv.py" train-cv \
    --cache-dir "$OUT/cache" --split-file "$SPLIT" --weights-dir "$OUT/weights" \
    --pretrained "$PRETRAIN" --device cuda
$PY "$REPO/scripts/task3/residual_cv.py" predict-oof \
    --cache-dir "$OUT/cache" --split-file "$SPLIT" --weights-dir "$OUT/weights" \
    --out-dir "$OUT/oof" --device cuda
echo; echo "--- [4] base + its residual ---"
$PY "$REPO/scripts/task3/residual_cv.py" score --data-dir "$MESH" \
    --base-pred-root "$BASE_PRED" --residual-pred-root "$OUT/oof" \
    --out-csv "$OUT/score_resid.csv" | tail -7

# ---- 5. ensemble with the champion line, weight grid ------------------------------------
echo; echo "--- [5] centroid ensemble vs the champion residual line ---"; date
for w in "0.85 0.15:0.65 0.35" "0.75 0.25:0.55 0.45" "0.65 0.35:0.55 0.45" "0.55 0.45:0.45 0.55"; do
    wr=${w%%:*}; wt=${w##*:}
    name="ens_r${wr// /_}_t${wt// /_}"
    $PY "$REPO/scripts/task3/ensemble_poses_centroid.py" \
        --roots "$CHAMP_RESID" "$OUT/oof" --obj-dir "$MESH" --out "$OUT/$name" \
        --w-rot $wr --w-trans $wt > /dev/null
    $PY "$REPO/scripts/task3/residual_cv.py" score --data-dir "$MESH" \
        --base-pred-root "$CHAMP_RESID" --residual-pred-root "$OUT/$name" \
        --out-csv "$OUT/score_$name.csv" > "$OUT/score_$name.log" 2>&1 &
done
wait

echo; echo "=== selection table (select on 001-183, confirm on locked 184-200) ==="
$PY - "$OUT" <<'PY'
import csv, glob, os, statistics as st, sys
out = sys.argv[1]
def part(rows, lo, hi): return [r for r in rows if lo <= int(r["case"]) <= hi]
print(f"{'weights':28s} | {'--- select 001-183 ---':^26s} | {'--- LOCKED 184-200 ---':^26s}")
print(f"{'':28s} | {'CD':>8s} {'PA%':>7s} {'Rot':>8s} | {'CD':>8s} {'PA%':>7s} {'Rot':>8s}")
first = True
for f in sorted(glob.glob(f"{out}/score_ens_*.csv")):
    rows = list(csv.DictReader(open(f)))
    if len(rows) < 170:
        continue
    if first:
        for lbl, sub in (("champion+residual", None),):
            for name, lo, hi in (("001-183", 0, 183), ("184-200", 184, 200)):
                p = part(rows, lo, hi)
                g = lambda k: st.mean(float(x[k]) for x in p)
                print(f"  base {lbl:22s} {name}: CD {g('base_cd_mm'):.4f}  PA {100*g('base_pa'):.2f}  Rot {g('base_rot_deg'):.3f}")
        first = False
    tag = os.path.basename(f)[len("score_ens_"):-4]
    o = []
    for lo, hi in ((0, 183), (184, 200)):
        p = part(rows, lo, hi)
        g = lambda k: st.mean(float(x[k]) for x in p)
        o.append((g("residual_cd_mm"), 100 * g("residual_pa"), g("residual_rot_deg")))
    print(f"{tag:28s} | {o[0][0]:8.4f} {o[0][1]:7.2f} {o[0][2]:8.3f} | {o[1][0]:8.4f} {o[1][1]:7.2f} {o[1][2]:8.3f}")
PY
echo; echo "=== done ==="; date
echo "artifacts under $OUT"
echo "to package:  python docker_task3_dual/package_model.py \\"
echo "    --assemblynet-a weights/assemble_4gpu_epoch984_simpretrain.ckpt \\"
echo "    --residual-dir-a output_task3_residual/weights_sim_actual_pretrain \\"
echo "    --assemblynet-b $CKPT --residual-dir-b $OUT/weights \\"
echo "    --out docker_task3_dual/model-task3-dual-<tag>.tar.gz"
