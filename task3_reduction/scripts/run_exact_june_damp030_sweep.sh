#!/usr/bin/env bash
# Score the exact-june milestone checkpoints under the CHAMPION-LINE recipe, not
# inference.py defaults: npoints 5000, max_iters 20, threshold 2mm, update_scale 0.3,
# convergence_metric max_point, seed 42.  See docs/PENGWIN_TASK3_EVAL_PROTOCOL.md --
# the previous milestone sweep used the zero-shot config and its numbers do not
# compare to anything in the champion line.
#
# One checkpoint per GPU, 8 in parallel.  Predictions land in the staging tree, which
# doubles as the prediction root for scoring.
set -euo pipefail

# REPO must point at the WORKING TREE that holds nnUNet_workdir/, weights/ and the
# output_* directories -- that tree is not part of this repository, because it contains
# trained weights and multi-GB intermediates. Set it before running:
#   REPO=/path/to/working/tree bash run_exact_june_damp030_sweep.sh
REPO="${REPO:?set REPO to the working tree containing nnUNet_workdir/ and weights/}"
BASELINE=$REPO/external/PENGWIN2026_Task3_Reduction_Baseline
PY="${PY:-python}"   # the interpreter with the AssemblyNet baseline deps installed
MESH=$REPO/datasets/task3_reduction/extracted/PENGWIN26_task3_clinical_fractures_train/mesh
STAGE=$REPO/output_task3_exact_june_damp030
LOGS=$STAGE/_logs

TAGS=(${TAGS_OVERRIDE:-s42_ep599 s42_ep699 s42_ep799 s42_ep899 s43_ep599 s43_ep699 s43_ep799 s43_ep899})

mkdir -p "$LOGS"

gpu=${GPU_START:-0}
for tag in "${TAGS[@]}"; do
  run_dir=$STAGE/$tag
  mkdir -p "$run_dir"
  # symlink the 170 OBJs; predictions get written next to them
  for case_dir in "$MESH"/*/; do
    case_name=$(basename "$case_dir")
    mkdir -p "$run_dir/$case_name"
    ln -sf "$case_dir/peripelvic-fracture-fragments.obj" \
           "$run_dir/$case_name/peripelvic-fracture-fragments.obj"
  done

  cfg=$BASELINE/configs/test_sweep_$tag.yaml
  cat > "$cfg" <<EOF
# exact-june $tag under the champion-line damping recipe (beta 0.3, max_point, 20 iters)
model_config: "model/AssemblyTransformer_coords"
checkpoint: $REPO/weights/exact_june/$tag.ckpt
input_dir: $run_dir
npoints: 5000
max_iters: 20
convergence_threshold: 2.0
seed: 42
update_scale: 0.3
convergence_metric: max_point
debug_vis: false
output_type: "coords"
experiment_name: "exact_june_${tag}_damp030"
EOF

  echo "[launch] gpu=$gpu $tag -> $run_dir"
  ( cd "$BASELINE" && CUDA_VISIBLE_DEVICES=$gpu "$PY" inference.py --config "$cfg" \
      > "$LOGS/$tag.log" 2>&1 ) &
  gpu=$((gpu + 1))
done

wait
echo "=== inference done ==="
for tag in "${TAGS[@]}"; do
  n=$(find "$STAGE/$tag" -name "reduction-poses-matrices.json" | wc -l)
  echo "$tag: $n/170 predictions"
done
