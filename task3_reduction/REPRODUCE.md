# Task 3 — reproduction

Three stages: a simulation-trained AssemblyNet backbone, a clinical residual pose head fitted
out-of-fold on that backbone's own errors, and a five-member ensemble. `$BASE` below is a
checkout of the
[Task 3 baseline](https://github.com/Sutuk/PENGWIN2026_Task3_Reduction_Baseline).

## 1. Backbone training (simulated data only)

No clinical data is used here; all clinical adaptation happens in stage 2.

```bash
cd $BASE
python train.py trainer.devices=4 experiment_name=<run_name> seed=<seed> \
    data.batch_size=10 trainer.accumulate_grad_batches=1 \
    model.optimizer.lr=1e-4 trainer.max_epochs=1000
```

Effective batch size is what matters: `devices x batch_size = 40`. Larger effective batches were
tried and are worse, because they cut the number of optimiser updates per epoch. Repeat with
different seeds to produce ensemble members — members from different runs disagree by 2.8–3.0°
per fragment, against 1.4–2.3° for two checkpoints of the same run, so separate runs are what
actually adds diversity.

## 2. Base inference under the damping recipe

```bash
REPO=$WORK TAGS_OVERRIDE="<tag>" bash scripts/run_exact_june_damp030_sweep.sh
```

The recipe is 5,000 bone / 1,000 fragment points, at most 20 iterations, update damping 0.3 and
a 2 mm `max_point` convergence threshold. Every number in the write-up assumes it; base
predictions produced with different settings are not comparable.

## 3. Residual head, five-fold, out-of-fold

```bash
REPO=$WORK bash scripts/promote_new_base.sh <ckpt> <tag> <gpu>
```

This caches the base predictions, fits one head per fold, writes out-of-fold predictions and
scores them. A residual head is tied to the base it was cached from and cannot be reused across
backbones. Inside each fold an inner validation split chooses the number of epochs, then the
head is refit on the whole outer-training split for exactly that many epochs, so the outer fold
is never used for selection.

`scripts/residual_cv.py` also exposes the individual steps (`cache`, `train-cv`, `predict-oof`,
`score`).

## 4. Ensemble

```bash
python scripts/ensemble_poses_centroid.py \
    --roots <oof_1> ... <oof_5> --obj-dir <meshes> --out <out_dir> \
    --w-rot 0.60 0.05 0.1167 0.1167 0.1166 \
    --w-trans 0.28 0.28 0.1467 0.1467 0.1466
```

Rotation and translation carry separate weights because the two are physically independent in
this parameterisation. Do not average in se(3) or on the raw 4x4 translation column: the
translation part there is a function of the rotation, and blending it across members with
different rotations degrades TRE from 3.14 mm to 8.7 mm.

## 5. Package and build

```bash
python container/package_model.py \
    --member "champ:<ckpt>:<residual_dir>" \
    --member "b:<ckpt>:<residual_dir>" \
    ... \
    --w-rot 0.60 0.05 0.1167 0.1167 0.1166 \
    --w-trans 0.28 0.28 0.1467 0.1467 0.1466 \
    --out model.tar.gz
bash container/build.sh
```

Member order and weights are written into the tar's `manifest.json`, and `process.py` reads
them from there, so changing the ensemble means replacing the model tar only — the image does
not need rebuilding. `W_ROT` / `W_TRANS` / `DEVICE` override the manifest at run time.
