# Task 1 — reproduction

Two nnU-Net v2 models and a deterministic instance decoder. `$WORK` below is a working tree
holding the raw data, `nnUNet_raw/`, `nnUNet_preprocessed/` and `nnUNet_results/`; it is not
part of this repository.

## 1. Datasets

The two nnU-Net datasets are built with the organisers' own preprocessing scripts from
[the Task 1 baseline](https://github.com/YzzLiu/PENGWIN2026_Task1_AutoSeg_Baseline):

* `preprocessing/gen_nnunet_dataset.py` → `Dataset001_PENGWIN_Anatomical` (5-class anatomy) and
  `Dataset002_PENGWIN_Frac` (images)
* `preprocessing/gen_CSM_dataset.py` → the 3-class contact-surface labels of `Dataset002`

`scripts/02_preprocess.sh` runs both, then `nnUNetv2_plan_and_preprocess`.

```bash
REPO_ROOT=$WORK bash scripts/02_preprocess.sh
REPO_ROOT=$WORK bash scripts/02c_replan_resenc.sh     # ResEnc XL plans
REPO_ROOT=$WORK bash scripts/02b_patient_splits.sh    # patient-level 5-fold + 20 held out
```

The split is patient-level: a patient's bones never straddle folds, and 20 patients are held
out of the cross-validation entirely.

## 2. Training

The anatomy model uses the custom trainer in `scripts/nnUNetTrainer_LRMirror.py`, which mirrors
only the left/right axis and swaps the left-hip/right-hip labels with it. Copy it into
nnU-Net's `nnunetv2/training/nnUNetTrainer/` (or put it on `PYTHONPATH`) before training.

```bash
# five-fold, used for out-of-fold validation
REPO_ROOT=$WORK GPUS="0 1" bash scripts/04_train_xl.sh

# submitted weights: all 340 patients
nnUNetv2_train 001 3d_fullres all -tr nnUNetTrainer_LRMirror -p nnUNetResEncUNetXLPlans -num_gpus 4
nnUNetv2_train 002 3d_fullres all -p nnUNetResEncUNetXLPlans -num_gpus 4
```

`-num_gpus 4` is the practical maximum: the plans batch size is 4, so four GPUs is one sample
each.

## 3. Package and build

```bash
REPO=$WORK PROFILE=alldata bash container/pack_model.sh model.tar.gz
bash container/build.sh
```

`pack_model.sh` strips optimiser state, relabels the anatomy checkpoint's `trainer_name` to
`nnUNetTrainerNoMirroring` (the custom trainer class is absent inside the container and nnU-Net
rebuilds the network from that name; the architecture is identical and mirror TTA stays off),
and makes the tar world-readable — Grand Challenge runs the algorithm as a non-root user and a
0700 root directory makes `/opt/ml/model` unreadable.

`KERNEL_SIZE` and `CCF_THRESHOLD` are environment-overridable at run time, so the decoder
operating point can be changed without a rebuild.
