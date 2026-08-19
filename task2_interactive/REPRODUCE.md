# Task 2 — reproduction

The shipped pipeline reuses Task 1's two models (anatomy, contact surface) and adds a per-click
fragment model. Build Task 1's datasets and models first — see `../task1_autoseg/REPRODUCE.md`.

## 1. The per-click fragment dataset

Each (fragment, CT) pair is one training sample: 3 channels (CT, this click as foreground, all
other clicks as background), 2 classes. The organisers'
[Task 2 baseline](https://github.com/Zrrr1997/PENGWIN2026_Task2_InteractiveSeg_Baseline)
provides `create_fragment_nnUNet_dataset.py`; it processes the 340 cases in a single loop, so
`scripts/task2_fragment_parallel.py` reuses its `process_case()` over a process pool.

```bash
REPO_ROOT=$WORK python scripts/task2_fragment_parallel.py --jobs 32
REPO_ROOT=$WORK bash scripts/task2_prepare.sh    # dataset.json, plan_and_preprocess, XL plans
REPO_ROOT=$WORK python scripts/task2_split.py    # same 20 held-out patients as Task 1
```

## 2. Training

```bash
nnUNetv2_train 457 3d_fullres 0 -p nnUNetResEncUNetXLPlans -num_gpus 4
```

Stock `nnUNetTrainer`; no custom trainer is needed, because the click channels already resolve
left from right and mirroring is not used.

## 3. Package and build

```bash
REPO=$WORK bash container/pack_model_flat.sh model.tar.gz
docker build -f container/Dockerfile -t pengwin-task2 .
```

## Runtime behaviour worth knowing

`process.py` is budget-aware. It first produces a complete contact-surface answer, then spends
the remaining wall clock upgrading bones with the per-click model, replacing a bone's labels
only once that bone finishes. If the budget runs out the earlier answer stands, so the output
is always complete and valid.

`UPGRADE_DS457=0` disables the upgrade and reverts to the contact-surface answer alone. Four
further switches (`STRICT_PARTITION`, hole fill, contested-voxel resolution, skip-clean-bones)
default to off and are not part of the evaluated configuration.
