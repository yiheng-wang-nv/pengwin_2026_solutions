# PENGWIN 2026 Task 2 — Team MIAGENT

Click-guided fragment segmentation. The container reads a CT and a fragment-click JSON and
writes a 0–200 instance map.

* `ALGORITHM_DESCRIPTION.md` — the method write-up.
* `REPRODUCE.md` — how the models were trained, from raw data.
* This file — how to build the container and run it on one case.

---

## Build

```bash
bash build.sh                       # -> pengwin-task2:latest
IMAGE=my-image TAG=v1 bash build.sh # or name it yourself
```

The build is self-contained: it starts from `pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime`
and installs `container/requirements.txt` on top. It takes about two minutes and needs no
files outside this directory.

`requirements.txt` deliberately does not list torch. torch 2.5.1 pins
`nvidia-cudnn-cu12==9.1.0.70`, which has since been withdrawn from the cu121 index, so a pip
install of torch at that version no longer resolves; taking torch from the base image avoids
the problem entirely.

## Model weights

The weights are **not** in this repository. Download the model tarball and extract it into a
directory that will be mounted at `/opt/ml/model`:

```bash
mkdir -p model && tar xzf model-task2.tar.gz -C model
```

After extraction `model/` must contain three nnU-Net result trees:

```
model/
  Dataset001_PENGWIN_Anatomical/nnUNetTrainerNoMirroring__nnUNetResEncUNetXLPlans__3d_fullres/fold_all/checkpoint_final.pth
  Dataset002_PENGWIN_Frac/nnUNetTrainer__nnUNetResEncUNetXLPlans__3d_fullres/fold_all/checkpoint_final.pth
  Dataset457_PENGWIN_frag/nnUNetTrainer__nnUNetResEncUNetXLPlans__3d_fullres/fold_0/checkpoint_final.pth
```

(plus each tree's `plans.json` and `dataset.json`). The container discovers `fold_all` or
`fold_0..4` at run time, so no path needs editing.

## Run one case

The interface matches Grand Challenge: a read-only `/input`, a writable `/output`, and the
model at `/opt/ml/model`.

```
input/
  images/pelvic-fracture-ct/<anything>.mha        # the CT
  peripelvic-fragment-clicks.json                 # the clicks
output/
  images/peripelvic-fracture-ct-segmentation/     # written by the container
```

```bash
docker run --rm --gpus '"device=0"' --shm-size=16g \
  -v $PWD/input:/input:ro \
  -v $PWD/output:/output \
  -v $PWD/model:/opt/ml/model:ro \
  --user $(id -u):$(id -g) --network none \
  pengwin-task2:latest
```

The CT and the clicks JSON are found by glob (`**/*.mha`, `**/*clicks*.json`), so the exact
file names do not matter. Output is a compressed `uint8` `.mha` with labels 0–200.

`--shm-size=16g` is only needed outside Grand Challenge: Docker's default `/dev/shm` is 64 MB
and nnU-Net's data workers need more.

`example_clicks.json` in this directory is a working clicks file for training case 001, useful
for a smoke test. Its points are `[z, y, x]` — note that the challenge's JSON stores them in
that order even though several parsers comment them as `[x, y, z]`.

## Expected behaviour

On one case with three bones and six fragments, on a single H200:

```
[UPGRADE] rightHip:1/1 leftHip:2/2 sacrum:3/3
[TIMING] total=131.4s
wrote /output/images/peripelvic-fracture-ct-segmentation/image.mha
```

The pipeline produces a complete answer from the contact-surface model first, then spends the
remaining wall clock re-segmenting bones with the per-click model, replacing each bone's
labels as it finishes. **It is budget-aware**: if time runs out the earlier answer stands, so
the output is always complete. The budget is `CASE_BUDGET_S` (default 600 s) times
`BUDGET_SAFETY` (default 0.90).

## Environment variables

Defaults reproduce the submitted configuration; none needs to be set.

| Variable | Default | Effect |
|---|---|---|
| `CASE_BUDGET_S` | `600` | wall-clock budget per case, in seconds |
| `BUDGET_SAFETY` | `0.90` | fraction of the budget the upgrade may use |
| `UPGRADE_DS457` | `1` | set `0` to skip the per-click upgrade and return the contact-surface answer alone |
| `CSM_KERNEL` | `5` | dilation kernel of the instance decoder |
| `CSM_CCF` | `100` | minimum component size, in voxels |
| `TILE_STEP` | `0.5` | nnU-Net sliding-window step |
| `OUTPUT_SLUG` | `peripelvic-fracture-ct-segmentation` | output sub-directory name |

`STRICT_PARTITION` and three related switches default to off and are **not** part of the
submitted configuration.

## Layout

```
build.sh              build the image
container/            Dockerfile, requirements.txt, process.py (the entry point), model packing
inference/            the pipeline modules process.py imports
scripts/              dataset preparation for the per-click fragment model
example_clicks.json   a clicks file for training case 001
```
