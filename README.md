# PENGWIN 2026 — Team MIAGENT solutions

Code for the three tasks of the MICCAI 2026 PENGWIN challenge, as submitted.

> **This repository is private during the challenge and will be made public once the
> results are announced.**

| Folder | Task | Approach |
|---|---|---|
| `task1_autoseg/` | Task 1 — automatic fragment segmentation | two-stage nnU-Net (anatomy → contact-surface) with a label-swapping L/R mirror augmentation, a geometry gate and a connected-component instance decoder |
| `task2_interactive/` | Task 2 — interactive fragment segmentation | click-gated anatomy + contact-surface answer, progressively upgraded per click by a merge-free fragment model within the wall-clock budget. Has its own `README.md` with self-contained build and run instructions. |
| `task3_reduction/` | Task 3 — fracture reduction | damped AssemblyNet + a geometry-conditioned residual pose head + a five-member SE(3) ensemble |

Each task folder contains two documents and the code they refer to:

* `ALGORITHM_DESCRIPTION.md` — the write-up submitted to the challenge forum.
* `REPRODUCE.md` — the exact command chain from raw data to a built container.

Experimental branches, ablations and abandoned ideas are deliberately not included.

Trained weights are distributed separately as the model tarballs each challenge task expects at
`/opt/ml/model`; they are not stored in this repository.

## Layout

```
task1_autoseg/
  container/     Dockerfile, inference entry point, instance decoder, model packing
  scripts/       dataset preparation, the custom L/R-mirror trainer, training
task2_interactive/
  container/     Dockerfile, process.py, model packing
  inference/     the pipeline modules the container imports
  scripts/       per-click fragment dataset creation, preprocessing, splits
task3_reduction/
  container/     Dockerfile, process.py, model packing
  scripts/       pose ensemble math, residual head, inference and reproduction scripts
```
