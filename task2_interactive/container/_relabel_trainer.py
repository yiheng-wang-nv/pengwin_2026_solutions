"""Relabel checkpoint trainer_name to a class that exists inside the GC container.

nnU-Net rebuilds the network from checkpoint["trainer_name"]; the container has none of our
custom trainers (nnUNetTrainer_LRMirror, nnUNetTrainer_4000ep, ...), so loading one of those
checkpoints fails with "Unable to locate trainer class". The architecture does not depend on
the trainer, and the only behavioural difference of NoMirroring is mirror TTA, which
process_flat.py never enables.
"""
import glob
import os
import sys

import torch

BUILTIN = {"Dataset001": "nnUNetTrainerNoMirroring", "Dataset002": "nnUNetTrainer",
           "Dataset456": "nnUNetTrainerNoMirroring", "Dataset457": "nnUNetTrainer"}
stage = sys.argv[1]
for ck in sorted(glob.glob(f"{stage}/*/*/fold_*/checkpoint_*.pth")):
    top = os.path.relpath(ck, stage).split("/")[0]
    want = next((v for k, v in BUILTIN.items() if top.startswith(k)), None)
    c = torch.load(ck, map_location="cpu", weights_only=False)
    if want and c.get("trainer_name") != want:
        print(f"  re-label {os.path.relpath(ck, stage)}: {c['trainer_name']} -> {want}")
        c["trainer_name"] = want
        if "inference_allowed_mirroring_axes" in c:
            c["inference_allowed_mirroring_axes"] = tuple()
        torch.save(c, ck)
