"""Custom trainer: mirror ONLY the left/right axis WITH a leftHip<->rightHip
label swap. For PENGWIN Task1 anatomical (ds001).

Rationale:
- Pelvis left/right hips are distinct classes. Plain mirror on the L/R axis
  swaps them spatially WITHOUT swapping labels -> that broke rightHip (why we
  used NoMirroring). Here we swap labels 2<->3 whenever the L/R axis is flipped,
  making the flip a VALID bilateral-symmetry augmentation.
- Axes 0,1 are NOT mirrored: nnUNetTrainer_onlyMirror01 (which mirrored 0,1)
  performed clearly worse than NoMirroring on this task, so we restrict to the
  one anatomically-meaningful axis.

Axis convention: ds001 plans transpose_forward=[0,1,2]; preprocessed data is
(z, y, x) so axis 2 = x = left/right. Verified: onlyMirror01 (axes 0,1) kept
rightHip non-zero, consistent with axis 2 being L/R.

inference_allowed_mirroring_axes is set to None: nnUNet's built-in mirror TTA
does NOT swap label channels, so it would be wrong for the L/R axis. Use the
external channel-swapping TTA in the e2e pipeline instead.
"""
import torch
from batchgeneratorsv2.transforms.spatial.mirroring import MirrorTransform
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


class LabelSwapMirrorTransform(MirrorTransform):
    """MirrorTransform that also swaps a pair of seg labels when the L/R axis flips."""

    def __init__(self, allowed_axes, lr_axis: int = 2, swap=(2, 3)):
        super().__init__(allowed_axes)
        self.lr_axis = lr_axis
        self.swap = swap

    def _apply_to_segmentation(self, segmentation: torch.Tensor, **params) -> torch.Tensor:
        seg = super()._apply_to_segmentation(segmentation, **params)
        if self.lr_axis in params.get("axes", []):
            a, b = self.swap
            seg = seg.clone()
            ma, mb = (seg == a), (seg == b)
            seg[ma] = b
            seg[mb] = a
        return seg


class nnUNetTrainer_LRMirror(nnUNetTrainer):
    LR_AXIS = 2          # x-axis = left/right
    SWAP = (2, 3)        # leftHip(2) <-> rightHip(3)

    def configure_rotation_dummyDA_mirroring_and_inital_patch_size(self):
        rot, dummy, patch, _ = super().configure_rotation_dummyDA_mirroring_and_inital_patch_size()
        mirror_axes = (self.LR_AXIS,)          # train-time mirror: L/R only
        self.inference_allowed_mirroring_axes = None  # built-in TTA can't swap labels; use external TTA
        return rot, dummy, patch, mirror_axes

    @staticmethod
    def get_training_transforms(patch_size, rotation_for_DA, deep_supervision_scales,
                                mirror_axes, do_dummy_2d_data_aug, use_mask_for_norm=None,
                                is_cascaded=False, foreground_labels=None, regions=None,
                                ignore_label=None):
        tr = nnUNetTrainer.get_training_transforms(
            patch_size, rotation_for_DA, deep_supervision_scales, mirror_axes,
            do_dummy_2d_data_aug, use_mask_for_norm, is_cascaded,
            foreground_labels, regions, ignore_label)
        for i, t in enumerate(tr.transforms):
            if type(t) is MirrorTransform:
                tr.transforms[i] = LabelSwapMirrorTransform(
                    t.allowed_axes, lr_axis=nnUNetTrainer_LRMirror.LR_AXIS,
                    swap=nnUNetTrainer_LRMirror.SWAP)
        return tr
