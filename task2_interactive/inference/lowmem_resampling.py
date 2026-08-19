"""GPU resampling drop-in for nnUNet preprocessing/export.

The stock nnUNet resampler (skimage `resize` -> scipy `ndi.zoom`, order-3 spline on
float64) is the dominant cost AND the OOM source for PENGWIN Task 2 inference:
  * Speed: ~28 s PER fragment on an idle box (much worse under CPU contention, as on
    the Grand Challenge node) — single-threaded scipy on 3-channel ~22 M-voxel volumes.
    With n fragments per case (one forward pass per click), this blows the 12-min limit.
  * Memory: `data.astype(float)` makes a full float64 copy of all channels (the 5-channel
    anatomy => ~4 GB spike) -> OOM on the ~12-16 GB container.

This switches resampling to nnUNet's own `resample_torch_fornnunet` (F.interpolate,
trilinear) on the GPU: ~1 s instead of ~28 s, no CPU float64 spike (transient GPU mem,
freed immediately), and CPU-contention-proof. Output differs from order-3 spline by
~1% of fragment-foreground voxels (trilinear vs cubic) — negligible for the instance
metric, and far better than a timeout/OOM (= 0 score).

install() monkeypatches the module-global `resample_data_or_seg_to_shape` that the
ConfigurationManager dispatches to for data, seg AND probabilities (export).
"""
import torch
import nnunetv2.preprocessing.resampling.default_resampling as _dr
from nnunetv2.preprocessing.resampling.resample_torch import resample_torch_fornnunet

# kwargs the torch resampler accepts (the scipy one is also handed order/order_z, which
# trilinear has no use for — filter them out so the call doesn't blow up).
_TORCH_OK = ("force_separate_z", "separate_z_anisotropy_threshold", "num_threads",
             "memefficient_seg_resampling", "mode", "aniso_axis_mode")


def _device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resample_to_shape_gpu(data, new_shape, current_spacing, new_spacing,
                          is_seg=False, **kw):
    kw2 = {k: v for k, v in kw.items() if k in _TORCH_OK}
    return resample_torch_fornnunet(data, new_shape, current_spacing, new_spacing,
                                    is_seg=is_seg, device=_device(), **kw2)


def install():
    """Route all nnUNet resampling (data/seg/probabilities) through the GPU torch path."""
    _dr.resample_data_or_seg_to_shape = resample_to_shape_gpu
