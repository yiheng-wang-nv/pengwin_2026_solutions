#!/usr/bin/env python3
"""Small geometry-conditioned residual pose head for PENGWIN Task 3.

The head predicts a *fragment-centric* correction: a displacement of the
predicted fragment centroid and a left-multiplied SO(3) rotation.  It never
regresses the translation column of a 4x4 matrix, because that quantity is
coupled to rotations about the CT origin.
"""

from __future__ import annotations

import math

import torch
from torch import nn


def axis_angle_to_matrix(rotvec: torch.Tensor) -> torch.Tensor:
    """Differentiable Rodrigues formula, including the small-angle limit."""
    theta2 = (rotvec * rotvec).sum(dim=-1, keepdim=True)
    theta = torch.sqrt(theta2.clamp_min(1e-16))
    x, y, z = rotvec.unbind(dim=-1)
    zero = torch.zeros_like(x)
    skew = torch.stack(
        (zero, -z, y, z, zero, -x, -y, x, zero), dim=-1
    ).reshape(*rotvec.shape[:-1], 3, 3)

    # sin(theta)/theta and (1-cos(theta))/theta^2 with stable Taylor limits.
    small = theta2 < 1e-8
    a = torch.where(
        small,
        1.0 - theta2 / 6.0 + theta2 * theta2 / 120.0,
        torch.sin(theta) / theta,
    )
    b = torch.where(
        small,
        0.5 - theta2 / 24.0 + theta2 * theta2 / 720.0,
        (1.0 - torch.cos(theta)) / theta2.clamp_min(1e-16),
    )
    eye = torch.eye(3, device=rotvec.device, dtype=rotvec.dtype)
    eye = eye.expand(*rotvec.shape[:-1], 3, 3)
    return eye + a[..., None] * skew + b[..., None] * (skew @ skew)


class PointEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_dim, 64, 1),
            nn.GELU(),
            nn.Conv1d(64, 96, 1),
            nn.GELU(),
            nn.Conv1d(96, hidden, 1),
            nn.GELU(),
        )

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        # [B,N,C] -> permutation-invariant [B,H]
        return self.net(points.transpose(1, 2)).amax(dim=-1)


class ResidualPointNet(nn.Module):
    """Shared fragment/contact geometry encoder with an identity-safe head."""

    def __init__(
        self,
        global_dim: int = 14,
        trans_limit_mm: float = 20.0,
        rot_limit_deg: float = 25.0,
    ):
        super().__init__()
        self.global_dim = int(global_dim)
        self.trans_limit_mm = float(trans_limit_mm)
        self.rot_limit_rad = math.radians(float(rot_limit_deg))
        self.fragment_encoder = PointEncoder(6)
        self.context_encoder = PointEncoder(7)
        self.head = nn.Sequential(
            nn.Linear(128 + 128 + self.global_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, 6),
        )
        # Initial model is exactly the identity correction. This makes both
        # failed optimisation and conservative regularisation safe by default.
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(
        self,
        fragment: torch.Tensor,
        context: torch.Tensor,
        global_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = torch.cat(
            (
                self.fragment_encoder(fragment),
                self.context_encoder(context),
                global_features,
            ),
            dim=-1,
        )
        raw = self.head(encoded)
        delta_centroid = torch.tanh(raw[:, :3]) * self.trans_limit_mm
        delta_rotvec = torch.tanh(raw[:, 3:]) * self.rot_limit_rad
        return delta_centroid, delta_rotvec


def correct_centered_points(
    centered_points_mm: torch.Tensor,
    delta_centroid_mm: torch.Tensor,
    delta_rotvec: torch.Tensor,
) -> torch.Tensor:
    """Apply a residual about the current predicted fragment centroid."""
    rotation = axis_angle_to_matrix(delta_rotvec)
    rotated = torch.einsum("bij,bnj->bni", rotation, centered_points_mm)
    return rotated + delta_centroid_mm[:, None, :]
