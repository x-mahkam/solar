from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def dice_loss(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    probs = probs.flatten(1)
    targets = targets.flatten(1)
    intersection = (probs * targets).sum(dim=1)
    union = probs.sum(dim=1) + targets.sum(dim=1)
    dice = (2 * intersection + eps) / (union + eps)
    return 1 - dice.mean()


class BCEDiceLoss(nn.Module):
    """BCE + soft-Dice, a standard and robust choice for imbalanced thin
    structures like filaments (mostly background pixels)."""

    def __init__(self, bce_weight: float = 0.5):
        super().__init__()
        self.bce_weight = bce_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, targets)
        dice = dice_loss(logits, targets)
        return self.bce_weight * bce + (1 - self.bce_weight) * dice
