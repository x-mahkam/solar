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


class TverskyLoss(nn.Module):
    """Generalizes Dice with independent false-positive/false-negative
    weights: Tversky = TP / (TP + alpha*FP + beta*FN). With beta > alpha,
    missed filament pixels (FN) are penalized harder than spurious ones (FP)
    -- appropriate for thin, easily-missed barbs under heavy class imbalance,
    where plain Dice/BCE gives the model little incentive to chase the last
    few pixels of a thin structure."""

    def __init__(self, alpha: float = 0.3, beta: float = 0.7, eps: float = 1e-6):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits).flatten(1)
        targets = targets.flatten(1)
        tp = (probs * targets).sum(dim=1)
        fp = (probs * (1 - targets)).sum(dim=1)
        fn = ((1 - probs) * targets).sum(dim=1)
        tversky = (tp + self.eps) / (tp + self.alpha * fp + self.beta * fn + self.eps)
        return 1 - tversky.mean()


class BCETverskyLoss(nn.Module):
    """BCE + Tversky, mirroring BCEDiceLoss's structure but with Tversky's
    recall-favoring behavior for the region term."""

    def __init__(self, bce_weight: float = 0.3, alpha: float = 0.3, beta: float = 0.7):
        super().__init__()
        self.bce_weight = bce_weight
        self.tversky = TverskyLoss(alpha=alpha, beta=beta)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, targets)
        tversky = self.tversky(logits, targets)
        return self.bce_weight * bce + (1 - self.bce_weight) * tversky


def get_loss(name: str, bce_weight: float = 0.5, tversky_alpha: float = 0.3, tversky_beta: float = 0.7) -> nn.Module:
    if name == "bce_dice":
        return BCEDiceLoss(bce_weight=bce_weight)
    if name == "tversky":
        return BCETverskyLoss(bce_weight=bce_weight, alpha=tversky_alpha, beta=tversky_beta)
    raise ValueError(f"unknown loss {name!r}, expected 'bce_dice' or 'tversky'")
