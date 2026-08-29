"""Validation metrics.

Pixel-level IoU/Dice/precision/recall are cheap and used every epoch during
training. Instance-level AP@IoU / hit-miss / mean-IoU-over-matched-instances
approximate the organizers' evaluation (exact MIoU formula is defined by the
competition and may differ in details -- treat these as a proxy for model
selection, not as the official leaderboard score).
"""
from __future__ import annotations

import numpy as np
import torch


@torch.no_grad()
def pixel_scores(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5, eps: float = 1e-6):
    preds = (torch.sigmoid(logits) > threshold).float()
    targets = targets.float()

    preds_f = preds.flatten(1)
    targets_f = targets.flatten(1)

    intersection = (preds_f * targets_f).sum(dim=1)
    union = ((preds_f + targets_f) > 0).float().sum(dim=1)
    iou = (intersection + eps) / (union + eps)

    dice = (2 * intersection + eps) / (preds_f.sum(dim=1) + targets_f.sum(dim=1) + eps)

    tp = intersection
    fp = (preds_f * (1 - targets_f)).sum(dim=1)
    fn = ((1 - preds_f) * targets_f).sum(dim=1)
    precision = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)

    return {
        "iou": iou.mean().item(),
        "dice": dice.mean().item(),
        "precision": precision.mean().item(),
        "recall": recall.mean().item(),
    }


def connected_components(mask: np.ndarray) -> list[np.ndarray]:
    """4-connected component labeling for a binary mask.

    Returns a list of boolean masks, one per component. Pure-numpy union-find;
    fine for occasional validation use, not meant for large-scale training loops.
    """
    h, w = mask.shape
    labels = -np.ones((h, w), dtype=np.int32)
    ys, xs = np.nonzero(mask)
    parent: dict[int, int] = {}

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    idx_of = -np.ones((h, w), dtype=np.int64)
    for i, (y, x) in enumerate(zip(ys, xs)):
        idx_of[y, x] = i
        parent[i] = i

    for i, (y, x) in enumerate(zip(ys, xs)):
        if y > 0 and mask[y - 1, x]:
            union(i, idx_of[y - 1, x])
        if x > 0 and mask[y, x - 1]:
            union(i, idx_of[y, x - 1])

    roots: dict[int, list[int]] = {}
    for i in range(len(ys)):
        r = find(i)
        roots.setdefault(r, []).append(i)

    components = []
    for members in roots.values():
        comp = np.zeros((h, w), dtype=bool)
        comp[ys[members], xs[members]] = True
        components.append(comp)
    return components


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter) / float(union) if union else 0.0


def instance_metrics(
    pred_mask: np.ndarray, gt_instances: list[np.ndarray], iou_thresholds=(0.3, 0.5, 0.7)
) -> dict:
    """Greedy-matched instance metrics between predicted semantic mask
    (split into connected components) and ground-truth instance masks."""
    pred_instances = connected_components(pred_mask.astype(bool))

    results = {}
    for thr in iou_thresholds:
        matched_gt = set()
        hits = 0
        ious = []
        for pred in pred_instances:
            best_iou, best_j = 0.0, -1
            for j, gt in enumerate(gt_instances):
                if j in matched_gt:
                    continue
                iou = _iou(pred, gt)
                if iou > best_iou:
                    best_iou, best_j = iou, j
            if best_iou >= thr and best_j >= 0:
                matched_gt.add(best_j)
                hits += 1
                ious.append(best_iou)

        n_pred = len(pred_instances)
        n_gt = len(gt_instances)
        precision = hits / n_pred if n_pred else 0.0
        recall = hits / n_gt if n_gt else 0.0
        results[f"AP@{thr}"] = precision
        results[f"hit_rate@{thr}"] = recall
        results[f"miss_rate@{thr}"] = 1 - recall
        results[f"mean_iou_matched@{thr}"] = float(np.mean(ious)) if ious else 0.0
    return results


def panoptic_quality(pred_instances: list[np.ndarray], gt_instances: list[np.ndarray], iou_threshold: float = 0.5) -> float:
    """Panoptic Quality, as defined on the competition's Evaluation page
    (Kirillov et al., 10.1109/CVPR.2019.00963):

        PQ = sum(IoU over matched pairs) / (|TP| + 0.5*|FP| + 0.5*|FN|)

    A predicted/ground-truth pair is a match (TP) if IoU > iou_threshold;
    at threshold 0.5 greedy matching is equivalent to optimal matching,
    since no ground-truth instance can have IoU > 0.5 with two different
    predictions simultaneously.
    """
    matched_gt: set[int] = set()
    tp_ious = []
    for pred in pred_instances:
        best_iou, best_j = 0.0, -1
        for j, gt in enumerate(gt_instances):
            if j in matched_gt:
                continue
            iou = _iou(pred, gt)
            if iou > best_iou:
                best_iou, best_j = iou, j
        if best_iou > iou_threshold and best_j >= 0:
            matched_gt.add(best_j)
            tp_ious.append(best_iou)

    tp = len(tp_ious)
    fp = len(pred_instances) - tp
    fn = len(gt_instances) - tp
    denom = tp + 0.5 * fp + 0.5 * fn
    return float(sum(tp_ious) / denom) if denom > 0 else 1.0
