"""Sliding-window (tiled) inference.

Companion to crop-based training in dataset.py: a model trained on
native-resolution crops needs native-resolution *tiles* at test time too --
resizing the whole 2048x2048 image down for inference would destroy the same
thin barbs the crop-based training was meant to preserve. This splits an
image into overlapping tiles, runs the model on each, and blends overlapping
predictions (simple averaging) back into one full-resolution probability map.
"""
from __future__ import annotations

import numpy as np
import torch
from PIL import Image

from .dataset import IMAGENET_MEAN, IMAGENET_STD


def _tile_starts(size: int, tile: int, stride: int) -> list[int]:
    if size <= tile:
        return [0]
    starts = list(range(0, size - tile + 1, stride))
    if starts[-1] != size - tile:
        starts.append(size - tile)  # ensure the far edge is always covered
    return starts


@torch.no_grad()
def tiled_predict(
    model: torch.nn.Module,
    image_path,
    tile_size: int = 512,
    overlap: int = 64,
    device: torch.device = torch.device("cpu"),
    batch_size: int = 8,
    tta: bool = False,
) -> tuple[np.ndarray, int, int]:
    """Returns (prob_map, height, width): a full-resolution float32
    probability map in [0, 1] plus the original image dimensions."""
    image = Image.open(image_path).convert("L")
    width, height = image.size
    image_np = np.asarray(image, dtype=np.float32) / 255.0
    image_np = (image_np - IMAGENET_MEAN) / IMAGENET_STD

    stride = max(1, tile_size - overlap)
    ys = _tile_starts(height, tile_size, stride)
    xs = _tile_starts(width, tile_size, stride)
    coords = [(y, x) for y in ys for x in xs]

    prob_sum = np.zeros((height, width), dtype=np.float32)
    weight_sum = np.zeros((height, width), dtype=np.float32)

    for batch_start in range(0, len(coords), batch_size):
        batch_coords = coords[batch_start:batch_start + batch_size]
        tiles = []
        for y, x in batch_coords:
            th, tw = min(tile_size, height - y), min(tile_size, width - x)
            tile = np.zeros((tile_size, tile_size), dtype=np.float32)
            tile[:th, :tw] = image_np[y:y + th, x:x + tw]
            tiles.append(tile)
        batch_t = torch.from_numpy(np.stack(tiles)).unsqueeze(1).to(device)

        probs = torch.sigmoid(model(batch_t))
        if tta:
            probs_h = torch.flip(torch.sigmoid(model(torch.flip(batch_t, dims=[3]))), dims=[3])
            probs_v = torch.flip(torch.sigmoid(model(torch.flip(batch_t, dims=[2]))), dims=[2])
            probs = (probs + probs_h + probs_v) / 3
        probs = probs[:, 0].cpu().numpy()

        for (y, x), prob_tile in zip(batch_coords, probs):
            th, tw = min(tile_size, height - y), min(tile_size, width - x)
            prob_sum[y:y + th, x:x + tw] += prob_tile[:th, :tw]
            weight_sum[y:y + th, x:x + tw] += 1.0

    prob_map = prob_sum / np.maximum(weight_sum, 1e-6)
    return prob_map, height, width
