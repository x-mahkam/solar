"""Run-length encoding helpers matching Kaggle's usual column-major RLE format
(as used in Airbus/HuBMAP/Sartorius-style segmentation competitions):
1-indexed pixel position, in column-major (Fortran) order, as
"start length start length ...".
"""
from __future__ import annotations

import numpy as np


def mask_to_rle(mask: np.ndarray) -> str:
    """Encode a binary (H, W) mask as a Kaggle-style RLE string."""
    pixels = mask.T.flatten()  # column-major order
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)


def rle_to_mask(rle: str, shape: tuple[int, int]) -> np.ndarray:
    """Decode a Kaggle-style RLE string back into a (H, W) binary mask."""
    height, width = shape
    if not rle or not rle.strip():
        return np.zeros((height, width), dtype=np.uint8)
    s = [int(x) for x in rle.split()]
    starts, lengths = s[0::2], s[1::2]
    mask = np.zeros(height * width, dtype=np.uint8)
    for start, length in zip(starts, lengths):
        start -= 1
        mask[start:start + length] = 1
    return mask.reshape((width, height)).T
