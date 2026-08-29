"""Run-length encoding helpers.

Two distinct RLE formats are used in this repo:

- ``mask_to_rle`` / ``rle_to_mask``: the simple numeric "start length ..."
  format used by many older Kaggle segmentation competitions (Airbus/HuBMAP/
  Sartorius-style). Handy for quick local round-trip checks.
- ``mask_to_coco_rle``: the compressed ASCII format produced by
  ``pycocotools.mask.encode`` (COCO RLE "counts" string). This is what
  filament-segmentation-2026 actually expects in ``segmentation_rle`` --
  see the competition's Evaluation page, which explicitly names
  ``pycocotools`` / ``annToMask`` as the expected conversion path. Do not
  submit the numeric format below; it will not be understood by their
  scorer.
"""
from __future__ import annotations

import numpy as np
from pycocotools import mask as _maskUtils


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


def mask_to_coco_rle(mask: np.ndarray) -> str:
    """Encode a binary (H, W) mask as a pycocotools RLE "counts" string --
    the format filament-segmentation-2026 expects in ``segmentation_rle``.
    Only the counts are returned (the competition fixes size at 2048x2048
    for every image, so it doesn't need to be stored per row)."""
    fortran_mask = np.asfortranarray(mask.astype(np.uint8))
    rle = _maskUtils.encode(fortran_mask)
    return rle["counts"].decode("ascii")


def coco_rle_to_mask(counts: str, shape: tuple[int, int]) -> np.ndarray:
    """Decode a pycocotools RLE "counts" string back into a (H, W) mask."""
    height, width = shape
    rle = {"size": [height, width], "counts": counts.encode("ascii")}
    return _maskUtils.decode(rle)
