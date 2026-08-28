"""Lightweight joint image+mask augmentations (no albumentations dependency)."""
from __future__ import annotations

import random

import numpy as np


class Compose:
    def __init__(self, transforms: list):
        self.transforms = transforms

    def __call__(self, image: np.ndarray, mask: np.ndarray):
        for t in self.transforms:
            image, mask = t(image, mask)
        return image, mask


class RandomHFlip:
    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, image, mask):
        if random.random() < self.p:
            image = np.ascontiguousarray(image[:, ::-1])
            mask = np.ascontiguousarray(mask[:, ::-1])
        return image, mask


class RandomVFlip:
    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, image, mask):
        if random.random() < self.p:
            image = np.ascontiguousarray(image[::-1, :])
            mask = np.ascontiguousarray(mask[::-1, :])
        return image, mask


class RandomRotate90:
    """Rotate by a random multiple of 90 degrees (safe for full-disk images)."""

    def __call__(self, image, mask):
        k = random.randint(0, 3)
        if k:
            image = np.ascontiguousarray(np.rot90(image, k))
            mask = np.ascontiguousarray(np.rot90(mask, k))
        return image, mask


def default_train_transform() -> Compose:
    return Compose([RandomHFlip(), RandomVFlip(), RandomRotate90()])
