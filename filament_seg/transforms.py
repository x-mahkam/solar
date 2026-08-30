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


class RandomBrightnessContrast:
    """Jitter brightness/contrast of the (already [0,1]-scaled) image only.
    GONG H-alpha frames vary in background noise/limb-darkening intensity
    (per the competition's own "Main Challenges" notes); this is meant to
    make the model robust to that rather than memorizing specific exposure
    levels, which should help close the gap between local validation and
    the real held-out test set."""

    def __init__(self, brightness: float = 0.15, contrast: float = 0.15, p: float = 0.5):
        self.brightness = brightness
        self.contrast = contrast
        self.p = p

    def __call__(self, image, mask):
        if random.random() < self.p:
            contrast_factor = 1.0 + random.uniform(-self.contrast, self.contrast)
            brightness_delta = random.uniform(-self.brightness, self.brightness)
            mean = image.mean()
            image = np.clip((image - mean) * contrast_factor + mean + brightness_delta, 0.0, 1.0)
        return image, mask


def default_train_transform() -> Compose:
    return Compose([RandomHFlip(), RandomVFlip(), RandomRotate90(), RandomBrightnessContrast()])
