"""PyTorch datasets for the MAGFiLO / filament-segmentation-2026 data.

Confirmed layout for the real filament-segmentation-2026 download:

    data/
      train/
        <file_name>.jpeg                  # matches COCO "file_name", flat, no subfolder
      train_annotations.json              # COCO-format polygons (2048x2048 full-disk H-alpha)

If your test set folder differs, pass ``images_dir`` explicitly -- nothing
here is hard-coded beyond the ``--images-dir`` / ``--ann-file`` CLI args.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from . import coco_utils

IMAGENET_MEAN = 0.449  # single-channel approximation, fine for grayscale H-alpha
IMAGENET_STD = 0.226


class FilamentDataset(Dataset):
    """Training/validation dataset: image + binary semantic mask of filaments."""

    def __init__(
        self,
        images_dir: str | Path,
        ann_file: str | Path,
        image_size: int = 512,
        transform: Callable | None = None,
        image_ids: list[int] | None = None,
    ):
        self.images_dir = Path(images_dir)
        self.image_size = image_size
        self.transform = transform

        coco = coco_utils.load_coco(ann_file)
        self.images = coco_utils.index_images(coco)
        self.anns_by_image = coco_utils.index_annotations(coco)
        self.image_ids = image_ids if image_ids is not None else sorted(self.images.keys())

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, idx: int):
        image_id = self.image_ids[idx]
        info = self.images[image_id]
        img_path = self.images_dir / info["file_name"]

        image = Image.open(img_path).convert("L")
        width, height = image.size
        # image dims should match COCO metadata; fall back to actual file size otherwise
        height = info.get("height", height)
        width = info.get("width", width)

        anns = self.anns_by_image.get(image_id, [])
        mask = coco_utils.annotations_to_semantic_mask(anns, height, width)

        image = image.resize((self.image_size, self.image_size), Image.BILINEAR)
        mask_img = Image.fromarray(mask * 255).resize(
            (self.image_size, self.image_size), Image.NEAREST
        )

        image_np = np.asarray(image, dtype=np.float32) / 255.0
        mask_np = (np.asarray(mask_img, dtype=np.float32) > 127).astype(np.float32)

        if self.transform is not None:
            image_np, mask_np = self.transform(image_np, mask_np)

        image_np = (image_np - IMAGENET_MEAN) / IMAGENET_STD

        image_t = torch.from_numpy(image_np).unsqueeze(0).float()
        mask_t = torch.from_numpy(mask_np).unsqueeze(0).float()
        return image_t, mask_t, image_id


class FilamentTestDataset(Dataset):
    """Inference dataset: just images, no annotations. Also returns original size
    so predictions can be resized back before RLE-encoding."""

    def __init__(self, images_dir: str | Path, image_size: int = 512, file_names: list[str] | None = None):
        self.images_dir = Path(images_dir)
        self.image_size = image_size
        if file_names is not None:
            self.files = sorted(self.images_dir / name for name in file_names)
        else:
            self.files = sorted(
                p for p in self.images_dir.iterdir()
                if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".fits"}
            )

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        path = self.files[idx]
        image = Image.open(path).convert("L")
        orig_w, orig_h = image.size
        image = image.resize((self.image_size, self.image_size), Image.BILINEAR)
        image_np = np.asarray(image, dtype=np.float32) / 255.0
        image_np = (image_np - IMAGENET_MEAN) / IMAGENET_STD
        image_t = torch.from_numpy(image_np).unsqueeze(0).float()
        return image_t, path.name, orig_h, orig_w
