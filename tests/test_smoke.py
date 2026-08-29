"""End-to-end smoke test on synthetic data: builds a tiny fake COCO dataset,
runs the dataset loader, one training step, metrics, RLE round-trip, and
inference. No real competition data required.

Run with: python tests/test_smoke.py
"""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from filament_seg.dataset import FilamentDataset, FilamentTestDataset
from filament_seg.losses import BCEDiceLoss
from filament_seg.metrics import connected_components, instance_metrics, panoptic_quality, pixel_scores
from filament_seg.model import UNet
from filament_seg.rle import coco_rle_to_mask, mask_to_coco_rle, mask_to_rle, rle_to_mask


def make_fake_data(root: Path, n_images: int = 4, size: int = 128):
    images_dir = root / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    coco = {"images": [], "annotations": [], "categories": [{"id": 1, "name": "filament"}]}
    ann_id = 0
    rng = np.random.default_rng(0)

    for img_id in range(n_images):
        arr = (rng.random((size, size)) * 40 + 20).astype(np.uint8)  # noisy background
        fname = f"img_{img_id}.png"
        Image.fromarray(arr, mode="L").save(images_dir / fname)
        coco["images"].append({"id": img_id, "file_name": fname, "height": size, "width": size})

        # a couple of simple polygon "filaments" per image
        for k in range(2):
            cx, cy = rng.integers(20, size - 20, size=2)
            poly = [
                cx - 15, cy - 3,
                cx + 15, cy - 3,
                cx + 15, cy + 3,
                cx - 15, cy + 3,
            ]
            coco["annotations"].append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": 1,
                "segmentation": [list(map(float, poly))],
                "chirality": "left" if k == 0 else "right",
            })
            ann_id += 1

    ann_file = root / "annotations.json"
    with open(ann_file, "w") as f:
        json.dump(coco, f)
    return images_dir, ann_file


def main():
    tmp_root = Path(tempfile.mkdtemp(prefix="filament_smoke_test_"))
    images_dir, ann_file = make_fake_data(tmp_root)

    ds = FilamentDataset(images_dir, ann_file, image_size=64)
    assert len(ds) == 4
    image, mask, image_id = ds[0]
    assert image.shape == (1, 64, 64)
    assert mask.shape == (1, 64, 64)
    assert mask.sum() > 0, "expected non-empty mask from synthetic polygons"
    print("dataset OK:", image.shape, mask.shape, "mask pixels:", int(mask.sum()))

    model = UNet(in_channels=1, num_classes=1, base_ch=8)
    criterion = BCEDiceLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    batch_img = image.unsqueeze(0)
    batch_mask = mask.unsqueeze(0)
    logits = model(batch_img)
    assert logits.shape == batch_mask.shape
    loss = criterion(logits, batch_mask)
    loss.backward()
    optimizer.step()
    print("train step OK: loss =", loss.item())

    scores = pixel_scores(logits, batch_mask)
    print("pixel metrics OK:", scores)

    pred_mask = (torch.sigmoid(logits)[0, 0] > 0.5).numpy().astype(np.uint8)
    gt_np = mask[0].numpy().astype(np.uint8)
    comps = connected_components(gt_np.astype(bool))
    print("connected_components OK: found", len(comps), "components in GT")

    inst_scores = instance_metrics(pred_mask, comps)
    print("instance metrics OK:", inst_scores)

    pred_comps = connected_components(pred_mask.astype(bool))
    pq = panoptic_quality(pred_comps, comps)
    print("panoptic_quality OK:", pq)

    rle = mask_to_rle(gt_np)
    recovered = rle_to_mask(rle, gt_np.shape)
    assert np.array_equal(gt_np, recovered), "RLE round-trip mismatch"
    print("RLE round-trip OK, length:", len(rle))

    coco_rle = mask_to_coco_rle(gt_np)
    coco_recovered = coco_rle_to_mask(coco_rle, gt_np.shape)
    assert np.array_equal(gt_np, coco_recovered), "COCO RLE round-trip mismatch"
    print("COCO RLE round-trip OK (submission format), counts:", coco_rle)

    test_ds = FilamentTestDataset(images_dir, image_size=64)
    t_img, name, h, w = test_ds[0]
    assert t_img.shape == (1, 64, 64)
    print("test dataset OK:", name, h, w)

    shutil.rmtree(tmp_root)
    print("\nALL SMOKE TESTS PASSED")


if __name__ == "__main__":
    main()
