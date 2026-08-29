"""Run inference and write the competition's submission CSV.

Per the official Evaluation page for filament-segmentation-2026, the
submission is a single CSV covering the whole test set with one row per
*predicted filament instance* (not one row per image):

    filament_id,segmentation_rle
    20150125172714Mh_1,"<pycocotools RLE counts>"
    20150125172714Mh_2,"<pycocotools RLE counts>"
    ...

- filament_id is "<image filename stem>_<1-based index>".
- segmentation_rle is the pycocotools RLE "counts" string (no quotes, no
  stored size -- every image is fixed at 2048x2048).

Predicted instances come from splitting the model's binary mask into
connected components; this is a reasonable proxy but will under-segment two
filaments that touch (see README "Where to take it from here").

Example:
    python -m filament_seg.predict \
        --images-dir data/test/test_images \
        --checkpoint runs/baseline/best.pt \
        --output submission.csv
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .dataset import FilamentTestDataset
from .metrics import connected_components
from .model import UNet
from .rle import mask_to_coco_rle


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--images-dir", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", default="submission.csv")
    p.add_argument("--image-size", type=int, default=512)
    p.add_argument("--base-ch", type=int, default=32)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--min-component-size", type=int, default=10,
                    help="drop predicted connected components smaller than this many pixels")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    model = UNet(in_channels=1, num_classes=1, base_ch=args.base_ch).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    ds = FilamentTestDataset(args.images_dir, image_size=args.image_size)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

    rows = []
    with torch.no_grad():
        for images, names, heights, widths in loader:
            images = images.to(device)
            logits = model(images)
            probs = torch.sigmoid(logits)

            for i in range(len(names)):
                h, w = int(heights[i]), int(widths[i])
                prob = probs[i, 0].unsqueeze(0).unsqueeze(0)
                prob = F.interpolate(prob, size=(h, w), mode="bilinear", align_corners=False)
                mask = (prob[0, 0].cpu().numpy() > args.threshold).astype(np.uint8)

                stem = Path(names[i]).stem
                components = connected_components(mask.astype(bool))
                components = [c for c in components if c.sum() >= args.min_component_size]
                for idx, comp in enumerate(components, start=1):
                    filament_id = f"{stem}_{idx}"
                    rows.append([filament_id, mask_to_coco_rle(comp.astype(np.uint8))])

    with open(args.output, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filament_id", "segmentation_rle"])
        writer.writerows(rows)

    print(f"Wrote {len(rows)} predicted filaments to {args.output}")


if __name__ == "__main__":
    main()
