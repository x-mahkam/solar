"""Run inference and write a Kaggle-style submission CSV.

The exact column names / one-row-per-image vs one-row-per-instance format for
filament-segmentation-2026 should be copied from the real sample_submission.csv
once you've downloaded the competition data (see scripts/download_data.md).
By default this produces one row per image with a single RLE covering all
predicted filament pixels (ImageId, EncodedPixels); pass --instance-level to
instead emit one row per predicted connected component.

Example:
    python -m filament_seg.predict \
        --images-dir data/test/images \
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
from PIL import Image
from torch.utils.data import DataLoader

from .dataset import FilamentTestDataset
from .metrics import connected_components
from .model import UNet
from .rle import mask_to_rle


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
    p.add_argument("--instance-level", action="store_true",
                    help="emit one row per connected component instead of one row per image")
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

                if args.instance_level:
                    components = connected_components(mask.astype(bool))
                    components = [c for c in components if c.sum() >= args.min_component_size]
                    if not components:
                        rows.append([names[i], ""])
                    for comp in components:
                        rows.append([names[i], mask_to_rle(comp.astype(np.uint8))])
                else:
                    rows.append([names[i], mask_to_rle(mask)])

    with open(args.output, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["ImageId", "EncodedPixels"])
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
