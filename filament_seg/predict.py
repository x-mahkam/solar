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

Two inference modes:
- Tiled (pass --tile-size): native-resolution sliding-window inference over
  the full image -- use this for a checkpoint trained with --crop-size, so
  thin barbs are never destroyed by a resize. Recommended.
- Whole-image resize (default, --image-size): legacy mode matching a
  checkpoint trained without --crop-size.

Example (tiled, matching --crop-size 512 training):
    python -m filament_seg.predict \
        --images-dir data/test/test_images \
        --checkpoint runs/baseline/best.pt \
        --tile-size 512 --tta \
        --output submission.csv
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage
from torch.utils.data import DataLoader

from .dataset import FilamentTestDataset
from .metrics import separate_touching_instances
from .model import UNet
from .rle import mask_to_coco_rle
from .tiling import tiled_predict


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--images-dir", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", default="submission.csv")
    p.add_argument("--image-size", type=int, default=512,
                    help="whole-image resize target (legacy mode, ignored if --tile-size is given)")
    p.add_argument("--tile-size", type=int, default=None,
                    help="enable tiled native-resolution inference with this tile size "
                         "(should match training --crop-size)")
    p.add_argument("--tile-overlap", type=int, default=64)
    p.add_argument("--tta", action="store_true",
                    help="average predictions over horizontal+vertical flips (tiled mode only)")
    p.add_argument("--base-ch", type=int, default=32)
    p.add_argument("--use-attention", action="store_true",
                    help="must match what the checkpoint was trained with (train.py --use-attention)")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--min-component-size", type=int, default=10,
                    help="drop predicted connected components smaller than this many pixels")
    p.add_argument("--closing-size", type=int, default=0,
                    help="morphological closing kernel size before splitting into components "
                         "(0 = off); merges nearby fragments of the same filament, reducing "
                         "over-segmentation -- tune with filament_seg.tune_postprocess")
    p.add_argument("--erosion-size", type=int, default=0,
                    help="erode-then-regrow kernel size (0 = off) to split filaments that touch "
                         "into separate instances before labeling -- tune with tune_postprocess")
    p.add_argument("--file-list", default=None,
                    help="optional text file, one image filename per line, to restrict prediction "
                         "to a subset (e.g. runs/baseline/val_file_names.txt for local self-evaluation)")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def _postprocess_to_rows(
    mask: np.ndarray, stem: str, closing_size: int, min_component_size: int, erosion_size: int = 0
) -> list[list[str]]:
    if closing_size > 0:
        structure = np.ones((closing_size, closing_size), dtype=bool)
        mask = ndimage.binary_closing(mask, structure=structure)
    components = separate_touching_instances(mask, erosion_size=erosion_size)
    components = [c for c in components if c.sum() >= min_component_size]
    return [[f"{stem}_{idx}", mask_to_coco_rle(comp.astype(np.uint8))] for idx, comp in enumerate(components, start=1)]


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    model = UNet(in_channels=1, num_classes=1, base_ch=args.base_ch, use_attention=args.use_attention).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    file_names = None
    if args.file_list:
        with open(args.file_list) as f:
            file_names = [line.strip() for line in f if line.strip()]

    rows = []
    if args.tile_size:
        images_dir = Path(args.images_dir)
        if file_names is not None:
            paths = sorted(images_dir / name for name in file_names)
        else:
            paths = sorted(
                p for p in images_dir.iterdir()
                if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
            )
        for path in paths:
            prob_map, _, _ = tiled_predict(
                model, path, tile_size=args.tile_size, overlap=args.tile_overlap,
                device=device, batch_size=args.batch_size, tta=args.tta,
            )
            mask = prob_map > args.threshold
            rows.extend(_postprocess_to_rows(mask, path.stem, args.closing_size, args.min_component_size, args.erosion_size))
    else:
        ds = FilamentTestDataset(args.images_dir, image_size=args.image_size, file_names=file_names)
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=4)
        with torch.no_grad():
            for images, names, heights, widths in loader:
                images = images.to(device)
                logits = model(images)
                probs = torch.sigmoid(logits)

                for i in range(len(names)):
                    h, w = int(heights[i]), int(widths[i])
                    prob = probs[i, 0].unsqueeze(0).unsqueeze(0)
                    prob = F.interpolate(prob, size=(h, w), mode="bilinear", align_corners=False)
                    mask = (prob[0, 0].cpu().numpy() > args.threshold)
                    stem = Path(names[i]).stem
                    rows.extend(_postprocess_to_rows(mask, stem, args.closing_size, args.min_component_size, args.erosion_size))

    with open(args.output, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filament_id", "segmentation_rle"])
        writer.writerows(rows)

    print(f"Wrote {len(rows)} predicted filaments to {args.output}")


if __name__ == "__main__":
    main()
