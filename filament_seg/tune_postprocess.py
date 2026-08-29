"""Sweep inference-time post-processing (probability threshold, minimum
component size, morphological closing) against an already-trained
checkpoint's cached probability maps, scoring each combination with the
organizers' exact Panoptic Quality formula (via self_eval).

This does NOT retrain anything -- it re-uses one forward pass per image and
then cheaply re-binarizes/re-labels it for every candidate combination, so
it's a fast way to find a better operating point before spending GPU time on
a full retrain. Useful when predicted instance count is far from the
ground-truth count (e.g. over-segmentation from noisy thresholded masks).

Example:
    python -m filament_seg.tune_postprocess \
        --ann-file MAGFiLO_1.0_Kaggle_2026/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json \
        --images-dir MAGFiLO_1.0_Kaggle_2026/train/train_images \
        --checkpoint runs/baseline/best.pt \
        --file-list runs/baseline/val_file_names.txt \
        --image-ids-file runs/baseline/val_image_ids.txt
"""
from __future__ import annotations

import argparse
import itertools

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy import ndimage

from . import self_eval
from .dataset import FilamentTestDataset
from .metrics import connected_components
from .model import UNet
from .rle import mask_to_coco_rle


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--images-dir", required=True)
    p.add_argument("--ann-file", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--file-list", required=True, help="e.g. runs/baseline/val_file_names.txt")
    p.add_argument("--image-ids-file", required=True, help="e.g. runs/baseline/val_image_ids.txt")
    p.add_argument("--image-size", type=int, default=512, help="inference resize, must match training")
    p.add_argument("--base-ch", type=int, default=32)
    p.add_argument("--thresholds", type=float, nargs="+", default=[0.3, 0.4, 0.5, 0.6, 0.7])
    p.add_argument("--min-sizes", type=int, nargs="+", default=[0, 10, 30, 60, 100])
    p.add_argument("--closing-sizes", type=int, nargs="+", default=[0, 3, 5],
                    help="morphological closing kernel size (0 = off); merges nearby fragments")
    p.add_argument("--output", default="tune_results.csv")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    model = UNet(in_channels=1, num_classes=1, base_ch=args.base_ch).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    with open(args.file_list) as f:
        file_names = [line.strip() for line in f if line.strip()]
    with open(args.image_ids_file) as f:
        image_ids = [line.strip() for line in f if line.strip()]

    ds = FilamentTestDataset(args.images_dir, image_size=args.image_size, file_names=file_names)

    print(f"Running {len(ds)} images through the model once, caching probability maps...")
    prob_maps: dict[str, np.ndarray] = {}
    sizes: dict[str, tuple[int, int]] = {}
    with torch.no_grad():
        for image_t, name, h, w in ds:
            logits = model(image_t.unsqueeze(0).to(device))
            prob = torch.sigmoid(logits)
            prob = F.interpolate(prob, size=(int(h), int(w)), mode="bilinear", align_corners=False)
            stem = name.rsplit(".", 1)[0]
            # quantize to uint8 to keep memory bounded across many full-res images
            prob_maps[stem] = (prob[0, 0].cpu().numpy() * 255).astype(np.uint8)
            sizes[stem] = (int(h), int(w))

    print("Building ground-truth dataframe...")
    gt_df = self_eval.build_gt_dataframe(args.ann_file, image_ids=image_ids)

    combos = list(itertools.product(args.thresholds, args.min_sizes, args.closing_sizes))
    results = []
    for threshold, min_size, closing_size in combos:
        rows = []
        for stem, prob_u8 in prob_maps.items():
            binary = prob_u8 > int(threshold * 255)
            if closing_size > 0:
                structure = np.ones((closing_size, closing_size), dtype=bool)
                binary = ndimage.binary_closing(binary, structure=structure)
            components = connected_components(binary)
            components = [c for c in components if c.sum() >= min_size]
            for idx, comp in enumerate(components, start=1):
                rows.append({
                    "filament_id": f"{stem}_{idx}",
                    "segmentation_rle": mask_to_coco_rle(comp.astype(np.uint8)),
                })
        pred_df = pd.DataFrame(rows, columns=["filament_id", "segmentation_rle"])
        h, w = next(iter(sizes.values()))
        pq = self_eval.panoptic_quality_from_dataframes(gt_df, pred_df, height=h, width=w)
        results.append({
            "threshold": threshold, "min_size": min_size, "closing_size": closing_size,
            "n_predicted": len(pred_df), "pq": pq,
        })
        print(f"threshold={threshold} min_size={min_size} closing={closing_size} "
              f"n_pred={len(pred_df)} PQ={pq:.4f}")

    results_df = pd.DataFrame(results).sort_values("pq", ascending=False)
    results_df.to_csv(args.output, index=False)
    best = results_df.iloc[0]
    print(f"\nBest: threshold={best.threshold} min_size={int(best.min_size)} "
          f"closing_size={int(best.closing_size)} -> PQ={best.pq:.4f}")
    print(f"Full results written to {args.output}")


if __name__ == "__main__":
    main()
