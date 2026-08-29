"""Local self-evaluation matching the organizers' official scorer exactly.

The competition published a self-evaluation notebook
(kaggle.com/code/azimahmadzadeh/self-evaluation-notebook) that computes
Panoptic Quality from two dataframes -- ``gt_df`` and ``pred_df`` -- each
with columns ``filament_id`` and ``segmentation_rle``. This module is a
faithful, tested port of that notebook's scoring logic (``get_overlap_df`` /
``get_pq_score``), plus a helper to build ``gt_df`` directly from a MAGFiLO
COCO annotations file, so you can validate a held-out split locally before
trusting the leaderboard number.

Ground-truth filament_id, per the official notebook, is
"<annotator_id>-<image_id>_<n>" (the COCO ``images[].id`` field already
carries the "<annotator_id>-<image_id>" prefix in MAGFiLO -- see
coco_utils.py); prediction filament_id is just "<image_id>_<n>"
(filament_seg.predict already produces this). The scorer matches the two by
stripping the annotator prefix off the GT side.

CLI:
    python -m filament_seg.self_eval build-gt --ann-file ... --output gt.csv \
        [--image-ids-file val_image_ids.txt]
    python -m filament_seg.self_eval score --gt-csv gt.csv --pred-csv submission.csv
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import torch
from pycocotools import mask as mask_util

from . import coco_utils
from .rle import mask_to_coco_rle


def build_gt_dataframe(ann_file: str, image_ids: list[str] | None = None) -> pd.DataFrame:
    """Ground-truth dataframe in the same (filament_id, segmentation_rle)
    schema as the competition's submission format, built from a COCO
    annotations file. Pass ``image_ids`` (COCO ``images[].id`` values, e.g.
    a validation split) to restrict to a subset."""
    coco = coco_utils.load_coco(ann_file)
    images = coco_utils.index_images(coco)
    anns_by_image = coco_utils.index_annotations(coco)

    ids = image_ids if image_ids is not None else list(images.keys())

    rows = []
    for image_id in ids:
        info = images[image_id]
        height, width = info["height"], info["width"]
        for idx, ann in enumerate(anns_by_image.get(image_id, []), start=1):
            mask = coco_utils.polygons_to_mask(ann["segmentation"], height, width)
            rows.append({
                "filament_id": f"{image_id}_{idx}",
                "segmentation_rle": mask_to_coco_rle(mask),
            })
    return pd.DataFrame(rows, columns=["filament_id", "segmentation_rle"])


# --- Faithful port of the organizers' self-evaluation notebook below ---

def _rles_to_layers(rles: list[str], height: int = 2048, width: int = 2048) -> np.ndarray:
    if not rles:
        return np.zeros((0, height, width), dtype=np.float32)
    rle_dicts = [{"size": [height, width], "counts": rle} for rle in rles]
    masks = mask_util.decode(rle_dicts)
    return masks.transpose(2, 0, 1).astype(np.float32)


def _get_overlap_matrices(gt_layers: torch.Tensor, pred_layers: torch.Tensor):
    n_gt, height, width = gt_layers.shape
    n_pred = pred_layers.shape[0]

    gt_flat = gt_layers.reshape(n_gt, height * width)
    pred_flat = pred_layers.reshape(n_pred, height * width)

    intersection = torch.matmul(gt_flat, pred_flat.t())
    gt_areas = gt_flat.sum(dim=1).view(-1, 1)
    pred_areas = pred_flat.sum(dim=1).view(1, -1)
    union = gt_areas + pred_areas - intersection

    iou_matrix = torch.where(union == 0, torch.tensor(0.0), intersection / union)
    return iou_matrix


def _fp_count_hit(hit_matrix: torch.Tensor) -> int:
    return (hit_matrix.sum(dim=0) == 0).sum().item()


def _fn_count_hit(hit_matrix: torch.Tensor) -> int:
    return (hit_matrix.sum(dim=1) == 0).sum().item()


def get_overlap_df(gt_df: pd.DataFrame, pred_df: pd.DataFrame, height: int = 2048, width: int = 2048) -> pd.DataFrame:
    gt_annotator_image_ids = gt_df["filament_id"].str.split("_", n=1).str[0]
    pred_image_ids = pred_df["filament_id"].str.split("_", n=1).str[0]

    annotator_images = gt_annotator_image_ids.unique()
    records = []
    for annotator_image in annotator_images:
        image_id = annotator_image.split("-", maxsplit=1)[1]

        gt_rles = gt_df.loc[gt_annotator_image_ids == annotator_image, "segmentation_rle"].tolist()
        gt_layers = torch.from_numpy(_rles_to_layers(gt_rles, height, width))

        pred_rles = pred_df.loc[pred_image_ids == image_id, "segmentation_rle"].tolist()
        pred_layers = torch.from_numpy(_rles_to_layers(pred_rles, height, width))

        n_gt, n_pred = len(gt_rles), len(pred_rles)
        iou_matrix = _get_overlap_matrices(gt_layers, pred_layers) if n_gt and n_pred else torch.zeros((n_gt, n_pred))
        records.append({"annotator_image": annotator_image, "iou_matrix": iou_matrix, "n_gt": n_gt, "n_pred": n_pred})

    return pd.DataFrame(records)


def get_pq_score(overlap_df: pd.DataFrame, iou_threshold: float = 0.5) -> float:
    tp_iou_scores: list[float] = []
    fp_count = 0
    fn_count = 0

    for row in overlap_df.itertuples(index=False):
        iou_matrix, n_gt, n_pred = row.iou_matrix, row.n_gt, row.n_pred

        if n_gt == 0:
            fp_count += n_pred
            continue
        if n_pred == 0:
            fn_count += n_gt
            continue

        hit_matrix = iou_matrix > iou_threshold
        tp_iou_scores.extend(iou_matrix[hit_matrix].tolist())
        fp_count += _fp_count_hit(hit_matrix)
        fn_count += _fn_count_hit(hit_matrix)

    tp_count = len(tp_iou_scores)
    denominator = tp_count + 0.5 * fp_count + 0.5 * fn_count
    return sum(tp_iou_scores) / denominator if denominator > 0 else 0.0


def panoptic_quality_from_dataframes(gt_df: pd.DataFrame, pred_df: pd.DataFrame, height: int = 2048, width: int = 2048) -> float:
    overlap_df = get_overlap_df(gt_df, pred_df, height, width)
    return get_pq_score(overlap_df)


def _main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    build = sub.add_parser("build-gt", help="build a ground-truth CSV from a COCO annotations file")
    build.add_argument("--ann-file", required=True)
    build.add_argument("--output", default="gt.csv")
    build.add_argument("--image-ids-file", default=None,
                        help="optional text file, one COCO image id per line, to restrict to a subset (e.g. your validation split)")

    score = sub.add_parser("score", help="compute Panoptic Quality between a GT CSV and a prediction CSV")
    score.add_argument("--gt-csv", required=True)
    score.add_argument("--pred-csv", required=True)
    score.add_argument("--height", type=int, default=2048)
    score.add_argument("--width", type=int, default=2048)

    args = p.parse_args()

    if args.command == "build-gt":
        image_ids = None
        if args.image_ids_file:
            with open(args.image_ids_file) as f:
                image_ids = [line.strip() for line in f if line.strip()]
        gt_df = build_gt_dataframe(args.ann_file, image_ids=image_ids)
        gt_df.to_csv(args.output, index=False)
        print(f"Wrote {len(gt_df)} ground-truth filaments to {args.output}")
    elif args.command == "score":
        gt_df = pd.read_csv(args.gt_csv, dtype={"segmentation_rle": str})
        pred_df = pd.read_csv(args.pred_csv, dtype={"segmentation_rle": str})
        pq = panoptic_quality_from_dataframes(gt_df, pred_df, args.height, args.width)
        print(f"Panoptic Quality: {pq:.4f}")


if __name__ == "__main__":
    _main()
