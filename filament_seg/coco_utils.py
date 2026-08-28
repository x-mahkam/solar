"""Helpers for reading MAGFiLO/COCO-style filament annotations and turning
polygon segmentations into raster masks.

MAGFiLO annotations follow the COCO format: a JSON file with top-level
``images`` and ``annotations`` lists. Each annotation carries a polygon
``segmentation`` (a filament's pixel-precise shape), a ``bbox``, and a
chirality label. The exact key used for chirality varies between dataset
releases, so :func:`chirality_of` checks a few common spots before giving up.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

CHIRALITY_KEYS = ("chirality", "label", "class", "chirality_label")
CHIRALITY_VALUES = {"left", "right", "unidentifiable"}


def load_coco(ann_file: str | Path) -> dict[str, Any]:
    with open(ann_file, "r") as f:
        return json.load(f)


def index_images(coco: dict[str, Any]) -> dict[int, dict[str, Any]]:
    """Map image_id -> image info dict (file_name, height, width, ...)."""
    return {img["id"]: img for img in coco["images"]}


def index_annotations(coco: dict[str, Any]) -> dict[int, list[dict[str, Any]]]:
    """Map image_id -> list of annotation dicts for that image."""
    by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for ann in coco["annotations"]:
        by_image[ann["image_id"]].append(ann)
    return by_image


def category_name_map(coco: dict[str, Any]) -> dict[int, str]:
    return {c["id"]: c.get("name", str(c["id"])) for c in coco.get("categories", [])}


def chirality_of(ann: dict[str, Any], cat_names: dict[int, str] | None = None) -> str:
    for key in CHIRALITY_KEYS:
        val = ann.get(key)
        if isinstance(val, str) and val.lower() in CHIRALITY_VALUES:
            return val.lower()
    attrs = ann.get("attributes") or {}
    for key in CHIRALITY_KEYS:
        val = attrs.get(key)
        if isinstance(val, str) and val.lower() in CHIRALITY_VALUES:
            return val.lower()
    if cat_names is not None:
        name = cat_names.get(ann.get("category_id"), "")
        if name.lower() in CHIRALITY_VALUES:
            return name.lower()
    return "unidentifiable"


def _polygon_rings(segmentation: Any) -> list[list[float]]:
    """Normalize a COCO ``segmentation`` field into a list of flat [x,y,...] rings.

    Handles the common "list of polygons" form. RLE-encoded segmentations
    (dict with ``counts``/``size``) are not expected for MAGFiLO but are
    skipped gracefully rather than raising.
    """
    if isinstance(segmentation, list):
        return [ring for ring in segmentation if len(ring) >= 6]
    return []


def polygons_to_mask(
    segmentation: Any, height: int, width: int
) -> np.ndarray:
    """Rasterize one annotation's polygon(s) into a single-instance binary mask."""
    mask_img = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask_img)
    for ring in _polygon_rings(segmentation):
        xy = list(zip(ring[0::2], ring[1::2]))
        if len(xy) >= 3:
            draw.polygon(xy, outline=1, fill=1)
    return np.array(mask_img, dtype=np.uint8)


def annotations_to_semantic_mask(
    anns: list[dict[str, Any]], height: int, width: int
) -> np.ndarray:
    """Union of every filament polygon for an image into one binary mask."""
    mask = np.zeros((height, width), dtype=np.uint8)
    for ann in anns:
        mask |= polygons_to_mask(ann["segmentation"], height, width)
    return mask


def annotations_to_instance_masks(
    anns: list[dict[str, Any]], height: int, width: int
) -> list[np.ndarray]:
    """One binary mask per filament instance (for instance-level metrics/RLE)."""
    return [polygons_to_mask(a["segmentation"], height, width) for a in anns]
