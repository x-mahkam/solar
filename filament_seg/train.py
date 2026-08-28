"""Train the U-Net baseline on filament-segmentation-2026 data.

Example:
    python -m filament_seg.train \
        --images-dir data/train/images \
        --ann-file data/train_annotations.json \
        --output-dir runs/baseline \
        --epochs 40 --batch-size 8 --image-size 512
"""
from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

from .dataset import FilamentDataset
from .losses import BCEDiceLoss
from .metrics import pixel_scores
from .model import UNet
from .transforms import default_train_transform


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--images-dir", required=True)
    p.add_argument("--ann-file", required=True)
    p.add_argument("--output-dir", default="runs/baseline")
    p.add_argument("--image-size", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--base-ch", type=int, default=32)
    p.add_argument("--val-fraction", type=float, default=0.15)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    full_ds = FilamentDataset(args.images_dir, args.ann_file, image_size=args.image_size)
    n_val = max(1, int(len(full_ds) * args.val_fraction))
    n_train = len(full_ds) - n_val
    generator = torch.Generator().manual_seed(args.seed)
    train_subset, val_subset = random_split(full_ds, [n_train, n_val], generator=generator)

    train_ds = FilamentDataset(
        args.images_dir, args.ann_file, image_size=args.image_size,
        transform=default_train_transform(), image_ids=[full_ds.image_ids[i] for i in train_subset.indices],
    )
    val_ds = FilamentDataset(
        args.images_dir, args.ann_file, image_size=args.image_size,
        image_ids=[full_ds.image_ids[i] for i in val_subset.indices],
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               num_workers=args.num_workers, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers)

    device = torch.device(args.device)
    model = UNet(in_channels=1, num_classes=1, base_ch=args.base_ch).to(device)
    criterion = BCEDiceLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    log_path = output_dir / "log.csv"
    best_iou = -1.0
    with open(log_path, "w", newline="") as log_file:
        writer = csv.writer(log_file)
        writer.writerow(["epoch", "train_loss", "val_loss", "val_iou", "val_dice", "val_precision", "val_recall"])

        for epoch in range(1, args.epochs + 1):
            model.train()
            train_loss = 0.0
            for images, masks, _ in train_loader:
                images, masks = images.to(device), masks.to(device)
                optimizer.zero_grad()
                logits = model(images)
                loss = criterion(logits, masks)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * images.size(0)
            train_loss /= len(train_ds)

            model.eval()
            val_loss = 0.0
            agg = {"iou": 0.0, "dice": 0.0, "precision": 0.0, "recall": 0.0}
            with torch.no_grad():
                for images, masks, _ in val_loader:
                    images, masks = images.to(device), masks.to(device)
                    logits = model(images)
                    loss = criterion(logits, masks)
                    val_loss += loss.item() * images.size(0)
                    scores = pixel_scores(logits, masks)
                    for k in agg:
                        agg[k] += scores[k] * images.size(0)
            val_loss /= len(val_ds)
            for k in agg:
                agg[k] /= len(val_ds)

            scheduler.step()
            print(
                f"epoch {epoch:3d}/{args.epochs} "
                f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
                f"val_iou={agg['iou']:.4f} val_dice={agg['dice']:.4f}"
            )
            writer.writerow([epoch, train_loss, val_loss, agg["iou"], agg["dice"], agg["precision"], agg["recall"]])
            log_file.flush()

            torch.save(model.state_dict(), output_dir / "last.pt")
            if agg["iou"] > best_iou:
                best_iou = agg["iou"]
                torch.save(model.state_dict(), output_dir / "best.pt")

    print(f"Best val IoU: {best_iou:.4f}. Checkpoints saved to {output_dir}")


if __name__ == "__main__":
    main()
