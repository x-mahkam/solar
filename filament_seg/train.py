"""Train the U-Net baseline on filament-segmentation-2026 data.

Recommended: train on native-resolution crops (--crop-size) rather than a
whole-image resize (--image-size) -- filament barbs are only a few pixels
wide at the native 2048x2048 resolution and get destroyed by resizing down
to 512-768px. Pair with tiled inference at test time (predict.py --tile-size
matching --crop-size).

Example (recommended, crop-based):
    python -m filament_seg.train \
        --images-dir data/train/images \
        --ann-file data/train_annotations.json \
        --output-dir runs/baseline \
        --epochs 40 --batch-size 16 --crop-size 512 --loss tversky

Example (legacy, whole-image resize):
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
from tqdm import tqdm

from .dataset import FilamentDataset
from .losses import get_loss
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
    p.add_argument("--image-size", type=int, default=512,
                    help="whole-image resize target; ignored if --crop-size is given")
    p.add_argument("--crop-size", type=int, default=None,
                    help="train on random native-resolution crops of this size instead of "
                         "resizing the whole image -- preserves thin filament barbs that a "
                         "global resize would destroy. Pair with tiled inference at test time "
                         "(predict.py --tile-size).")
    p.add_argument("--positive-crop-prob", type=float, default=0.8,
                    help="with --crop-size, fraction of crops centered (with jitter) on a "
                         "filament rather than fully random -- filaments cover a small fraction "
                         "of each frame, so pure random crops mostly land on empty background")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--base-ch", type=int, default=32)
    p.add_argument("--use-attention", action="store_true",
                    help="add Attention U-Net-style gates on the skip connections (Oktay et al. "
                         "2018) -- helps focus on small/thin structures against clutter. Pass the "
                         "same flag to predict.py/tune_postprocess.py to load this checkpoint.")
    p.add_argument("--loss", choices=["bce_dice", "tversky"], default="bce_dice")
    p.add_argument("--tversky-alpha", type=float, default=0.3, help="Tversky false-positive weight")
    p.add_argument("--tversky-beta", type=float, default=0.7, help="Tversky false-negative weight (>alpha favors recall)")
    p.add_argument("--val-fraction", type=float, default=0.15)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--no-amp", action="store_true", help="disable automatic mixed precision (on by default on CUDA)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    full_ds = FilamentDataset(args.images_dir, args.ann_file, image_size=args.image_size,
                               crop_size=args.crop_size, positive_crop_prob=args.positive_crop_prob)
    n_val = max(1, int(len(full_ds) * args.val_fraction))
    n_train = len(full_ds) - n_val
    generator = torch.Generator().manual_seed(args.seed)
    train_subset, val_subset = random_split(full_ds, [n_train, n_val], generator=generator)

    train_ds = FilamentDataset(
        args.images_dir, args.ann_file, image_size=args.image_size, crop_size=args.crop_size,
        positive_crop_prob=args.positive_crop_prob,
        transform=default_train_transform(), image_ids=[full_ds.image_ids[i] for i in train_subset.indices],
    )
    val_ds = FilamentDataset(
        args.images_dir, args.ann_file, image_size=args.image_size, crop_size=args.crop_size,
        positive_crop_prob=args.positive_crop_prob,
        image_ids=[full_ds.image_ids[i] for i in val_subset.indices],
    )

    val_image_ids = val_ds.image_ids
    with open(output_dir / "val_image_ids.txt", "w") as f:
        f.write("\n".join(str(i) for i in val_image_ids))
    val_file_names = [full_ds.images[i]["file_name"] for i in val_image_ids]
    with open(output_dir / "val_file_names.txt", "w") as f:
        f.write("\n".join(val_file_names))
    print(f"Saved {len(val_image_ids)} validation image ids/filenames to {output_dir} "
          f"(use with filament_seg.self_eval to check PQ on the exact held-out split)")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               num_workers=args.num_workers, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers)

    device = torch.device(args.device)
    model = UNet(in_channels=1, num_classes=1, base_ch=args.base_ch, use_attention=args.use_attention).to(device)
    criterion = get_loss(args.loss, tversky_alpha=args.tversky_alpha, tversky_beta=args.tversky_beta)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    use_amp = device.type == "cuda" and not args.no_amp
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    if use_amp:
        print("Using automatic mixed precision (AMP)")

    log_path = output_dir / "log.csv"
    best_iou = -1.0
    with open(log_path, "w", newline="") as log_file:
        writer = csv.writer(log_file)
        writer.writerow(["epoch", "train_loss", "val_loss", "val_iou", "val_dice", "val_precision", "val_recall"])

        for epoch in range(1, args.epochs + 1):
            model.train()
            train_loss = 0.0
            train_bar = tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs} [train]", unit="batch")
            for images, masks, _ in train_bar:
                images, masks = images.to(device), masks.to(device)
                optimizer.zero_grad()
                with torch.amp.autocast("cuda", enabled=use_amp):
                    logits = model(images)
                    loss = criterion(logits, masks)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                train_loss += loss.item() * images.size(0)
                train_bar.set_postfix(loss=f"{loss.item():.4f}")
            train_loss /= len(train_ds)

            model.eval()
            val_loss = 0.0
            agg = {"iou": 0.0, "dice": 0.0, "precision": 0.0, "recall": 0.0}
            val_bar = tqdm(val_loader, desc=f"epoch {epoch}/{args.epochs} [val]", unit="batch")
            with torch.no_grad(), torch.amp.autocast("cuda", enabled=use_amp):
                for images, masks, _ in val_bar:
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
