# Solar Filament Segmentation Challenge 2026 — starter kit

A working PyTorch baseline for
[kaggle.com/competitions/filament-segmentation-2026](https://www.kaggle.com/competitions/filament-segmentation-2026)
(mirrored at the [IEEE Big Data Cup 2026](https://bigdataieee.org/BigData2026/cup/solar-filament-segmentation/)).

## The competition, in short

- **Task**: pixel-precise segmentation of solar filaments in full-disk GONG
  H-alpha images — including thin barbs, faint material, and separating each
  filament as one coherent object.
- **Data**: [MAGFiLO](https://www.nature.com/articles/s41597-024-03876-y), the
  largest gold-standard filament dataset — 10,244 manually annotated
  filaments across 1,593 observations (2011-2022), with polygon masks,
  spines, bounding boxes, and chirality labels ('left' / 'right' /
  'unidentifiable'), shipped as a COCO-format JSON.
- **Metrics**: IoU, precision, recall, AP@IoU, hit/miss rate, and the
  organizers' own multi-scale IoU (MIoU) for fine structures.
- **Notes**: any method is allowed (classical CV to deep nets); it's a code
  competition — source code is required from top teams; prizes include cash
  and/or IEEE Big Data 2026 registration.

I could not reach `kaggle.com` from this sandbox (network egress is
blocked here), so I wasn't able to confirm the *exact* file layout, submission
CSV columns, or deadline first-hand — the above is pieced together from the
IEEE mirror, the MAGFiLO paper, and the [EdgeAttNet paper/code](https://arxiv.org/abs/2509.02964)
(same research group's own baseline model for this exact dataset). See
`scripts/download_data.md` before your first real run — you'll need to
download the data yourself and double check paths/columns against what
Kaggle actually gives you.

## What's here

```
filament_seg/
  coco_utils.py   # COCO polygon parsing -> raster masks (semantic + per-instance)
  dataset.py      # PyTorch Datasets (train/val with masks, test without)
  transforms.py   # flip/rotate augmentation (no albumentations dependency)
  model.py        # plain U-Net (single-channel in, works with no pretrained weights)
  losses.py       # BCE + soft-Dice
  metrics.py      # pixel IoU/Dice/precision/recall + instance-level AP@IoU/hit-miss
  rle.py          # Kaggle-style RLE encode/decode
  train.py        # training CLI
  predict.py      # inference CLI -> submission.csv
tests/test_smoke.py  # end-to-end test on synthetic data, no real data needed
scripts/download_data.md
```

Everything has been smoke-tested end to end (dataset loading, a training
step, metrics, RLE round-trip, and both CLIs) against synthetic data — see
`tests/test_smoke.py`. It has **not** been run against the real competition
data, since that requires a Kaggle account/API token this environment can't
use.

## Quickstart

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# sanity check the whole pipeline on fake data (no download needed)
python tests/test_smoke.py

# get the real data — see scripts/download_data.md
# then train (confirmed layout: images flat under data/train/, e.g. data/train/20110109104734Ch.jpeg):
python -m filament_seg.train \
  --images-dir data/train \
  --ann-file data/train_annotations.json \
  --output-dir runs/baseline \
  --epochs 40 --batch-size 8 --image-size 512

# predict on the test set (path depends on how the test split is shipped):
python -m filament_seg.predict \
  --images-dir data/test \
  --checkpoint runs/baseline/best.pt \
  --output submission.csv
```

`--instance-level` on `predict.py` switches from one RLE row per image
(union of all filaments) to one row per predicted connected component —
check the real `sample_submission.csv` once you have it to see which shape
Kaggle expects, and adjust the column names in `predict.py` if needed.

## Where to take it from here

- **Confirmed against the real download**: images are `.jpeg`, 2048x2048,
  flat under `data/train/` (matches COCO `file_name` directly, no FITS
  conversion needed). Chirality is the category itself — categories are
  `Left` / `Right` / `Unidentifiable` / `Ambiguous` (id 1-4) — `coco_utils.chirality_of`
  already resolves this via `category_id`.
- **Still worth checking**: full-disk frames are 2048x2048 and filaments are
  thin; a single resize to 512 (the default) will shrink barbs a lot —
  worth trying a larger `--image-size` (e.g. 768-1024, memory permitting) or
  tiling into crops instead of one global resize once the baseline is
  working end-to-end. Also check how the test split is shipped (folder name,
  and whether ground truth is withheld or a `sample_submission.csv` is
  provided) before running `predict.py`.
- **Model upgrades**: the U-Net here is intentionally minimal. The
  [EdgeAttNet](https://github.com/dasjar/EdgeAttNet) paper from the same
  group that built MAGFiLO reports mIoU 0.645 (pairwise) / 0.703
  (multi-scale) using a U-Net + edge-guided self-attention bottleneck aimed
  specifically at capturing thin barbs — a natural next step once the plain
  U-Net baseline is working end-to-end on real data.
- **Class imbalance**: filaments are thin structures on a mostly-background
  disk; `BCEDiceLoss` is a reasonable start, but also worth trying: Tversky
  loss (penalize false negatives harder), boundary/edge losses, or heavier
  positive-pixel weighting.
- **Instance separation**: the current pipeline predicts one semantic mask
  and splits it into instances via connected components at inference time —
  fine as a baseline, but two filaments that touch will get merged into one
  predicted instance. An instance-segmentation head (Mask R-CNN-style) or
  a learned watershed/energy map would fix that if AP@IoU matters a lot for
  scoring.
