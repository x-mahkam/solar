# Solar Filament Segmentation Challenge 2026 — starter kit

A working PyTorch baseline for
[kaggle.com/competitions/filament-segmentation-2026](https://www.kaggle.com/competitions/filament-segmentation-2026)
(the IEEE Big Data Cup 2026 "Pixel-Precise Segmentation of Solar Filaments"
challenge, hosted by the Earth-Space AI Research (ESAIR) Lab).

The facts below are confirmed against the competition's own Overview/Data/
Rules pages (the user shared screenshots of these — this sandbox cannot
reach kaggle.com directly), not guessed.

## The competition, in short

- **Task**: from GONG H-alpha full-disk images, produce a segmentation mask
  for every individual solar filament — capturing fine structures (barbs),
  separating filament material from background noise, and keeping each
  filament as one coherent instance.
- **Data (MAGFiLO)**: `MAGFiLO_1.0_Kaggle_2026/train/train_images/*.jpeg` +
  `MAGFiLO_1.0_Annotations_kaggle2026_train.json` (COCO-format polygons),
  and `MAGFiLO_1.0_Kaggle_2026/test/test_images/*.jpeg` for the held-out
  test set. All images are 2048x2048. Categories double as chirality labels:
  `Left` / `Right` / `Unidentifiable` / `Ambiguous` (ids 1-4). License:
  CC BY-NC 4.0.
- **Metric**: **Panoptic Quality (PQ)** (Kirillov et al., CVPR 2019):
  `PQ = (sum of IoU over matched pred/GT pairs) / (|TP| + 0.5|FP| + 0.5|FN|)`,
  matched at IoU > 0.5. Implemented in `filament_seg/metrics.py:panoptic_quality`.
  Organizers also track Dice-score and IoU distributions, and one-to-many /
  many-to-one (fragmentation/over-merging) rates.
- **Submission**: one CSV for the whole test set, **one row per predicted
  filament instance** (not per image):
  ```
  filament_id,segmentation_rle
  20150125172714Mh_1,<pycocotools RLE counts, no quotes>
  20150125172714Mh_2,<pycocotools RLE counts, no quotes>
  ```
  `filament_id` is `<image filename stem>_<1-based index>`. `segmentation_rle`
  is the **pycocotools** RLE "counts" string (`mask.encode` on a Fortran-order
  binary mask) — *not* a plain numeric RLE. Size isn't stored per row (fixed
  2048x2048). `filament_seg/predict.py` already produces this format.
- **Scoring rubric** (partially seen): Qualitative Comparison is 30% (pipeline
  description, segmentation morphology on H-alpha images, code quality) —
  the rest is quantitative (PQ/Dice/IoU on the leaderboard), all **contingent
  on your source code being available** to the organizers.
- **Open-Access Policy**: to be eligible for final evaluation, your Git repo
  must be **publicly accessible** from the moment the competition closes
  through at least the winners' announcement. Code quality (modularity,
  reproducibility, documentation) factors into the score.
- **Final submission** (separate from the Kaggle leaderboard, due **Nov 15,
  2026**): a Google Form with (1) a 4-page technical report from the
  organizers' Overleaf template, and (2) your public repo link, containing a
  `requirements.txt` with pinned versions and a Jupyter notebook that runs
  the *entire* pipeline end-to-end with no extra files needed. **This repo
  currently has no such notebook — add one before the deadline** (see
  "Still needed" below).
- **Dates**: launched Jul 10, 2026 · final reports due **Nov 15, 2026** ·
  winners announced Nov 30, 2026 · optional IEEE BigData conference Dec
  14-17, 2026, Phoenix, AZ.
- **Prizes**: pool up to $3,000, split across top 3 teams proportional to
  score (`weight = score^2.0`, normalized). No Kaggle points/medals.
- **Integrity note**: the organizers posted a warning (Aug 20, 2026) about
  participants "hacking the system" instead of solving the actual problem,
  and said any PQ > 0.3 is already valuable — don't chase leaderboard hacks.

## What's here

```
filament_seg/
  coco_utils.py   # COCO polygon parsing -> raster masks (semantic + per-instance)
  dataset.py      # PyTorch Datasets (train/val with masks, test without)
  transforms.py   # flip/rotate augmentation (no albumentations dependency)
  model.py        # plain U-Net (single-channel in, works with no pretrained weights)
  losses.py       # BCE + soft-Dice
  metrics.py      # pixel IoU/Dice/precision/recall + instance AP/hit-miss + Panoptic Quality
  rle.py          # numeric RLE (local checks) + pycocotools COCO RLE (actual submission format)
  train.py        # training CLI (tqdm progress bars)
  predict.py      # inference CLI -> submission.csv (filament_id, segmentation_rle)
tests/test_smoke.py  # end-to-end test on synthetic data, no real data needed
scripts/download_data.md
```

Confirmed working against the real dataset (not just synthetic data): a
laptop CPU run (3 epochs, 256px) reached val IoU 0.41, and a Kaggle
Notebook GPU run (T4, 40 epochs, 512px) reached val IoU 0.51 in about an
hour. `tests/test_smoke.py` end-to-end tests the whole pipeline including
the real submission RLE format on synthetic data.

## Quickstart

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# sanity check the whole pipeline on fake data (no download needed)
python tests/test_smoke.py

# get the real data — see scripts/download_data.md
python -m filament_seg.train \
  --images-dir MAGFiLO_1.0_Kaggle_2026/train/train_images \
  --ann-file MAGFiLO_1.0_Kaggle_2026/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json \
  --output-dir runs/baseline \
  --epochs 40 --batch-size 16 --image-size 512

# predict on the test set -> submission.csv in the exact competition format
python -m filament_seg.predict \
  --images-dir MAGFiLO_1.0_Kaggle_2026/test/test_images \
  --checkpoint runs/baseline/best.pt \
  --output submission.csv
```

On Windows, add `--num-workers 0` to `train.py` (multiprocessing DataLoader
workers can hang/slow down badly there). On a CPU-only machine, expect
~35-40 min/epoch at these settings; a free Kaggle Notebook GPU (T4) does the
same epoch in ~80 seconds — see `scripts/download_data.md` for both paths.

`train.py` also writes `val_image_ids.txt` / `val_file_names.txt` to its
output dir (the held-out split it evaluated on each epoch), so you can
score that exact split locally:

```bash
# predict on just the held-out validation images
python -m filament_seg.predict \
  --images-dir MAGFiLO_1.0_Kaggle_2026/train/train_images \
  --checkpoint runs/baseline/best.pt \
  --file-list runs/baseline/val_file_names.txt \
  --output val_predictions.csv

# build a matching ground-truth CSV for that same split
python -m filament_seg.self_eval build-gt \
  --ann-file MAGFiLO_1.0_Kaggle_2026/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json \
  --image-ids-file runs/baseline/val_image_ids.txt \
  --output val_gt.csv

# score with the organizers' exact Panoptic Quality formula
python -m filament_seg.self_eval score --gt-csv val_gt.csv --pred-csv val_predictions.csv
```

`filament_seg/self_eval.py` is a faithful, tested port of the organizers'
own self-evaluation notebook (`kaggle.com/code/azimahmadzadeh/self-evaluation-notebook`)
-- same `filament_id`/`segmentation_rle` matching logic (including the
annotator-prefix stripping on the GT side) and the same pooled PQ formula,
verified against it to give PQ=1.0 for an exact match and PQ=0.0 for an
empty prediction.

## Pipeline notebook

`notebooks/pipeline.ipynb` is the actual Kaggle Notebook run (GPU, T4) that
produced these results: clones this repo, trains 40 epochs (val IoU 0.51),
predicts on the held-out validation split, and scores it with
`filament_seg.self_eval` against the organizers' own Panoptic Quality
formula:

**Panoptic Quality: 0.3386** on the local validation split (1232 ground-truth
filaments vs. 1759 predicted) -- above the organizers' own stated bar ("any
PQ score greater than 0.3 is of great value to us", Aug 20 2026 announcement).
Kept as-run (including the mid-session recovery cells after a `git pull`)
rather than cleaned up, so it's an honest record of what actually produced
the checkpoint and the score above.

## Still needed before the Nov 15 deadline

This starter kit gets you a working, correctly-formatted baseline with a
locally-verified PQ score of 0.3386 and a pipeline notebook in the repo. It
does **not** yet fully satisfy the competition's *Final Submission*
requirements:

- [ ] **Make the GitHub repo public** before the competition closes (Open-Access
  Policy) — currently unverified from this session.
- [x] ~~Add a Jupyter notebook that runs the complete pipeline~~ -- done:
  `notebooks/pipeline.ipynb`.
- [ ] **Write the 4-page technical report** from the organizers' Overleaf
  template, describing preprocessing, architecture, training, post-processing
  (this is 30% of the score — "Qualitative Comparison" explicitly asks for
  this description plus code quality and segmentation morphology).
- [ ] **Submit the Google Form** with the report and repo link by Nov 15, 2026.
- [x] ~~Run the organizers' self-evaluation notebook against your predictions~~
  -- done: `filament_seg/self_eval.py` reproduces it locally (still worth a
  one-time cross-check against the actual notebook before the deadline).

## Where to take it from here

- **Model upgrades**: the U-Net here is intentionally minimal. The
  [EdgeAttNet](https://github.com/dasjar/EdgeAttNet) paper from the same
  group that built MAGFiLO reports mIoU 0.645 (pairwise) / 0.703
  (multi-scale) using a U-Net + edge-guided self-attention bottleneck aimed
  specifically at capturing thin barbs — a natural next step once the plain
  U-Net baseline is solid.
- **Class imbalance**: filaments are thin structures on a mostly-background
  disk; `BCEDiceLoss` is a reasonable start, but also worth trying: Tversky
  loss (penalize false negatives harder), boundary/edge losses, or heavier
  positive-pixel weighting.
- **Instance separation**: the current pipeline predicts one semantic mask
  and splits it into instances via connected components at inference time —
  fine as a baseline, but two filaments that touch will get merged into one
  predicted instance, hurting PQ (counted as fragmentation/over-merging by
  the organizers' rubric too). An instance-segmentation head (Mask
  R-CNN-style) or a learned watershed/energy map would address this.
- **Resolution**: full-disk frames are 2048x2048 and filaments/barbs are
  thin; a single resize to 512 (the default) loses a lot of fine detail —
  worth trying a larger `--image-size` (memory permitting) or tiling into
  crops instead of one global resize.

## Pushing PQ higher: `tune_postprocess.py`

The 0.3386 PQ run had 1759 predicted filaments against 1232 ground-truth --
noticeable over-segmentation, likely fixable without retraining. Before
spending GPU time on a bigger model, sweep the post-processing operating
point (probability threshold, minimum component size, morphological
closing to merge nearby fragments) against the *already-trained* checkpoint:

```bash
python -m filament_seg.tune_postprocess \
  --images-dir MAGFiLO_1.0_Kaggle_2026/train/train_images \
  --ann-file MAGFiLO_1.0_Kaggle_2026/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json \
  --checkpoint runs/baseline/best.pt \
  --file-list runs/baseline/val_file_names.txt \
  --image-ids-file runs/baseline/val_image_ids.txt
```

This runs one forward pass per validation image (reusing the cached
probability maps), then cheaply re-binarizes/re-labels/re-scores every
(threshold, min-size, closing-size) combination with the real PQ formula,
and writes a sorted `tune_results.csv`. Whatever combination wins, pass the
same `--threshold` / `--min-component-size` / `--closing-size` to
`predict.py` when generating the real submission.

Rough order of expected impact for further PQ gains, cheapest first:
1. Post-processing tuning above (no retrain).
2. **Crop-based training + tiled inference** (see below) -- addresses the
   root cause: barbs are only a few pixels wide at 2048x2048 and a global
   resize to 512-768px destroys them outright, which is why bumping
   `--image-size` alone barely moved val IoU (0.5118 -> 0.5236). Training on
   native-resolution crops keeps every barb pixel-exact.
3. A loss suited to thin structures under class imbalance (`--loss tversky`)
   instead of plain BCE+Dice.
4. Architecture change -- EdgeAttNet-style edge-guided attention (see
   "Model upgrades" above); this is the one that requires real engineering
   work, not just hyperparameters.

### Crop-based training + tiled inference

Train on native-resolution random crops instead of a whole-image resize, and
predict with a matching sliding-window (tiled) pass so the full 2048x2048
image is covered without ever downsampling it:

```bash
python -m filament_seg.train \
  --images-dir MAGFiLO_1.0_Kaggle_2026/train/train_images \
  --ann-file MAGFiLO_1.0_Kaggle_2026/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json \
  --output-dir runs/baseline \
  --epochs 50 --batch-size 16 --crop-size 512 --base-ch 32 \
  --loss tversky --tversky-alpha 0.3 --tversky-beta 0.7

python -m filament_seg.predict \
  --images-dir MAGFiLO_1.0_Kaggle_2026/test/test_images \
  --checkpoint runs/baseline/best.pt \
  --tile-size 512 --tile-overlap 64 --tta --base-ch 32 \
  --output submission.csv
```

Notes:
- **Positive-biased crop sampling** (`--positive-crop-prob`, default 0.8):
  filaments cover a small fraction of a 2048x2048 frame, so purely random
  512x512 crops mostly land on empty background -- verified on synthetic
  data at the same sparsity as the real dataset, only 35.5% of pure-random
  crops contained any filament pixels at all, vs. 88.5% once crops are
  centered (with jitter) on a randomly chosen filament most of the time.
  The remaining crops stay fully random so the model still sees
  pure-background negatives.
- AMP (mixed precision) is on automatically on CUDA (`--no-amp` to disable);
  it roughly halves memory use and speeds up training, which is what makes a
  larger `--batch-size` viable at native resolution without OOM (unlike the
  earlier whole-image `--image-size 768` run, which had to drop to
  `--batch-size 4` and still barely fit in 15GB).
- `--tile-overlap` should be generous enough that no filament falls entirely
  within the discarded seam between tiles; 64-128px at `--tile-size 512` is
  a reasonable start.
- `filament_seg/tune_postprocess.py` also accepts `--tile-size`/`--tile-overlap`/`--tta`
  so the threshold/min-size/closing sweep matches how `predict.py` will
  actually run.
- `RandomBrightnessContrast` was added to the default training augmentations
  (`transforms.py`) to make the model robust to the background-noise/
  exposure variation the competition's own "Main Challenges" section calls
  out -- aimed at closing the gap between local validation PQ and the real
  (lower) leaderboard score.

### Attention gates + touching-instance separation

Two more additions, usable together with crop/tiled training above:

- **`--use-attention`** (train.py/predict.py/tune_postprocess.py, must match
  between training and inference): adds Attention U-Net-style gates on the
  skip connections (Oktay et al. 2018, arxiv.org/abs/1804.03999) --
  each gate learns to weight encoder features by relevance to the decoder's
  context before concatenating them, which tends to help the model focus on
  small/thin structures against background clutter. This is a well-documented
  stand-in for EdgeAttNet's own attention mechanism, whose exact
  implementation isn't public.
- **`--erosion-size`** (predict.py/tune_postprocess.py, no retraining
  needed): splits filaments that touch into separate predicted instances --
  erode the mask first (breaks thin bridges between adjacent filaments),
  label the eroded mask, then grow each label back out to its original
  extent. Verified on a synthetic case (two 10x10 blobs joined by a 1px
  bridge): plain connected-components merges them into 1 instance;
  `separate_touching_instances` with `erosion_size=3` correctly recovers 2,
  each at their full original size. Sweep it alongside threshold/min-size/
  closing in `tune_postprocess.py --erosion-sizes`.

## Integrity note

While setting this up, a *different, unrelated* Kaggle dataset containing
another participant's full solution (code, technical report, "rules
compliance" docs) was briefly attached to a Notebook by mistake while
searching for the official competition data. **None of that code was used
here** — everything in this repo was written from scratch against the
official rules. Do not reuse another team's submission; the organizers have
explicitly flagged leaderboard-gaming behavior as a concern.
