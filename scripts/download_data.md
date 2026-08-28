# Getting the competition data

This sandbox's network egress does not allow reaching `kaggle.com`, so the
data has to be pulled down wherever you actually train (your own machine,
Colab, or a Kaggle Notebook). Steps:

1. Join the competition at
   https://www.kaggle.com/competitions/filament-segmentation-2026 and accept
   its rules (required before the API will let you download anything).
2. Get an API token: Kaggle account settings -> "Create New Token", which
   downloads `kaggle.json`.
3. Install the CLI and place the token:
   ```bash
   pip install kaggle
   mkdir -p ~/.kaggle
   cp /path/to/kaggle.json ~/.kaggle/
   chmod 600 ~/.kaggle/kaggle.json
   ```
4. Download and unzip into `data/`:
   ```bash
   kaggle competitions download -c filament-segmentation-2026 -p data/
   unzip -q data/filament-segmentation-2026.zip -d data/
   ```
5. Check the actual folder/file names Kaggle gives you (they may not exactly
   match the `data/train/images` + `data/train_annotations.json` layout this
   repo assumes) and pass the real paths to `--images-dir` / `--ann-file`.
6. Copy the exact column names from the real `sample_submission.csv` into
   `filament_seg/predict.py` if they differ from `ImageId,EncodedPixels`.

If running on Kaggle Notebooks instead, the competition data is simply
mounted at `/kaggle/input/filament-segmentation-2026/` and step 1-4 are
unnecessary -- just point `--images-dir` / `--ann-file` there.

## Background reading

- Competition mirror: https://bigdataieee.org/BigData2026/cup/solar-filament-segmentation/
- MAGFiLO dataset paper: https://www.nature.com/articles/s41597-024-03876-y
- EdgeAttNet (same research group's baseline model + code):
  https://arxiv.org/abs/2509.02964 / https://github.com/dasjar/EdgeAttNet
