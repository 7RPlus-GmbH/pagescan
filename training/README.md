# training/

Training pipelines for the models pagescan ships.

## Layout

```
training/
├── README.md           (this file)
├── yolo/               canonical training pipeline (YOLO11n doc detector)
├── labels_3k/          1000 YOLO-OBB labels for Dec photos (tracked)
├── labels_april/       50 YOLO-OBB labels for April hold-out (tracked)
└── data/               local working data (GITIGNORED)
    ├── dec_1000_images/    198 MB — the 1000 Dec photos
    └── yolo_build/         built YOLO dataset, regen via build_dataset.py
```

Labels are small and live in git. Photos and built datasets are too large
and live under `training/data/` which is gitignored. See
`training/data/README.md` for restore instructions.

## What's active

### `yolo/` — document detector (canonical)

YOLO11n trained as a single-class document bbox detector. Output is the box prompt fed to HQ-SAM in the production cascade.

```
yolo/
├── README.md           # Full instructions
├── build_dataset.py    # OBB labels → YOLO-detect bbox dataset
└── train_colab.py      # Colab training script (Ultralytics)
```

See `yolo/README.md` for the full prep + train + export workflow.

### `labels_3k/` — Dec 2025 real photos

1000 labels for `training/data/dec_1000_images/*.jpg`, YOLO-OBB format
(`class x1 y1 x2 y2 x3 y3 x4 y4`, normalized). Mostly white paper on dark
table, well-lit, axis-aligned.

Used for the v1 YOLO training. Insufficient on its own — the April
benchmark exposes a Dec→April distribution shift (different document types
and backgrounds).

### `labels_april/` — held-out 50

Labels for the 50 April 2026 photos in `benchmark/comparison/images/`.
**This is the test set.** Never train on these. The cascade's reported
performance on these images is the only number that goes in the README.

## What's not active

- HRNet-era training (`_archive/training/train_heatmap_colab.py`,
  `_archive/training/train_hrnet_multitask.py`,
  `_archive/training/scripts/`). Failed approach; capped at ~9–10/50 on
  April due to data distribution.
- The annotation tool (`_archive/training/scripts/annotate.py`) may still
  be useful for the Week-1 labeling sprint; lift it out if needed.

## What's missing

- **More labeled real photos in failure distributions.** See [PLAN.md](../PLAN.md) week 1.
- **SmartDoc 2015 integration** (~25k frames, public, quad-annotated). Week 1.
- **Combined dataset prep** that mixes Dec + new April-style + SmartDoc with
  proper sampling. Week 2.
