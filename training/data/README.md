# `training/data/` — local working data (gitignored)

This whole directory is **gitignored**. It holds working-state data — photos
and built datasets too large to live in git. Labels themselves are tracked
under `training/labels_3k/` (Dec) and `training/labels_april/` (April test set).

Treat anything in here as reproducible scratch: if it gets deleted, you can
rebuild it from the labels in git plus the photo source (currently your
local copy; eventually an HF Hub dataset per PLAN.md week 1).

## Current contents

```
training/data/
├── dec_1000_images/   1000 Dec phone photos (~198 MB), source for YOLO training.
│                      Labels live in training/labels_3k/ (tracked in git).
└── yolo_build/        Built YOLO-detect dataset (~8 MB on disk; images are
                       symlinks into dec_1000_images/). Rebuild any time via:
                           python training/yolo/build_dataset.py
```

## Future contents (PLAN.md weeks 1–2)

```
training/data/
├── dec_1000_images/      (existing)
├── april_v2_images/      ≥300 new April-distribution photos (your data sprint)
├── smartdoc_2015/        ~25k frames + quad annotations from SmartDoc
└── yolo_build/           rebuilt against the combined source pool
```

## Why gitignored

- Binary photos bloat the repo and clones.
- Built datasets are reproducible via `build_dataset.py`.
- Long-term home for raw data is HF Hub (tracked in PLAN.md week 1, not done yet).

## Restoring after a clone

If you clone the repo and need to train, you have to provide the photos
locally — they're not in git. Until the HF Hub dataset is published, ask
Alexander for the source bundle, or re-photograph and re-label.
