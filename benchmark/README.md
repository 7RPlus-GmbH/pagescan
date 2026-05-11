# benchmark/

Evaluation harness for pagescan. Three things live here:

## `cascade/` — production benchmark

The canonical evaluation: YOLO11n → HQ-SAM ViT-B → quad fit on the 50-photo April hold-out.

```bash
python benchmark/cascade/run.py
```

Reads images from `benchmark/comparison/images/` and labels from `training/labels_april/`, writes overlays to `benchmark/cascade/out/` and `summary.json`. Reports IoU, per-corner pixel error, and best-cyclic-rotation alignment metrics.

This is what we measure ourselves against. Acceptance gate: ≥45/50 at IoU ≥ 0.90 before 0.1.0 ships.

## `ceiling/` — architectural upper bound

Same cascade with the YOLO replaced by ground-truth bbox. Tells us what the cascade would achieve with a perfect detector. Used during development to attribute failures to detector vs. segmenter vs. quad-fitting.

```bash
python benchmark/ceiling/run.py
```

Result on the current April benchmark: **50/50 at IoU ≥ 0.80, 44/50 at IoU ≥ 0.90, mean IoU 0.966**. The architecture is sound; the open work is making the trained detector match this ceiling.

## `comparison/` — competitive comparison (in progress)

`run_comparison.py` runs pagescan vs `docscan` vs the OpenCV contour scanner on the same photo set. **Currently measures only `success` (= "didn't throw") and self-reported `quality_score`** — both insufficient for a public release. Replaced by a real IoU + OCR-WER + SSIM benchmark in week 4 of [PLAN.md](../PLAN.md).

`colab_benchmark.py` runs against SmartDoc 2015 on Colab. Pre-existing; left as-is for SmartDoc-specific runs.

## What's *not* here

- **Old experimental benchmarks** (MobileSAM standalone, HQ-SAM standalone, MobileSAM cascade, HRNet) live under `_archive/benchmark/`. They're kept for reference; they don't ship.
