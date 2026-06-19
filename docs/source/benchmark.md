# Benchmark

pagescan ships with a real-data benchmark harness that compares its scan quality against `docscan` (the most popular Python document scanner on PyPI) and a hand-written OpenCV contour-based scanner. Both pagescan detection paths are evaluated separately: the **legacy ML chain** (SA24 + LCNet), which is the default in 0.1.0, and the **YOLO + HQ-SAM cascade**, which is opt-in (`ScanConfig(use_cascade=True)`) on the current weights. The numbers below are why legacy is the default — see [the use_cascade decision](#use-cascade-default).

## Methodology

- **Test set:** 50 photos of real documents, shot with consumer phones on varied backgrounds (wood tables, white desks, magazines, hands holding the page). Held out from training data.
- **Ground truth:** four-corner annotations per image, plus a clean reference scan for OCR comparison.
- **Metrics:**
  - **Pass rate @ IoU ≥ 0.90** — strict bar; the corners must be tight to the document.
  - **Pass rate @ IoU ≥ 0.85** — accepts mild over/under-crop.
  - **Mean IoU** across the set.
  - **Median max corner pixel error** — for the corners predicted, how far off (in pixels) is the worst one.
  - **Mean PSNR** vs the reference scan — pixel-level reconstruction quality.
  - **Mean WER (word error rate)** of Tesseract output vs the reference — does the final scan still OCR well?
  - **Mean latency** per image on CPU.
- **Harness:** {file}`benchmark/comparison/run_real.py`. Each scanner runs in-process through a common adapter (`benchmark/comparison/scanners.py`).
- **Reference for PSNR/WER:** the image perspective-warped with the *ground-truth* corners — a fair upper bound, not a separate hand-scanned capture.

## Current results (v1 weights)

50-photo April benchmark, v1 weights. Default path listed first.

| Scanner | Pass ≥ 0.90 | Pass ≥ 0.85 | Mean IoU | Median max-px | Mean PSNR | Mean WER | Mean latency |
|---|---|---|---|---|---|---|---|
| **pagescan-legacy** (default) | **44 / 50** | **44 / 50** | **0.916** | 39 px | 18.1 dB | 0.53 | **62 ms** |
| pagescan-cascade (opt-in) | 35 / 50 | 38 / 50 | 0.857 | **32 px** | **18.7 dB** | 0.53 | 5044 ms |
| opencv-recipe | 12 / 50 | 12 / 50 | 0.507 | 70 px | 15.1 dB | 0.81 | **10 ms** |
| docscan | 0 / 50 | 0 / 50 | — | — | 14.3 dB | 0.87 | 0 ms |

(`docscan` exposes no corners, so its IoU columns are blank; it is scored on output PSNR/WER only.)

(use-cascade-default)=

**Observations — why legacy is the 0.1.0 default:**

- On the current v1 weights `pagescan-legacy` **wins on pass-rate** (44/50 vs 35/50 at IoU ≥ 0.90), is **~80× faster** (62 ms vs 5 s on CPU), and — critically — **fails less catastrophically**: its mean max-corner error is 133 px vs the cascade's 294 px. When the cascade's YOLO bbox misses on this held-out distribution, HQ-SAM segments the wrong region and the error is large. So pagescan defaults to legacy (`use_cascade=False`).
- The cascade has the **higher ceiling**, though: better median max-corner error (32 px vs 39 px), more near-perfect detections (pass ≥ 0.95: 28 vs 24), and higher PSNR. When its detector fires correctly, HQ-SAM gives the tightest corners of any path. Enable it with `ScanConfig(use_cascade=True)` (requires the `[ml]` extras). The gap is detector misses, not segmentation quality.
- `docscan` scores 0/50 — its corner heuristics don't fire on phone photos with non-trivial backgrounds; it's tuned for scans of scans. The OpenCV contour recipe manages 12/50. Both pagescan paths beat these baselines by a wide margin.

## Roadmap: the cascade becomes the default after v2

The cascade is in its v1 state, trained on a 1000-photo December corpus that under-covers the April distribution. A v2 retrain on an extended dataset (≥300 new April-distribution photos + SmartDoc 2015 frames) is the planned path to flip the comparison; the [ceiling benchmark](#ceiling-benchmark) below shows the segmentation is already good enough, so v2 is purely a detector-data effort. When v2 lands and beats legacy on this set, `use_cascade` flips back to `True` and this page is regenerated. You can re-run the head-to-head any time with {file}`benchmark/comparison/sweep_cascade.py`.

If you want to reproduce these numbers locally:

```bash
git clone https://github.com/7RPlus-GmbH/pagescan.git
cd pagescan
pip install -e ".[dev,ml]"
python -u benchmark/comparison/run_real.py
# Results: benchmark/comparison/results/{summary.json, table.md}
```

The `-u` flag is important — without it, the benchmark's stdout buffers and you'll see no progress until completion (~10 min on a modern laptop CPU).

## Ceiling benchmark

A separate harness (`benchmark/ceiling/`) measures the *upper bound* on cascade quality by giving the segmenter ground-truth bounding boxes instead of YOLO predictions. On the same 50-photo set, the ceiling hits **50 / 50 at IoU ≥ 0.80** — confirming that HQ-SAM's segmentation quality is sufficient, and the remaining gap to perfect performance is entirely on the detector side. That's why v2 work focuses on detector training, not segmentation.
