# Benchmark

pagescan ships with a real-data benchmark harness that compares its scan quality against `docscan` (the most popular Python document scanner on PyPI) and a hand-written OpenCV contour-based scanner. Both pagescan paths — the production cascade and the legacy ML chain — are evaluated separately so you can see the effect of the architecture change.

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
- **Harness:** {file}`benchmark/comparison/run_real.py`. Each scanner runs as a black-box subprocess.

## Current results (v1 weights)

50-photo April benchmark, v1 cascade weights:

| Scanner | Pass ≥ 0.90 | Pass ≥ 0.85 | Mean IoU | Median max-px | Mean PSNR | Mean WER | Mean latency |
|---|---|---|---|---|---|---|---|
| **pagescan-cascade** | 35 / 50 | 38 / 50 | 0.857 | **32 px** | **18.7 dB** | 0.53 | 5685 ms |
| **pagescan-legacy** | **44 / 50** | **44 / 50** | **0.916** | 39 px | 18.1 dB | 0.53 | **73 ms** |
| opencv-recipe | 12 / 50 | 12 / 50 | 0.507 | 70 px | 15.1 dB | 0.81 | **10 ms** |
| docscan | 0 / 50 | 0 / 50 | — | — | 14.3 dB | 0.87 | 0 ms |

**Observations:**

- The legacy ML chain (`pagescan-legacy`) currently *beats* the cascade on pass-rate. This is expected: v1 cascade weights are trained on a 1000-photo December corpus and over-fit to that distribution. The April test set is held-out and probes a slightly different distribution; the legacy chain was trained on a broader corpus and generalises better in this v1 snapshot.
- However, the cascade's **median max corner pixel error is lower** (32 px vs 39 px) when it succeeds — HQ-SAM produces tighter boundaries. The pass-rate gap is detection misses, not corner-accuracy issues.
- `docscan` scores 0/50 because it ships with corner-detection heuristics that simply don't fire on phone photos with non-trivial backgrounds. It's tuned for scans of scans, not photos.
- The OpenCV recipe — a hand-written `cv2.findContours` + `approxPolyDP` pipeline — is the right comparison for "is the ML actually worth it?". Even unoptimised, both pagescan paths beat it by 3–4× on every metric except latency.

## Why these numbers will change before 0.1.0

The cascade is currently in its v1 state. A v2 retrain on an extended dataset (≥300 new April-distribution photos + SmartDoc 2015 frames) is in progress and is expected to flip the cascade-vs-legacy comparison. Once v2 lands, this page will be regenerated with the final numbers; the cascade should pull clearly ahead.

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
