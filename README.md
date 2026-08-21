<picture>
  <source media="(prefers-color-scheme: dark)"
    srcset="https://raw.githubusercontent.com/7RPlus-GmbH/pagescan/main/brand/banners/readme-header-dark-1280x320.png">
  <img alt="pagescan" width="640"
    src="https://raw.githubusercontent.com/7RPlus-GmbH/pagescan/main/brand/banners/readme-header-light-1280x320.png">
</picture>

[![Status](https://img.shields.io/badge/status-in%20development-d97706.svg)](https://pagescan.7rplus.com)
[![Python](https://img.shields.io/badge/python-3.9%20%E2%80%93%203.13-blue.svg)](pyproject.toml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Weights: Apache-2.0](https://img.shields.io/badge/weights-Apache--2.0-blue.svg)](https://huggingface.co/7rplus/pagescan-weights)

> ⚠️ **Pre-release.** pagescan is on the path to its first public release (`0.1.0`). The README below describes the *current* state and the *target* release.

A privacy-first document scanner for Python. Phone photo in, deskewed print-ready PDF out — without uploading anything to a cloud.

Built for batch / server-side / on-prem use cases where Apple's VisionKit and Google's ML Kit aren't an option: regulated environments, EU data-residency requirements, headless Linux, scriptable pipelines.

## Installation

Not on PyPI yet — install from source:

```bash
pip install git+https://github.com/7RPlus-GmbH/pagescan
```

The pre-trained models (~50 MB) download from Hugging Face Hub on first use and cache locally.

**Optional system dependency** for orientation cross-check: [Tesseract OCR](https://github.com/tesseract-ocr/tesseract).

```bash
sudo apt install tesseract-ocr tesseract-ocr-deu   # Ubuntu/Debian (+ a language pack)
brew install tesseract                              # macOS (bundles many languages)
```

Install one language pack per document language you scan, e.g. `tesseract-ocr-fra`,
`tesseract-ocr-spa`, `tesseract-ocr-ita`, `tesseract-ocr-jpn`, then set
`ScanConfig.ocr_lang` accordingly (default `"eng+deu+fra+spa+ita+jpn"`). pagescan
uses whichever requested packs are actually installed and ignores the rest.

Without Tesseract, pagescan still works — orientation falls back to a CNN-only heuristic.

## Quick start

```python
import pagescan

# Scan a single document
result = pagescan.scan("photo.jpg", "output.pdf")

# Batch
pagescan.scan_batch("input/", "output/")

# Custom config
from pagescan import ScanConfig
config = ScanConfig(auto_orient=True, enhance=True, jpeg_quality=60)
pagescan.scan("photo.jpg", "out.pdf", config=config)
```

CLI:

```bash
pagescan photo.jpg output.pdf
pagescan --batch --input-dir photos/ --output-dir scans/ --workers 4
pagescan photo.jpg --raw     # crop + perspective only, no enhancement
```

## How it works

1. **Corner detection** — FastViT-SA24 heatmap regression finds the four corners,
   with LCNet100 as a second backbone when it fails. This is what `ScanConfig()`
   runs today.
2. **Quad fit and repair** — sanity checks on the predicted quad.
3. **Orientation** — small CNN classifier + (optional) Tesseract OCR cross-check.
4. **Perspective transform** — original document aspect ratio preserved (no forced A4 stretch).
5. **Enhancement** — shadow removal, white balance, contrast stretch, unsharp mask.
6. **PDF output** — A4 canvas @ 300 DPI by default; configurable.

A traditional contour-based fallback runs if corner detection finds nothing.

**The YOLO11n → HQ-SAM cascade is opt-in**, not the default. On the current
weights the legacy chain wins on the held-out benchmark (44/50 vs 35/50 at
IoU ≥ 0.90), runs ~75× faster and fails less catastrophically, so
`use_cascade` defaults to `False`. Enable it with `ScanConfig(use_cascade=True)`
and the `[ml]` extras. See [the benchmark](https://7rplus-gmbh.github.io/pagescan/benchmark.html)
for the numbers behind that choice.

## Benchmarks

The 0.1.0 release will include published numbers vs `docscan` and the OpenCV contour scanner on a 50-photo benchmark of real phone shots, measuring detection IoU, corner pixel error, OCR word-error rate, and end-to-end visual quality.

Until then, the architecture has been validated to hit **50/50 at IoU ≥ 0.80** on the held-out April benchmark with a perfect detector (`benchmark/ceiling/`); the trained-detector cascade currently sits below that ceiling and is the active focus of pre-release work.

## Configuration

All parameters live on `ScanConfig` (dataclass). Common knobs:

```python
from pagescan import ScanConfig

# Different background (e.g. blue tablecloth)
ScanConfig(background_hsv_low=(100, 50, 30), background_hsv_high=(130, 255, 255))

# US Letter instead of A4
ScanConfig(output_width=2550, output_height=3300)

# High quality
ScanConfig(jpeg_quality=85, output_dpi=300)
```

See `src/pagescan/config.py` for the full surface.

## Why pagescan

- **No cloud round-trip.** Everything runs locally; nothing leaves the machine.
- **Headless / scriptable.** No GUI, no mobile dependency, batch-friendly.
- **Open weights.** Models are MIT-compatible and hosted under our HF org; you can audit, fine-tune, and redistribute.
- **Tuned for real-world phone photos.** Wood tables, uneven lighting, hand occlusion, perspective tilt.
- **Built for regulated environments.** EU data-residency, no third-party cloud calls during inference.

If you need a mobile SDK, Apple's VisionKit and Google's ML Kit are excellent. If you need server/desktop/headless and care about data residency, pagescan is built for that case.

## Project status

pagescan is developed by [7R+ GmbH](https://7rplus.com) — same team behind [xaitalk](https://xaitalk.com).

Site: <https://pagescan.7rplus.com>
Documentation: <https://7rplus-gmbh.github.io/pagescan/> (auto-deployed from `main`).

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). The most useful contribution right now is **labeled real-world document photos** in our failure distributions (magazines, white-on-cream, hand-occluded scenes).

## License

MIT — see [LICENSE](LICENSE). Copyright © 2026 7R+ GmbH.
