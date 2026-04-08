# pagescan

[![Python 3.9-3.13](https://img.shields.io/badge/python-3.9--3.13-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![PyPI version](https://img.shields.io/pypi/v/pagescan.svg)](https://pypi.org/project/pagescan/)
[![Tests](https://img.shields.io/badge/tests-30%20passed-brightgreen.svg)](#)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.5%2B-red.svg)](https://opencv.org/)
[![Tesseract](https://img.shields.io/badge/Tesseract-OCR-orange.svg)](https://github.com/tesseract-ocr/tesseract)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Turn phone photos of documents into clean, deskewed, print-ready PDFs.

Built for the real world: documents photographed on wood tables, under uneven lighting, at odd angles. Handles corner detection, perspective correction, shadow removal, deskew, auto-rotation, and scan-like enhancement in a single pipeline.

## Installation

```bash
pip install pagescan
```

Includes ML corner detection (DocAligner) and OCR-based auto-rotation (Tesseract) out of the box.

**Note:** Tesseract must be installed separately on your system (`apt install tesseract-ocr` / `brew install tesseract`).

## Quick Start

### Python API

```python
import pagescan

# Scan a single document
result = pagescan.scan("photo.jpg", "output.pdf")

# Batch process a directory
pagescan.scan_batch("input_photos/", "scanned_pdfs/")

# Custom configuration
from pagescan import ScanConfig

config = ScanConfig(
    auto_orient=True,     # OCR-based rotation correction
    enhance=True,         # Scan-like grayscale output
    jpeg_quality=60,      # Higher quality PDF output
)
result = pagescan.scan("photo.jpg", "output.pdf", config=config)
```

### Command Line

```bash
# Single file
pagescan photo.jpg output.pdf

# Batch processing
pagescan --batch --input-dir photos/ --output-dir scans/ --workers 4

# Fast mode (no ML, no OCR rotation)
pagescan photo.jpg --no-ml --no-rotate

# Keep color (no grayscale enhancement)
pagescan photo.jpg --no-enhance

# Raw output (just crop + perspective, no enhancement)
pagescan photo.jpg --raw
```

## What it does

1. **Corner detection** - ML-based (DocAligner) with geometric validation: parallelism check, aspect ratio filter, and automatic repair of misplaced corners. Falls back to contour-based detection.
2. **Perspective transform** - Preserves original document aspect ratio (no forced A4 stretching).
3. **Edge trimming** - Two-phase background removal: strip-scan + column density analysis to handle corner triangles from perspective tilt.
4. **Auto-rotation** - Tesseract OSD across 4 orientations with OCR word-count validation to pick the correct direction.
5. **Deskew** - Hough line detection on center 60% (avoids background edges), median angle for robustness.
6. **Shadow removal** - Multiplicative illumination normalization (never paints white onto content).
7. **White balance** - Per-channel gain from paper pixel sampling.
8. **Enhancement** - Contrast stretch, brightening gamma, unsharp mask, background whitening.
9. **PDF output** - Placed on A4 canvas at 300 DPI.

## Configuration

All parameters are exposed via `ScanConfig`:

```python
from pagescan import ScanConfig

# Custom background color (not wood)
config = ScanConfig(
    background_hsv_low=(100, 50, 30),   # blue tablecloth
    background_hsv_high=(130, 255, 255),
)

# US Letter instead of A4
config = ScanConfig(output_width=2550, output_height=3300)

# High quality output
config = ScanConfig(jpeg_quality=85, output_dpi=300)
```

## Prerequisites

All Python dependencies (OpenCV, NumPy, Pillow, img2pdf, DocAligner, pytesseract) are installed automatically via `pip install pagescan`.

The only manual requirement is the **Tesseract OCR** system binary:

```bash
# Ubuntu / Debian
sudo apt install tesseract-ocr tesseract-ocr-deu

# macOS
brew install tesseract

# Windows
# Download installer from https://github.com/tesseract-ocr/tesseract
```

## License

MIT
