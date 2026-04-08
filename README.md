# pagescan

Turn phone photos of documents into clean, deskewed, print-ready PDFs.

Built for the real world: documents photographed on wood tables, under uneven lighting, at odd angles. Handles corner detection, perspective correction, shadow removal, deskew, auto-rotation, and scan-like enhancement in a single pipeline.

## Installation

```bash
pip install pagescan
```

With optional ML corner detection (recommended):

```bash
pip install pagescan[ml]
```

With OCR-based auto-rotation:

```bash
pip install pagescan[ocr]
```

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

## Dependencies

**Required:** OpenCV, NumPy, Pillow, img2pdf

**Optional:**
- `docaligner` - ML corner detection (`pip install pagescan[ml]`)
- `pytesseract` - Auto-rotation via OCR (`pip install pagescan[ocr]`)

## License

MIT
