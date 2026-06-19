# Quickstart

## Single document

```python
import pagescan

result = pagescan.scan("photo.jpg", "output.pdf")
print(result)
# {'success': True, 'output_path': 'output.pdf', 'method': 'cascade', ...}
```

The return dict tells you:

- `success` — `True` / `False`
- `output_path` — absolute path to the produced PDF
- `method` — `"cascade"`, `"legacy"` (legacy ML), or `"conservative"` (fallback)
- `quality_score` — `0.0`–`1.0` heuristic confidence
- `message` — human-readable description of what happened

## Batch directory

```python
import pagescan

summary = pagescan.scan_batch("input_photos/", "output_pdfs/")
print(f"Processed {summary['processed']}, "
      f"low quality {summary['low_quality']}, failed {summary['failed']}")
```

By default this uses `min(4, cpu_count)` workers. Cap parallelism if you need to:

```python
pagescan.scan_batch("input/", "output/", workers=4)
```

## Custom configuration

All tunables live on the {class}`~pagescan.ScanConfig` dataclass:

```python
from pagescan import ScanConfig, scan

config = ScanConfig(
    auto_orient=True,       # OCR-based 90/180/270° correction
    enhance=True,           # shadow removal + sharpening
    jpeg_quality=60,        # bigger files, less compression artefact
    output_width=2550,      # US Letter @ 300 DPI
    output_height=3300,
)
scan("photo.jpg", "out.pdf", config=config)
```

See the [Configuration page](config.md) for every field, defaults, and the four built-in presets (`PRESET_A4_300`, `PRESET_LETTER_300`, `PRESET_FAST`, `PRESET_RAW`).

## Command-line interface

```bash
# Single
pagescan photo.jpg output.pdf

# Batch
pagescan --batch --input-dir photos/ --output-dir scans/ --workers 4

# No enhancement — perspective + crop only
pagescan photo.jpg --raw

# Specific config knobs
pagescan photo.jpg out.pdf --quality 80 --no-deskew
```

Run `pagescan --help` for the full flag list.

## Common patterns

### Disable ML entirely (fast / headless mode)

For environments where the ML extras are unavailable or first-run model download is undesirable:

```python
from pagescan import ScanConfig, scan, PRESET_FAST

scan("photo.jpg", "out.pdf", config=PRESET_FAST)
# Equivalent to: ScanConfig(use_ml=False, auto_orient=False)
```

This skips ML detection and goes straight to the conservative contour-based fallback — useful in CI, headless servers, or when latency matters more than the last 10% of corner accuracy.

### Inspect what the pipeline did

```python
from pagescan import ScanConfig, scan

config = ScanConfig(debug=True, debug_dir="./debug/")
scan("photo.jpg", "out.pdf", config=config)
# ./debug/ now contains: corners.jpg, enhanced_color.jpg, result.jpg
```

### Don't ship a PDF, just get the corrected image

You can call the pipeline pieces directly:

```python
import cv2
import numpy as np
from pagescan.corners import detect_corners
from pagescan.transform import perspective_transform

img = cv2.imread("photo.jpg")
# detect_corners returns (corners, rotation_k, method); corners is None on failure
corners, rotation_k, method = detect_corners(img)
if corners is not None:
    if rotation_k:
        img = np.rot90(img, k=rotation_k)
    straight = perspective_transform(img, corners)
    cv2.imwrite("straight.jpg", straight)
```
