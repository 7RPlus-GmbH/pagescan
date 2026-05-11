# Configuration

All tunable parameters live on the {class}`~pagescan.ScanConfig` dataclass. Construct one, pass it to {func}`~pagescan.scan` or {func}`~pagescan.scan_batch`.

```python
from pagescan import ScanConfig, scan

config = ScanConfig(jpeg_quality=80, auto_orient=False)
scan("photo.jpg", "out.pdf", config=config)
```

## Field reference

### Background detection

| Field | Type | Default | Purpose |
|---|---|---|---|
| `background_hsv_low` | `(int, int, int)` | `(0, 65, 30)` | Lower HSV bound for background. Default targets warm wood (`H=0–45`, `S≥65`, `V≥30`). |
| `background_hsv_high` | `(int, int, int)` | `(45, 255, 255)` | Upper HSV bound for background. |
| `background_hsv_strict_s` | `int` | `90` | Stricter saturation minimum for edge detection. Prevents cream-coloured paper (`S=65–85`) from being matched as background. |

```python
# Blue tablecloth instead of wood
ScanConfig(
    background_hsv_low=(100, 50, 30),
    background_hsv_high=(130, 255, 255),
)
```

### Output dimensions

| Field | Type | Default | Notes |
|---|---|---|---|
| `output_width` | `int` | `2480` | A4 width at 300 DPI. |
| `output_height` | `int` | `3508` | A4 height at 300 DPI. |
| `output_dpi` | `int` | `300` | DPI tag written to the PDF. |
| `output_margin` | `int` | `50` | Margin around document on the output canvas. |
| `jpeg_quality` | `int` | `50` | JPEG quality for PDF embedding. Lower = smaller file. |

```python
# US Letter
ScanConfig(output_width=2550, output_height=3300)
```

### Pipeline toggles

| Field | Type | Default | What it controls |
|---|---|---|---|
| `auto_orient` | `bool` | `True` | OCR-based 90/180/270° rotation correction. Disable when input orientation is known. |
| `deskew` | `bool` | `True` | Hough-based skew correction for residual text tilt after perspective. |
| `enhance` | `bool` | `True` | Master switch for the enhancement stages. |
| `shadow_removal` | `bool` | `True` | Illumination normalisation before enhancement. |
| `white_balance` | `bool` | `True` | Push the paper background toward pure white. |
| `use_ml` | `bool` | `True` | Master switch for ML-based corner detection. When `False`, skips both cascade and legacy ML and goes straight to the contour fallback. |

### ML detection path

| Field | Type | Default | What it controls |
|---|---|---|---|
| `use_cascade` | `bool` | `True` | Use the YOLO + HQ-SAM cascade as the primary ML detection path. When `False` (or when the cascade weights are missing), pagescan falls back to the legacy SA24+LCNet ML chain. The cascade requires the `[ml]` extras. |
| `detector_conf_threshold` | `float` | `0.25` | Minimum YOLO confidence for accepting a detection. Below this, cascade falls through to legacy. |
| `min_doc_coverage` | `float` | `0.05` | Over-crop guard. Predicted quads with area below this fraction of the image are rejected as too small (typical failure: SAM segments an inner text block instead of the full page). Raise toward `0.10` if your documents always fill ≥10% of the frame. |

### Debug

| Field | Type | Default | Notes |
|---|---|---|---|
| `debug` | `bool` | `False` | When `True`, intermediate stages write images to `debug_dir`. Useful for diagnosing bad scans. |
| `debug_dir` | `str` | `"pagescan_debug"` | Directory for debug output. Created if missing. |

## Built-in presets

The {mod}`pagescan.config` module exposes four preset configurations for the most common use cases:

| Preset | Equivalent of | Use case |
|---|---|---|
| `PRESET_A4_300` | `ScanConfig()` (defaults) | Standard A4 office scan at 300 DPI. |
| `PRESET_LETTER_300` | `output_width=2550, output_height=3300` | US Letter at 300 DPI. |
| `PRESET_FAST` | `use_ml=False, auto_orient=False` | Headless / fast path. ~10× faster, slightly worse corner accuracy. |
| `PRESET_RAW` | `enhance=False, shadow_removal=False, white_balance=False` | Crop + perspective only. No tone changes — preserves original colours and texture. |

```python
from pagescan import scan, PRESET_FAST

scan("photo.jpg", "out.pdf", config=PRESET_FAST)
```

## Tuning guide

### "It's cropping too aggressively"

Raise the over-crop guard:

```python
ScanConfig(min_doc_coverage=0.15)
```

Then `pagescan.scan(...)` rejects any quad smaller than 15% of the frame, falling back to the conservative contour path. This is the single most common knob to adjust.

### "It's not detecting the document at all"

Lower the detector confidence threshold:

```python
ScanConfig(detector_conf_threshold=0.10)
```

This accepts weaker YOLO detections. Risk: more false positives on document-like background objects.

### "Whites are coming out grey"

Enable white balance (default), or raise JPEG quality so the post-enhancement whites don't get crushed:

```python
ScanConfig(white_balance=True, jpeg_quality=80)
```

### "Output looks oversaturated / cartoonish"

Disable enhancement entirely:

```python
ScanConfig(enhance=False)
# Or use the raw preset:
from pagescan import PRESET_RAW
```

### "First scan is slow"

That's the one-time HF download (~50 MB cascade weights, ~360 MB HQ-SAM checkpoint). Subsequent scans use the cached copy. Pre-warm in a build step:

```python
from pagescan.detector import _ensure_model as ensure_detector
from pagescan.segmenter import _ensure_model as ensure_sam
ensure_detector()
ensure_sam()
```
