# Troubleshooting

Common issues, their root causes, and how to fix them. If your problem isn't listed here, please [open an issue](https://github.com/7RPlus-GmbH/pagescan/issues) with a minimal reproducer and (if possible) the input photo.

## Installation

### `ImportError: cannot import name 'sam_model_registry' from 'segment_anything_hq'`

The `[ml]` extras aren't installed. pagescan still works without them — it falls back to the legacy ML chain. To enable the production cascade:

```bash
pip install "pagescan[ml]"
```

### `ModuleNotFoundError: No module named 'torch'`

Same root cause as above. The cascade requires PyTorch. If you don't want the torch dependency, accept the fallback to the legacy chain — pagescan auto-detects.

### `pytesseract.TesseractNotFoundError: tesseract is not installed`

The Tesseract binary isn't on `PATH`. pagescan only needs it for the orientation cross-check; without it, orientation falls back to a CNN-only heuristic that's slightly weaker on 180° rotations.

To install (see [Installation](installation.md) for OS-specific commands):

```bash
sudo apt install tesseract-ocr   # Linux
brew install tesseract           # macOS
```

## Detection

### "It cropped off part of my document"

This is the single most reported issue. Pagescan has an over-crop guard ({attr}`~pagescan.ScanConfig.min_doc_coverage`, default `0.05`) but the default may be too permissive for your distribution.

Raise the floor:

```python
from pagescan import ScanConfig, scan
scan("photo.jpg", "out.pdf", config=ScanConfig(min_doc_coverage=0.15))
```

If the document still gets cropped, run with `debug=True` and inspect `debug/corners.jpg` to see what the model predicted:

```python
ScanConfig(debug=True, debug_dir="./debug/")
```

### "It detected the document but rotated it wrong"

The orientation classifier got confused — usually because:

1. The page has very little text (e.g. a form, a diagram).
2. The page is in a language not in `auto_orient`'s OCR cross-check.

Disable auto-orient and rotate manually:

```python
ScanConfig(auto_orient=False)
```

Or supply the right Tesseract language data with `TESSDATA_PREFIX` set to a directory containing `deu.traineddata`, `fra.traineddata`, etc.

### "ML detection failed, falling back to contour"

You'll see log output like:

```text
Detection: conservative
```

Causes:

1. **First-run download failed** — check network access to `huggingface.co`. The first scan needs to fetch ~50 MB (and the 362 MB HQ-SAM checkpoint if `[ml]` is installed).
2. **YOLO confidence below threshold** — the document is small, partially out of frame, or shot at an extreme angle. Lower `detector_conf_threshold`:
   ```python
   ScanConfig(detector_conf_threshold=0.10)
   ```
3. **Validation guard rejected the quad** — coverage, dimension, or parallelism failed. Bump `min_doc_coverage` *down* (carefully) or inspect with `debug=True`.

The conservative contour fallback is the safety net — it produces a reasonable result on most images even when ML fails, just with looser corner accuracy.

## Output

### "PDF is way too big"

Lower the JPEG quality:

```python
ScanConfig(jpeg_quality=30)   # default is 50
```

A typical A4 scan at quality 30 is ~150 KB; at 50, ~250 KB; at 80, ~800 KB.

### "Output is too small / too big on the page"

Adjust the canvas dimensions and the margin:

```python
ScanConfig(
    output_width=2480, output_height=3508,   # A4 @ 300 DPI (default)
    output_margin=100,                       # tighter margin = bigger document
)
```

### "Output has weird colours / looks oversaturated"

Disable enhancement entirely:

```python
from pagescan import PRESET_RAW, scan
scan("photo.jpg", "out.pdf", config=PRESET_RAW)
```

Or selectively disable individual stages:

```python
ScanConfig(white_balance=False, shadow_removal=False, enhance=False)
```

## Performance

### "First scan takes 30+ seconds"

One-time weight download:

- YOLO ONNX: ~11 MB
- HQ-SAM ViT-B: ~362 MB

After that, weights are cached under `~/.cache/huggingface/hub/` and load instantly. Pre-warm in CI or build steps:

```python
from pagescan.detector import _ensure_model as ensure_detector
from pagescan.segmenter import _ensure_model as ensure_sam
ensure_detector()
ensure_sam()
```

### "Each scan still takes 5+ seconds with the cascade"

That's expected for the v1 cascade on CPU. HQ-SAM ViT-B inference dominates. Options:

1. **Use the legacy chain** — `ScanConfig(use_cascade=False)` runs at ~70 ms/image.
2. **Use the fast preset** — `PRESET_FAST` skips ML entirely, ~30 ms/image.
3. **GPU** — install a CUDA-enabled torch; HQ-SAM will use it automatically and drop to ~150 ms/image.

### "Tests are slow / hanging"

The default test suite uses `use_ml=False` and shouldn't take more than 5 seconds total. If yours is slow, you likely have stale cached state or a debug log redirected to disk — clear `pagescan_debug/` and `~/.cache/pagescan/`.

## Where to get help

- Read the [API reference](api.md).
- Search [existing issues](https://github.com/7RPlus-GmbH/pagescan/issues).
- Open a new issue with: pagescan version, Python version, OS, a minimal reproducer, and the input photo (if you can share it) or a description of the document type and shooting conditions.
