# API reference

## Public API

The public surface of pagescan is intentionally small: two functions and one configuration class.

```{eval-rst}
.. currentmodule:: pagescan

.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   scan
   scan_batch
   ScanConfig
```

## Configuration presets

Four `ScanConfig` instances ready to use without further tuning:

- `pagescan.PRESET_A4_300` — defaults; A4 at 300 DPI.
- `pagescan.PRESET_LETTER_300` — US Letter at 300 DPI.
- `pagescan.PRESET_FAST` — `use_ml=False, auto_orient=False`. ~10× faster, slightly worse corner accuracy.
- `pagescan.PRESET_RAW` — no enhancement, no white balance, no shadow removal. Crop + perspective only.

See the [Configuration](config.md) page for tuning guidance.

## Pipeline

```{eval-rst}
.. automodule:: pagescan.pipeline
   :members: scan, scan_batch
```

## Configuration

```{eval-rst}
.. autoclass:: pagescan.config.ScanConfig
   :members:
   :member-order: bysource
```

## Internal modules

These modules are documented for contributors and advanced users who want to compose the pipeline from individual stages. They are **not** part of the stable public API — signatures may change between minor versions.

### `pagescan.detector`

YOLO11 ONNX document detector. First stage of the production cascade.

```{eval-rst}
.. automodule:: pagescan.detector
   :members:
```

### `pagescan.segmenter`

HQ-SAM ViT-B box-prompted segmentation. Second stage of the cascade. Imports torch lazily.

```{eval-rst}
.. automodule:: pagescan.segmenter
   :members:
```

### `pagescan.corners`

Corner-detection orchestration: cascade and legacy ML paths share validate-and-repair logic here.

```{eval-rst}
.. automodule:: pagescan.corners
   :members:
```

### `pagescan.edges`

Contour-based fallback used when ML corner detection fails entirely.

```{eval-rst}
.. automodule:: pagescan.edges
   :members:
```

### `pagescan.transform`

Perspective transform and canvas placement.

```{eval-rst}
.. automodule:: pagescan.transform
   :members:
```

### `pagescan.enhance`

Shadow removal, white balance, contrast stretch, unsharp mask.

```{eval-rst}
.. automodule:: pagescan.enhance
   :members:
```

### `pagescan.orientation`

Deskew and auto-rotation (CNN + optional Tesseract).

```{eval-rst}
.. automodule:: pagescan.orientation
   :members:
```

### `pagescan.quality`

Heuristic quality scoring for scanned documents.

```{eval-rst}
.. automodule:: pagescan.quality
   :members:
```

### `pagescan.output`

PDF rendering.

```{eval-rst}
.. automodule:: pagescan.output
   :members:
```

### `pagescan.model`

Legacy SA24 + LCNet + DeepLabV3 ONNX inference. Used as the fallback when the cascade is unavailable.

```{eval-rst}
.. automodule:: pagescan.model
   :members:
```
