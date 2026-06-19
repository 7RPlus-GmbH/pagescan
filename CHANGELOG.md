# Changelog

All notable changes to pagescan will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Planned for 0.1.0

- New ML pipeline: YOLO11n document detector → HQ-SAM ViT-B box-prompted segmentation → quad fit, available via `ScanConfig(use_cascade=True)`. On the current v1 weights the legacy SA24 + LCNet chain still wins on the April benchmark (44/50 vs 35/50 at IoU≥0.90, ~75× faster), so `use_cascade` **defaults to `False`** for 0.1.0; the cascade becomes the default once the v2 YOLO closes the training-distribution gap. The `benchmark/comparison/sweep_cascade.py` harness produces this comparison.
- Over-crop guard on the ML path (regression fix; the conservative path already had one).
- Real benchmark suite measuring detection IoU, corner pixel error, OCR word-error rate, and end-to-end visual quality against `docscan` and the OpenCV contour scanner.
- Cascade weights (YOLO11n + HQ-SAM ViT-B) hosted on Hugging Face Hub at [`7rplus/pagescan-weights`](https://huggingface.co/7rplus/pagescan-weights); auto-downloaded on first use via `huggingface_hub`.
- Legacy ONNX weights (DocAligner SA24 / LCNet + a DeepLabV3 doc-segmentation net) now download from the Hugging Face mirror ([`7rplus/pagescan-weights`](https://huggingface.co/7rplus/pagescan-weights)) as the primary source, with DocsaidLab's Google Drive (the original DocAligner upstream) as a transparent fallback. Replaces the previous Google-Drive-only path.
- PDF page size now follows the configured canvas (`output_width` / `output_height` / `output_dpi`) instead of always being A4 — `PRESET_LETTER_300` now produces a genuine US Letter page.
- OCR orientation cross-check language is configurable via `ScanConfig.ocr_lang` (default `"eng+deu+fra+spa+ita+jpn"`); previously hard-coded to German only. Requested language packs that aren't installed are dropped automatically, so multilingual document sets work out of the box with whatever Tesseract packs are present.
- Sphinx documentation site (pydata-sphinx-theme + MyST markdown) covering install, quickstart, architecture, full `ScanConfig` reference, benchmark methodology, troubleshooting, and autodoc-generated API reference. Auto-deploys to GitHub Pages on push to `main`.
- `examples/quickstart.ipynb` — runnable end-to-end notebook covering single scan, batch, custom config, debug visualisation, and the optional ML cascade path.
- CI: ruff + mypy on Python 3.12; pytest matrix on Python 3.9–3.13 (`.github/workflows/ci.yml`).
- Release workflow: stable tags publish to PyPI, RC tags (`vX.Y.Z-rc.N`) stage to TestPyPI; both flows auto-attach the sdist + wheel to a GitHub Release.
- Type-checked end-to-end with mypy (non-strict; `strict = true` is a v0.2 follow-up).
- `scan()` return now includes a `method` field: `"cascade"`, `"legacy"`, or `"conservative"` — tells you which detection path actually produced the corners.
- New top-level exports: `PRESET_A4_300`, `PRESET_LETTER_300`, `PRESET_FAST`, `PRESET_RAW` (previously only reachable via `pagescan.config`).

### Removed in 0.1.0 (relative to 0.0.x)

- Dead-code paths in `edges.py` (`trim_edges`, `find_precise_edges`, `find_receipt_bounds`).

## [0.0.2] — internal

Initial private prototype. SA24 + LCNet + DeepLabV3 corner-detection chain,
edge fallback, perspective transform, A4 PDF output, document-orientation
heuristics. Not released to PyPI.
