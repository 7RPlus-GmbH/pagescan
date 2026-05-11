# Changelog

All notable changes to pagescan will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Planned for 0.1.0

- New ML pipeline: YOLO11n document detector → HQ-SAM ViT-B box-prompted segmentation → quad fit. Replaces the legacy SA24 + LCNet + DeepLabV3 chain (kept as fallback).
- Over-crop guard on the ML path (regression fix; the conservative path already had one).
- Real benchmark suite measuring detection IoU, corner pixel error, OCR word-error rate, and end-to-end visual quality against `docscan` and the OpenCV contour scanner.
- Model weights mirrored to Hugging Face Hub under the `7rplus-gmbh` org (no more Google Drive dependency).
- mkdocs-material documentation site with quickstart, architecture overview, API reference, and benchmark numbers.
- Type-checked under `mypy --strict`.
- CI matrix on Python 3.9–3.13.

### Removed in 0.1.0 (relative to 0.0.x)

- The HSV-corner-based `quality.check_quality` heuristic — replaced with a meaningful confidence score (or removed from the public return dict; decision pending).
- Dead-code paths in `edges.py` (`trim_edges`, `find_precise_edges`, `find_receipt_bounds`).

## [0.0.1] — internal

Initial private prototype. SA24 + LCNet + DeepLabV3 corner-detection chain,
edge fallback, perspective transform, A4 PDF output, document-orientation
heuristics. Not released to PyPI.
