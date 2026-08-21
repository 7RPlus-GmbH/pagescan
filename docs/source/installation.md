# Installation

## From PyPI (recommended)

```bash
pip install git+https://github.com/7RPlus-GmbH/pagescan
```

pagescan is not on PyPI yet, so install from the repository. `pip install
pagescan` will work from the first release.

That installs the core package with all Python dependencies. The pre-trained models (~50 MB total) download automatically from [Hugging Face Hub](https://huggingface.co/7rplus/pagescan-weights) on first use and cache under `~/.cache/huggingface/`.

## Optional system dependency: Tesseract

For the OCR-based orientation cross-check, install Tesseract:

**Ubuntu / Debian:**

```bash
sudo apt install tesseract-ocr tesseract-ocr-deu tesseract-ocr-eng
```

**macOS:**

```bash
brew install tesseract
```

**Windows:** download from [UB-Mannheim's Tesseract distribution](https://github.com/UB-Mannheim/tesseract/wiki).

**Tesseract is optional.** Without it, pagescan still works — orientation falls back to a CNN-only heuristic that is slightly less robust on 180° rotations of text-heavy pages.

## Optional `[ml]` extras

The default detection cascade (YOLO11 + HQ-SAM) needs PyTorch:

```bash
pip install "pagescan[ml]"
```

This pulls in `torch` and `segment-anything-hq`. Without `[ml]`, pagescan automatically falls back to the legacy SA24 + LCNet ONNX detection chain (slightly weaker on hard cases, no torch required).

## Development install

Clone the repository and install with the `[dev]` extras for the test and lint toolchain:

```bash
git clone https://github.com/7RPlus-GmbH/pagescan.git
cd pagescan
pip install -e ".[dev,ml,docs]"
```

| Extras group | Adds | Use when |
|---|---|---|
| `[ml]` | torch, segment-anything-hq | Running the YOLO + HQ-SAM cascade |
| `[dev]` | pytest, ruff, mypy | Running the test suite and quality gates |
| `[docs]` | sphinx, pydata-sphinx-theme, myst-parser, … | Building this documentation site |

## Verifying the install

```bash
python -c "import pagescan; print(pagescan.__version__)"
```

## Where weights are cached

The cascade weights (`yolo_doc_v1.onnx`, `sam_hq_vit_b.pth`, …) live under:

```text
~/.cache/huggingface/hub/models--7rplus--pagescan-weights/
```

To override the cache location, set `HF_HOME` or `HUGGINGFACE_HUB_CACHE`. To override pagescan's own cache scratch directory for legacy weights, set `PAGESCAN_CACHE`.
