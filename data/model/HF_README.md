---
license: apache-2.0
tags:
  - document-scanner
  - document-detection
  - perspective-correction
  - object-detection
  - segmentation
  - yolo
  - sam
library_name: pagescan
pipeline_tag: object-detection
---

# pagescan weights

Pre-trained model weights used by [pagescan](https://github.com/7RPlus-GmbH/pagescan), a privacy-first document scanner that turns phone photos into clean, deskewed, print-ready PDFs.

These weights power the default detection cascade:

1. **YOLO11** detects a coarse axis-aligned bounding box around the document.
2. **HQ-SAM ViT-B**, prompted by that bbox, returns a precise binary mask.
3. The mask is fitted to a 4-corner quadrilateral for perspective correction.

A **legacy fallback chain** is also bundled, used when the cascade is unavailable
(e.g. no torch) or fails on a given image: DocAligner's FastViT-SA24 and LCNet100
corner-heatmap regressors, plus a DeepLabV3-MobileNetV3 document-segmentation net
for the conservative-crop path. These are re-hosted third-party weights — see
attribution below.

## Files

| File | Size | Description |
|---|---|---|
| `yolo_doc_v1.onnx` | 11 MB | Exported YOLO11 detector — used at inference time by `pagescan.detector` (no torch required). |
| `yolo_doc_v1.pt` | 5.3 MB | Original PyTorch checkpoint — kept for re-export / fine-tuning. |
| `sam_hq_vit_b.pth` | 362 MB | HQ-SAM ViT-B checkpoint — re-hosted copy (see attribution below). |
| `fastvit_sa24_h_e_bifpn_256_fp32.onnx` | 83 MB | DocAligner FastViT-SA24 corner-heatmap regressor — legacy primary (re-hosted; see attribution). |
| `lcnet100_h_e_bifpn_256_fp32.onnx` | 4.8 MB | DocAligner LCNet100 corner-heatmap regressor — legacy fallback backbone (re-hosted; see attribution). |
| `deeplabv3_mbv3_docseg.onnx` | 44 MB | DeepLabV3-MobileNetV3 document segmentation — conservative-crop fallback (re-hosted; see attribution). |

## Usage

These weights are downloaded automatically by pagescan on first use:

```bash
pip install pagescan
python -c "from pagescan import scan; scan('photo.jpg', 'out.pdf')"
```

To pin a specific revision or use a local cache:

```python
from huggingface_hub import hf_hub_download

onnx_path = hf_hub_download(
    repo_id="7rplus/pagescan-weights",
    filename="yolo_doc_v1.onnx",
)
```

## Training

`yolo_doc_v1` was trained from `yolo11n-obb.pt` on a 1000-photo private corpus
of phone-captured documents (Dec 2025) with oriented bounding box (OBB) labels.
The training script lives in [`training/yolo/`](https://github.com/7RPlus-GmbH/pagescan/tree/main/training/yolo).
A v2 trained on an extended, distribution-balanced corpus is in progress.

## Attribution — HQ-SAM

`sam_hq_vit_b.pth` is **not original work from this repository**. It is a re-hosted copy of the ViT-B HQ-SAM checkpoint from the HQ-SAM authors, included here so pagescan installs ship with a single, stable weight source.

> **HQ-SAM** — *Segment Anything in High Quality*
> Lei Ke, Mingqiao Ye, Martin Danelljan, Yifan Liu, Yu-Wing Tai, Chi-Keung Tang, Fisher Yu. *NeurIPS 2023*.
> Paper: <https://arxiv.org/abs/2306.01567>
> Code & original weights: <https://github.com/SysCV/sam-hq>
> Original Hugging Face mirror: <https://huggingface.co/lkeab/hq-sam>

HQ-SAM is released under the **Apache 2.0** license. We re-host the ViT-B checkpoint unchanged; all credit for the model itself belongs to the original authors. If you use pagescan in research that depends on the cascade's segmentation quality, please cite the HQ-SAM paper:

```bibtex
@inproceedings{ke2023segment,
  title     = {Segment Anything in High Quality},
  author    = {Ke, Lei and Ye, Mingqiao and Danelljan, Martin and Liu, Yifan and Tai, Yu-Wing and Tang, Chi-Keung and Yu, Fisher},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2023}
}
```

## Attribution — DocAligner (legacy weights)

`fastvit_sa24_h_e_bifpn_256_fp32.onnx` and `lcnet100_h_e_bifpn_256_fp32.onnx` are
**not original work from this repository**. They are re-hosted copies of the
corner-detection ONNX weights from the **DocAligner** project by DocsaidLab,
included so pagescan's legacy fallback path has a single, stable weight source
(the DocAligner upstream is an undocumented auto-downloader).

> **DocAligner** — DocsaidLab
> Code & docs: <https://github.com/DocsaidLab/DocAligner> · <https://docsaid.org/en/docs/docaligner/>

DocAligner is released under the **Apache 2.0** license. We re-host these
checkpoints unchanged; all credit for the models belongs to DocsaidLab.

`deeplabv3_mbv3_docseg.onnx` is a re-hosted DeepLabV3-MobileNetV3 document-segmentation
model used only by the conservative-crop fallback. Its upstream provenance is not
definitively traced; if you redistribute it, confirm its original source and license.

## License

- `yolo_doc_v1.{pt,onnx}` — released under Apache 2.0 by 7R+ GmbH.
- `sam_hq_vit_b.pth` — Apache 2.0, HQ-SAM authors (see above).
- `fastvit_sa24_*.onnx`, `lcnet100_*.onnx` — Apache 2.0, DocAligner / DocsaidLab (see above).
- `deeplabv3_mbv3_docseg.onnx` — re-hosted; provenance/license to be confirmed.

The pagescan package itself is MIT-licensed; see the [main repository](https://github.com/7RPlus-GmbH/pagescan).
