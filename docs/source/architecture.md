# Architecture

pagescan turns a phone photo into a clean PDF by running a *cascade* of vision models followed by classical image-processing steps. The cascade is built so each stage's failure mode is the next stage's domain, and so that the whole thing degrades gracefully when the ML stages aren't available.

## Pipeline overview

```{mermaid}
flowchart LR
    A[Input photo] --> B[YOLO11 detector]
    B -->|bbox| C[HQ-SAM ViT-B segmenter]
    C -->|mask| D[Quad fit<br/>hull → polyDP]
    D -->|4 corners| E[Validate &<br/>repair]
    E -->|valid| F[Perspective transform]

    B -.->|no detection| G[Legacy SA24+LCNet]
    G -.->|fallback| E

    E -.->|reject| H[Contour fallback]
    H -.->|bbox crop| F

    F --> I[Orientation correction]
    I --> J[Enhancement<br/>shadow + WB + sharpen]
    J --> K[PDF output]

    style B fill:#dbeafe,stroke:#2563eb
    style C fill:#dbeafe,stroke:#2563eb
    style D fill:#dbeafe,stroke:#2563eb
    style E fill:#dbeafe,stroke:#2563eb
    style G fill:#fef3c7,stroke:#d97706
    style H fill:#fef3c7,stroke:#d97706
```

Solid arrows are the **production cascade**. Dashed arrows are **fallback paths** when the primary chain fails or its dependencies aren't installed.

## Stages

### 1. Detection — YOLO11

A single-class ([`document`](https://huggingface.co/7rplus/pagescan-weights/blob/main/yolo_doc_v1.onnx)) YOLO11n detector trained on phone photos of documents. ONNX-only at inference time — no torch required, ~145 lines in {mod}`pagescan.detector`.

Output: one axis-aligned bounding box with a confidence score. If confidence falls below {attr}`~pagescan.ScanConfig.detector_conf_threshold` (default `0.25`), the cascade falls through to the legacy path.

### 2. Segmentation — HQ-SAM ViT-B

The detection bbox is used as a *box prompt* to HQ-SAM ViT-B. The segmenter returns a precise binary mask of the document — sharper boundaries than the legacy heatmap models, especially on tilted documents and folded paper.

Lives in {mod}`pagescan.segmenter`. Torch is imported lazily — environments without `[ml]` extras still get a working `import pagescan` (the cascade just becomes unavailable and the legacy path takes over).

### 3. Quad fit

Convex hull of the mask, then `cv2.approxPolyDP` with an adaptive epsilon to reduce the hull to four corners. When the polygon approximation fails to produce a clean 4-vertex quad, the system falls back to `cv2.minAreaRect` on the contour.

### 4. Validate & repair

Every candidate quad is checked against three guards before being accepted:

1. **Coverage** — the quad area must be at least {attr}`~pagescan.ScanConfig.min_doc_coverage` of the frame (default `5%`). Catches the common SAM failure of segmenting an inner text block instead of the full page.
2. **Dimensions** — width and height must be roughly comparable; rejects degenerate slivers.
3. **Parallelism** — opposing sides should be near-parallel. Documents are rectangles; severely non-rectangular quads are perspective artefacts and rejected.

If a quad fails validation, the pipeline drops to the conservative contour-based fallback.

### 5. Perspective transform

Standard four-corner perspective warp via `cv2.getPerspectiveTransform` + `cv2.warpPerspective`. The output's aspect ratio is computed from the detected corners — no forced A4 stretch. The original document shape is preserved.

### 6. Orientation correction

A small CNN classifier predicts the dominant text orientation (0/90/180/270°). When confidence is low or the prediction is 180°, Tesseract OCR scores all four rotations and picks the one with the most recognised words. Without Tesseract, the CNN result is trusted directly.

### 7. Enhancement

Optional steps (all toggleable on {class}`~pagescan.ScanConfig`):

- **Shadow removal** — illumination normalization to flatten uneven lighting.
- **White balance** — paper background pushed toward pure white.
- **Contrast stretch + unsharp mask** — scanner-like crispness.

### 8. PDF output

A4 at 300 DPI by default (overridable). Output is JPEG-encoded inside a PDF wrapper via `img2pdf`. Quality tunable via {attr}`~pagescan.ScanConfig.jpeg_quality`.

## Fallback chain

The cascade is the *primary* path, but pagescan is built to keep working when its dependencies aren't present:

| Condition | Behaviour |
|---|---|
| Cascade weights missing | Falls back to legacy SA24+LCNet ONNX chain. |
| `[ml]` extras not installed | Same as above (torch isn't importable). |
| Legacy ML also fails | Falls back to contour-based detection on the edge map. |
| Contour also fails | Returns the original image with a flag set on `result['success']`. |
| `use_ml=False` | Skips all ML, goes straight to contour fallback. Useful for headless/CI. |

This means a `pip install pagescan` *without* `[ml]` still produces good results — just slightly weaker on hard cases. The cascade is an accuracy upgrade, not a hard requirement.

## Why a cascade?

The previous architecture used heatmap regression models (SA24 + LCNet) that predict corner pixel locations directly. They are fast (~70 ms total) and self-contained, but the model's loss is at the per-pixel level — small detection errors compound to large IoU errors on tilted documents.

The cascade decouples the two sub-problems:

1. *"Where is the document?"* — a detection problem, solved at the bbox level. YOLO is purpose-built for this.
2. *"What are its exact pixel boundaries?"* — a segmentation problem. HQ-SAM is purpose-built for that, and adding a box prompt drastically narrows the search space.

The architectural cost is one extra inference pass and the torch dependency for HQ-SAM. The benefit is sharper boundaries on tilted, occluded, and shadowed documents — exactly the failure cases the legacy chain struggled with.

See [Benchmark](benchmark.md) for measured comparisons on a held-out 50-photo test set.
