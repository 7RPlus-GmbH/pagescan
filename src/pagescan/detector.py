"""YOLO11 document detector — first stage of the production cascade.

The detector finds an axis-aligned bounding box around the document; that
bbox becomes the box prompt to the HQ-SAM segmenter (next stage). Pure
ONNX-runtime inference with no torch dependency.

Public API:
    detector.detect(image_bgr, conf_threshold=0.25) -> (bbox_xyxy, conf) | None
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_MODEL_NAME = "yolo_doc_v1.onnx"
INPUT_SIZE = 960  # imgsz used at training time

# Resolves to <repo>/data/model/ in development; in installed package the
# model is fetched from the user cache (or eventually downloaded from HF).
_DATA_MODEL_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "model"

_session = None  # singleton onnxruntime.InferenceSession


def _get_cache_dir() -> Path:
    cache = Path(os.environ.get("PAGESCAN_CACHE",
                                Path.home() / ".cache" / "pagescan"))
    cache.mkdir(parents=True, exist_ok=True)
    return cache


def _ensure_model() -> Path:
    """Locate the YOLO ONNX. Search order: project data/, user cache."""
    local = _DATA_MODEL_DIR / DEFAULT_MODEL_NAME
    if local.exists() and local.stat().st_size > 100_000:
        return local
    cache_path = _get_cache_dir() / DEFAULT_MODEL_NAME
    if cache_path.exists() and cache_path.stat().st_size > 100_000:
        return cache_path
    raise FileNotFoundError(
        f"YOLO detector '{DEFAULT_MODEL_NAME}' not found in {_DATA_MODEL_DIR} "
        f"or {cache_path}. The cascade is unavailable; pagescan will fall back "
        f"to the legacy SA24+LCNet pipeline."
    )


def _get_session():
    global _session
    if _session is not None:
        return _session
    try:
        import onnxruntime as ort
    except ImportError as e:
        raise ImportError(
            "onnxruntime is required for the YOLO detector. "
            "Install it with: pip install onnxruntime"
        ) from e
    path = _ensure_model()
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    _session = ort.InferenceSession(str(path), sess_options=opts,
                                    providers=["CPUExecutionProvider"])
    logger.info(f"Loaded YOLO detector from {path}")
    return _session


def _letterbox(img: np.ndarray, new_shape: int = INPUT_SIZE
                ) -> tuple[np.ndarray, float, tuple[int, int]]:
    """Resize to (new_shape, new_shape) preserving aspect ratio with gray padding.

    Returns:
        canvas:     (new_shape, new_shape, 3) uint8 image
        ratio:      scale factor applied to original
        (pad_x, pad_y): pixel offsets to undo letterbox
    """
    h, w = img.shape[:2]
    r = new_shape / max(h, w)
    new_h, new_w = round(h * r), round(w * r)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((new_shape, new_shape, 3), 114, dtype=np.uint8)
    pad_y = (new_shape - new_h) // 2
    pad_x = (new_shape - new_w) // 2
    canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized
    return canvas, r, (pad_x, pad_y)


def detect(image: np.ndarray, conf_threshold: float = 0.25
           ) -> tuple[np.ndarray, float] | None:
    """Find the highest-confidence document bbox.

    Args:
        image: BGR image (cv2.imread output).
        conf_threshold: minimum detection confidence.

    Returns:
        (bbox_xyxy, conf) where bbox is float32 [x1, y1, x2, y2] in original
        image coordinates, or None if no detection above threshold.
    """
    sess = _get_session()
    h0, w0 = image.shape[:2]

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    canvas, ratio, (pad_x, pad_y) = _letterbox(rgb)
    tensor = canvas.transpose(2, 0, 1).astype(np.float32)[None] / 255.0

    inputs = {sess.get_inputs()[0].name: tensor}
    output = sess.run(None, inputs)[0]  # (1, 5, N) for single-class detect

    # Output may come back as (1, 5, N) or (1, N, 5) depending on opset; normalize.
    out = output[0]
    if out.shape[0] == 5:
        bboxes_cxcywh = out[:4].T  # (N, 4)
        scores = out[4]            # (N,)
    else:
        bboxes_cxcywh = out[:, :4]
        scores = out[:, 4]

    mask = scores >= conf_threshold
    if not mask.any():
        return None

    bboxes_cxcywh = bboxes_cxcywh[mask]
    scores = scores[mask]
    best_i = int(np.argmax(scores))
    cx, cy, bw, bh = bboxes_cxcywh[best_i]
    conf = float(scores[best_i])

    # cxcywh (in letterboxed coords) -> xyxy in original-image coords
    x1 = (cx - bw / 2 - pad_x) / ratio
    y1 = (cy - bh / 2 - pad_y) / ratio
    x2 = (cx + bw / 2 - pad_x) / ratio
    y2 = (cy + bh / 2 - pad_y) / ratio

    x1 = max(0.0, min(float(w0 - 1), float(x1)))
    y1 = max(0.0, min(float(h0 - 1), float(y1)))
    x2 = max(0.0, min(float(w0 - 1), float(x2)))
    y2 = max(0.0, min(float(h0 - 1), float(y2)))

    return np.array([x1, y1, x2, y2], dtype=np.float32), conf
