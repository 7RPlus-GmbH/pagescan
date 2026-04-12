"""Lightweight ONNX inference for document corner detection.

Replaces the docaligner-docsaid + capybara dependency chain with a
self-contained implementation using only onnxruntime, OpenCV, and NumPy.

The ONNX model (fastvit_sa24, ~80MB) is downloaded automatically from
Google Drive on first use and cached in ~/.cache/pagescan/.
"""

import hashlib
import logging
import os
import shutil
import tempfile
import urllib.request
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

MODEL_FILENAME = "fastvit_sa24_h_e_bifpn_256_fp32.onnx"
MODEL_GDRIVE_ID = "14vUH77v6yGg7zFctUgcT6BzV5Iisg4Dl"
MODEL_SHA256 = None  # Set after first verified download
MODEL_SIZE_INFER = (256, 256)
HEATMAP_THRESHOLD = 0.1

_ort_session = None


def _get_cache_dir() -> Path:
    cache = Path(os.environ.get("PAGESCAN_CACHE",
                                Path.home() / ".cache" / "pagescan"))
    cache.mkdir(parents=True, exist_ok=True)
    return cache


def _download_from_gdrive(file_id: str, dest: Path) -> None:
    """Download a file from Google Drive, handling the confirmation token."""
    url = f"https://drive.google.com/uc?export=download&id={file_id}&confirm=t"
    logger.info(f"  Downloading model to {dest} ...")

    tmp = dest.with_suffix('.tmp')
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "pagescan"})
        with urllib.request.urlopen(req, timeout=120) as resp, open(tmp, 'wb') as f:
            shutil.copyfileobj(resp, f)
        tmp.rename(dest)
        logger.info(f"  Model downloaded ({dest.stat().st_size / 1e6:.0f} MB)")
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


def _ensure_model() -> Path:
    """Ensure the ONNX model is available, downloading if needed."""
    model_path = _get_cache_dir() / MODEL_FILENAME
    if model_path.exists() and model_path.stat().st_size > 1_000_000:
        return model_path
    _download_from_gdrive(MODEL_GDRIVE_ID, model_path)
    return model_path


def _get_session():
    """Get or create the ONNX runtime session (singleton)."""
    global _ort_session
    if _ort_session is not None:
        return _ort_session

    try:
        import onnxruntime as ort
    except ImportError:
        raise ImportError(
            "onnxruntime is required for ML corner detection. "
            "Install it with: pip install onnxruntime"
        )

    model_path = _ensure_model()
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    _ort_session = ort.InferenceSession(
        str(model_path),
        sess_options=opts,
        providers=["CPUExecutionProvider"],
    )
    return _ort_session


def _preprocess(img: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int]]:
    """Resize to model input size and convert to NCHW float tensor."""
    h, w = img.shape[:2]
    resized = cv2.resize(img, MODEL_SIZE_INFER, interpolation=cv2.INTER_LINEAR)
    tensor = resized.transpose(2, 0, 1).astype(np.float32)[None] / 255.0
    return tensor, (h, w)


def _postprocess(heatmaps: np.ndarray,
                 orig_size: Tuple[int, int]) -> Optional[np.ndarray]:
    """Extract 4 corner points from heatmap predictions.

    Each of the 4 heatmap channels represents one corner. We threshold,
    binarize, find the largest contour, and take its centroid.
    """
    oh, ow = orig_size
    points = []

    for i in range(min(4, heatmaps.shape[1])):
        hmap = heatmaps[0, i]

        # Resize heatmap back to original image dimensions
        hmap = cv2.resize(hmap, (ow, oh), interpolation=cv2.INTER_LINEAR)

        # Threshold and binarize
        hmap[hmap < HEATMAP_THRESHOLD] = 0
        binary = (hmap * 255).astype(np.uint8)
        _, binary = cv2.threshold(binary, 127, 255, cv2.THRESH_BINARY)

        # Find contours, pick largest by area
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        largest = max(contours, key=cv2.contourArea)
        M = cv2.moments(largest)
        if M["m00"] < 1:
            return None

        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]
        points.append([cx, cy])

    if len(points) != 4:
        return None

    return np.array(points, dtype=np.float32)


def detect_corners_onnx(image: np.ndarray) -> Optional[np.ndarray]:
    """Detect document corners using the ONNX heatmap model.

    Returns 4 corner points as float32 array of shape (4, 2), or None.
    """
    if image.ndim != 3 or image.shape[2] != 3:
        return None

    session = _get_session()
    tensor, orig_size = _preprocess(image)

    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    outputs = session.run([output_name], {input_name: tensor})

    return _postprocess(outputs[0], orig_size)
