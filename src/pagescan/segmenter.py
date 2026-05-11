"""HQ-SAM ViT-B box-prompted segmentation — second stage of the cascade.

Takes the YOLO-detected document bbox as a box prompt to HQ-SAM and
returns a precise binary mask. The mask is then fitted to a 4-corner
quadrilateral via convex hull + polygon approximation.

Public API:
    segmenter.segment(image_bgr, bbox_xyxy) -> mask | None
    segmenter.mask_to_quad(mask) -> (4, 2) ndarray | None

Imports torch + segment_anything_hq lazily so that environments without
the cascade extras still get a working `import pagescan` (the cascade
just becomes unavailable and `corners.detect_corners_ml` falls back to
the legacy SA24+LCNet path).
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_MODEL_NAME = "sam_hq_vit_b.pth"
MODEL_TYPE = "vit_b"
HF_REPO_ID = "7rplus/pagescan-weights"

_DATA_MODEL_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "model"

_predictor = None  # singleton SamPredictor


def _get_cache_dir() -> Path:
    cache = Path(os.environ.get("PAGESCAN_CACHE",
                                Path.home() / ".cache" / "pagescan"))
    cache.mkdir(parents=True, exist_ok=True)
    return cache


def _ensure_model() -> Path:
    """Locate the HQ-SAM checkpoint. Search order: project data/, user cache, HF Hub."""
    local = _DATA_MODEL_DIR / DEFAULT_MODEL_NAME
    if local.exists() and local.stat().st_size > 100_000_000:
        return local
    cache_path = _get_cache_dir() / DEFAULT_MODEL_NAME
    if cache_path.exists() and cache_path.stat().st_size > 100_000_000:
        return cache_path

    from huggingface_hub import hf_hub_download
    logger.info(f"Downloading {DEFAULT_MODEL_NAME} from {HF_REPO_ID} (first run, ~360 MB)...")
    return Path(hf_hub_download(repo_id=HF_REPO_ID, filename=DEFAULT_MODEL_NAME))


def _get_predictor():
    """Lazy-load HQ-SAM. Raises ImportError if torch / segment_anything_hq are missing."""
    global _predictor
    if _predictor is not None:
        return _predictor
    try:
        import torch
        from segment_anything_hq import SamPredictor, sam_model_registry
    except ImportError as e:
        raise ImportError(
            "HQ-SAM segmentation requires torch and segment-anything-hq. "
            "Install with: pip install pagescan[ml]"
        ) from e

    path = _ensure_model()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam = sam_model_registry[MODEL_TYPE](checkpoint=None)
    state = torch.load(str(path), map_location=device, weights_only=True)
    sam.load_state_dict(state)
    sam.to(device).eval()
    _predictor = SamPredictor(sam)
    logger.info(f"Loaded HQ-SAM ({MODEL_TYPE}) from {path} on {device}")
    return _predictor


def segment(image: np.ndarray, bbox: np.ndarray) -> np.ndarray | None:
    """Segment the document given an axis-aligned box prompt.

    Args:
        image: BGR image.
        bbox: float32 [x1, y1, x2, y2] axis-aligned bbox.

    Returns:
        Binary mask (H, W) bool, or None on failure.
    """
    try:
        import torch
    except ImportError as e:
        raise ImportError("torch is required for HQ-SAM") from e
    import torch

    predictor = _get_predictor()
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    predictor.set_image(rgb)

    with torch.no_grad():
        masks, _scores, _ = predictor.predict(
            box=np.asarray(bbox, dtype=np.float32),
            multimask_output=False,   # box prompt is unambiguous
            hq_token_only=True,       # use HQ token exclusively for sharpest edges
        )
    if masks is None or len(masks) == 0:
        return None
    return masks[0].astype(bool)


def mask_to_quad(mask: np.ndarray, min_area: int = 1000) -> np.ndarray | None:
    """Fit a 4-corner quadrilateral to a binary mask.

    Strategy: largest external contour -> convex hull -> walk approxPolyDP
    epsilon until exactly 4 vertices. Falls back to minAreaRect (rotated
    rect) if no epsilon yields 4 vertices.

    Returns:
        (4, 2) float32 array of corner coords, or None if the mask is empty.
    """
    contours, _ = cv2.findContours(mask.astype(np.uint8),
                                   cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < min_area:
        return None
    hull = cv2.convexHull(contour)
    perim = cv2.arcLength(hull, True)
    for eps_frac in (0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.12):
        approx = cv2.approxPolyDP(hull, eps_frac * perim, True)
        if len(approx) == 4:
            return approx.reshape(4, 2).astype(np.float32)
    rect = cv2.minAreaRect(hull)
    return cv2.boxPoints(rect).astype(np.float32)
