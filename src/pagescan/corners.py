"""Document corner detection.

Two paths share the same validate-and-repair logic:

1. **Cascade** (production): YOLO11 detector -> HQ-SAM ViT-B segmenter ->
   convex-hull + approxPolyDP quad fit. Lives in
   `pagescan.detector` + `pagescan.segmenter`.

2. **Legacy ML** (fallback): SA24 + LCNet100 ONNX heatmap regression,
   in `pagescan.model`. Used when the cascade is disabled, unavailable
   (no weights / no torch), or fails on a particular image.

Both paths return raw 4-corner candidates; `_validate_and_repair` enforces
coverage, dimension, and parallelism guards before the corners are accepted.
The over-crop guard (`config.min_doc_coverage`) is applied here too —
cascade or legacy, no quad smaller than the configured floor is accepted.
"""
from __future__ import annotations

import logging
from typing import Optional, Tuple

import cv2
import numpy as np

from pagescan.config import ScanConfig

logger = logging.getLogger(__name__)

# --- Cascade modules (production path) ---
try:
    from pagescan import detector, segmenter  # noqa: F401
    HAS_CASCADE = True
except ImportError:
    HAS_CASCADE = False

# --- Legacy ML chain (fallback path) ---
try:
    from pagescan.model import detect_corners_onnx
    HAS_LEGACY_ML = True
except ImportError:
    HAS_LEGACY_ML = False


# ---------- shared geometry helpers ----------

def order_corners(pts: np.ndarray) -> np.ndarray:
    """Order 4 points as: top-left, top-right, bottom-right, bottom-left.

    Uses the spatial sum/difference heuristic — works for documents that
    are roughly axis-aligned. Heavily-rotated documents may need the
    orientation module to re-label after a perspective warp.
    """
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).flatten()
    rect[0] = pts[np.argmin(s)]   # TL: smallest x+y
    rect[2] = pts[np.argmax(s)]   # BR: largest x+y
    rect[1] = pts[np.argmin(d)]   # TR: smallest y-x
    rect[3] = pts[np.argmax(d)]   # BL: largest y-x
    return rect


def _edge_angle(v: np.ndarray) -> float:
    return np.degrees(np.arctan2(v[1], v[0]))


def _check_parallel(pts: np.ndarray) -> Tuple[float, float]:
    """Return (top-bottom angular diff, left-right angular diff) in degrees."""
    top = pts[1] - pts[0]
    bottom = pts[2] - pts[3]
    left = pts[3] - pts[0]
    right = pts[2] - pts[1]
    tb = abs(_edge_angle(top) - _edge_angle(bottom))
    lr = abs(_edge_angle(left) - _edge_angle(right))
    return tb, lr


def _check_quad_dimensions(pts: np.ndarray, h: int, w: int) -> bool:
    """Reject quads whose short side is less than 15% of the image short side."""
    wt = np.linalg.norm(pts[1] - pts[0])
    wb = np.linalg.norm(pts[2] - pts[3])
    hl = np.linalg.norm(pts[3] - pts[0])
    hr = np.linalg.norm(pts[2] - pts[1])
    quad_short = min(max(wt, wb), max(hl, hr))
    img_short = min(h, w)
    if quad_short < img_short * 0.15:
        logger.debug(f"  quad too narrow: {quad_short:.0f} < {img_short * 0.15:.0f}")
        return False
    return True


def _repair_corners(ordered: np.ndarray, tb_diff: float, lr_diff: float,
                    h: int, w: int) -> Optional[np.ndarray]:
    """Try to make opposite edges parallel by adjusting the outlier corner.

    When top/bottom angles diverge, one bottom corner is misplaced.
    Reconstruct it so bottom is parallel to top (keeping the longer side's
    endpoint fixed). Symmetric logic for left/right divergence.
    """
    repaired = ordered.copy()

    if tb_diff > lr_diff:
        top_dir = ordered[1] - ordered[0]
        top_dir = top_dir / np.linalg.norm(top_dir)
        bottom_len = np.linalg.norm(ordered[2] - ordered[3])
        left_len = np.linalg.norm(ordered[3] - ordered[0])
        right_len = np.linalg.norm(ordered[2] - ordered[1])
        if right_len < left_len * 0.85:
            repaired[2] = repaired[3] + top_dir * bottom_len
        else:
            repaired[3] = repaired[2] - top_dir * bottom_len
    else:
        left_dir = ordered[3] - ordered[0]
        left_dir = left_dir / np.linalg.norm(left_dir)
        right_len = np.linalg.norm(ordered[2] - ordered[1])
        top_len = np.linalg.norm(ordered[1] - ordered[0])
        bottom_len = np.linalg.norm(ordered[2] - ordered[3])
        if bottom_len < top_len * 0.85:
            repaired[2] = repaired[1] + left_dir * right_len
        else:
            top_dir = (ordered[1] - ordered[0]) / np.linalg.norm(ordered[1] - ordered[0])
            repaired[1] = repaired[0] + top_dir * top_len

    repaired[:, 0] = np.clip(repaired[:, 0], 0, w - 1)
    repaired[:, 1] = np.clip(repaired[:, 1], 0, h - 1)

    tb2, lr2 = _check_parallel(repaired)
    if tb2 <= 10 and lr2 <= 10:
        repair_area = cv2.contourArea(repaired)
        repair_coverage = repair_area / (h * w)
        if 0.10 < repair_coverage < 0.98 and _check_quad_dimensions(repaired, h, w):
            logger.info(f"  corners repaired: TB {tb_diff:.1f}->{tb2:.1f} "
                        f"LR {lr_diff:.1f}->{lr2:.1f}")
            return repaired
    return None


def _validate_and_repair(corners: np.ndarray, h: int, w: int,
                         min_coverage: float = 0.05) -> Optional[np.ndarray]:
    """Validate raw corners (coverage, dimensions, parallelism) and attempt repair.

    `min_coverage` is the over-crop guard: any quad whose area is below
    this fraction of the image is rejected. Defaults to 0.05 (5%); tune
    via `ScanConfig.min_doc_coverage`.
    """
    corners = corners.astype(np.float32)
    area = cv2.contourArea(corners)
    coverage = area / (h * w)

    if coverage < min_coverage:
        logger.debug(f"  corners rejected by over-crop guard: "
                     f"coverage {coverage:.3f} < {min_coverage:.3f}")
        return None
    if coverage > 0.99:
        return None

    ordered = order_corners(corners)
    if not _check_quad_dimensions(ordered, h, w):
        return None

    tb_diff, lr_diff = _check_parallel(ordered)
    if tb_diff <= 15 and lr_diff <= 15:
        return ordered

    repaired = _repair_corners(ordered, tb_diff, lr_diff, h, w)
    if repaired is not None:
        return repaired

    if tb_diff <= 25 and lr_diff <= 25:
        logger.info(f"  corners moderately skewed (TB={tb_diff:.1f} "
                    f"LR={lr_diff:.1f}), accepting")
        return ordered

    logger.debug(f"  corners not parallel: TB={tb_diff:.1f} LR={lr_diff:.1f}")
    return None


# ---------- detection backends ----------

def _detect_via_cascade(image: np.ndarray, config: ScanConfig) -> Optional[np.ndarray]:
    """YOLO bbox -> HQ-SAM mask -> quad. Returns raw (unvalidated) 4 corners or None."""
    if not HAS_CASCADE:
        return None
    try:
        det = detector.detect(image, conf_threshold=config.detector_conf_threshold)
        if det is None:
            logger.debug("  cascade: no YOLO detection")
            return None
        bbox, conf = det
        logger.info(f"  cascade: YOLO conf={conf:.2f}")
        mask = segmenter.segment(image, bbox)
        if mask is None:
            logger.debug("  cascade: HQ-SAM returned no mask")
            return None
        quad = segmenter.mask_to_quad(mask)
        if quad is None:
            logger.debug("  cascade: mask_to_quad failed")
            return None
        return quad
    except FileNotFoundError as e:
        logger.info(f"  cascade unavailable ({e.filename or 'missing weights'}); falling back to legacy")
        return None
    except ImportError as e:
        logger.info(f"  cascade unavailable (missing dependency: {e}); falling back to legacy")
        return None
    except Exception as e:
        logger.warning(f"  cascade failed unexpectedly: {e}")
        return None


def _detect_via_legacy(image: np.ndarray) -> Optional[np.ndarray]:
    """Legacy SA24 + LCNet100 ONNX chain. Returns raw (unvalidated) 4 corners or None."""
    if not HAS_LEGACY_ML:
        return None
    try:
        result = detect_corners_onnx(image)
        if result is None or not hasattr(result, 'shape') or result.shape != (4, 2):
            return None
        return np.asarray(result, dtype=np.float32)
    except Exception as e:
        logger.warning(f"  legacy ML failed: {e}")
        return None


# ---------- public API ----------

def detect_corners_ml(image: np.ndarray,
                      config: Optional[ScanConfig] = None) -> Optional[np.ndarray]:
    """Detect document corners on a single image (no rotation retry).

    Tries the cascade first when `config.use_cascade` is True; falls back
    to the legacy ML chain otherwise or on cascade failure. Both paths
    are validated and repaired through the shared geometry pipeline.

    Returns ordered corners (TL, TR, BR, BL) or None.
    """
    if config is None:
        config = ScanConfig()

    h, w = image.shape[:2]
    min_coverage = config.min_doc_coverage

    if config.use_cascade:
        raw = _detect_via_cascade(image, config)
        if raw is not None:
            validated = _validate_and_repair(raw, h, w, min_coverage)
            if validated is not None:
                logger.info("  corners via cascade")
                return validated

    raw = _detect_via_legacy(image)
    if raw is not None:
        validated = _validate_and_repair(raw, h, w, min_coverage)
        if validated is not None:
            logger.info("  corners via legacy ML")
            return validated

    return None


def detect_corners(image: np.ndarray,
                   config: Optional[ScanConfig] = None
                   ) -> Tuple[Optional[np.ndarray], int]:
    """Detect document corners with rotation retry.

    Tries detection on the original image; on failure, retries at 90 and
    270 degrees CCW. Returns (corners, rotation_k) where rotation_k is
    the number of 90° CCW rotations applied to find the document.

    Returns:
        (corners, rotation_k) or (None, 0) on total failure.
    """
    if config is None:
        config = ScanConfig()

    if not config.use_ml:
        return None, 0

    corners = detect_corners_ml(image, config)
    if corners is not None:
        return corners, 0

    for k in (1, 3):
        rotated = np.rot90(image, k=k)
        corners = detect_corners_ml(rotated, config)
        if corners is not None:
            logger.info(f"  detection succeeded after {k * 90}° CCW rotation")
            return corners, k

    return None, 0
