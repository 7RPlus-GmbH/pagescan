"""Document corner detection with ML validation and geometric repair."""

import logging
from typing import Optional

import cv2
import numpy as np

from pagescan.config import ScanConfig

logger = logging.getLogger(__name__)

try:
    from docaligner import DocAligner
    _docaligner_model = None
    HAS_DOCALIGNER = True
except ImportError:
    HAS_DOCALIGNER = False


def order_corners(pts: np.ndarray) -> np.ndarray:
    """Order 4 points as: top-left, top-right, bottom-right, bottom-left."""
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).flatten()
    rect[0] = pts[np.argmin(s)]   # top-left: smallest x+y
    rect[2] = pts[np.argmax(s)]   # bottom-right: largest x+y
    rect[1] = pts[np.argmin(d)]   # top-right: smallest x-y
    rect[3] = pts[np.argmax(d)]   # bottom-left: largest x-y
    return rect


def _edge_angle(v: np.ndarray) -> float:
    """Angle of a 2D vector in degrees."""
    return np.degrees(np.arctan2(v[1], v[0]))


def _check_parallel(pts: np.ndarray):
    """Check parallelism of opposite edges. Returns (top-bottom diff, left-right diff)."""
    top = pts[1] - pts[0]
    bottom = pts[2] - pts[3]
    left = pts[3] - pts[0]
    right = pts[2] - pts[1]
    tb = abs(_edge_angle(top) - _edge_angle(bottom))
    lr = abs(_edge_angle(left) - _edge_angle(right))
    return tb, lr


def _check_quad_dimensions(pts: np.ndarray, h: int, w: int) -> bool:
    """Validate that the detected quad covers a reasonable portion of the image."""
    wt = np.linalg.norm(pts[1] - pts[0])
    wb = np.linalg.norm(pts[2] - pts[3])
    hl = np.linalg.norm(pts[3] - pts[0])
    hr = np.linalg.norm(pts[2] - pts[1])

    # Short side must be >= 45% of image short side
    quad_short = min(max(wt, wb), max(hl, hr))
    img_short = min(h, w)
    if quad_short < img_short * 0.45:
        logger.debug(f"  ML quad too narrow: {quad_short:.0f} < {img_short * 0.45:.0f}")
        return False

    # Reject near-square aspect ratios (real documents are portrait or landscape)
    quad_w = max(wt, wb)
    quad_h = max(hl, hr)
    aspect = quad_w / quad_h if quad_h > 0 else 1.0
    if 0.88 < aspect < 1.15:
        logger.debug(f"  ML quad nearly square (aspect={aspect:.2f}), likely partial detection")
        return False

    return True


def _repair_corners(ordered: np.ndarray, tb_diff: float, lr_diff: float,
                    h: int, w: int) -> Optional[np.ndarray]:
    """Attempt to repair non-parallel corners by adjusting the outlier.

    When top/bottom angles diverge, one bottom corner is misplaced.
    Reconstruct it to make bottom parallel to top (keeping the longer
    side's endpoint fixed). Same logic for left/right divergence.
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
            repaired[1] = repaired[0] + (ordered[1] - ordered[0]) / np.linalg.norm(ordered[1] - ordered[0]) * top_len

    tb2, lr2 = _check_parallel(repaired)
    if tb2 <= 10 and lr2 <= 10:
        repair_area = cv2.contourArea(repaired)
        repair_coverage = repair_area / (h * w)
        if 0.10 < repair_coverage < 0.98 and _check_quad_dimensions(repaired, h, w):
            logger.info(f"  ML corners repaired: TB {tb_diff:.1f}->{tb2:.1f} LR {lr_diff:.1f}->{lr2:.1f}")
            return repaired

    return None


def detect_corners_ml(image: np.ndarray) -> Optional[np.ndarray]:
    """Detect document corners using DocAligner deep learning model.

    Returns ordered corners (TL, TR, BR, BL) or None if detection fails.
    Validates coverage, aspect ratio, and edge parallelism. Attempts
    geometric repair when opposite edges are not parallel.
    """
    if not HAS_DOCALIGNER:
        return None

    global _docaligner_model
    if _docaligner_model is None:
        _docaligner_model = DocAligner()

    result = _docaligner_model(image)
    if result is None or not hasattr(result, 'shape') or result.shape != (4, 2):
        return None

    corners = result.astype(np.float32)
    h, w = image.shape[:2]

    # Coverage validation
    area = cv2.contourArea(corners)
    coverage = area / (h * w)
    if coverage < 0.10 or coverage > 0.98:
        return None

    ordered = order_corners(corners)

    if not _check_quad_dimensions(ordered, h, w):
        return None

    # Parallelism check
    tb_diff, lr_diff = _check_parallel(ordered)
    if tb_diff <= 10 and lr_diff <= 10:
        return ordered

    # Attempt repair
    repaired = _repair_corners(ordered, tb_diff, lr_diff, h, w)
    if repaired is not None:
        return repaired

    logger.debug(f"  ML corners not parallel: TB={tb_diff:.1f} LR={lr_diff:.1f}")
    return None


def detect_corners(image: np.ndarray, config: ScanConfig = None) -> Optional[np.ndarray]:
    """Detect document corners with rotation retry.

    Tries ML detection on the original image, then retries at 90 and 270
    degree rotations if the first attempt fails. Returns (corners, rotation_k)
    where rotation_k indicates how many 90-degree CCW rotations were applied.

    Returns:
        Tuple of (corners, rotation_k) or (None, 0) if detection fails.
    """
    if config is None:
        config = ScanConfig()

    if not config.use_ml:
        return None, 0

    corners = detect_corners_ml(image)
    if corners is not None:
        return corners, 0

    for k in [1, 3]:
        rotated = np.rot90(image, k=k)
        corners = detect_corners_ml(rotated)
        if corners is not None:
            logger.info(f"  ML detection succeeded after {k * 90} CCW rotation")
            return corners, k

    return None, 0
