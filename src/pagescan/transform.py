"""Perspective transform and A4/canvas placement."""

import logging

import cv2
import numpy as np

from pagescan.config import ScanConfig
from pagescan.corners import order_corners

logger = logging.getLogger(__name__)


def perspective_transform(image: np.ndarray, corners: np.ndarray) -> np.ndarray:
    """Apply perspective transform using detected corners.

    Preserves the document's original aspect ratio (no A4 forcing).
    Output dimensions are derived from the corner positions.
    """
    ordered = order_corners(corners)

    width_top = np.linalg.norm(ordered[1] - ordered[0])
    width_bot = np.linalg.norm(ordered[2] - ordered[3])
    width = int(max(width_top, width_bot))

    height_left = np.linalg.norm(ordered[3] - ordered[0])
    height_right = np.linalg.norm(ordered[2] - ordered[1])
    height = int(max(height_left, height_right))

    width = max(width, 200)
    height = max(height, 200)

    dst = np.array([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1]
    ], dtype=np.float32)

    M = cv2.getPerspectiveTransform(ordered, dst)
    return cv2.warpPerspective(image, M, (width, height), flags=cv2.INTER_LANCZOS4)


def place_on_canvas(image: np.ndarray, config: ScanConfig = None) -> np.ndarray:
    """Place document image on a white canvas (default A4 at 300 DPI) with margin."""
    if config is None:
        config = ScanConfig()

    h, w = image.shape[:2]
    is_gray = (image.ndim == 2)

    max_w = config.output_width - 2 * config.output_margin
    max_h = config.output_height - 2 * config.output_margin

    scale = min(max_w / w, max_h / h)
    new_w, new_h = int(w * scale), int(h * scale)

    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

    if is_gray:
        canvas = np.ones((config.output_height, config.output_width), dtype=np.uint8) * 255
    else:
        canvas = np.ones((config.output_height, config.output_width, 3), dtype=np.uint8) * 255

    x = (config.output_width - new_w) // 2
    y = (config.output_height - new_h) // 2
    canvas[y:y + new_h, x:x + new_w] = resized

    return canvas
