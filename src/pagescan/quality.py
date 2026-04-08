"""Quality assessment for scanned documents."""

import logging
from typing import Tuple

import cv2
import numpy as np

from pagescan.config import ScanConfig

logger = logging.getLogger(__name__)


def check_quality(image: np.ndarray, config: ScanConfig = None) -> Tuple[bool, float, str]:
    """Check for background contamination at document corners.

    Samples 5% of each corner and measures background (wood/shadow) ratio
    using a strict saturation threshold to avoid false positives on
    cream-colored paper.

    Returns (passed, score, message) where score is 1.0 (clean) to 0.0 (all background).
    """
    if config is None:
        config = ScanConfig()

    h, w = image.shape[:2]
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    edge_h = max(20, h // 20)
    edge_w = max(20, w // 20)

    corners = [
        hsv[0:edge_h, 0:edge_w],
        hsv[0:edge_h, w - edge_w:w],
        hsv[h - edge_h:h, 0:edge_w],
        hsv[h - edge_h:h, w - edge_w:w],
    ]

    bg_low = config.background_hsv_low
    bg_high = config.background_hsv_high
    strict_s = config.background_hsv_strict_s

    wood_ratios = []
    for corner in corners:
        wood = cv2.inRange(corner, (bg_low[0], strict_s, bg_low[2]), bg_high)
        ratio = np.sum(wood > 0) / max(corner.size // 3, 1)
        wood_ratios.append(ratio)

    max_wood = max(wood_ratios)
    avg_wood = sum(wood_ratios) / 4
    score = 1.0 - avg_wood

    if max_wood > 0.50:
        return False, score, f"Significant background in corners (max={max_wood:.1%})"
    if avg_wood > 0.30:
        return False, score, f"Background contamination (avg={avg_wood:.1%})"

    return True, score, "OK"
