"""Document orientation correction: deskew and auto-rotation."""

import logging
from typing import Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def deskew(image: np.ndarray) -> Tuple[np.ndarray, float]:
    """Correct text skew using Hough line detection.

    Detects near-horizontal lines (text baselines, rules, table borders)
    via probabilistic Hough transform on the center 60% of the image
    (avoids background edges). Uses median angle for robustness to outliers.

    Returns (corrected_image, detected_angle). Angle is 0.0 if no
    correction was needed.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    h, w = gray.shape

    # Center 60% to avoid background edges
    mh, mw = h // 5, w // 5
    center = gray[mh:h - mh, mw:w - mw]

    edges = cv2.Canny(center, 50, 150, apertureSize=3)

    # Bridge text into horizontal strokes
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 3))
    edges = cv2.dilate(edges, kernel, iterations=1)
    edges = cv2.erode(edges, kernel, iterations=1)

    lines = cv2.HoughLinesP(edges, 1, np.pi / 720, threshold=80,
                            minLineLength=min(center.shape[1] // 8, 200),
                            maxLineGap=20)
    if lines is None or len(lines) < 3:
        return image, 0.0

    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if abs(x2 - x1) < 10:
            continue
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        if abs(angle) < 25:
            angles.append(angle)

    if len(angles) < 3:
        return image, 0.0

    median_angle = float(np.median(angles))

    if abs(median_angle) < 0.3 or abs(median_angle) > 25:
        return image, 0.0

    logger.info(f"  Deskew: {median_angle:.1f} ({len(angles)} lines)")

    h, w = image.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), median_angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h),
                             flags=cv2.INTER_CUBIC,
                             borderValue=(255, 255, 255))
    return rotated, median_angle


def _ocr_word_score(gray_image: np.ndarray) -> int:
    """Count high-confidence OCR words as a proxy for correct orientation."""
    try:
        import pytesseract
        from pytesseract import Output
        data = pytesseract.image_to_data(
            gray_image, lang='deu', config='--psm 6',
            output_type=Output.DICT)
        return sum(1 for c in data['conf'] if int(c) > 50)
    except Exception:
        return 0


def auto_rotate(image: np.ndarray) -> np.ndarray:
    """Auto-rotate document to correct orientation using OCR word scoring.

    Tests all 4 orientations (0, 90, 180, 270) and picks the one that
    produces the most high-confidence OCR words. This is more reliable than
    Tesseract OSD which often fails on colorful or non-text-heavy documents.

    Requires pytesseract (optional dependency). Returns image unchanged
    if pytesseract is not installed.
    """
    try:
        import pytesseract
        from pytesseract import Output
    except ImportError:
        logger.debug("  pytesseract not available, skipping auto-rotate")
        return image

    h, w = image.shape[:2]

    # Prepare a smaller grayscale version for OCR speed
    ocr_scale = min(1.0, 1000 / max(h, w))
    small = cv2.resize(image, None, fx=ocr_scale, fy=ocr_scale) if ocr_scale < 1.0 else image
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY) if small.ndim == 3 else small

    # First try OSD (fast) — if it works with high confidence, use it
    try:
        osd = pytesseract.image_to_osd(gray, output_type=Output.DICT)
        osd_rot = osd.get('rotate', 0)
        osd_conf = osd.get('orientation_conf', 0)
        if osd_conf >= 5.0 and osd_rot != 0:
            # OSD is very confident — map rotation to np.rot90 k value
            k_map = {90: 3, 180: 2, 270: 1}
            best_k = k_map.get(osd_rot, 0)
            if best_k != 0:
                logger.info(f"  Auto-rotating {best_k * 90} CCW (OSD conf={osd_conf:.1f})")
                return np.rot90(image, k=best_k)
    except Exception:
        pass

    # OSD failed or low confidence — use OCR word scoring on all 4 orientations
    scores = {}
    for k in [0, 1, 2, 3]:
        rotated = np.rot90(gray, k=k) if k != 0 else gray
        scores[k] = _ocr_word_score(rotated)

    best_k = max(scores, key=scores.get)
    best_score = scores[best_k]
    current_score = scores[0]

    logger.debug(f"  OCR scores: 0°={scores[0]} 90°={scores[1]} 180°={scores[2]} 270°={scores[3]}")

    # Only rotate if the best orientation is clearly better than current
    if best_k != 0 and best_score > current_score * 1.2 and best_score >= 3:
        logger.info(f"  Auto-rotating {best_k * 90} CCW (OCR: {best_score} words vs {current_score} at 0°)")
        image = np.rot90(image, k=best_k)
    elif best_k != 0 and best_score > 0 and current_score == 0:
        # Current orientation has zero OCR hits but another has some
        logger.info(f"  Auto-rotating {best_k * 90} CCW (OCR: {best_score} words, current=0)")
        image = np.rot90(image, k=best_k)

    return image
