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


def auto_rotate(image: np.ndarray) -> np.ndarray:
    """Auto-rotate document to correct orientation using Tesseract OSD.

    Tries all 4 orientations and picks the one where Tesseract confirms
    text is upright (rotate=0) with highest confidence. Validates 90-degree
    rotations with OCR word scoring to avoid wrong-direction picks.

    Skips 180-degree rotation (OSD is unreliable for upside-down detection).
    Falls back to aspect-ratio heuristic if OSD fails entirely.

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
    scale = min(1.0, 1500 / max(h, w))
    small = cv2.resize(image, None, fx=scale, fy=scale) if scale < 1.0 else image
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY) if small.ndim == 3 else small

    best_k = 0
    best_conf = -1.0
    osd_failures = 0

    for k in [0, 1, 3, 2]:  # original, 90 CCW, 90 CW, 180
        rotated = np.rot90(gray, k=k) if k != 0 else gray
        try:
            osd = pytesseract.image_to_osd(rotated, output_type=Output.DICT)
            rot = osd.get('rotate', 0)
            conf = osd.get('orientation_conf', 0)
            if rot == 0 and conf > best_conf:
                best_k = k
                best_conf = conf
        except Exception:
            osd_failures += 1
            continue

    if best_k != 0 and best_conf > 0.5:
        # Skip 180 (unreliable)
        if best_k == 2:
            logger.debug(f"  Skipping 180 rotation (disabled, conf={best_conf:.1f})")
        else:
            # Validate 90-degree direction with OCR word count
            if best_k in (1, 3):
                other_k = 3 if best_k == 1 else 1
                try:
                    ocr_scale = min(1.0, 1000 / max(gray.shape))
                    ocr_img = cv2.resize(gray, None, fx=ocr_scale, fy=ocr_scale) if ocr_scale < 1.0 else gray
                    data_best = pytesseract.image_to_data(
                        np.rot90(ocr_img, k=best_k), lang='deu', config='--psm 6',
                        output_type=Output.DICT)
                    data_other = pytesseract.image_to_data(
                        np.rot90(ocr_img, k=other_k), lang='deu', config='--psm 6',
                        output_type=Output.DICT)
                    score_best = sum(1 for c in data_best['conf'] if int(c) > 50)
                    score_other = sum(1 for c in data_other['conf'] if int(c) > 50)
                    if score_other > score_best * 1.2:
                        logger.info(f"  OCR validates opposite direction: k={other_k} "
                                    f"({score_other} words) > k={best_k} ({score_best} words)")
                        best_k = other_k
                except Exception:
                    pass

            logger.info(f"  Auto-rotating {best_k * 90} CCW (conf={best_conf:.1f})")
            image = np.rot90(image, k=best_k)

    elif osd_failures >= 3 and w > h * 1.3:
        # OSD mostly failed + landscape: assume portrait, validate with OCR
        try:
            ocr_scale = min(1.0, 1000 / max(gray.shape))
            ocr_img = cv2.resize(gray, None, fx=ocr_scale, fy=ocr_scale) if ocr_scale < 1.0 else gray
            data_k1 = pytesseract.image_to_data(
                np.rot90(ocr_img, k=1), lang='deu', config='--psm 6',
                output_type=Output.DICT)
            data_k3 = pytesseract.image_to_data(
                np.rot90(ocr_img, k=3), lang='deu', config='--psm 6',
                output_type=Output.DICT)
            score_k1 = sum(1 for c in data_k1['conf'] if int(c) > 50)
            score_k3 = sum(1 for c in data_k3['conf'] if int(c) > 50)
            fallback_k = 1 if score_k1 >= score_k3 else 3
        except Exception:
            fallback_k = 1
        logger.info(f"  Auto-rotating {fallback_k * 90} CCW (aspect {w / h:.1f}:1, OSD failed)")
        image = np.rot90(image, k=fallback_k)

    return image
