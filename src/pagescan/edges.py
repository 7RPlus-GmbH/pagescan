"""Edge detection and background trimming.

Three complementary strategies for removing background (wood, shadow)
from document edges after perspective transform or direct crop:

1. trim_edges      - Strip-scan each edge for heavy contamination
2. find_precise_edges - Two-phase strip-scan + column density analysis
3. find_paper_contour - Contour-based paper region detection (fallback)
"""

import logging
from typing import Tuple

import cv2
import numpy as np

from pagescan.config import ScanConfig

logger = logging.getLogger(__name__)


def trim_edges(image: np.ndarray, config: ScanConfig = None,
               max_trim_ratio: float = 0.08) -> np.ndarray:
    """Trim background contamination from edges after perspective transform.

    Analyzes each edge strip for high-saturation (wood) or dark (shadow)
    pixels and crops inward until clean paper is reached.

    Uses corner-aware trimming: checks triangular corner regions for
    contamination and trims enough to eliminate them, even if the middle
    of an edge strip is clean.

    Args:
        image: Document image (BGR).
        config: Scan configuration (for HSV ranges).
        max_trim_ratio: Maximum fraction of image to trim from any single edge.
    """
    if config is None:
        config = ScanConfig()

    h, w = image.shape[:2]
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    max_trim_x = int(w * max_trim_ratio)
    max_trim_y = int(h * max_trim_ratio)

    bg_low = config.background_hsv_low
    bg_high = config.background_hsv_high

    def is_contaminated(strip_hsv, strip_gray):
        if strip_hsv.size == 0 or strip_gray.size == 0:
            return False
        wood = cv2.inRange(strip_hsv, bg_low, bg_high)
        wood_ratio = np.sum(wood > 0) / wood.size
        shadow_ratio = np.sum(strip_gray < 50) / strip_gray.size
        paper = cv2.inRange(strip_hsv, (0, 0, 120), (180, 50, 255))
        paper_ratio = np.sum(paper > 0) / paper.size
        return (wood_ratio > 0.5 or shadow_ratio > 0.5) and paper_ratio < 0.3

    strip_width = max(5, min(15, w // 50, h // 50))

    # Top
    top_trim = 0
    for y in range(0, max_trim_y, strip_width):
        if is_contaminated(hsv[y:y + strip_width, w // 4:3 * w // 4],
                           gray[y:y + strip_width, w // 4:3 * w // 4]):
            top_trim = y + strip_width
        else:
            break

    # Bottom
    bottom_trim = 0
    for y in range(h - strip_width, h - max_trim_y, -strip_width):
        if is_contaminated(hsv[y:y + strip_width, w // 4:3 * w // 4],
                           gray[y:y + strip_width, w // 4:3 * w // 4]):
            bottom_trim = h - y
        else:
            break

    # Left
    left_trim = 0
    for x in range(0, max_trim_x, strip_width):
        if is_contaminated(hsv[h // 4:3 * h // 4, x:x + strip_width],
                           gray[h // 4:3 * h // 4, x:x + strip_width]):
            left_trim = x + strip_width
        else:
            break

    # Right
    right_trim = 0
    for x in range(w - strip_width, w - max_trim_x, -strip_width):
        if is_contaminated(hsv[h // 4:3 * h // 4, x:x + strip_width],
                           gray[h // 4:3 * h // 4, x:x + strip_width]):
            right_trim = w - x
        else:
            break

    # Corner-aware trimming: check corner quadrants for contamination
    # that edge-strip scanning misses (e.g. triangular background from tilt)
    corner_size_y = min(h // 8, max_trim_y)
    corner_size_x = min(w // 8, max_trim_x)
    corner_regions = [
        (hsv[0:corner_size_y, 0:corner_size_x],
         gray[0:corner_size_y, 0:corner_size_x], 'TL'),
        (hsv[0:corner_size_y, w-corner_size_x:w],
         gray[0:corner_size_y, w-corner_size_x:w], 'TR'),
        (hsv[h-corner_size_y:h, 0:corner_size_x],
         gray[h-corner_size_y:h, 0:corner_size_x], 'BL'),
        (hsv[h-corner_size_y:h, w-corner_size_x:w],
         gray[h-corner_size_y:h, w-corner_size_x:w], 'BR'),
    ]
    for c_hsv, c_gray, label in corner_regions:
        if is_contaminated(c_hsv, c_gray):
            # This corner has heavy background — ensure trim reaches it
            extra = strip_width * 2
            if label[0] == 'T' and top_trim < extra:
                top_trim = min(extra, max_trim_y)
            if label[0] == 'B' and bottom_trim < extra:
                bottom_trim = min(extra, max_trim_y)
            if label[1] == 'L' and left_trim < extra:
                left_trim = min(extra, max_trim_x)
            if label[1] == 'R' and right_trim < extra:
                right_trim = min(extra, max_trim_x)

    if top_trim > 0 or bottom_trim > 0 or left_trim > 0 or right_trim > 0:
        new_h = h - top_trim - bottom_trim
        new_w = w - left_trim - right_trim
        if new_h > h * 0.5 and new_w > w * 0.5:
            logger.info(f"  Trimming edges: T={top_trim} B={bottom_trim} L={left_trim} R={right_trim}")
            return image[top_trim:h - bottom_trim if bottom_trim > 0 else h,
                         left_trim:w - right_trim if right_trim > 0 else w]

    return image


def find_precise_edges(image: np.ndarray, config: ScanConfig = None,
                       max_scan_ratio: float = 0.15,
                       min_keep_ratio: float = 0.50) -> Tuple[int, int, int, int]:
    """Two-phase edge detection to crop background.

    Phase 1 (Top/Bottom): Strip-scan using the middle 2/3 of width to
    avoid corner triangle contamination. Uses HSV to separate background
    from document content (including colored headers).

    Phase 2 (Left/Right): Per-column background density within the
    Phase-1-cropped vertical range, eliminating corner triangles.

    Returns (top, bottom, left, right) crop coordinates.
    """
    if config is None:
        config = ScanConfig()

    h, w = image.shape[:2]
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    strip_w = max(3, min(10, w // 100, h // 100))
    bg_low = config.background_hsv_low
    bg_high = config.background_hsv_high
    strict_s = config.background_hsv_strict_s

    def is_background(strip_hsv, strip_gray):
        wood = cv2.inRange(strip_hsv, (bg_low[0], strict_s, bg_low[2]), bg_high)
        wood_ratio = np.sum(wood > 0) / max(wood.size, 1)
        shadow_ratio = np.sum(strip_gray < 50) / max(strip_gray.size, 1)
        paper = cv2.inRange(strip_hsv, (0, 0, 120), (180, 50, 255))
        paper_ratio = np.sum(paper > 0) / max(paper.size, 1)
        header = cv2.inRange(strip_hsv, (0, 130, 180), (25, 255, 255))
        header2 = cv2.inRange(strip_hsv, (165, 130, 180), (180, 255, 255))
        header_ratio = (np.sum(header > 0) + np.sum(header2 > 0)) / max(header.size, 1)
        doc_ratio = paper_ratio + header_ratio
        if doc_ratio > 0.5:
            return False
        return wood_ratio > 0.35 or shadow_ratio > 0.35

    max_scan = int(min(h, w) * max_scan_ratio)

    def _find_edge_forward(positions, get_strip):
        last_bg = -1
        doc_run = 0
        for pos in positions:
            s_hsv, s_gray = get_strip(pos)
            if is_background(s_hsv, s_gray):
                if doc_run < 2:
                    last_bg = pos
                    doc_run = 0
                else:
                    break
            else:
                doc_run += 1
        return last_bg

    # Phase 1: Top/Bottom (middle 2/3 width)
    x1, x2 = w // 6, 5 * w // 6

    positions = list(range(0, min(max_scan, h - strip_w), strip_w))
    last_bg = _find_edge_forward(
        positions, lambda y: (hsv[y:y + strip_w, x1:x2], gray[y:y + strip_w, x1:x2]))
    top = (last_bg + strip_w) if last_bg >= 0 else 0

    positions = list(range(h - strip_w, max(h - max_scan, top + strip_w), -strip_w))
    last_bg = _find_edge_forward(
        positions, lambda y: (hsv[y:y + strip_w, x1:x2], gray[y:y + strip_w, x1:x2]))
    bottom = last_bg if last_bg >= 0 else h

    # Phase 2: Left/Right (column density within cropped vertical range)
    wood_mask = cv2.inRange(hsv, (bg_low[0], strict_s, bg_low[2]), bg_high)
    shadow_mask = (gray < 50).astype(np.uint8) * 255
    bg_mask = cv2.bitwise_or(wood_mask, shadow_mask)

    crop_h = bottom - top
    y1 = top + crop_h // 6
    y2 = bottom - crop_h // 6
    col_bg = np.mean(bg_mask[y1:y2, :] > 0, axis=0)
    bg_threshold = 0.10

    left = 0
    for x in range(0, min(max_scan, w), strip_w):
        if np.mean(col_bg[x:x + strip_w]) > bg_threshold:
            left = x + strip_w
        else:
            break

    right = w
    for x in range(w - strip_w, max(w - max_scan, left + strip_w), -strip_w):
        if np.mean(col_bg[x:x + strip_w]) > bg_threshold:
            right = x
        else:
            break

    crop_h = bottom - top
    crop_w = right - left
    if crop_h < h * min_keep_ratio or crop_w < w * min_keep_ratio:
        logger.warning(f"  Precise edges would over-crop, keeping full image")
        return 0, h, 0, w

    return top, bottom, left, right


def find_paper_contour(image: np.ndarray, config: ScanConfig = None,
                       min_area_ratio: float = 0.05) -> Tuple[int, int, int, int]:
    """Find the largest paper region via contour detection.

    Fallback for when find_precise_edges fails (e.g. small document on
    large background). Uses HSV paper mask (low saturation, bright) with
    morphology to find the largest contiguous paper region.

    Returns (top, bottom, left, right) crop coordinates.
    """
    if config is None:
        config = ScanConfig()

    h, w = image.shape[:2]
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Saturation up to 100 to handle paper under warm indoor lighting
    paper = cv2.inRange(hsv, (0, 0, 120), (180, 100, 255))

    short = min(h, w)
    ks = max(25, short // 60) | 1
    kernel = np.ones((ks, ks), np.uint8)
    paper = cv2.morphologyEx(paper, cv2.MORPH_CLOSE, kernel, iterations=3)
    paper = cv2.morphologyEx(paper, cv2.MORPH_OPEN, kernel, iterations=2)

    contours, _ = cv2.findContours(paper, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0, h, 0, w

    significant = [c for c in contours if cv2.contourArea(c) >= h * w * min_area_ratio]
    if not significant:
        return 0, h, 0, w

    x1, y1, x2, y2 = w, h, 0, 0
    total_area = 0
    for c in significant:
        bx, by, bw, bh = cv2.boundingRect(c)
        x1 = min(x1, bx)
        y1 = min(y1, by)
        x2 = max(x2, bx + bw)
        y2 = max(y2, by + bh)
        total_area += cv2.contourArea(c)

    margin_x = max(10, int(w * 0.01))
    margin_y = max(10, int(h * 0.01))
    x1 = max(0, x1 - margin_x)
    y1 = max(0, y1 - margin_y)
    x2 = min(w, x2 + margin_x)
    y2 = min(h, y2 + margin_y)

    logger.info(f"  Paper contour: ({x1},{y1}) {x2 - x1}x{y2 - y1}, "
                f"area={total_area / (h * w) * 100:.0f}% ({len(significant)} regions)")
    return y1, y2, x1, x2


def _find_document_contours(image: np.ndarray):
    """Shared edge analysis: returns scored contours sorted by quality.

    Strategy: downscale to ~500px (suppresses text, keeps document boundary),
    then Canny + morphology to find contours. Coordinates are scaled back
    to original resolution.

    Returns list of (score, contour, approx_polygon) tuples, best first.
    """
    h, w = image.shape[:2]

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Downscale to suppress text edges while keeping document boundary
    max_dim = 500
    scale = min(1.0, max_dim / max(h, w))
    if scale < 1.0:
        small = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    else:
        small = gray

    sh, sw = small.shape[:2]

    # Light blur to smooth remaining noise
    blurred = cv2.GaussianBlur(small, (5, 5), 1)

    # Multi-scale Canny
    edges1 = cv2.Canny(blurred, 20, 60)
    edges2 = cv2.Canny(blurred, 40, 120)
    edges = cv2.bitwise_or(edges1, edges2)

    # Dilate to connect edge fragments into continuous boundary
    k_size = max(3, min(sh, sw) // 50) | 1
    kernel = np.ones((k_size, k_size), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=3)
    edges = cv2.erode(edges, kernel, iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []

    scored = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < sh * sw * 0.05:
            continue

        peri = cv2.arcLength(c, True)

        # Try multiple epsilon values to find 4-sided approximation
        approx = None
        for eps in [0.015, 0.02, 0.03, 0.04, 0.05, 0.07, 0.10]:
            a = cv2.approxPolyDP(c, eps * peri, True)
            if len(a) == 4:
                # Validate: all angles should be roughly 60-120 degrees
                pts = a.reshape(4, 2)
                angles_ok = True
                for i in range(4):
                    v1 = pts[(i - 1) % 4] - pts[i]
                    v2 = pts[(i + 1) % 4] - pts[i]
                    cos_a = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
                    angle = np.degrees(np.arccos(np.clip(cos_a, -1, 1)))
                    if angle < 50 or angle > 140:
                        angles_ok = False
                        break
                if angles_ok:
                    approx = a
                    break
        if approx is None:
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        bx, by, bw, bh = cv2.boundingRect(c)
        rect_area = bw * bh
        if rect_area < 1:
            continue
        rectangularity = area / rect_area

        vertex_bonus = 1.5 if len(approx) == 4 else 1.0
        score = area * rectangularity * vertex_bonus

        # Scale contour and approx back to original resolution
        if scale < 1.0:
            c_scaled = (c.astype(np.float32) / scale).astype(np.int32)
            approx_scaled = (approx.astype(np.float32) / scale).astype(np.int32)
        else:
            c_scaled = c
            approx_scaled = approx

        scored.append((score, c_scaled, approx_scaled))

    scored.sort(key=lambda x: x[0], reverse=True)
    return scored


def detect_corners_contour(image: np.ndarray, config: ScanConfig = None):
    """Detect document corners via edge-based contour analysis.

    Background-agnostic alternative to ML corner detection. Returns
    4 corners as np.ndarray shape (4, 2) or None.

    This is the key improvement over bounding-box fallback: returns
    a proper quadrilateral that matches the document's actual tilt,
    dramatically improving IoU on tilted documents.
    """
    if config is None:
        config = ScanConfig()

    scored = _find_document_contours(image)
    if not scored:
        return None

    # Take the best contour
    _, contour, approx = scored[0]

    if len(approx) == 4:
        # Perfect: got a 4-sided polygon
        corners = approx.reshape(4, 2).astype(np.float32)
        return corners

    # Fallback: use minimum area rotated rectangle (still a proper quad)
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    return box.astype(np.float32)


def find_document_edges(image: np.ndarray, config: ScanConfig = None) -> Tuple[int, int, int, int]:
    """Background-agnostic document detection via edge analysis.

    Works on ANY background by detecting the document's sharp edges
    rather than trying to classify background pixels by color.

    Returns (top, bottom, left, right) crop coordinates for the pipeline.
    Falls back to find_paper_contour if edge detection finds nothing.
    """
    if config is None:
        config = ScanConfig()

    h, w = image.shape[:2]

    scored = _find_document_contours(image)
    if not scored:
        return find_paper_contour(image, config)

    _, contour, approx = scored[0]
    bx, by, bw, bh = cv2.boundingRect(contour)

    margin_x = max(10, int(w * 0.01))
    margin_y = max(10, int(h * 0.01))
    x1 = max(0, bx - margin_x)
    y1 = max(0, by - margin_y)
    x2 = min(w, bx + bw + margin_x)
    y2 = min(h, by + bh + margin_y)

    logger.info(f"  Edge detection: ({x1},{y1}) {x2 - x1}x{y2 - y1}")
    return y1, y2, x1, x2


def find_receipt_bounds(image: np.ndarray, config: ScanConfig = None) -> Tuple[int, int, int, int]:
    """Find bounding box of a small receipt on a background surface.

    Inverse approach: detect BACKGROUND (warm hue + medium-high saturation),
    then find the largest non-background region. This works better than
    detecting paper directly because some backgrounds (e.g. end-grain wood)
    are very bright, making brightness-based separation unreliable.

    Returns (y1, y2, x1, x2) or raises ValueError if no receipt found.
    """
    if config is None:
        config = ScanConfig()

    h, w = image.shape[:2]
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    bg_low = config.background_hsv_low
    bg_high = config.background_hsv_high
    wood = cv2.inRange(hsv, (bg_low[0], max(60, bg_low[1]), max(50, bg_low[2])),
                       bg_high)

    kernel = np.ones((15, 15), np.uint8)
    wood = cv2.morphologyEx(wood, cv2.MORPH_CLOSE, kernel, iterations=5)
    non_wood = cv2.bitwise_not(wood)
    non_wood = cv2.morphologyEx(non_wood, cv2.MORPH_OPEN, kernel, iterations=2)

    contours, _ = cv2.findContours(non_wood, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("No receipt found on background")

    candidates = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < h * w * 0.05:
            continue
        x, y, cw, ch = cv2.boundingRect(c)
        candidates.append((area, x, y, cw, ch))

    if not candidates:
        raise ValueError("No receipt region large enough (min 5% of image)")

    candidates.sort(reverse=True)
    _, x, y, cw, ch = candidates[0]

    margin = 20
    x1 = max(0, x - margin)
    y1 = max(0, y - margin)
    x2 = min(w, x + cw + margin)
    y2 = min(h, y + ch + margin)

    return y1, y2, x1, x2
