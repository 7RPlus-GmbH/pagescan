"""Edge detection and background trimming for the conservative-crop fallback path.

Used only when ML corner detection fails entirely. The strategies are
ordered by aggressiveness; the conservative path picks the first one
that produces a plausible crop.

Public surface:
    find_paper_contour       - Contour-based paper region detection
    find_document_edges      - HSV-mask based bbox
    detect_corners_contour   - 4-corner quad from largest contour
    detect_paper_quad        - Stricter 4-corner quad with paper-shape priors
    estimate_paper_coverage  - Sanity-check helper for over-crop guards
"""
from __future__ import annotations

import logging

import cv2
import numpy as np

from pagescan.config import ScanConfig

logger = logging.getLogger(__name__)


def find_paper_contour(image: np.ndarray, config: ScanConfig | None = None,
                       min_area_ratio: float = 0.05) -> tuple[int, int, int, int]:
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
    paper = cv2.inRange(hsv, (0, 0, 120), (180, 100, 255))  # type: ignore[call-overload]

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
    total_area = 0.0
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

        _bx, _by, bw, bh = cv2.boundingRect(c)
        rect_area = bw * bh
        if rect_area < 1:  # pragma: no cover - findContours rects are always >=1px
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


def detect_corners_contour(image: np.ndarray, config: ScanConfig | None = None):
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


def find_document_edges(image: np.ndarray, config: ScanConfig | None = None) -> tuple[int, int, int, int]:
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

    _, contour, _approx = scored[0]
    bx, by, bw, bh = cv2.boundingRect(contour)

    margin_x = max(10, int(w * 0.01))
    margin_y = max(10, int(h * 0.01))
    x1 = max(0, bx - margin_x)
    y1 = max(0, by - margin_y)
    x2 = min(w, bx + bw + margin_x)
    y2 = min(h, by + bh + margin_y)

    logger.info(f"  Edge detection: ({x1},{y1}) {x2 - x1}x{y2 - y1}")
    return y1, y2, x1, x2


def detect_paper_quad(image: np.ndarray, config: ScanConfig | None = None) -> np.ndarray | None:
    """Detect document boundary via paper-mask segmentation.

    Works by finding bright, low-saturation pixels (paper) and fitting a
    quadrilateral around them. Uses aggressive morphological closing to
    bridge fold lines, shadows, and other internal features that confuse
    edge-based detection.

    Key advantage over edge-based detection: fold lines in letters create
    strong edges but do NOT break the paper mask, so this method handles
    folded documents naturally.

    Returns 4 corners as np.ndarray shape (4, 2) or None.
    """
    if config is None:
        config = ScanConfig()

    h, w = image.shape[:2]
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Paper mask: low saturation, reasonably bright
    # S<100 handles paper under warm indoor lighting; V>100 is slightly more
    # permissive than find_paper_contour (V>120) to catch fold-line shadows
    paper = cv2.inRange(hsv, (0, 0, 100), (180, 100, 255))  # type: ignore[call-overload]

    # Aggressive morphological closing to bridge fold lines
    # Fold lines are typically 20-80px wide on a 3024px image; we need a
    # kernel large enough to close them in a few iterations
    short = min(h, w)
    ks = max(51, short // 30) | 1
    kernel = np.ones((ks, ks), np.uint8)
    paper = cv2.morphologyEx(paper, cv2.MORPH_CLOSE, kernel, iterations=5)
    paper = cv2.morphologyEx(paper, cv2.MORPH_OPEN, kernel, iterations=2)

    contours, _ = cv2.findContours(paper, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    # Merge all significant contours via convex hull
    # This handles cases where fold lines still split the paper into 2-3 regions
    significant = [c for c in contours
                   if cv2.contourArea(c) >= h * w * 0.05]
    if not significant:
        return None

    all_pts = np.vstack(significant)
    hull = cv2.convexHull(all_pts)
    hull_area = cv2.contourArea(hull)
    coverage = hull_area / (h * w)

    if coverage < 0.15:
        return None

    # Fit a 4-sided polygon
    peri = cv2.arcLength(hull, True)
    for eps in [0.015, 0.02, 0.03, 0.04, 0.05, 0.07, 0.10]:
        approx = cv2.approxPolyDP(hull, eps * peri, True)
        if len(approx) == 4:
            pts = approx.reshape(4, 2).astype(np.float32)
            # Validate angles (should be roughly rectangular)
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
                logger.info(f"  Paper quad: coverage={coverage:.2f} (eps={eps})")
                return pts

    # Fallback: minimum area rotated rectangle
    rect = cv2.minAreaRect(hull)
    box = cv2.boxPoints(rect).astype(np.float32)
    logger.info(f"  Paper quad (minAreaRect): coverage={coverage:.2f}")
    return box


def estimate_paper_coverage(image: np.ndarray) -> float:
    """Estimate what fraction of the image is paper (bright, low-saturation).

    Quick check used to cross-validate ML corner detection: if ML detects
    a small region but paper fills most of the image, the ML corners are
    probably wrong (detecting a fold section, not the whole document).
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    paper = cv2.inRange(hsv, (0, 0, 100), (180, 100, 255))  # type: ignore[call-overload]
    return float(np.mean(paper > 0))


