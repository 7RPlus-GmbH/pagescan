"""Document-scanner adapters for the comparison harness.

Each scanner exposes the same minimal contract:

    name: str
    detect(image_bgr: np.ndarray) -> Optional[np.ndarray]   # (4, 2) quad or None

Some scanners (`docscan`) only expose a warp operation, not corner
coordinates — they implement `warp(image)` directly and return None from
`detect`. The harness handles both cases.

The OpenCV recipe is included as the textbook contour-scanner baseline
that pagescan claims to beat. `docscan` is included if pip-installed
(it's a popular alternative library); silently skipped otherwise.
"""
from __future__ import annotations

import logging
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# ---------- base interface ----------

class Scanner:
    """ABC. Subclasses must set `name` and implement `detect` and/or `warp`."""

    name: str = "scanner"
    available: bool = True
    notes: str = ""  # human-readable note shown in the report

    def detect(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Return predicted (4, 2) quad in original-image coords, or None."""
        return None

    def warp(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Return scanner's warped output. Default: detect → perspective-warp."""
        from benchmark.comparison.metrics import perspective_warp
        quad = self.detect(image)
        if quad is None:
            return None
        return perspective_warp(image, quad)


# ---------- pagescan (cascade + legacy) ----------

class _PagescanBase(Scanner):
    use_cascade: bool

    def detect(self, image: np.ndarray) -> Optional[np.ndarray]:
        from pagescan.config import ScanConfig
        from pagescan.corners import detect_corners
        cfg = ScanConfig(use_cascade=self.use_cascade)
        corners, rotation_k, _method = detect_corners(image, cfg)
        if corners is None:
            return None
        if rotation_k != 0:
            # detect_corners returns corners in the rotated-image coordinate frame;
            # un-rotate them back to original coords.
            h_rot, w_rot = np.rot90(image, k=rotation_k).shape[:2]
            corners = _unrotate_corners(corners, rotation_k, h_rot, w_rot,
                                        image.shape[0], image.shape[1])
        return corners.astype(np.float32)


class PagescanCascade(_PagescanBase):
    """Production pagescan path: YOLO + HQ-SAM cascade primary, legacy fallback."""
    name = "pagescan-cascade"
    use_cascade = True
    notes = "YOLO11 + HQ-SAM cascade (production path); legacy SA24+LCNet fallback."


class PagescanLegacy(_PagescanBase):
    """Legacy-only pagescan: SA24 + LCNet ML chain. Cascade disabled."""
    name = "pagescan-legacy"
    use_cascade = False
    notes = "SA24 + LCNet ONNX heatmap regression (cascade disabled)."


def _unrotate_corners(corners: np.ndarray, k: int,
                      h_rot: int, w_rot: int,
                      h_orig: int, w_orig: int) -> np.ndarray:
    """Map (4, 2) corners from a k * 90° CCW-rotated frame back to original."""
    if k == 0:
        return corners
    out = np.zeros_like(corners)
    for i, (x, y) in enumerate(corners):
        if k == 1:    # original was rotated 90 CCW -> map back 90 CW
            out[i] = [y, w_rot - 1 - x]
        elif k == 2:
            out[i] = [w_rot - 1 - x, h_rot - 1 - y]
        elif k == 3:
            out[i] = [h_rot - 1 - y, x]
        else:
            out[i] = [x, y]
    return out


# ---------- OpenCV contour recipe ----------

class OpenCVRecipe(Scanner):
    """Classic gray + blur + Canny + contour + approxPolyDP scanner.

    The textbook OpenCV recipe (`pyimagesearch` style); a fair "no-ML"
    baseline. Implemented inline so we don't depend on a third-party
    package whose contour parameters drift over time.
    """
    name = "opencv-recipe"
    notes = "Gray + Gaussian + Canny + dilate + max-contour + approxPolyDP."

    def detect(self, image: np.ndarray) -> Optional[np.ndarray]:
        h_orig, w_orig = image.shape[:2]
        scale = h_orig / 500.0
        small = cv2.resize(image, (int(w_orig / scale), 500),
                           interpolation=cv2.INTER_AREA)

        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(gray, 75, 200)
        kernel = np.ones((5, 5), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)

        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

        for c in contours:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            if len(approx) == 4:
                return (approx.reshape(4, 2).astype(np.float32) * scale)
        return None


# ---------- docscan (optional pip dep) ----------

class DocscanLib(Scanner):
    """The `docscan` PyPI package (rembg-based). Outputs warped image only.

    Skipped silently if the package isn't installed. We mark `available =
    False` and the harness skips it without erroring.
    """
    name = "docscan"
    notes = "rembg-based contour scanner; warps directly (no quad output)."

    def __init__(self) -> None:
        try:
            from docscan.doc import scan as _docscan_scan  # noqa: F401
            self.available = True
        except ImportError:
            self.available = False
            self.notes += " — not installed; skipped."

    def detect(self, image: np.ndarray) -> Optional[np.ndarray]:
        # docscan doesn't expose corners; harness will rely on `warp` only.
        return None

    def warp(self, image: np.ndarray) -> Optional[np.ndarray]:
        if not self.available:
            return None
        try:
            from docscan.doc import scan as _docscan_scan
            ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 92])
            if not ok:
                return None
            result_bytes = _docscan_scan(buf.tobytes())
            if result_bytes is None:
                return None
            arr = np.frombuffer(result_bytes, np.uint8)
            return cv2.imdecode(arr, cv2.IMREAD_COLOR)
        except Exception as e:
            logger.debug(f"  docscan failed: {e}")
            return None


# ---------- registry ----------

def all_scanners() -> list[Scanner]:
    """Return all available scanners, in display order."""
    candidates: list[Scanner] = [
        PagescanCascade(),
        PagescanLegacy(),
        OpenCVRecipe(),
        DocscanLib(),
    ]
    return [s for s in candidates if s.available]
