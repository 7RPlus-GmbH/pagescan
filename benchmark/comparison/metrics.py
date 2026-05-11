"""Per-image metrics for the document-scanner comparison harness.

Exports the only six functions `run_real.py` ever needs from this module:

    polygon_iou        - polygon IoU between two 4-corner quads
    align_quad         - best cyclic + reflected match of pred to GT label order
    perspective_warp   - quad -> deskewed-and-cropped document
    psnr               - peak signal-to-noise ratio (uint8 RGB images)
    wer                - word error rate via Levenshtein on tokenised text
    ocr                - tesseract OCR wrapper, returns "" if Tesseract missing

All functions are pure / side-effect-free except `ocr`, which shells out
to Tesseract once per call and returns the empty string on any failure.
"""
from __future__ import annotations

import logging
from typing import Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# ---------- geometry ----------

def polygon_iou(pts1: np.ndarray, pts2: np.ndarray) -> float:
    """IoU between two convex 4-corner quads via cv2.intersectConvexConvex."""
    pts1 = pts1.astype(np.float32).reshape(-1, 2)
    pts2 = pts2.astype(np.float32).reshape(-1, 2)
    h1 = cv2.convexHull(pts1)
    h2 = cv2.convexHull(pts2)
    a1 = cv2.contourArea(h1)
    a2 = cv2.contourArea(h2)
    if a1 < 1 or a2 < 1:
        return 0.0
    ret, inter = cv2.intersectConvexConvex(h1, h2)
    if ret <= 0 or inter is None or len(inter) < 3:
        return 0.0
    ai = cv2.contourArea(inter)
    union = a1 + a2 - ai
    return float(ai / union) if union >= 1 else 0.0


def align_quad(pred: np.ndarray, gt: np.ndarray
               ) -> Tuple[int, np.ndarray, np.ndarray]:
    """Find the cyclic rotation + optional reflection of `pred` that minimises
    sum-of-corner-distances against `gt` (taken in its original label order).

    Returns:
        rotation_offset: 0..3 forward, 4..7 reflected.
        pred_aligned:    pred rolled to match gt's per-corner index.
        dists:           per-corner L2 distance after alignment.
    """
    gt = gt.astype(np.float32).reshape(4, 2)
    pred = pred.astype(np.float32).reshape(4, 2)
    best = (0, pred, np.linalg.norm(pred - gt, axis=1))
    best_total = best[2].sum()
    for reflect in (False, True):
        base = pred[::-1] if reflect else pred
        for k in range(4):
            cand = np.roll(base, -k, axis=0)
            dists = np.linalg.norm(cand - gt, axis=1)
            total = dists.sum()
            if total < best_total:
                best_total = total
                best = (k + (4 if reflect else 0), cand, dists)
    return best


def perspective_warp(image: np.ndarray, quad: np.ndarray) -> np.ndarray:
    """Warp the quad-bounded region of `image` into an axis-aligned rectangle.

    Output size is derived from the quad's edge lengths so the document is
    rendered at roughly its native pixel density.
    """
    quad = quad.astype(np.float32).reshape(4, 2)
    # Order corners as TL, TR, BR, BL spatially
    s = quad.sum(axis=1)
    d = np.diff(quad, axis=1).ravel()
    ordered = np.array([
        quad[np.argmin(s)],
        quad[np.argmin(d)],
        quad[np.argmax(s)],
        quad[np.argmax(d)],
    ], dtype=np.float32)
    tl, tr, br, bl = ordered
    width = int(round(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl))))
    height = int(round(max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr))))
    width = max(width, 1)
    height = max(height, 1)
    dst = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
                   dtype=np.float32)
    M = cv2.getPerspectiveTransform(ordered, dst)
    return cv2.warpPerspective(image, M, (width, height))


# ---------- image quality ----------

def psnr(a: np.ndarray, b: np.ndarray, target_size: Optional[Tuple[int, int]] = None) -> float:
    """Peak SNR between two uint8 BGR images. Resizes b to a's shape if needed.

    `target_size = (width, height)` resamples both before comparing — useful
    when comparing scans of different output resolutions.
    """
    if target_size is not None:
        a = cv2.resize(a, target_size, interpolation=cv2.INTER_AREA)
        b = cv2.resize(b, target_size, interpolation=cv2.INTER_AREA)
    elif a.shape != b.shape:
        b = cv2.resize(b, (a.shape[1], a.shape[0]), interpolation=cv2.INTER_AREA)
    mse = np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2)
    if mse < 1e-9:
        return float("inf")
    return float(20 * np.log10(255.0 / np.sqrt(mse)))


# ---------- text ----------

def _levenshtein(a: list, b: list) -> int:
    """Token-level edit distance. O(n*m) memory; fine for OCR-sized inputs."""
    n, m = len(a), len(b)
    if n == 0:
        return m
    if m == 0:
        return n
    prev = list(range(m + 1))
    cur = [0] * (m + 1)
    for i in range(1, n + 1):
        cur[0] = i
        for j in range(1, m + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            cur[j] = min(prev[j] + 1,        # deletion
                         cur[j - 1] + 1,     # insertion
                         prev[j - 1] + cost) # substitution
        prev, cur = cur, prev
    return prev[m]


def wer(reference: str, hypothesis: str) -> float:
    """Word error rate between two text strings.

    Returns 0.0 for identical text; 1.0 when reference is empty and hypothesis
    is not (or vice versa); >1.0 is possible when the hypothesis has many
    insertions relative to a short reference.
    """
    ref = reference.split()
    hyp = hypothesis.split()
    if not ref:
        return 0.0 if not hyp else float("inf")
    return _levenshtein(ref, hyp) / len(ref)


def ocr(image: np.ndarray, lang: str = "eng+deu") -> str:
    """Run Tesseract on a BGR image and return the text. Returns "" on any error.

    `pytesseract` is optional — without it, this returns "" silently and the
    OCR-WER metric is reported as N/A in the harness aggregate.
    """
    try:
        import pytesseract
    except ImportError:
        return ""
    try:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return pytesseract.image_to_string(rgb, lang=lang)
    except Exception as e:
        logger.debug(f"  ocr failed: {e}")
        return ""
