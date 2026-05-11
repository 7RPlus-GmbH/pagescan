"""Image enhancement: shadow removal, white balance, scan-like output."""

import logging

import cv2
import numpy as np

from pagescan.config import ScanConfig

logger = logging.getLogger(__name__)


def remove_shadows(image: np.ndarray) -> np.ndarray:
    """Remove uneven lighting via divide-by-background illumination.

    Downscales to ~1000px for fast background estimation (illumination
    varies slowly). Per-channel morphological closing estimates the
    background, then divides the original by it. Multiplicative
    normalization — never paints white onto content.
    """
    h, w = image.shape[:2]

    max_bg_dim = 1000
    scale = min(1.0, max_bg_dim / max(h, w))
    small = cv2.resize(image, None, fx=scale, fy=scale) if scale < 1.0 else image

    sh, sw = small.shape[:2]
    kernel_size = max(31, min(sh, sw) // 8) | 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

    def _normalize_channel(channel, small_channel):
        bg_small = cv2.morphologyEx(small_channel, cv2.MORPH_CLOSE, kernel)
        bg_small = cv2.GaussianBlur(bg_small, (kernel_size, kernel_size), 0)
        bg = cv2.resize(bg_small, (w, h), interpolation=cv2.INTER_LINEAR) if scale < 1.0 else bg_small
        bg = bg.astype(np.float32)
        bg[bg < 1] = 1
        return np.clip(channel.astype(np.float32) / bg * 255, 0, 255).astype(np.uint8)

    if image.ndim == 2:
        return _normalize_channel(image, small)

    channels = cv2.split(image)
    small_channels = cv2.split(small)
    result_channels = [_normalize_channel(ch, sch)
                       for ch, sch in zip(channels, small_channels)]
    return cv2.merge(result_channels)


def white_balance(image: np.ndarray) -> np.ndarray:
    """Adjust white balance so paper background becomes pure white.

    Samples the central 50% region, finds paper pixels (low saturation,
    bright), and computes per-channel gain to map paper color to white.
    """
    if image.ndim == 2:
        return image

    h, w = image.shape[:2]
    y1, y2 = h // 4, 3 * h // 4
    x1, x2 = w // 4, 3 * w // 4
    center = image[y1:y2, x1:x2]

    hsv_center = cv2.cvtColor(center, cv2.COLOR_BGR2HSV)
    paper_mask = cv2.inRange(hsv_center, (0, 0, 160), (180, 60, 255))

    paper_pixels = center[paper_mask > 0]
    if len(paper_pixels) < 100:
        return image

    paper_median = np.median(paper_pixels, axis=0)
    gains = np.clip(255.0 / (paper_median + 1e-6), 1.0, 1.5)

    result = image.astype(np.float32)
    for i in range(3):
        result[:, :, i] *= gains[i]

    return np.clip(result, 0, 255).astype(np.uint8)


def enhance_document(image: np.ndarray) -> np.ndarray:
    """Scan-like enhancement: grayscale + contrast stretch + gamma + sharpen + whiten.

    Converts to grayscale, applies percentile-based contrast stretch,
    brightening gamma, unsharp mask for text sharpness, and pushes
    near-white pixels to pure white for a clean scan appearance.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image

    # Percentile-based contrast stretch
    p2, p98 = np.percentile(gray, [2, 98])
    if p98 > p2:
        gray = np.clip((gray.astype(np.float32) - p2) / (p98 - p2) * 255,
                        0, 255).astype(np.uint8)

    # Brightening gamma (< 1.0 pushes paper toward white)
    gray = (255 * (gray / 255.0) ** 0.8).astype(np.uint8)

    # Unsharp mask for text edges
    blurred = cv2.GaussianBlur(gray, (0, 0), sigmaX=2)
    gray = cv2.addWeighted(gray, 1.5, blurred, -0.5, 0)

    # Push near-white to pure white
    gray[gray >= 230] = 255

    return gray
