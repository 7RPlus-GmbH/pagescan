"""Configuration for the scanning pipeline.

All tunable parameters are exposed here so users can adapt pagescan
to different backgrounds, paper types, and output requirements.
"""

from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class ScanConfig:
    """Configuration for the document scanning pipeline.

    Attributes:
        background_hsv_low: Lower HSV bound for background detection (e.g. wood table).
            Default targets warm wood: H=0-45, S>=65 (paper is S=40-60), V>=30.
        background_hsv_high: Upper HSV bound for background detection.
        background_hsv_strict_s: Stricter saturation minimum for edge detection.
            Prevents cream-colored paper (S=65-85) from being matched as background.
        output_width: Output canvas width in pixels.
        output_height: Output canvas height in pixels.
        output_dpi: DPI for PDF output.
        output_margin: Margin in pixels when placing document on canvas.
        jpeg_quality: JPEG quality for PDF embedding (lower = smaller file).
        auto_orient: Run OCR-based auto-rotation to fix document orientation.
        deskew: Run Hough-based deskew to correct text tilt.
        enhance: Apply scan-like enhancement (grayscale, contrast, sharpen).
        shadow_removal: Apply illumination normalization before enhancement.
        white_balance: Adjust white balance so paper becomes pure white.
        use_ml: Try ML-based corner detection (DocAligner) before fallback.
        debug: Save intermediate images to debug_dir.
        debug_dir: Directory for debug output.
    """

    # Background detection (default: wood table)
    background_hsv_low: Tuple[int, int, int] = (0, 65, 30)
    background_hsv_high: Tuple[int, int, int] = (45, 255, 255)
    background_hsv_strict_s: int = 90

    # Output dimensions (default: A4 at 300 DPI)
    output_width: int = 2480
    output_height: int = 3508
    output_dpi: int = 300
    output_margin: int = 50
    jpeg_quality: int = 50

    # Pipeline toggles
    auto_orient: bool = True
    deskew: bool = True
    enhance: bool = True
    shadow_removal: bool = True
    white_balance: bool = True
    use_ml: bool = True

    # Debug
    debug: bool = False
    debug_dir: str = "pagescan_debug"


# Common presets

PRESET_A4_300 = ScanConfig()

PRESET_LETTER_300 = ScanConfig(
    output_width=2550,
    output_height=3300,
)

PRESET_FAST = ScanConfig(
    auto_orient=False,
    use_ml=False,
)

PRESET_RAW = ScanConfig(
    enhance=False,
    shadow_removal=False,
    white_balance=False,
)
