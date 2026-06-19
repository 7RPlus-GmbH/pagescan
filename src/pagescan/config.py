"""Configuration for the scanning pipeline.

All tunable parameters are exposed here so users can adapt pagescan
to different backgrounds, paper types, and output requirements.
"""
from __future__ import annotations

from dataclasses import dataclass


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
        ocr_lang: Tesseract language(s) for the OCR orientation cross-check.
            Accepts any Tesseract language code, or several joined with ``+``
            (e.g. ``"eng+deu+fra+spa+ita+jpn"``, ``"eng"``, ``"fra+jpn"``).
            Used only when ``auto_orient`` is True and Tesseract is installed,
            and only as a tiebreaker on ambiguous orientations — not on every
            scan. Requested packs that aren't installed are dropped
            automatically (it uses whatever is available), so a multilingual
            default is safe; a wholly-uninstalled set degrades to the
            CNN-only orientation heuristic. Default
            ``"eng+deu+fra+spa+ita+jpn"`` targets common business-document
            languages — set it to match your own corpus to keep the
            cross-check fast and focused.
        deskew: Run Hough-based deskew to correct text tilt.
        enhance: Apply scan-like enhancement (grayscale, contrast, sharpen).
        shadow_removal: Apply illumination normalization before enhancement.
        white_balance: Adjust white balance so paper becomes pure white.
        use_ml: Master switch for ML-based corner detection. When False,
            skips both cascade and legacy ML and goes straight to the
            conservative-crop fallback. Useful for fast/headless modes.
        use_cascade: Use the YOLO11 + HQ-SAM cascade as the primary
            detection path instead of the legacy SA24+LCNet ML chain.
            Defaults to False: on the current (v1) weights the legacy chain
            beats the cascade on the April benchmark (44/50 vs 35/50 at
            IoU>=0.90), is ~75x faster, and fails less catastrophically. The
            cascade has a higher ceiling and becomes the default again once
            the v2 YOLO closes the training-distribution gap. Set True to opt
            in now; it requires the optional `[ml]` extras (torch + HQ-SAM).
        detector_conf_threshold: Minimum YOLO confidence for accepting a
            detection. Below this, cascade falls through to legacy.
        min_doc_coverage: Over-crop guard. Predicted quads whose area is
            below this fraction of the image are rejected as too small
            (typical failure: SAM segments an inner text block instead
            of the full page). Default 0.05; raise toward 0.10 if your
            documents always fill ≥10% of the frame.
        debug: Save intermediate images to debug_dir.
        debug_dir: Directory for debug output.
    """

    # Background detection (default: wood table)
    background_hsv_low: tuple[int, int, int] = (0, 65, 30)
    background_hsv_high: tuple[int, int, int] = (45, 255, 255)
    background_hsv_strict_s: int = 90

    # Output dimensions (default: A4 at 300 DPI)
    output_width: int = 2480
    output_height: int = 3508
    output_dpi: int = 300
    output_margin: int = 50
    jpeg_quality: int = 50

    # Pipeline toggles
    auto_orient: bool = True
    ocr_lang: str = "eng+deu+fra+spa+ita+jpn"
    deskew: bool = True
    enhance: bool = True
    shadow_removal: bool = True
    white_balance: bool = True
    use_ml: bool = True

    # ML detection path
    use_cascade: bool = False  # legacy beats cascade on v1 weights; see docstring
    detector_conf_threshold: float = 0.25
    min_doc_coverage: float = 0.05

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
