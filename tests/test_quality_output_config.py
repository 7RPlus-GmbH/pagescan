"""Hermetic tests for pagescan.quality, pagescan.output, and pagescan.config.

Pure functions plus file I/O confined to ``tmp_path``. No network, no ML weights.
"""
from __future__ import annotations

import re
from pathlib import Path

import cv2
import numpy as np
import pytest

from pagescan.config import (
    PRESET_A4_300,
    PRESET_FAST,
    PRESET_LETTER_300,
    PRESET_RAW,
    ScanConfig,
)
from pagescan.output import save_image, save_pdf
from pagescan.quality import check_quality

# A BGR colour that maps into the default background HSV window
# (H<=45, S>=strict_s=90, V>=30): warm brown / wood.  Verified: HSV ~ (10,219,140).
BROWN_BGR = (20, 60, 140)
WHITE_BGR = (255, 255, 255)

# Image size chosen so the corner patch is exactly h//20 x w//20 (>= 20 each)
# and large enough that fill fractions are exact.
IMG_H = 400
IMG_W = 400
EDGE = 20  # max(20, 400//20) == 20


def _white_image(h: int = IMG_H, w: int = IMG_W) -> np.ndarray:
    img = np.empty((h, w, 3), dtype=np.uint8)
    img[:] = WHITE_BGR
    return img


def _fill_corner(img: np.ndarray, corner: str, frac: float) -> None:
    """Fill ``frac`` of an EDGE x EDGE corner patch with BROWN_BGR.

    The patch is EDGE rows x EDGE cols; we paint the first ``round(frac*EDGE)``
    *rows* of the patch so the matched-pixel ratio equals that fraction.
    """
    rows = round(frac * EDGE)
    if corner == "tl":
        img[0:rows, 0:EDGE] = BROWN_BGR
    elif corner == "tr":
        img[0:rows, IMG_W - EDGE:IMG_W] = BROWN_BGR
    elif corner == "bl":
        # Paint the top ``rows`` rows of the bottom-left patch.
        img[IMG_H - EDGE:IMG_H - EDGE + rows, 0:EDGE] = BROWN_BGR
    elif corner == "br":
        img[IMG_H - EDGE:IMG_H - EDGE + rows, IMG_W - EDGE:IMG_W] = BROWN_BGR
    else:  # pragma: no cover - guard
        raise ValueError(corner)


# --------------------------------------------------------------------------- #
# quality.py
# --------------------------------------------------------------------------- #

def test_brown_maps_into_background_window():
    """Sanity check that our test colour is detected and white is not."""
    cfg = ScanConfig()
    px = np.array([[list(BROWN_BGR)]], dtype=np.uint8)
    hsv = cv2.cvtColor(px, cv2.COLOR_BGR2HSV)[0, 0]
    assert hsv[0] <= cfg.background_hsv_high[0]
    assert hsv[1] >= cfg.background_hsv_strict_s
    assert hsv[2] >= cfg.background_hsv_low[2]


def test_check_quality_clean_passes():
    """A clean white scan -> passed, score ~1.0, message OK.

    Passing config=None also exercises the ScanConfig() default (line 24).
    """
    img = _white_image()
    passed, score, message = check_quality(img, config=None)
    assert passed is True
    assert message == "OK"
    assert score == pytest.approx(1.0, abs=1e-6)


def test_check_quality_significant_background_fails():
    """A corner fully covered by background -> max>0.50 branch (line 54)."""
    img = _white_image()
    # Fill all four corners completely: max == avg == 1.0.
    for c in ("tl", "tr", "bl", "br"):
        _fill_corner(img, c, 1.0)

    passed, score, message = check_quality(img)
    assert passed is False
    assert "Significant background" in message
    # avg_wood == 1.0 here, so score == 0.0.
    assert score == pytest.approx(0.0, abs=1e-6)


def test_check_quality_significant_single_corner():
    """One fully-covered corner still trips max>0.50 even with low average."""
    img = _white_image()
    _fill_corner(img, "tl", 1.0)  # ratio 1.0 in one corner, 0 in others

    passed, score, message = check_quality(img)
    assert passed is False
    assert "Significant background" in message
    # avg_wood == 1.0/4 == 0.25 -> score == 0.75.
    assert score == pytest.approx(0.75, abs=1e-6)


def test_check_quality_moderate_contamination_fails():
    """Moderate, even contamination -> avg_wood>0.30 branch (line 56).

    Fill 0.40 of every corner: max == 0.40 (NOT > 0.50) so the first branch
    is skipped, avg == 0.40 (> 0.30) so the contamination branch fires.
    """
    frac = 0.40
    img = _white_image()
    for c in ("tl", "tr", "bl", "br"):
        _fill_corner(img, c, frac)

    passed, score, message = check_quality(img)
    assert passed is False
    assert "contamination" in message
    # score == 1.0 - avg_wood, and avg_wood == frac here.
    assert score == pytest.approx(1.0 - frac, abs=1e-6)
    # Confirm it was the avg branch, not the max branch.
    assert "Significant background" not in message


# --------------------------------------------------------------------------- #
# output.py
# --------------------------------------------------------------------------- #

_MEDIABOX_RE = re.compile(rb"/MediaBox\s*\[\s*([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s*\]")
PT_TO_MM = 25.4 / 72.0


def _mediabox_mm(pdf_bytes: bytes) -> tuple[float, float]:
    m = _MEDIABOX_RE.search(pdf_bytes)
    assert m is not None, "no /MediaBox found in PDF"
    x0, y0, x1, y1 = (float(g) for g in m.groups())
    width_mm = (x1 - x0) * PT_TO_MM
    height_mm = (y1 - y0) * PT_TO_MM
    return width_mm, height_mm


def _color_canvas(cfg: ScanConfig) -> np.ndarray:
    """A colour canvas at the configured pixel dimensions.

    The PDF page size is derived from pixel_count / output_dpi, so the canvas
    must use the real configured dimensions for MediaBox assertions to hold.
    """
    img = np.full((cfg.output_height, cfg.output_width, 3), 255, dtype=np.uint8)
    img[100:200, 100:200] = (0, 0, 200)  # a bit of colour content
    return img


def test_save_pdf_a4_page_size(tmp_path: Path):
    cfg = PRESET_A4_300
    out = tmp_path / "a4.pdf"
    save_pdf(_color_canvas(cfg), str(out), cfg)

    assert out.exists()
    data = out.read_bytes()
    assert data.startswith(b"%PDF")

    w_mm, h_mm = _mediabox_mm(data)
    assert w_mm == pytest.approx(210.0, abs=1.0)
    assert h_mm == pytest.approx(297.0, abs=1.0)


def test_save_pdf_letter_page_size(tmp_path: Path):
    cfg = PRESET_LETTER_300
    out = tmp_path / "letter.pdf"
    save_pdf(_color_canvas(cfg), str(out), cfg)

    assert out.exists()
    data = out.read_bytes()
    assert data.startswith(b"%PDF")

    w_mm, h_mm = _mediabox_mm(data)
    assert w_mm == pytest.approx(215.9, abs=1.0)
    assert h_mm == pytest.approx(279.4, abs=1.0)


def test_save_pdf_grayscale(tmp_path: Path):
    """A 2-D (grayscale) image exercises the mode='L' branch (line 34)."""
    gray = np.full((120, 100), 255, dtype=np.uint8)
    gray[10:20, 10:20] = 0
    out = tmp_path / "gray.pdf"
    save_pdf(gray, str(out))  # config=None -> ScanConfig() default

    assert out.exists()
    data = out.read_bytes()
    assert data.startswith(b"%PDF")
    # Still a valid, parseable single-page PDF.
    assert _MEDIABOX_RE.search(data) is not None


def test_save_pdf_creates_nested_dirs(tmp_path: Path):
    cfg = ScanConfig()
    out = tmp_path / "deep" / "nested" / "dir" / "doc.pdf"
    assert not out.parent.exists()
    save_pdf(_color_canvas(cfg), str(out), cfg)
    assert out.exists()
    assert out.read_bytes().startswith(b"%PDF")


def test_save_image_png_roundtrip(tmp_path: Path):
    img = np.zeros((30, 40, 3), dtype=np.uint8)
    img[:, :, 2] = 255  # red in BGR
    out = tmp_path / "out.png"
    save_image(img, str(out))

    assert out.exists()
    loaded = cv2.imread(str(out), cv2.IMREAD_COLOR)
    assert loaded is not None
    assert loaded.shape == (30, 40, 3)


def test_save_image_jpg_roundtrip(tmp_path: Path):
    img = np.full((24, 36, 3), 128, dtype=np.uint8)
    out = tmp_path / "sub" / "out.jpg"
    save_image(img, str(out))  # also covers parent-dir creation in save_image

    assert out.exists()
    loaded = cv2.imread(str(out), cv2.IMREAD_COLOR)
    assert loaded is not None
    assert loaded.shape == (24, 36, 3)


# --------------------------------------------------------------------------- #
# config.py
# --------------------------------------------------------------------------- #

def test_default_field_values():
    cfg = ScanConfig()
    assert cfg.background_hsv_low == (0, 65, 30)
    assert cfg.background_hsv_high == (45, 255, 255)
    assert cfg.background_hsv_strict_s == 90
    assert cfg.output_width == 2480
    assert cfg.output_height == 3508
    assert cfg.output_dpi == 300
    assert cfg.output_margin == 50
    assert cfg.jpeg_quality == 50
    assert cfg.auto_orient is True
    assert cfg.ocr_lang == "eng+deu+fra+spa+ita+jpn"
    assert cfg.deskew is True
    assert cfg.enhance is True
    assert cfg.shadow_removal is True
    assert cfg.white_balance is True
    assert cfg.use_ml is True
    assert cfg.use_cascade is False  # legacy is the default path on v1 weights
    assert cfg.detector_conf_threshold == 0.25
    assert cfg.min_doc_coverage == 0.05
    assert cfg.debug is False
    assert cfg.debug_dir == "pagescan_debug"


def test_preset_a4_equals_default():
    assert ScanConfig() == PRESET_A4_300


def test_preset_letter_differs_from_a4():
    assert PRESET_LETTER_300.output_width == 2550
    assert PRESET_LETTER_300.output_height == 3300
    assert PRESET_LETTER_300 != PRESET_A4_300
    # Only the dimensions differ from the A4 default.
    assert ScanConfig(output_width=2550, output_height=3300) == PRESET_LETTER_300


def test_preset_fast():
    assert PRESET_FAST.use_ml is False
    assert PRESET_FAST.auto_orient is False
    # Everything else stays at the default.
    assert ScanConfig(use_ml=False, auto_orient=False) == PRESET_FAST


def test_preset_raw():
    assert PRESET_RAW.enhance is False
    assert PRESET_RAW.shadow_removal is False
    assert PRESET_RAW.white_balance is False
    assert ScanConfig(
        enhance=False, shadow_removal=False, white_balance=False
    ) == PRESET_RAW
