"""Integration tests for the scan pipeline."""

import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest

from pagescan import scan, ScanConfig


def _create_test_image(path: str, h=3000, w=2200):
    """Create a synthetic document photo: white paper on brown wood."""
    img = np.zeros((h, w, 3), dtype=np.uint8)

    # Brown wood background (HSV: H=20, S=120, V=140)
    img[:] = (60, 80, 140)  # BGR for warm brown

    # White paper region (slightly inset)
    margin = 200
    img[margin:h - margin, margin:w - margin] = (240, 240, 245)

    # Text lines
    for y in range(margin + 100, h - margin - 100, 40):
        x_end = margin + 100 + np.random.randint(800, 1400)
        img[y:y + 3, margin + 100:x_end] = (30, 30, 30)

    cv2.imwrite(path, img)
    return path


@pytest.fixture
def test_image(tmp_path):
    return _create_test_image(str(tmp_path / "test_doc.jpg"))


class TestScan:
    def test_basic_scan(self, test_image, tmp_path):
        output = str(tmp_path / "output.pdf")
        config = ScanConfig(use_ml=False, auto_orient=False)
        result = scan(test_image, output, config=config)
        assert result['success'] is True
        assert Path(output).exists()
        assert Path(output).stat().st_size > 0

    def test_nonexistent_input(self, tmp_path):
        result = scan("/nonexistent/image.jpg", str(tmp_path / "out.pdf"))
        assert result['success'] is False

    def test_auto_output_path(self, test_image):
        config = ScanConfig(use_ml=False, auto_orient=False)
        result = scan(test_image, config=config)
        assert result['success'] is True
        assert Path(result['output_path']).suffix == '.pdf'
        # Cleanup
        Path(result['output_path']).unlink(missing_ok=True)

    def test_raw_mode(self, test_image, tmp_path):
        output = str(tmp_path / "raw.pdf")
        config = ScanConfig(
            use_ml=False, auto_orient=False,
            enhance=False, shadow_removal=False, white_balance=False,
        )
        result = scan(test_image, output, config=config)
        assert result['success'] is True

    def test_debug_mode(self, test_image, tmp_path):
        output = str(tmp_path / "debug.pdf")
        debug_dir = str(tmp_path / "debug")
        config = ScanConfig(
            use_ml=False, auto_orient=False,
            debug=True, debug_dir=debug_dir,
        )
        result = scan(test_image, output, config=config)
        assert result['success'] is True
        assert Path(debug_dir).exists()

    def test_quality_score_range(self, test_image, tmp_path):
        output = str(tmp_path / "quality.pdf")
        config = ScanConfig(use_ml=False, auto_orient=False)
        result = scan(test_image, output, config=config)
        assert 0.0 <= result['quality_score'] <= 1.0
