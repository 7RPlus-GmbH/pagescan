"""Tests for perspective transform and canvas placement."""

import numpy as np
import pytest

from pagescan.config import ScanConfig
from pagescan.transform import perspective_transform, place_on_canvas


class TestPerspectiveTransform:
    def test_simple_rectangle(self):
        # 500x700 image with a document occupying most of it
        img = np.ones((700, 500, 3), dtype=np.uint8) * 128
        corners = np.array([
            [20, 20], [480, 20], [480, 680], [20, 680]
        ], dtype=np.float32)
        result = perspective_transform(img, corners)
        assert result.shape[0] > 200
        assert result.shape[1] > 200

    def test_preserves_aspect_ratio(self):
        img = np.ones((1000, 800, 3), dtype=np.uint8) * 128
        # ~A4 ratio document
        corners = np.array([
            [50, 50], [550, 50], [550, 750], [50, 750]
        ], dtype=np.float32)
        result = perspective_transform(img, corners)
        ratio = result.shape[0] / result.shape[1]
        expected_ratio = 700 / 500  # 1.4
        assert abs(ratio - expected_ratio) < 0.1


class TestPlaceOnCanvas:
    def test_default_a4(self):
        doc = np.ones((2000, 1600), dtype=np.uint8) * 128
        config = ScanConfig()
        result = place_on_canvas(doc, config)
        assert result.shape == (3508, 2480)

    def test_custom_dimensions(self):
        doc = np.ones((1000, 800, 3), dtype=np.uint8) * 128
        config = ScanConfig(output_width=2550, output_height=3300)
        result = place_on_canvas(doc, config)
        assert result.shape[:2] == (3300, 2550)

    def test_grayscale(self):
        doc = np.ones((1000, 800), dtype=np.uint8) * 128
        result = place_on_canvas(doc)
        assert result.ndim == 2

    def test_color(self):
        doc = np.ones((1000, 800, 3), dtype=np.uint8) * 128
        result = place_on_canvas(doc)
        assert result.ndim == 3

    def test_white_border(self):
        doc = np.ones((1000, 800), dtype=np.uint8) * 0  # black doc
        result = place_on_canvas(doc)
        # Corners should be white (margin area)
        assert result[0, 0] == 255
        assert result[-1, -1] == 255
