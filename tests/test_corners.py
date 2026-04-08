"""Tests for corner detection utilities."""

import numpy as np
import pytest

from pagescan.corners import order_corners, _check_parallel, _check_quad_dimensions


class TestOrderCorners:
    def test_already_ordered(self):
        pts = np.array([[0, 0], [100, 0], [100, 150], [0, 150]], dtype=np.float32)
        result = order_corners(pts)
        np.testing.assert_array_almost_equal(result[0], [0, 0])    # TL
        np.testing.assert_array_almost_equal(result[1], [100, 0])  # TR
        np.testing.assert_array_almost_equal(result[2], [100, 150])  # BR
        np.testing.assert_array_almost_equal(result[3], [0, 150])  # BL

    def test_shuffled(self):
        pts = np.array([[100, 150], [0, 0], [0, 150], [100, 0]], dtype=np.float32)
        result = order_corners(pts)
        np.testing.assert_array_almost_equal(result[0], [0, 0])
        np.testing.assert_array_almost_equal(result[1], [100, 0])
        np.testing.assert_array_almost_equal(result[2], [100, 150])
        np.testing.assert_array_almost_equal(result[3], [0, 150])

    def test_with_perspective(self):
        # Slightly tilted document
        pts = np.array([[10, 5], [95, 2], [98, 148], [8, 152]], dtype=np.float32)
        result = order_corners(pts)
        # TL should be the point with smallest sum
        assert result[0][0] < result[1][0]  # TL.x < TR.x
        assert result[0][1] < result[3][1]  # TL.y < BL.y


class TestCheckParallel:
    def test_perfect_rectangle(self):
        pts = np.array([[0, 0], [100, 0], [100, 150], [0, 150]], dtype=np.float32)
        tb, lr = _check_parallel(pts)
        assert tb < 1.0
        assert lr < 1.0

    def test_slight_tilt(self):
        pts = np.array([[0, 0], [100, 5], [100, 155], [0, 150]], dtype=np.float32)
        tb, lr = _check_parallel(pts)
        assert tb < 5.0  # Parallel-ish


class TestCheckQuadDimensions:
    def test_full_page(self):
        pts = np.array([[50, 50], [950, 50], [950, 1400], [50, 1400]], dtype=np.float32)
        assert _check_quad_dimensions(pts, 1500, 1000) is True

    def test_too_narrow(self):
        # Quad only covers a thin strip
        pts = np.array([[400, 50], [600, 50], [600, 1400], [400, 1400]], dtype=np.float32)
        assert _check_quad_dimensions(pts, 1500, 1000) is False

    def test_near_square_rejected(self):
        pts = np.array([[50, 50], [550, 50], [550, 550], [50, 550]], dtype=np.float32)
        assert _check_quad_dimensions(pts, 1000, 1000) is False
