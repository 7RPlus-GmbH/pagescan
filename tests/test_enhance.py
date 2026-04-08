"""Tests for image enhancement functions."""

import numpy as np
import pytest

from pagescan.enhance import remove_shadows, white_balance, enhance_document


def _make_document_image(h=400, w=300):
    """Create a synthetic document image (white paper with dark text)."""
    img = np.ones((h, w, 3), dtype=np.uint8) * 230  # off-white paper
    # Simulate text (dark rectangles)
    img[50:55, 30:200] = 30
    img[70:75, 30:180] = 30
    img[90:95, 30:220] = 30
    return img


def _make_shadowed_image(h=400, w=300):
    """Create a document with uneven lighting (shadow on left side)."""
    img = _make_document_image(h, w)
    # Darken left side to simulate shadow
    gradient = np.linspace(0.5, 1.0, w).reshape(1, -1, 1)
    img = (img.astype(np.float32) * gradient).astype(np.uint8)
    return img


class TestRemoveShadows:
    def test_preserves_shape(self):
        img = _make_document_image()
        result = remove_shadows(img)
        assert result.shape == img.shape

    def test_grayscale_input(self):
        gray = np.ones((200, 150), dtype=np.uint8) * 200
        result = remove_shadows(gray)
        assert result.shape == gray.shape
        assert result.ndim == 2

    def test_reduces_shadow_variance(self):
        img = _make_shadowed_image()
        result = remove_shadows(img)
        # After shadow removal, brightness should be more uniform
        orig_std = np.std(img[:, :, 0].astype(float))
        result_std = np.std(result[:, :, 0].astype(float))
        assert result_std < orig_std


class TestWhiteBalance:
    def test_preserves_shape(self):
        img = _make_document_image()
        result = white_balance(img)
        assert result.shape == img.shape

    def test_grayscale_passthrough(self):
        gray = np.ones((200, 150), dtype=np.uint8) * 200
        result = white_balance(gray)
        np.testing.assert_array_equal(result, gray)

    def test_brightens_paper(self):
        img = _make_document_image()
        result = white_balance(img)
        # Paper should be brighter after white balance
        center = result[150:250, 100:200]
        assert np.mean(center) >= np.mean(img[150:250, 100:200])


class TestEnhanceDocument:
    def test_output_is_grayscale(self):
        img = _make_document_image()
        result = enhance_document(img)
        assert result.ndim == 2

    def test_near_white_becomes_white(self):
        img = _make_document_image()
        result = enhance_document(img)
        # Background pixels should be pushed to 255
        bg_region = result[150:200, 100:200]
        assert np.mean(bg_region) > 250

    def test_text_stays_dark(self):
        img = _make_document_image()
        result = enhance_document(img)
        # Text line at row 50-55 should remain dark
        text_region = result[50:55, 30:200]
        assert np.mean(text_region) < 150
