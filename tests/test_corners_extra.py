"""Extra hermetic tests for pagescan.corners (no network, ML mocked)."""

import numpy as np
import pytest

from pagescan import corners
from pagescan.config import ScanConfig

# ---------- geometry helpers ----------

class TestEdgeAngle:
    def test_horizontal(self):
        assert corners._edge_angle(np.array([10.0, 0.0])) == pytest.approx(0.0)

    def test_vertical(self):
        assert corners._edge_angle(np.array([0.0, 10.0])) == pytest.approx(90.0)

    def test_diagonal(self):
        assert corners._edge_angle(np.array([10.0, 10.0])) == pytest.approx(45.0)


class TestCheckParallelExtra:
    def test_rectangle_zero(self):
        pts = np.array([[0, 0], [100, 0], [100, 150], [0, 150]], dtype=np.float32)
        tb, lr = corners._check_parallel(pts)
        assert tb == pytest.approx(0.0, abs=0.5)
        assert lr == pytest.approx(0.0, abs=0.5)


class TestCheckQuadDimensionsExtra:
    def test_short_side_just_below_threshold(self):
        # short side ~10% of img short side (1000) -> rejected
        pts = np.array([[0, 0], [1500, 0], [1500, 100], [0, 100]], dtype=np.float32)
        assert corners._check_quad_dimensions(pts, 1000, 1500) is False

    def test_short_side_above_threshold(self):
        pts = np.array([[0, 0], [1500, 0], [1500, 300], [0, 300]], dtype=np.float32)
        assert corners._check_quad_dimensions(pts, 1000, 1500) is True


# ---------- _repair_corners ----------

class TestRepairCorners:
    def test_lr_repair_branch_executes(self):
        # Force the else-branch (left/right repair) by passing lr_diff > tb_diff.
        # Right edge is short relative to left -> bottom_len < top_len * 0.85 is
        # false, so the final sub-branch (rebuild TR from top_len) runs.
        h, w = 1000, 800
        ordered = np.array([
            [50, 50],     # TL
            [700, 90],    # TR (slightly off)
            [750, 920],   # BR
            [50, 900],    # BL
        ], dtype=np.float32)
        # else-branch when tb_diff <= lr_diff; call directly with forced values.
        repaired = corners._repair_corners(ordered, tb_diff=1.0, lr_diff=20.0, h=h, w=w)
        # The repair branch ran; result is either a repaired quad or None.
        assert repaired is None or repaired.shape == (4, 2)

    def test_repairable_top_bottom_divergence(self):
        h, w = 1000, 800
        # Good top edge, bottom-right corner dropped so bottom diverges from top.
        ordered = np.array([
            [50, 50],     # TL
            [750, 50],    # TR
            [760, 700],   # BR (skewed outward / down)
            [50, 900],    # BL
        ], dtype=np.float32)
        tb, lr = corners._check_parallel(ordered)
        assert tb > 10  # genuinely divergent before repair
        repaired = corners._repair_corners(ordered, tb, lr, h, w)
        assert repaired is not None
        tb2, lr2 = corners._check_parallel(repaired)
        assert tb2 <= 10
        assert lr2 <= 10


# ---------- _validate_and_repair ----------

class TestValidateAndRepair:
    def _clean_quad(self, h, w):
        return np.array([
            [int(0.1 * w), int(0.1 * h)],
            [int(0.9 * w), int(0.1 * h)],
            [int(0.9 * w), int(0.9 * h)],
            [int(0.1 * w), int(0.9 * h)],
        ], dtype=np.float32)

    def test_clean_quad_returns_ordered(self):
        h, w = 1000, 800
        quad = self._clean_quad(h, w)
        result = corners._validate_and_repair(quad, h, w, min_coverage=0.05)
        assert result is not None
        # ordered TL, TR, BR, BL
        np.testing.assert_array_almost_equal(result[0], [0.1 * w, 0.1 * h], decimal=0)
        np.testing.assert_array_almost_equal(result[1], [0.9 * w, 0.1 * h], decimal=0)
        np.testing.assert_array_almost_equal(result[2], [0.9 * w, 0.9 * h], decimal=0)
        np.testing.assert_array_almost_equal(result[3], [0.1 * w, 0.9 * h], decimal=0)

    def test_below_min_coverage_rejected(self):
        h, w = 1000, 800
        # tiny quad ~ a few percent of area but still > 15% short-side; force tiny area
        quad = np.array([
            [10, 10], [60, 10], [60, 200], [10, 200]
        ], dtype=np.float32)
        # coverage = (50*190)/(800*1000) ~= 0.0119 < 0.05
        result = corners._validate_and_repair(quad, h, w, min_coverage=0.05)
        assert result is None

    def test_coverage_over_99_rejected(self):
        h, w = 1000, 800
        quad = np.array([
            [0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]
        ], dtype=np.float32)
        result = corners._validate_and_repair(quad, h, w, min_coverage=0.05)
        assert result is None

    def test_too_narrow_rejected(self):
        h, w = 1000, 800
        # short side < 15% of img short side (800) -> < 120px wide
        quad = np.array([
            [300, 50], [380, 50], [380, 950], [300, 950]
        ], dtype=np.float32)
        result = corners._validate_and_repair(quad, h, w, min_coverage=0.01)
        assert result is None

    def test_moderately_skewed_accepted(self):
        h, w = 1000, 800
        # Verified quad: tb~17.7, lr~17.0 (both in the (15, 25] moderate band)
        # and _repair_corners returns None -> the moderate-accept branch runs.
        quad = np.array([
            [158, 150],
            [633, 221],
            [727, 767],
            [66, 874],
        ], dtype=np.float32)
        ordered = corners.order_corners(quad)
        tb, lr = corners._check_parallel(ordered)
        assert 15 < tb <= 25 and 15 < lr <= 25
        assert corners._repair_corners(ordered, tb, lr, h, w) is None
        result = corners._validate_and_repair(quad, h, w, min_coverage=0.01)
        # accepted as moderately skewed -> returns the ordered quad unchanged
        assert result is not None
        np.testing.assert_array_almost_equal(result, ordered)

    def test_badly_skewed_rejected(self):
        h, w = 1000, 800
        # Wildly non-parallel quad; tb and lr both >> 25 and not repairable.
        quad = np.array([
            [100, 100],
            [700, 400],
            [600, 850],
            [150, 200],
        ], dtype=np.float32)
        ordered = corners.order_corners(quad)
        tb, lr = corners._check_parallel(ordered)
        assert tb > 25 or lr > 25
        result = corners._validate_and_repair(quad, h, w, min_coverage=0.01)
        assert result is None


# ---------- backends ----------

def _valid_quad(h=1000, w=800):
    return np.array([
        [int(0.1 * w), int(0.1 * h)],
        [int(0.9 * w), int(0.1 * h)],
        [int(0.9 * w), int(0.9 * h)],
        [int(0.1 * w), int(0.9 * h)],
    ], dtype=np.float32)


class TestDetectViaCascade:
    def test_not_available_returns_none(self, monkeypatch):
        monkeypatch.setattr(corners, "HAS_CASCADE", False)
        img = np.zeros((1000, 800, 3), dtype=np.uint8)
        assert corners._detect_via_cascade(img, ScanConfig()) is None


class TestDetectViaLegacy:
    def test_not_available_returns_none(self, monkeypatch):
        monkeypatch.setattr(corners, "HAS_LEGACY_ML", False)
        img = np.zeros((1000, 800, 3), dtype=np.uint8)
        assert corners._detect_via_legacy(img) is None


# ---------- detect_corners_ml ----------

class TestDetectCornersML:
    def _img(self):
        return np.zeros((1000, 800, 3), dtype=np.uint8)

    def test_cascade_success(self, monkeypatch):
        monkeypatch.setattr(corners, "HAS_CASCADE", True)
        monkeypatch.setattr(corners, "HAS_LEGACY_ML", True)
        monkeypatch.setattr(corners, "_detect_via_cascade",
                            lambda img, cfg: _valid_quad())
        monkeypatch.setattr(corners, "_detect_via_legacy",
                            lambda img: (_ for _ in ()).throw(AssertionError("legacy not expected")))
        cfg = ScanConfig(use_cascade=True)
        result, method = corners.detect_corners_ml(self._img(), cfg)
        assert method == "cascade"
        assert result is not None

    def test_cascade_none_legacy_success(self, monkeypatch):
        monkeypatch.setattr(corners, "HAS_CASCADE", True)
        monkeypatch.setattr(corners, "HAS_LEGACY_ML", True)
        monkeypatch.setattr(corners, "_detect_via_cascade", lambda img, cfg: None)
        monkeypatch.setattr(corners, "_detect_via_legacy", lambda img: _valid_quad())
        cfg = ScanConfig(use_cascade=True)
        result, method = corners.detect_corners_ml(self._img(), cfg)
        assert method == "legacy"
        assert result is not None

    def test_both_none(self, monkeypatch):
        monkeypatch.setattr(corners, "_detect_via_cascade", lambda img, cfg: None)
        monkeypatch.setattr(corners, "_detect_via_legacy", lambda img: None)
        cfg = ScanConfig(use_cascade=True)
        result, method = corners.detect_corners_ml(self._img(), cfg)
        assert result is None
        assert method is None

    def test_cascade_disabled_uses_legacy(self, monkeypatch):
        # use_cascade False -> cascade backend not called
        monkeypatch.setattr(corners, "_detect_via_cascade",
                            lambda img, cfg: (_ for _ in ()).throw(AssertionError("cascade not expected")))
        monkeypatch.setattr(corners, "_detect_via_legacy", lambda img: _valid_quad())
        cfg = ScanConfig(use_cascade=False)
        _result, method = corners.detect_corners_ml(self._img(), cfg)
        assert method == "legacy"


# ---------- detect_corners (rotation retry) ----------

class TestDetectCorners:
    def test_use_ml_false(self):
        img = np.zeros((1000, 800, 3), dtype=np.uint8)
        cfg = ScanConfig(use_ml=False)
        assert corners.detect_corners(img, cfg) == (None, 0, None)

    def test_success_no_rotation(self, monkeypatch):
        img = np.zeros((1000, 800, 3), dtype=np.uint8)
        monkeypatch.setattr(corners, "detect_corners_ml",
                            lambda image, cfg: (_valid_quad(), "cascade"))
        c, k, method = corners.detect_corners(img, ScanConfig())
        assert k == 0
        assert method == "cascade"
        assert c is not None

    def test_success_after_rotation(self, monkeypatch):
        img = np.zeros((1000, 800, 3), dtype=np.uint8)
        calls = {"n": 0}

        def fake_ml(image, cfg):
            # Fail on original orientation, succeed once rotated.
            calls["n"] += 1
            if calls["n"] == 1:
                return None, None
            return _valid_quad(image.shape[0], image.shape[1]), "legacy"

        monkeypatch.setattr(corners, "detect_corners_ml", fake_ml)
        c, k, method = corners.detect_corners(img, ScanConfig())
        assert k in (1, 3)
        assert method == "legacy"
        assert c is not None

    def test_total_failure(self, monkeypatch):
        img = np.zeros((1000, 800, 3), dtype=np.uint8)
        monkeypatch.setattr(corners, "detect_corners_ml",
                            lambda image, cfg: (None, None))
        assert corners.detect_corners(img, ScanConfig()) == (None, 0, None)

    def test_config_none_defaults(self, monkeypatch):
        # config=None -> builds a default ScanConfig (use_ml True), then fails out.
        img = np.zeros((1000, 800, 3), dtype=np.uint8)
        monkeypatch.setattr(corners, "detect_corners_ml",
                            lambda image, cfg: (None, None))
        assert corners.detect_corners(img, None) == (None, 0, None)


class TestDetectCornersMLConfigNone:
    def test_config_none_defaults(self, monkeypatch):
        monkeypatch.setattr(corners, "_detect_via_cascade", lambda img, cfg: None)
        monkeypatch.setattr(corners, "_detect_via_legacy", lambda img: None)
        img = np.zeros((1000, 800, 3), dtype=np.uint8)
        # config=None exercises the default-construction line
        assert corners.detect_corners_ml(img, None) == (None, None)
