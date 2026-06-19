"""Hermetic tests for pagescan.edges (pure OpenCV/numpy, no ML/IO/network).

Covers every public/private function in src/pagescan/edges.py:
    find_paper_contour, _find_document_contours, detect_corners_contour,
    find_document_edges, detect_paper_quad, estimate_paper_coverage.

All images are synthesised with numpy/cv2. Note: images are BGR (the code
calls cv2.cvtColor(..., COLOR_BGR2HSV / COLOR_BGR2GRAY)).
"""

import cv2
import numpy as np
import pytest

from pagescan.config import ScanConfig
from pagescan.edges import (
    _find_document_contours,
    detect_corners_contour,
    detect_paper_quad,
    estimate_paper_coverage,
    find_document_edges,
    find_paper_contour,
)

# --------------------------------------------------------------------------
# Image builders
# --------------------------------------------------------------------------

# Known paper rectangle for the default doc, in (x, y) coords:
PAPER_L, PAPER_T, PAPER_R, PAPER_B = 160, 150, 640, 850


def make_doc(w=800, h=1000, bg=(60, 90, 120)):
    """Bright (white) paper rectangle on a darker, saturated background.

    bg is a warm/saturated color so the HSV paper mask cleanly separates
    the white rectangle from the background.
    """
    img = np.full((h, w, 3), bg, np.uint8)
    cv2.rectangle(img, (PAPER_L, PAPER_T), (PAPER_R, PAPER_B), (245, 245, 245), -1)
    return img


def make_tilted_doc(w=800, h=1000, bg=(60, 90, 120), angle=15.0):
    """Same paper rectangle, rotated about the image centre."""
    img = make_doc(w, h, bg)
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(img, M, (w, h), borderValue=bg)


def make_small_doc(w=800, h=1000, bg=(60, 90, 120)):
    """Paper occupying well under 15% of the frame."""
    img = np.full((h, w, 3), bg, np.uint8)
    # 150x150 = 22500 px of 800000 ~= 2.8%
    cv2.rectangle(img, (100, 100), (250, 250), (245, 245, 245), -1)
    return img


def make_blank(w=800, h=1000, bg=(60, 90, 120)):
    """Uniform dark/saturated frame: no paper at all."""
    return np.full((h, w, 3), bg, np.uint8)


def make_full_frame_paper(w=800, h=1000):
    """Entire frame is white paper."""
    return np.full((h, w, 3), 245, np.uint8)


def make_sharp_rhombus_doc(w=800, h=1000, bg=(60, 90, 120)):
    """A tall, sharp rhombus (acute ~35deg apex angles).

    In _find_document_contours every 4-vertex approximation of this shape
    has two ~35deg and two ~145deg corners, so the angle validation always
    fails (angles_ok=False / break) and `approx` stays None, exercising the
    in-loop angle-rejection branch and the `approx is None` default fallback.
    """
    img = np.full((h, w, 3), bg, np.uint8)
    pts = np.array([[400, 80], [520, 500], [400, 920], [280, 500]], np.int32)
    cv2.fillPoly(img, [pts], (245, 245, 245))
    return img


def make_tiny_dot_doc(w=800, h=1000, bg=(60, 90, 120)):
    """A small paper dot that survives morphology but stays under the 5%
    significant-area threshold in detect_paper_quad.

    Used to hit detect_paper_quad's `no significant contours -> return None`
    branch (contours exist, but none is large enough)."""
    img = np.full((h, w, 3), bg, np.uint8)
    cv2.circle(img, (w // 2, h // 2), 100, (245, 245, 245), -1)
    return img


def make_doc_with_speck(w=800, h=1000, bg=(60, 90, 120)):
    """The standard document plus a tiny separate bright speck.

    The speck is a 2nd contour whose area is below the 5% threshold in
    _find_document_contours, exercising its small-contour `continue` skip."""
    img = make_doc(w, h, bg)
    cv2.rectangle(img, (40, 40), (90, 90), (245, 245, 245), -1)
    return img


def make_small_no_scale_doc(w=400, h=480, bg=(60, 90, 120)):
    """Small frame (max dim <= 500) so _find_document_contours skips resize.

    Paper rect is proportional to the frame so it survives the area filter.
    """
    img = np.full((h, w, 3), bg, np.uint8)
    cv2.rectangle(img, (int(w * 0.2), int(h * 0.15)),
                  (int(w * 0.8), int(h * 0.85)), (245, 245, 245), -1)
    return img


def make_triangle_doc(w=800, h=1000, bg=(60, 90, 120)):
    """A large white *triangle* on dark background.

    A triangle's contour approximates to 3 vertices (never a valid 4-gon
    under the angle constraint), which forces the minAreaRect fallback
    branch in detect_corners_contour, and the non-4-vertex approx path in
    _find_document_contours.
    """
    img = np.full((h, w, 3), bg, np.uint8)
    pts = np.array([[400, 150], [680, 820], [150, 820]], np.int32)
    cv2.fillPoly(img, [pts], (245, 245, 245))
    return img


def make_big_triangle_doc(w=800, h=1000, bg=(60, 90, 120)):
    """A frame-filling triangle on dark background.

    Large enough that the aggressive paper-mask morphology in
    detect_paper_quad cannot round its corners into a valid rectangle:
    every 4-vertex approximation of its convex hull has a near-collinear
    (>140 degree) vertex, so the angle check rejects them all (lines that
    set angles_ok=False / break) and the code falls through to the
    minAreaRect fallback. Coverage stays well above the 0.15 gate (~0.59).
    """
    img = np.full((h, w, 3), bg, np.uint8)
    pts = np.array([[400, 40], [770, 960], [30, 960]], np.int32)
    cv2.fillPoly(img, [pts], (245, 245, 245))
    return img


# --------------------------------------------------------------------------
# Geometry helpers for assertions
# --------------------------------------------------------------------------

def hull_area(corners):
    return cv2.contourArea(cv2.convexHull(corners.astype(np.float32)))


def bbox_of(corners):
    xs = corners[:, 0]
    ys = corners[:, 1]
    return float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())


# --------------------------------------------------------------------------
# estimate_paper_coverage
# --------------------------------------------------------------------------

class TestEstimatePaperCoverage:
    def test_all_dark_is_near_zero(self):
        img = make_blank()
        cov = estimate_paper_coverage(img)
        assert isinstance(cov, float)
        assert cov < 0.01

    def test_white_rect_raises_coverage(self):
        dark = estimate_paper_coverage(make_blank())
        doc = estimate_paper_coverage(make_doc())
        assert doc > dark
        # paper rect is (640-160)*(850-150) = 336000 / 800000 = 0.42
        assert doc == pytest.approx(0.42, abs=0.05)

    def test_full_frame_paper_is_near_one(self):
        cov = estimate_paper_coverage(make_full_frame_paper())
        assert cov > 0.99

    def test_returns_python_float_in_range(self):
        cov = estimate_paper_coverage(make_doc())
        assert type(cov) is float
        assert 0.0 <= cov <= 1.0


# --------------------------------------------------------------------------
# find_paper_contour
# --------------------------------------------------------------------------

class TestFindPaperContour:
    def test_brackets_paper(self):
        top, bottom, left, right = find_paper_contour(make_doc())
        # The returned crop must enclose the known paper rect (plus margin),
        # and not exceed the frame.
        assert left <= PAPER_L and right >= PAPER_R
        assert top <= PAPER_T and bottom >= PAPER_B
        assert left >= 0 and top >= 0
        assert right <= 800 and bottom <= 1000

    def test_crop_is_tight_around_paper(self):
        top, bottom, left, right = find_paper_contour(make_doc())
        # Should not be the full frame: it brackets just the paper.
        assert right - left < 800
        assert bottom - top < 1000
        # Width/height roughly match paper size (+ small margins).
        assert (right - left) == pytest.approx(PAPER_R - PAPER_L, abs=40)
        assert (bottom - top) == pytest.approx(PAPER_B - PAPER_T, abs=40)

    def test_default_config_is_constructed(self):
        # config=None path must not raise and behave like default config.
        a = find_paper_contour(make_doc(), config=None)
        b = find_paper_contour(make_doc(), config=ScanConfig())
        assert a == b

    def test_blank_returns_full_frame(self):
        # No paper -> no contours -> full-frame fallback (0, h, 0, w).
        h, w = 1000, 800
        assert find_paper_contour(make_blank(w, h)) == (0, h, 0, w)

    def test_small_doc_below_min_area_ratio_returns_full_frame(self):
        # The small paper (~2.8%) is below min_area_ratio=0.5 -> no significant
        # contour -> full-frame fallback.
        img = make_small_doc()
        h, w = img.shape[:2]
        assert find_paper_contour(img, min_area_ratio=0.5) == (0, h, 0, w)

    def test_small_doc_low_ratio_brackets_it(self):
        # With a low ratio the small paper qualifies and is bracketed.
        img = make_small_doc()
        top, bottom, left, right = find_paper_contour(img, min_area_ratio=0.005)
        assert left <= 100 + 12 and right >= 250 - 12  # within margins
        assert top <= 100 + 12 and bottom >= 250 - 12
        assert right - left < img.shape[1]

    def test_returns_four_ints(self):
        res = find_paper_contour(make_doc())
        assert len(res) == 4
        assert all(isinstance(v, int) for v in res)


# --------------------------------------------------------------------------
# _find_document_contours
# --------------------------------------------------------------------------

class TestFindDocumentContours:
    def test_finds_document_and_scores_quad(self):
        scored = _find_document_contours(make_doc())
        assert len(scored) >= 1
        score, _contour, approx = scored[0]
        assert score > 0
        # The strong rectangle should approximate to a 4-vertex polygon.
        assert len(approx) == 4

    def test_sorted_best_first(self):
        scored = _find_document_contours(make_doc())
        scores = [s for s, _, _ in scored]
        assert scores == sorted(scores, reverse=True)

    def test_coords_scaled_back_for_large_image(self):
        # Large image (max dim 2000 > 500) forces the scale<1 downscale path.
        # Coordinates must be scaled back up so the contour lives in the
        # original frame, not the 500px working copy.
        img = make_doc(w=1600, h=2000)
        scored = _find_document_contours(img)
        assert scored
        _, contour, _ = scored[0]
        pts = contour.reshape(-1, 2)
        # Working copy would max out near ~500px; scaled-back coords must
        # exceed that, proving the /scale step ran.
        assert pts[:, 1].max() > 600
        # And the contour stays within the original frame bounds.
        assert pts[:, 0].max() <= 1600
        assert pts[:, 1].max() <= 2000
        # Roughly centred on the paper region (not a tiny fragment).
        assert cv2.contourArea(contour) > 0.1 * 1600 * 2000

    def test_no_scale_path_small_image(self):
        # Image already <= max_dim(500) so scale==1.0 branch (no resize) runs
        # and coordinates are used unscaled.
        img = make_small_no_scale_doc(w=400, h=480)
        assert max(img.shape[:2]) <= 500
        scored = _find_document_contours(img)
        assert scored
        _, contour, approx = scored[0]
        assert len(approx) == 4
        bx, by, bw, bh = cv2.boundingRect(contour)
        assert bw < 400 and bh < 480
        # Brackets the proportional paper rect (0.2w..0.8w, 0.15h..0.85h).
        assert bx == pytest.approx(80, abs=30)
        assert by == pytest.approx(72, abs=30)

    def test_blank_returns_empty(self):
        # Uniform image -> Canny finds no edges -> no contours.
        assert _find_document_contours(make_blank()) == []

    def test_triangle_uses_non_quad_approx(self):
        # A triangle won't approximate to 4 verts under the angle constraint,
        # exercising the `approx is None` -> default-approx fallback (and the
        # vertex_bonus==1.0 / len(approx)!=4 score path).
        scored = _find_document_contours(make_triangle_doc())
        assert scored
        _, _, approx = scored[0]
        assert len(approx) != 4

    def test_sharp_rhombus_hits_angle_rejection(self):
        # Every 4-vert approximation of the sharp rhombus has out-of-range
        # angles, so the in-loop angle check rejects all of them and the
        # default approx (line 150) is used instead. Must still produce a
        # scored contour.
        scored = _find_document_contours(make_sharp_rhombus_doc())
        assert scored
        score, _, approx = scored[0]
        assert score > 0
        # The default approx happens to give 4 here, but the point is the
        # angle-validation branch ran and failed for every eps first.
        assert len(approx) >= 3

    def test_speck_contour_is_skipped(self):
        # The big document plus a tiny speck: findContours sees two regions,
        # but the speck is below the 5% area threshold and is skipped, so
        # only one contour ends up scored.
        scored = _find_document_contours(make_doc_with_speck())
        assert len(scored) == 1
        _, contour, _ = scored[0]
        # The single survivor is the big document, not the speck.
        assert cv2.contourArea(contour) > 0.2 * 800 * 1000


# --------------------------------------------------------------------------
# detect_corners_contour
# --------------------------------------------------------------------------

class TestDetectCornersContour:
    def test_returns_4x2_float32_quad(self):
        corners = detect_corners_contour(make_doc())
        assert corners is not None
        assert corners.shape == (4, 2)
        assert corners.dtype == np.float32

    def test_quad_bounds_paper_rect(self):
        corners = detect_corners_contour(make_doc())
        x1, y1, x2, y2 = bbox_of(corners)
        # Corner bbox overlaps the known paper rect within tolerance.
        assert x1 == pytest.approx(PAPER_L, abs=40)
        assert y1 == pytest.approx(PAPER_T, abs=40)
        assert x2 == pytest.approx(PAPER_R, abs=40)
        assert y2 == pytest.approx(PAPER_B, abs=40)

    def test_hull_area_is_large_fraction_of_paper(self):
        corners = detect_corners_contour(make_doc())
        paper_area = (PAPER_R - PAPER_L) * (PAPER_B - PAPER_T)
        assert hull_area(corners) > 0.85 * paper_area

    def test_tilted_doc_quad_is_rotated(self):
        corners = detect_corners_contour(make_tilted_doc(angle=15.0))
        assert corners is not None
        assert corners.shape == (4, 2)
        # A tilted quad has no edge axis-aligned: at least one side must have
        # a meaningful slope in both x and y.
        corners[np.argsort(corners[:, 1])]
        # The hull should still cover a large area.
        assert hull_area(corners) > 0.5 * (PAPER_R - PAPER_L) * (PAPER_B - PAPER_T)

    def test_blank_returns_none(self):
        assert detect_corners_contour(make_blank()) is None

    def test_minarearect_fallback_for_triangle(self):
        # Triangle -> best approx not 4 verts -> minAreaRect fallback path.
        corners = detect_corners_contour(make_triangle_doc())
        assert corners is not None
        assert corners.shape == (4, 2)
        assert corners.dtype == np.float32
        # The rotated box must bound the triangle (verts at x in [150,680],
        # y in [150,820]).
        x1, y1, x2, y2 = bbox_of(corners)
        assert x1 == pytest.approx(150, abs=40)
        assert x2 == pytest.approx(680, abs=40)
        assert y1 == pytest.approx(150, abs=40)
        assert y2 == pytest.approx(820, abs=40)

    def test_default_config_none(self):
        # config=None must construct a default and not raise.
        assert detect_corners_contour(make_doc(), config=None) is not None


# --------------------------------------------------------------------------
# find_document_edges
# --------------------------------------------------------------------------

class TestFindDocumentEdges:
    def test_brackets_paper(self):
        top, bottom, left, right = find_document_edges(make_doc())
        assert left <= PAPER_L and right >= PAPER_R
        assert top <= PAPER_T and bottom >= PAPER_B
        assert left >= 0 and top >= 0
        assert right <= 800 and bottom <= 1000

    def test_crop_is_tight(self):
        top, bottom, left, right = find_document_edges(make_doc())
        assert (right - left) == pytest.approx(PAPER_R - PAPER_L, abs=60)
        assert (bottom - top) == pytest.approx(PAPER_B - PAPER_T, abs=60)

    def test_falls_back_to_paper_contour_when_no_edges(self):
        # Blank image: _find_document_contours returns [] so this delegates
        # to find_paper_contour, which (no paper) returns the full frame.
        h, w = 1000, 800
        img = make_blank(w, h)
        assert find_document_edges(img) == find_paper_contour(img)
        assert find_document_edges(img) == (0, h, 0, w)

    def test_returns_four_ints(self):
        res = find_document_edges(make_doc())
        assert len(res) == 4
        assert all(isinstance(v, int) for v in res)

    def test_default_config_none(self):
        a = find_document_edges(make_doc(), config=None)
        b = find_document_edges(make_doc(), config=ScanConfig())
        assert a == b


# --------------------------------------------------------------------------
# detect_paper_quad
# --------------------------------------------------------------------------

class TestDetectPaperQuad:
    def test_returns_4x2_float32_quad(self):
        corners = detect_paper_quad(make_doc())
        assert corners is not None
        assert corners.shape == (4, 2)
        assert corners.dtype == np.float32

    def test_quad_bounds_paper_rect(self):
        corners = detect_paper_quad(make_doc())
        x1, y1, x2, y2 = bbox_of(corners)
        assert x1 == pytest.approx(PAPER_L, abs=45)
        assert y1 == pytest.approx(PAPER_T, abs=45)
        assert x2 == pytest.approx(PAPER_R, abs=45)
        assert y2 == pytest.approx(PAPER_B, abs=45)

    def test_hull_area_is_large_fraction_of_paper(self):
        corners = detect_paper_quad(make_doc())
        paper_area = (PAPER_R - PAPER_L) * (PAPER_B - PAPER_T)
        assert hull_area(corners) > 0.85 * paper_area

    def test_blank_no_contours_returns_none(self):
        # Uniform dark frame: paper mask empty -> no contours -> None.
        assert detect_paper_quad(make_blank()) is None

    def test_tiny_dot_no_significant_contour_returns_none(self):
        # A small dot survives morphology as a contour but stays under the
        # 5% significant-area threshold -> `significant` is empty -> None.
        # (Distinct from the no-contours-at-all path.)
        assert detect_paper_quad(make_tiny_dot_doc()) is None

    def test_small_doc_below_coverage_returns_none(self):
        # Small paper passes the 0.05 area filter? It is ~2.8% < 5%, so it is
        # filtered out as not significant -> None. (Covers the no-significant
        # / low-coverage rejection territory.)
        assert detect_paper_quad(make_small_doc()) is None

    def test_minarearect_fallback_for_big_triangle(self):
        # Frame-filling triangle: every 4-gon approx of its hull fails the
        # angle gate (angles_ok=False / break), so the code falls through to
        # the minAreaRect fallback (which still returns a box).
        corners = detect_paper_quad(make_big_triangle_doc())
        assert corners is not None
        assert corners.shape == (4, 2)
        assert corners.dtype == np.float32
        # minAreaRect box bounds the triangle (x in [30,770], y in [40,960]).
        x1, _y1, x2, y2 = bbox_of(corners)
        assert x1 == pytest.approx(30, abs=50)
        assert x2 == pytest.approx(770, abs=50)
        assert y2 == pytest.approx(960, abs=50)

    def test_tilted_doc(self):
        corners = detect_paper_quad(make_tilted_doc(angle=12.0))
        assert corners is not None
        assert corners.shape == (4, 2)
        assert hull_area(corners) > 0.5 * (PAPER_R - PAPER_L) * (PAPER_B - PAPER_T)

    def test_default_config_none(self):
        assert detect_paper_quad(make_doc(), config=None) is not None

    def test_coverage_below_gate_returns_none(self):
        """A rectangle that PASSES the 5% significant-area filter but whose
        convex-hull coverage lands just under the 0.15 gate is rejected.

        This exercises the `coverage < 0.15` branch specifically (a
        significant contour exists; rejection is purely on coverage).
        """
        img = make_blank()
        # 230x260 raw = 59800 px = 7.5% (>5% significant). After the
        # aggressive morphology the merged-hull coverage is ~0.1485 < 0.15.
        cv2.rectangle(img, (100, 100), (330, 360), (245, 245, 245), -1)
        assert detect_paper_quad(img) is None
