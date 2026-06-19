"""Hermetic tests for the untested parts of pagescan.pipeline.

ML and edge backends are mocked; no network and no real model weights.
"""

import numpy as np
import cv2
import pytest

from pagescan import pipeline
from pagescan.config import ScanConfig


# ---------- helpers ----------

def _quad(h, w, inset=0.1):
    return np.array([
        [int(inset * w), int(inset * h)],
        [int((1 - inset) * w), int(inset * h)],
        [int((1 - inset) * w), int((1 - inset) * h)],
        [int(inset * w), int((1 - inset) * h)],
    ], dtype=np.float32)


def _make_image(path, h=400, w=300):
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:] = (60, 80, 140)
    img[40:h - 40, 30:w - 30] = (240, 240, 245)
    cv2.imwrite(str(path), img)
    return str(path)


def _disable_all_edges(monkeypatch):
    """Make every _conservative_crop backend return 'nothing found'."""
    monkeypatch.setattr(pipeline, "detect_paper_quad", lambda img, cfg: None)
    monkeypatch.setattr(pipeline, "detect_corners_segmentation", lambda img: None)
    monkeypatch.setattr(pipeline, "detect_corners_contour", lambda img, cfg: None)
    h_w = {}

    def full_paper(img, cfg):
        h, w = img.shape[:2]
        return (0, h, 0, w)  # no crop

    def full_edges(img, cfg):
        h, w = img.shape[:2]
        return (0, h, 0, w)

    monkeypatch.setattr(pipeline, "find_paper_contour", full_paper)
    monkeypatch.setattr(pipeline, "find_document_edges", full_edges)


# ---------- _conservative_crop ----------

class TestConservativeCrop:
    def _img(self, h=400, w=300):
        return np.full((h, w, 3), 200, dtype=np.uint8)

    def test_paper_quad_branch(self, monkeypatch):
        img = self._img()
        h, w = img.shape[:2]
        sentinel = np.full((50, 50, 3), 7, dtype=np.uint8)
        monkeypatch.setattr(pipeline, "detect_paper_quad", lambda i, c: _quad(h, w))
        monkeypatch.setattr(pipeline, "perspective_transform",
                            lambda i, corners: sentinel)
        # other backends must not be reached
        monkeypatch.setattr(pipeline, "detect_corners_segmentation",
                            lambda i: pytest.fail("seg should not run"))
        out = pipeline._conservative_crop(img, ScanConfig())
        assert out is sentinel

    def test_segmentation_branch(self, monkeypatch):
        img = self._img()
        h, w = img.shape[:2]
        sentinel = np.full((50, 50, 3), 8, dtype=np.uint8)
        monkeypatch.setattr(pipeline, "detect_paper_quad", lambda i, c: None)
        monkeypatch.setattr(pipeline, "detect_corners_segmentation",
                            lambda i: _quad(h, w))
        monkeypatch.setattr(pipeline, "perspective_transform",
                            lambda i, corners: sentinel)
        monkeypatch.setattr(pipeline, "detect_corners_contour",
                            lambda i, c: pytest.fail("contour should not run"))
        out = pipeline._conservative_crop(img, ScanConfig())
        assert out is sentinel

    def test_contour_branch(self, monkeypatch):
        img = self._img()
        h, w = img.shape[:2]
        sentinel = np.full((50, 50, 3), 9, dtype=np.uint8)
        monkeypatch.setattr(pipeline, "detect_paper_quad", lambda i, c: None)
        monkeypatch.setattr(pipeline, "detect_corners_segmentation", lambda i: None)
        # contour quad with coverage in (0.05, 0.95)
        monkeypatch.setattr(pipeline, "detect_corners_contour",
                            lambda i, c: _quad(h, w, inset=0.2))
        monkeypatch.setattr(pipeline, "perspective_transform",
                            lambda i, corners: sentinel)
        out = pipeline._conservative_crop(img, ScanConfig())
        assert out is sentinel

    def test_hsv_bbox_branch(self, monkeypatch):
        img = self._img(h=400, w=300)
        monkeypatch.setattr(pipeline, "detect_paper_quad", lambda i, c: None)
        monkeypatch.setattr(pipeline, "detect_corners_segmentation", lambda i: None)
        monkeypatch.setattr(pipeline, "detect_corners_contour", lambda i, c: None)
        # crop a modest margin (> 60% retained in each dim)
        monkeypatch.setattr(pipeline, "find_paper_contour",
                            lambda i, c: (20, 380, 15, 285))
        out = pipeline._conservative_crop(img, ScanConfig())
        assert out.shape[0] == 360  # 380 - 20
        assert out.shape[1] == 270  # 285 - 15

    def test_edge_bbox_fallback_branch(self, monkeypatch):
        img = self._img(h=400, w=300)
        h, w = img.shape[:2]
        monkeypatch.setattr(pipeline, "detect_paper_quad", lambda i, c: None)
        monkeypatch.setattr(pipeline, "detect_corners_segmentation", lambda i: None)
        monkeypatch.setattr(pipeline, "detect_corners_contour", lambda i, c: None)
        # paper contour finds nothing -> falls through to edges
        monkeypatch.setattr(pipeline, "find_paper_contour", lambda i, c: (0, h, 0, w))
        monkeypatch.setattr(pipeline, "find_document_edges",
                            lambda i, c: (10, 390, 10, 290))
        out = pipeline._conservative_crop(img, ScanConfig())
        assert out.shape[0] == 380
        assert out.shape[1] == 280

    def test_no_crop_returns_full(self, monkeypatch):
        img = self._img()
        _disable_all_edges(monkeypatch)
        out = pipeline._conservative_crop(img, ScanConfig())
        assert out is img  # unchanged full image

    def test_overcrop_safety_returns_full(self, monkeypatch):
        img = self._img(h=400, w=300)
        monkeypatch.setattr(pipeline, "detect_paper_quad", lambda i, c: None)
        monkeypatch.setattr(pipeline, "detect_corners_segmentation", lambda i: None)
        monkeypatch.setattr(pipeline, "detect_corners_contour", lambda i, c: None)
        # crop to <60% of height -> over-crop guard keeps full image
        monkeypatch.setattr(pipeline, "find_paper_contour",
                            lambda i, c: (0, 150, 0, 300))  # 150 < 0.6*400=240
        out = pipeline._conservative_crop(img, ScanConfig())
        assert out is img


# ---------- _process_single ----------

class TestProcessSingle:
    def test_success(self, monkeypatch, tmp_path):
        inp = _make_image(tmp_path / "a.jpg")
        out = str(tmp_path / "a.pdf")
        monkeypatch.setattr(pipeline, "scan",
                            lambda i, o, config=None: {"success": True})
        name, result = pipeline._process_single((inp, out, ScanConfig()))
        assert name == "a.jpg"
        assert result["success"] is True

    def test_nonexistent_file(self, tmp_path):
        inp = str(tmp_path / "missing.jpg")
        out = str(tmp_path / "missing.pdf")
        # scan() returns success False for unreadable image; no mock needed
        name, result = pipeline._process_single((inp, out, ScanConfig(use_ml=False)))
        assert name == "missing.jpg"
        assert result["success"] is False

    def test_exception_caught(self, monkeypatch, tmp_path):
        monkeypatch.setattr(pipeline, "scan",
                            lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
        name, result = pipeline._process_single(
            (str(tmp_path / "x.jpg"), str(tmp_path / "x.pdf"), ScanConfig()))
        assert result["success"] is False
        assert "boom" in result["message"]


# ---------- scan_batch ----------

class TestScanBatch:
    def test_processes_all_workers1(self, tmp_path):
        in_dir = tmp_path / "in"
        in_dir.mkdir()
        out_dir = tmp_path / "out"
        for i in range(3):
            _make_image(in_dir / f"doc{i}.jpg")
        cfg = ScanConfig(use_ml=False, enhance=False, auto_orient=False,
                         deskew=False, shadow_removal=False, white_balance=False)
        results = pipeline.scan_batch(str(in_dir), str(out_dir),
                                      config=cfg, workers=1)
        assert set(results) == {"processed", "failed", "low_quality"}
        assert results["processed"] == 3
        assert results["failed"] == 0
        # PDFs written
        assert len(list(out_dir.glob("*.pdf"))) == 3

    def test_empty_dir(self, tmp_path):
        in_dir = tmp_path / "empty"
        in_dir.mkdir()
        results = pipeline.scan_batch(str(in_dir), str(tmp_path / "o"),
                                      config=ScanConfig(use_ml=False), workers=1)
        assert results == {"processed": 0, "failed": 0, "low_quality": 0}

    def test_workers1_counts_failures(self, monkeypatch, tmp_path):
        in_dir = tmp_path / "in2"
        in_dir.mkdir()
        _make_image(in_dir / "good.jpg")
        _make_image(in_dir / "bad.jpg")

        def fake_scan(img, out, config=None):
            if "bad" in img:
                return {"success": False, "message": "nope"}
            return {"success": True, "quality_passed": False,
                    "quality_score": 0.1, "message": "low"}

        monkeypatch.setattr(pipeline, "scan", fake_scan)
        results = pipeline.scan_batch(str(in_dir), str(tmp_path / "o2"),
                                      config=ScanConfig(), workers=1)
        assert results["processed"] == 1
        assert results["failed"] == 1
        assert results["low_quality"] == 1


# ---------- scan() ML branches ----------

class TestScanMLBranch:
    def _setup_identity(self, monkeypatch):
        monkeypatch.setattr(pipeline, "auto_rotate", lambda doc, lang=None: doc)
        monkeypatch.setattr(pipeline, "deskew", lambda doc: (doc, 0.0))
        monkeypatch.setattr(pipeline, "remove_shadows", lambda doc: doc)
        monkeypatch.setattr(pipeline, "white_balance", lambda doc: doc)
        monkeypatch.setattr(pipeline, "enhance_document", lambda doc: doc)
        monkeypatch.setattr(pipeline, "perspective_transform",
                            lambda img, corners: img)
        monkeypatch.setattr(pipeline, "check_quality",
                            lambda doc, cfg: (True, 0.9, "ok"))

    def test_cascade_method_writes_pdf(self, monkeypatch, tmp_path):
        inp = _make_image(tmp_path / "doc.jpg")
        out = str(tmp_path / "doc.pdf")
        self._setup_identity(monkeypatch)
        monkeypatch.setattr(pipeline, "detect_corners",
                            lambda img, cfg: (_quad(*img.shape[:2]), 0, "cascade"))
        cfg = ScanConfig(auto_orient=False, deskew=False, enhance=False,
                         shadow_removal=False, white_balance=False)
        result = pipeline.scan(inp, out, config=cfg)
        assert result["success"] is True
        assert result["method"] == "cascade"
        import os
        assert os.path.exists(out) and os.path.getsize(out) > 0

    def test_ml_pre_rotation_path(self, monkeypatch, tmp_path):
        inp = _make_image(tmp_path / "rot.jpg", h=400, w=300)
        out = str(tmp_path / "rot.pdf")
        self._setup_identity(monkeypatch)

        def fake_detect(img, cfg):
            # rotation_k=1 -> exercises the rotate + undo lines
            rh, rw = img.shape[:2]
            return _quad(rh, rw), 1, "legacy"

        monkeypatch.setattr(pipeline, "detect_corners", fake_detect)
        cfg = ScanConfig(auto_orient=False, deskew=False, enhance=False,
                         shadow_removal=False, white_balance=False)
        result = pipeline.scan(inp, out, config=cfg)
        assert result["success"] is True
        assert result["method"] == "legacy"

    def test_conservative_method_when_ml_none(self, monkeypatch, tmp_path):
        inp = _make_image(tmp_path / "cons.jpg")
        out = str(tmp_path / "cons.pdf")
        self._setup_identity(monkeypatch)
        monkeypatch.setattr(pipeline, "detect_corners",
                            lambda img, cfg: (None, 0, None))
        _disable_all_edges(monkeypatch)
        cfg = ScanConfig(auto_orient=False, deskew=False, enhance=False,
                         shadow_removal=False, white_balance=False)
        result = pipeline.scan(inp, out, config=cfg)
        assert result["success"] is True
        assert result["method"] == "conservative"

    def test_debug_branch_writes_intermediates(self, monkeypatch, tmp_path):
        inp = _make_image(tmp_path / "dbg.jpg")
        out = str(tmp_path / "dbg.pdf")
        self._setup_identity(monkeypatch)
        monkeypatch.setattr(pipeline, "detect_corners",
                            lambda img, cfg: (_quad(*img.shape[:2]), 0, "cascade"))
        dbg = tmp_path / "dbgdir"
        cfg = ScanConfig(auto_orient=False, deskew=False, shadow_removal=False,
                         white_balance=False, debug=True, debug_dir=str(dbg))
        result = pipeline.scan(inp, out, config=cfg)
        assert result["success"] is True
        # corners.jpg + enhanced_color.jpg + result.jpg written
        assert (dbg / "corners.jpg").exists()
        assert (dbg / "result.jpg").exists()

    def test_config_none_default(self, monkeypatch, tmp_path):
        inp = _make_image(tmp_path / "def.jpg")
        out = str(tmp_path / "def.pdf")
        self._setup_identity(monkeypatch)
        monkeypatch.setattr(pipeline, "detect_corners",
                            lambda img, cfg: (None, 0, None))
        _disable_all_edges(monkeypatch)
        # auto_orient default True path also exercised via identity mocks
        result = pipeline.scan(inp, out, config=None)
        assert result["success"] is True

    def test_output_path_none_derives_pdf(self, monkeypatch, tmp_path):
        inp = _make_image(tmp_path / "noout.jpg")
        self._setup_identity(monkeypatch)
        monkeypatch.setattr(pipeline, "detect_corners",
                            lambda img, cfg: (None, 0, None))
        _disable_all_edges(monkeypatch)
        cfg = ScanConfig(auto_orient=False, deskew=False, enhance=False,
                         shadow_removal=False, white_balance=False)
        result = pipeline.scan(inp, None, config=cfg)
        assert result["output_path"].endswith(".pdf")
        import os
        assert os.path.exists(result["output_path"])

    def test_unreadable_returns_failure(self, tmp_path):
        result = pipeline.scan(str(tmp_path / "nope.jpg"), str(tmp_path / "o.pdf"),
                               config=ScanConfig(use_ml=False))
        assert result["success"] is False
        assert "Cannot read" in result["message"]
