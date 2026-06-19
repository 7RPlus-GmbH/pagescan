"""Targeted tests closing the last reachable coverage gaps.

Covers the cascade/legacy detection-backend wrappers in ``corners`` (mocked
at the module boundary), the ``_validate_and_repair`` repair-return branches,
the ``scan_batch`` default/multiprocessing paths in ``pipeline``, and the
``white_balance`` no-paper early return in ``enhance``.
"""
from __future__ import annotations

import cv2
import numpy as np
import pytest

import pagescan.corners as C
import pagescan.pipeline as P
from pagescan.config import ScanConfig
from pagescan.enhance import white_balance


def _make_doc(w=320, h=420, bg=(60, 90, 120)):
    img = np.full((h, w, 3), bg, np.uint8)
    cv2.rectangle(img, (60, 60), (w - 60, h - 60), (245, 245, 245), -1)
    return img


def _quad():
    return np.array([[10, 10], [200, 10], [200, 300], [10, 300]], np.float32)


# --------------------------------------------------------------------------
# corners._detect_via_cascade  (the YOLO -> HQ-SAM -> quad orchestration body)
# --------------------------------------------------------------------------

def test_cascade_success(monkeypatch):
    assert C.HAS_CASCADE  # detector + segmenter import cleanly in this env
    monkeypatch.setattr(C.detector, "detect",
                        lambda img, conf_threshold=0.25: (np.array([0, 0, 200, 300], np.float32), 0.9))
    monkeypatch.setattr(C.segmenter, "segment", lambda img, bbox: np.ones((300, 200), bool))
    monkeypatch.setattr(C.segmenter, "mask_to_quad", lambda mask: _quad())
    out = C._detect_via_cascade(np.zeros((400, 300, 3), np.uint8), ScanConfig())
    assert out is not None and out.shape == (4, 2)


def test_cascade_no_detection(monkeypatch):
    monkeypatch.setattr(C.detector, "detect", lambda img, conf_threshold=0.25: None)
    assert C._detect_via_cascade(np.zeros((400, 300, 3), np.uint8), ScanConfig()) is None


def test_cascade_no_mask(monkeypatch):
    monkeypatch.setattr(C.detector, "detect",
                        lambda img, conf_threshold=0.25: (np.array([0, 0, 200, 300], np.float32), 0.9))
    monkeypatch.setattr(C.segmenter, "segment", lambda img, bbox: None)
    assert C._detect_via_cascade(np.zeros((400, 300, 3), np.uint8), ScanConfig()) is None


def test_cascade_mask_to_quad_fails(monkeypatch):
    monkeypatch.setattr(C.detector, "detect",
                        lambda img, conf_threshold=0.25: (np.array([0, 0, 200, 300], np.float32), 0.9))
    monkeypatch.setattr(C.segmenter, "segment", lambda img, bbox: np.ones((300, 200), bool))
    monkeypatch.setattr(C.segmenter, "mask_to_quad", lambda mask: None)
    assert C._detect_via_cascade(np.zeros((400, 300, 3), np.uint8), ScanConfig()) is None


@pytest.mark.parametrize("exc", [
    FileNotFoundError("weights.onnx"),
    ImportError("torch missing"),
    RuntimeError("unexpected"),
])
def test_cascade_exceptions_return_none(monkeypatch, exc):
    def _raise(*a, **k):
        raise exc
    monkeypatch.setattr(C.detector, "detect", _raise)
    assert C._detect_via_cascade(np.zeros((400, 300, 3), np.uint8), ScanConfig()) is None


# --------------------------------------------------------------------------
# corners._detect_via_legacy
# --------------------------------------------------------------------------

def test_legacy_success(monkeypatch):
    assert C.HAS_LEGACY_ML
    monkeypatch.setattr(C, "detect_corners_onnx", lambda img: _quad())
    out = C._detect_via_legacy(np.zeros((400, 300, 3), np.uint8))
    assert out is not None and out.shape == (4, 2) and out.dtype == np.float32


def test_legacy_none(monkeypatch):
    monkeypatch.setattr(C, "detect_corners_onnx", lambda img: None)
    assert C._detect_via_legacy(np.zeros((400, 300, 3), np.uint8)) is None


def test_legacy_wrong_shape(monkeypatch):
    monkeypatch.setattr(C, "detect_corners_onnx", lambda img: np.zeros((3, 2), np.float32))
    assert C._detect_via_legacy(np.zeros((400, 300, 3), np.uint8)) is None


def test_legacy_raises(monkeypatch):
    def _raise(img):
        raise RuntimeError("boom")
    monkeypatch.setattr(C, "detect_corners_onnx", _raise)
    assert C._detect_via_legacy(np.zeros((400, 300, 3), np.uint8)) is None


# --------------------------------------------------------------------------
# corners._validate_and_repair repair-return branch (both repair sub-branches)
# --------------------------------------------------------------------------

def test_validate_repairs_and_returns_lower_branch():
    # Off bottom-right corner; tb diverges >15deg, right side is the SHORT one
    # -> hits the `right_len < left_len*0.85` repair branch, repaired quad valid.
    quad = np.array([[100, 100], [900, 120], [880, 500], [120, 700]], np.float32)
    out = C._validate_and_repair(quad, h=1000, w=1000, min_coverage=0.05)
    assert out is not None
    tb, lr = C._check_parallel(out)
    assert tb <= 10 and lr <= 10


def test_validate_repairs_and_returns_upper_branch():
    # Off bottom-left corner; tb diverges, right side is the LONG one
    # -> hits the else (`repaired[3] = ...`) repair branch.
    quad = np.array([[100, 100], [900, 120], [880, 800], [120, 500]], np.float32)
    out = C._validate_and_repair(quad, h=1000, w=1000, min_coverage=0.05)
    assert out is not None
    tb, lr = C._check_parallel(out)
    assert tb <= 10 and lr <= 10


# --------------------------------------------------------------------------
# pipeline.scan_batch default args + multiprocessing path
# --------------------------------------------------------------------------

def test_scan_batch_defaults_conservative(tmp_path, monkeypatch):
    # config=None and output_dir=None defaults, via the conservative path
    # (no ML / orientation download — those boundaries are stubbed).
    for i in range(2):
        cv2.imwrite(str(tmp_path / f"p{i}.jpg"), _make_doc())
    monkeypatch.setattr(P, "detect_corners", lambda img, cfg: (None, 0, None))
    monkeypatch.setattr(P, "auto_rotate", lambda img, lang="x": img)
    res = P.scan_batch(str(tmp_path))  # output_dir=None, config=None, workers=None
    assert res["processed"] == 2 and res["failed"] == 0
    assert sorted(p.suffix for p in tmp_path.glob("*.pdf")) == [".pdf", ".pdf"]


def test_scan_batch_sequential_exception(tmp_path, monkeypatch):
    # scan() raising inside the workers<=1 loop -> counted as failed.
    cv2.imwrite(str(tmp_path / "a.jpg"), _make_doc())

    def _boom(*a, **k):
        raise RuntimeError("scan exploded")
    monkeypatch.setattr(P, "scan", _boom)
    res = P.scan_batch(str(tmp_path), workers=1)
    assert res["failed"] == 1 and res["processed"] == 0


_OFFLINE = dict(use_ml=False, auto_orient=False, deskew=False,
                enhance=False, shadow_removal=False, white_balance=False)


def test_scan_batch_multiprocessing(tmp_path):
    # workers>1 spawns real subprocesses; use a fully offline config so the
    # children need no models or network.
    for i in range(2):
        cv2.imwrite(str(tmp_path / f"m{i}.jpg"), _make_doc())
    res = P.scan_batch(str(tmp_path), output_dir=str(tmp_path / "out"),
                       config=ScanConfig(**_OFFLINE), workers=2)
    assert res["processed"] == 2 and res["failed"] == 0


def test_scan_batch_multiprocessing_failure(tmp_path):
    # A worker returning success=False drives the parallel-branch failure path.
    cv2.imwrite(str(tmp_path / "good.jpg"), _make_doc())
    (tmp_path / "bad.jpg").write_bytes(b"not a real image")
    res = P.scan_batch(str(tmp_path), output_dir=str(tmp_path / "o"),
                       config=ScanConfig(**_OFFLINE), workers=2)
    assert res["processed"] == 1 and res["failed"] == 1


# --------------------------------------------------------------------------
# enhance.white_balance no-paper early return
# --------------------------------------------------------------------------

def test_white_balance_no_paper_returns_unchanged():
    img = np.zeros((60, 60, 3), np.uint8)  # all black: no bright low-sat pixels
    out = white_balance(img)
    assert np.array_equal(out, img)
