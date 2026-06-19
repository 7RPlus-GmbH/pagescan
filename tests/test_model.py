"""Hermetic tests for pagescan.model.

ONNX inference and all network downloads (HF Hub, Google Drive) are mocked.
No real weights are ever loaded and no network access occurs.
"""
from __future__ import annotations

import sys
import types

import numpy as np
import pytest

from pagescan import model as m

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def reset_sessions():
    """Reset the module-level session singleton cache between tests."""
    m._sessions.clear()
    yield
    m._sessions.clear()


@pytest.fixture
def isolated_dirs(tmp_path, monkeypatch):
    """Point both the project-local data dir and the cache dir at tmp_path.

    Returns (data_dir, cache_dir) — both empty to start with.
    """
    data_dir = tmp_path / "data" / "model"
    cache_dir = tmp_path / "cache"
    data_dir.mkdir(parents=True)
    cache_dir.mkdir(parents=True)
    monkeypatch.setattr(m, "_DATA_MODEL_DIR", data_dir)
    monkeypatch.setenv("PAGESCAN_CACHE", str(cache_dir))
    return data_dir, cache_dir


def _make_big_file(path, size=200_000):
    """Create a file larger than the 100KB validity threshold."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\x00" * size)


class _FakeIO:
    """Stand-in for an onnxruntime input/output descriptor with a `.name`."""

    def __init__(self, name):
        self.name = name


class _FakeSession:
    """Fake onnxruntime InferenceSession returning a fixed output."""

    def __init__(self, output, input_name="input", output_name="output", path=None):
        self._output = output
        self._input_name = input_name
        self._output_name = output_name
        self.path = path
        self.run_calls = []

    def get_inputs(self):
        return [_FakeIO(self._input_name)]

    def get_outputs(self):
        return [_FakeIO(self._output_name)]

    def run(self, output_names, feed):
        self.run_calls.append((output_names, feed))
        return [self._output]


# ---------------------------------------------------------------------------
# _get_cache_dir
# ---------------------------------------------------------------------------

class TestGetCacheDir:
    def test_uses_env_var(self, tmp_path, monkeypatch):
        target = tmp_path / "mycache"
        monkeypatch.setenv("PAGESCAN_CACHE", str(target))
        result = m._get_cache_dir()
        assert result == target
        assert result.is_dir()

    def test_default_location_created(self, tmp_path, monkeypatch):
        monkeypatch.delenv("PAGESCAN_CACHE", raising=False)
        monkeypatch.setattr(m.Path, "home", staticmethod(lambda: tmp_path))
        result = m._get_cache_dir()
        assert result == tmp_path / ".cache" / "pagescan"
        assert result.is_dir()


# ---------------------------------------------------------------------------
# _preprocess_heatmap  (PURE)
# ---------------------------------------------------------------------------

class TestPreprocessHeatmap:
    def test_shape_dtype_range_and_origsize(self):
        h, w = 480, 640
        img = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
        input_size = (256, 256)
        tensor, orig = m._preprocess_heatmap(img, input_size)

        assert tensor.shape == (1, 3, 256, 256)
        assert tensor.dtype == np.float32
        assert tensor.min() >= 0.0
        assert tensor.max() <= 1.0
        assert orig == (h, w)

    def test_nonsquare_input_size(self):
        img = np.zeros((100, 200, 3), dtype=np.uint8)
        tensor, orig = m._preprocess_heatmap(img, (320, 256))
        # cv2.resize takes (width, height); NCHW => (1,3,height,width)
        assert tensor.shape == (1, 3, 256, 320)
        assert orig == (100, 200)

    def test_white_image_maps_to_one(self):
        img = np.full((10, 10, 3), 255, dtype=np.uint8)
        tensor, _ = m._preprocess_heatmap(img, (8, 8))
        assert np.allclose(tensor, 1.0)


# ---------------------------------------------------------------------------
# _postprocess_heatmap  (PURE)
# ---------------------------------------------------------------------------

class TestPostprocessHeatmap:
    def test_single_hot_pixel_per_channel_scales(self):
        hm_h, hm_w = 64, 64
        oh, ow = 256, 128
        heatmaps = np.zeros((1, 4, hm_h, hm_w), dtype=np.float32)
        # (row, col) peak per channel
        peaks = [(0, 0), (0, 32), (16, 8), (48, 60)]
        for i, (py, px) in enumerate(peaks):
            heatmaps[0, i, py, px] = 1.0

        pts = m._postprocess_heatmap(heatmaps, (oh, ow))
        assert pts is not None
        assert pts.shape == (4, 2)
        assert pts.dtype == np.float32
        for i, (py, px) in enumerate(peaks):
            expected_cx = px * ow / hm_w
            expected_cy = py * oh / hm_h
            assert pts[i, 0] == pytest.approx(expected_cx)
            assert pts[i, 1] == pytest.approx(expected_cy)

    def test_below_threshold_returns_none(self):
        heatmaps = np.zeros((1, 4, 16, 16), dtype=np.float32)
        heatmaps[0, 0, 5, 5] = 1.0
        heatmaps[0, 1, 5, 5] = 1.0
        heatmaps[0, 2, 5, 5] = 1.0
        # channel 3 max is below HEATMAP_THRESHOLD (0.1)
        heatmaps[0, 3, 5, 5] = m.HEATMAP_THRESHOLD / 2
        assert m._postprocess_heatmap(heatmaps, (100, 100)) is None

    def test_fewer_than_four_channels_returns_none(self):
        heatmaps = np.ones((1, 3, 16, 16), dtype=np.float32)
        assert m._postprocess_heatmap(heatmaps, (100, 100)) is None


# ---------------------------------------------------------------------------
# _heatmap_confidence
# ---------------------------------------------------------------------------

class TestHeatmapConfidence:
    def test_mean_of_channel_maxima(self):
        heatmaps = np.zeros((1, 4, 8, 8), dtype=np.float32)
        maxima = [0.2, 0.4, 0.6, 0.8]
        for i, v in enumerate(maxima):
            heatmaps[0, i, 0, 0] = v
        conf = m._heatmap_confidence(heatmaps)
        assert conf == pytest.approx(np.mean(maxima))

    def test_none_returns_zero(self):
        assert m._heatmap_confidence(None) == 0.0


# ---------------------------------------------------------------------------
# _ensure_model
# ---------------------------------------------------------------------------

class TestEnsureModel:
    def test_returns_local_data_path(self, isolated_dirs):
        data_dir, _ = isolated_dirs
        filename = m.MODELS["sa24"]["filename"]
        local = data_dir / filename
        _make_big_file(local)
        assert m._ensure_model("sa24") == local

    def test_small_local_file_ignored(self, isolated_dirs, monkeypatch):
        data_dir, cache_dir = isolated_dirs
        filename = m.MODELS["sa24"]["filename"]
        # local exists but too small -> should not be returned
        _make_big_file(data_dir / filename, size=1000)

        def fake_hf(fn, dest):
            _make_big_file(dest)

        monkeypatch.setattr(m, "_download_from_hf", fake_hf)
        result = m._ensure_model("sa24")
        assert result == cache_dir / filename

    def test_returns_cache_path(self, isolated_dirs):
        _, cache_dir = isolated_dirs
        filename = m.MODELS["sa24"]["filename"]
        cached = cache_dir / filename
        _make_big_file(cached)
        assert m._ensure_model("sa24") == cached

    def test_downloads_from_gdrive_first(self, isolated_dirs, monkeypatch):
        _, cache_dir = isolated_dirs
        filename = m.MODELS["sa24"]["filename"]
        calls = []

        def fake_hf(fn, dest):
            calls.append("hf")
            _make_big_file(dest)

        def fake_gdrive(file_id, dest):
            calls.append("gdrive")
            assert file_id == m.MODELS["sa24"]["gdrive_id"]
            _make_big_file(dest)

        monkeypatch.setattr(m, "_download_from_hf", fake_hf)
        monkeypatch.setattr(m, "_download_from_gdrive", fake_gdrive)

        result = m._ensure_model("sa24")
        assert result == cache_dir / filename
        # gdrive (original upstream) tried first and succeeded; HF never touched
        assert calls == ["gdrive"]

    def test_falls_back_to_hf_mirror(self, isolated_dirs, monkeypatch):
        _, cache_dir = isolated_dirs
        filename = m.MODELS["sa24"]["filename"]
        calls = []

        def fake_gdrive(file_id, dest):
            calls.append("gdrive")
            raise RuntimeError("Drive down")

        def fake_hf(fn, dest):
            calls.append("hf")
            assert fn == filename
            _make_big_file(dest)

        monkeypatch.setattr(m, "_download_from_hf", fake_hf)
        monkeypatch.setattr(m, "_download_from_gdrive", fake_gdrive)

        result = m._ensure_model("sa24")
        assert result == cache_dir / filename
        assert calls == ["gdrive", "hf"]

    def test_both_fail_raises_filenotfound(self, isolated_dirs, monkeypatch):
        def fake_hf(fn, dest):
            raise RuntimeError("HF down")

        def fake_gdrive(file_id, dest):
            raise RuntimeError("Drive down")

        monkeypatch.setattr(m, "_download_from_hf", fake_hf)
        monkeypatch.setattr(m, "_download_from_gdrive", fake_gdrive)

        with pytest.raises(FileNotFoundError):
            m._ensure_model("sa24")

    def test_hf_fails_no_gdrive_id_raises(self, isolated_dirs, monkeypatch):
        """A model without a gdrive_id raises FileNotFoundError directly."""
        _, _cache_dir = isolated_dirs
        # Register a temporary model entry lacking gdrive_id.
        monkeypatch.setitem(
            m.MODELS, "no_gd",
            {"filename": "no_gd.onnx", "input_size": (256, 256), "type": "heatmap"},
        )

        def fake_hf(fn, dest):
            raise RuntimeError("HF down")

        monkeypatch.setattr(m, "_download_from_hf", fake_hf)
        with pytest.raises(FileNotFoundError):
            m._ensure_model("no_gd")


# ---------------------------------------------------------------------------
# _download_from_hf / _download_from_gdrive  (mocked, no network)
# ---------------------------------------------------------------------------

class TestDownloadFromHf:
    def test_copies_hub_file_to_dest(self, tmp_path, monkeypatch):
        src = tmp_path / "src.onnx"
        src.write_bytes(b"weights")
        dest = tmp_path / "dest.onnx"

        fake_hub = types.ModuleType("huggingface_hub")
        recorded = {}

        def fake_download(repo_id, filename):
            recorded["repo_id"] = repo_id
            recorded["filename"] = filename
            return str(src)

        fake_hub.hf_hub_download = fake_download
        monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)

        m._download_from_hf("model.onnx", dest)
        assert dest.read_bytes() == b"weights"
        assert recorded["repo_id"] == m.HF_REPO_ID
        assert recorded["filename"] == "model.onnx"


class TestDownloadFromGdrive:
    def test_downloads_and_renames(self, tmp_path, monkeypatch):
        dest = tmp_path / "model.onnx"
        payload = b"gdrive-weights"

        class _FakeResp:
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def read(self, *a):
                # used by shutil.copyfileobj
                data, self._done = (payload, True) if not getattr(self, "_done", False) else (b"", True)
                return data

        def fake_urlopen(req, timeout=None):
            return _FakeResp()

        monkeypatch.setattr(m.urllib.request, "urlopen", fake_urlopen)
        m._download_from_gdrive("someid", dest)
        assert dest.read_bytes() == payload
        # tmp file cleaned up via rename
        assert not dest.with_suffix(".tmp").exists()

    def test_cleans_up_tmp_on_failure(self, tmp_path, monkeypatch):
        dest = tmp_path / "model.onnx"

        def fake_urlopen(req, timeout=None):
            raise RuntimeError("network error")

        monkeypatch.setattr(m.urllib.request, "urlopen", fake_urlopen)
        with pytest.raises(RuntimeError):
            m._download_from_gdrive("someid", dest)
        assert not dest.with_suffix(".tmp").exists()
        assert not dest.exists()


# ---------------------------------------------------------------------------
# _get_session
# ---------------------------------------------------------------------------

class TestGetSession:
    def _install_fake_ort(self, monkeypatch):
        """Install a fake onnxruntime module; returns a list recording paths."""
        created_paths = []

        fake_ort = types.ModuleType("onnxruntime")

        class _SessOpts:
            graph_optimization_level = None

        class _GraphOpt:
            ORT_ENABLE_ALL = "ALL"

        class _FakeInferenceSession:
            def __init__(self, path, sess_options=None, providers=None):
                created_paths.append(path)
                self.path = path
                self.providers = providers

        fake_ort.SessionOptions = _SessOpts
        fake_ort.GraphOptimizationLevel = _GraphOpt
        fake_ort.InferenceSession = _FakeInferenceSession
        monkeypatch.setitem(sys.modules, "onnxruntime", fake_ort)
        return created_paths

    def test_creates_and_caches_singleton(self, tmp_path, monkeypatch):
        dummy_path = tmp_path / "dummy.onnx"
        monkeypatch.setattr(m, "_ensure_model", lambda name: dummy_path)
        created_paths = self._install_fake_ort(monkeypatch)

        s1 = m._get_session("sa24")
        s2 = m._get_session("sa24")
        assert s1 is s2  # cached singleton
        assert created_paths == [str(dummy_path)]  # built only once
        assert m._sessions["sa24"] is s1

    def test_missing_onnxruntime_raises_importerror(self, tmp_path, monkeypatch):
        monkeypatch.setattr(m, "_ensure_model", lambda name: tmp_path / "x.onnx")
        # Force the `import onnxruntime` to fail.
        monkeypatch.setitem(sys.modules, "onnxruntime", None)
        with pytest.raises(ImportError):
            m._get_session("sa24")


# ---------------------------------------------------------------------------
# _run_heatmap_raw / _detect_heatmap
# ---------------------------------------------------------------------------

def _hot_heatmaps(hm_h=64, hm_w=64):
    heatmaps = np.zeros((1, 4, hm_h, hm_w), dtype=np.float32)
    peaks = [(0, 0), (0, hm_w - 1), (hm_h - 1, hm_w - 1), (hm_h - 1, 0)]
    for i, (py, px) in enumerate(peaks):
        heatmaps[0, i, py, px] = 1.0
    return heatmaps


class TestRunHeatmapRaw:
    def test_non_three_channel_returns_none(self):
        gray = np.zeros((50, 50), dtype=np.uint8)
        assert m._run_heatmap_raw(gray, "sa24") is None
        four = np.zeros((50, 50, 4), dtype=np.uint8)
        assert m._run_heatmap_raw(four, "sa24") is None

    def test_session_failure_returns_none(self, monkeypatch):
        def boom(name):
            raise RuntimeError("no session")

        monkeypatch.setattr(m, "_get_session", boom)
        img = np.zeros((50, 50, 3), dtype=np.uint8)
        assert m._run_heatmap_raw(img, "sa24") is None

    def test_returns_session_output(self, monkeypatch):
        heatmaps = _hot_heatmaps()
        session = _FakeSession(heatmaps)
        monkeypatch.setattr(m, "_get_session", lambda name: session)
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        out = m._run_heatmap_raw(img, "sa24")
        assert out is heatmaps
        # The tensor fed in is the preprocessed NCHW tensor.
        _, feed = session.run_calls[0]
        fed = next(iter(feed.values()))
        assert fed.shape == (1, 3, 256, 256)


class TestDetectHeatmap:
    def test_returns_four_corners(self, monkeypatch):
        heatmaps = _hot_heatmaps()
        session = _FakeSession(heatmaps)
        monkeypatch.setattr(m, "_get_session", lambda name: session)
        img = np.zeros((200, 100, 3), dtype=np.uint8)
        corners = m._detect_heatmap(img, "sa24")
        assert corners is not None
        assert corners.shape == (4, 2)
        assert corners.dtype == np.float32

    def test_none_when_raw_none(self, monkeypatch):
        monkeypatch.setattr(m, "_run_heatmap_raw", lambda img, name: None)
        img = np.zeros((50, 50, 3), dtype=np.uint8)
        assert m._detect_heatmap(img, "sa24") is None


# ---------------------------------------------------------------------------
# _detect_segmentation
# ---------------------------------------------------------------------------

def _rect_logits(in_h=384, in_w=384):
    """(1,2,H,W) logits whose argmax is 1 inside a centered rectangle."""
    logits = np.zeros((1, 2, in_h, in_w), dtype=np.float32)
    # background channel slightly positive everywhere
    logits[0, 0] = 0.1
    y0, y1 = in_h // 5, in_h * 4 // 5
    x0, x1 = in_w // 5, in_w * 4 // 5
    logits[0, 1, y0:y1, x0:x1] = 1.0  # foreground wins in the rectangle
    return logits


class TestDetectSegmentation:
    def test_non_three_channel_returns_none(self):
        gray = np.zeros((50, 50), dtype=np.uint8)
        assert m._detect_segmentation(gray, "deeplabv3_mbv3") is None

    def test_two_channel_argmax_rect(self, monkeypatch):
        logits = _rect_logits()
        session = _FakeSession(logits)
        monkeypatch.setattr(m, "_get_session", lambda name: session)
        img = np.zeros((768, 384, 3), dtype=np.uint8)
        corners = m._detect_segmentation(img, "deeplabv3_mbv3")
        assert corners is not None
        assert corners.shape == (4, 2)
        assert corners.dtype == np.float32
        # Corners scaled to original coords (orig 768x384, input 384x384):
        #   scale_x = 384/384 = 1, scale_y = 768/384 = 2
        # Rectangle spans roughly x in [76, 307], y*2 in [153, 614].
        assert corners[:, 0].max() <= 384 + 1
        assert corners[:, 1].max() <= 768 + 2
        assert corners[:, 1].max() > corners[:, 0].max()  # y scaled by 2

    def test_three_dim_threshold_branch(self, monkeypatch):
        in_h = in_w = 384
        pred = np.zeros((1, in_h, in_w), dtype=np.float32)
        y0, y1 = in_h // 5, in_h * 4 // 5
        x0, x1 = in_w // 5, in_w * 4 // 5
        pred[0, y0:y1, x0:x1] = 1.0  # > 0.5 inside rectangle
        session = _FakeSession(pred)
        monkeypatch.setattr(m, "_get_session", lambda name: session)
        img = np.zeros((384, 384, 3), dtype=np.uint8)
        corners = m._detect_segmentation(img, "deeplabv3_mbv3")
        assert corners is not None
        assert corners.shape == (4, 2)

    def test_unexpected_pred_shape_returns_none(self, monkeypatch):
        # 4D but channel dim != 2 -> neither branch matches -> None
        pred = np.zeros((1, 5, 384, 384), dtype=np.float32)
        session = _FakeSession(pred)
        monkeypatch.setattr(m, "_get_session", lambda name: session)
        img = np.zeros((384, 384, 3), dtype=np.uint8)
        assert m._detect_segmentation(img, "deeplabv3_mbv3") is None

    def test_empty_mask_returns_none(self, monkeypatch):
        # all-background -> no contours
        logits = np.zeros((1, 2, 384, 384), dtype=np.float32)
        logits[0, 0] = 1.0  # background everywhere
        session = _FakeSession(logits)
        monkeypatch.setattr(m, "_get_session", lambda name: session)
        img = np.zeros((384, 384, 3), dtype=np.uint8)
        assert m._detect_segmentation(img, "deeplabv3_mbv3") is None

    def test_tiny_contour_below_area_threshold_returns_none(self, monkeypatch):
        """A contour smaller than 2% of mask area is rejected (line 294)."""
        import cv2
        in_h = in_w = 384
        logits = np.zeros((1, 2, in_h, in_w), dtype=np.float32)
        logits[0, 0] = 0.1
        # Tiny blob (~ a few px) — survives morphology but is < 2% of area.
        fg = np.zeros((in_h, in_w), dtype=np.uint8)
        cv2.rectangle(fg, (10, 10), (24, 24), 255, -1)
        logits[0, 1][fg > 0] = 1.0
        session = _FakeSession(logits)
        monkeypatch.setattr(m, "_get_session", lambda name: session)
        img = np.zeros((384, 384, 3), dtype=np.uint8)
        assert m._detect_segmentation(img, "deeplabv3_mbv3") is None

    def test_minarearect_fallback(self, monkeypatch):
        """A non-quadrilateral mask falls through to minAreaRect (lines 310-314).

        An L-shaped polygon never reduces to exactly 4 vertices across the
        epsilon sweep, so approxPolyDP fails and minAreaRect is used.
        """
        import cv2
        in_h = in_w = 384
        logits = np.zeros((1, 2, in_h, in_w), dtype=np.float32)
        logits[0, 0] = 0.1
        fg = np.zeros((in_h, in_w), dtype=np.uint8)
        # L-shape (6 distinct corners) covering well over 2% of the area.
        l_poly = np.array([
            [40, 40], [300, 40], [300, 140], [160, 140],
            [160, 320], [40, 320],
        ], dtype=np.int32)
        cv2.fillPoly(fg, [l_poly], 255)
        logits[0, 1][fg > 0] = 1.0
        session = _FakeSession(logits)
        monkeypatch.setattr(m, "_get_session", lambda name: session)
        img = np.zeros((384, 384, 3), dtype=np.uint8)
        corners = m._detect_segmentation(img, "deeplabv3_mbv3")
        assert corners is not None
        assert corners.shape == (4, 2)  # minAreaRect always yields 4 points


# ---------------------------------------------------------------------------
# detect_corners_onnx (public API fallback chain)
# ---------------------------------------------------------------------------

class TestDetectCornersOnnx:
    def test_sa24_primary_path(self, monkeypatch):
        sa24_corners = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32)
        calls = []

        def fake_detect(image, name):
            calls.append(name)
            return sa24_corners if name == "sa24" else None

        monkeypatch.setattr(m, "_detect_heatmap", fake_detect)
        img = np.zeros((50, 50, 3), dtype=np.uint8)
        out = m.detect_corners_onnx(img)
        assert np.array_equal(out, sa24_corners)
        assert calls == ["sa24"]  # lcnet100 never reached

    def test_lcnet100_fallback(self, monkeypatch):
        lc_corners = np.array([[2, 2], [3, 2], [3, 3], [2, 3]], dtype=np.float32)
        calls = []

        def fake_detect(image, name):
            calls.append(name)
            return None if name == "sa24" else lc_corners

        monkeypatch.setattr(m, "_detect_heatmap", fake_detect)
        img = np.zeros((50, 50, 3), dtype=np.uint8)
        out = m.detect_corners_onnx(img)
        assert np.array_equal(out, lc_corners)
        assert calls == ["sa24", "lcnet100"]

    def test_both_fail_returns_none(self, monkeypatch):
        monkeypatch.setattr(m, "_detect_heatmap", lambda image, name: None)
        img = np.zeros((50, 50, 3), dtype=np.uint8)
        assert m.detect_corners_onnx(img) is None


# ---------------------------------------------------------------------------
# detect_corners_segmentation (public API thin wrapper)
# ---------------------------------------------------------------------------

class TestDetectCornersSegmentation:
    def test_delegates_to_detect_segmentation(self, monkeypatch):
        sentinel = np.zeros((4, 2), dtype=np.float32)
        recorded = {}

        def fake_seg(image, name):
            recorded["name"] = name
            return sentinel

        monkeypatch.setattr(m, "_detect_segmentation", fake_seg)
        img = np.zeros((50, 50, 3), dtype=np.uint8)
        out = m.detect_corners_segmentation(img)
        assert out is sentinel
        assert recorded["name"] == "deeplabv3_mbv3"
