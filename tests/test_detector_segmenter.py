"""Hermetic unit tests for pagescan.detector and pagescan.segmenter.

All ONNX-runtime, torch/SAM and Hugging Face downloads are mocked. No network
access and no real model weights are touched.
"""
from __future__ import annotations

import sys
import types

import cv2
import numpy as np
import pytest

from pagescan import detector, segmenter


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture(autouse=True)
def _reset_singletons():
    """Reset module-level singletons so tests never leak a session/predictor."""
    detector._session = None
    segmenter._predictor = None
    yield
    detector._session = None
    segmenter._predictor = None


@pytest.fixture
def isolated_cache(tmp_path, monkeypatch):
    """Point PAGESCAN_CACHE and the project data dir at empty tmp locations."""
    cache = tmp_path / "cache"
    data_model = tmp_path / "data_model"
    cache.mkdir()
    data_model.mkdir()
    monkeypatch.setenv("PAGESCAN_CACHE", str(cache))
    monkeypatch.setattr(detector, "_DATA_MODEL_DIR", data_model)
    monkeypatch.setattr(segmenter, "_DATA_MODEL_DIR", data_model)
    return cache, data_model


def _install_fake_hf(monkeypatch, returns_path):
    """Inject a fake huggingface_hub module whose hf_hub_download is recorded."""
    calls = []

    def fake_download(repo_id, filename, **kwargs):
        calls.append({"repo_id": repo_id, "filename": filename, **kwargs})
        return str(returns_path)

    fake_mod = types.ModuleType("huggingface_hub")
    fake_mod.hf_hub_download = fake_download
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_mod)
    return calls


# =========================================================================== #
# detector._letterbox  (PURE)
# =========================================================================== #
def test_letterbox_canvas_shape_dtype_and_ratio():
    h, w = 600, 1000
    img = np.zeros((h, w, 3), dtype=np.uint8)
    canvas, ratio, (pad_x, pad_y) = detector._letterbox(img)

    assert canvas.shape == (detector.INPUT_SIZE, detector.INPUT_SIZE, 3)
    assert canvas.dtype == np.uint8
    assert ratio == pytest.approx(detector.INPUT_SIZE / max(h, w))


def test_letterbox_padding_centres_and_gray_fill():
    h, w = 600, 1000
    # Distinctive non-gray content so we can locate the resized region.
    img = np.full((h, w, 3), 200, dtype=np.uint8)
    canvas, ratio, (pad_x, pad_y) = detector._letterbox(img)

    new_h, new_w = round(h * ratio), round(w * ratio)
    # Width is the long side -> fills the canvas width, no horizontal padding.
    assert pad_x == (detector.INPUT_SIZE - new_w) // 2
    assert pad_y == (detector.INPUT_SIZE - new_h) // 2
    assert pad_x == 0
    assert pad_y > 0

    # Pad region (top strip) must be the 114 gray fill.
    assert np.all(canvas[:pad_y] == 114)
    assert np.all(canvas[pad_y + new_h:] == 114)
    # Content region carries the resized image (value 200).
    assert np.all(canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] == 200)


def test_letterbox_point_roundtrips():
    """A point in the original maps through ratio+pad into the canvas region."""
    h, w = 600, 1000
    img = np.zeros((h, w, 3), dtype=np.uint8)
    canvas, ratio, (pad_x, pad_y) = detector._letterbox(img)

    # Original centre.
    ox, oy = w / 2.0, h / 2.0
    # Forward map: original -> letterboxed canvas coords.
    cx = ox * ratio + pad_x
    cy = oy * ratio + pad_y
    # The canvas centre column lands at INPUT_SIZE/2 (width is the long side).
    assert cx == pytest.approx(detector.INPUT_SIZE / 2.0, abs=1.0)
    # And lies inside the content band.
    new_h = round(h * ratio)
    assert pad_y <= cy <= pad_y + new_h

    # Inverse map (the formula detect() uses) returns the original point.
    back_x = (cx - pad_x) / ratio
    back_y = (cy - pad_y) / ratio
    assert back_x == pytest.approx(ox, abs=1.0)
    assert back_y == pytest.approx(oy, abs=1.0)


# =========================================================================== #
# detector.detect
# =========================================================================== #
class _FakeInput:
    def __init__(self, name):
        self.name = name


class _FakeSession:
    """Mimics an onnxruntime InferenceSession for a single-class YOLO head."""

    def __init__(self, output):
        self._output = output
        self.last_inputs = None

    def get_inputs(self):
        return [_FakeInput("images")]

    def run(self, output_names, inputs):
        self.last_inputs = inputs
        return [self._output]


def _make_output_5xN(boxes, scores):
    """Build a (1, 5, N) output: rows cx, cy, w, h, score."""
    n = len(scores)
    arr = np.zeros((5, n), dtype=np.float32)
    for i, (cx, cy, bw, bh) in enumerate(boxes):
        arr[0, i] = cx
        arr[1, i] = cy
        arr[2, i] = bw
        arr[3, i] = bh
    arr[4, :] = scores
    return arr[None]  # (1, 5, N)


def _make_output_Nx5(boxes, scores):
    """Build a (1, N, 5) output: cols cx, cy, w, h, score."""
    n = len(scores)
    arr = np.zeros((n, 5), dtype=np.float32)
    for i, (cx, cy, bw, bh) in enumerate(boxes):
        arr[i, :4] = (cx, cy, bw, bh)
    arr[:, 4] = scores
    return arr[None]  # (1, N, 5)


@pytest.mark.parametrize("layout", ["5xN", "Nx5"])
def test_detect_picks_argmax_and_maps_back(monkeypatch, layout):
    # Original image: 1000 wide x 600 tall.
    h0, w0 = 600, 1000
    image = np.zeros((h0, w0, 3), dtype=np.uint8)

    # Recompute the letterbox transform that detect() will apply.
    ratio = detector.INPUT_SIZE / max(h0, w0)
    new_h, new_w = round(h0 * ratio), round(w0 * ratio)
    pad_x = (detector.INPUT_SIZE - new_w) // 2
    pad_y = (detector.INPUT_SIZE - new_h) // 2

    # Target box in ORIGINAL coords we want to recover: x1,y1,x2,y2.
    tx1, ty1, tx2, ty2 = 100.0, 80.0, 700.0, 500.0
    # Forward into letterboxed cxcywh.
    lx1 = tx1 * ratio + pad_x
    ly1 = ty1 * ratio + pad_y
    lx2 = tx2 * ratio + pad_x
    ly2 = ty2 * ratio + pad_y
    cx = (lx1 + lx2) / 2.0
    cy = (ly1 + ly2) / 2.0
    bw = lx2 - lx1
    bh = ly2 - ly1

    boxes = [
        (10.0, 10.0, 5.0, 5.0),     # below threshold
        (cx, cy, bw, bh),           # winner
        (20.0, 20.0, 8.0, 8.0),     # below threshold
    ]
    scores = [0.10, 0.91, 0.05]

    builder = _make_output_5xN if layout == "5xN" else _make_output_Nx5
    fake = _FakeSession(builder(boxes, scores))
    monkeypatch.setattr(detector, "_get_session", lambda: fake)

    result = detector.detect(image, conf_threshold=0.25)
    assert result is not None
    bbox, conf = result

    assert conf == pytest.approx(0.91, abs=1e-5)
    assert bbox.dtype == np.float32
    assert bbox.shape == (4,)
    np.testing.assert_allclose(bbox, [tx1, ty1, tx2, ty2], atol=1.0)

    # Confirm the fake actually received the letterboxed tensor under input name.
    assert "images" in fake.last_inputs
    assert fake.last_inputs["images"].shape == (1, 3, detector.INPUT_SIZE, detector.INPUT_SIZE)


def test_detect_clamps_to_image_bounds(monkeypatch):
    h0, w0 = 400, 400
    image = np.zeros((h0, w0, 3), dtype=np.uint8)

    # Box that extends far beyond canvas -> must clamp to [0, dim-1].
    big = float(detector.INPUT_SIZE * 4)
    boxes = [(detector.INPUT_SIZE / 2, detector.INPUT_SIZE / 2, big, big)]
    fake = _FakeSession(_make_output_5xN(boxes, [0.99]))
    monkeypatch.setattr(detector, "_get_session", lambda: fake)

    bbox, conf = detector.detect(image, conf_threshold=0.25)
    assert bbox[0] >= 0.0 and bbox[1] >= 0.0
    assert bbox[2] <= w0 - 1 and bbox[3] <= h0 - 1


def test_detect_returns_none_below_threshold(monkeypatch):
    image = np.zeros((400, 400, 3), dtype=np.uint8)
    boxes = [(100, 100, 50, 50), (200, 200, 30, 30)]
    fake = _FakeSession(_make_output_5xN(boxes, [0.10, 0.20]))
    monkeypatch.setattr(detector, "_get_session", lambda: fake)

    assert detector.detect(image, conf_threshold=0.25) is None


# =========================================================================== #
# detector._ensure_model
# =========================================================================== #
def _write_sized_file(path, size):
    with open(path, "wb") as f:
        f.truncate(size)


def test_detector_ensure_model_local_branch(isolated_cache):
    _, data_model = isolated_cache
    local = data_model / detector.DEFAULT_MODEL_NAME
    _write_sized_file(local, 100_001)  # > 100_000 gate

    assert detector._ensure_model() == local


def test_detector_ensure_model_cache_branch(isolated_cache):
    cache, _ = isolated_cache
    cache_file = cache / detector.DEFAULT_MODEL_NAME
    _write_sized_file(cache_file, 100_001)

    assert detector._ensure_model() == cache_file


def test_detector_ensure_model_too_small_falls_through_to_hf(isolated_cache, monkeypatch, tmp_path):
    _, data_model = isolated_cache
    # Present but below the 100_000 size gate -> ignored.
    _write_sized_file(data_model / detector.DEFAULT_MODEL_NAME, 50_000)

    downloaded = tmp_path / "downloaded_yolo.onnx"
    downloaded.write_bytes(b"x")
    calls = _install_fake_hf(monkeypatch, downloaded)

    result = detector._ensure_model()
    assert result == downloaded
    assert calls == [{"repo_id": detector.HF_REPO_ID,
                      "filename": detector.DEFAULT_MODEL_NAME}]


def test_detector_ensure_model_hf_branch(isolated_cache, monkeypatch, tmp_path):
    downloaded = tmp_path / "hf_yolo.onnx"
    downloaded.write_bytes(b"weights")
    calls = _install_fake_hf(monkeypatch, downloaded)

    assert detector._ensure_model() == downloaded
    assert len(calls) == 1


# =========================================================================== #
# segmenter.mask_to_quad  (PURE)
# =========================================================================== #
def test_mask_to_quad_rectangle_corners():
    mask = np.zeros((300, 400), dtype=bool)
    mask[50:250, 80:320] = True  # rectangle: x in [80,319], y in [50,249]

    quad = segmenter.mask_to_quad(mask)
    assert quad is not None
    assert quad.shape == (4, 2)
    assert quad.dtype == np.float32

    expected = {(80, 50), (319, 50), (319, 249), (80, 249)}
    for px, py in quad:
        # Each returned corner is near one of the rectangle corners.
        assert min(abs(px - ex) + abs(py - ey) for ex, ey in expected) <= 5


def test_mask_to_quad_complex_blob_minarearect_fallback():
    # An L-shaped blob: its convex hull won't reduce to exactly 4 vertices
    # across the epsilon sweep, forcing the minAreaRect fallback.
    mask = np.zeros((400, 400), dtype=bool)
    mask[50:350, 50:150] = True   # vertical arm
    mask[250:350, 50:350] = True  # horizontal arm

    quad = segmenter.mask_to_quad(mask)
    assert quad is not None
    assert quad.shape == (4, 2)
    assert quad.dtype == np.float32


def test_mask_to_quad_empty_mask_returns_none():
    mask = np.zeros((100, 100), dtype=bool)
    assert segmenter.mask_to_quad(mask) is None


def test_mask_to_quad_tiny_below_min_area_returns_none():
    mask = np.zeros((100, 100), dtype=bool)
    mask[10:15, 10:15] = True  # area 25 << default min_area 1000
    assert segmenter.mask_to_quad(mask) is None


# =========================================================================== #
# segmenter.segment
# =========================================================================== #
class _FakePredictor:
    def __init__(self, predict_return):
        self._predict_return = predict_return
        self.set_image_called_with = None
        self.predict_kwargs = None

    def set_image(self, rgb):
        self.set_image_called_with = rgb

    def predict(self, **kwargs):
        self.predict_kwargs = kwargs
        return self._predict_return


def test_segment_returns_bool_mask(monkeypatch):
    h, w = 120, 160
    raw = np.zeros((1, h, w), dtype=np.float32)
    raw[0, 30:90, 40:120] = 1.0  # foreground region
    scores = np.array([0.97], dtype=np.float32)
    low_res = np.zeros((1, 256, 256), dtype=np.float32)

    fake = _FakePredictor((raw, scores, low_res))
    monkeypatch.setattr(segmenter, "_get_predictor", lambda: fake)

    image = np.zeros((h, w, 3), dtype=np.uint8)
    bbox = np.array([40, 30, 120, 90], dtype=np.float32)

    mask = segmenter.segment(image, bbox)
    assert mask is not None
    assert mask.dtype == bool
    assert mask.shape == (h, w)
    assert mask[50, 50] == True
    assert mask[0, 0] == False

    # Predictor was driven with an RGB image and the box prompt.
    assert fake.set_image_called_with is not None
    assert fake.set_image_called_with.shape == (h, w, 3)
    assert fake.predict_kwargs["multimask_output"] is False
    np.testing.assert_array_equal(
        fake.predict_kwargs["box"], bbox.astype(np.float32))


def test_segment_empty_masks_returns_none(monkeypatch):
    empty = np.zeros((0, 10, 10), dtype=np.float32)
    fake = _FakePredictor((empty, np.zeros((0,)), None))
    monkeypatch.setattr(segmenter, "_get_predictor", lambda: fake)

    image = np.zeros((10, 10, 3), dtype=np.uint8)
    bbox = np.array([0, 0, 5, 5], dtype=np.float32)
    assert segmenter.segment(image, bbox) is None


def test_segment_none_masks_returns_none(monkeypatch):
    fake = _FakePredictor((None, None, None))
    monkeypatch.setattr(segmenter, "_get_predictor", lambda: fake)

    image = np.zeros((10, 10, 3), dtype=np.uint8)
    bbox = np.array([0, 0, 5, 5], dtype=np.float32)
    assert segmenter.segment(image, bbox) is None


def test_segment_missing_torch_raises(monkeypatch):
    # Force the function-local `import torch` to fail.
    monkeypatch.setitem(sys.modules, "torch", None)
    image = np.zeros((10, 10, 3), dtype=np.uint8)
    bbox = np.array([0, 0, 5, 5], dtype=np.float32)
    with pytest.raises(ImportError, match="torch is required"):
        segmenter.segment(image, bbox)


# =========================================================================== #
# segmenter._ensure_model  (>100 MB gate)
# =========================================================================== #
_SAM_GATE = 100_000_000


def test_segmenter_ensure_model_local_branch(isolated_cache):
    _, data_model = isolated_cache
    local = data_model / segmenter.DEFAULT_MODEL_NAME
    _write_sized_file(local, _SAM_GATE + 1)  # sparse file > 100 MB

    assert segmenter._ensure_model() == local


def test_segmenter_ensure_model_cache_branch(isolated_cache):
    cache, _ = isolated_cache
    cache_file = cache / segmenter.DEFAULT_MODEL_NAME
    _write_sized_file(cache_file, _SAM_GATE + 1)

    assert segmenter._ensure_model() == cache_file


def test_segmenter_ensure_model_too_small_falls_through_to_hf(isolated_cache, monkeypatch, tmp_path):
    _, data_model = isolated_cache
    # Present but below the 100 MB gate -> ignored.
    _write_sized_file(data_model / segmenter.DEFAULT_MODEL_NAME, 50_000)

    downloaded = tmp_path / "downloaded_sam.pth"
    downloaded.write_bytes(b"x")
    calls = _install_fake_hf(monkeypatch, downloaded)

    result = segmenter._ensure_model()
    assert result == downloaded
    assert calls == [{"repo_id": segmenter.HF_REPO_ID,
                      "filename": segmenter.DEFAULT_MODEL_NAME}]


def test_segmenter_ensure_model_hf_branch(isolated_cache, monkeypatch, tmp_path):
    downloaded = tmp_path / "hf_sam.pth"
    downloaded.write_bytes(b"weights")
    calls = _install_fake_hf(monkeypatch, downloaded)

    assert segmenter._ensure_model() == downloaded
    assert len(calls) == 1


# =========================================================================== #
# Cache-dir helpers
# =========================================================================== #
def test_get_cache_dir_uses_env(tmp_path, monkeypatch):
    target = tmp_path / "mycache"
    monkeypatch.setenv("PAGESCAN_CACHE", str(target))
    assert detector._get_cache_dir() == target
    assert target.is_dir()
    assert segmenter._get_cache_dir() == target


# =========================================================================== #
# detector._get_session  (onnxruntime fully mocked)
# =========================================================================== #
class _FakeOrtSession:
    def __init__(self, model_path, sess_options=None, providers=None):
        self.model_path = model_path
        self.sess_options = sess_options
        self.providers = providers


def _install_fake_onnxruntime(monkeypatch):
    mod = types.ModuleType("onnxruntime")

    class _Opts:
        def __init__(self):
            self.graph_optimization_level = None

    class _Level:
        ORT_ENABLE_ALL = "all"

    mod.SessionOptions = _Opts
    mod.GraphOptimizationLevel = _Level
    mod.InferenceSession = _FakeOrtSession
    monkeypatch.setitem(sys.modules, "onnxruntime", mod)
    return mod


def test_get_session_builds_and_caches(monkeypatch, tmp_path):
    _install_fake_onnxruntime(monkeypatch)
    model = tmp_path / "yolo.onnx"
    model.write_bytes(b"x")
    monkeypatch.setattr(detector, "_ensure_model", lambda: model)

    sess = detector._get_session()
    assert isinstance(sess, _FakeOrtSession)
    assert sess.model_path == str(model)
    assert sess.providers == ["CPUExecutionProvider"]
    assert sess.sess_options.graph_optimization_level == "all"

    # Second call returns the cached singleton without rebuilding.
    monkeypatch.setattr(detector, "_ensure_model",
                        lambda: (_ for _ in ()).throw(AssertionError("rebuilt")))
    assert detector._get_session() is sess


def test_get_session_missing_onnxruntime_raises(monkeypatch):
    # Force the lazy `import onnxruntime` to fail.
    monkeypatch.setitem(sys.modules, "onnxruntime", None)
    with pytest.raises(ImportError, match="onnxruntime is required"):
        detector._get_session()


# =========================================================================== #
# segmenter._get_predictor  (torch + segment_anything_hq fully mocked)
# =========================================================================== #
def _install_fake_sam_stack(monkeypatch, tmp_path):
    captured = {}

    # Fake segment_anything_hq.
    sam_mod = types.ModuleType("segment_anything_hq")

    class _FakeSam:
        def __init__(self):
            self.device = None
            self.eval_called = False
            self.loaded_state = None

        def load_state_dict(self, state):
            self.loaded_state = state

        def to(self, device):
            self.device = device
            return self

        def eval(self):
            self.eval_called = True
            return self

    class _SamPredictor:
        def __init__(self, sam):
            self.sam = sam
            captured["predictor_sam"] = sam

    def _builder(checkpoint=None):
        captured["checkpoint"] = checkpoint
        return _FakeSam()

    sam_mod.SamPredictor = _SamPredictor
    sam_mod.sam_model_registry = {"vit_b": _builder}

    # Fake torch (only the bits _get_predictor touches).
    torch_mod = types.ModuleType("torch")

    class _Cuda:
        @staticmethod
        def is_available():
            return False

    torch_mod.cuda = _Cuda()

    def _load(path, map_location=None, weights_only=None):
        captured["load"] = {"path": path, "map_location": map_location,
                            "weights_only": weights_only}
        return {"fake": "state"}

    torch_mod.load = _load

    monkeypatch.setitem(sys.modules, "segment_anything_hq", sam_mod)
    monkeypatch.setitem(sys.modules, "torch", torch_mod)
    return captured


def test_get_predictor_builds_and_caches(monkeypatch, tmp_path):
    captured = _install_fake_sam_stack(monkeypatch, tmp_path)
    model = tmp_path / "sam.pth"
    model.write_bytes(b"x")
    monkeypatch.setattr(segmenter, "_ensure_model", lambda: model)

    pred = segmenter._get_predictor()
    assert pred is not None
    assert captured["checkpoint"] is None
    assert captured["load"]["weights_only"] is True
    assert captured["load"]["map_location"] == "cpu"  # cuda mocked unavailable
    sam = captured["predictor_sam"]
    assert sam.loaded_state == {"fake": "state"}
    assert sam.device == "cpu"
    assert sam.eval_called is True

    # Cached on second call.
    monkeypatch.setattr(segmenter, "_ensure_model",
                        lambda: (_ for _ in ()).throw(AssertionError("rebuilt")))
    assert segmenter._get_predictor() is pred


def test_get_predictor_missing_deps_raises(monkeypatch):
    monkeypatch.setitem(sys.modules, "segment_anything_hq", None)
    with pytest.raises(ImportError, match="segment-anything-hq"):
        segmenter._get_predictor()
