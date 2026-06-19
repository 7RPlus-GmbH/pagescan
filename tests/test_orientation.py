"""Hermetic tests for pagescan.orientation.

All ONNX / Tesseract / network dependencies are mocked. No real model is
downloaded and no real OCR is run. Tests use monkeypatch and sys.modules
injection so they pass regardless of whether onnxruntime/pytesseract are
actually installed.
"""
from __future__ import annotations

import sys
import types

import cv2
import numpy as np
import pytest

from pagescan import orientation

# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _reset_module_globals():
    """Isolate module-level singletons / probe caches across tests."""
    orig_session = orientation._ori_session
    orig_avail = orientation._available_ocr_langs
    orig_probed = orientation._ocr_langs_probed
    orientation._ori_session = None
    orientation._available_ocr_langs = None
    orientation._ocr_langs_probed = False
    yield
    orientation._ori_session = orig_session
    orientation._available_ocr_langs = orig_avail
    orientation._ocr_langs_probed = orig_probed


def _nonsquare_bgr(h=120, w=80):
    """A non-square BGR image so rot90 changes shape (detectable)."""
    return np.full((h, w, 3), 200, dtype=np.uint8)


# --------------------------------------------------------------------------
# _get_cache_dir
# --------------------------------------------------------------------------

class TestGetCacheDir:
    def test_uses_env_override(self, monkeypatch, tmp_path):
        target = tmp_path / "mycache"
        monkeypatch.setenv("PAGESCAN_CACHE", str(target))
        result = orientation._get_cache_dir()
        assert result == target
        assert result.is_dir()  # created

    def test_default_path_created(self, monkeypatch, tmp_path):
        monkeypatch.delenv("PAGESCAN_CACHE", raising=False)
        fake_home = tmp_path / "home"
        monkeypatch.setattr(orientation.Path, "home", staticmethod(lambda: fake_home))
        result = orientation._get_cache_dir()
        assert result == fake_home / ".cache" / "pagescan"
        assert result.is_dir()


# --------------------------------------------------------------------------
# _ensure_orientation_model
# --------------------------------------------------------------------------

class TestEnsureOrientationModel:
    def test_returns_cached_when_present_and_large(self, monkeypatch, tmp_path):
        monkeypatch.setattr(orientation, "_get_cache_dir", lambda: tmp_path)
        model_path = tmp_path / orientation.ORI_MODEL_FILENAME
        model_path.write_bytes(b"x" * 200_000)  # > 100_000 threshold
        result = orientation._ensure_orientation_model()
        assert result == model_path

    def test_downloads_via_hf_hub(self, monkeypatch, tmp_path):
        monkeypatch.setattr(orientation, "_get_cache_dir", lambda: tmp_path)
        # No cached file yet -> go to download path.
        # Build a fake huggingface_hub module that returns a source file.
        src = tmp_path / "hf_source.onnx"
        src.write_bytes(b"y" * 150_000)

        fake_hf = types.ModuleType("huggingface_hub")

        def fake_dl(repo_id, filename):
            assert repo_id == orientation.ORI_MODEL_HF_REPO
            assert filename == orientation.ORI_MODEL_HF_FILE
            return str(src)

        fake_hf.hf_hub_download = fake_dl
        monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hf)

        result = orientation._ensure_orientation_model()
        assert result == tmp_path / orientation.ORI_MODEL_FILENAME
        assert result.read_bytes() == b"y" * 150_000

    def test_url_fallback_when_hf_missing(self, monkeypatch, tmp_path):
        monkeypatch.setattr(orientation, "_get_cache_dir", lambda: tmp_path)
        # Make `import huggingface_hub` raise ImportError.
        monkeypatch.setitem(sys.modules, "huggingface_hub", None)

        captured = {}

        class FakeResp:
            def __init__(self, data):
                self._data = data

            def read(self, n=-1):
                d = self._data
                self._data = b""
                return d

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        def fake_urlopen(req, timeout=None):
            captured["url"] = req.full_url
            return FakeResp(b"z" * 130_000)

        monkeypatch.setattr(orientation.urllib.request, "urlopen", fake_urlopen)

        result = orientation._ensure_orientation_model()
        assert result == tmp_path / orientation.ORI_MODEL_FILENAME
        assert result.read_bytes() == b"z" * 130_000
        assert orientation.ORI_MODEL_HF_REPO in captured["url"]

    def test_url_fallback_cleans_tmp_on_error(self, monkeypatch, tmp_path):
        monkeypatch.setattr(orientation, "_get_cache_dir", lambda: tmp_path)
        monkeypatch.setitem(sys.modules, "huggingface_hub", None)

        def boom(req, timeout=None):
            raise RuntimeError("network down")

        monkeypatch.setattr(orientation.urllib.request, "urlopen", boom)

        with pytest.raises(RuntimeError):
            orientation._ensure_orientation_model()
        # No leftover tmp / model files.
        assert not (tmp_path / orientation.ORI_MODEL_FILENAME).exists()
        assert not (tmp_path / orientation.ORI_MODEL_FILENAME).with_suffix(".tmp").exists()


# --------------------------------------------------------------------------
# _get_orientation_session
# --------------------------------------------------------------------------

class TestGetOrientationSession:
    def test_returns_existing_singleton(self, monkeypatch):
        sentinel = object()
        orientation._ori_session = sentinel
        assert orientation._get_orientation_session() is sentinel

    def test_returns_none_when_onnx_missing(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "onnxruntime", None)
        assert orientation._get_orientation_session() is None

    def test_returns_none_when_model_load_fails(self, monkeypatch):
        # Provide a fake onnxruntime so import succeeds...
        fake_ort = types.ModuleType("onnxruntime")
        fake_ort.SessionOptions = lambda: types.SimpleNamespace()
        fake_ort.GraphOptimizationLevel = types.SimpleNamespace(ORT_ENABLE_ALL=1)
        monkeypatch.setitem(sys.modules, "onnxruntime", fake_ort)
        # ...but make model acquisition fail.
        monkeypatch.setattr(orientation, "_ensure_orientation_model",
                            lambda: (_ for _ in ()).throw(RuntimeError("no model")))
        assert orientation._get_orientation_session() is None

    def test_builds_session(self, monkeypatch, tmp_path):
        created = {}

        class FakeSessionOptions:
            pass

        def fake_inference_session(path, sess_options=None, providers=None):
            created["path"] = path
            created["providers"] = providers
            return "FAKE_SESSION"

        fake_ort = types.ModuleType("onnxruntime")
        fake_ort.SessionOptions = FakeSessionOptions
        fake_ort.GraphOptimizationLevel = types.SimpleNamespace(ORT_ENABLE_ALL=99)
        fake_ort.InferenceSession = fake_inference_session
        monkeypatch.setitem(sys.modules, "onnxruntime", fake_ort)

        mp = tmp_path / orientation.ORI_MODEL_FILENAME
        mp.write_bytes(b"m" * 200_000)
        monkeypatch.setattr(orientation, "_ensure_orientation_model", lambda: mp)

        sess = orientation._get_orientation_session()
        assert sess == "FAKE_SESSION"
        assert created["path"] == str(mp)
        assert created["providers"] == ["CPUExecutionProvider"]
        # Cached as singleton.
        assert orientation._ori_session == "FAKE_SESSION"


# --------------------------------------------------------------------------
# classify_orientation
# --------------------------------------------------------------------------

class _FakeInput:
    name = "input"


class _FakeSession:
    """Returns crafted logits shaped like the real model output."""

    def __init__(self, logits):
        # Real code does session.run(...)[0][0] -> a 1-D vector of 4 logits.
        self._out = [np.array([logits], dtype=np.float32)]
        self.run_calls = []

    def get_inputs(self):
        return [_FakeInput()]

    def run(self, output_names, feed):
        self.run_calls.append(feed)
        return self._out


class TestClassifyOrientation:
    def test_no_session_returns_zero(self, monkeypatch):
        monkeypatch.setattr(orientation, "_get_orientation_session", lambda: None)
        angle, conf = orientation.classify_orientation(_nonsquare_bgr())
        assert angle == 0
        assert conf == 0.0

    @pytest.mark.parametrize("idx", [0, 1, 2, 3])
    def test_argmax_maps_to_class(self, monkeypatch, idx):
        logits = [0.0, 0.0, 0.0, 0.0]
        logits[idx] = 10.0  # clear winner
        sess = _FakeSession(logits)
        monkeypatch.setattr(orientation, "_get_orientation_session", lambda: sess)

        angle, conf = orientation.classify_orientation(_nonsquare_bgr())
        assert angle == orientation.ORI_CLASSES[idx]
        # softmax max in (0, 1]
        assert 0.0 < conf <= 1.0
        assert conf > 0.9  # dominant logit
        # confirm the model was actually fed a blob
        assert sess.run_calls and "input" in sess.run_calls[0]

    def test_handles_grayscale_input(self, monkeypatch):
        sess = _FakeSession([5.0, 0.0, 0.0, 0.0])
        monkeypatch.setattr(orientation, "_get_orientation_session", lambda: sess)
        gray = np.full((120, 80), 128, dtype=np.uint8)
        angle, conf = orientation.classify_orientation(gray)
        assert angle == 0
        assert 0.0 < conf <= 1.0

    def test_confidence_is_softmax_max(self, monkeypatch):
        logits = [1.0, 2.0, 3.0, 0.0]
        sess = _FakeSession(logits)
        monkeypatch.setattr(orientation, "_get_orientation_session", lambda: sess)
        angle, conf = orientation.classify_orientation(_nonsquare_bgr())
        # Expected softmax max
        arr = np.array(logits)
        probs = np.exp(arr - arr.max())
        probs /= probs.sum()
        assert angle == orientation.ORI_CLASSES[int(np.argmax(probs))]
        assert conf == pytest.approx(float(probs.max()), rel=1e-5)


# --------------------------------------------------------------------------
# deskew  (pure cv2, no mocks)
# --------------------------------------------------------------------------

def _make_tilted_lines_image(angle_deg, h=600, w=800):
    """White canvas with several long black lines tilted by angle_deg."""
    img = np.full((h, w, 3), 255, dtype=np.uint8)
    theta = np.radians(angle_deg)
    cx, cy = w // 2, h // 2
    length = w  # long lines spanning most of the width
    dx = (length / 2) * np.cos(theta)
    dy = (length / 2) * np.sin(theta)
    # Several parallel baselines spread across the center region.
    for offset in range(-150, 151, 40):
        oy = cy + offset
        p1 = (int(cx - dx), int(oy - dy))
        p2 = (int(cx + dx), int(oy + dy))
        cv2.line(img, p1, p2, (0, 0, 0), 2)
    return img


class TestDeskew:
    def test_detects_tilt_and_preserves_shape(self):
        induced = 5.0
        img = _make_tilted_lines_image(induced)
        out, angle = orientation.deskew(img)
        assert out.shape == img.shape
        # Magnitude within a few degrees of the induced tilt.
        assert abs(angle) == pytest.approx(induced, abs=2.0)
        # Sign: positive tilt -> positive detected angle.
        assert angle > 0

    def test_negative_tilt(self):
        induced = -4.0
        img = _make_tilted_lines_image(induced)
        out, angle = orientation.deskew(img)
        assert out.shape == img.shape
        assert angle < 0
        assert abs(angle) == pytest.approx(abs(induced), abs=2.0)

    def test_no_lines_returns_unchanged(self):
        blank = np.full((600, 800, 3), 255, dtype=np.uint8)
        out, angle = orientation.deskew(blank)
        assert angle == 0.0
        # Same object returned (no rotation applied).
        assert out is blank

    def test_grayscale_input_accepted(self):
        img = _make_tilted_lines_image(5.0)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        out, angle = orientation.deskew(gray)
        assert out.shape == gray.shape
        assert abs(angle) == pytest.approx(5.0, abs=2.5)

    def test_near_zero_tilt_returns_unchanged(self):
        # Perfectly horizontal lines -> median angle below 0.3 deg threshold.
        img = _make_tilted_lines_image(0.0)
        out, angle = orientation.deskew(img)
        assert angle == 0.0
        assert out is img

    def test_vertical_lines_skipped_and_too_few_kept(self, monkeypatch):
        # Drive the per-line filters deterministically by mocking Hough output:
        #  - one near-vertical segment  -> abs(x2-x1) < 10 -> `continue`
        #  - the rest steep (>25 deg)   -> filtered out by abs(angle) < 25
        # Net: fewer than 3 angles kept -> early `return image, 0.0`.
        img = _make_tilted_lines_image(5.0)
        fake_lines = np.array([
            [[100, 50, 105, 400]],   # near-vertical: dx=5 (<10) -> continue
            [[100, 100, 200, 200]],  # 45 deg -> dropped by abs(angle)<25
            [[100, 100, 150, 250]],  # ~71 deg -> dropped
            [[100, 100, 130, 260]],  # steep -> dropped
        ], dtype=np.int32)
        monkeypatch.setattr(orientation.cv2, "HoughLinesP",
                            lambda *a, **k: fake_lines)
        out, angle = orientation.deskew(img)
        assert angle == 0.0
        assert out is img


# --------------------------------------------------------------------------
# _effective_ocr_lang
# --------------------------------------------------------------------------

class TestEffectiveOcrLang:
    def test_intersects_with_installed(self, monkeypatch, caplog):
        orientation._available_ocr_langs = {"eng", "deu"}
        orientation._ocr_langs_probed = True
        import logging
        with caplog.at_level(logging.WARNING):
            result = orientation._effective_ocr_lang("eng+deu+fra+jpn")
        assert result == "eng+deu"
        # Dropped packs are logged.
        assert "fra" in caplog.text and "jpn" in caplog.text

    def test_none_installed_returns_empty(self):
        orientation._available_ocr_langs = {"eng", "deu"}
        orientation._ocr_langs_probed = True
        assert orientation._effective_ocr_lang("jpn") == ""

    def test_cannot_introspect_returns_input(self):
        orientation._available_ocr_langs = None
        orientation._ocr_langs_probed = True
        assert orientation._effective_ocr_lang("eng+deu+xyz") == "eng+deu+xyz"

    def test_probes_pytesseract_once(self, monkeypatch):
        # Unprobed state -> should call pytesseract.get_languages exactly once.
        orientation._available_ocr_langs = None
        orientation._ocr_langs_probed = False

        calls = {"n": 0}
        fake_pt = types.ModuleType("pytesseract")

        def get_languages(config=""):
            calls["n"] += 1
            return ["eng", "deu", "osd"]

        fake_pt.get_languages = get_languages
        monkeypatch.setitem(sys.modules, "pytesseract", fake_pt)

        assert orientation._effective_ocr_lang("eng+fra") == "eng"
        # Second call uses cache, no further probing.
        assert orientation._effective_ocr_lang("deu") == "deu"
        assert calls["n"] == 1

    def test_probe_exception_sets_none(self, monkeypatch):
        orientation._available_ocr_langs = None
        orientation._ocr_langs_probed = False
        fake_pt = types.ModuleType("pytesseract")

        def boom(config=""):
            raise RuntimeError("tesseract missing")

        fake_pt.get_languages = boom
        monkeypatch.setitem(sys.modules, "pytesseract", fake_pt)
        # Falls into except -> _available_ocr_langs = None -> returns input as-is.
        assert orientation._effective_ocr_lang("eng+deu") == "eng+deu"
        assert orientation._available_ocr_langs is None


# --------------------------------------------------------------------------
# _ocr_word_score
# --------------------------------------------------------------------------

def _install_fake_pytesseract(monkeypatch, data=None, raise_exc=None):
    fake_pt = types.ModuleType("pytesseract")
    fake_pt.Output = types.SimpleNamespace(DICT="dict")

    def image_to_data(img, lang=None, config=None, output_type=None):
        if raise_exc is not None:
            raise raise_exc
        return data

    fake_pt.image_to_data = image_to_data
    fake_pt.get_languages = lambda config="": ["eng", "deu", "fra", "spa", "ita", "jpn"]
    monkeypatch.setitem(sys.modules, "pytesseract", fake_pt)
    return fake_pt


class TestOcrWordScore:
    def test_counts_conf_above_50(self, monkeypatch):
        orientation._available_ocr_langs = {"eng"}
        orientation._ocr_langs_probed = True
        _install_fake_pytesseract(
            monkeypatch,
            data={"conf": ["95", "51", "50", "-1", "80", "12"]},
        )
        gray = np.zeros((50, 50), dtype=np.uint8)
        # conf > 50: 95, 51, 80 -> 3
        assert orientation._ocr_word_score(gray, "eng") == 3

    def test_empty_effective_lang_returns_zero(self, monkeypatch):
        orientation._available_ocr_langs = {"deu"}
        orientation._ocr_langs_probed = True
        _install_fake_pytesseract(monkeypatch, data={"conf": ["99"]})
        gray = np.zeros((50, 50), dtype=np.uint8)
        # "jpn" not installed -> eff_lang == "" -> short-circuit to 0,
        # never calls image_to_data.
        assert orientation._ocr_word_score(gray, "jpn") == 0

    def test_exception_path_returns_zero(self, monkeypatch):
        orientation._available_ocr_langs = {"eng"}
        orientation._ocr_langs_probed = True
        _install_fake_pytesseract(monkeypatch, raise_exc=RuntimeError("boom"))
        gray = np.zeros((50, 50), dtype=np.uint8)
        assert orientation._ocr_word_score(gray, "eng") == 0

    def test_import_error_returns_zero(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "pytesseract", None)
        gray = np.zeros((50, 50), dtype=np.uint8)
        assert orientation._ocr_word_score(gray, "eng") == 0


# --------------------------------------------------------------------------
# auto_rotate
# --------------------------------------------------------------------------

class TestAutoRotate:
    def test_high_conf_90_rotates(self, monkeypatch):
        img = _nonsquare_bgr(120, 80)  # h=120, w=80
        monkeypatch.setattr(orientation, "classify_orientation",
                            lambda im: (90, 0.95))
        out = orientation.auto_rotate(img)
        # k=1 (90 CCW) swaps the two leading axes.
        assert out.shape[:2] == (80, 120)
        np.testing.assert_array_equal(out, np.rot90(img, k=1))

    def test_high_conf_270_rotates(self, monkeypatch):
        img = _nonsquare_bgr(120, 80)
        monkeypatch.setattr(orientation, "classify_orientation",
                            lambda im: (-90, 0.9))
        out = orientation.auto_rotate(img)
        # k=3
        assert out.shape[:2] == (80, 120)
        np.testing.assert_array_equal(out, np.rot90(img, k=3))

    def test_high_conf_upright_unchanged(self, monkeypatch):
        img = _nonsquare_bgr()
        monkeypatch.setattr(orientation, "classify_orientation",
                            lambda im: (0, 0.92))
        out = orientation.auto_rotate(img)
        assert out is img

    def test_no_ocr_trusts_cnn(self, monkeypatch):
        img = _nonsquare_bgr(120, 80)
        # 180 with conf below 0.6 won't take the high-conf 90/270 branch;
        # force the no-OCR branch by removing pytesseract, with conf>=0.5.
        monkeypatch.setattr(orientation, "classify_orientation",
                            lambda im: (180, 0.55))
        monkeypatch.setitem(sys.modules, "pytesseract", None)
        out = orientation.auto_rotate(img)
        np.testing.assert_array_equal(out, np.rot90(img, k=2))

    def test_no_ocr_low_conf_unchanged(self, monkeypatch):
        img = _nonsquare_bgr(120, 80)
        monkeypatch.setattr(orientation, "classify_orientation",
                            lambda im: (180, 0.2))
        monkeypatch.setitem(sys.modules, "pytesseract", None)
        out = orientation.auto_rotate(img)
        assert out is img

    def test_cnn_180_ocr_agrees(self, monkeypatch):
        img = _nonsquare_bgr(120, 80)
        monkeypatch.setattr(orientation, "classify_orientation",
                            lambda im: (180, 0.8))
        # pytesseract import must succeed in auto_rotate's probe.
        monkeypatch.setitem(sys.modules, "pytesseract", types.ModuleType("pytesseract"))

        # First call (score_0) low, second (score_180) high.
        seq = iter([2, 10])
        monkeypatch.setattr(orientation, "_ocr_word_score",
                            lambda g, lang=None: next(seq))
        out = orientation.auto_rotate(img)
        np.testing.assert_array_equal(out, np.rot90(img, k=2))

    def test_cnn_180_ocr_disagrees(self, monkeypatch):
        img = _nonsquare_bgr(120, 80)
        monkeypatch.setattr(orientation, "classify_orientation",
                            lambda im: (180, 0.8))
        monkeypatch.setitem(sys.modules, "pytesseract", types.ModuleType("pytesseract"))
        # score_0 >= score_180 -> keep as-is.
        seq = iter([10, 2])
        monkeypatch.setattr(orientation, "_ocr_word_score",
                            lambda g, lang=None: next(seq))
        out = orientation.auto_rotate(img)
        assert out is img

    def test_cnn_180_ocr_inconclusive(self, monkeypatch):
        img = _nonsquare_bgr(120, 80)
        monkeypatch.setattr(orientation, "classify_orientation",
                            lambda im: (180, 0.8))
        monkeypatch.setitem(sys.modules, "pytesseract", types.ModuleType("pytesseract"))
        # score_180 > score_0 but not > score_0*1.2, and score_0 < score_180
        # -> "OCR inconclusive" else branch still rotates k=2.
        seq = iter([10, 11])  # 11 > 10 but 11 < 12; 10 < 11 -> else branch
        monkeypatch.setattr(orientation, "_ocr_word_score",
                            lambda g, lang=None: next(seq))
        out = orientation.auto_rotate(img)
        np.testing.assert_array_equal(out, np.rot90(img, k=2))

    def test_low_conf_full_ocr_cnn_agrees(self, monkeypatch):
        img = _nonsquare_bgr(120, 80)
        # Low CNN conf but a correction_angle so cnn_k != 0; CNN says 90 (k=1).
        monkeypatch.setattr(orientation, "classify_orientation",
                            lambda im: (90, 0.4))
        monkeypatch.setitem(sys.modules, "pytesseract", types.ModuleType("pytesseract"))
        # Full OCR over k=0,1,2,3: best is k=1 (agrees with CNN), >=3.
        scores = {0: 1, 1: 9, 2: 2, 3: 0}
        calls = {"n": 0}

        def fake_score(rotated, lang=None):
            k = calls["n"]
            calls["n"] += 1
            return scores[k]

        monkeypatch.setattr(orientation, "_ocr_word_score", fake_score)
        out = orientation.auto_rotate(img)
        np.testing.assert_array_equal(out, np.rot90(img, k=1))

    def test_low_conf_full_ocr_clear_winner(self, monkeypatch):
        img = _nonsquare_bgr(120, 80)
        # CNN says upright-ish but low conf; OCR clear winner at k=2,
        # disagreeing with cnn_k (0).
        monkeypatch.setattr(orientation, "classify_orientation",
                            lambda im: (0, 0.3))
        monkeypatch.setitem(sys.modules, "pytesseract", types.ModuleType("pytesseract"))
        scores = {0: 2, 1: 1, 2: 9, 3: 0}  # best_k=2, 9 > 2*1.2, >=3
        calls = {"n": 0}

        def fake_score(rotated, lang=None):
            k = calls["n"]
            calls["n"] += 1
            return scores[k]

        monkeypatch.setattr(orientation, "_ocr_word_score", fake_score)
        out = orientation.auto_rotate(img)
        np.testing.assert_array_equal(out, np.rot90(img, k=2))

    def test_low_conf_full_ocr_current_zero(self, monkeypatch):
        img = _nonsquare_bgr(120, 80)
        monkeypatch.setattr(orientation, "classify_orientation",
                            lambda im: (0, 0.3))
        monkeypatch.setitem(sys.modules, "pytesseract", types.ModuleType("pytesseract"))
        # current_score (k=0) == 0, best_k=1 with score 2 (>0). Triggers the
        # "current=0" branch. Note best_score < 3 so prior branches skip.
        scores = {0: 0, 1: 2, 2: 1, 3: 0}
        calls = {"n": 0}

        def fake_score(rotated, lang=None):
            k = calls["n"]
            calls["n"] += 1
            return scores[k]

        monkeypatch.setattr(orientation, "_ocr_word_score", fake_score)
        out = orientation.auto_rotate(img)
        np.testing.assert_array_equal(out, np.rot90(img, k=1))

    def test_low_conf_ocr_inconclusive_uses_cnn(self, monkeypatch):
        img = _nonsquare_bgr(120, 80)
        # cnn_k != 0 (90) and conf >= 0.35, OCR gives no useful signal.
        monkeypatch.setattr(orientation, "classify_orientation",
                            lambda im: (90, 0.4))
        monkeypatch.setitem(sys.modules, "pytesseract", types.ModuleType("pytesseract"))
        scores = {0: 0, 1: 0, 2: 0, 3: 0}  # all zero -> no OCR winner
        calls = {"n": 0}

        def fake_score(rotated, lang=None):
            k = calls["n"]
            calls["n"] += 1
            return scores[k]

        monkeypatch.setattr(orientation, "_ocr_word_score", fake_score)
        out = orientation.auto_rotate(img)
        # Falls through to CNN low-conf branch -> rotate k=1.
        np.testing.assert_array_equal(out, np.rot90(img, k=1))

    def test_low_conf_no_signal_unchanged(self, monkeypatch):
        img = _nonsquare_bgr(120, 80)
        # cnn correction 0 -> cnn_k == 0; OCR all zero -> nothing applies.
        monkeypatch.setattr(orientation, "classify_orientation",
                            lambda im: (0, 0.2))
        monkeypatch.setitem(sys.modules, "pytesseract", types.ModuleType("pytesseract"))
        monkeypatch.setattr(orientation, "_ocr_word_score", lambda g, lang=None: 0)
        out = orientation.auto_rotate(img)
        assert out is img
