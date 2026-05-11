"""Document orientation correction: deskew and auto-rotation."""

import logging
import os
import shutil
import urllib.request
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# -- CNN orientation model (docTR MobileNetV3-Small) --

ORI_MODEL_FILENAME = "mobilenet_v3_small_page_orientation.onnx"
ORI_MODEL_HF_REPO = "Felix92/onnxtr-mobilenet-v3-small-page-orientation"
ORI_MODEL_HF_FILE = "model.onnx"
ORI_MODEL_SIZE_INFER = (512, 512)
ORI_MODEL_MEAN = np.array([0.694, 0.695, 0.693], dtype=np.float32)
ORI_MODEL_STD = np.array([0.299, 0.296, 0.301], dtype=np.float32)
ORI_CLASSES = [0, -90, 180, 90]  # correction angles

_ori_session = None


def _get_cache_dir() -> Path:
    cache = Path(os.environ.get("PAGESCAN_CACHE",
                                Path.home() / ".cache" / "pagescan"))
    cache.mkdir(parents=True, exist_ok=True)
    return cache


def _ensure_orientation_model() -> Path:
    """Download orientation ONNX model from HuggingFace if not cached."""
    model_path = _get_cache_dir() / ORI_MODEL_FILENAME
    if model_path.exists() and model_path.stat().st_size > 100_000:
        return model_path

    # Try huggingface_hub first (handles caching, mirrors, etc.)
    try:
        from huggingface_hub import hf_hub_download
        hf_path = hf_hub_download(repo_id=ORI_MODEL_HF_REPO,
                                  filename=ORI_MODEL_HF_FILE)
        shutil.copy2(hf_path, model_path)
        logger.info(f"  Orientation model cached ({model_path.stat().st_size / 1e6:.1f} MB)")
        return model_path
    except ImportError:
        pass

    # Fallback: direct URL download
    url = (f"https://huggingface.co/{ORI_MODEL_HF_REPO}"
           f"/resolve/main/{ORI_MODEL_HF_FILE}")
    logger.info(f"  Downloading orientation model to {model_path} ...")
    tmp = model_path.with_suffix('.tmp')
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "pagescan"})
        with urllib.request.urlopen(req, timeout=120) as resp, open(tmp, 'wb') as f:
            shutil.copyfileobj(resp, f)
        tmp.rename(model_path)
        logger.info(f"  Orientation model downloaded ({model_path.stat().st_size / 1e6:.1f} MB)")
    except Exception:
        tmp.unlink(missing_ok=True)
        raise
    return model_path


def _get_orientation_session():
    """Get or create the orientation ONNX session (singleton)."""
    global _ori_session
    if _ori_session is not None:
        return _ori_session

    try:
        import onnxruntime as ort
    except ImportError:
        return None

    try:
        model_path = _ensure_orientation_model()
    except Exception as e:
        logger.warning(f"  Cannot load orientation model: {e}")
        return None

    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    _ori_session = ort.InferenceSession(
        str(model_path), sess_options=opts,
        providers=["CPUExecutionProvider"],
    )
    return _ori_session


def classify_orientation(image: np.ndarray) -> Tuple[int, float]:
    """Classify document orientation using CNN model.

    Returns (correction_angle, confidence) where correction_angle is one of
    [0, -90, 180, 90] indicating the rotation needed to make the document
    upright. Confidence is 0.0-1.0.

    Returns (0, 0.0) if the model is not available.
    """
    session = _get_orientation_session()
    if session is None:
        return 0, 0.0

    resized = cv2.resize(image, ORI_MODEL_SIZE_INFER)
    if resized.ndim == 2:
        resized = cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

    blob = rgb.astype(np.float32) / 255.0
    blob = (blob - ORI_MODEL_MEAN) / ORI_MODEL_STD
    blob = blob.transpose(2, 0, 1)[np.newaxis, ...].astype(np.float32)

    input_name = session.get_inputs()[0].name
    logits = session.run(None, {input_name: blob})[0][0]

    # Softmax
    probs = np.exp(logits - logits.max())
    probs /= probs.sum()

    idx = int(np.argmax(probs))
    return ORI_CLASSES[idx], float(probs[idx])


def deskew(image: np.ndarray) -> Tuple[np.ndarray, float]:
    """Correct text skew using Hough line detection.

    Detects near-horizontal lines (text baselines, rules, table borders)
    via probabilistic Hough transform on the center 60% of the image
    (avoids background edges). Uses median angle for robustness to outliers.

    Returns (corrected_image, detected_angle). Angle is 0.0 if no
    correction was needed.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    h, w = gray.shape

    # Center 60% to avoid background edges
    mh, mw = h // 5, w // 5
    center = gray[mh:h - mh, mw:w - mw]

    edges = cv2.Canny(center, 50, 150, apertureSize=3)

    # Bridge text into horizontal strokes
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 3))
    edges = cv2.dilate(edges, kernel, iterations=1)
    edges = cv2.erode(edges, kernel, iterations=1)

    lines = cv2.HoughLinesP(edges, 1, np.pi / 720, threshold=80,
                            minLineLength=min(center.shape[1] // 8, 200),
                            maxLineGap=20)
    if lines is None or len(lines) < 3:
        return image, 0.0

    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if abs(x2 - x1) < 10:
            continue
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        if abs(angle) < 25:
            angles.append(angle)

    if len(angles) < 3:
        return image, 0.0

    median_angle = float(np.median(angles))

    if abs(median_angle) < 0.3 or abs(median_angle) > 25:
        return image, 0.0

    logger.info(f"  Deskew: {median_angle:.1f} ({len(angles)} lines)")

    h, w = image.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), median_angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h),
                             flags=cv2.INTER_CUBIC,
                             borderValue=(255, 255, 255))
    return rotated, median_angle


def _ocr_word_score(gray_image: np.ndarray) -> int:
    """Count high-confidence OCR words as a proxy for correct orientation."""
    try:
        import pytesseract
        from pytesseract import Output
        data = pytesseract.image_to_data(
            gray_image, lang='deu', config='--psm 6',
            output_type=Output.DICT)
        return sum(1 for c in data['conf'] if int(c) > 50)
    except Exception:
        return 0


def auto_rotate(image: np.ndarray) -> np.ndarray:
    """Auto-rotate document to correct orientation.

    Strategy:
    - For 90/270 rotations: CNN is reliable (aspect ratio change is obvious).
      Trust CNN at conf >= 0.6.
    - For 180 rotations: CNN is often wrong (text looks similar upside-down).
      Always verify 180 with OCR — only apply if OCR agrees.
    - Low CNN confidence: fall back to OCR word scoring on all 4 orientations.

    The CNN model (docTR MobileNetV3-Small) is fast (~6ms) but has a known
    weakness for 180° detection. OCR verification adds ~2s but prevents
    wrong 180° flips.
    """
    k_map = {-90: 3, 180: 2, 90: 1}

    # -- Primary: CNN orientation classifier --
    correction_angle, cnn_conf = classify_orientation(image)
    cnn_k = k_map.get(correction_angle, 0)

    # For 90/270 rotations with high confidence, trust CNN directly
    if cnn_conf >= 0.6 and correction_angle in (-90, 90):
        logger.info(f"  Auto-rotating {cnn_k * 90} CCW (CNN conf={cnn_conf:.2f}, correction={correction_angle})")
        return np.rot90(image, k=cnn_k)

    if cnn_conf >= 0.6 and correction_angle == 0:
        logger.debug(f"  CNN orientation: upright (conf={cnn_conf:.2f})")
        return image

    # -- For 180 or low confidence: verify with OCR --
    try:
        import pytesseract
    except ImportError:
        # No OCR available — trust CNN if reasonable confidence
        if cnn_k != 0 and cnn_conf >= 0.5:
            logger.info(f"  Auto-rotating {cnn_k * 90} CCW (CNN conf={cnn_conf:.2f}, no OCR)")
            return np.rot90(image, k=cnn_k)
        return image

    h, w = image.shape[:2]
    ocr_scale = min(1.0, 1000 / max(h, w))
    small = cv2.resize(image, None, fx=ocr_scale, fy=ocr_scale) if ocr_scale < 1.0 else image
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY) if small.ndim == 3 else small

    if cnn_conf >= 0.6 and correction_angle == 180:
        # CNN says 180 — verify with OCR (only check 0 vs 180, skip 90/270)
        score_0 = _ocr_word_score(gray)
        score_180 = _ocr_word_score(np.rot90(gray, k=2))

        if score_180 > score_0 * 1.2 and score_180 >= 3:
            logger.info(f"  Auto-rotating 180 CCW (CNN+OCR agree: 180={score_180} vs 0={score_0})")
            return np.rot90(image, k=2)
        elif score_0 >= score_180:
            logger.info(f"  CNN said 180 but OCR disagrees (0={score_0} vs 180={score_180}), keeping as-is")
            return image
        else:
            # OCR inconclusive but CNN has reasonable confidence
            logger.info(f"  Auto-rotating 180 CCW (CNN conf={cnn_conf:.2f}, OCR inconclusive: 0={score_0} 180={score_180})")
            return np.rot90(image, k=2)

    # -- Full OCR fallback for low CNN confidence --
    logger.debug(f"  CNN low confidence ({cnn_conf:.2f}), running full OCR")

    scores = {}
    for k in [0, 1, 2, 3]:
        rotated = np.rot90(gray, k=k) if k != 0 else gray
        scores[k] = _ocr_word_score(rotated)

    best_k = max(scores, key=scores.get)
    best_score = scores[best_k]
    current_score = scores[0]

    logger.debug(f"  OCR scores: 0°={scores[0]} 90°={scores[1]} 180°={scores[2]} 270°={scores[3]}")

    # CNN and OCR agree
    if cnn_k == best_k and best_k != 0 and best_score >= 3:
        logger.info(f"  Auto-rotating {best_k * 90} CCW (CNN+OCR agree, OCR={best_score} words)")
        return np.rot90(image, k=best_k)

    # OCR has a clear winner
    if best_k != 0 and best_score > current_score * 1.2 and best_score >= 3:
        logger.info(f"  Auto-rotating {best_k * 90} CCW (OCR: {best_score} words vs {current_score} at 0°)")
        return np.rot90(image, k=best_k)

    if best_k != 0 and best_score > 0 and current_score == 0:
        logger.info(f"  Auto-rotating {best_k * 90} CCW (OCR: {best_score} words, current=0)")
        return np.rot90(image, k=best_k)

    # No strong signal — use CNN if it has some confidence
    if cnn_k != 0 and cnn_conf >= 0.35:
        logger.info(f"  Auto-rotating {cnn_k * 90} CCW (CNN low-conf={cnn_conf:.2f}, OCR inconclusive)")
        return np.rot90(image, k=cnn_k)

    return image
