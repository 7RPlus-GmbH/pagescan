"""ONNX inference for document corner detection.

Multi-model approach:

- **Primary:** FastViT_SA24 heatmap regression — perspective-aware corners.
- **Fallback:** LCNet100 heatmap regression — different backbone, catches images where SA24 fails.
- **Conservative:** DeepLabV3-MobileNetV3 segmentation — pixel-level mask, used in conservative crop fallback only.

All models are ONNX and downloaded automatically on first use from the
Hugging Face Hub mirror (``7rplus/pagescan-weights``), falling back to
DocsaidLab's Google Drive (the original DocAligner upstream) if the mirror
is unavailable.
"""
from __future__ import annotations

import logging
import os
import shutil
import urllib.request
from pathlib import Path
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# -- Model registry --
MODELS: dict[str, dict[str, Any]] = {
    'sa24': {
        'filename': 'fastvit_sa24_h_e_bifpn_256_fp32.onnx',
        'gdrive_id': '14vUH77v6yGg7zFctUgcT6BzV5Iisg4Dl',
        'input_size': (256, 256),
        'type': 'heatmap',
    },
    'lcnet100': {
        'filename': 'lcnet100_h_e_bifpn_256_fp32.onnx',
        'gdrive_id': '1IlbLPkCv-TdaBLOPh_4J97P1_KHYYJ7a',
        'input_size': (256, 256),
        'type': 'heatmap',
    },
    'deeplabv3_mbv3': {
        'filename': 'deeplabv3_mbv3_docseg.onnx',
        'gdrive_id': '1ROtCvke02aFzRrVOSBpiIkjMB1oMnMCU',
        'input_size': (384, 384),
        'type': 'segmentation',
        'mean': (0.4611, 0.4359, 0.3905),
        'std': (0.2193, 0.2150, 0.2109),
    },
}

HEATMAP_THRESHOLD = 0.1

# Hugging Face Hub mirror for the legacy weights — the primary download
# source. DocsaidLab's Google Drive (the `gdrive_id` on each model) is the
# upstream fallback, used only if the HF mirror is unreachable.
HF_REPO_ID = "7rplus/pagescan-weights"

_sessions: dict[str, Any] = {}  # model_name -> ort.InferenceSession


def _get_cache_dir() -> Path:
    cache = Path(os.environ.get("PAGESCAN_CACHE",
                                Path.home() / ".cache" / "pagescan"))
    cache.mkdir(parents=True, exist_ok=True)
    return cache


def _download_from_gdrive(file_id: str, dest: Path) -> None:
    """Download a file from Google Drive, handling the confirmation token."""
    url = f"https://drive.google.com/uc?export=download&id={file_id}&confirm=t"
    logger.info(f"  Downloading model to {dest} ...")

    tmp = dest.with_suffix('.tmp')
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "pagescan"})
        with urllib.request.urlopen(req, timeout=120) as resp, open(tmp, 'wb') as f:
            shutil.copyfileobj(resp, f)
        tmp.rename(dest)
        logger.info(f"  Model downloaded ({dest.stat().st_size / 1e6:.0f} MB)")
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


_DATA_MODEL_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "model"


def _download_from_hf(filename: str, dest: Path) -> None:
    """Download a weight file from the Hugging Face Hub mirror into `dest`."""
    from huggingface_hub import hf_hub_download
    logger.info(f"  Downloading {filename} from {HF_REPO_ID} (HF mirror)...")
    hf_path = hf_hub_download(repo_id=HF_REPO_ID, filename=filename)
    shutil.copy2(hf_path, dest)


def _ensure_model(name: str) -> Path:
    """Ensure an ONNX model is available, downloading if needed.

    Search order: data/model/ (project-local) → cache dir → Hugging Face Hub
    mirror (primary download) → DocsaidLab Google Drive (upstream fallback).

    The legacy weights are DocAligner models (DocsaidLab); the canonical
    upstream is their auto-downloader with no stable public URL, so the HF
    mirror at ``HF_REPO_ID`` is the reliable primary source and Google Drive
    is only used if the mirror is unreachable.
    """
    info = MODELS[name]
    filename = info['filename']

    # 1. Check project-local data/model/ directory
    local_path = _DATA_MODEL_DIR / filename
    if local_path.exists() and local_path.stat().st_size > 100_000:
        return local_path

    # 2. Check user cache directory
    model_path = _get_cache_dir() / filename
    if model_path.exists() and model_path.stat().st_size > 100_000:
        return model_path

    # 3. Download from the HF Hub mirror (primary); Google Drive is the fallback.
    try:
        _download_from_hf(filename, model_path)
        return model_path
    except Exception as hf_err:
        gdrive_id = info.get('gdrive_id')
        if gdrive_id is None:
            raise FileNotFoundError(
                f"Model '{name}' ({filename}) not found in {local_path} or "
                f"{model_path}, and the HF Hub mirror ({HF_REPO_ID}) failed: {hf_err}"
            ) from hf_err
        logger.warning(f"  HF Hub download failed for {filename}: {hf_err}; "
                       f"falling back to Google Drive (upstream)")
        try:
            _download_from_gdrive(gdrive_id, model_path)
            return model_path
        except Exception as gd_err:
            raise FileNotFoundError(
                f"Model '{name}' ({filename}) not found locally; both the HF Hub "
                f"mirror ({HF_REPO_ID}) and Google Drive failed "
                f"(HF: {hf_err}; Drive: {gd_err})"
            ) from gd_err


def _get_session(name: str):
    """Get or create an ONNX runtime session for a model (singleton)."""
    if name in _sessions:
        return _sessions[name]

    try:
        import onnxruntime as ort
    except ImportError as err:
        raise ImportError(
            "onnxruntime is required for ML corner detection. "
            "Install it with: pip install onnxruntime"
        ) from err

    model_path = _ensure_model(name)
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    _sessions[name] = ort.InferenceSession(
        str(model_path),
        sess_options=opts,
        providers=["CPUExecutionProvider"],
    )
    return _sessions[name]


# -- Heatmap model inference --

def _preprocess_heatmap(img: np.ndarray, input_size: tuple[int, int]
                        ) -> tuple[np.ndarray, tuple[int, int]]:
    """Resize to model input size and convert to NCHW float tensor."""
    h, w = img.shape[:2]
    resized = cv2.resize(img, input_size, interpolation=cv2.INTER_LINEAR)
    tensor = resized.transpose(2, 0, 1).astype(np.float32)[None] / 255.0
    return tensor, (h, w)


def _postprocess_heatmap(heatmaps: np.ndarray,
                         orig_size: tuple[int, int]) -> np.ndarray | None:
    """Extract 4 corner points from heatmap predictions.

    Uses argmax to find the peak in each channel, then scales to original
    image coordinates. Falls back to contour-based extraction for models
    with sharper peaks.
    """
    oh, ow = orig_size
    hm_h, hm_w = heatmaps.shape[2], heatmaps.shape[3]
    points = []

    for i in range(min(4, heatmaps.shape[1])):
        hmap = heatmaps[0, i]

        if hmap.max() < HEATMAP_THRESHOLD:
            return None

        # Argmax-based: find peak location, scale to original coords
        idx = np.argmax(hmap)
        py, px = np.unravel_index(idx, hmap.shape)
        cx = px * ow / hm_w
        cy = py * oh / hm_h
        points.append([cx, cy])

    if len(points) != 4:
        return None

    return np.array(points, dtype=np.float32)


def _run_heatmap_raw(image: np.ndarray, model_name: str) -> np.ndarray | None:
    """Run a heatmap model and return raw heatmaps (1, 4, H, W) or None."""
    if image.ndim != 3 or image.shape[2] != 3:
        return None

    try:
        session = _get_session(model_name)
    except Exception:
        return None

    info = MODELS[model_name]
    tensor, _ = _preprocess_heatmap(image, info['input_size'])

    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    return session.run([output_name], {input_name: tensor})[0]


def _heatmap_confidence(heatmaps: np.ndarray) -> float:
    """Mean of the 4 channel peak values — higher = more confident."""
    if heatmaps is None:
        return 0.0
    return float(np.mean([heatmaps[0, i].max() for i in range(4)]))


def _detect_heatmap(image: np.ndarray, model_name: str) -> np.ndarray | None:
    """Run a heatmap model and return 4 corners or None."""
    heatmaps = _run_heatmap_raw(image, model_name)
    if heatmaps is None:
        return None
    h, w = image.shape[:2]
    return _postprocess_heatmap(heatmaps, (h, w))


# -- Segmentation model inference --

def _detect_segmentation(image: np.ndarray, model_name: str) -> np.ndarray | None:
    """Run a segmentation model and extract document corners from the mask."""
    if image.ndim != 3 or image.shape[2] != 3:
        return None

    info = MODELS[model_name]
    session = _get_session(model_name)
    h, w = image.shape[:2]

    # Preprocess: resize, RGB, normalize
    input_size = info['input_size']
    resized = cv2.resize(image, input_size, interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    mean = np.array(info['mean'], dtype=np.float32)
    std = np.array(info['std'], dtype=np.float32)
    rgb = (rgb - mean) / std

    tensor = rgb.transpose(2, 0, 1)[np.newaxis, ...].astype(np.float32)

    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    outputs = session.run([output_name], {input_name: tensor})

    # Post-process: argmax to get binary mask
    pred = outputs[0]  # (1, 2, H, W) or (1, H, W)
    if pred.ndim == 4 and pred.shape[1] == 2:
        mask = (pred[0].argmax(axis=0) * 255).astype(np.uint8)
    elif pred.ndim == 3:
        mask = ((pred[0] > 0.5) * 255).astype(np.uint8)
    else:
        return None

    # Morphological cleanup
    ks = max(5, min(mask.shape) // 30) | 1
    kernel = np.ones((ks, ks), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)

    # Find contours and fit quadrilateral
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < mask.shape[0] * mask.shape[1] * 0.02:
        return None

    # Scale contour to original image coordinates
    scale_x = w / input_size[0]
    scale_y = h / input_size[1]

    peri = cv2.arcLength(largest, True)
    for eps in [0.015, 0.02, 0.03, 0.04, 0.05, 0.07, 0.10]:
        approx = cv2.approxPolyDP(largest, eps * peri, True)
        if len(approx) == 4:
            pts = approx.reshape(4, 2).astype(np.float32)
            pts[:, 0] *= scale_x
            pts[:, 1] *= scale_y
            return pts

    # Fallback: minimum area rotated rectangle
    rect = cv2.minAreaRect(largest)
    box = cv2.boxPoints(rect).astype(np.float32)
    box[:, 0] *= scale_x
    box[:, 1] *= scale_y
    return box


# -- Public API --

def detect_corners_onnx(image: np.ndarray) -> np.ndarray | None:
    """Detect document corners using multi-model fallback chain.

    1. FastViT_SA24 heatmap (primary) — perspective-aware corners
    2. LCNet100 heatmap — different backbone, catches SA24 failures

    Returns 4 corner points as float32 array of shape (4, 2), or None.
    """
    _h, _w = image.shape[:2]

    # Primary: SA24 heatmap
    corners = _detect_heatmap(image, 'sa24')
    if corners is not None:
        logger.info("  Corner detection: SA24")
        return corners

    # Fallback: LCNet100 heatmap (different backbone)
    corners = _detect_heatmap(image, 'lcnet100')
    if corners is not None:
        logger.info("  Corner detection: LC100 fallback")
        return corners

    return None


def detect_corners_segmentation(image: np.ndarray) -> np.ndarray | None:
    """Detect document corners using DeepLabV3 segmentation.

    Separate from the heatmap chain — used in conservative crop fallback.
    Segments the document region at pixel level, then fits a quadrilateral.

    Returns 4 corner points as float32 array of shape (4, 2), or None.
    """
    return _detect_segmentation(image, 'deeplabv3_mbv3')
