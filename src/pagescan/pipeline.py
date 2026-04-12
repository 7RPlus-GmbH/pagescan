"""Main scanning pipeline: photo of document -> clean PDF."""

import logging
import multiprocessing
import os
from pathlib import Path
from typing import Dict, Optional

import cv2
import numpy as np

from pagescan.config import ScanConfig
from pagescan.corners import detect_corners, order_corners
from pagescan.edges import trim_edges, find_precise_edges, find_paper_contour, find_document_edges, detect_corners_contour
from pagescan.enhance import remove_shadows, white_balance, enhance_document
from pagescan.orientation import deskew, auto_rotate
from pagescan.output import save_pdf, save_image
from pagescan.quality import check_quality
from pagescan.transform import perspective_transform, place_on_canvas

logger = logging.getLogger(__name__)


def _conservative_crop(image: np.ndarray, config: ScanConfig) -> np.ndarray:
    """Crop background edges without perspective transform.

    Conservative approach: only crop when very confident we're removing
    background, never document content. Prefers to leave some background
    rather than risk cutting text.
    """
    h, w = image.shape[:2]

    # Try contour-based perspective correction first (gives best results for tilted docs)
    contour_corners = detect_corners_contour(image, config)
    if contour_corners is not None:
        c_ordered = order_corners(contour_corners)
        c_area = cv2.contourArea(c_ordered)
        c_coverage = c_area / (h * w)
        if 0.05 < c_coverage < 0.95:
            logger.info(f"  Contour perspective (coverage={c_coverage:.2f})")
            return perspective_transform(image, contour_corners)

    # Fallback: HSV paper contour detection
    top, bottom, left, right = find_paper_contour(image, config)
    cropped = (top > 0 or bottom < h or left > 0 or right < w)

    if not cropped:
        # Try edge-based detection (for white/beige backgrounds)
        top, bottom, left, right = find_document_edges(image, config)
        cropped = (top > 0 or bottom < h or left > 0 or right < w)

    if not cropped:
        return image

    # Safety check: never remove more than 40% of any dimension
    crop_h = bottom - top
    crop_w = right - left
    if crop_h < h * 0.60 or crop_w < w * 0.60:
        logger.warning(f"  Conservative crop would over-crop ({crop_w}x{crop_h} from {w}x{h}), keeping full image")
        return image

    logger.info(f"  Conservative crop: T={top} B={h - bottom} L={left} R={w - right}")
    return image[top:bottom, left:right]


def scan(image_path: str, output_path: str = None,
         config: ScanConfig = None) -> Dict:
    """Scan a single document photo into a clean PDF.

    Pipeline:
        1. Load image
        2. Detect corners (ML with validation, or conservative crop)
        3. Perspective transform or direct crop
        4. Auto-rotate (OCR-based orientation correction)
        5. Deskew (Hough-based text tilt correction)
        6. Shadow removal + white balance
        7. Scan-like enhancement (grayscale, contrast, sharpen)
        8. Place on canvas (A4) and save as PDF

    Args:
        image_path: Path to input image (JPEG, PNG, TIFF).
        output_path: Path for output PDF. If None, replaces extension with .pdf.
        config: Scan configuration. Uses defaults if None.

    Returns:
        Dict with keys: success, quality_score, quality_passed, message, output_path.
    """
    if config is None:
        config = ScanConfig()

    image = cv2.imread(image_path)
    if image is None:
        return {'success': False, 'message': f"Cannot read: {image_path}"}

    logger.info(f"Processing: {image_path}")
    h, w = image.shape[:2]
    logger.info(f"  Input: {w}x{h}")

    if config.debug:
        debug_dir = Path(config.debug_dir)
        debug_dir.mkdir(exist_ok=True)

    # -- Corner detection with rotation retry --
    ml_corners, ml_pre_rotation = detect_corners(image, config)

    if ml_pre_rotation != 0:
        image = np.rot90(image, k=ml_pre_rotation)
        h, w = image.shape[:2]

    if ml_corners is not None:
        method = 'docaligner'
        corners = ml_corners
    else:
        method = 'conservative'
        corners = None
    logger.info(f"  Detection: {method}")

    if config.debug and corners is not None:
        vis = image.copy()
        pts = order_corners(corners).astype(np.int32)
        cv2.polylines(vis, [pts], True, (0, 255, 0), 3)
        for pt in pts:
            cv2.circle(vis, tuple(pt), 15, (0, 0, 255), -1)
        save_image(vis, str(Path(config.debug_dir) / 'corners.jpg'))

    # -- Transform or crop --
    if method == 'conservative':
        document = _conservative_crop(image.copy(), config)
    else:
        document = perspective_transform(image, corners)
        logger.info(f"  Straightened: {document.shape[1]}x{document.shape[0]}")
        # NO aggressive trimming after ML perspective — the ML corners define
        # the document boundary. Trimming here cuts off actual content.
        if ml_pre_rotation != 0:
            undo_k = (4 - ml_pre_rotation) % 4
            document = np.rot90(document, k=undo_k)
            logger.info(f"  Undoing ML pre-rotation: {undo_k * 90} CCW")

    logger.info(f"  Final crop: {document.shape[1]}x{document.shape[0]}")

    # -- Auto-rotate FIRST (text must be horizontal before deskew) --
    if config.auto_orient:
        document = auto_rotate(document)

    # -- Deskew --
    if config.deskew:
        document, deskew_angle = deskew(document)

    # -- Quality check (informational only, no retries that would over-crop) --
    passed, quality_score, qa_message = check_quality(document, config)
    logger.info(f"  Quality: {quality_score:.2f} - {qa_message}")

    # -- Enhancement --
    if config.shadow_removal:
        document = remove_shadows(document)
    if config.white_balance:
        document = white_balance(document)

    if config.debug:
        save_image(document, str(Path(config.debug_dir) / 'enhanced_color.jpg'))

    if config.enhance:
        document = enhance_document(document)

    result = place_on_canvas(document, config)

    if config.debug:
        save_image(result, str(Path(config.debug_dir) / 'result.jpg'))

    # -- Output --
    if output_path is None:
        output_path = str(Path(image_path).with_suffix('.pdf'))

    save_pdf(result, output_path, config)
    logger.info(f"  Saved: {output_path}")

    return {
        'success': True,
        'quality_score': quality_score,
        'quality_passed': passed,
        'message': qa_message,
        'output_path': output_path,
    }


def _process_single(args):
    """Worker for multiprocessing batch."""
    img_path, out_path, config = args
    try:
        result = scan(img_path, out_path, config=config)
        return Path(img_path).name, result
    except Exception as e:
        return Path(img_path).name, {'success': False, 'message': str(e)}


def scan_batch(input_dir: str, output_dir: str = None,
               config: ScanConfig = None, workers: int = None) -> Dict:
    """Process all images in a directory.

    Args:
        input_dir: Directory containing input images.
        output_dir: Directory for output PDFs. Defaults to input_dir.
        config: Scan configuration.
        workers: Number of parallel workers. Default: min(4, cpu_count).
                 Use 1 for sequential/debug mode.

    Returns:
        Dict with processed, failed, low_quality counts.
    """
    if config is None:
        config = ScanConfig()

    input_path = Path(input_dir)
    output_path = Path(output_dir) if output_dir else input_path
    output_path.mkdir(parents=True, exist_ok=True)

    extensions = {'.jpg', '.jpeg', '.png', '.tiff', '.tif'}
    images = sorted([f for f in input_path.iterdir()
                     if f.is_file() and f.suffix.lower() in extensions])

    if not images:
        logger.warning(f"No images found in {input_dir}")
        return {'processed': 0, 'failed': 0, 'low_quality': 0}

    if workers is None:
        workers = min(4, os.cpu_count() or 1)

    logger.info(f"Processing {len(images)} images with {workers} worker(s)...")

    results = {'processed': 0, 'failed': 0, 'low_quality': 0}

    tasks = []
    for img_path in images:
        out = output_path / (img_path.stem + '.pdf')
        tasks.append((str(img_path), str(out), config))

    if workers <= 1:
        for i, (img_str, out_str, cfg) in enumerate(tasks):
            name = Path(img_str).name
            print(f"\n[{i + 1}/{len(tasks)}] {name}")
            try:
                result = scan(img_str, out_str, config=cfg)
                if result['success']:
                    results['processed'] += 1
                    if not result['quality_passed']:
                        results['low_quality'] += 1
                        print(f"  Warning: {result['message']}")
                else:
                    results['failed'] += 1
                    print(f"  Error: {result['message']}")
            except Exception as e:
                results['failed'] += 1
                print(f"  Error: {e}")
    else:
        ctx = multiprocessing.get_context('spawn')
        with ctx.Pool(workers) as pool:
            for i, (name, result) in enumerate(pool.imap_unordered(_process_single, tasks)):
                idx = i + 1
                if result['success']:
                    results['processed'] += 1
                    q = result['quality_score']
                    if not result.get('quality_passed', True):
                        results['low_quality'] += 1
                        print(f"[{idx}/{len(tasks)}] {name}  Warning: {result['message']}")
                    else:
                        print(f"[{idx}/{len(tasks)}] {name}  OK (q={q:.2f})")
                else:
                    results['failed'] += 1
                    print(f"[{idx}/{len(tasks)}] {name}  Error: {result['message']}")

    print(f"\nProcessed: {results['processed']}")
    print(f"Low quality: {results['low_quality']}")
    print(f"Failed: {results['failed']}")

    return results
