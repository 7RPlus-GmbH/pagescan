#!/usr/bin/env python3
"""
Benchmark pagescan corner detection against SmartDoc 2015 Challenge 1.

Evaluates document corner detection accuracy using Intersection over Union (IoU).
The dataset contains ~25k frames across 5 backgrounds and 30 document models.

Usage:
    python benchmark/run_smartdoc.py                    # Full benchmark
    python benchmark/run_smartdoc.py --sample 0.1       # 10% sample (fast)
    python benchmark/run_smartdoc.py --backgrounds 01   # Single background
"""

import argparse
import csv
import gzip
import logging
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from pagescan.corners import detect_corners_ml, order_corners
from pagescan.edges import detect_corners_contour, find_document_edges
from pagescan.config import ScanConfig

logging.basicConfig(level=logging.WARNING, format='%(levelname)s: %(message)s')

DATASET_DIR = Path(__file__).parent / "data"


def load_metadata(dataset_dir: Path):
    """Load ground truth from metadata.csv.gz."""
    meta_path = dataset_dir / "metadata.csv.gz"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"metadata.csv.gz not found in {dataset_dir}.\n"
            f"Download from: https://github.com/jchazalon/smartdoc15-ch1-dataset/releases"
        )

    entries = []
    with gzip.open(meta_path, 'rt') as f:
        reader = csv.DictReader(f)
        for row in reader:
            entries.append(row)
    return entries


def polygon_iou(pts1: np.ndarray, pts2: np.ndarray) -> float:
    """Compute IoU between two quadrilateral polygons.

    Uses cv2.intersectConvexConvex for robust polygon intersection.
    """
    pts1 = pts1.astype(np.float32).reshape(-1, 2)
    pts2 = pts2.astype(np.float32).reshape(-1, 2)

    # Ensure convex and proper ordering
    hull1 = cv2.convexHull(pts1)
    hull2 = cv2.convexHull(pts2)

    area1 = cv2.contourArea(hull1)
    area2 = cv2.contourArea(hull2)

    if area1 < 1 or area2 < 1:
        return 0.0

    ret, intersection = cv2.intersectConvexConvex(hull1, hull2)
    if ret <= 0 or intersection is None or len(intersection) < 3:
        return 0.0

    inter_area = cv2.contourArea(intersection)
    union_area = area1 + area2 - inter_area

    if union_area < 1:
        return 0.0

    return inter_area / union_area


def detect_with_pagescan(image: np.ndarray) -> np.ndarray:
    """Detect corners using pagescan's ML + edge fallback cascade.

    Cascade:
    1. ML corners (best quality — proper perspective quad)
    2. Edge-based quad (contour approxPolyDP or minAreaRect)
    3. Edge-based bounding box (axis-aligned, worst IoU but high detection rate)

    Returns 4 corners as np.ndarray (4, 2) or None.
    """
    # 1. ML detection
    corners = detect_corners_ml(image)
    if corners is not None:
        return order_corners(corners)

    config = ScanConfig()

    # 2. Edge-based quad (proper quadrilateral)
    corners = detect_corners_contour(image, config)
    if corners is not None:
        return order_corners(corners)

    # 3. Edge-based bounding box (last resort)
    h, w = image.shape[:2]
    top, bottom, left, right = find_document_edges(image, config)
    if top > 0 or bottom < h or left > 0 or right < w:
        return np.array([
            [left, top], [right, top], [right, bottom], [left, bottom]
        ], dtype=np.float32)

    return None


def run_benchmark(dataset_dir: Path, sample: float = 1.0,
                  backgrounds: list = None, verbose: bool = False):
    """Run the SmartDoc 2015 benchmark."""
    entries = load_metadata(dataset_dir)
    print(f"Loaded {len(entries)} ground truth entries")

    # Filter by background if specified
    if backgrounds:
        entries = [e for e in entries if any(
            e['bg_name'].startswith(bg) or bg in e['bg_name']
            for bg in backgrounds
        )]
        print(f"Filtered to {len(entries)} entries for backgrounds: {backgrounds}")

    # Sample if requested
    if sample < 1.0:
        rng = np.random.RandomState(42)
        n = max(1, int(len(entries) * sample))
        indices = rng.choice(len(entries), n, replace=False)
        entries = [entries[i] for i in sorted(indices)]
        print(f"Sampled {len(entries)} entries ({sample:.0%})")

    # Results tracking
    results_by_bg = defaultdict(list)
    results_all = []
    ml_detections = 0
    fallback_detections = 0
    failures = 0
    total_time = 0.0

    for i, entry in enumerate(entries):
        image_path = dataset_dir / entry['image_path']
        if not image_path.exists():
            if verbose:
                print(f"  SKIP: {image_path} not found")
            continue

        # Ground truth corners: TL, BL, BR, TR -> reorder to TL, TR, BR, BL
        gt_corners = np.array([
            [float(entry['tl_x']), float(entry['tl_y'])],
            [float(entry['tr_x']), float(entry['tr_y'])],
            [float(entry['br_x']), float(entry['br_y'])],
            [float(entry['bl_x']), float(entry['bl_y'])],
        ], dtype=np.float32)

        # Load and detect
        image = cv2.imread(str(image_path))
        if image is None:
            continue

        t0 = time.time()
        pred_corners = detect_with_pagescan(image)
        dt = time.time() - t0
        total_time += dt

        bg_name = entry['bg_name']

        if pred_corners is None:
            failures += 1
            results_by_bg[bg_name].append(0.0)
            results_all.append(0.0)
            if verbose:
                print(f"  [{i+1}/{len(entries)}] {bg_name} FAIL (no detection)")
            continue

        iou = polygon_iou(pred_corners, gt_corners)
        results_by_bg[bg_name].append(iou)
        results_all.append(iou)

        if verbose and (iou < 0.5 or (i + 1) % 500 == 0):
            print(f"  [{i+1}/{len(entries)}] {bg_name} IoU={iou:.3f} ({dt*1000:.0f}ms)")

        # Progress
        if (i + 1) % 1000 == 0:
            avg = np.mean(results_all)
            fps = (i + 1) / total_time if total_time > 0 else 0
            print(f"  Progress: {i+1}/{len(entries)} | "
                  f"Mean IoU: {avg:.3f} | {fps:.1f} fps")

    # Summary
    print(f"\n{'='*60}")
    print(f"SmartDoc 2015 Challenge 1 — pagescan Benchmark Results")
    print(f"{'='*60}")
    print(f"Total frames:     {len(results_all)}")
    print(f"Detections:       {len(results_all) - failures} "
          f"({(len(results_all) - failures) / max(len(results_all), 1) * 100:.1f}%)")
    print(f"Failures:         {failures}")
    if total_time > 0:
        print(f"Avg time/frame:   {total_time / max(len(results_all), 1) * 1000:.0f}ms")
    print()

    # Per-background results
    print(f"{'Background':<25} {'Count':>6} {'Mean IoU':>10} {'Median':>10} "
          f"{'IoU>0.5':>8} {'IoU>0.9':>8}")
    print("-" * 75)
    for bg_name in sorted(results_by_bg.keys()):
        scores = results_by_bg[bg_name]
        arr = np.array(scores)
        n = len(arr)
        mean_iou = np.mean(arr)
        median_iou = np.median(arr)
        above_50 = np.sum(arr > 0.5) / n * 100
        above_90 = np.sum(arr > 0.9) / n * 100
        print(f"{bg_name:<25} {n:>6} {mean_iou:>10.3f} {median_iou:>10.3f} "
              f"{above_50:>7.1f}% {above_90:>7.1f}%")

    # Overall
    arr = np.array(results_all)
    mean_iou = np.mean(arr)
    median_iou = np.median(arr)
    above_50 = np.sum(arr > 0.5) / len(arr) * 100
    above_90 = np.sum(arr > 0.9) / len(arr) * 100
    print("-" * 75)
    print(f"{'OVERALL':<25} {len(arr):>6} {mean_iou:>10.3f} {median_iou:>10.3f} "
          f"{above_50:>7.1f}% {above_90:>7.1f}%")
    print()

    return {
        'mean_iou': mean_iou,
        'median_iou': median_iou,
        'above_50': above_50,
        'above_90': above_90,
        'total': len(results_all),
        'failures': failures,
        'by_background': {
            bg: {'mean': np.mean(s), 'median': np.median(s), 'count': len(s)}
            for bg, s in results_by_bg.items()
        }
    }


def main():
    parser = argparse.ArgumentParser(
        description='Benchmark pagescan against SmartDoc 2015 Challenge 1')
    parser.add_argument('--data-dir', type=str, default=str(DATASET_DIR),
                        help='Path to SmartDoc frames directory')
    parser.add_argument('--sample', type=float, default=1.0,
                        help='Fraction of dataset to use (0.0-1.0)')
    parser.add_argument('--backgrounds', nargs='+', default=None,
                        help='Filter to specific backgrounds (e.g. 01 02)')
    parser.add_argument('--verbose', '-v', action='store_true')
    args = parser.parse_args()

    dataset_dir = Path(args.data_dir)
    if not dataset_dir.exists():
        print(f"Dataset not found at {dataset_dir}")
        print("Download from: https://github.com/jchazalon/smartdoc15-ch1-dataset/releases")
        print("Extract frames.tar.gz into benchmark/data/")
        return

    run_benchmark(dataset_dir, sample=args.sample,
                  backgrounds=args.backgrounds, verbose=args.verbose)


if __name__ == '__main__':
    main()
