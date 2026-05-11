"""
SmartDoc 2015 Benchmark for pagescan — Google Colab version.

Paste this entire script into a Colab cell, or upload and run with:
    !python colab_benchmark.py

It will:
1. Install pagescan from GitHub
2. Install system Tesseract
3. Download SmartDoc 2015 dataset (~1GB)
4. Run full benchmark (25k frames)
"""

import subprocess
import sys

# ── Step 1: Install dependencies ──
print("=" * 60)
print("Installing pagescan and dependencies...")
print("=" * 60)
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                       "git+https://github.com/7RPlus-GmbH/pagescan.git"])
subprocess.check_call(["apt-get", "-qq", "install", "-y", "tesseract-ocr", "tesseract-ocr-deu"],
                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
print("Done.\n")

# ── Step 2: Download SmartDoc 2015 ──
import os
from pathlib import Path

DATA_DIR = Path("/content/smartdoc15")
DATA_DIR.mkdir(exist_ok=True)

if not (DATA_DIR / "metadata.csv.gz").exists():
    print("=" * 60)
    print("Downloading SmartDoc 2015 Challenge 1 dataset (~1GB)...")
    print("=" * 60)
    subprocess.check_call([
        "curl", "-L", "-o", str(DATA_DIR / "frames.tar.gz"),
        "https://github.com/jchazalon/smartdoc15-ch1-dataset/releases/download/v2.0.0/frames.tar.gz"
    ])
    print("Extracting...")
    subprocess.check_call(["tar", "xzf", str(DATA_DIR / "frames.tar.gz"), "-C", str(DATA_DIR)])
    print("Done.\n")
else:
    print("SmartDoc dataset already present.\n")

# ── Step 3: Run benchmark ──
import csv
import gzip
import time
from collections import defaultdict

import cv2
import numpy as np

from pagescan.corners import detect_corners_ml, order_corners
from pagescan.edges import find_paper_contour
from pagescan.config import ScanConfig


def polygon_iou(pts1, pts2):
    pts1 = pts1.astype(np.float32).reshape(-1, 2)
    pts2 = pts2.astype(np.float32).reshape(-1, 2)
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
    return inter_area / union_area if union_area >= 1 else 0.0


def detect_with_pagescan(image):
    corners = detect_corners_ml(image)
    if corners is not None:
        return order_corners(corners)
    config = ScanConfig()
    top, bottom, left, right = find_paper_contour(image, config)
    h, w = image.shape[:2]
    if top > 0 or bottom < h or left > 0 or right < w:
        return np.array([
            [left, top], [right, top], [right, bottom], [left, bottom]
        ], dtype=np.float32)
    return None


# Load metadata
meta_path = DATA_DIR / "metadata.csv.gz"
entries = []
with gzip.open(meta_path, 'rt') as f:
    for row in csv.DictReader(f):
        entries.append(row)

print("=" * 60)
print(f"Running benchmark on {len(entries)} frames...")
print("=" * 60)

results_by_bg = defaultdict(list)
results_all = []
failures = 0
total_time = 0.0

for i, entry in enumerate(entries):
    image_path = DATA_DIR / entry['image_path']
    if not image_path.exists():
        continue

    gt_corners = np.array([
        [float(entry['tl_x']), float(entry['tl_y'])],
        [float(entry['tr_x']), float(entry['tr_y'])],
        [float(entry['br_x']), float(entry['br_y'])],
        [float(entry['bl_x']), float(entry['bl_y'])],
    ], dtype=np.float32)

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
        continue

    iou = polygon_iou(pred_corners, gt_corners)
    results_by_bg[bg_name].append(iou)
    results_all.append(iou)

    if (i + 1) % 2000 == 0:
        avg = np.mean(results_all)
        fps = (i + 1) / total_time if total_time > 0 else 0
        print(f"  [{i+1}/{len(entries)}] Mean IoU: {avg:.3f} | {fps:.1f} fps")

# ── Results ──
print(f"\n{'=' * 60}")
print(f"SmartDoc 2015 Challenge 1 — pagescan Benchmark Results")
print(f"{'=' * 60}")
print(f"Total frames:     {len(results_all)}")
print(f"Detections:       {len(results_all) - failures} "
      f"({(len(results_all) - failures) / max(len(results_all), 1) * 100:.1f}%)")
print(f"Failures:         {failures}")
if total_time > 0:
    print(f"Total time:       {total_time:.0f}s ({total_time/60:.1f}min)")
    print(f"Avg time/frame:   {total_time / max(len(results_all), 1) * 1000:.0f}ms")
print()

print(f"{'Background':<25} {'Count':>6} {'Mean IoU':>10} {'Median':>10} "
      f"{'IoU>0.5':>8} {'IoU>0.9':>8}")
print("-" * 75)
for bg_name in sorted(results_by_bg.keys()):
    scores = results_by_bg[bg_name]
    arr = np.array(scores)
    n = len(arr)
    print(f"{bg_name:<25} {n:>6} {np.mean(arr):>10.3f} {np.median(arr):>10.3f} "
          f"{np.sum(arr > 0.5) / n * 100:>7.1f}% {np.sum(arr > 0.9) / n * 100:>7.1f}%")

arr = np.array(results_all)
print("-" * 75)
print(f"{'OVERALL':<25} {len(arr):>6} {np.mean(arr):>10.3f} {np.median(arr):>10.3f} "
      f"{np.sum(arr > 0.5) / len(arr) * 100:>7.1f}% {np.sum(arr > 0.9) / len(arr) * 100:>7.1f}%")
