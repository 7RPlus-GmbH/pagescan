#!/usr/bin/env python3
"""
End-to-end comparison of document scanning packages.

Compares pagescan vs docscan vs OpenCV contour scanner on real phone photos.
Measures both detection quality and full pipeline output.

Usage:
    python benchmark/comparison/run_comparison.py
    python benchmark/comparison/run_comparison.py --images-dir path/to/photos
"""

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

logging.basicConfig(level=logging.WARNING)

IMAGES_DIR = Path(__file__).parent / "images"
RESULTS_DIR = Path(__file__).parent / "results"


# ═══════════════════════════════════════════════════════════════════════
# Scanner implementations
# ═══════════════════════════════════════════════════════════════════════

def scan_pagescan(image_path: str, output_path: str) -> Dict:
    """pagescan: full pipeline scan."""
    from pagescan import scan, ScanConfig
    config = ScanConfig(debug=False)
    t0 = time.time()
    result = scan(image_path, output_path, config=config)
    dt = time.time() - t0
    return {
        "success": result.get("success", False),
        "quality": result.get("quality_score", 0),
        "time_ms": dt * 1000,
        "method": "pagescan",
    }


def scan_docscan(image_path: str, output_path: str) -> Dict:
    """docscan: rembg + contour detection + perspective."""
    try:
        from docscan.doc import scan as docscan_scan
        t0 = time.time()
        with open(image_path, "rb") as f:
            data = f.read()
        result_bytes = docscan_scan(data)
        dt = time.time() - t0

        if result_bytes is None:
            return {"success": False, "quality": 0, "time_ms": dt * 1000, "method": "docscan"}

        # docscan returns image bytes, save as PDF
        img_array = np.frombuffer(result_bytes, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if img is None:
            return {"success": False, "quality": 0, "time_ms": dt * 1000, "method": "docscan"}

        # Save as image (docscan doesn't produce PDFs)
        cv2.imwrite(output_path.replace(".pdf", ".jpg"), img)
        return {
            "success": True,
            "quality": 1.0,  # No quality metric from docscan
            "time_ms": dt * 1000,
            "method": "docscan",
        }
    except Exception as e:
        return {"success": False, "quality": 0, "time_ms": 0, "method": "docscan", "error": str(e)}


def scan_opencv(image_path: str, output_path: str) -> Dict:
    """Classic OpenCV contour-based scanner (the StackOverflow approach)."""
    t0 = time.time()
    try:
        image = cv2.imread(image_path)
        if image is None:
            return {"success": False, "quality": 0, "time_ms": 0, "method": "opencv"}

        orig = image.copy()
        h, w = image.shape[:2]

        # Resize for processing
        ratio = h / 500.0
        image = cv2.resize(image, (int(w / ratio), 500))

        # Grayscale + blur + Canny
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(gray, 75, 200)

        # Dilate to close gaps
        kernel = np.ones((5, 5), np.uint8)
        edged = cv2.dilate(edged, kernel, iterations=1)

        # Find contours
        contours, _ = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

        doc_contour = None
        for c in contours:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            if len(approx) == 4:
                doc_contour = approx
                break

        dt_detect = time.time() - t0

        if doc_contour is None:
            return {"success": False, "quality": 0, "time_ms": dt_detect * 1000, "method": "opencv"}

        # Scale back to original
        corners = doc_contour.reshape(4, 2) * ratio

        # Order corners: TL, TR, BR, BL
        rect = np.zeros((4, 2), dtype=np.float32)
        s = corners.sum(axis=1)
        d = np.diff(corners, axis=1).flatten()
        rect[0] = corners[np.argmin(s)]
        rect[2] = corners[np.argmax(s)]
        rect[1] = corners[np.argmin(d)]
        rect[3] = corners[np.argmax(d)]

        # Perspective transform
        width = int(max(
            np.linalg.norm(rect[1] - rect[0]),
            np.linalg.norm(rect[2] - rect[3])
        ))
        height = int(max(
            np.linalg.norm(rect[3] - rect[0]),
            np.linalg.norm(rect[2] - rect[1])
        ))

        dst = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32)
        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(orig, M, (width, height))

        # Simple enhancement: grayscale + threshold
        gray_out = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        gray_out = cv2.adaptiveThreshold(
            gray_out, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 10)

        dt = time.time() - t0
        cv2.imwrite(output_path.replace(".pdf", ".jpg"), gray_out)

        return {
            "success": True,
            "quality": 1.0,
            "time_ms": dt * 1000,
            "method": "opencv",
        }
    except Exception as e:
        return {"success": False, "quality": 0, "time_ms": 0, "method": "opencv", "error": str(e)}


# ═══════════════════════════════════════════════════════════════════════
# Evaluation
# ═══════════════════════════════════════════════════════════════════════

SCANNERS = {
    "pagescan": scan_pagescan,
    "docscan": scan_docscan,
    "opencv": scan_opencv,
}


def evaluate_image(image_path: Path, results_dir: Path) -> Dict:
    """Run all scanners on a single image and collect results."""
    stem = image_path.stem  # e.g. "invoice_wood"
    parts = stem.rsplit("_", 1)
    doc_type = parts[0] if len(parts) == 2 else stem
    background = parts[1] if len(parts) == 2 else "unknown"

    results = {
        "image": image_path.name,
        "doc_type": doc_type,
        "background": background,
        "scanners": {},
    }

    for name, scanner_fn in SCANNERS.items():
        output_dir = results_dir / name
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = str(output_dir / f"{stem}.pdf")

        print(f"  {name}...", end=" ", flush=True)
        try:
            result = scanner_fn(str(image_path), output_path)
            status = "OK" if result["success"] else "FAIL"
            print(f"{status} ({result['time_ms']:.0f}ms)")
        except Exception as e:
            result = {"success": False, "quality": 0, "time_ms": 0, "method": name, "error": str(e)}
            print(f"ERROR: {e}")

        results["scanners"][name] = result

    return results


def print_summary(all_results: List[Dict]):
    """Print comparison summary table."""
    # Aggregate by scanner
    scanner_stats = {}
    for name in SCANNERS:
        successes = [r["scanners"][name] for r in all_results if r["scanners"][name]["success"]]
        failures = [r["scanners"][name] for r in all_results if not r["scanners"][name]["success"]]
        total = len(all_results)
        scanner_stats[name] = {
            "success_rate": len(successes) / total * 100,
            "avg_time": np.mean([s["time_ms"] for s in successes]) if successes else 0,
            "successes": len(successes),
            "failures": len(failures),
            "total": total,
        }

    print(f"\n{'='*70}")
    print(f"End-to-End Comparison Results")
    print(f"{'='*70}")
    print(f"{'Scanner':<15} {'Success':>8} {'Rate':>8} {'Avg Time':>10} {'Fails':>8}")
    print("-" * 55)
    for name in SCANNERS:
        s = scanner_stats[name]
        print(f"{name:<15} {s['successes']:>5}/{s['total']:<3} {s['success_rate']:>7.1f}% "
              f"{s['avg_time']:>8.0f}ms {s['failures']:>8}")

    # Per background
    backgrounds = sorted(set(r["background"] for r in all_results))
    if len(backgrounds) > 1:
        print(f"\n{'Success rate by background':}")
        print(f"{'Scanner':<15}", end="")
        for bg in backgrounds:
            print(f" {bg:>10}", end="")
        print()
        print("-" * (15 + 11 * len(backgrounds)))
        for name in SCANNERS:
            print(f"{name:<15}", end="")
            for bg in backgrounds:
                bg_results = [r for r in all_results if r["background"] == bg]
                bg_success = sum(1 for r in bg_results if r["scanners"][name]["success"])
                rate = bg_success / len(bg_results) * 100 if bg_results else 0
                print(f" {rate:>9.0f}%", end="")
            print()

    # Per document type
    doc_types = sorted(set(r["doc_type"] for r in all_results))
    if len(doc_types) > 1:
        print(f"\n{'Success rate by document type':}")
        print(f"{'Scanner':<15}", end="")
        for dt in doc_types:
            label = dt[:8]
            print(f" {label:>10}", end="")
        print()
        print("-" * (15 + 11 * len(doc_types)))
        for name in SCANNERS:
            print(f"{name:<15}", end="")
            for dt in doc_types:
                dt_results = [r for r in all_results if r["doc_type"] == dt]
                dt_success = sum(1 for r in dt_results if r["scanners"][name]["success"])
                rate = dt_success / len(dt_results) * 100 if dt_results else 0
                print(f" {rate:>9.0f}%", end="")
            print()


def main():
    parser = argparse.ArgumentParser(description="Compare document scanning packages")
    parser.add_argument("--images-dir", type=str, default=str(IMAGES_DIR),
                        help="Directory with test photos")
    parser.add_argument("--results-dir", type=str, default=str(RESULTS_DIR))
    parser.add_argument("--scanners", nargs="+", default=list(SCANNERS.keys()),
                        help="Which scanners to run")
    args = parser.parse_args()

    images_dir = Path(args.images_dir)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Filter scanners
    global SCANNERS
    SCANNERS = {k: v for k, v in SCANNERS.items() if k in args.scanners}

    extensions = {".jpg", ".jpeg", ".png", ".tiff", ".tif"}
    images = sorted([f for f in images_dir.iterdir()
                     if f.is_file() and f.suffix.lower() in extensions])

    if not images:
        print(f"No images found in {images_dir}")
        print(f"Add test photos with naming: {{doctype}}_{{background}}.jpg")
        print(f"Example: invoice_wood.jpg, receipt_white.jpg")
        return

    print(f"Found {len(images)} test images")
    print(f"Scanners: {', '.join(SCANNERS.keys())}")
    print()

    all_results = []
    for i, image_path in enumerate(images):
        print(f"[{i+1}/{len(images)}] {image_path.name}")
        result = evaluate_image(image_path, results_dir)
        all_results.append(result)

    print_summary(all_results)

    # Save raw results
    results_file = results_dir / "comparison_results.json"
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nDetailed results: {results_file}")
    print(f"Output scans:     {results_dir}/{{pagescan,docscan,opencv}}/")


if __name__ == "__main__":
    main()
