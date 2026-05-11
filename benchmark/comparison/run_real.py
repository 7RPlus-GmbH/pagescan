"""Real benchmark: pagescan vs docscan vs OpenCV recipe on 50 April photos.

Measures, per scanner, per image:
    - Detection IoU vs labelled GT quad
    - Mean / max corner pixel error (under best cyclic alignment)
    - Detection latency (wall clock)
    - PSNR of scanner-warped vs GT-warped image (output quality proxy)
    - OCR word-error rate of scanner-warped vs GT-warped (text-fidelity proxy)

GT corners come from training/labels_april/*.txt (YOLO-OBB normalised).
The "GT-warped" reference is the perspective-corrected image using GT
corners — not a hand-scanned reference, but a fair upper bound for the
output-quality and OCR proxies.

Outputs:
    benchmark/comparison/results/<scanner>/*_overlay.jpg     visual diff
    benchmark/comparison/results/<scanner>/*_warped.jpg      warped output
    benchmark/comparison/results/summary.json                full per-image data
    benchmark/comparison/results/table.md                    drop-in README table

Run:
    python benchmark/comparison/run_real.py [--scanners pagescan-cascade,opencv-recipe]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import cv2
import numpy as np

# Make this file runnable from the repo root with no install needed.
REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from benchmark.comparison.metrics import (
    align_quad,
    ocr,
    perspective_warp,
    polygon_iou,
    psnr,
    wer,
)
from benchmark.comparison.scanners import Scanner, all_scanners

IMAGES_DIR = REPO / "benchmark" / "comparison" / "images"
LABELS_DIR = REPO / "training" / "labels_april"
OUT_DIR = REPO / "benchmark" / "comparison" / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_gt_quad(label_path: Path, w: int, h: int) -> np.ndarray | None:
    if not label_path.exists():
        return None
    parts = label_path.read_text().strip().split()
    if len(parts) < 9:
        return None
    coords = np.array([float(x) for x in parts[1:9]], dtype=np.float32).reshape(4, 2)
    coords[:, 0] *= w
    coords[:, 1] *= h
    return coords


def draw_overlay(img: np.ndarray,
                 pred: np.ndarray | None,
                 pred_aligned: np.ndarray | None,
                 gt: np.ndarray,
                 metrics: dict) -> np.ndarray:
    out = img.copy()
    cv2.polylines(out, [gt.astype(int)], True, (0, 255, 255), 6, cv2.LINE_AA)
    if pred is not None:
        cv2.polylines(out, [pred.astype(int)], True, (0, 255, 0), 4, cv2.LINE_AA)
    text_lines = [
        f"IoU {metrics.get('iou', 0):.3f}   max_corner {metrics.get('max_corner_px', float('nan')):.0f} px",
        f"PSNR {metrics.get('psnr', float('nan')):.1f} dB   WER {metrics.get('wer', float('nan')):.2f}",
        f"latency {metrics.get('latency_ms', 0):.0f} ms",
    ]
    for i, line in enumerate(text_lines):
        y = 50 + i * 50
        cv2.putText(out, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 6, cv2.LINE_AA)
        cv2.putText(out, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 2, cv2.LINE_AA)
    return out


def evaluate_image(scanner: Scanner,
                   image: np.ndarray,
                   gt_quad: np.ndarray,
                   gt_warped: np.ndarray,
                   gt_text: str,
                   *,
                   compute_text: bool = True) -> dict:
    """Run a scanner on a single image and compute every metric we have."""
    h, w = image.shape[:2]

    t0 = time.perf_counter()
    pred = scanner.detect(image)
    detect_ms = (time.perf_counter() - t0) * 1000

    metrics: dict = {
        "success": pred is not None,
        "latency_ms": round(detect_ms, 1),
        "iou": float("nan"),
        "mean_corner_px": float("nan"),
        "max_corner_px": float("nan"),
        "rotation_offset": -1,
        "psnr": float("nan"),
        "wer": float("nan"),
    }

    pred_aligned = None
    if pred is not None and pred.shape == (4, 2):
        metrics["iou"] = round(polygon_iou(pred, gt_quad), 4)
        rot, pred_aligned, dists = align_quad(pred, gt_quad)
        metrics["rotation_offset"] = int(rot)
        metrics["mean_corner_px"] = round(float(dists.mean()), 2)
        metrics["max_corner_px"] = round(float(dists.max()), 2)

    # Output-quality metrics need a warped image. Scanners that don't expose
    # corners (docscan) override `warp`; others fall back to detect→warp.
    t1 = time.perf_counter()
    warped = scanner.warp(image)
    metrics["warp_ms"] = round((time.perf_counter() - t1) * 1000, 1)

    if warped is not None:
        try:
            metrics["psnr"] = round(
                psnr(gt_warped, warped, target_size=(gt_warped.shape[1], gt_warped.shape[0])), 2
            )
        except Exception:
            pass
        if compute_text:
            text = ocr(warped)
            metrics["wer"] = round(wer(gt_text, text), 4)

    return metrics, pred, pred_aligned, warped


def aggregate(rows: list[dict]) -> dict:
    """Per-scanner rollup: pass rates, central tendencies, latencies."""
    agg: dict = {}
    for scanner_name in {r["scanner"] for r in rows}:
        sub = [r for r in rows if r["scanner"] == scanner_name]
        n = len(sub)
        ious = [r["iou"] for r in sub if not np.isnan(r["iou"])]
        max_pxs = [r["max_corner_px"] for r in sub if not np.isnan(r["max_corner_px"])]
        psnrs = [r["psnr"] for r in sub if not np.isnan(r["psnr"]) and r["psnr"] != float("inf")]
        wers = [r["wer"] for r in sub if not np.isnan(r["wer"]) and r["wer"] != float("inf")]
        latencies = [r["latency_ms"] for r in sub]
        successes = sum(1 for r in sub if r["success"])

        agg[scanner_name] = {
            "n_images": n,
            "n_with_quad": successes,
            "success_rate": round(successes / n, 3) if n else 0,
            "iou_mean": round(float(np.mean(ious)), 3) if ious else None,
            "iou_median": round(float(np.median(ious)), 3) if ious else None,
            "pass_iou_0_85": int(sum(1 for x in ious if x >= 0.85)),
            "pass_iou_0_90": int(sum(1 for x in ious if x >= 0.90)),
            "pass_iou_0_95": int(sum(1 for x in ious if x >= 0.95)),
            "max_corner_mean": round(float(np.mean(max_pxs)), 1) if max_pxs else None,
            "max_corner_median": round(float(np.median(max_pxs)), 1) if max_pxs else None,
            "psnr_mean": round(float(np.mean(psnrs)), 2) if psnrs else None,
            "wer_mean": round(float(np.mean(wers)), 3) if wers else None,
            "latency_mean_ms": round(float(np.mean(latencies)), 1),
            "latency_median_ms": round(float(np.median(latencies)), 1),
        }
    return agg


def render_markdown_table(agg: dict, scanner_order: list[str]) -> str:
    """Drop-in markdown table for the README."""
    headers = ["Scanner", "Pass≥.90/50", "Pass≥.85/50", "mean IoU",
               "median max-px", "mean PSNR", "mean WER", "mean latency"]
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] * len(headers)) + "|",
    ]
    for name in scanner_order:
        a = agg.get(name)
        if a is None:
            continue
        n = a["n_images"]
        cells = [
            f"**{name}**" if name == "pagescan-cascade" else name,
            f"{a['pass_iou_0_90']}/{n}",
            f"{a['pass_iou_0_85']}/{n}",
            f"{a['iou_mean']:.3f}" if a["iou_mean"] is not None else "—",
            f"{a['max_corner_median']:.0f} px" if a["max_corner_median"] is not None else "—",
            f"{a['psnr_mean']:.1f} dB" if a["psnr_mean"] is not None else "—",
            f"{a['wer_mean']:.2f}" if a["wer_mean"] is not None else "—",
            f"{a['latency_mean_ms']:.0f} ms",
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--scanners", default=None,
                    help="Comma-separated scanner names. Default: all available.")
    ap.add_argument("--limit", type=int, default=None,
                    help="Process only the first N images (for quick smoke runs).")
    ap.add_argument("--no-text", action="store_true",
                    help="Skip OCR/WER computation (Tesseract may be slow).")
    ap.add_argument("--no-overlays", action="store_true",
                    help="Skip writing overlay JPGs (faster, smaller output).")
    args = ap.parse_args()

    paths = sorted(IMAGES_DIR.glob("*.jpg"))
    if args.limit is not None:
        paths = paths[: args.limit]

    scanners = all_scanners()
    if args.scanners:
        wanted = {s.strip() for s in args.scanners.split(",") if s.strip()}
        scanners = [s for s in scanners if s.name in wanted]
    print(f"running {len(scanners)} scanners on {len(paths)} images: "
          f"{[s.name for s in scanners]}")
    print(f"text metrics: {'on' if not args.no_text else 'off'}")
    print()

    rows: list[dict] = []
    for img_path in paths:
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"  ! unreadable: {img_path.name}")
            continue
        h, w = image.shape[:2]
        gt_quad = load_gt_quad(LABELS_DIR / f"{img_path.stem}.txt", w, h)
        if gt_quad is None:
            print(f"  ! no GT label for {img_path.name}; skipping")
            continue

        gt_warped = perspective_warp(image, gt_quad)
        gt_text = ocr(gt_warped) if not args.no_text else ""

        for scanner in scanners:
            scanner_dir = OUT_DIR / scanner.name
            scanner_dir.mkdir(parents=True, exist_ok=True)

            metrics, pred, pred_aligned, warped = evaluate_image(
                scanner, image, gt_quad, gt_warped, gt_text,
                compute_text=not args.no_text,
            )

            if not args.no_overlays:
                overlay = draw_overlay(image, pred, pred_aligned, gt_quad, metrics)
                cv2.imwrite(
                    str(scanner_dir / f"{img_path.stem}_overlay.jpg"),
                    overlay, [cv2.IMWRITE_JPEG_QUALITY, 80],
                )
                if warped is not None:
                    cv2.imwrite(
                        str(scanner_dir / f"{img_path.stem}_warped.jpg"),
                        warped, [cv2.IMWRITE_JPEG_QUALITY, 85],
                    )

            row = {"file": img_path.name, "scanner": scanner.name, **metrics}
            rows.append(row)
            iou_str = "—" if np.isnan(row["iou"]) else f"{row['iou']:.3f}"
            mc = "—" if np.isnan(row["max_corner_px"]) else f"{row['max_corner_px']:.0f}px"
            psnr_s = "—" if np.isnan(row["psnr"]) else f"{row['psnr']:.1f}dB"
            wer_s = "—" if np.isnan(row["wer"]) else f"{row['wer']:.2f}"
            print(
                f"  {img_path.name:35s} {scanner.name:18s} "
                f"iou={iou_str:6s} maxc={mc:7s} "
                f"psnr={psnr_s:8s} wer={wer_s:5s} "
                f"lat={row['latency_ms']:6.0f}ms"
            )

    agg = aggregate(rows)
    summary_path = OUT_DIR / "summary.json"
    summary_path.write_text(json.dumps({"per_image": rows, "aggregate": agg}, indent=2))

    scanner_order = [s.name for s in scanners]
    table = render_markdown_table(agg, scanner_order)
    table_path = OUT_DIR / "table.md"
    table_path.write_text(
        "# pagescan benchmark — April 50-photo set\n\n"
        f"v1 weights, {len(rows) // max(1, len(scanners))} images per scanner.\n\n"
        + table
    )

    print()
    print("=" * 70)
    print(table)
    print(f"summary: {summary_path}")
    print(f"table:   {table_path}")


if __name__ == "__main__":
    main()
