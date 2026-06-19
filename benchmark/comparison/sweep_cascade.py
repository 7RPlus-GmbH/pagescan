"""use_cascade sweep: cascade vs legacy on the sealed April hold-out set.

Answers the single release-gating question: does the YOLO+HQ-SAM cascade
(`use_cascade=True`, the current default) actually beat the legacy SA24+LCNet
chain (`use_cascade=False`) on the 50-photo benchmark — and therefore which
value should `ScanConfig.use_cascade` default to before tagging 0.1.0?

Reuses the comparison harness (metrics, scanners, evaluation) from
``run_real.py`` and runs only the two pagescan paths, then prints a verdict
plus the exact one-line change to flip the default if legacy wins.

Run:
    python benchmark/comparison/sweep_cascade.py [--limit N] [--text]
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[2]
for p in (REPO, REPO / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from benchmark.comparison.metrics import ocr, perspective_warp
from benchmark.comparison.run_real import (
    IMAGES_DIR,
    LABELS_DIR,
    OUT_DIR,
    aggregate,
    evaluate_image,
    load_gt_quad,
)
from benchmark.comparison.scanners import PagescanCascade, PagescanLegacy

# IoU threshold that defines a "pass" — the headline 0.1.0 quality bar.
PASS_THRESHOLD = 0.90
MARGIN = 1  # how many more passes the cascade must win by to keep being default


def run(limit: int | None, compute_text: bool) -> dict:
    scanners = [PagescanCascade(), PagescanLegacy()]
    paths = sorted(IMAGES_DIR.glob("*.jpg"))
    if limit is not None:
        paths = paths[:limit]
    if not paths:
        raise SystemExit(f"no images in {IMAGES_DIR} — the sealed hold-out set is missing")

    print(f"sweep: {[s.name for s in scanners]} on {len(paths)} images "
          f"(text metrics: {'on' if compute_text else 'off'})\n")

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
        gt_text = ocr(gt_warped) if compute_text else ""

        line = f"  {img_path.name:34s}"
        for scanner in scanners:
            metrics, *_ = evaluate_image(
                scanner, image, gt_quad, gt_warped, gt_text, compute_text=compute_text,
            )
            rows.append({"file": img_path.name, "scanner": scanner.name, **metrics})
            iou = metrics["iou"]
            line += f"  {scanner.name}={'—' if np.isnan(iou) else f'{iou:.3f}'}"
        print(line)

    return aggregate(rows)


def verdict(agg: dict) -> None:
    cas = agg.get("pagescan-cascade")
    leg = agg.get("pagescan-legacy")
    if not cas or not leg:
        print("\nverdict: could not run both scanners — check weights/deps.")
        return

    n = cas["n_images"]
    c_pass = cas["pass_iou_0_90"]
    l_pass = leg["pass_iou_0_90"]

    print("\n" + "=" * 64)
    print(f"use_cascade sweep — pass @ IoU >= {PASS_THRESHOLD:.2f}  (n={n})")
    print("-" * 64)
    print(f"  cascade (use_cascade=True) : {c_pass}/{n}   "
          f"mean IoU {cas['iou_mean']}   {cas['latency_mean_ms']:.0f} ms/img")
    print(f"  legacy  (use_cascade=False): {l_pass}/{n}   "
          f"mean IoU {leg['iou_mean']}   {leg['latency_mean_ms']:.0f} ms/img")
    print(f"  also pass @ .85 — cascade {cas['pass_iou_0_85']}/{n}, "
          f"legacy {leg['pass_iou_0_85']}/{n}")
    print("-" * 64)

    if c_pass >= l_pass + MARGIN:
        print("  VERDICT: cascade wins -> KEEP ScanConfig.use_cascade = True (default).")
    elif l_pass >= c_pass + MARGIN:
        print("  VERDICT: legacy wins -> FLIP the default to legacy until v2 YOLO lands:")
        print("           in src/pagescan/config.py set  use_cascade: bool = False")
        print("           (and note it in CHANGELOG.md).")
    else:
        print(f"  VERDICT: within {MARGIN} of a tie — cascade ahead on quality keeps the")
        print("           default at True; if latency matters, prefer legacy. Judgement call.")
    print("=" * 64)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--limit", type=int, default=None,
                    help="Process only the first N images (quick smoke run).")
    ap.add_argument("--text", action="store_true",
                    help="Also compute OCR/WER (slower; not needed for the use_cascade decision).")
    args = ap.parse_args()

    agg = run(args.limit, compute_text=args.text)
    out = OUT_DIR / "sweep_cascade.json"
    out.write_text(json.dumps(agg, indent=2))
    verdict(agg)
    print(f"\nsummary: {out}")


if __name__ == "__main__":
    main()
