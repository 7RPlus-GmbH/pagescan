"""Full YOLO -> HQ-SAM cascade benchmark on the April 50-photo set.

Pipeline:
    photo --[YOLO11n doc-detect]--> bbox
         --[HQ-SAM ViT-B, box prompt]--> mask
         --[convex hull + approxPolyDP]--> quad

Compares predicted quads against training/labels_april/ ground truth, including
the cyclic-rotation alignment metric. The GT label here is *only* used for
evaluation; YOLO never sees the April photos.

Run:
    python benchmark/cascade/run.py
"""
from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import cv2
import numpy as np
import torch

from segment_anything_hq import SamPredictor, sam_model_registry
from ultralytics import YOLO

REPO = Path(__file__).resolve().parents[2]
SAM_TYPE = "vit_b"
SAM_CHECKPOINT = REPO / "data" / "model" / "sam_hq_vit_b.pth"
YOLO_CHECKPOINT = REPO / "data" / "model" / "yolo_doc_v1.pt"
YOLO_IMGSZ = 960
YOLO_CONF = 0.25  # generous; we only need the doc, false positives are rare for this dataset
IMAGES = REPO / "benchmark" / "comparison" / "images"
LABELS = REPO / "training" / "labels_april"
OUT = Path(__file__).parent / "out"
OUT.mkdir(parents=True, exist_ok=True)

IOU_PASS = 0.90
# Slack on the GT bbox (in pixels) so SAM has a little room.
# YOLO output won't be tighter than this anyway, so a small dilation is realistic.
BBOX_DILATE_FRAC = 0.02


def load_gt(label_path: Path, w: int, h: int) -> np.ndarray | None:
    if not label_path.exists():
        return None
    parts = label_path.read_text().strip().split()
    if len(parts) < 9:
        return None
    coords = np.array([float(x) for x in parts[1:9]], dtype=np.float32).reshape(4, 2)
    coords[:, 0] *= w
    coords[:, 1] *= h
    return coords


def order_corners(pts: np.ndarray) -> np.ndarray:
    """Spatial TL/TR/BR/BL — useful when the doc is roughly axis-aligned, but
    NOT a substitute for matching against semantic GT labels."""
    pts = pts.astype(np.float32).reshape(-1, 2)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()
    return np.array(
        [pts[np.argmin(s)], pts[np.argmin(d)], pts[np.argmax(s)], pts[np.argmax(d)]],
        dtype=np.float32,
    )


def best_cyclic_alignment(pred_quad: np.ndarray, gt_quad: np.ndarray) -> tuple[int, np.ndarray, np.ndarray]:
    """Find the cyclic rotation (and optional reflection) of pred that minimizes
    sum of per-corner distances to gt (gt is taken in its original label order).

    Returns (rotation_offset, pred_aligned, per_corner_dists).
    rotation_offset is in 0..3 for forward order, 4..7 for reflected order.
    """
    gt = gt_quad.astype(np.float32).reshape(4, 2)
    pred = pred_quad.astype(np.float32).reshape(4, 2)
    best_idx, best_total, best_aligned, best_dists = 0, float("inf"), pred, None
    for reflect in (False, True):
        base = pred[::-1] if reflect else pred
        for k in range(4):
            cand = np.roll(base, -k, axis=0)
            dists = np.linalg.norm(cand - gt, axis=1)
            total = dists.sum()
            if total < best_total:
                best_total = total
                best_idx = k + (4 if reflect else 0)
                best_aligned = cand
                best_dists = dists
    return best_idx, best_aligned, best_dists


def polygon_iou(pts1: np.ndarray, pts2: np.ndarray) -> float:
    pts1 = pts1.astype(np.float32).reshape(-1, 2)
    pts2 = pts2.astype(np.float32).reshape(-1, 2)
    h1 = cv2.convexHull(pts1)
    h2 = cv2.convexHull(pts2)
    a1 = cv2.contourArea(h1)
    a2 = cv2.contourArea(h2)
    if a1 < 1 or a2 < 1:
        return 0.0
    ret, inter = cv2.intersectConvexConvex(h1, h2)
    if ret <= 0 or inter is None or len(inter) < 3:
        return 0.0
    ai = cv2.contourArea(inter)
    union = a1 + a2 - ai
    return float(ai / union) if union >= 1 else 0.0


def quad_to_bbox(quad: np.ndarray, w: int, h: int, dilate_frac: float) -> np.ndarray:
    """Axis-aligned bbox around quad, optionally dilated by `dilate_frac` of img dims."""
    x_min, y_min = quad.min(axis=0)
    x_max, y_max = quad.max(axis=0)
    dx = dilate_frac * w
    dy = dilate_frac * h
    return np.array([
        max(0, x_min - dx),
        max(0, y_min - dy),
        min(w - 1, x_max + dx),
        min(h - 1, y_max + dy),
    ], dtype=np.float32)


def mask_to_quad(mask: np.ndarray) -> tuple[np.ndarray, str]:
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.zeros((4, 2), dtype=np.float32), "empty"
    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < 100:
        return np.zeros((4, 2), dtype=np.float32), "tiny"
    hull = cv2.convexHull(contour)
    perim = cv2.arcLength(hull, True)
    for eps_frac in (0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.12):
        approx = cv2.approxPolyDP(hull, eps_frac * perim, True)
        if len(approx) == 4:
            return approx.reshape(4, 2).astype(np.float32), f"approxPolyDP@{eps_frac}"
    rect = cv2.minAreaRect(hull)
    box = cv2.boxPoints(rect).astype(np.float32)
    return box, "minAreaRect"


CORNER_NAMES = ["TL", "TR", "BR", "BL"]  # GT label order: clockwise from paper's TL


def _label_corners(img, pts, names, color, *, side: str):
    """Draw small labeled circles at each corner, offsetting text away from polygon."""
    pts_i = pts.astype(int)
    cy = pts_i[:, 1].mean()
    for p, name in zip(pts_i, names):
        cv2.circle(img, tuple(p), 12, color, -1, cv2.LINE_AA)
        cv2.circle(img, tuple(p), 12, (0, 0, 0), 2, cv2.LINE_AA)
        # Offset text up if corner is in upper half, down otherwise; left/right by side.
        dy = -22 if p[1] < cy else 36
        dx = 18 if side == "right" else -55
        tx, ty = int(p[0]) + dx, int(p[1]) + dy
        cv2.putText(img, name, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 5, cv2.LINE_AA)
        cv2.putText(img, name, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)


def draw_overlay(img, pred_aligned, gt, bbox, iou, fit_method, rot_offset, per_corner_dists):
    """Draw GT and pred polygons. Both are labeled in the SAME logical order
    (GT in its original label order; pred reordered by best_cyclic_alignment),
    so a green TL should sit near a yellow TL when rotation is correct."""
    out = img.copy()
    x1, y1, x2, y2 = bbox.astype(int)
    cv2.rectangle(out, (x1, y1), (x2, y2), (255, 200, 0), 3, cv2.LINE_AA)
    if gt is not None:
        cv2.polylines(out, [gt.astype(int)], True, (0, 255, 255), 6, cv2.LINE_AA)
        _label_corners(out, gt, CORNER_NAMES, (0, 255, 255), side="left")
    cv2.polylines(out, [pred_aligned.astype(int)], True, (0, 255, 0), 4, cv2.LINE_AA)
    _label_corners(out, pred_aligned, CORNER_NAMES, (0, 255, 0), side="right")

    rot_flag = "OK" if rot_offset == 0 else f"ROT={rot_offset}"
    max_d = float(per_corner_dists.max()) if per_corner_dists is not None else 0.0
    lines = [
        f"IoU {iou:.3f}   fit={fit_method}",
        f"corner-rot={rot_flag}   max_corner_px={max_d:.0f}",
    ]
    for i, line in enumerate(lines):
        y = 50 + i * 50
        cv2.putText(out, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 6, cv2.LINE_AA)
        col = (0, 255, 0) if i == 0 else ((0, 255, 0) if rot_offset == 0 else (0, 0, 255))
        cv2.putText(out, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 1.3, col, 2, cv2.LINE_AA)
    return out


def yolo_predict_bbox(yolo_model, img_bgr, conf: float, imgsz: int) -> tuple[np.ndarray | None, float, float]:
    """Run YOLO on a BGR image. Return (xyxy bbox, conf, inference_ms) or (None, 0, ms) if no detection.
    If multiple detections, pick the highest-confidence one (single-doc-per-photo assumption).
    """
    t0 = time.perf_counter()
    res = yolo_model.predict(img_bgr, imgsz=imgsz, conf=conf, verbose=False)[0]
    dt_ms = (time.perf_counter() - t0) * 1000
    if res.boxes is None or len(res.boxes) == 0:
        return None, 0.0, dt_ms
    confs = res.boxes.conf.cpu().numpy()
    idx = int(np.argmax(confs))
    xyxy = res.boxes.xyxy[idx].cpu().numpy().astype(np.float32)
    return xyxy, float(confs[idx]), dt_ms


def main() -> None:
    assert SAM_CHECKPOINT.exists(), f"missing: {SAM_CHECKPOINT}"
    assert YOLO_CHECKPOINT.exists(), f"missing: {YOLO_CHECKPOINT} — train via training/yolo first"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"loading HQ-SAM {SAM_TYPE} on {device}...")
    sam = sam_model_registry[SAM_TYPE](checkpoint=None)
    state = torch.load(str(SAM_CHECKPOINT), map_location=device, weights_only=True)
    sam.load_state_dict(state)
    sam.to(device).eval()
    predictor = SamPredictor(sam)

    print(f"loading YOLO from {YOLO_CHECKPOINT.name}...")
    yolo_model = YOLO(str(YOLO_CHECKPOINT))

    paths = sorted(IMAGES.glob("*.jpg"))
    print(f"running {len(paths)} images (YOLO -> HQ-SAM cascade)")

    rows = []
    for p in paths:
        img = cv2.imread(str(p))
        if img is None:
            continue
        h, w = img.shape[:2]
        gt = load_gt(LABELS / f"{p.stem}.txt", w, h)
        if gt is None:
            continue

        bbox, conf, yolo_ms = yolo_predict_bbox(yolo_model, img, YOLO_CONF, YOLO_IMGSZ)
        if bbox is None:
            print(f"  ! {p.name}: YOLO produced no detection (skipping)")
            rows.append({
                "file": p.name, "image_size": [w, h], "iou": 0.0,
                "mean_corner_px": float("nan"), "max_corner_px": float("nan"),
                "rotation_offset": 0, "per_corner_dists": [],
                "fit_method": "no_yolo_detection",
                "yolo_detected": False, "yolo_conf": 0.0, "yolo_ms": round(yolo_ms, 1),
                "embed_ms": 0, "predict_ms": 0,
                "gt_corners": gt.tolist(),
            })
            continue

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        t0 = time.perf_counter()
        predictor.set_image(rgb)
        embed_ms = (time.perf_counter() - t0) * 1000

        t1 = time.perf_counter()
        with torch.no_grad():
            masks, scores, _ = predictor.predict(
                box=bbox,
                multimask_output=False,
                hq_token_only=True,
            )
        predict_ms = (time.perf_counter() - t1) * 1000

        mask = masks[0]
        mask_score = float(scores[0])
        mask_area_frac = float(mask.sum()) / (w * h)

        pred, fit_method = mask_to_quad(mask)
        iou = polygon_iou(pred, gt)
        rot_offset, pred_aligned, per_corner_dists = best_cyclic_alignment(pred, gt)
        mean_corner_px = float(per_corner_dists.mean())
        max_corner_px = float(per_corner_dists.max())

        overlay = draw_overlay(img, pred_aligned, gt, bbox, iou, fit_method,
                               rot_offset, per_corner_dists)
        cv2.imwrite(str(OUT / f"{p.stem}_overlay.jpg"), overlay, [cv2.IMWRITE_JPEG_QUALITY, 85])

        flag = "+" if iou >= IOU_PASS else "-"
        rot_marker = "  " if rot_offset == 0 else f"R{rot_offset}"
        print(
            f"  {flag} {p.name:35s} iou={iou:.3f} {rot_marker} "
            f"corner mean={mean_corner_px:5.0f} max={max_corner_px:5.0f}px  "
            f"yolo conf={conf:.2f} {yolo_ms:4.0f}ms  "
            f"sam embed={embed_ms:5.0f}ms predict={predict_ms:5.0f}ms"
        )

        rows.append({
            "file": p.name,
            "image_size": [w, h],
            "iou": round(iou, 4),
            "mean_corner_px": round(mean_corner_px, 2),
            "max_corner_px": round(max_corner_px, 2),
            "rotation_offset": rot_offset,
            "per_corner_dists": [round(float(x), 2) for x in per_corner_dists],
            "fit_method": fit_method,
            "mask_score": round(mask_score, 4),
            "mask_area_frac": round(mask_area_frac, 4),
            "yolo_detected": True,
            "yolo_conf": round(conf, 4),
            "yolo_ms": round(yolo_ms, 1),
            "embed_ms": round(embed_ms, 1),
            "predict_ms": round(predict_ms, 1),
            "bbox": bbox.tolist(),
            "pred_corners_raw": pred.tolist(),
            "pred_corners_aligned": pred_aligned.tolist(),
            "gt_corners": gt.tolist(),
        })

    summary_path = Path(__file__).parent / "summary.json"
    summary_path.write_text(json.dumps(rows, indent=2))

    detected_rows = [r for r in rows if r.get("yolo_detected", True)]
    ious = [r["iou"] for r in rows]  # missed detections count as IoU=0
    max_pxs = [r["max_corner_px"] for r in detected_rows
               if r["max_corner_px"] == r["max_corner_px"]]  # NaN-filter
    rotated = [r for r in detected_rows if r["rotation_offset"] != 0]
    print()
    print(f"images:                  {len(rows)}")
    print(f"yolo missed detection:   {len(rows) - len(detected_rows)}")
    for thr in (0.95, 0.90, 0.85, 0.80, 0.75, 0.50):
        n = sum(1 for x in ious if x >= thr)
        print(f"pass (IoU >= {thr:.2f}):    {n}/{len(rows)}")
    print(f"mean IoU:                {np.mean(ious):.3f}")
    print(f"median IoU:              {np.median(ious):.3f}")
    if max_pxs:
        print(f"mean max-corner-px:      {np.mean(max_pxs):.0f}")
        print(f"median max-corner-px:    {np.median(max_pxs):.0f}")
    print(f"rotation mismatches:     {len(rotated)}/{len(detected_rows)} (offset != 0)")
    if rotated:
        print("  rotation breakdown:", {k: sum(1 for r in rotated if r['rotation_offset'] == k)
                                         for k in sorted({r['rotation_offset'] for r in rotated})})
    if detected_rows:
        yolo_ms = [r["yolo_ms"] for r in detected_rows]
        print(f"mean yolo ms:            {np.mean(yolo_ms):.0f}")
    print(f"summary:                 {summary_path}")


if __name__ == "__main__":
    main()
