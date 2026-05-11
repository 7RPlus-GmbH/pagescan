"""Build a YOLO-detect dataset for the document detector.

Source (relative to repo root, gitignored — see training/data/README.md):
    images: training/data/dec_1000_images/*.jpg   (1000)
    labels: training/labels_3k/*.txt              (1000, tracked in git)

Each source label is YOLO-OBB:  class x1 y1 x2 y2 x3 y3 x4 y4   (normalized)

Output (default training/data/yolo_build/):
    yolo_build/
        images/{train,val}/*.jpg     (symlinked from source)
        labels/{train,val}/*.txt     (YOLO-detect: class cx cy w h, normalized)
        dataset.yaml                 (Ultralytics config)

The conversion takes the axis-aligned bounding rect of the 4-corner quad,
which matches what we feed SAM as a box prompt at inference time.

Usage:
    python training/yolo/build_dataset.py
    python training/yolo/build_dataset.py --val-frac 0.1 --seed 42
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SRC_IMAGES = REPO / "training" / "data" / "dec_1000_images"
SRC_LABELS = REPO / "training" / "labels_3k"
DEFAULT_OUT = REPO / "training" / "data" / "yolo_build"


def obb_to_bbox(parts: list[str]) -> tuple[int, float, float, float, float] | None:
    """Convert YOLO-OBB line tokens to (class, cx, cy, w, h) normalized bbox.

    Returns None if the line is malformed or produces a degenerate bbox.
    """
    if len(parts) < 9:
        return None
    try:
        cls = int(float(parts[0]))
        coords = [float(x) for x in parts[1:9]]
    except ValueError:
        return None
    xs = coords[0::2]
    ys = coords[1::2]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    # clamp to [0, 1] in case labels nudged slightly out
    x_min, x_max = max(0.0, x_min), min(1.0, x_max)
    y_min, y_max = max(0.0, y_min), min(1.0, y_max)
    w = x_max - x_min
    h = y_max - y_min
    if w <= 0.001 or h <= 0.001:
        return None
    cx = x_min + w / 2
    cy = y_min + h / 2
    return cls, cx, cy, w, h


def build(out_dir: Path, val_frac: float, seed: int) -> dict:
    assert SRC_IMAGES.exists(), f"missing: {SRC_IMAGES}"
    assert SRC_LABELS.exists(), f"missing: {SRC_LABELS}"

    pairs = []
    for label_path in sorted(SRC_LABELS.glob("*.txt")):
        img_path = SRC_IMAGES / f"{label_path.stem}.jpg"
        if not img_path.exists():
            continue
        pairs.append((img_path, label_path))

    rng = random.Random(seed)
    rng.shuffle(pairs)
    n_val = max(1, int(round(len(pairs) * val_frac)))
    val_pairs = pairs[:n_val]
    train_pairs = pairs[n_val:]

    for split in ("train", "val"):
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    n_dropped = 0
    for split, items in (("train", train_pairs), ("val", val_pairs)):
        for img_path, label_path in items:
            # symlink image (avoid copying ~10 GB)
            link = out_dir / "images" / split / img_path.name
            if link.is_symlink() or link.exists():
                link.unlink()
            link.symlink_to(img_path)

            # convert label
            out_label = out_dir / "labels" / split / label_path.name
            lines_out = []
            for line in label_path.read_text().splitlines():
                parts = line.strip().split()
                if not parts:
                    continue
                conv = obb_to_bbox(parts)
                if conv is None:
                    n_dropped += 1
                    continue
                cls, cx, cy, w, h = conv
                lines_out.append(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
            out_label.write_text("\n".join(lines_out) + "\n")

    yaml_path = out_dir / "dataset.yaml"
    yaml_path.write_text(
        f"path: {out_dir}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"names:\n"
        f"  0: document\n"
    )

    return {
        "out_dir": str(out_dir),
        "n_train": len(train_pairs),
        "n_val": len(val_pairs),
        "n_total": len(pairs),
        "n_dropped_lines": n_dropped,
        "yaml": str(yaml_path),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--val-frac", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    info = build(args.out, args.val_frac, args.seed)
    print(f"output:      {info['out_dir']}")
    print(f"train:       {info['n_train']}")
    print(f"val:         {info['n_val']}")
    print(f"total pairs: {info['n_total']}")
    print(f"dropped:     {info['n_dropped_lines']} malformed/degenerate label lines")
    print(f"yaml:        {info['yaml']}")


if __name__ == "__main__":
    main()
