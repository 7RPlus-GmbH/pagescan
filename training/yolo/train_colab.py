"""Train a YOLO11n document detector on Colab Pro (T4/A100).

Steps:
1. Bundle the dataset locally:
       cd ~/pagescan-training/yolo_dataset
       zip -r yolo_dataset.zip images labels dataset.yaml
   (Or rsync to a Colab-mounted Drive folder.)

2. In a fresh Colab notebook with GPU, run:
       !pip install -q ultralytics
       # upload yolo_dataset.zip then:
       !unzip -q yolo_dataset.zip -d /content/yolo_dataset
       !python train_colab.py

3. Outputs land in runs/detect/<run_name>/. The relevant files:
       weights/best.pt   <- use this for inference
       weights/best.onnx <- if you exported (see EXPORT_ONNX)
       results.png       <- training curves
       val_batch*.jpg    <- val-set predictions

Tunables are at the top of the file.
"""
from __future__ import annotations

from pathlib import Path

# ---------------- config ----------------
DATA_YAML = Path("/content/yolo_dataset/dataset.yaml")
MODEL = "yolo11n.pt"        # nano (~2.6M params); bump to yolo11s.pt for more headroom
EPOCHS = 80
IMGSZ = 960                 # docs are large in frame; 960 keeps text edges crisp
BATCH = 16                  # T4-safe; raise to 32 on A100
RUN_NAME = "doc_detect_yolo11n"
EXPORT_ONNX = True
PATIENCE = 20               # early-stop if val mAP stagnates


def main() -> None:
    from ultralytics import YOLO

    assert DATA_YAML.exists(), f"missing: {DATA_YAML} — did you unzip the dataset?"

    model = YOLO(MODEL)

    model.train(
        data=str(DATA_YAML),
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        patience=PATIENCE,
        name=RUN_NAME,
        # Augmentations: docs are rotated/photographed in varied conditions,
        # so default coco-tuned augs are mostly fine. Tighten the hue jitter
        # since docs are mostly grayscale/cream and we don't want hue drift.
        hsv_h=0.010,
        hsv_s=0.30,
        hsv_v=0.30,
        # No mosaic in last 10 epochs so val mAP isn't noisy at the end.
        close_mosaic=10,
        # We have a single class, so increase the obj loss weight slightly.
        cls=0.6,
        box=7.5,
        # Save best-by-mAP automatically (default).
    )

    metrics = model.val()
    print(f"final val mAP50:    {metrics.box.map50:.4f}")
    print(f"final val mAP50-95: {metrics.box.map:.4f}")

    if EXPORT_ONNX:
        # opset=12 is broadly compatible with onnxruntime/mobile runtimes
        onnx_path = model.export(format="onnx", imgsz=IMGSZ, opset=12, simplify=True)
        print(f"onnx export:        {onnx_path}")


if __name__ == "__main__":
    main()
