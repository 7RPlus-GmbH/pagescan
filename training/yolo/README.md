# YOLO11 document detector

Single-class detector trained as the first stage of the YOLO -> SAM cascade.

## Data

- Source: `~/pagescan-training/bundle_stage/training/{real_photos,real_labels}/` (1000 Dec photos, YOLO-OBB labels)
- 50 April photos under `training/labels_april/` are the held-out benchmark and **must not** be added to training.

## Pipeline

1. **Build the dataset** (local, fast):
   ```
   python training/yolo/build_dataset.py
   ```
   Output: `~/pagescan-training/yolo_dataset/` (900 train, 100 val, OBB->bbox converted, dataset.yaml written).

2. **Bundle for Colab**:
   ```
   cd ~/pagescan-training/yolo_dataset
   zip -r yolo_dataset.zip images labels dataset.yaml
   ```

3. **Train on Colab Pro** (T4 sufficient, A100 faster):
   ```
   !pip install -q ultralytics
   # upload yolo_dataset.zip, then:
   !unzip -q yolo_dataset.zip -d /content/yolo_dataset
   !python train_colab.py
   ```
   Expected: ~30–60 min on T4 for 80 epochs at imgsz=960.

4. **Pull `best.pt` / `best.onnx` back** to `data/model/yolo_doc_v1.{pt,onnx}` for the cascade inference script.

## Tunables in `train_colab.py`

- `MODEL`: `yolo11n.pt` (default) → bump to `yolo11s.pt` if mAP plateaus low
- `IMGSZ`: 960 default; lower to 640 for faster training / mobile-deploy parity
- `EPOCHS` + `PATIENCE`: 80 with early-stop at 20 epochs of no improvement

## What "good enough" looks like

This detector only needs to produce a bbox loose enough to contain the document.
Target: val mAP50 ≥ 0.95 on the held-out 100 photos. Tightness matters less than
recall; SAM will refine boundaries from a slightly loose bbox.
