# RetinaScan — Colab Terminal Setup Guide

## Prerequisites
Colab notebook → Runtime → Change runtime type → **T4 GPU**.

---

## Installation

### 1. Mount Drive & Check GPU
```python
from google.colab import drive
drive.mount('/content/drive')
!nvidia-smi
```

### 2. Clone Repo
```bash
!git clone https://github.com/xmadmaxdx/RetinaScan.git
%cd RetinaScan
```

### 3. Install Dependencies
```bash
!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
!pip install open-clip-torch==2.24.0
!pip install "numpy<2.0.0"
!pip install opencv-python==4.9.0.80 scikit-image==0.23.2 Pillow==10.3.0
!pip install tqdm pyyaml scikit-learn matplotlib seaborn gdown
!pip install onnx onnxruntime-gpu onnxscript
```

### 4. Download GDRBench Merged Data
```bash
!gdown 1ZJOEZ73OdWSG0YbFtgaH8hcE_NGfb8D8 -O gdrbench.zip
!mkdir -p data/gdrbench/images && unzip -qo gdrbench.zip -d data/gdrbench/images/
!python merge_datasets.py
```

### 5. Verify
```bash
!python -c "import torch; import open_clip; import cv2; print('OK', torch.__version__)"
```

---

## Pull Latest Code (after updates)
```bash
%cd RetinaScan
!git pull
```

---

## Training

### Fresh Training (50 epochs, ~6h)
```bash
!python src/train.py --config configs/train_config.yaml --drive-path /content/drive/MyDrive/RetinaScan/checkpoints
```

### Resume After Crash
Sync checkpoint from Drive, then resume:
```bash
!cp /content/drive/MyDrive/RetinaScan/checkpoints/latest.pt checkpoints/latest.pt
!python src/train.py --config configs/train_config.yaml --drive-path /content/drive/MyDrive/RetinaScan/checkpoints --resume
```

### Tweak Run (Rebalance + Extra Epochs)
After initial training completes, improve minority recall with rebalanced CORAL loss:
```bash
!cp /content/drive/MyDrive/RetinaScan/checkpoints/latest.pt checkpoints/latest.pt
!python src/train.py --config configs/train_config.yaml --drive-path /content/drive/MyDrive/RetinaScan/checkpoints --resume --tweak --num-epochs 5
```
- `--tweak` enables per-task pos_weight for ordinal loss (sqrt of class ratios)
- `--num-epochs N` trains N additional epochs from current point

---

## Evaluation

### Full Evaluation Report (Val)
```bash
!python src/evaluate/metrics.py --config configs/train_config.yaml --checkpoint checkpoints/final.pt --split val
```
Auto-detects KNN mode if checkpoint contains training features.

### Test Set (Unseen Holdout)
```bash
!python src/evaluate/metrics.py --config configs/train_config.yaml --checkpoint checkpoints/final.pt --split test
```

### With Threshold Tuning
```bash
!python src/evaluate/metrics.py --config configs/train_config.yaml --checkpoint checkpoints/best.pt --split val
```
Adds tuned threshold metrics with confusion matrix plots.

---

## Calibration

```bash
!python src/calibrate.py --config configs/train_config.yaml --checkpoint checkpoints/best.pt
```
Searches ordinal temperature (maximizes kappa) + prototype temperature (minimizes ECE).

---

## Final Pipeline (Multi-Mode Comparison)

Compare all inference strategies and pick the best:
```bash
!python src/evaluate/final_pipeline.py --config configs/train_config.yaml --checkpoints checkpoints/best.pt
```
Runs: baseline, calibrated+tuned, SWA, ensemble, KNN. Auto-saves `checkpoints/final.pt` if kappa improves.

With multiple checkpoints:
```bash
!python src/evaluate/final_pipeline.py --config configs/train_config.yaml --checkpoints checkpoints/best.pt checkpoints/latest.pt
```

---

## Other Tools

### Grad-CAM Heatmap
```bash
!python src/evaluate/gradcam.py --config configs/train_config.yaml --checkpoint checkpoints/best.pt --image sample.jpeg
```

### Export ONNX
```bash
!python deploy/export_onnx.py --config configs/train_config.yaml --checkpoint checkpoints/best.pt
```

### Zero-Shot Evaluation (No Training)
```bash
!python src/evaluate/metrics.py --config configs/train_config.yaml
```

---

## Sync Artifacts to Drive
```bash
!cp -r checkpoints /content/drive/MyDrive/RetinaScan/
!cp -r logs /content/drive/MyDrive/RetinaScan/
```
