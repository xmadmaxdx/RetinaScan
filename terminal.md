# RetinaScan — Colab Terminal Setup Guide

## Prerequisites
Open a **Colab** notebook → Runtime → Change runtime type → **T4 GPU**.

---

## Step-by-step Installation (paste in order)

### 1. Mount Google Drive & check GPU
```python
from google.colab import drive
drive.mount('/content/drive')
!nvidia-smi
```

### 2. Clone the repo
```bash
!git clone https://github.com/xmadmaxdx/RetinaScan.git
%cd RetinaScan
```

### 3. Install core ML libraries
```bash
!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
!pip install open-clip-torch==2.24.0
!pip install "numpy<2.0.0"  # wandb/timm compat with numpy 1.x
```

### 4. Install image processing
```bash
!pip install opencv-python==4.9.0.80
!pip install scikit-image==0.23.2
!pip install Pillow==10.3.0
```

### 5a. [Primary] Download GDRBench merged pack (~10 GB, 6 datasets)
Download from Google Drive and extract:
```bash
!pip install gdown -q
!gdown 1ZJOEZ73OdWSG0YbFtgaH8hcE_NGfb8D8 -O gdrbench.zip
!mkdir -p data/gdrbench/images && unzip -qo gdrbench.zip -d data/gdrbench/images/
!python merge_datasets.py
```
Expected output: ~108,000 images across all 6 sources.

### 5b. [Legacy] Single-dataset EyePACS only (HuggingFace, no auth)
```python
!pip install datasets -q
from datasets import load_dataset
ds = load_dataset("bumbledeep/eyepacs", split="train")
print(f"Loaded {len(ds)} images — already cropped and resized")
# Then switch config back: data.source = huggingface in train_config.yaml
```

### 6. Install training & evaluation utilities
```bash
!pip install tqdm==4.66.2
!pip install wandb==0.17.0
!pip install pyyaml==6.2
!pip install scikit-learn==1.4.2
!pip install matplotlib==3.8.4
!pip install seaborn==0.13.2
```

### 7. Install deployment tools
```bash
!pip install onnx==1.16.0
!pip install onnxruntime-gpu==1.17.1
!pip install onnxscript
```

### 8. Verify everything
```bash
!python -c "import torch; import open_clip; import cv2; print('All imports OK. Torch:', torch.__version__)"
```

---

## Choose Your Data Source

The config defaults to `source: merged` (GDRBench, 6 datasets, ~108k images).
To use **EyePACS only** (HuggingFace), edit `configs/train_config.yaml` and change:
```yaml
data:
  source: "huggingface"            # was "merged"
  hf_dataset: "bumbledeep/eyepacs"
```

Both code paths are fully supported — switch back any time.

---

## Run the Pipeline

### Option A: Pure Zero-Shot (no training needed)
```bash
# Evaluate zero-shot — CLIP text vs image similarity
# Ensure data is available first (Step 5a for merged, or 5b + config switch for HF)
!python src/evaluate/metrics.py --config configs/train_config.yaml
```

### Option B: Train Projection Head (better accuracy)

Mount Drive first:
```python
from google.colab import drive
drive.mount('/content/drive')
```

Train (~2.5h for 50 epochs on T4):
```bash
# batch_size must be divisible by 5 (set to 40 in config for balanced batches)
!python src/train.py --config configs/train_config.yaml --drive-path /content/drive/MyDrive/RetinaScan/checkpoints
```

Calibrate confidence scores (post-hoc temperature scaling):
```bash
!python src/calibrate.py --config configs/train_config.yaml --checkpoint checkpoints/best.pt
```

Evaluate trained model (val set, also tunes optimal thresholds):
```bash
!python src/evaluate/metrics.py --config configs/train_config.yaml --checkpoint checkpoints/best.pt --drive-path /content/drive/MyDrive/RetinaScan/logs
```
Final test set evaluation (after threshold tuning on val):
```bash
!python src/evaluate/metrics.py --config configs/train_config.yaml --checkpoint checkpoints/best.pt --split test
```

Grad-CAM on a sample:
```bash
!python src/evaluate/gradcam.py --config configs/train_config.yaml --checkpoint checkpoints/best.pt --image data/raw/sample.jpeg
```

Export ONNX:
```bash
!python deploy/export_onnx.py --config configs/train_config.yaml --checkpoint checkpoints/best.pt
```

**Resume after crash/disconnect:**
```bash
!cp /content/drive/MyDrive/RetinaScan/checkpoints/latest.pt checkpoints/latest.pt
!python src/train.py --config configs/train_config.yaml --drive-path /content/drive/MyDrive/RetinaScan/checkpoints --resume
```

---

## Download Artifacts to Drive
```bash
!cp -r checkpoints /content/drive/MyDrive/RetinaScan/
!cp -r logs /content/drive/MyDrive/RetinaScan/
!cp deploy/*.onnx /content/drive/MyDrive/RetinaScan/
```

---

## Hugging Face Spaces Quick Deploy
```bash
# After training, create a Space:
# 1. huggingface.co → New Space → Gradio SDK → free tier
# 2. Upload: app.py, requirements.txt, checkpoints/best.pt
# 3. HF auto-deploys. Set UptimeRobot to ping every 5 min.
```
