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
!git clone https://github.com/YOUR_USER/RetinaScan.git
%cd RetinaScan
```

### 3. Install core ML libraries
```bash
!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
!pip install open-clip-torch==2.24.0
```

### 4. Install image processing
```bash
!pip install opencv-python==4.9.0.80
!pip install scikit-image==0.23.2
!pip install Pillow==10.3.0
```

### 5. Install Kaggle CLI & download EyePACS

**Method A — Token auth (new, recommended):**
```python
# Paste your token here
import os
os.environ["KAGGLE_API_TOKEN"] = "KGAT_your_token_here"
!pip install kaggle>=2.0.0
!kaggle competitions download -c diabetic-retinopathy-detection -p data/raw/
!unzip -q "data/raw/diabetic-retinopathy-detection.zip" -d data/raw/
```

**Method B — File auth (legacy):**
```bash
!pip install kaggle==1.6.14
!mkdir -p ~/.kaggle
!cp /content/drive/MyDrive/kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
!kaggle competitions download -c diabetic-retinopathy-detection -p data/raw/
!unzip -q "data/raw/diabetic-retinopathy-detection.zip" -d data/raw/
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
```

### 8. Verify everything
```bash
!python -c "import torch; import open_clip; import cv2; print('All imports OK. Torch:', torch.__version__)"
```

---

## Run the Pipeline

### Option A: Pure Zero-Shot (no training needed)
```bash
# Preprocess images
!python src/preprocess.py --config configs/train_config.yaml

# Evaluate zero-shot — CLIP text vs image similarity, no training
!python src/evaluate/metrics.py --config configs/train_config.yaml
```

### Option B: Train Projection Head (better accuracy)
```bash
# Preprocess
!python src/preprocess.py --config configs/train_config.yaml

# Train (projection head only, ~2-3h on T4)
!python src/train.py --config configs/train_config.yaml

# Evaluate trained model
!python src/evaluate/metrics.py --config configs/train_config.yaml --checkpoint checkpoints/best.pt

# Grad-CAM on sample
!python src/evaluate/gradcam.py --config configs/train_config.yaml --checkpoint checkpoints/best.pt --image data/raw/sample.jpeg

# Export ONNX
!python deploy/export_onnx.py --config configs/train_config.yaml --checkpoint checkpoints/best.pt
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
