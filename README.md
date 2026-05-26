# RetinaScan — Zero-Shot Diabetic Retinopathy Grading

[![Hugging Face](https://img.shields.io/badge/HF-Space-blue)](https://huggingface.co/spaces/YOUR_USER/retinascan)
[![Colab](https://img.shields.io/badge/Open%20in-Colab-orange)](https://colab.research.google.com/github/YOUR_USER/RetinaScan/blob/main/notebooks/Train.ipynb)

Grades retinopathy severity (Grade 0–4) **without requiring labeled retina data at training time**, using **CLIP text-guided visual prototypes**.

## The Problem

> 80% of diabetic patients in low-income countries never receive retina screening due to specialist scarcity. Existing graders need thousands of labeled images per clinic.

## Impact

A mobile-deployable model that outputs Grade 0-4 severity with heatmaps, allowing non-specialists to screen patients in **under 2 seconds per image**.

## Architecture

```
CLIP Text Encoder (frozen)           CLIP Image Encoder (frozen)
         │                                    │
         ▼                                    ▼
  Severity Descriptions ──► text     retina fundus ──► image
  (Grade 0-4 clinical        features     image        features
   language)                    │                        │
                                 ▼                        ▼
                      ┌──────────────────────────────────┐
                      │     Shared Projection Head       │
                      │     (only trainable part)        │
                      └────────────┬─────────────────────┘
                                   │
                    ┌──────────────┴──────────────┐
                    ▼                              ▼
           Cosine Similarity               Ordinal Regression
           + Temperature Scaling           (CORAL head, 4 tasks)
                    │                              │
                    ▼                              ▼
           Interpretable Similarities       Clinically-grounded
           to each severity text            Grade 0-4 prediction
                    │                              │
                    └──────────┬───────────────────┘
                               ▼
                    Final Grade + Grad-CAM Heatmap
```

### Text-Derived Prototypes (Interpretability)

Instead of learning prototypes from labeled images, we encode **clinically accurate severity descriptions** through CLIP's text encoder. These text embeddings serve as fixed, interpretable anchors:

| Grade | Text Prototype |
|-------|----------------|
| 0 | "no diabetic retinopathy, healthy retina with normal blood vessels..." |
| 1 | "mild NPDR with only a few microaneurysms, no hemorrhages..." |
| 2 | "moderate NPDR with microaneurysms, dot-blot hemorrhages, hard exudates..." |
| 3 | "severe NPDR with venous beading, intraretinal hemorrhages in four quadrants..." |
| 4 | "proliferative DR with neovascularization, vitreous hemorrhage..." |

Cosine similarity against these prototypes produces interpretable confidence scores per grade — the model can explain *why* it chose a grade by showing which clinical description matched best.

### Ordinal Regression Head (Clinical Accuracy)

Diabetic retinopathy grading is inherently **ordinal** — misclassifying Grade 3 as Grade 4 is clinically acceptable, but Grade 0 as Grade 4 is dangerous. Standard cross-entropy treats all errors equally, which is wrong for this task.

A **CORAL (COnsistent RAnk Logits)** head solves this by decomposing the problem into 4 binary tasks:
- Is severity ≥ Grade 1?
- Is severity ≥ Grade 2?
- Is severity ≥ Grade 3?
- Is severity ≥ Grade 4?

The final grade is the sum of positive tasks. This:
- Penalizes distant errors more than near errors (matches clinical reality)
- Improves **Quadratic Weighted Kappa** — the gold standard metric for DR grading
- Produces calibrated confidence scores per decision threshold

Both heads share the same learned projection features, so the interpretability of prototypes is preserved while the ordinal head drives the final prediction.

### Two Operating Modes

| Mode | Training Data | Use Case |
|------|--------------|----------|
| **Pure Zero-Shot** | None — just CLIP | Instant deploy, no training |
| **Projection Tuning** | Labeled retina | Higher accuracy, trained ordinal head |

## Results

| Metric | Zero-Shot | With Projection Tuning |
|--------|-----------|----------------------|
| Accuracy | TBD | TBD |
| Quadratic Kappa | TBD | TBD |
| F1 (weighted) | TBD | TBD |

## Project Structure

```
RetinaScan/
├── data/raw/                      # Raw EyePACS dataset
├── data/processed/                # Preprocessed images
├── src/
│   ├── preprocess.py              # CLAHE + Ben Graham + crop-to-circle
│   ├── train.py                   # Training loop (CSV-based, patient-level split)
│   ├── model/
│   │   ├── clip_proto.py          # CLIP dual encoder (image + text)
│   │   └── prototype_bank.py      # Text-derived prototypes
│   ├── losses/
│   │   └── proto_loss.py          # Text-alignment + entropy + diversity loss
│   └── evaluate/
│       ├── metrics.py             # Accuracy, Kappa, F1, confusion matrix
│       └── gradcam.py             # ViT Grad-CAM severity heatmaps
├── deploy/
│   └── export_onnx.py             # ONNX export + latency benchmark
├── configs/
│   └── train_config.yaml          # All hyperparameters
├── notebooks/
│   ├── EDA.ipynb
│   ├── Train.ipynb
│   └── Evaluate.ipynb
├── app.py                         # Gradio interface (HF Spaces)
├── terminal.md                    # Colab installation guide
├── requirements.txt
└── README.md
```

## Dataset

Two data sources supported (set `data.source` in config):

| Source | Size | Auth | Preprocessing |
|--------|------|------|---------------|
| `huggingface` (default) | 6.5GB — `bumbledeep/eyepacs` | None | Already cropped + resized |
| `local` | 88GB — EyePACS Kaggle | Kaggle API token | Run `src/preprocess.py` |

## Quick Start

### Pure Zero-Shot (2 minutes, no GPU needed for inference)
```bash
pip install -r requirements.txt
python src/evaluate/metrics.py --config configs/train_config.yaml
```

### With Projection Tuning (Colab T4, ~2-3h)
```bash
# Dataset loads automatically from HuggingFace — no download step needed
python src/train.py --config configs/train_config.yaml
python src/evaluate/metrics.py --config configs/train_config.yaml --checkpoint checkpoints/best.pt
```

### Grad-CAM Visualization
```bash
python src/evaluate/gradcam.py --config configs/train_config.yaml --checkpoint checkpoints/best.pt --image sample.jpeg
```

## Deployment

1. **Train** → `checkpoints/best.pt`
2. **Push to Hugging Face Spaces** (Gradio SDK — `app.py`)
3. **UptimeRobot** ping every 5 min → keeps Space warm
4. **Inference**: <500ms per image on CPU, <100ms on GPU

## Citation

```bibtex
@misc{retinascan2026,
  title={RetinaScan: Zero-Shot Diabetic Retinopathy Grading via CLIP Text-Guided Visual Prototypes},
  author={Simanta Das},
  year={2026}
}
```
