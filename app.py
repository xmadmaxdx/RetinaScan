import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import gradio as gr
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from src.model.clip_proto import CLIPZeroShotNetwork
from src.model.prototype_bank import SEVERITY_LABELS, SEVERITY_DESCRIPTIONS


CONFIG = {
    "data": {"image_size": 512},
    "model": {
        "backbone": "ViT-B/16",
        "pretrained": "openai",
        "prototype_dim": 512,
        "temperature": 0.07,
        "zero_shot_only": False,
        "severities": SEVERITY_DESCRIPTIONS,
        "num_prototypes": 5,
    },
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CLIPZeroShotNetwork(CONFIG, device=device)

checkpoint_path = "checkpoints/best.pt"
if os.path.exists(checkpoint_path):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    if "ordinal_temperature" in ckpt:
        model.set_temperatures(ord_temp=ckpt["ordinal_temperature"], proto_temp=ckpt["prototype_temperature"])
        print(f"Loaded temperatures: ordinal={ckpt['ordinal_temperature']:.3f}, prototype={ckpt['prototype_temperature']:.3f}")
    print(f"Loaded checkpoint: {checkpoint_path}")
else:
    print("No checkpoint found — running in pure zero-shot mode")

model.eval()

transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


SEVERITY_COLORS = ["green", "yellow", "orange", "red", "darkred"]
N_RUNS = 20


def predict(image):
    img_pil = Image.fromarray(image).convert("RGB")
    img_tensor = transform(img_pil).unsqueeze(0).to(device)

    with torch.no_grad():
        grade, probs = model.predict_grade(img_tensor)

    grade = grade.item()
    probs = probs[0].tolist()

    mean_grade, confidence, mean_probs = model.predict_with_uncertainty([img_pil], n_runs=N_RUNS)
    confidence = confidence[0].item()

    severity_label = SEVERITY_LABELS[int(round(mean_grade.item()))]
    html_color = "green" if confidence > 0.7 else "orange" if confidence > 0.4 else "red"
    confidence_text = f"<span style='color:{html_color}; font-weight:bold'>{confidence:.0%}</span>"
    grade_color = SEVERITY_COLORS[int(round(mean_grade.item()))]

    matched_desc = model.get_prototype_descriptions()[int(round(mean_grade.item()))]

    uncertainty_breakdown = ""
    if confidence < 0.7:
        second_best = mean_probs[0].argsort(descending=True)
        alt_grades = [SEVERITY_LABELS[int(g)] for g in second_best[1:3] if mean_probs[0][int(g)] > 0.1]
        if alt_grades:
            uncertainty_breakdown = f"Also possible: {', '.join(alt_grades)}"

    result = (
        f"## <span style='color:{grade_color}'>{severity_label}</span>\n\n"
        f"**Confidence**: {confidence_text}\n\n"
        f"**Prototype Match**: {matched_desc}\n\n"
        f"{'⚠️ ' + uncertainty_breakdown if uncertainty_breakdown else ''}\n\n"
        "### Per-Grade Similarity\n"
    )
    for i, (label, p) in enumerate(zip(SEVERITY_LABELS, mean_probs[0].tolist())):
        pct = max(p * 100, 0.5)
        sub = label.split(" — ")[1] if " — " in label else label
        result += f"**{sub}**: {p*100:.1f}%\n"
        result += f"<div style='background:#e0e0e0; border-radius:4px; height:16px; width:100%'>"
        result += f"<div style='background:{SEVERITY_COLORS[i]}; width:{pct:.0f}%; height:16px; border-radius:4px'></div></div>\n"

    return result


description_text = "\n".join(
    [f"**{SEVERITY_LABELS[i]}**: {desc}" for i, desc in enumerate(SEVERITY_DESCRIPTIONS)]
)

demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="numpy", label="Retina Fundus Image"),
    outputs=gr.Markdown(label="Prediction Results"),
    title="RetinaScan — Zero-Shot DR Grading via CLIP Text Prototypes",
    description=(
        "Upload a retina fundus image. The model compares your image "
        "against **CLIP text embeddings** of clinical severity descriptions.\n\n"
        "Uncertainty is estimated via **test-time augmentation** (20 runs): "
        "if confidence is low, alternative grades are shown — indicating the "
        "model needs a clearer image or is near a decision boundary.\n\n"
        "Confidence scores are **temperature-calibrated** to match true accuracy.\n\n"
        f"### Prototype Descriptions\n{description_text}"
    ),
    examples=[["data/raw/sample.jpeg"]] if os.path.exists("data/raw/sample.jpeg") else None,
)

if __name__ == "__main__":
    demo.launch()
