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
    print(f"Loaded checkpoint: {checkpoint_path}")
else:
    print("No checkpoint found — running in pure zero-shot mode")

model.eval()

transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def predict(image):
    img_pil = Image.fromarray(image).convert("RGB")
    img_tensor = transform(img_pil).unsqueeze(0).to(device)

    with torch.no_grad():
        grade, probs = model.predict_grade(img_tensor)

    grade = grade.item()
    probs = probs[0].tolist()

    result = {SEVERITY_LABELS[i]: round(p, 4) for i, p in enumerate(probs)}
    result["Predicted Grade"] = int(grade)
    result["Severity"] = SEVERITY_LABELS[grade]

    proto_descriptions = model.get_prototype_descriptions()
    matched_description = proto_descriptions[grade]
    result["Prototype Match"] = matched_description

    return result


description_text = "\n".join(
    [f"**{SEVERITY_LABELS[i]}**: {desc}" for i, desc in enumerate(SEVERITY_DESCRIPTIONS)]
)

demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="numpy", label="Retina Fundus Image"),
    outputs=gr.JSON(label="Prediction Results"),
    title="RetinaScan — Zero-Shot DR Grading via CLIP Text Prototypes",
    description=(
        "Upload a retina fundus image. The model compares your image "
        "against **CLIP text embeddings** of clinical severity descriptions — "
        "no training labels required.\n\n"
        f"### Prototype Descriptions\n{description_text}"
    ),
    examples=[["data/raw/sample.jpeg"]] if os.path.exists("data/raw/sample.jpeg") else None,
)

if __name__ == "__main__":
    demo.launch()
