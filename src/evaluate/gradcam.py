import os
import yaml
import argparse
import torch
import torch.nn.functional as F
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from src.model.clip_proto import CLIPZeroShotNetwork


class GradCAM:
    def __init__(self, model, target_module):
        self.model = model
        self.gradients = None
        self.activations = None
        self._register_hooks(target_module)

    def _register_hooks(self, module):
        def forward_hook(m, input, output):
            self.activations = output[0] if isinstance(output, tuple) else output
        def backward_hook(m, grad_input, grad_output):
            self.gradients = grad_output[0]

        module.register_forward_hook(forward_hook)
        module.register_full_backward_hook(backward_hook)

    def generate(self, image_tensor, class_idx=None):
        logits, _ = self.model.forward_gradcam(image_tensor)
        if class_idx is None:
            class_idx = logits.argmax(dim=-1).item()

        self.model.zero_grad()
        logits[0, class_idx].backward()

        act = self.activations
        if act.dim() == 3:
            b, n, d = act.shape
            h = w = int(n ** 0.5)
            act = act[:, 1:].transpose(1, 2).reshape(b, d, h, w)

        grad = self.gradients
        if grad.dim() == 3:
            b, n, d = grad.shape
            h = w = int(n ** 0.5)
            grad = grad[:, 1:].transpose(1, 2).reshape(b, d, h, w)

        weights = grad.mean(dim=(2, 3), keepdim=True)
        cam = (weights * act).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)
        cam = cam.squeeze().cpu().numpy()

        cam = cv2.resize(cam, (image_tensor.shape[2], image_tensor.shape[3]))
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam


def overlay_heatmap(img_np, cam, alpha=0.5):
    heatmap = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = (alpha * heatmap + (1 - alpha) * img_np).astype(np.uint8)
    return overlay


def main(config, checkpoint_path, image_path, save_dir="outputs/gradcam"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CLIPZeroShotNetwork(config, device=device)
    if os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
        print(f"Loaded checkpoint: {checkpoint_path}")
    else:
        print("No checkpoint found — running in pure zero-shot mode")

    model.eval()

    target_layer = model.clip_model.visual.transformer.resblocks[-1]
    gradcam = GradCAM(model, target_layer)

    transform = transforms.Compose([
        transforms.Resize((config["data"]["image_size"], config["data"]["image_size"])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    img_pil = Image.open(image_path).convert("RGB")
    img_tensor = transform(img_pil).unsqueeze(0).to(device)
    img_np = np.array(img_pil.resize((config["data"]["image_size"], config["data"]["image_size"])))

    grade, probs = model.predict_grade(img_tensor)
    cam = gradcam.generate(img_tensor, class_idx=grade.item())
    overlay = overlay_heatmap(img_np, cam)

    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, os.path.basename(image_path))
    plt.figure(figsize=(15, 5))
    for i, (title, img) in enumerate([
        ("Original", img_np),
        ("Grad-CAM Heatmap", (cam * 255).astype(np.uint8)),
        (f"Overlay | Grade {grade.item()}", overlay),
    ]):
        plt.subplot(1, 3, i + 1)
        plt.imshow(img, cmap="jet" if i == 1 else None)
        plt.title(title)
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Grad-CAM saved -> {out_path}")
    print(f"Predicted grade: {grade.item()} | Probs: {probs[0].cpu().tolist()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train_config.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--image", required=True)
    parser.add_argument("--save_dir", default="outputs/gradcam")
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    main(cfg, args.checkpoint, args.image, args.save_dir)
