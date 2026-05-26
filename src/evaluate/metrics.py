import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import csv
import yaml
import argparse
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, cohen_kappa_score, f1_score,
    confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from src.model.clip_proto import CLIPZeroShotNetwork
from src.train import EyePACSDataset, HuggingFaceEyePACSDataset, get_val_transform
from src.model.prototype_bank import SEVERITY_LABELS


def get_eval_dataset(config):
    source = config["data"].get("source", "local")
    transform = get_val_transform(config)
    if source == "huggingface":
        return HuggingFaceEyePACSDataset(
            hf_dataset_name=config["data"]["hf_dataset"],
            split=config["data"].get("hf_split", "train"),
            transform=transform,
        )
    csv_path = config["data"]["labels_csv"]
    image_dir = config["data"]["processed_path"]
    if not os.path.exists(image_dir):
        image_dir = config["data"]["raw_path"]
    return EyePACSDataset(csv_path, image_dir, transform=transform)


@torch.no_grad()
def evaluate(config, checkpoint_path=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CLIPZeroShotNetwork(config, device=device)
    if checkpoint_path and os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
        print(f"Loaded checkpoint: {checkpoint_path}")
    else:
        print("No checkpoint — running pure zero-shot evaluation")

    model.eval()

    dataset = get_eval_dataset(config)
    loader = DataLoader(dataset, batch_size=config["training"]["batch_size"], shuffle=False, num_workers=2)

    all_preds, all_labels, all_probs = [], [], []
    for images, labels in loader:
        images = images.to(device)
        grades, probs = model.predict_grade(images)
        all_preds.extend(grades.cpu().tolist())
        all_labels.extend(labels.tolist())
        all_probs.extend(probs.cpu().tolist())

    acc = accuracy_score(all_labels, all_preds)
    kappa = cohen_kappa_score(all_labels, all_preds, weights="quadratic")
    f1_macro = f1_score(all_labels, all_preds, average="macro")
    f1_weighted = f1_score(all_labels, all_preds, average="weighted")

    print(f"\n{'='*50}")
    print(f"Zero-Shot Accuracy:  {acc:.4f}")
    print(f"Quadratic Kappa:     {kappa:.4f}")
    print(f"F1 Macro:            {f1_macro:.4f}")
    print(f"F1 Weighted:         {f1_weighted:.4f}")
    print(f"{'='*50}\n")

    print("Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=SEVERITY_LABELS, digits=4))

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=SEVERITY_LABELS, yticklabels=SEVERITY_LABELS)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix - DR Grades (0-4)")
    os.makedirs(config["paths"]["log_dir"], exist_ok=True)
    plt.savefig(os.path.join(config["paths"]["log_dir"], "confusion_matrix.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
    print("Confusion matrix saved to logs/")

    return {"accuracy": acc, "kappa": kappa, "f1_macro": f1_macro, "f1_weighted": f1_weighted}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train_config.yaml")
    parser.add_argument("--checkpoint", default=None)
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    evaluate(cfg, args.checkpoint)
