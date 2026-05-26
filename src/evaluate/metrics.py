import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from copy import deepcopy
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
from src.train import build_dataset, get_val_transform
from src.model.prototype_bank import SEVERITY_LABELS


def get_eval_dataset(config):
    transform = get_val_transform(config)
    return build_dataset(config, transform, split="val")


def expected_calibration_error(labels, probs, n_bins=10):
    confs, preds = probs.max(dim=-1)
    accs = (preds == labels).float()
    bounds = torch.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bounds[i], bounds[i + 1]
        mask = (confs > lo) & (confs <= hi)
        n = mask.sum()
        if n > 0:
            ece += (accs[mask].mean() - confs[mask].mean()).abs().item() * n.item() / len(labels)
    return ece


def sync_to_drive(src_path, drive_dir):
    if not os.path.exists(drive_dir):
        os.makedirs(drive_dir, exist_ok=True)
    import shutil
    shutil.copy2(src_path, os.path.join(drive_dir, os.path.basename(src_path)))


@torch.no_grad()
def evaluate(config, checkpoint_path=None, drive_path=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if checkpoint_path and os.path.exists(checkpoint_path):
        model = CLIPZeroShotNetwork(config, device=device)
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
        if "ordinal_temperature" in ckpt:
            model.set_temperatures(ord_temp=ckpt["ordinal_temperature"], proto_temp=ckpt["prototype_temperature"])
            print(f"Loaded temperatures: ordinal={ckpt['ordinal_temperature']:.3f}, prototype={ckpt['prototype_temperature']:.3f}")
        print(f"Loaded checkpoint: {checkpoint_path}")
    else:
        print("No checkpoint — running pure zero-shot evaluation")
        zs_config = deepcopy(config)
        zs_config["model"]["zero_shot_only"] = True
        model = CLIPZeroShotNetwork(zs_config, device=device)

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

    probs_tensor = torch.tensor(all_probs)
    labels_tensor = torch.tensor(all_labels)
    ece = expected_calibration_error(labels_tensor, probs_tensor)

    print(f"\n{'='*50}")
    print(f"Zero-Shot Accuracy:  {acc:.4f}")
    print(f"Quadratic Kappa:     {kappa:.4f}")
    print(f"F1 Macro:            {f1_macro:.4f}")
    print(f"F1 Weighted:         {f1_weighted:.4f}")
    print(f"ECE (calibration):   {ece:.4f}")
    print(f"{'='*50}\n")

    print("Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=SEVERITY_LABELS, digits=4, zero_division=0))

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=SEVERITY_LABELS, yticklabels=SEVERITY_LABELS)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix - DR Grades (0-4)")
    log_dir = config["paths"]["log_dir"]
    os.makedirs(log_dir, exist_ok=True)
    cm_path = os.path.join(log_dir, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=150, bbox_inches="tight")
    plt.close()

    if drive_path:
        os.makedirs(drive_path, exist_ok=True)
        sync_to_drive(cm_path, drive_path)
        if checkpoint_path and os.path.exists(checkpoint_path):
            sync_to_drive(checkpoint_path, drive_path)
        print(f"Synced to Drive: {drive_path}")
    print("Confusion matrix saved to logs/")

    return {"accuracy": acc, "kappa": kappa, "f1_macro": f1_macro, "f1_weighted": f1_weighted}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train_config.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--drive-path", default=None, help="Sync results to Google Drive path")
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    evaluate(cfg, args.checkpoint, drive_path=args.drive_path)
