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
    confusion_matrix, classification_report, precision_recall_fscore_support
)
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from src.model.clip_proto import CLIPZeroShotNetwork
from src.train import build_dataset, get_val_transform
from src.model.prototype_bank import SEVERITY_LABELS


def get_eval_dataset(config, split="val"):
    transform = get_val_transform(config)
    return build_dataset(config, transform, split=split)


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


def maximum_calibration_error(labels, probs, n_bins=10):
    confs, preds = probs.max(dim=-1)
    accs = (preds == labels).float()
    bounds = torch.linspace(0, 1, n_bins + 1)
    mce = 0.0
    for i in range(n_bins):
        lo, hi = bounds[i], bounds[i + 1]
        mask = (confs > lo) & (confs <= hi)
        n = mask.sum()
        if n > 0:
            mce = max(mce, (accs[mask].mean() - confs[mask].mean()).abs().item())
    return mce


def find_optimal_thresholds(ord_logits, labels):
    thresholds = []
    for k in range(4):
        targets = (labels > k).float()
        best_t = 0.0
        best_f1 = 0.0
        for t in [i * 0.1 for i in range(-100, 101)]:
            preds = (ord_logits[:, k] > t).float()
            f1 = f1_score(targets.cpu().numpy(), preds.cpu().numpy(), zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_t = t
        thresholds.append(best_t)
    return torch.tensor(thresholds)


def plot_reliability_diagram(labels, probs, save_path, n_bins=10):
    confs, preds = probs.max(dim=-1)
    accs = (preds == labels).float()
    bounds = torch.linspace(0, 1, n_bins + 1)
    bin_conf = []
    bin_acc = []
    bin_counts = []
    for i in range(n_bins):
        lo, hi = bounds[i], bounds[i + 1]
        mask = (confs > lo) & (confs <= hi)
        n = mask.sum()
        bin_counts.append(n.item())
        if n > 0:
            bin_conf.append(confs[mask].mean().item())
            bin_acc.append(accs[mask].mean().item())
        else:
            bin_conf.append((lo + hi) / 2)
            bin_acc.append(0)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Perfect Calibration")
    ax.plot(bin_conf, bin_acc, "o-", color="#1a76b5", linewidth=2, markersize=6)
    for i, (c, a, n) in enumerate(zip(bin_conf, bin_acc, bin_counts)):
        ax.annotate(f"n={n}", (c, a), textcoords="offset points", xytext=(5, 5), fontsize=7)
    ax.set_xlabel("Confidence", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title("Reliability Diagram", fontsize=14)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def sync_to_drive(src_path, drive_dir):
    if not os.path.exists(drive_dir):
        os.makedirs(drive_dir, exist_ok=True)
    import shutil
    shutil.copy2(src_path, os.path.join(drive_dir, os.path.basename(src_path)))


def print_ordinal_report(labels, preds):
    diff = np.array(preds) - np.array(labels)
    off_by_one = np.abs(diff) <= 1
    mae = np.abs(diff).mean()
    undershoot = (diff < 0).mean()
    overshoot = (diff > 0).mean()
    exact = (diff == 0).mean()

    print(f"\n  Off-by-0 (exact):     {exact:.4f}")
    print(f"  Off-by-1 or better:   {off_by_one.mean():.4f}")
    print(f"  MAE (mean grade err): {mae:.4f}")
    print(f"  Undershoot rate:      {undershoot:.4f}")
    print(f"  Overshoot rate:       {overshoot:.4f}")

    print(f"\n  Error Distribution:")
    for d in range(-4, 5):
        pct = (diff == d).mean()
        if pct > 0.001:
            direction = "under" if d < 0 else "over"
            print(f"    Off by {abs(d)} ({direction}): {pct*100:.1f}%")


@torch.no_grad()
def evaluate(config, checkpoint_path=None, drive_path=None, tune_thresholds=True, split="val"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if checkpoint_path and os.path.exists(checkpoint_path):
        model = CLIPZeroShotNetwork(config, device=device)
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
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
        tune_thresholds = False

    model.eval()

    dataset = get_eval_dataset(config, split=split)
    loader = DataLoader(dataset, batch_size=config["training"]["batch_size"], shuffle=False, num_workers=2)

    all_preds, all_labels, all_probs, all_ord_logits = [], [], [], []
    for images, labels in loader:
        images = images.to(device)
        if model.zero_shot_only:
            grades, probs = model.predict_grade(images)
            ordinal_logits = None
        else:
            proto_logits, projected, ordinal_logits = model(images)
            if ordinal_logits is not None:
                cal_ord = ordinal_logits / model.ordinal_temperature
                grades = (cal_ord > 0.0).sum(dim=-1)
                cal_proto = proto_logits / model.prototype_temperature
                probs = torch.softmax(cal_proto, dim=-1)
            else:
                grades = proto_logits.argmax(dim=-1)
                probs = torch.softmax(proto_logits, dim=-1)
        all_preds.extend(grades.cpu().tolist())
        all_labels.extend(labels.tolist())
        all_probs.extend(probs.cpu().tolist())
        if not model.zero_shot_only and ordinal_logits is not None:
            all_ord_logits.append(ordinal_logits.cpu())

    labels_np = np.array(all_labels)
    preds_np = np.array(all_preds)
    probs_tensor = torch.tensor(all_probs)
    labels_tensor = torch.tensor(all_labels)

    if tune_thresholds and len(all_ord_logits) > 0:
        ord_logits = torch.cat(all_ord_logits)
        optimal_thresholds = find_optimal_thresholds(ord_logits, labels_tensor)
        print(f"\nOptimal ordinal thresholds: {optimal_thresholds.tolist()}")
        print(f"  (default was [0.0, 0.0, 0.0, 0.0])")

        model.eval()
        tuned_preds = []
        for images, labels in loader:
            images = images.to(device)
            grades, _ = model.predict_grade(images, thresholds=optimal_thresholds)
            tuned_preds.extend(grades.cpu().tolist())
        tuned_preds_np = np.array(tuned_preds)
        tuned_acc = accuracy_score(all_labels, tuned_preds_np)
        tuned_kappa = cohen_kappa_score(all_labels, tuned_preds_np, weights="quadratic")
        tuned_f1 = f1_score(all_labels, tuned_preds_np, average="weighted")
        tuned_offby1 = (np.abs(tuned_preds_np - labels_np) <= 1).mean()
        tuned_mae = np.abs(tuned_preds_np - labels_np).mean()

        ckpt["optimal_thresholds"] = optimal_thresholds.tolist()
        torch.save(ckpt, checkpoint_path)
        print(f"Saved optimal thresholds to checkpoint.")

    default_threshold_str = " (default 0.0)" if not tune_thresholds else ""
    acc = accuracy_score(all_labels, all_preds)
    kappa = cohen_kappa_score(all_labels, all_preds, weights="quadratic")
    f1_macro = f1_score(all_labels, all_preds, average="macro")
    f1_weighted = f1_score(all_labels, all_preds, average="weighted")
    ece = expected_calibration_error(labels_tensor, probs_tensor)
    mce = maximum_calibration_error(labels_tensor, probs_tensor)
    off_by_one = (np.abs(preds_np - labels_np) <= 1).mean()
    mae = np.abs(preds_np - labels_np).mean()
    spearman_corr, _ = spearmanr(preds_np, labels_np)

    print(f"\n{'='*56}")
    print(f"  RETINASCAN — FULL EVALUATION REPORT")
    print(f"{'='*56}")
    print(f"  Thresholds:           default [0.0, 0.0, 0.0, 0.0]{default_threshold_str}")
    print(f"{'─'*56}")
    print(f"  Accuracy:             {acc:.4f}")
    print(f"  Quadratic Kappa:      {kappa:.4f}")
    print(f"  Spearman Correlation: {spearman_corr:.4f}")
    print(f"  F1 Macro:             {f1_macro:.4f}")
    print(f"  F1 Weighted:          {f1_weighted:.4f}")
    print(f"  MAE (grade error):    {mae:.4f}")
    print(f"  Off-by-1 Accuracy:    {off_by_one:.4f}")
    print(f"  ECE (calibration):    {ece:.4f}")
    print(f"  MCE (max cal err):    {mce:.4f}")
    print(f"{'='*56}")

    if tune_thresholds and len(all_ord_logits) > 0:
        print(f"\n  >>> WITH TUNED THRESHOLDS <<<")
        print(f"  Accuracy:             {tuned_acc:.4f}")
        print(f"  Quadratic Kappa:      {tuned_kappa:.4f}")
        print(f"  F1 Weighted:          {tuned_f1:.4f}")
        print(f"  Off-by-1 Accuracy:    {tuned_offby1:.4f}")
        print(f"  MAE (grade error):    {tuned_mae:.4f}")
        print(f"{'='*56}")

    print(f"\n  Ordinal Error Analysis{default_threshold_str}:")
    print_ordinal_report(labels_np, preds_np)

    print(f"\n  Per-Class Metrics{default_threshold_str}:")
    print(f"  {'Class':<20} {'Precision':<10} {'Recall':<10} {'F1':<10} {'Support':<10}")
    print(f"  {'─'*60}")
    p, r, f1_c, s = precision_recall_fscore_support(labels_np, preds_np, zero_division=0)
    for i in range(5):
        print(f"  {SEVERITY_LABELS[i]:<20} {p[i]:<10.4f} {r[i]:<10.4f} {f1_c[i]:<10.4f} {s[i]:<10}")

    cm = confusion_matrix(labels_np, preds_np)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)

    log_dir = config["paths"]["log_dir"]
    os.makedirs(log_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=SEVERITY_LABELS, yticklabels=SEVERITY_LABELS, ax=axes[0])
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("True")
    axes[0].set_title("Raw Counts")

    sns.heatmap(cm_norm, annot=True, fmt=".2%", cmap="Blues",
                xticklabels=SEVERITY_LABELS, yticklabels=SEVERITY_LABELS, ax=axes[1])
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("True")
    axes[1].set_title("Row-Normalized (Recall per class)")

    fig.suptitle(f"Confusion Matrix — DR Grades (default thresholds)", fontsize=14, y=1.02)
    fig.tight_layout()
    cm_path = os.path.join(log_dir, "confusion_matrix.png")
    fig.savefig(cm_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    rel_path = os.path.join(log_dir, "reliability_diagram.png")
    plot_reliability_diagram(labels_tensor, probs_tensor, rel_path)

    if tune_thresholds and len(all_ord_logits) > 0:
        fig2, axes2 = plt.subplots(1, 2, figsize=(16, 6))
        cm_tuned = confusion_matrix(labels_np, tuned_preds_np)
        cm_tuned_norm = cm_tuned.astype(float) / cm_tuned.sum(axis=1, keepdims=True).clip(min=1)
        sns.heatmap(cm_tuned, annot=True, fmt="d", cmap="Greens",
                    xticklabels=SEVERITY_LABELS, yticklabels=SEVERITY_LABELS, ax=axes2[0])
        axes2[0].set_xlabel("Predicted")
        axes2[0].set_ylabel("True")
        axes2[0].set_title("Raw Counts (Tuned)")
        sns.heatmap(cm_tuned_norm, annot=True, fmt=".2%", cmap="Greens",
                    xticklabels=SEVERITY_LABELS, yticklabels=SEVERITY_LABELS, ax=axes2[1])
        axes2[1].set_xlabel("Predicted")
        axes2[1].set_ylabel("True")
        axes2[1].set_title("Row-Normalized (Tuned)")
        fig2.suptitle(f"Confusion Matrix — DR Grades (tuned thresholds)", fontsize=14, y=1.02)
        fig2.tight_layout()
        cm_tuned_path = os.path.join(log_dir, "confusion_matrix_tuned.png")
        fig2.savefig(cm_tuned_path, dpi=150, bbox_inches="tight")
        plt.close(fig2)

    if drive_path:
        os.makedirs(drive_path, exist_ok=True)
        for fname in ["confusion_matrix.png", "reliability_diagram.png", "confusion_matrix_tuned.png"]:
            fpath = os.path.join(log_dir, fname)
            if os.path.exists(fpath):
                sync_to_drive(fpath, drive_path)
        if checkpoint_path and os.path.exists(checkpoint_path):
            sync_to_drive(checkpoint_path, drive_path)
        print(f"\nSynced to Drive: {drive_path}")

    print(f"\nPlots saved to {log_dir}/")
    print(f"  - confusion_matrix.png (default thresholds)")
    print(f"  - reliability_diagram.png")
    if tune_thresholds and len(all_ord_logits) > 0:
        print(f"  - confusion_matrix_tuned.png (tuned thresholds)")

    results = {
        "accuracy": acc, "kappa": kappa, "spearman": spearman_corr,
        "f1_macro": f1_macro, "f1_weighted": f1_weighted,
        "mae": mae, "off_by_one": off_by_one,
        "ece": ece, "mce": mce,
    }
    if tune_thresholds and len(all_ord_logits) > 0:
        results["tuned_accuracy"] = tuned_acc
        results["tuned_kappa"] = tuned_kappa
        results["tuned_f1_weighted"] = tuned_f1
        results["tuned_off_by_one"] = tuned_offby1
        results["tuned_mae"] = tuned_mae
        results["optimal_thresholds"] = optimal_thresholds.tolist()
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train_config.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--drive-path", default=None, help="Sync results to Google Drive path")
    parser.add_argument("--no-tune", action="store_true", help="Skip threshold tuning")
    parser.add_argument("--split", default="val", choices=["val", "test"], help="Which split to evaluate")
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    evaluate(cfg, args.checkpoint, drive_path=args.drive_path, tune_thresholds=not args.no_tune, split=args.split)
