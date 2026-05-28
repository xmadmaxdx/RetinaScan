import sys, os, warnings
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import yaml
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score
import numpy as np
from tqdm import tqdm
from copy import deepcopy

from src.model.clip_proto import CLIPZeroShotNetwork
from src.train import build_dataset, get_val_transform
from src.evaluate.metrics import evaluate
from src.calibrate import calibrate

warnings.filterwarnings("ignore", message=".*QuickGELU.*")

FINAL_PATH = "checkpoints/final.pt"
BEST_KAPPA_SO_FAR = -1.0


def _load_model(config, checkpoint_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CLIPZeroShotNetwork(config, device=device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval()
    if "ordinal_temperature" in ckpt:
        model.set_temperatures(
            ord_temp=ckpt["ordinal_temperature"],
            proto_temp=ckpt["prototype_temperature"],
        )
    return model, ckpt


def _save_final(model, ckpt, source_label, metrics, kappa, knn_features=None, knn_labels=None):
    global BEST_KAPPA_SO_FAR
    if kappa <= BEST_KAPPA_SO_FAR:
        return
    BEST_KAPPA_SO_FAR = kappa
    save_ckpt = {
        "model_state_dict": model.state_dict(),
        "text_descriptions": model.get_prototype_descriptions(),
        "ordinal_temperature": model.ordinal_temperature.item(),
        "prototype_temperature": model.prototype_temperature.item(),
        "source": source_label,
        "kappa": metrics.get("kappa", 0),
        "accuracy": metrics.get("accuracy", 0),
        "mae": metrics.get("mae", 0),
        "off_by_one": metrics.get("off_by_one", 0),
    }
    if knn_features is not None and knn_labels is not None:
        save_ckpt["knn_features"] = knn_features
        save_ckpt["knn_labels"] = knn_labels
    if "optimal_thresholds" in ckpt:
        save_ckpt["optimal_thresholds"] = ckpt["optimal_thresholds"]
    torch.save(save_ckpt, FINAL_PATH)
    print(f"  >>> NEW BEST: {source_label} (kappa={kappa:.4f}) -> {FINAL_PATH}")


def evaluate_baseline(config, checkpoint_path):
    print(f"\n{'='*60}")
    print(f"  BASELINE: {os.path.basename(checkpoint_path)}")
    print(f"{'='*60}")
    results = evaluate(config, checkpoint_path, tune_thresholds=False, split="val")
    return {
        "accuracy": results["accuracy"],
        "kappa": results["kappa"],
        "f1": results.get("f1_weighted", 0),
        "mae": results.get("mae", 0),
        "off_by_one": results.get("off_by_one", 0),
        "ece": results.get("ece", 0),
    }


def evaluate_calibrated_tuned(config, checkpoint_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    name = os.path.basename(checkpoint_path)
    cal_path = checkpoint_path.replace(".pt", "_cal.pt")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    torch.save(ckpt, cal_path)

    print(f"\n{'='*60}")
    print(f"  CALIBRATE: {name}")
    print(f"{'='*60}")
    calibrate(config, cal_path)

    print(f"\n{'='*60}")
    print(f"  TUNED: {name}")
    print(f"{'='*60}")
    results = evaluate(config, cal_path, tune_thresholds=True, split="val")

    if "tuned_accuracy" not in results:
        print(f"  WARNING: tuned metrics not available for {name}, using raw")
        tuned = {
            "accuracy": results.get("accuracy", 0),
            "kappa": results.get("kappa", 0),
            "f1": results.get("f1_weighted", 0),
            "mae": results.get("mae", 0),
            "off_by_one": results.get("off_by_one", 0),
            "ece": results.get("ece", 0),
        }
    else:
        tuned = {
            "accuracy": results["tuned_accuracy"],
            "kappa": results["tuned_kappa"],
            "f1": results.get("tuned_f1_weighted", 0),
            "mae": results.get("tuned_mae", 0),
            "off_by_one": results.get("tuned_off_by_one", 0),
            "ece": results.get("ece", 0),
        }

    cal_ckpt = torch.load(cal_path, map_location=device, weights_only=False)
    if os.path.exists(cal_path):
        os.remove(cal_path)
    return tuned, cal_ckpt


def swa_checkpoints(config, paths, output_path):
    print(f"\n{'='*60}")
    print(f"  SWA: averaging {[os.path.basename(p) for p in paths]}")
    print(f"{'='*60}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CLIPZeroShotNetwork(config, device=device)

    avg_sd = None
    n = 0
    for path in paths:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        sd = ckpt["model_state_dict"]
        if avg_sd is None:
            avg_sd = {k: v.clone() for k, v in sd.items()}
        else:
            for k in avg_sd:
                avg_sd[k] += sd[k]
        n += 1
    for k in avg_sd:
        avg_sd[k] /= n

    model.load_state_dict(avg_sd, strict=False)
    torch.save({
        "model_state_dict": model.state_dict(),
        "text_descriptions": model.get_prototype_descriptions(),
        "ordinal_temperature": model.ordinal_temperature.item(),
        "prototype_temperature": model.prototype_temperature.item(),
    }, output_path)
    print(f"  Saved SWA checkpoint: {output_path}")
    return output_path


def ensemble_evaluate(config, checkpoints):
    print(f"\n{'='*60}")
    print(f"  ENSEMBLE: logit avg of {[os.path.basename(p) for p in checkpoints]}")
    print(f"{'='*60}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    models = []
    temps = []
    for path in checkpoints:
        m, ckpt = _load_model(config, path)
        models.append(m)
        ot = ckpt.get("ordinal_temperature", 1.0)
        temps.append(ot)

    val_ds = build_dataset(config, get_val_transform(config), split="val")
    loader = DataLoader(
        val_ds, batch_size=config["training"]["batch_size"],
        shuffle=False, num_workers=2
    )

    all_preds, all_labels = [], []
    for images, labels in tqdm(loader, desc="Ensemble eval"):
        images = images.to(device)
        avg_ord = None
        for i, m in enumerate(models):
            _, _, ordinal_logits = m(images)
            if ordinal_logits is not None:
                scaled = ordinal_logits / temps[i]
                if avg_ord is None:
                    avg_ord = scaled
                else:
                    avg_ord += scaled
        if avg_ord is not None:
            avg_ord /= len(models)
            grades = (avg_ord > 0.0).sum(dim=-1)
        else:
            grades = models[0](images)[0].argmax(dim=-1)
        all_preds.extend(grades.cpu().tolist())
        all_labels.extend(labels.tolist())

    labels_np = np.array(all_labels)
    preds_np = np.array(all_preds)
    acc = accuracy_score(labels_np, preds_np)
    kappa = cohen_kappa_score(labels_np, preds_np, weights="quadratic")
    print(f"  Ensemble Accuracy: {acc:.4f} | Kappa: {kappa:.4f}")
    return {
        "accuracy": acc,
        "kappa": kappa,
        "mae": np.abs(preds_np - labels_np).mean(),
        "off_by_one": (np.abs(preds_np - labels_np) <= 1).mean(),
    }


def knn_evaluate(config, checkpoint_path, k=10):
    print(f"\n{'='*60}")
    print(f"  KNN (k={k}): {os.path.basename(checkpoint_path)}")
    print(f"{'='*60}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = _load_model(config, checkpoint_path)

    train_ds = build_dataset(config, get_val_transform(config), split="train")
    train_loader = DataLoader(
        train_ds, batch_size=config["training"]["batch_size"],
        shuffle=False, num_workers=2
    )

    all_train_feats, all_train_labels = [], []
    for images, labels in tqdm(train_loader, desc="Extracting train features"):
        images = images.to(device)
        _, projected, _ = model(images)
        all_train_feats.append(projected.cpu())
        all_train_labels.append(labels)
    train_features = torch.cat(all_train_feats)
    train_labels = torch.cat(all_train_labels)
    print(f"  Train features: {train_features.shape}")

    val_ds = build_dataset(config, get_val_transform(config), split="val")
    val_loader = DataLoader(
        val_ds, batch_size=config["training"]["batch_size"],
        shuffle=False, num_workers=2
    )

    all_preds, all_labels = [], []
    for images, labels in tqdm(val_loader, desc="KNN eval"):
        images = images.to(device)
        _, projected, _ = model(images)
        proj_cpu = projected.cpu()
        dist = torch.cdist(proj_cpu, train_features)
        _, idx = dist.topk(k, largest=False)
        knn_labels = train_labels[idx]
        preds = torch.mode(knn_labels, dim=-1).values
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.tolist())

    labels_np = np.array(all_labels)
    preds_np = np.array(all_preds)
    acc = accuracy_score(labels_np, preds_np)
    kappa = cohen_kappa_score(labels_np, preds_np, weights="quadratic")
    print(f"  KNN Accuracy: {acc:.4f} | Kappa: {kappa:.4f}")
    return {
        "accuracy": acc,
        "kappa": kappa,
        "mae": np.abs(preds_np - labels_np).mean(),
        "off_by_one": (np.abs(preds_np - labels_np) <= 1).mean(),
    }, train_features, train_labels


def smoothing_evaluate(config, checkpoint_path):
    print(f"\n{'='*60}")
    print(f"  PREDICTION SMOOTHING: {os.path.basename(checkpoint_path)}")
    print(f"{'='*60}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ckpt = _load_model(config, checkpoint_path)

    val_ds = build_dataset(config, get_val_transform(config), split="val")
    loader = DataLoader(
        val_ds, batch_size=config["training"]["batch_size"],
        shuffle=False, num_workers=2
    )

    kernel = torch.tensor([0.05, 0.25, 0.40, 0.25, 0.05], dtype=torch.float32, device=device)

    all_preds, all_labels = [], []
    for images, labels in tqdm(loader, desc="Smoothing eval"):
        images = images.to(device)
        proto_logits, _, ordinal_logits = model(images)
        if ordinal_logits is not None:
            cal_proto = proto_logits / model.prototype_temperature
            probs = torch.softmax(cal_proto, dim=-1)
        else:
            probs = torch.softmax(proto_logits, dim=-1)
        smoothed = F.conv1d(
            probs.unsqueeze(1).float(),
            kernel.view(1, 1, -1),
            padding=2,
        ).squeeze(1)
        smoothed = smoothed / smoothed.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        grades = smoothed.argmax(dim=-1)
        all_preds.extend(grades.cpu().tolist())
        all_labels.extend(labels.tolist())

    labels_np = np.array(all_labels)
    preds_np = np.array(all_preds)
    acc = accuracy_score(labels_np, preds_np)
    kappa = cohen_kappa_score(labels_np, preds_np, weights="quadratic")
    print(f"  Smoothing Accuracy: {acc:.4f} | Kappa: {kappa:.4f}")
    return {
        "accuracy": acc,
        "kappa": kappa,
        "mae": np.abs(preds_np - labels_np).mean(),
        "off_by_one": (np.abs(preds_np - labels_np) <= 1).mean(),
    }


def build_table(results, baseline_kappa, baseline_acc):
    print(f"\n{'='*72}")
    print(f"  FINAL COMPARISON — All Modes")
    print(f"{'='*72}")
    print(f"  Baseline: kappa={baseline_kappa:.4f}  acc={baseline_acc:.4f}")
    print(f"  Discard rules: kappa drop → trash | acc drop >5% → trash")
    print()
    header = f"  {'Mode':<28} {'Acc':>8} {'Kappa':>8} {'MAE':>8} {'Off-by-1':>10} {'Saved?':>8}"
    sep = f"  {'─'*72}"
    print(header)
    print(sep)

    saved_label = None
    saved_metrics = None

    for label, metrics, final_ckpt in results:
        acc = metrics.get("accuracy", 0)
        kappa = metrics.get("kappa", 0)
        mae = metrics.get("mae", 0)
        off1 = metrics.get("off_by_one", 0)

        kappa_ok = kappa >= baseline_kappa - 0.001
        acc_ok = acc >= baseline_acc - 0.05
        if kappa_ok and acc_ok:
            if final_ckpt:
                saved_label = label
                saved_metrics = metrics
                status = "✓ SAVED"
            else:
                status = "✓ KEEP"
        else:
            reasons = []
            if not kappa_ok:
                reasons.append(f"κΔ={kappa - baseline_kappa:.4f}")
            if not acc_ok:
                reasons.append(f"aΔ={acc - baseline_acc:.4f}")
            status = f"✗ TRASH ({', '.join(reasons)})"

        print(f"  {label:<28} {acc:>8.4f} {kappa:>8.4f} {mae:>8.4f} {off1:>10.4f} {status}")

    print(sep)
    if saved_label:
        print(f"\n  >>> BEST SAVED: {saved_label} -> {FINAL_PATH} <<<")
        print(f"  Acc: {saved_metrics['accuracy']:.4f} | Kappa: {saved_metrics['kappa']:.4f} | MAE: {saved_metrics.get('mae', 0):.4f} | Off-by-1: {saved_metrics.get('off_by_one', 0):.4f}")
    else:
        print(f"\n  >>> No mode improved over baseline. {FINAL_PATH} not saved. <<<")
    print()


def final_pipeline(config, checkpoints):
    print(f"{'='*72}")
    print(f"  RETINASCAN — FINAL VALIDATION PIPELINE")
    print(f"{'='*72}")
    print(f"  Checkpoints: {[os.path.basename(c) for c in checkpoints]}")
    print(f"  Modes: baseline, calibrate+tune, SWA, ensemble, KNN, smoothing")
    print()

    global BEST_KAPPA_SO_FAR
    BEST_KAPPA_SO_FAR = -1.0
    all_results = []

    # 1. Baseline on each checkpoint
    for ckpt_path in checkpoints:
        label = f"{os.path.basename(ckpt_path)} baseline"
        metrics = evaluate_baseline(config, ckpt_path)
        is_best = metrics["kappa"] > BEST_KAPPA_SO_FAR
        all_results.append((label, metrics, is_best))
        if is_best:
            m, ckpt = _load_model(config, ckpt_path)
            _save_final(m, ckpt, label, metrics, metrics["kappa"])
        print(f"  {label}: acc={metrics['accuracy']:.4f} kappa={metrics['kappa']:.4f}")

    # 2. Calibrate + tune on each checkpoint
    for ckpt_path in checkpoints:
        label = f"{os.path.basename(ckpt_path)} tuned"
        metrics, cal_ckpt = evaluate_calibrated_tuned(config, ckpt_path)
        is_best = metrics["kappa"] > BEST_KAPPA_SO_FAR
        all_results.append((label, metrics, is_best))
        if is_best:
            m, _ = _load_model(config, ckpt_path)
            m.set_temperatures(
                ord_temp=cal_ckpt.get("ordinal_temperature", 1.0),
                proto_temp=cal_ckpt.get("prototype_temperature", 1.0),
            )
            _save_final(m, cal_ckpt, label, metrics, metrics["kappa"])
        print(f"  {label}: acc={metrics['accuracy']:.4f} kappa={metrics['kappa']:.4f}")

    # 3. SWA
    if len(checkpoints) >= 2:
        swa_path = "checkpoints/swa.pt"
        swa_checkpoints(config, checkpoints, swa_path)

        label = "SWA baseline"
        metrics = evaluate_baseline(config, swa_path)
        is_best = metrics["kappa"] > BEST_KAPPA_SO_FAR
        all_results.append((label, metrics, is_best))
        if is_best:
            m, _ = _load_model(config, swa_path)
            _save_final(m, {"optimal_thresholds": None}, label, metrics, metrics["kappa"])
        print(f"  {label}: acc={metrics['accuracy']:.4f} kappa={metrics['kappa']:.4f}")

        label = "SWA tuned"
        metrics, cal_ckpt = evaluate_calibrated_tuned(config, swa_path)
        is_best = metrics["kappa"] > BEST_KAPPA_SO_FAR
        all_results.append((label, metrics, is_best))
        if is_best:
            m, _ = _load_model(config, swa_path)
            m.set_temperatures(
                ord_temp=cal_ckpt.get("ordinal_temperature", 1.0),
                proto_temp=cal_ckpt.get("prototype_temperature", 1.0),
            )
            _save_final(m, cal_ckpt, label, metrics, metrics["kappa"])
        print(f"  {label}: acc={metrics['accuracy']:.4f} kappa={metrics['kappa']:.4f}")

    # 4. Ensemble on all checkpoints (eval-only — would need multi-model at inference)
    if len(checkpoints) >= 2:
        label = f"Ensemble ({' + '.join(os.path.basename(c).replace('.pt','') for c in checkpoints)})"
        try:
            metrics = ensemble_evaluate(config, checkpoints)
            all_results.append((label, metrics, False))
            print(f"  {label}: acc={metrics['accuracy']:.4f} kappa={metrics['kappa']:.4f}")
        except Exception as e:
            print(f"  Ensemble failed: {e}")
            all_results.append((label, {"accuracy": 0, "kappa": 0, "mae": 0, "off_by_one": 0}, False))

    # 5. KNN on best checkpoint
    label = f"KNN (10-NN) on {os.path.basename(checkpoints[0])}"
    try:
        metrics, knn_feats, knn_labels = knn_evaluate(config, checkpoints[0], k=10)
        is_best = metrics["kappa"] > BEST_KAPPA_SO_FAR
        all_results.append((label, metrics, is_best))
        if is_best:
            m, _ = _load_model(config, checkpoints[0])
            _save_final(m, {}, label, metrics, metrics["kappa"],
                        knn_features=knn_feats, knn_labels=knn_labels)
        print(f"  {label}: acc={metrics['accuracy']:.4f} kappa={metrics['kappa']:.4f}")
    except Exception as e:
        import traceback; traceback.print_exc()
        print(f"  KNN failed: {e}")
        all_results.append((label, {"accuracy": 0, "kappa": 0, "mae": 0, "off_by_one": 0}, False))

    # 6. Prediction smoothing on best checkpoint (eval-only — applied post-prediction)
    label = f"Smoothing on {os.path.basename(checkpoints[0])}"
    try:
        metrics = smoothing_evaluate(config, checkpoints[0])
        all_results.append((label, metrics, False))
        print(f"  {label}: acc={metrics['accuracy']:.4f} kappa={metrics['kappa']:.4f}")
    except Exception as e:
        print(f"  Smoothing failed: {e}")
        all_results.append((label, {"accuracy": 0, "kappa": 0, "mae": 0, "off_by_one": 0}, False))

    baseline = all_results[0][1]
    baseline_kappa = baseline["kappa"]
    baseline_acc = baseline["accuracy"]
    display_results = [(label, metrics, saved) for label, metrics, saved in all_results]
    build_table(display_results, baseline_kappa, baseline_acc)

    if BEST_KAPPA_SO_FAR >= 0:
        print(f"  Final best kappa: {BEST_KAPPA_SO_FAR:.4f} -> {FINAL_PATH}")
    else:
        print(f"  No improvement. Delete {FINAL_PATH} if it exists.")
    print(f"{'='*72}")
    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train_config.yaml")
    parser.add_argument("--checkpoints", nargs="+", default=["checkpoints/best.pt", "checkpoints/latest.pt"])
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    final_pipeline(cfg, args.checkpoints)
