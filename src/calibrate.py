import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import yaml
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from src.model.clip_proto import CLIPZeroShotNetwork
from src.train import get_val_transform, build_dataset


def expected_calibration_error(labels, probs, n_bins=10):
    confs, preds = probs.max(dim=-1)
    accs = (preds == labels).float()
    bounds = torch.linspace(0, 1, n_bins + 1, device=probs.device)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bounds[i], bounds[i + 1]
        mask = (confs > lo) & (confs <= hi)
        n = mask.sum()
        if n > 0:
            ece += (accs[mask].mean() - confs[mask].mean()).abs().item() * n.item() / len(labels)
    return ece


def calibrate(config, checkpoint_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CLIPZeroShotNetwork(config, device=device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval()
    print(f"Loaded checkpoint: {checkpoint_path}")

    val_ds = build_dataset(config, get_val_transform(config), split="val")
    loader = DataLoader(val_ds, batch_size=config["training"]["batch_size"], shuffle=False, num_workers=2)

    all_proto_logits, all_ord_logits, all_labels = [], [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            proto_logits, _, ordinal_logits = model(images)
            all_proto_logits.append(proto_logits.cpu())
            all_labels.append(labels)
            if ordinal_logits is not None:
                all_ord_logits.append(ordinal_logits.cpu())

    proto_logits = torch.cat(all_proto_logits)
    labels = torch.cat(all_labels).to(device)
    has_ordinal = len(all_ord_logits) > 0
    if has_ordinal:
        ord_logits = torch.cat(all_ord_logits)

    def best_proto_temp(temps):
        best_ece = float("inf")
        best_t = 1.0
        for t in temps:
            probs = torch.softmax(proto_logits.to(device) / t, dim=-1)
            ece = expected_calibration_error(labels.cpu(), probs.cpu())
            if ece < best_ece:
                best_ece = ece
                best_t = t
        return best_t, best_ece

    def best_ord_temp(temps):
        from sklearn.metrics import cohen_kappa_score
        best_kappa = -1.0
        best_t = 1.0
        for t in temps:
            preds = (ord_logits.to(device) / t > 0.0).sum(dim=-1).cpu()
            kappa = cohen_kappa_score(labels.cpu().numpy(), preds.numpy(), weights="quadratic")
            if kappa > best_kappa:
                best_kappa = kappa
                best_t = t
        return best_t, best_kappa

    uncal_probs = torch.softmax(proto_logits, dim=-1)
    ece_before = expected_calibration_error(labels.cpu(), uncal_probs)
    print(f"\nECE before calibration: {ece_before:.4f}")

    temps = [round(0.2 + i * 0.1, 2) for i in range(50)]  # 0.2 to 5.0
    proto_temp, ece_after = best_proto_temp(temps)

    if has_ordinal:
        ord_temp, ord_kappa = best_ord_temp(temps)
        print(f"Optimal ordinal temperature: {ord_temp:.4f} (kappa={ord_kappa:.4f})")
    else:
        ord_temp = 1.0
        print(f"Optimal ordinal temperature: {ord_temp:.4f} (no ordinal head)")

    print(f"Optimal prototype temperature: {proto_temp:.4f}")
    print(f"ECE after  calibration:  {ece_after:.4f}")

    model.set_temperatures(ord_temp=ord_temp, proto_temp=proto_temp)
    ckpt["model_state_dict"] = model.state_dict()
    ckpt["ordinal_temperature"] = ord_temp
    ckpt["prototype_temperature"] = proto_temp
    torch.save(ckpt, checkpoint_path)
    print(f"Saved temperatures to checkpoint: {checkpoint_path}")

    return ord_temp, proto_temp


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train_config.yaml")
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    calibrate(cfg, args.checkpoint)
