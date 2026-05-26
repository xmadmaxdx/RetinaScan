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
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval()
    print(f"Loaded checkpoint: {checkpoint_path}")

    val_ds = build_dataset(config, get_val_transform(config), split="val")
    loader = DataLoader(val_ds, batch_size=config["training"]["batch_size"], shuffle=False, num_workers=2)

    all_ord_logits, all_proto_logits, all_labels = [], [], []
    with torch.no_grad():
        for images, labels, *_ in loader:
            images = images.to(device)
            proto_logits, _, ordinal_logits = model(images)
            all_ord_logits.append(ordinal_logits.cpu())
            all_proto_logits.append(proto_logits.cpu())
            all_labels.append(labels)

    ord_logits = torch.cat(all_ord_logits)
    proto_logits = torch.cat(all_proto_logits)
    labels = torch.cat(all_labels).to(device)

    targets = torch.stack([
        (labels >= 1).float(), (labels >= 2).float(),
        (labels >= 3).float(), (labels >= 4).float(),
    ], dim=-1)

    def optimal_temp(logits, targets, loss_fn, n_params=1):
        temp = torch.nn.Parameter(torch.tensor(1.0, device=device))
        opt = torch.optim.LBFGS([temp], lr=0.01, max_iter=100)
        def closure():
            opt.zero_grad()
            loss = loss_fn(logits.to(device) / temp, targets)
            loss.backward()
            return loss
        opt.step(closure)
        return temp.item()

    def ord_nll(scaled, tgt):
        return F.binary_cross_entropy_with_logits(scaled, tgt)

    def proto_nll(scaled, tgt):
        return F.cross_entropy(scaled, tgt.long())

    uncal_probs = torch.softmax(proto_logits, dim=-1)
    ece_before = expected_calibration_error(labels.cpu(), uncal_probs)

    print(f"\nECE before calibration: {ece_before:.4f}")

    ord_temp = optimal_temp(ord_logits, targets, ord_nll)
    proto_temp = optimal_temp(proto_logits, labels, proto_nll)

    cal_probs = torch.softmax(proto_logits.to(device) / proto_temp, dim=-1)
    ece_after = expected_calibration_error(labels.cpu(), cal_probs.cpu())

    print(f"Optimal ordinal temperature: {ord_temp:.4f}")
    print(f"Optimal prototype temperature: {proto_temp:.4f}")
    print(f"ECE after  calibration:  {ece_after:.4f}")

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
