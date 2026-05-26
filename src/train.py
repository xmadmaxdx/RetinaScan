import os
import csv
import yaml
import argparse
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from tqdm import tqdm
import numpy as np
from src.model.clip_proto import CLIPZeroShotNetwork
from src.losses.proto_loss import TextPrototypeLoss


class EyePACSDataset(Dataset):
    def __init__(self, csv_path, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.samples = []
        with open(csv_path, newline="") as f:
            reader = csv.reader(f)
            next(reader, None)
            for row in reader:
                image_id, label = row[0], int(row[1])
                for ext in [".jpeg", ".jpg", ".png"]:
                    path = os.path.join(image_dir, image_id + ext)
                    if os.path.exists(path):
                        self.samples.append((path, label))
                        break

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        from PIL import Image
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


class EyePACSWithPatientID(EyePACSDataset):
    def __init__(self, csv_path, image_dir, transform=None):
        super().__init__(csv_path, image_dir, transform)
        self.patient_ids = []
        with open(csv_path, newline="") as f:
            reader = csv.reader(f)
            next(reader, None)
            for row in reader:
                self.patient_ids.append(row[0].split("_")[0])

    def __getitem__(self, idx):
        img, label = super().__getitem__(idx)
        return img, label, self.patient_ids[idx]


def get_train_transform(config):
    ac = config["augmentation"]
    return transforms.Compose([
        transforms.Resize((config["data"]["image_size"], config["data"]["image_size"])),
        transforms.RandomResizedCrop(ac["random_crop"]),
        transforms.RandomHorizontalFlip(ac.get("horizontal_flip", True)),
        transforms.ColorJitter(brightness=ac["color_jitter"], contrast=ac["color_jitter"]),
        transforms.GaussianBlur(kernel_size=5, sigma=ac.get("gaussian_blur", 0.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def get_val_transform(config):
    size = config["data"]["image_size"]
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def patient_level_split(dataset, train_ratio=0.8, val_ratio=0.1):
    patients = sorted(set(dataset.patient_ids))
    np.random.shuffle(patients)
    n = len(patients)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)
    train_patients = set(patients[:train_end])
    val_patients = set(patients[train_end:val_end])
    test_patients = set(patients[val_end:])

    train_idx = [i for i, p in enumerate(dataset.patient_ids) if p in train_patients]
    val_idx = [i for i, p in enumerate(dataset.patient_ids) if p in val_patients]
    test_idx = [i for i, p in enumerate(dataset.patient_ids) if p in test_patients]
    return train_idx, val_idx, test_idx


def train_epoch(model, loader, optimizer, criterion, device, scaler=None):
    model.train()
    total_loss = 0
    metrics = {"sup_loss": 0, "align_loss": 0, "entropy_loss": 0, "diversity_loss": 0}
    pbar = tqdm(loader, desc="Train")
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        with torch.amp.autocast("cuda", enabled=(scaler is not None)):
            logits, projected = model(images)
            text_protos = model.get_text_prototypes()
            losses = criterion(logits, projected, text_protos, labels=labels)

        if scaler:
            scaler.scale(losses["loss"]).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            losses["loss"].backward()
            optimizer.step()

        total_loss += losses["loss"].item()
        for k in metrics:
            metrics[k] += losses[k]
        pbar.set_postfix(loss=losses["loss"].item())
    n = len(loader)
    return total_loss / n, {k: v / n for k, v in metrics.items()}


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    for images, labels in tqdm(loader, desc="Val"):
        images, labels = images.to(device), labels.to(device)
        grades, probs = model.predict_grade(images)
        correct += (grades == labels).sum().item()
        total += labels.size(0)
    return correct / total


def main(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    mc = config["model"]

    csv_path = config["data"]["labels_csv"]
    image_dir = config["data"]["processed_path"]
    if not os.path.exists(image_dir):
        image_dir = config["data"]["raw_path"]

    do_patient_split = config["training"].get("patient_level_split", False)
    if do_patient_split:
        full_dataset = EyePACSWithPatientID(csv_path, image_dir, transform=get_train_transform(config))
        train_idx, val_idx, test_idx = patient_level_split(
            full_dataset, config["data"]["train_ratio"], config["data"]["val_ratio"]
        )
        train_ds = Subset(full_dataset, train_idx)
        val_dataset = EyePACSWithPatientID(csv_path, image_dir, transform=get_val_transform(config))
        val_ds = Subset(val_dataset, val_idx)
    else:
        full_dataset = EyePACSDataset(csv_path, image_dir, transform=get_train_transform(config))
        n = len(full_dataset)
        train_len = int(n * config["data"]["train_ratio"])
        val_len = int(n * config["data"]["val_ratio"])
        indices = np.random.permutation(n).tolist()
        train_idx = indices[:train_len]
        val_idx = indices[train_len:train_len + val_len]
        train_ds = Subset(full_dataset, train_idx)
        val_dataset = EyePACSDataset(csv_path, image_dir, transform=get_val_transform(config))
        val_ds = Subset(val_dataset, val_idx)

    train_loader = DataLoader(train_ds, batch_size=config["training"]["batch_size"], shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=config["training"]["batch_size"], shuffle=False, num_workers=2)

    model = CLIPZeroShotNetwork(config, device=device)

    if mc.get("zero_shot_only", False):
        print("Pure zero-shot mode: no training, evaluating directly...")
        val_acc = validate(model, val_loader, device)
        print(f"Zero-shot validation accuracy: {val_acc:.4f}")
        return

    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    total_trainable = sum(p.numel() for p in trainable_params)
    print(f"Trainable parameters: {total_trainable:,} (projection head only)")

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["training"]["epochs"])
    criterion = TextPrototypeLoss(
        temperature=config["model"]["temperature"],
        entropy_weight=config["loss"]["entropy_weight"],
        diversity_weight=config["loss"]["diversity_weight"],
        supervised_weight=config["loss"]["supervised_weight"],
    )

    os.makedirs(config["paths"]["checkpoint_dir"], exist_ok=True)
    best_acc = 0.0
    scaler = torch.amp.GradScaler("cuda", enabled=(config["training"]["mixed_precision"] and torch.cuda.is_available()))

    for epoch in range(config["training"]["epochs"]):
        print(f"\nEpoch {epoch+1}/{config['training']['epochs']}")
        train_loss, train_metrics = train_epoch(model, train_loader, optimizer, criterion, device, scaler)
        val_acc = validate(model, val_loader, device)
        scheduler.step()
        lr = optimizer.param_groups[0]["lr"]
        print(f"Loss: {train_loss:.4f} | Acc: {val_acc:.4f} | LR: {lr:.2e}")
        print(f"  sup={train_metrics['sup_loss']:.4f} align={train_metrics['align_loss']:.4f} ent={train_metrics['entropy_loss']:.4f} div={train_metrics['diversity_loss']:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            ckpt_path = os.path.join(config["paths"]["checkpoint_dir"], "best.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_acc": best_acc,
                "text_descriptions": model.get_prototype_descriptions(),
            }, ckpt_path)
            print(f"Checkpoint saved -> {ckpt_path}")

    print(f"Done. Best val acc: {best_acc:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train_config.yaml")
    parser.add_argument("--zero-shot", action="store_true", help="Run pure zero-shot, no training")
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    if args.zero_shot:
        cfg["model"]["zero_shot_only"] = True
    main(cfg)
