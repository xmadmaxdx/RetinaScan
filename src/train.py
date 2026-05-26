import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

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


class HuggingFaceEyePACSDataset(Dataset):
    def __init__(self, hf_dataset_name="bumbledeep/eyepacs", split="train", transform=None):
        from datasets import load_dataset
        self.ds = load_dataset(hf_dataset_name, split=split, streaming=False)
        self.transform = transform

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        row = self.ds[idx]
        img = row["image"].convert("RGB")
        label = row["label_code"]
        if self.transform:
            img = self.transform(img)
        return img, label


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
    metrics = {"sup_loss": 0, "align_loss": 0, "entropy_loss": 0, "diversity_loss": 0, "ordinal_loss": 0}
    pbar = tqdm(loader, desc="Train")
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        with torch.amp.autocast("cuda", enabled=(scaler is not None)):
            logits, projected, ordinal_logits = model(images)
            text_protos = model.get_text_prototypes()
            losses = criterion(logits, projected, text_protos, labels=labels, ordinal_logits=ordinal_logits)

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


def build_dataset(config, transform, split="train"):
    source = config["data"].get("source", "local")
    if source == "huggingface":
        hf_name = config["data"]["hf_dataset"]
        hf_split = config["data"].get("hf_split", "train")
        dataset = HuggingFaceEyePACSDataset(
            hf_dataset_name=hf_name, split=hf_split, transform=transform
        )
        n = len(dataset)
        indices = np.random.permutation(n).tolist()
        train_end = int(n * config["data"]["train_ratio"])
        val_end = train_end + int(n * config["data"]["val_ratio"])
        if split == "train":
            return Subset(dataset, indices[:train_end])
        elif split == "val":
            return Subset(dataset, indices[train_end:val_end])
        return Subset(dataset, indices[val_end:])
    else:
        csv_path = config["data"]["labels_csv"]
        image_dir = config["data"]["processed_path"]
        if not os.path.exists(image_dir):
            image_dir = config["data"]["raw_path"]
        if config["training"].get("patient_level_split", False):
            full_dataset = EyePACSWithPatientID(csv_path, image_dir, transform=transform)
            train_idx, val_idx, _ = patient_level_split(
                full_dataset, config["data"]["train_ratio"], config["data"]["val_ratio"]
            )
            if split == "train":
                return Subset(full_dataset, train_idx)
            return Subset(full_dataset, val_idx)
        else:
            full_dataset = EyePACSDataset(csv_path, image_dir, transform=transform)
            n = len(full_dataset)
            indices = np.random.permutation(n).tolist()
            train_end = int(n * config["data"]["train_ratio"])
            val_end = train_end + int(n * config["data"]["val_ratio"])
            if split == "train":
                return Subset(full_dataset, indices[:train_end])
            return Subset(full_dataset, indices[train_end:val_end])


def sync_to_drive(src_path, drive_dir):
    if not os.path.exists(drive_dir):
        os.makedirs(drive_dir, exist_ok=True)
    import shutil
    shutil.copy2(src_path, os.path.join(drive_dir, os.path.basename(src_path)))
    print(f"Synced to Drive: {drive_dir}")


def save_checkpoint(path, model, optimizer, scheduler, epoch, best_acc, drive_path=None):
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "best_acc": best_acc,
        "text_descriptions": model.get_prototype_descriptions(),
    }, path)
    print(f"Checkpoint saved -> {path}")
    if drive_path:
        sync_to_drive(path, drive_path)


def load_checkpoint(path, model, optimizer, scheduler, device):
    ckpt = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    return ckpt["epoch"], ckpt["best_acc"]


def main(config, drive_path=None, resume=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Data source: {config['data'].get('source', 'local')}")
    mc = config["model"]

    train_ds = build_dataset(config, get_train_transform(config), split="train")
    val_ds = build_dataset(config, get_val_transform(config), split="val")

    train_loader = DataLoader(train_ds, batch_size=config["training"]["batch_size"], shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=config["training"]["batch_size"], shuffle=False, num_workers=2)

    model = CLIPZeroShotNetwork(config, device=device)

    if mc.get("zero_shot_only", False):
        print("Pure zero-shot mode: no training, evaluating directly...")
        val_acc = validate(model, val_loader, device)
        print(f"Zero-shot validation accuracy: {val_acc:.4f}")
        return

    trainable_params = list(filter(lambda p: p.requires_grad, model.parameters()))
    total_trainable = sum(p.numel() for p in trainable_params)
    print(f"Trainable parameters: {total_trainable:,} (projection head only)")

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )

    total_epochs = config["training"]["epochs"]
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs)
    criterion = TextPrototypeLoss(
        temperature=config["model"]["temperature"],
        entropy_weight=config["loss"]["entropy_weight"],
        diversity_weight=config["loss"]["diversity_weight"],
        supervised_weight=config["loss"]["supervised_weight"],
        ordinal_weight=config["loss"].get("ordinal_weight", 0.5),
    )

    ckpt_dir = config["paths"]["checkpoint_dir"]
    latest_path = os.path.join(ckpt_dir, "latest.pt")
    best_path = os.path.join(ckpt_dir, "best.pt")
    os.makedirs(ckpt_dir, exist_ok=True)
    if drive_path:
        os.makedirs(drive_path, exist_ok=True)

    start_epoch = 0
    best_acc = 0.0

    if resume:
        resume_path = latest_path if os.path.exists(latest_path) else best_path
        if os.path.exists(resume_path):
            start_epoch, best_acc = load_checkpoint(resume_path, model, optimizer, scheduler, device)
            start_epoch += 1
            print(f"Resuming from epoch {start_epoch+1}/{total_epochs} (best_acc={best_acc:.4f})")
        else:
            print("No checkpoint found for resume — starting from scratch")

    scaler = torch.amp.GradScaler("cuda", enabled=(config["training"]["mixed_precision"] and torch.cuda.is_available()))

    for epoch in range(start_epoch, total_epochs):
        print(f"\nEpoch {epoch+1}/{total_epochs}")
        train_loss, train_metrics = train_epoch(model, train_loader, optimizer, criterion, device, scaler)
        val_acc = validate(model, val_loader, device)
        scheduler.step()
        lr = optimizer.param_groups[0]["lr"]
        print(f"Loss: {train_loss:.4f} | Acc: {val_acc:.4f} | LR: {lr:.2e}")
        print(f"  sup={train_metrics['sup_loss']:.4f} align={train_metrics['align_loss']:.4f} ent={train_metrics['entropy_loss']:.4f} div={train_metrics['diversity_loss']:.4f} ord={train_metrics['ordinal_loss']:.4f}")

        save_checkpoint(latest_path, model, optimizer, scheduler, epoch, best_acc, drive_path)

        if val_acc > best_acc:
            best_acc = val_acc
            save_checkpoint(best_path, model, optimizer, scheduler, epoch, best_acc, drive_path)

    if drive_path:
        log_dir = config["paths"]["log_dir"]
        if os.path.exists(log_dir):
            import shutil
            for f in os.listdir(log_dir):
                shutil.copy2(os.path.join(log_dir, f), os.path.join(drive_path, f))
            print(f"Logs synced to Drive: {drive_path}")

    print(f"Done. Best val acc: {best_acc:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train_config.yaml")
    parser.add_argument("--zero-shot", action="store_true", help="Run pure zero-shot, no training")
    parser.add_argument("--drive-path", default=None, help="Sync checkpoints to Google Drive path")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoints/latest.pt")
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    if args.zero_shot:
        cfg["model"]["zero_shot_only"] = True
    main(cfg, drive_path=args.drive_path, resume=args.resume)
