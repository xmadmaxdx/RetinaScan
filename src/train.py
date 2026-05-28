import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import csv
from collections import defaultdict
import yaml
import argparse
import torch
from torch.utils.data import DataLoader, Dataset, Sampler, Subset
from torchvision import transforms
from tqdm import tqdm
import numpy as np
from src.model.clip_proto import CLIPZeroShotNetwork
from src.losses.balanced_loss import ClassWeightedCORALLoss, PrototypeFocalLoss


class HuggingFaceEyePACSDataset(Dataset):
    def __init__(self, hf_dataset_name="bumbledeep/eyepacs", split="train", transform=None, heavy_transform=None):
        from datasets import load_dataset
        self.ds = load_dataset(hf_dataset_name, split=split, streaming=False)
        self.labels = self.ds["label_code"]
        self.transform = transform
        self.heavy_transform = heavy_transform

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        row = self.ds[idx]
        img = row["image"].convert("RGB")
        label = row["label_code"]
        if label >= 3 and self.heavy_transform is not None:
            img = self.heavy_transform(img)
        elif self.transform is not None:
            img = self.transform(img)
        return img, label


class EyePACSDataset(Dataset):
    def __init__(self, csv_path, image_dir, transform=None, heavy_transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.heavy_transform = heavy_transform
        self.samples = []
        self._labels = []
        with open(csv_path, newline="") as f:
            reader = csv.reader(f)
            next(reader, None)
            for row in reader:
                image_id, label = row[0], int(row[1])
                for ext in [".jpeg", ".jpg", ".png"]:
                    path = os.path.join(image_dir, image_id + ext)
                    if os.path.exists(path):
                        self.samples.append((path, label))
                        self._labels.append(label)
                        break

    @property
    def labels(self):
        return self._labels

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        from PIL import Image
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if label >= 3 and self.heavy_transform is not None:
            img = self.heavy_transform(img)
        elif self.transform is not None:
            img = self.transform(img)
        return img, label


class EyePACSWithPatientID(EyePACSDataset):
    def __init__(self, csv_path, image_dir, transform=None, heavy_transform=None):
        super().__init__(csv_path, image_dir, transform, heavy_transform)
        self.patient_ids = []
        with open(csv_path, newline="") as f:
            reader = csv.reader(f)
            next(reader, None)
            for row in reader:
                self.patient_ids.append(row[0].split("_")[0])

    def __getitem__(self, idx):
        img, label = super().__getitem__(idx)
        return img, label, self.patient_ids[idx]


class MergedDataset(Dataset):
    def __init__(self, csv_path, transform=None, heavy_transform=None):
        self.transform = transform
        self.heavy_transform = heavy_transform
        self.samples = []
        self._labels = []
        self._sources = []
        self._source_set = set()
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.samples.append((row["image_path"], int(row["grade"])))
                self._labels.append(int(row["grade"]))
                src = row.get("source", "unknown")
                self._sources.append(src)
                self._source_set.add(src)

    @property
    def labels(self):
        return self._labels

    @property
    def sources(self):
        return self._sources

    @property
    def source_set(self):
        return self._source_set

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        from PIL import Image
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if label >= 3 and self.heavy_transform is not None:
            img = self.heavy_transform(img)
        elif self.transform is not None:
            img = self.transform(img)
        return img, label


class BalancedStageSampler(Sampler):
    def __init__(self, labels, batch_size):
        self.labels = np.array(labels)
        self.batch_size = batch_size
        self.num_classes = 5
        self.per_class = batch_size // self.num_classes
        assert batch_size % self.num_classes == 0, "batch_size must be multiple of 5"
        self.class_indices = {
            c: np.where(self.labels == c)[0].tolist() for c in range(self.num_classes)
        }
        for c, idxs in self.class_indices.items():
            if len(idxs) == 0:
                print(f"  ⚠ Class {c} has 0 samples — will be absent from training")
                self.class_indices[c] = [0] * self.per_class
            elif len(idxs) < self.per_class:
                repeats = (self.per_class // len(idxs)) + 1
                self.class_indices[c] = (idxs * repeats)[:self.per_class]
        self.num_batches = len(self.labels) // batch_size

    def __iter__(self):
        for c in range(self.num_classes):
            np.random.shuffle(self.class_indices[c])
        ptr = {c: 0 for c in range(self.num_classes)}
        indices = []
        for _ in range(self.num_batches):
            batch = []
            for c in range(self.num_classes):
                start = ptr[c]
                end = start + self.per_class
                n = len(self.class_indices[c])
                if end > n:
                    np.random.shuffle(self.class_indices[c])
                    start = 0
                    end = min(self.per_class, n)
                    ptr[c] = 0
                batch.extend(self.class_indices[c][start:end])
                ptr[c] = end
            np.random.shuffle(batch)
            indices.extend(batch)
        return iter(indices)

    def __len__(self):
        return self.num_batches * self.batch_size


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


def get_heavy_transform(config):
    ac = config["augmentation"]
    return transforms.Compose([
        transforms.Resize((config["data"]["image_size"], config["data"]["image_size"])),
        transforms.RandomResizedCrop(ac["random_crop"]),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.85, 1.15)),
        transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.4, hue=0.1),
        transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
        transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.3),
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


def patient_level_split(dataset, train_ratio=0.8, val_ratio=0.1, seed=42):
    patients = sorted(set(dataset.patient_ids))
    rng = np.random.RandomState(seed)
    rng.shuffle(patients)
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


def source_aware_split(dataset, train_r=0.8, val_r=0.1, seed=42):
    rng = np.random.RandomState(seed)
    by_source = defaultdict(list)
    for i, src in enumerate(dataset.sources):
        by_source[src].append(i)

    train_idx, val_idx, test_idx = [], [], []
    for src, idxs in by_source.items():
        idxs = rng.permutation(idxs).tolist()
        n = len(idxs)
        t = int(n * train_r)
        v = int(n * val_r)
        train_idx.extend(idxs[:t])
        val_idx.extend(idxs[t:t+v])
        test_idx.extend(idxs[t+v:])

    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)
    return train_idx, val_idx, test_idx


def train_epoch(model, loader, optimizer, loss_coral, loss_proto, device, epoch=0, scaler=None, grad_clip=1.0):
    model.train()
    total_loss = 0
    coral_loss_sum = 0
    proto_loss_sum = 0
    cw = 0.1 if epoch < 5 else 1.0
    pw = 1.0 if epoch < 5 else 0.2
    pbar = tqdm(loader, desc="Train")
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        with torch.amp.autocast("cuda", enabled=(scaler is not None)):
            _, projected, ordinal_logits = model(images)
            text_protos = model.get_text_prototypes()

            targets = torch.zeros(len(labels), 4, device=device)
            for k in range(4):
                targets[:, k] = (labels > k).float()

            lc = loss_coral(ordinal_logits, targets)
            lp = loss_proto(projected, text_protos, labels)
            loss = cw * lc + pw * lp

        if scaler:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()

        total_loss += loss.item()
        coral_loss_sum += lc.item()
        proto_loss_sum += lp.item()
        pbar.set_postfix(loss=loss.item())
    n = len(loader)
    return total_loss / n, {"coral_loss": coral_loss_sum / n, "proto_loss": proto_loss_sum / n}


@torch.no_grad()
def validate(model, loader, device):
    from sklearn.metrics import cohen_kappa_score
    model.eval()
    if model.zero_shot_only:
        correct = 0
        total = 0
        for images, labels in tqdm(loader, desc="Val"):
            images, labels = images.to(device), labels.to(device)
            grades, _ = model.predict_grade(images)
            correct += (grades == labels).sum().item()
            total += labels.size(0)
        acc = correct / total
        return acc, acc, 1.0, 0.0

    all_ord_logits = []
    all_labels = []
    all_proto_raw = []
    for images, labels in tqdm(loader, desc="Val"):
        images, labels = images.to(device), labels.to(device)
        proto_logits, _, ordinal_logits = model(images)
        if ordinal_logits is None:
            all_proto_raw.append(proto_logits.argmax(dim=-1).cpu())
        else:
            all_ord_logits.append(ordinal_logits.cpu())
        all_labels.append(labels.cpu())
    labels_cat = torch.cat(all_labels)
    n_total = len(labels_cat)

    if len(all_proto_raw) > 0:
        proto_preds = torch.cat(all_proto_raw)
        acc = (proto_preds == labels_cat).float().mean().item()
        kappa = cohen_kappa_score(labels_cat.numpy(), proto_preds.numpy(), weights="quadratic")
        return acc, acc, 1.0, kappa

    ord_logits = torch.cat(all_ord_logits)
    labels_cat = torch.cat(all_labels)
    n_total = len(labels_cat)

    raw_preds = (ord_logits / model.ordinal_temperature.cpu() > 0.0).sum(dim=-1)
    raw_acc = (raw_preds == labels_cat).float().mean().item()
    raw_kappa = cohen_kappa_score(labels_cat.numpy(), raw_preds.numpy(), weights="quadratic")

    best_correct = 0
    best_temp = 1.0
    for t in [i * 0.1 for i in range(2, 51)]:
        preds = (ord_logits / t > 0.0).sum(dim=-1)
        n_correct = (preds == labels_cat).sum().item()
        if n_correct > best_correct:
            best_correct = n_correct
            best_temp = t
    cal_acc = best_correct / n_total

    return raw_acc, cal_acc, best_temp, raw_kappa


def build_dataset(config, transform, split="train", heavy_transform=None):
    source = config["data"].get("source", "local")
    if source == "merged":
        csv_path = config["data"]["merged_csv"]
        full_dataset = MergedDataset(
            csv_path, transform=transform, heavy_transform=heavy_transform,
        )
        train_idx, val_idx, test_idx = source_aware_split(
            full_dataset,
            train_r=config["data"]["train_ratio"],
            val_r=config["data"]["val_ratio"],
        )
        if split == "train":
            return Subset(full_dataset, train_idx)
        elif split == "val":
            return Subset(full_dataset, val_idx)
        return Subset(full_dataset, test_idx)
    elif source == "huggingface":
        hf_name = config["data"]["hf_dataset"]
        hf_base = config["data"].get("hf_split", "train")
        tr = config["data"]["train_ratio"]
        vr = config["data"]["val_ratio"]
        pct_map = {
            "train": f"{hf_base}[:{int(tr*100)}%]",
            "val": f"{hf_base}[{int(tr*100)}%:{int((tr+vr)*100)}%]",
            "test": f"{hf_base}[{int((tr+vr)*100)}%:]",
        }
        dataset = HuggingFaceEyePACSDataset(
            hf_dataset_name=hf_name, split=pct_map.get(split, split),
            transform=transform, heavy_transform=heavy_transform,
        )
        return Subset(dataset, range(len(dataset)))
    else:
        csv_path = config["data"]["labels_csv"]
        image_dir = config["data"]["processed_path"]
        if not os.path.exists(image_dir):
            image_dir = config["data"]["raw_path"]
        if config["training"].get("patient_level_split", False):
            full_dataset = EyePACSWithPatientID(csv_path, image_dir, transform=transform, heavy_transform=heavy_transform)
            train_idx, val_idx, test_idx = patient_level_split(
                full_dataset, config["data"]["train_ratio"], config["data"]["val_ratio"]
            )
            if split == "train":
                return Subset(full_dataset, train_idx)
            elif split == "val":
                return Subset(full_dataset, val_idx)
            return Subset(full_dataset, test_idx)
        else:
            full_dataset = EyePACSDataset(csv_path, image_dir, transform=transform, heavy_transform=heavy_transform)
            n = len(full_dataset)
            rng = np.random.RandomState(42)
            indices = rng.permutation(n).tolist()
            train_end = int(n * config["data"]["train_ratio"])
            val_end = train_end + int(n * config["data"]["val_ratio"])
            if split == "train":
                return Subset(full_dataset, indices[:train_end])
            elif split == "val":
                return Subset(full_dataset, indices[train_end:val_end])
            return Subset(full_dataset, indices[val_end:])


def sync_to_drive(src_path, drive_dir):
    if not os.path.exists(drive_dir):
        os.makedirs(drive_dir, exist_ok=True)
    import shutil
    shutil.copy2(src_path, os.path.join(drive_dir, os.path.basename(src_path)))
    print(f"Synced to Drive: {drive_dir}")


def save_checkpoint(path, model, optimizer, scheduler, epoch, best_kappa, drive_path=None):
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "best_kappa": best_kappa,
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
    epoch = ckpt["epoch"]
    best_kappa = ckpt.get("best_kappa", 0.0)
    return epoch, best_kappa


def main(config, drive_path=None, resume=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Data source: {config['data'].get('source', 'local')}")
    mc = config["model"]

    train_ds = build_dataset(config, get_train_transform(config), split="train", heavy_transform=get_heavy_transform(config))
    val_ds = build_dataset(config, get_val_transform(config), split="val")

    train_labels = [train_ds.dataset.labels[i] for i in train_ds.indices]
    train_sampler = BalancedStageSampler(train_labels, batch_size=config["training"]["batch_size"])
    train_loader = DataLoader(train_ds, batch_size=config["training"]["batch_size"], sampler=train_sampler, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=config["training"]["batch_size"], shuffle=False, num_workers=2)

    model = CLIPZeroShotNetwork(config, device=device)

    if mc.get("zero_shot_only", False):
        print("Pure zero-shot mode: no training, evaluating directly...")
        raw_acc, _, _, _ = validate(model, val_loader, device)
        print(f"Zero-shot validation accuracy: {raw_acc:.4f}")
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
    warmup_epochs = config["training"].get("warmup_epochs", 0)
    if warmup_epochs > 0:
        warmup_sched = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.01, total_iters=warmup_epochs
        )
        cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(total_epochs - warmup_epochs, 1)
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup_sched, cosine_sched], milestones=[warmup_epochs]
        )
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs)
    loss_coral = ClassWeightedCORALLoss()
    loss_proto = PrototypeFocalLoss(gamma=2.5)

    ckpt_dir = config["paths"]["checkpoint_dir"]
    latest_path = os.path.join(ckpt_dir, "latest.pt")
    best_path = os.path.join(ckpt_dir, "best.pt")
    os.makedirs(ckpt_dir, exist_ok=True)
    if drive_path:
        os.makedirs(drive_path, exist_ok=True)

    start_epoch = 0
    best_kappa = -1.0

    if resume:
        resume_path = latest_path if os.path.exists(latest_path) else best_path
        if os.path.exists(resume_path):
            start_epoch, best_kappa = load_checkpoint(resume_path, model, optimizer, scheduler, device)
            start_epoch += 1
            print(f"Resuming from epoch {start_epoch+1}/{total_epochs} (best_kappa={best_kappa:.4f})")
        else:
            print("No checkpoint found for resume — starting from scratch")

    scaler = torch.amp.GradScaler("cuda", enabled=(config["training"]["mixed_precision"] and torch.cuda.is_available()))
    grad_clip = config["training"].get("gradient_clip", 1.0)

    for epoch in range(start_epoch, total_epochs):
        print(f"\nEpoch {epoch+1}/{total_epochs}")
        train_loss, train_metrics = train_epoch(model, train_loader, optimizer, loss_coral, loss_proto, device, epoch=epoch, scaler=scaler, grad_clip=grad_clip)
        raw_acc, cal_acc, cal_temp, kappa = validate(model, val_loader, device)
        scheduler.step()
        lr = optimizer.param_groups[0]["lr"]
        print(f"Loss: {train_loss:.4f} | Raw: {raw_acc:.4f} | Cal: {cal_acc:.4f} | Kappa: {kappa:.4f} | LR: {lr:.2e}")
        print(f"  coral={train_metrics['coral_loss']:.4f} proto={train_metrics['proto_loss']:.4f} temp={cal_temp:.2f}")

        save_checkpoint(latest_path, model, optimizer, scheduler, epoch, best_kappa, drive_path)

        if kappa > best_kappa:
            best_kappa = kappa
            save_checkpoint(best_path, model, optimizer, scheduler, epoch, best_kappa, drive_path)

    if drive_path:
        log_dir = config["paths"]["log_dir"]
        if os.path.exists(log_dir):
            import shutil
            for f in os.listdir(log_dir):
                shutil.copy2(os.path.join(log_dir, f), os.path.join(drive_path, f))
            print(f"Logs synced to Drive: {drive_path}")

    print(f"Done. Best val kappa: {best_kappa:.4f}")


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
