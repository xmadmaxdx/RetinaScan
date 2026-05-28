import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
import open_clip
from torchvision import transforms as T
from .prototype_bank import TextPrototypeBank, SEVERITY_DESCRIPTIONS


class CoralOrdinalHead(nn.Module):
    def __init__(self, input_dim, num_tasks=4):
        super().__init__()
        self.num_tasks = num_tasks
        self.linear = nn.Linear(input_dim, num_tasks)

    def forward(self, x):
        return self.linear(x)

    def predict(self, x, threshold=0.0):
        logits = self.forward(x)
        grades = (logits > threshold).sum(dim=-1)
        probs = torch.sigmoid(logits)
        return grades, probs


class CLIPZeroShotNetwork(nn.Module):
    def __init__(self, config, device=None):
        super().__init__()
        self.config = config
        mc = config["model"]
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*QuickGELU.*")
            clip_model, _, _ = open_clip.create_model_and_transforms(
                mc["backbone"],
                pretrained=mc["pretrained"],
                device=self.device,
            )
        self.clip_model = clip_model
        self.clip_model.eval()
        for p in self.clip_model.parameters():
            p.requires_grad = False

        image_size = config["data"].get("image_size", 224)
        patch_size = getattr(clip_model.visual, "patch_size", 16)
        if isinstance(patch_size, (tuple, list)):
            patch_size = patch_size[0]
        pretrained_pos = clip_model.visual.positional_embedding
        n_pretrained = pretrained_pos.shape[0]  # 197 for 224x224
        pretrained_grid = int((n_pretrained - 1) ** 0.5)
        target_grid = image_size // patch_size
        if target_grid != pretrained_grid:
            cls_token = pretrained_pos[0:1]
            patch_embeds = pretrained_pos[1:]
            patch_embeds = patch_embeds.reshape(1, pretrained_grid, pretrained_grid, -1)
            patch_embeds = patch_embeds.permute(0, 3, 1, 2)
            patch_embeds = F.interpolate(patch_embeds, size=(target_grid, target_grid), mode="bicubic", align_corners=False)
            patch_embeds = patch_embeds.permute(0, 2, 3, 1).reshape(-1, pretrained_pos.shape[-1])
            clip_model.visual.positional_embedding = nn.Parameter(torch.cat([cls_token, patch_embeds], dim=0))
            print(f"  Interpolated positional embeddings: {n_pretrained} → {target_grid**2 + 1} positions")

        self.visual_dim = clip_model.visual.output_dim
        self.prototype_dim = mc["prototype_dim"]
        self.temperature = mc["temperature"]
        self.zero_shot_only = mc.get("zero_shot_only", False)
        self.use_ordinal = mc.get("use_ordinal", True)

        self.projection = nn.Sequential(
            nn.Linear(self.visual_dim, self.visual_dim),
            nn.LayerNorm(self.visual_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(self.visual_dim, self.prototype_dim),
            nn.LayerNorm(self.prototype_dim),
            nn.Dropout(p=0.2),
        ).to(self.device)

        descriptions = mc.get("severities", SEVERITY_DESCRIPTIONS)
        self.prototypes = TextPrototypeBank(
            clip_model=clip_model,
            prototype_dim=self.prototype_dim,
            temperature=mc["temperature"],
            descriptions=descriptions,
        ).to(self.device)

        if not self.zero_shot_only:
            self.ordinal_head = CoralOrdinalHead(
                input_dim=self.prototype_dim, num_tasks=4
            ).to(self.device)

        self.ordinal_temperature = nn.Parameter(torch.tensor(1.0), requires_grad=False)
        self.prototype_temperature = nn.Parameter(torch.tensor(1.0), requires_grad=False)

        if self.zero_shot_only:
            for p in self.projection.parameters():
                p.requires_grad = False

    def forward(self, images):
        with torch.no_grad():
            raw_features = self.clip_model.encode_image(images).float()
        projected = F.normalize(self.projection(raw_features), dim=-1)
        proto_logits, text_protos = self.prototypes(projected, self.projection)
        ordinal_logits = None
        if not self.zero_shot_only and self.use_ordinal:
            ordinal_logits = self.ordinal_head(projected)
        return proto_logits, projected, ordinal_logits

    def forward_gradcam(self, images):
        raw_features = self.clip_model.encode_image(images).float()
        projected = F.normalize(self.projection(raw_features), dim=-1)
        proto_logits, text_protos = self.prototypes(projected, self.projection)
        ordinal_logits = None
        if not self.zero_shot_only and self.use_ordinal:
            ordinal_logits = self.ordinal_head(projected)
        return proto_logits, projected, ordinal_logits

    @torch.no_grad()
    def zero_shot_predict(self, images):
        raw_features = self.clip_model.encode_image(images).float()
        img_norm = F.normalize(raw_features, dim=-1)
        raw_text = F.normalize(self.prototypes.raw_text_features, dim=-1)
        logits = img_norm @ raw_text.T / self.temperature
        probs = torch.softmax(logits, dim=-1)
        grades = logits.argmax(dim=-1)
        return grades, probs

    def extract_features(self, images):
        with torch.no_grad():
            return self.clip_model.encode_image(images).float()

    @torch.no_grad()
    def predict_grade(self, images, thresholds=None):
        self.eval()
        if self.zero_shot_only:
            return self.zero_shot_predict(images)
        proto_logits, projected, ordinal_logits = self.forward(images)
        if self.use_ordinal and ordinal_logits is not None:
            cal_ord = ordinal_logits / self.ordinal_temperature
            if thresholds is not None:
                thresh = torch.as_tensor(thresholds, device=cal_ord.device, dtype=cal_ord.dtype)
            else:
                thresh = 0.0
            grades = (cal_ord > thresh).sum(dim=-1)
            cal_proto = proto_logits / self.prototype_temperature
            probs = torch.softmax(cal_proto, dim=-1)
            return grades, probs
        cal_proto = proto_logits / self.prototype_temperature
        probs = torch.softmax(cal_proto, dim=-1)
        grades = proto_logits.argmax(dim=-1)
        return grades, probs

    def predict_with_uncertainty(self, images_pil, n_runs=20):
        self.train()
        device = self.device
        tta_size = self.config.get("data", {}).get("image_size", 224)
        tta = T.Compose([
            T.Resize((tta_size, tta_size)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomAffine(degrees=5, translate=(0.05, 0.05)),
            T.ColorJitter(brightness=0.1, contrast=0.1),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        all_grades, all_probs = [], []
        for _ in range(n_runs):
            batch = torch.stack([tta(img) for img in images_pil]).to(device)
            with torch.no_grad():
                raw = self.clip_model.encode_image(batch).float()
                proj = F.normalize(self.projection(raw), dim=-1)
                proto_logits, _ = self.prototypes(proj, self.projection)
                if self.use_ordinal and not self.zero_shot_only:
                    ordinal_logits = self.ordinal_head(proj)
                    cal_ord = ordinal_logits / self.ordinal_temperature
                    grades = (cal_ord > 0.0).sum(dim=-1)
                else:
                    grades = proto_logits.argmax(dim=-1)
                cal_proto = proto_logits / self.prototype_temperature
                probs = torch.softmax(cal_proto, dim=-1)
            all_grades.append(grades)
            all_probs.append(probs)
        self.eval()
        stacked_probs = torch.stack(all_probs, dim=0)
        stacked_grades = torch.stack(all_grades, dim=0)
        mean_probs = stacked_probs.mean(dim=0)
        mean_grade = stacked_grades.float().mean(dim=0)
        entropy = -(mean_probs * torch.log(mean_probs + 1e-8)).sum(dim=-1)
        max_entropy = torch.log(torch.tensor(5.0, device=device))
        confidence = 1.0 - (entropy / max_entropy)
        return mean_grade, confidence, mean_probs

    def set_temperatures(self, ord_temp=None, proto_temp=None):
        if ord_temp is not None:
            self.ordinal_temperature.fill_(ord_temp)
        if proto_temp is not None:
            self.prototype_temperature.fill_(proto_temp)

    def get_text_prototypes(self):
        return self.prototypes.get_prototypes(projection=self.projection)

    def get_prototype_descriptions(self):
        return self.prototypes.descriptions
