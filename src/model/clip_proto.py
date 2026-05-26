import torch
import torch.nn as nn
import torch.nn.functional as F
import open_clip
from .prototype_bank import TextPrototypeBank, SEVERITY_DESCRIPTIONS


class CoralOrdinalHead(nn.Module):
    def __init__(self, input_dim, num_tasks=4):
        super().__init__()
        self.num_tasks = num_tasks
        self.linear = nn.Linear(input_dim, 1)
        self.biases = nn.Parameter(torch.zeros(num_tasks))

    def forward(self, x):
        shared = self.linear(x)
        return shared + self.biases

    def predict(self, x, threshold=0.0):
        logits = self.forward(x)
        probs = torch.sigmoid(logits)
        grades = (probs > threshold).sum(dim=-1)
        return grades, probs


class CLIPZeroShotNetwork(nn.Module):
    def __init__(self, config, device=None):
        super().__init__()
        self.config = config
        mc = config["model"]
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        clip_model, _, _ = open_clip.create_model_and_transforms(
            mc["backbone"],
            pretrained=mc["pretrained"],
            device=self.device,
        )
        self.clip_model = clip_model
        self.clip_model.eval()
        for p in self.clip_model.parameters():
            p.requires_grad = False

        self.visual_dim = clip_model.visual.output_dim
        self.prototype_dim = mc["prototype_dim"]
        self.temperature = mc["temperature"]
        self.zero_shot_only = mc.get("zero_shot_only", False)
        self.use_ordinal = mc.get("use_ordinal", True)

        self.projection = nn.Sequential(
            nn.Linear(self.visual_dim, self.visual_dim),
            nn.LayerNorm(self.visual_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.visual_dim, self.prototype_dim),
            nn.LayerNorm(self.prototype_dim),
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
    def predict_grade(self, images):
        self.eval()
        if self.zero_shot_only:
            return self.zero_shot_predict(images)
        proto_logits, projected, ordinal_logits = self.forward(images)
        if self.use_ordinal and ordinal_logits is not None:
            grades, _ = self.ordinal_head.predict(projected)
            probs = torch.softmax(proto_logits, dim=-1)
            return grades, probs
        probs = torch.softmax(proto_logits, dim=-1)
        grades = proto_logits.argmax(dim=-1)
        return grades, probs

    def get_text_prototypes(self):
        return self.prototypes.get_prototypes(projection=self.projection)

    def get_prototype_descriptions(self):
        return self.prototypes.descriptions
