import torch
import torch.nn as nn
import torch.nn.functional as F
import open_clip


SEVERITY_DESCRIPTIONS = [
    "no diabetic retinopathy, healthy retina with normal blood vessels, optic disc, and macula",
    "mild nonproliferative diabetic retinopathy with only a few microaneurysms, no hemorrhages or exudates",
    "moderate nonproliferative diabetic retinopathy with microaneurysms, dot-blot hemorrhages, hard exudates, and cotton wool spots",
    "severe nonproliferative diabetic retinopathy with venous beading, intraretinal hemorrhages in four quadrants, and IRMA",
    "proliferative diabetic retinopathy with neovascularization, vitreous hemorrhage, and high-risk characteristics",
]

SEVERITY_LABELS = [
    "Grade 0 — No DR",
    "Grade 1 — Mild NPDR",
    "Grade 2 — Moderate NPDR",
    "Grade 3 — Severe NPDR",
    "Grade 4 — Proliferative DR",
]


class TextPrototypeBank(nn.Module):
    def __init__(self, clip_model, prototype_dim=512, temperature=0.07, descriptions=None):
        super().__init__()
        self.prototype_dim = prototype_dim
        self.temperature = temperature
        self.descriptions = descriptions or SEVERITY_DESCRIPTIONS
        self.num_prototypes = len(self.descriptions)

        with torch.no_grad():
            text_tokens = open_clip.tokenize(self.descriptions)
            text_tokens = text_tokens.to(next(clip_model.parameters()).device)
            text_features = clip_model.encode_text(text_tokens).float()
            self.register_buffer("raw_text_features", F.normalize(text_features, dim=-1))

    def forward(self, projected_features, projection):
        text_projected = projection(self.raw_text_features)
        text_prototypes = F.normalize(text_projected, dim=-1)
        logits = projected_features @ text_prototypes.T / self.temperature
        return logits, text_prototypes

    def get_prototypes(self, projection=None):
        if projection is not None and self.raw_text_features.device != next(projection.parameters()).device:
            projection = projection.to(self.raw_text_features.device)
        if projection is not None:
            return F.normalize(projection(self.raw_text_features), dim=-1)
        return self.raw_text_features

    @torch.no_grad()
    def get_severity_scores(self, projected_features, projection):
        logits, _ = self.forward(projected_features, projection)
        probs = F.softmax(logits, dim=-1)
        grades = torch.arange(self.num_prototypes, device=projected_features.device).float()
        scores = probs @ grades
        return scores, probs
