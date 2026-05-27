import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassWeightedCORALLoss(nn.Module):
    def __init__(self, pos_weight=None):
        super().__init__()
        if pos_weight is not None:
            self.register_buffer("pos_weight", torch.tensor(pos_weight))
        else:
            self.pos_weight = None

    def forward(self, logits, targets):
        w = self.pos_weight.to(device=logits.device) if self.pos_weight is not None else None
        loss = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=w)
        return loss


class PrototypeFocalLoss(nn.Module):
    def __init__(self, gamma=2.5):
        super().__init__()
        self.gamma = gamma

    def forward(self, image_features, text_prototypes, labels, temperature=0.05):
        img_norm = F.normalize(image_features, dim=-1)
        proto_norm = F.normalize(text_prototypes, dim=-1)
        sim = torch.matmul(img_norm, proto_norm.t()) / temperature
        probs = F.softmax(sim, dim=-1)
        true_probs = probs[torch.arange(len(labels), device=labels.device), labels]
        focal = torch.pow(1.0 - true_probs, self.gamma)
        ce = F.cross_entropy(sim, labels, reduction="none")
        return (focal * ce).mean()
