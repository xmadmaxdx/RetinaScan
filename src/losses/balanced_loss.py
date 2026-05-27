import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassWeightedCORALLoss(nn.Module):
    def __init__(self, class_counts=None):
        super().__init__()
        if class_counts is not None:
            counts = torch.FloatTensor(class_counts)
            total = counts.sum()
            pos_rates = torch.zeros(4)
            for k in range(4):
                pos_rate = counts[k + 1:].sum() / total
                pos_rates[k] = (1.0 - pos_rate) / (pos_rate + 1e-8)
            self.register_buffer("task_weights", pos_rates.clamp(1.0, 50.0))
        else:
            self.register_buffer("task_weights", torch.tensor([2.7, 4.0, 19.0, 49.0]))

    def forward(self, logits, targets):
        w = self.task_weights.to(device=logits.device)
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
