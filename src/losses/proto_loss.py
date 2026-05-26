import torch
import torch.nn as nn
import torch.nn.functional as F


class TextPrototypeLoss(nn.Module):
    def __init__(self, temperature=0.07, entropy_weight=0.5, diversity_weight=0.2, supervised_weight=1.0):
        super().__init__()
        self.temperature = temperature
        self.entropy_weight = entropy_weight
        self.diversity_weight = diversity_weight
        self.supervised_weight = supervised_weight

    def forward(self, logits, projected_features, text_prototypes, labels=None):
        probs = F.softmax(logits / self.temperature, dim=-1)
        log_probs = F.log_softmax(logits / self.temperature, dim=-1)

        sup_loss = torch.tensor(0.0, device=logits.device)
        if labels is not None:
            sup_loss = F.cross_entropy(logits / self.temperature, labels)

        assigned = probs.argmax(dim=-1)
        selected = F.one_hot(assigned, num_classes=text_prototypes.size(0)).float()
        proto_selected = selected @ text_prototypes
        proto_selected = F.normalize(proto_selected, dim=-1)
        proj_norm = F.normalize(projected_features, dim=-1)
        align_loss = 1.0 - (proj_norm * proto_selected).sum(dim=-1).mean()

        entropy = -(probs * log_probs).sum(dim=-1).mean()
        entropy_loss = -entropy

        p_norm = F.normalize(text_prototypes, dim=-1)
        sim_matrix = p_norm @ p_norm.T
        mask = torch.eye(text_prototypes.size(0), device=text_prototypes.device)
        diversity_loss = (sim_matrix * (1 - mask)).abs().mean()

        total = (
            self.supervised_weight * sup_loss
            + align_loss
            + self.entropy_weight * entropy_loss
            + self.diversity_weight * diversity_loss
        )

        return {
            "loss": total,
            "sup_loss": sup_loss.item(),
            "align_loss": align_loss.item(),
            "entropy_loss": entropy_loss.item(),
            "diversity_loss": diversity_loss.item(),
        }
