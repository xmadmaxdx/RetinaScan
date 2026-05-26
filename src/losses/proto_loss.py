import torch
import torch.nn as nn
import torch.nn.functional as F


def coral_ordinal_loss(ordinal_logits, labels, num_tasks=4):
    targets = torch.zeros(len(labels), num_tasks, device=labels.device)
    for k in range(num_tasks):
        targets[:, k] = (labels > k).float()
    return F.binary_cross_entropy_with_logits(ordinal_logits, targets)


class TextPrototypeLoss(nn.Module):
    def __init__(self, temperature=0.07, entropy_weight=0.5, diversity_weight=0.2,
                 supervised_weight=1.0, ordinal_weight=0.5):
        super().__init__()
        self.temperature = temperature
        self.entropy_weight = entropy_weight
        self.diversity_weight = diversity_weight
        self.supervised_weight = supervised_weight
        self.ordinal_weight = ordinal_weight

    def forward(self, proto_logits, projected_features, text_prototypes,
                labels=None, ordinal_logits=None):
        probs = F.softmax(proto_logits / self.temperature, dim=-1)
        log_probs = F.log_softmax(proto_logits / self.temperature, dim=-1)

        sup_loss = torch.tensor(0.0, device=proto_logits.device)
        if labels is not None:
            sup_loss = F.cross_entropy(proto_logits / self.temperature, labels)

        align_targets = labels if labels is not None else probs.argmax(dim=-1)
        selected = F.one_hot(align_targets, num_classes=text_prototypes.size(0)).float()
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

        ord_loss = torch.tensor(0.0, device=proto_logits.device)
        if ordinal_logits is not None and labels is not None:
            ord_loss = coral_ordinal_loss(ordinal_logits, labels)

        total = (
            self.supervised_weight * sup_loss
            + align_loss
            + self.entropy_weight * entropy_loss
            + self.diversity_weight * diversity_loss
            + self.ordinal_weight * ord_loss
        )

        return {
            "loss": total,
            "sup_loss": sup_loss.item(),
            "align_loss": align_loss.item(),
            "entropy_loss": entropy_loss.item(),
            "diversity_loss": diversity_loss.item(),
            "ordinal_loss": ord_loss.item(),
        }
