"""Paper-facing EventRefine segmentation objective."""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def lovasz_grad(sorted_labels: torch.Tensor) -> torch.Tensor:
    count = len(sorted_labels)
    positives = sorted_labels.sum()
    intersection = positives - sorted_labels.float().cumsum(0)
    union = positives + (1.0 - sorted_labels).float().cumsum(0)
    gradient = 1.0 - intersection / union
    if count > 1:
        gradient[1:] -= gradient[:-1].clone()
    return gradient


def lovasz_softmax_flat(probabilities: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    if probabilities.numel() == 0:
        return probabilities.sum() * 0.0
    values = []
    for class_id in range(probabilities.shape[1]):
        foreground = (labels == class_id).float()
        if foreground.sum() == 0:
            continue
        errors = (foreground - probabilities[:, class_id]).abs()
        errors, permutation = torch.sort(errors, descending=True)
        values.append(torch.dot(errors, lovasz_grad(foreground[permutation])))
    return torch.stack(values).mean() if values else probabilities.sum() * 0.0


def eventrefine_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    lovasz_weight: float = 0.15,
    threshold: float = 0.90,
) -> tuple[torch.Tensor, dict[str, float]]:
    if logits.ndim != 2 or logits.shape[-1] != 2 or labels.shape != (logits.shape[0],):
        raise ValueError("logits and labels are not aligned")
    if not 0.0 < float(threshold) < 1.0:
        raise ValueError("threshold must be in (0,1)")
    ce = F.cross_entropy(logits.float(), labels.long())
    margin = logits[:, 1].float() - logits[:, 0].float()
    shifted = torch.stack(
        [torch.zeros_like(margin), margin - math.log(float(threshold) / (1.0 - float(threshold)))],
        dim=-1,
    )
    lovasz = lovasz_softmax_flat(torch.softmax(shifted, dim=-1), labels)
    total = ce + float(lovasz_weight) * lovasz
    return total, {"loss": float(total.detach()), "ce": float(ce.detach()), "lovasz": float(lovasz.detach())}


__all__ = ["eventrefine_loss", "lovasz_softmax_flat"]
