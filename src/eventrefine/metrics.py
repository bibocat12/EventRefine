"""Event-wise segmentation metrics."""
from __future__ import annotations

import numpy as np


def confusion(probabilities: np.ndarray, labels: np.ndarray, threshold: float = 0.90) -> dict[str, int]:
    probabilities = np.asarray(probabilities, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int64)
    if probabilities.shape != labels.shape or not np.isfinite(probabilities).all():
        raise ValueError("probabilities and labels must be aligned and finite")
    predicted, target = probabilities >= float(threshold), labels == 1
    return {
        "TP": int(np.count_nonzero(predicted & target)),
        "FP": int(np.count_nonzero(predicted & ~target)),
        "FN": int(np.count_nonzero(~predicted & target)),
        "TN": int(np.count_nonzero(~predicted & ~target)),
        "events": int(labels.size),
    }


def summarize(counts: dict[str, int]) -> dict[str, float | int]:
    tp, fp, fn = (int(counts[k]) for k in ("TP", "FP", "FN"))
    return {
        **counts,
        "IoU": 100.0 * tp / max(1, tp + fp + fn),
        "Precision": 100.0 * tp / max(1, tp + fp),
        "Recall": 100.0 * tp / max(1, tp + fn),
    }


__all__ = ["confusion", "summarize"]
