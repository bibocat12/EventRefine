"""TRAIN-only normalization for the seven EventRefine input features."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .dataset import EvUAV
from .models.event_features import bucketize, stem_features


def _clip_features(clip: dict[str, Any]) -> np.ndarray:
    locs = np.asarray(clip["locs"], dtype=np.int64)
    features = np.asarray(clip["feats"], dtype=np.float32)
    timestamps = np.asarray(clip["ts"], dtype=np.float64)
    origin = float(locs[:, 3].min())
    values = stem_features(
        locs,
        features,
        bucketize(locs, time_origin=origin),
        feature_time_origin=origin,
        current_raw_timestamps=timestamps,
    )
    if values.shape[1:] != (7,) or not np.isfinite(values).all():
        raise ValueError("TRAIN feature construction produced non-finite or non-seven-channel values")
    return values


def compute_train_normalization(data_root: str | Path, *, split: str = "train") -> dict[str, Any]:
    if split != "train":
        raise ValueError("normalization must be computed from TRAIN")
    dataset = EvUAV(data_root, split="train")
    count = 0
    total = np.zeros(7, dtype=np.float64)
    total_sq = np.zeros(7, dtype=np.float64)
    for index in range(len(dataset)):
        values = _clip_features(dataset[index]).astype(np.float64)
        count += len(values)
        total += values.sum(axis=0)
        total_sq += np.square(values).sum(axis=0)
    if count <= 0:
        raise ValueError("TRAIN contains no events")
    mean = total / count
    variance = np.maximum(0.0, total_sq / count - np.square(mean))
    std = np.sqrt(variance)
    if not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std <= 0):
        raise ValueError("TRAIN normalization is non-finite or has non-positive standard deviation")
    return {
        "mean": mean.astype(np.float32).tolist(),
        "std": std.astype(np.float32).tolist(),
        "split": "train",
        "clips": len(dataset),
        "events": count,
        "feature_channels": 7,
        "tau_scale": 64.0,
    }


def write_normalization(path: str | Path, payload: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


__all__ = ["compute_train_normalization", "write_normalization"]
