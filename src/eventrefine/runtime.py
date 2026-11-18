"""Window preparation and direct metric helpers for EventRefine."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from .models import event_features
from .models.sensor_backbone import prep


@dataclass(frozen=True)
class ClipContext:
    clip: dict[str, Any]
    raw_features: np.ndarray
    windows: tuple[np.ndarray, ...]
    time_origin: float
    time_center: float


@dataclass(frozen=True)
class CachedContext:
    context: ClipContext
    prepared_windows: tuple[dict[str, torch.Tensor] | None, ...]


def make_context(clip: dict[str, Any], *, window_duration_ms: float = 1000.0) -> ClipContext:
    locs = np.asarray(clip["locs"], dtype=np.int64)
    features = np.asarray(clip["feats"], dtype=np.float32)
    timestamps = np.asarray(clip["ts"], dtype=np.float64)
    if len(locs) == 0 or not (len(locs) == len(features) == len(timestamps)):
        raise ValueError("clip arrays must be nonempty and aligned")
    if np.any(timestamps[1:] < timestamps[:-1]):
        raise ValueError("clip timestamps must be nondecreasing")
    origin = float(locs[:, 3].min())
    raw_features = event_features.stem_features(
        locs,
        features,
        event_features.bucketize(locs, time_origin=origin),
        feature_time_origin=origin,
        current_raw_timestamps=timestamps,
    )
    window_ids = np.floor((timestamps - timestamps[0]) / float(window_duration_ms)).astype(np.int64)
    windows = tuple(np.flatnonzero(window_ids == index) for index in range(int(window_ids[-1]) + 1))
    if sum(len(item) for item in windows) != len(locs):
        raise RuntimeError("temporal windows do not cover every event exactly once")
    tau = (locs[:, 3].astype(np.float64) - origin) / 64.0
    return ClipContext(clip, raw_features, windows, origin, float(tau.mean()))


def prepare_window(
    context: ClipContext,
    indices: np.ndarray,
    *,
    mean: np.ndarray,
    std: np.ndarray,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    if len(indices) == 0:
        raise ValueError("cannot prepare an empty model window")
    indices = np.asarray(indices, dtype=np.int64)
    expected = np.arange(int(indices[0]), int(indices[-1]) + 1, dtype=np.int64)
    if not np.array_equal(indices, expected):
        raise ValueError("model windows must preserve contiguous event order")
    clip = context.clip
    locs = np.asarray(clip["locs"], dtype=np.int64)[indices]
    segment = np.asarray(clip["seg"], dtype=np.int64)[indices]
    features = np.asarray(clip["feats"], dtype=np.float32)[indices]
    raw_features = context.raw_features[indices]
    batch = prep(
        {"name": str(clip["name"]), "locs": locs, "seg": segment, "feats": features},
        device,
        np.asarray(mean, dtype=np.float32),
        np.asarray(std, dtype=np.float32),
        tau_scale=64.0,
        n_blocks=32,
        precomputed_features=raw_features,
        clip_time_origin=context.time_origin,
        clip_time_end=float(np.asarray(clip["locs"])[-1, 3]),
        clip_time_center=context.time_center,
    )
    batch["x"] = torch.as_tensor(locs[:, 1], dtype=torch.float32, device=device)
    batch["y"] = torch.as_tensor(locs[:, 2], dtype=torch.float32, device=device)
    batch["lab"] = torch.as_tensor(segment, dtype=torch.long, device=device)
    batch["global_indices"] = torch.as_tensor(indices, dtype=torch.long, device=device)
    return batch


def _pin_batch(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    if not torch.cuda.is_available():
        return batch
    return {key: value.pin_memory() if isinstance(value, torch.Tensor) else value for key, value in batch.items()}


def cache_contexts(dataset, *, mean: np.ndarray, std: np.ndarray, window_duration_ms: float = 1000.0) -> tuple[CachedContext, ...]:
    cached = []
    cpu = torch.device("cpu")
    for index in range(len(dataset)):
        context = make_context(dataset[index], window_duration_ms=window_duration_ms)
        prepared = tuple(
            None if len(window) == 0 else _pin_batch(
                prepare_window(context, window, mean=mean, std=std, device=cpu)
            )
            for window in context.windows
        )
        cached.append(CachedContext(context=context, prepared_windows=prepared))
    return tuple(cached)


def move_batch(batch: dict[str, torch.Tensor], device: torch.device | str) -> dict[str, torch.Tensor]:
    return {
        key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def confusion(logits: torch.Tensor, labels: torch.Tensor, threshold: float = 0.90) -> dict[str, int]:
    predicted = torch.softmax(logits.float(), dim=-1)[:, 1] >= float(threshold)
    target = labels == 1
    return {
        "TP": int((predicted & target).sum()),
        "FP": int((predicted & ~target).sum()),
        "FN": int((~predicted & target).sum()),
        "TN": int((~predicted & ~target).sum()),
        "events": int(labels.numel()),
    }


def metrics(counts: dict[str, int]) -> dict[str, float | int]:
    tp, fp, fn, _tn = (int(counts[key]) for key in ("TP", "FP", "FN", "TN"))
    return {
        "IoU": 100.0 * tp / max(1, tp + fp + fn),
        "Precision": 100.0 * tp / max(1, tp + fp),
        "Recall": 100.0 * tp / max(1, tp + fn),
        **counts,
    }


__all__ = ["CachedContext", "ClipContext", "cache_contexts", "confusion", "make_context", "metrics", "move_batch", "prepare_window"]
