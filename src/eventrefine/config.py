"""Fail-closed configuration for the fixed public EventRefine model."""
from __future__ import annotations

import math
from collections.abc import Mapping

_REQUIRED = {"project", "version", "model", "training", "paths"}
_MODEL_KEYS = {
    "c_stem", "c_graph", "k", "encoder", "encoder_context", "graph_operator",
    "refinement_mode", "threshold", "window_duration_ms", "tau_scale", "n_blocks",
    "image_width", "image_height",
}
_TRAINING_KEYS = {
    "objective", "final_ce_weight", "lovasz_weight", "threshold_logit_shift",
    "checkpoint_selection", "optimizer", "learning_rate", "weight_decay", "epochs", "seeds",
}
_PATH_KEYS = {"data_root_env", "normalization_env", "output_root_env"}


def _number(value: object, expected: float) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isclose(
        float(value), expected, rel_tol=0.0, abs_tol=1e-9
    )


def validate_config(raw: Mapping[str, object]) -> dict[str, object]:
    if not isinstance(raw, Mapping):
        raise TypeError("config must be a mapping")
    unknown = set(raw) - _REQUIRED
    missing = _REQUIRED - set(raw)
    if unknown:
        raise ValueError(f"unknown top-level config keys: {sorted(unknown)}")
    if missing:
        raise ValueError(f"missing top-level config keys: {sorted(missing)}")
    model, training, paths = raw["model"], raw["training"], raw["paths"]
    for name, value, expected in (("model", model, _MODEL_KEYS), ("training", training, _TRAINING_KEYS), ("paths", paths, _PATH_KEYS)):
        if not isinstance(value, Mapping):
            raise TypeError(f"{name} must be a mapping")
        extra, absent = set(value) - expected, expected - set(value)
        if extra:
            raise ValueError(f"unknown {name} keys: {sorted(extra)}")
        if absent:
            raise ValueError(f"missing {name} keys: {sorted(absent)}")
    exact_model = {
        "c_stem": 64, "c_graph": 16, "k": 8, "encoder": "sensor",
        "encoder_context": "local_attention", "graph_operator": "clipped_global",
        "refinement_mode": "feature_graph", "threshold": 0.90,
        "window_duration_ms": 1000.0, "tau_scale": 64.0, "n_blocks": 32,
        "image_width": 346, "image_height": 260,
    }
    for key, expected in exact_model.items():
        actual = model[key]
        valid = _number(actual, expected) if isinstance(expected, float) else type(actual) is type(expected) and actual == expected
        if not valid:
            raise ValueError(f"model protocol drift for {key}: expected {expected!r}, got {actual!r}")
    exact_training = {
        "objective": "final_ce_plus_shifted_lovasz", "final_ce_weight": 1.0,
        "lovasz_weight": 0.15, "threshold_logit_shift": math.log(9.0),
        "checkpoint_selection": "max_val_iou_at_0.90", "optimizer": "AdamW",
        "learning_rate": 0.0002, "weight_decay": 0.01, "epochs": 120,
    }
    for key, expected in exact_training.items():
        actual = training[key]
        valid = _number(actual, expected) if isinstance(expected, float) else type(actual) is type(expected) and actual == expected
        if not valid:
            raise ValueError(f"training protocol drift for {key}: expected {expected!r}, got {actual!r}")
    if tuple(training["seeds"]) != (37, 1337, 2024):
        raise ValueError("training seeds must be [37, 1337, 2024]")
    if any(not isinstance(paths[key], str) or not paths[key] for key in _PATH_KEYS):
        raise ValueError("path environment names must be nonempty strings")
    return {
        "model": dict(model), "training": dict(training), "paths": dict(paths),
    }


__all__ = ["validate_config"]
