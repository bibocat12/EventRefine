"""Explicit split-aware evaluation for EventRefine."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from .checkpoints import load_strict, make_receipt, sha256, write_receipt
from .config import validate_config
from .dataset import EvUAV
from .metrics import confusion, summarize
from .model import EventRefineModel
from .official_evaluator import OfficialEvaluator
from .runtime import make_context, prepare_window
from .training import load_normalization


def evaluate(*, config_path: Path, data_root: Path, normalization_path: Path, checkpoint: Path, output: Path, split: str, device: str = "cpu", allow_test: bool = False, official_evaluator_root: Path | None = None, official_interval_ms: float = 50.0, official_correct_threshold: float = 1e-4) -> dict[str, Any]:
    if split == "test" and not allow_test:
        raise ValueError("TEST evaluation requires --allow-test")
    if split not in {"val", "test"}:
        raise ValueError("evaluation split must be val or test")
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config = validate_config(raw)
    official = (OfficialEvaluator(
        official_evaluator_root, threshold=float(config["model"]["threshold"]),
        interval_ms=official_interval_ms, correct_threshold=official_correct_threshold,
    ) if official_evaluator_root is not None else None)
    mean, std = load_normalization(normalization_path)
    model = EventRefineModel(**{key: config["model"][key] for key in ("c_stem", "c_graph", "k")})
    payload = load_strict(model, checkpoint, device=torch.device(device))
    dataset = EvUAV(data_root, split=split)
    counts = {key: 0 for key in ("TP", "FP", "FN", "TN", "events")}
    per_clip = []
    with torch.inference_mode():
        for index in range(len(dataset)):
            clip = dataset[index]
            context = make_context(clip, window_duration_ms=float(config["model"]["window_duration_ms"]))
            probabilities = np.full(len(clip["seg"]), np.nan, dtype=np.float32)
            for window in context.windows:
                if len(window) == 0:
                    continue
                batch = prepare_window(context, window, mean=mean, std=std, device=torch.device(device))
                logits, _, _ = model(batch)
                probabilities[window] = torch.softmax(logits.float(), dim=-1)[:, 1].cpu().numpy()
            if not np.isfinite(probabilities).all():
                raise RuntimeError("evaluation did not produce one finite prediction per event")
            observed = confusion(probabilities, np.asarray(clip["seg"], dtype=np.int64), threshold=float(config["model"]["threshold"]))
            for key in counts:
                counts[key] += int(observed[key])
            per_clip.append({"clip": clip["name"], "metrics": summarize(observed)})
            if official is not None:
                official.update(clip, probabilities)
    result: dict[str, Any] = {"status": "COMPLETED", "split": split, "uses_test": split == "test", "metrics": summarize(counts), "clips": len(per_clip), "events": counts["events"], "per_clip": per_clip, "checkpoint_epoch": payload.get("epoch"), "official_evaluator": None}
    if official is not None:
        result["official_evaluator"] = official.report()
        result["metrics"].update(result["official_evaluator"]["metrics"])
    receipt = make_receipt(root=Path(__file__).resolve().parents[2], config_path=config_path, split=split, uses_test=split == "test", checkpoint=checkpoint, clips=len(per_clip), events=counts["events"], metrics=result["metrics"])
    receipt["status"] = "COMPLETED"
    receipt["normalization"] = {"path": str(normalization_path.resolve()), "sha256": sha256(normalization_path)}
    receipt["official_evaluator"] = result["official_evaluator"]
    output.parent.mkdir(parents=True, exist_ok=True)
    write_receipt(output, {"result": result, "receipt": receipt})
    return result


__all__ = ["evaluate"]

