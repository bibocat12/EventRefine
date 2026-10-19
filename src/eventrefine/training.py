"""Small, explicit TRAIN/VAL runner for the fixed EventRefine model."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from .config import validate_config
from .dataset import EvUAV
from .losses import eventrefine_loss
from .model import EventRefineModel
from .runtime import confusion, make_context, metrics, prepare_window

ARM_NAME = "eventrefine_global_reference"


def load_normalization(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    mean = np.asarray(payload["mean"], dtype=np.float32)
    std = np.asarray(payload["std"], dtype=np.float32)
    if mean.shape != (7,) or std.shape != (7,) or not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std <= 0):
        raise ValueError("normalization must contain finite seven-channel mean/std vectors")
    return mean, std


def _add_counts(total: dict[str, int], observed: dict[str, int]) -> None:
    for key in total:
        total[key] += int(observed[key])


def evaluate_val(model: EventRefineModel, dataset: EvUAV, *, mean: np.ndarray, std: np.ndarray, device: torch.device, window_duration_ms: float = 1000.0) -> dict[str, Any]:
    model.eval()
    total = {key: 0 for key in ("TP", "FP", "FN", "TN", "events")}
    per_clip = []
    with torch.inference_mode():
        for index in range(len(dataset)):
            context = make_context(dataset[index], window_duration_ms=window_duration_ms)
            probabilities = np.full(len(context.clip["seg"]), np.nan, dtype=np.float32)
            for window in context.windows:
                if len(window) == 0:
                    continue
                batch = prepare_window(context, window, mean=mean, std=std, device=device)
                logits, _, _ = model(batch)
                probabilities[window] = torch.softmax(logits.float(), dim=-1)[:, 1].cpu().numpy()
            if not np.isfinite(probabilities).all():
                raise RuntimeError("validation did not produce one finite prediction per event")
            observed = confusion(probabilities, np.asarray(context.clip["seg"], dtype=np.int64))
            _add_counts(total, observed)
            per_clip.append({"clip": context.clip["name"], "metrics": metrics(observed)})
    return {"metrics": metrics(total), "clips": len(per_clip), "events": total["events"], "per_clip": per_clip, "split": "val", "uses_test": False}


def train_one(*, config_path: Path, data_root: Path, normalization_path: Path, output_root: Path, seed: int, epochs: int | None = None, device: str = "cpu") -> dict[str, Any]:
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config = validate_config(raw)
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"output root is not fresh: {output_root}")
    output_root.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    train_dataset = EvUAV(data_root, split="train")
    val_dataset = EvUAV(data_root, split="val")
    mean, std = load_normalization(normalization_path)
    target_epochs = int(config["training"]["epochs"] if epochs is None else epochs)
    if target_epochs < 1:
        raise ValueError("epochs must be positive")
    model = EventRefineModel(**{key: config["model"][key] for key in ("c_stem", "c_graph", "k")}).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["training"]["learning_rate"]), weight_decay=float(config["training"]["weight_decay"]))
    best_iou = -1.0
    best_epoch = -1
    best_val: dict[str, Any] = {}
    history = []
    for epoch in range(target_epochs):
        model.train()
        loss_values = []
        for index in np.random.permutation(len(train_dataset)):
            context = make_context(train_dataset[int(index)], window_duration_ms=float(config["model"]["window_duration_ms"]))
            for window in context.windows:
                if len(window) == 0:
                    continue
                batch = prepare_window(context, window, mean=mean, std=std, device=device)
                logits, _, _ = model(batch)
                loss, parts = eventrefine_loss(logits, batch["lab"], lovasz_weight=float(config["training"]["lovasz_weight"]), threshold=float(config["model"]["threshold"]))
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                loss_values.append(parts)
        val = evaluate_val(model, val_dataset, mean=mean, std=std, device=torch.device(device), window_duration_ms=float(config["model"]["window_duration_ms"]))
        record = {"epoch": epoch + 1, "train_loss": float(np.mean([item["loss"] for item in loss_values])), "val": val["metrics"]}
        history.append(record)
        if float(val["metrics"]["IoU"]) > best_iou:
            best_iou = float(val["metrics"]["IoU"])
            best_epoch = epoch + 1
            best_val = val
            torch.save({"epoch": best_epoch, "model": model.state_dict(), "config": config, "seed": int(seed), "arm": ARM_NAME, "val_metrics": val["metrics"]}, output_root / "best.pt")
        torch.save({"epoch": epoch + 1, "model": model.state_dict(), "config": config, "seed": int(seed), "arm": ARM_NAME}, output_root / "last.pt")
    summary = {"status": "COMPLETED", "arm": ARM_NAME, "seed": int(seed), "epochs": target_epochs, "best_epoch": best_epoch, "best_val_iou_09": best_iou, "best_val": best_val, "history": history, "split": "val", "uses_test": False, "train_clips": len(train_dataset), "val_clips": len(val_dataset)}
    (output_root / "training_summary.json").write_text(__import__("json").dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


__all__ = ["ARM_NAME", "evaluate_val", "load_normalization", "train_one"]
