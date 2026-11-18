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
from .metrics import confusion, summarize
from .model import EventRefineModel
from .runtime import CachedContext, cache_contexts, make_context, move_batch, prepare_window

ARM_NAME = "eventrefine_pair_reference"


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


def build_optimizer(
    model: EventRefineModel,
    config: dict[str, Any],
    *,
    target_epochs: int | None = None,
) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.CosineAnnealingLR, dict[str, int]]:
    """Build the fixed AdamW groups and cosine schedule used by the paper recipe."""
    training = config["training"]
    groups: dict[str, list[torch.nn.Parameter]] = {
        "decay": [], "no_decay": [], "stem": [], "local": [], "graph": [],
    }
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if "base.stem" in name:
            groups["stem"].append(parameter)
        elif "base.local" in name or "base.eventwise" in name:
            groups["local"].append(parameter)
        elif name.startswith("cc_mar.") or name.startswith("prediction_refinement."):
            groups["graph"].append(parameter)
        elif parameter.ndim >= 2 and not name.endswith(".bias"):
            groups["decay"].append(parameter)
        else:
            groups["no_decay"].append(parameter)
    parameter_groups = [
        {"params": groups["decay"], "lr": float(training["learning_rate"]),
         "weight_decay": float(training["weight_decay"]), "name": "decay"},
        {"params": groups["no_decay"], "lr": float(training["learning_rate"]),
         "weight_decay": 0.0, "name": "no_decay"},
        {"params": groups["stem"], "lr": float(training["stem_learning_rate"]),
         "weight_decay": float(training["weight_decay"]), "name": "stem"},
        {"params": groups["local"], "lr": float(training["local_learning_rate"]),
         "weight_decay": float(training["weight_decay"]), "name": "local"},
        {"params": groups["graph"], "lr": float(training["graph_learning_rate"]),
         "weight_decay": float(training["weight_decay"]), "name": "graph"},
    ]
    parameter_groups = [group for group in parameter_groups if group["params"]]
    optimizer = torch.optim.AdamW(parameter_groups)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=int(target_epochs if target_epochs is not None else training["epochs"]),
        eta_min=float(training["eta_min"]),
    )
    return optimizer, scheduler, {name: len(values) for name, values in groups.items()}


def evaluate_val(model: EventRefineModel, dataset: EvUAV, *, mean: np.ndarray, std: np.ndarray, device: torch.device, window_duration_ms: float = 1000.0, cached: tuple[CachedContext, ...] | None = None) -> dict[str, Any]:
    model.eval()
    contexts = cached if cached is not None else tuple(cache_contexts(dataset, mean=mean, std=std, window_duration_ms=window_duration_ms))
    total = {key: 0 for key in ("TP", "FP", "FN", "TN", "events")}
    per_clip = []
    with torch.inference_mode():
        for item in contexts:
            context = item.context
            probabilities = np.full(len(context.clip["seg"]), np.nan, dtype=np.float32)
            for window, cached_batch in zip(context.windows, item.prepared_windows):
                if cached_batch is None:
                    continue
                batch = move_batch(cached_batch, device)
                logits, _, _ = model(batch)
                probabilities[window] = torch.softmax(logits.float(), dim=-1)[:, 1].cpu().numpy()
            if not np.isfinite(probabilities).all():
                raise RuntimeError("validation did not produce one finite prediction per event")
            observed = confusion(probabilities, np.asarray(context.clip["seg"], dtype=np.int64))
            _add_counts(total, observed)
            per_clip.append({"clip": context.clip["name"], "metrics": summarize(observed)})
    return {"metrics": summarize(total), "clips": len(per_clip), "events": total["events"], "per_clip": per_clip, "split": "val", "uses_test": False}


def train_one(*, config_path: Path, data_root: Path, normalization_path: Path, output_root: Path, seed: int, epochs: int | None = None, device: str = "cpu") -> dict[str, Any]:
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config = validate_config(raw)
    output_root.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    train_dataset = EvUAV(data_root, split="train", preload=True)
    val_dataset = EvUAV(data_root, split="val", preload=True)
    mean, std = load_normalization(normalization_path)
    train_cached = cache_contexts(train_dataset, mean=mean, std=std, window_duration_ms=float(config["model"]["window_duration_ms"]))
    val_cached = cache_contexts(val_dataset, mean=mean, std=std, window_duration_ms=float(config["model"]["window_duration_ms"]))
    target_epochs = int(config["training"]["epochs"] if epochs is None else epochs)
    if target_epochs < 1:
        raise ValueError("epochs must be positive")
    model = EventRefineModel(**{key: config["model"][key] for key in ("c_stem", "c_graph", "k")}).to(device)
    optimizer, scheduler, optimizer_group_counts = build_optimizer(
        model, config, target_epochs=target_epochs,
    )
    accumulation_windows = int(config["training"]["gradient_accumulation_windows"])
    gradient_clip_norm = float(config["training"]["gradient_clip_norm"])
    best_iou = -1.0
    best_epoch = -1
    best_val: dict[str, Any] = {}
    history = []
    total_optimizer_steps = 0
    for epoch in range(target_epochs):
        model.train()
        loss_values = []
        pending_losses: list[torch.Tensor] = []
        pending_window_count = 0
        epoch_optimizer_steps = 0
        optimizer.zero_grad(set_to_none=True)

        def flush_pending() -> None:
            nonlocal pending_window_count, epoch_optimizer_steps, total_optimizer_steps
            if pending_losses:
                torch.stack(pending_losses).mean().backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
                optimizer.step()
                epoch_optimizer_steps += 1
                total_optimizer_steps += 1
            optimizer.zero_grad(set_to_none=True)
            pending_losses.clear()
            pending_window_count = 0

        for index in np.random.permutation(len(train_cached)):
            item = train_cached[int(index)]
            for window, cached_batch in zip(item.context.windows, item.prepared_windows):
                pending_window_count += 1
                if cached_batch is None:
                    if pending_window_count >= accumulation_windows:
                        flush_pending()
                    continue
                batch = move_batch(cached_batch, device)
                logits, _, _ = model(batch)
                loss, parts = eventrefine_loss(
                    logits,
                    batch["lab"],
                    lovasz_weight=float(config["training"]["lovasz_weight"]),
                    threshold=float(config["model"]["threshold"]),
                )
                pending_losses.append(loss)
                loss_values.append(parts)
                if pending_window_count >= accumulation_windows:
                    flush_pending()
            if pending_window_count:
                flush_pending()
        if pending_window_count:
            flush_pending()
        scheduler.step()
        val = evaluate_val(
            model,
            val_dataset,
            mean=mean,
            std=std,
            device=torch.device(device),
            window_duration_ms=float(config["model"]["window_duration_ms"]),
            cached=val_cached,
        )
        record = {
            "epoch": epoch + 1,
            "train_loss": float(np.mean([item["loss"] for item in loss_values])),
            "val": val["metrics"],
            "optimizer_steps": epoch_optimizer_steps,
            "lr": [float(group["lr"]) for group in optimizer.param_groups],
        }
        history.append(record)
        if float(val["metrics"]["IoU"]) > best_iou:
            best_iou = float(val["metrics"]["IoU"])
            best_epoch = epoch + 1
            best_val = val
            torch.save({"epoch": best_epoch, "model": model.state_dict(), "config": config, "seed": int(seed), "arm": ARM_NAME, "val_metrics": val["metrics"]}, output_root / "best.pt")
        torch.save({"epoch": epoch + 1, "model": model.state_dict(), "config": config, "seed": int(seed), "arm": ARM_NAME}, output_root / "last.pt")
    summary = {"status": "COMPLETED", "arm": ARM_NAME, "seed": int(seed), "epochs": target_epochs, "best_epoch": best_epoch, "best_val_iou_09": best_iou, "best_val": best_val, "history": history, "total_optimizer_steps": total_optimizer_steps, "optimizer_group_parameter_counts": optimizer_group_counts, "split": "val", "uses_test": False, "train_clips": len(train_dataset), "val_clips": len(val_dataset)}
    (output_root / "training_summary.json").write_text(__import__("json").dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


__all__ = ["ARM_NAME", "build_optimizer", "evaluate_val", "load_normalization", "train_one"]

