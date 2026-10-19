"""Strict checkpoint loading and compact provenance receipts."""
from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_identity(root: str | Path) -> dict[str, Any]:
    root = str(Path(root).resolve())
    try:
        commit = subprocess.check_output(["git", "-C", root, "rev-parse", "HEAD"], text=True).strip()
        dirty = bool(subprocess.check_output(["git", "-C", root, "status", "--porcelain"], text=True).strip())
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = None, None
    return {"commit": commit, "dirty": dirty}


def load_strict(model: torch.nn.Module, path: str | Path, *, device: torch.device) -> dict[str, Any]:
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "model" not in payload:
        raise ValueError("checkpoint must be a mapping containing a model state_dict")
    model.load_state_dict(payload["model"], strict=True)
    model.to(device).eval()
    return payload


def write_receipt(path: str | Path, payload: dict[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(destination)


def make_receipt(*, root: Path, config_path: Path, split: str, uses_test: bool, checkpoint: Path | None = None, **extra: Any) -> dict[str, Any]:
    receipt: dict[str, Any] = {
        "status": "COMPLETED",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source": git_identity(root),
        "config": {"path": str(config_path.resolve()), "sha256": sha256(config_path)},
        "split": split,
        "uses_test": bool(uses_test),
        **extra,
    }
    if checkpoint is not None:
        receipt["checkpoint"] = {"path": str(checkpoint.resolve()), "sha256": sha256(checkpoint)}
    return receipt


__all__ = ["git_identity", "load_strict", "make_receipt", "sha256", "write_receipt"]
