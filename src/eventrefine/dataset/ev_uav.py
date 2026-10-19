"""Read-only adapter for the official EV-UAV archive layout."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def convert_official_arrays(arrays: dict[str, Any], name: str) -> dict[str, Any]:
    """Convert an EV-UAV archive to the EventRefine event contract.

    The released EV-UAV archives contain ``ev_loc`` with ``[x, y, t]`` and
    ``evs_norm`` whose first four columns are normalized event features; columns
    four and five contain the binary label and target identity. A structured
    ``ev`` array is also accepted for archives produced by the evaluation tools.
    """
    required = {"evs_norm", "ev_loc"}
    missing = required - set(arrays)
    if missing:
        raise ValueError(f"EV-UAV archive missing: {sorted(missing)}")
    norm = np.asarray(arrays["evs_norm"])
    ev_loc = np.asarray(arrays["ev_loc"])
    if norm.ndim != 2 or norm.shape[1] < 6 or ev_loc.ndim != 2 or ev_loc.shape[1] != 3:
        raise ValueError("EV-UAV evs_norm/ev_loc arrays have unexpected shapes")
    if len(norm) == 0 or len(norm) != len(ev_loc):
        raise ValueError("EV-UAV arrays must have the same nonzero event count")

    x = np.asarray(ev_loc[:, 0], dtype=np.int64)
    y = np.asarray(ev_loc[:, 1], dtype=np.int64)
    t = np.asarray(ev_loc[:, 2], dtype=np.int64)
    if np.any(x < 0) or np.any(x >= 346) or np.any(y < 0) or np.any(y >= 260):
        raise ValueError("EV-UAV coordinates fall outside the 346x260 sensor")
    if np.any(t[1:] < t[:-1]):
        raise ValueError("EV-UAV timestamps must be nondecreasing")

    labels = np.asarray(norm[:, 4], dtype=np.int64)
    identities = np.asarray(norm[:, 5], dtype=np.int64)
    features = np.asarray(norm[:, :4], dtype=np.float32)
    if not np.isfinite(features).all() or np.any((labels != 0) & (labels != 1)):
        raise ValueError("EV-UAV features must be finite and labels binary")

    # Validate optional structured fields when present, without requiring them
    # for the public EV-UAV archive format.
    if "ev" in arrays:
        ev = np.asarray(arrays["ev"])
        required_fields = {"t", "p", "label", "name"}
        if not ev.dtype.names or not required_fields.issubset(ev.dtype.names):
            raise ValueError("structured EV-UAV array lacks required fields")
        if not np.array_equal(np.asarray(ev["t"], dtype=np.int64), t):
            raise ValueError("structured timestamps disagree with ev_loc")
        if not np.array_equal(np.asarray(ev["label"], dtype=np.int64), labels):
            raise ValueError("structured labels disagree with evs_norm")
        polarity = np.asarray(ev["p"], dtype=np.int64)
    else:
        polarity = (features[:, 3] > 0.5).astype(np.int64)

    locs = np.column_stack([np.zeros(len(t), dtype=np.int64), x, y, t])
    return {
        "name": str(name),
        "locs": locs,
        "ts": t.astype(np.float64),
        "seg": labels,
        "feats": features,
        "idx": identities,
        "polarity": polarity,
    }


class EvUAV:
    """Read only the requested split; other split directories are untouched."""

    def __init__(self, root: str | Path, *, split: str = "val") -> None:
        if split not in {"train", "val", "test"}:
            raise ValueError(f"unknown split {split!r}")
        self.split = split
        self.root = Path(root).expanduser().resolve() / split
        self.files = sorted(self.root.glob(f"{split}_*.npz"))
        if not self.files:
            raise FileNotFoundError(f"no {split}_*.npz archives found in {self.root}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> dict[str, Any]:
        path = self.files[int(index)]
        with np.load(path, allow_pickle=True) as arrays:
            return convert_official_arrays({key: arrays[key] for key in arrays.files}, path.stem)

    def __iter__(self):
        for index in range(len(self)):
            yield self[index]


__all__ = ["EvUAV", "convert_official_arrays"]
