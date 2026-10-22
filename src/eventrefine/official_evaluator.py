"""Adapter to a user-supplied EV-UAV evaluator; no upstream code is vendored."""
from __future__ import annotations

import importlib.util
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

from .checkpoints import git_identity, sha256


class OfficialEvaluator:
    """Run the upstream ``evalute.roc_update/cal_roc`` API on whole clips.

    Preserve archive timestamps and the upstream interval boundaries, counters,
    and component-counting behavior. Prediction windows are reassembled by the
    caller; they must not be submitted as separate detection-evaluation clips.
    """

    def __init__(self, root: Path, *, threshold: float = 0.90,
                 interval_ms: float = 50.0, correct_threshold: float = 1e-4) -> None:
        self.root = Path(root).expanduser().resolve()
        self.source = self.root / "utils" / "eval.py"
        if not self.source.is_file():
            raise FileNotFoundError(f"official evaluator not found: {self.source}")
        if not math.isfinite(interval_ms) or interval_ms <= 0:
            raise ValueError("official interval must be positive and finite")
        if not 0 < correct_threshold <= 1 or not 0 < threshold < 1:
            raise ValueError("official thresholds are outside their valid ranges")
        self.threshold = float(threshold)
        self.interval_ms = float(interval_ms)
        self.correct_threshold = float(correct_threshold)
        self.source_hash = sha256(self.source)
        spec = importlib.util.spec_from_file_location("_eventrefine_external_eval", self.source)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load official evaluator: {self.source}")
        module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)
        except ImportError as error:
            raise ImportError(
                "Could not import the supplied EV-UAV evaluator. Install its "
                "dependencies (pip install -e '.[evaluator]') and retry."
            ) from error
        factory = getattr(module, "evalute", None)
        if not callable(factory):
            raise TypeError("expected official evaluator class 'evalute' in utils/eval.py")
        self.evaluator = factory(SimpleNamespace(
            roc=True, pd_detT=self.interval_ms, correct_thresh=self.correct_threshold,
        ))
        if not all(callable(getattr(self.evaluator, name, None))
                   for name in ("roc_update", "cal_roc")):
            raise TypeError("official evaluator must implement roc_update and cal_roc")
        self.clips = 0

    def update(self, clip: dict[str, Any], probabilities: np.ndarray) -> None:
        probabilities = np.asarray(probabilities, dtype=np.float32)
        labels = np.asarray(clip["seg"], dtype=np.int64)
        identities = np.asarray(clip["idx"])
        locations = np.asarray(clip["locs"])
        if (probabilities.ndim != 1 or probabilities.size == 0
                or labels.shape != probabilities.shape or identities.shape != probabilities.shape
                or locations.shape != (len(probabilities), 4)):
            raise ValueError("official evaluator requires one aligned whole-clip prediction per event")
        if not np.isfinite(probabilities).all() or np.any((probabilities < 0) | (probabilities > 1)):
            raise ValueError("official evaluator probabilities must be finite and in [0,1]")
        # Match the upstream test.py interface: CPU float locations/labels and
        # predictions, NumPy instance IDs. Never mutate the caller's arrays or
        # subtract a timestamp origin that the official caller does not subtract.
        ev_locs = torch.tensor(locations, dtype=torch.float32)
        self.evaluator.roc_update(
            ev_locs[:, 3], torch.tensor(probabilities), identities.copy(),
            torch.tensor(labels, dtype=torch.float32), ev_locs, thresh=self.threshold,
        )
        self.clips += 1

    def report(self) -> dict[str, Any]:
        # These counters define the denominator in the inspected upstream API.
        counters = {name: int(getattr(self.evaluator, name)) for name in
                    ("obj_num", "correct_num", "frame_num", "false_num", "whole_ev_num")}
        if counters["obj_num"] <= 0 or counters["frame_num"] <= 0:
            raise ValueError("official Pd/Fa are undefined: no target instances or effective intervals")
        pd, fa = (float(value) for value in self.evaluator.cal_roc())
        if not math.isfinite(pd) or not math.isfinite(fa):
            raise ValueError("official evaluator returned non-finite Pd/Fa")
        if sha256(self.source) != self.source_hash:
            raise RuntimeError("official evaluator source changed during evaluation")
        return {
            "status": "COMPLETED", "root": str(self.root), "clips": self.clips,
            "source": {**git_identity(self.root), "file": "utils/eval.py", "sha256": self.source_hash},
            "settings": {"threshold": self.threshold, "interval_ms": self.interval_ms,
                         "correct_threshold": self.correct_threshold,
                         "timestamp_policy": "archive timestamps unchanged; upstream test.py convention",
                         "input_unit": "one reassembled full clip"},
            "counters": counters,
            "metrics": {"Pd": 100.0 * pd, "Fa": fa, "Fa_x1e4": 1e4 * fa},
        }
