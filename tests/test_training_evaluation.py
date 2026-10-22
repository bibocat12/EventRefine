from pathlib import Path
import os

import numpy as np
import pytest
import torch
import yaml

from eventrefine.evaluation import evaluate
from eventrefine.training import train_one


CONFIG = Path(__file__).parents[1] / "configs/eventrefine_global_reference.yaml"


def _archive(root, split):
    directory = root / split
    directory.mkdir()
    # Two occupied model windows separated by one empty window.
    times = np.array([1, 20, 40, 80, 2001, 2020, 2040, 2080])
    locs = np.column_stack([np.arange(8) % 4, np.arange(8) % 3, times])
    norm = np.zeros((8, 6), dtype=np.float32)
    norm[:, 3] = np.arange(8) % 2
    norm[:, 4] = [1, 1, 0, 0, 1, 0, 1, 0]
    norm[:, 5] = norm[:, 4]
    np.savez(directory / f"{split}_000.npz", ev_loc=locs, evs_norm=norm)


def test_one_epoch_train_validate_reload_and_evaluate(tmp_path):
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        data = tmp_path / "data"
        data.mkdir()
        _archive(data, "train")
        _archive(data, "val")
        # A malformed TEST archive must never be touched by training/VAL.
        (data / "test").mkdir()
        (data / "test" / "test_000.npz").write_text("must not be opened")
        normalizer = tmp_path / "normalization.yaml"
        normalizer.write_text(yaml.safe_dump({"mean": [0.] * 7, "std": [1.] * 7}))
        output = tmp_path / "run"
        summary = train_one(config_path=CONFIG, data_root=data, normalization_path=normalizer,
                            output_root=output, seed=37, epochs=1)
        assert summary["best_epoch"] == 1
        assert summary["uses_test"] is False
        assert summary["best_val"]["events"] == 8
        payload = torch.load(output / "best.pt", weights_only=False)
        assert payload["model"]["cc_mar.up.weight"].abs().sum() > 0
        result = evaluate(config_path=CONFIG, data_root=data, normalization_path=normalizer,
                          checkpoint=output / "best.pt", output=tmp_path / "evaluation.json", split="val")
        assert result["metrics"] == summary["best_val"]["metrics"]
        assert result["official_evaluator"] is None
        external_root = os.environ.get("EVENTREFINE_OFFICIAL_EVALUATOR_ROOT")
        if external_root:
            official = evaluate(
                config_path=CONFIG, data_root=data, normalization_path=normalizer,
                checkpoint=output / "best.pt", output=tmp_path / "official.json", split="val",
                official_evaluator_root=Path(external_root),
            )
            assert official["official_evaluator"]["clips"] == 1
            assert official["official_evaluator"]["counters"]["whole_ev_num"] == 8
            for key, value in result["metrics"].items():
                assert official["metrics"][key] == value
            assert {"Pd", "Fa", "Fa_x1e4"}.issubset(official["metrics"])
    finally:
        torch.set_num_threads(old_threads)


def test_test_requires_opt_in_before_any_files_are_opened(tmp_path):
    missing = tmp_path / "missing"
    with pytest.raises(ValueError, match="allow-test"):
        evaluate(config_path=missing, data_root=missing, normalization_path=missing,
                 checkpoint=missing, output=missing, split="test", official_evaluator_root=missing)
