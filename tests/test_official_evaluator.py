import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from eventrefine.official_evaluator import OfficialEvaluator


def sample():
    times = np.array([1, 49, 50, 51, 100, 101, 151])
    clip = {"locs": np.column_stack([np.zeros(7), np.arange(7), np.ones(7), times]),
            "seg": np.array([1, 0, 0, 1, 0, 0, 1]), "idx": np.array([1, 0, 0, 1, 0, 0, 1])}
    probabilities = np.array([.95, .95, .95, .2, .95, .95, .95], dtype=np.float32)
    return clip, probabilities


def test_adapter_loads_external_api_records_settings_and_copies_inputs(tmp_path):
    # Independent interface fixture, not an implementation of EV-UAV metrics.
    (tmp_path / "utils").mkdir()
    (tmp_path / "utils" / "eval.py").write_text('''
class evalute:
    def __init__(self, cfg):
        assert cfg.roc and cfg.pd_detT == 50 and cfg.correct_thresh == 0.0001
        self.obj_num, self.correct_num, self.frame_num, self.false_num = 3, 2, 3, 2
        self.whole_ev_num = 0
    def roc_update(self, ts, preds, idx, label, ev_locs, thresh):
        assert ts[0] == 1 and ts[-1] == 151 and thresh == 0.9
        assert preds.shape == label.shape == idx.shape == (7,)
        self.whole_ev_num += len(preds)
        preds[:] = 0
        idx[:] = 99
    def cal_roc(self):
        return self.correct_num / self.obj_num, self.false_num / (self.frame_num * 346 * 260)
''')
    clip, probabilities = sample()
    original_probabilities, original_ids = probabilities.copy(), clip["idx"].copy()
    adapter = OfficialEvaluator(tmp_path)
    adapter.update(clip, probabilities)
    report = adapter.report()
    assert report["status"] == "COMPLETED"
    assert report["metrics"]["Pd"] == pytest.approx(200 / 3)
    assert report["metrics"]["Fa_x1e4"] == pytest.approx(2e4 / (3 * 346 * 260))
    assert report["counters"]["whole_ev_num"] == 7
    assert len(report["source"]["sha256"]) == 64
    np.testing.assert_array_equal(probabilities, original_probabilities)
    np.testing.assert_array_equal(clip["idx"], original_ids)
    adapter.evaluator.frame_num = 0
    with pytest.raises(ValueError, match="undefined"):
        adapter.report()


def test_missing_official_evaluator_fails_explicitly(tmp_path):
    with pytest.raises(FileNotFoundError, match="official evaluator"):
        OfficialEvaluator(tmp_path)


@pytest.mark.skipif(not os.environ.get("EVENTREFINE_OFFICIAL_EVALUATOR_ROOT"),
                    reason="external EV-UAV checkout not supplied")
def test_against_direct_external_evaluator():
    root = Path(os.environ["EVENTREFINE_OFFICIAL_EVALUATOR_ROOT"])
    spec = importlib.util.spec_from_file_location("_direct_evuav", root / "utils" / "eval.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    direct = module.evalute(SimpleNamespace(roc=True, pd_detT=50, correct_thresh=1e-4))
    clip, probabilities = sample()
    locations = torch.tensor(clip["locs"], dtype=torch.float32)
    direct.roc_update(locations[:, 3], torch.tensor(probabilities), clip["idx"].copy(),
                      torch.tensor(clip["seg"], dtype=torch.float32), locations, thresh=.9)
    expected_pd, expected_fa = direct.cal_roc()
    adapter = OfficialEvaluator(root)
    adapter.update(clip, probabilities)
    report = adapter.report()
    assert report["metrics"]["Pd"] == pytest.approx(float(expected_pd) * 100)
    assert report["metrics"]["Fa"] == pytest.approx(float(expected_fa))
    # Preserve the upstream strict interval boundaries and effective denominator.
    assert report["counters"]["obj_num"] == 3
    assert report["counters"]["correct_num"] == 2
    assert report["counters"]["frame_num"] == 3
    assert report["counters"]["false_num"] == 2
