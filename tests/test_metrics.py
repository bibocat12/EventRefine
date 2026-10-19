import numpy as np

from eventrefine.metrics import confusion, summarize


def test_metrics_use_foreground_iou_and_recall():
    counts = confusion(np.array([0.95, 0.10, 0.91, 0.20]), np.array([1, 0, 0, 1]))
    result = summarize(counts)
    assert counts == {"TP": 1, "FP": 1, "FN": 1, "TN": 1, "events": 4}
    assert result["IoU"] == 100.0 / 3.0
    assert result["Recall"] == 50.0
