import numpy as np
import pytest

from eventrefine.dataset import convert_official_arrays


def test_ev_uav_archive_conversion_and_alignment():
    arrays = {
        "ev_loc": np.array([[1, 2, 10], [3, 4, 20]], dtype=np.int64),
        "evs_norm": np.array([[0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 1, 2]], dtype=np.float32),
    }
    converted = convert_official_arrays(arrays, "val_000")
    assert converted["locs"].shape == (2, 4)
    assert converted["locs"][:, 3].tolist() == [10, 20]
    assert converted["seg"].tolist() == [0, 1]
    assert converted["idx"].tolist() == [0, 2]


def test_ev_uav_archive_rejects_nonmonotonic_time():
    arrays = {
        "ev_loc": np.array([[1, 2, 20], [3, 4, 10]], dtype=np.int64),
        "evs_norm": np.zeros((2, 6), dtype=np.float32),
    }
    with pytest.raises(ValueError, match="timestamps"):
        convert_official_arrays(arrays, "val_000")
