import pytest
import torch

from eventrefine.model import EventRefineModel


def test_model_has_fixed_public_seams_and_finite_forward():
    model = EventRefineModel()
    assert model.encoder_context == "local_attention"
    assert model.graph_operator == "clipped_global"
    assert model.refinement_mode == "feature_graph"
    with pytest.raises(ValueError):
        EventRefineModel(k=4)


def test_checkpoint_shape_seam_is_deterministic():
    first = EventRefineModel()
    second = EventRefineModel()
    assert list(first.state_dict()) == list(second.state_dict())
    assert all(torch.isfinite(value).all() for value in first.state_dict().values())
