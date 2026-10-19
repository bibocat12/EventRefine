from pathlib import Path

import pytest
import yaml

from eventrefine.config import validate_config

CONFIG = Path(__file__).parents[1] / "configs/eventrefine_global_reference.yaml"


def test_public_config_is_fixed_and_fail_closed():
    clean = validate_config(yaml.safe_load(CONFIG.read_text()))
    assert clean["model"]["graph_operator"] == "clipped_global"
    assert tuple(clean["training"]["seeds"]) == (37, 1337, 2024)

    bad = yaml.safe_load(CONFIG.read_text())
    bad["model"]["unexpected"] = True
    with pytest.raises(ValueError, match="unknown model"):
        validate_config(bad)
