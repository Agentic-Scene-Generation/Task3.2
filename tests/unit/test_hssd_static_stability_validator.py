import importlib.util
from pathlib import Path

import pytest


SCRIPT = Path(__file__).parents[2] / "scripts" / "validate_hssd_static_stability.py"
SPEC = importlib.util.spec_from_file_location("hssd_static_stability_validator", SCRIPT)
VALIDATOR = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(VALIDATOR)


def test_wide_centered_support_survives_five_degree_tilt():
    pytest.importorskip("mujoco")
    result = VALIDATOR._simulate_box(
        extents_hssd=[1.0, 2.0, 1.0],
        mass=10.0,
        friction=0.5,
        support={
            "half_extents_z_up_m": [0.5, 0.5, 0.02],
            "center_relative_to_bbox_z_up_m": [0.0, 0.0, -0.98],
        },
        seconds=2.0,
        timestep=0.002,
        perturb_deg=5.0,
        axis="y",
    )
    assert result["fell"] is False
    assert result["final_tilt_deg"] < 1.0


def test_offset_support_topples_without_added_tilt():
    pytest.importorskip("mujoco")
    result = VALIDATOR._simulate_box(
        extents_hssd=[1.0, 2.0, 1.0],
        mass=10.0,
        friction=0.5,
        support={
            "half_extents_z_up_m": [0.08, 0.08, 0.02],
            "center_relative_to_bbox_z_up_m": [0.18, 0.0, -0.98],
        },
        seconds=2.0,
        timestep=0.002,
        perturb_deg=0.0,
        axis="y",
    )
    assert result["fell"] is True
    assert result["max_tilt_deg"] > 15.0

