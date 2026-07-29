"""Tests for optional stage-specific placement-order references."""

import json

from pathlib import Path
from types import SimpleNamespace

import numpy as np

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from scenesmith.agent_utils.furniture_placement_order import (
    build_furniture_placement_order_reference,
)
from scenesmith.agent_utils.manipuland_placement_order import (
    build_manipuland_placement_order_reference,
)
from scenesmith.agent_utils.stage_placement_order_config import (
    append_placement_order_reference,
    get_stage_placement_order_config,
)


class FakeVLMService:
    def __init__(self, response: str | Exception):
        self.response = response
        self.calls = 0
        self.kwargs = []

    def create_completion(self, **kwargs) -> str:
        self.calls += 1
        self.kwargs.append(kwargs)
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


def _surface():
    return SimpleNamespace(
        bounding_box_min=np.array([-0.3, -0.2, 0.0]),
        bounding_box_max=np.array([0.3, 0.2, 0.0]),
        link_name=None,
    )


def _furniture_reference(tmp_path, cfg, vlm):
    return build_furniture_placement_order_reference(
        cfg=cfg,
        scene_prompt="A bedroom with a bed, two nightstands, and a rug.",
        scene_dir=tmp_path,
        vlm_service=vlm,
        model="test-model",
        room_dimensions={"length_m": 5.0, "width_m": 4.0},
    )


def _manipuland_reference(
    tmp_path,
    cfg,
    vlm,
    suggested_items="lamp, clock",
):
    return build_manipuland_placement_order_reference(
        cfg=cfg,
        scene_prompt="A serene bedroom",
        scene_dir=tmp_path,
        vlm_service=vlm,
        model="test-model",
        furniture_id="nightstand_0",
        furniture_description="oak nightstand",
        suggested_items=suggested_items,
        prompt_constraints="A table lamp is required",
        style_notes="Keep the surface minimal",
        support_surfaces={"S_10": _surface()},
    )


def test_stage_config_defaults_closed_and_supports_independent_switches():
    assert not get_stage_placement_order_config({}, "furniture")["enabled"]
    assert not get_stage_placement_order_config({}, "manipuland")["enabled"]

    cfg = {
        "stage_placement_order": {
            "furniture": {"enabled": False},
            "manipuland": {"enabled": True},
        }
    }
    assert not get_stage_placement_order_config(cfg, "furniture")["enabled"]
    assert get_stage_placement_order_config(cfg, "manipuland")["enabled"]


def test_legacy_flat_override_applies_only_to_furniture(caplog):
    cfg = {
        "stage_placement_order": {
            "enabled": True,
            "cache": False,
            "furniture": {"enabled": False, "cache": True},
            "manipuland": {"enabled": False},
        }
    }
    furniture = get_stage_placement_order_config(cfg, "furniture")
    manipuland = get_stage_placement_order_config(cfg, "manipuland")

    assert furniture["enabled"]
    assert not furniture["cache"]
    assert not manipuland["enabled"]
    assert "using the legacy value" in caplog.text


def test_legacy_hydra_override_does_not_require_plus():
    config_dir = Path(__file__).resolve().parents[2] / "configurations"
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        cfg = compose(
            config_name="config",
            overrides=["experiment.stage_placement_order.enabled=true"],
        )
    assert cfg.experiment.stage_placement_order.enabled
    assert not cfg.experiment.stage_placement_order.manipuland.enabled


def test_malformed_config_fails_closed():
    cfg = OmegaConf.create({"stage_placement_order": "${missing.value}"})
    stage_cfg = get_stage_placement_order_config(cfg, "furniture")
    assert stage_cfg == {"enabled": False}


def test_disabled_builders_do_not_call_vlm_or_touch_cache(tmp_path):
    furniture_vlm = FakeVLMService(RuntimeError("must not be called"))
    manipuland_vlm = FakeVLMService(RuntimeError("must not be called"))

    assert _furniture_reference(tmp_path, {}, furniture_vlm) == ""
    assert _manipuland_reference(tmp_path, {}, manipuland_vlm) == ""
    assert furniture_vlm.calls == 0
    assert manipuland_vlm.calls == 0
    assert list(tmp_path.iterdir()) == []


def test_furniture_filters_precise_positions_and_reuses_cache(tmp_path):
    vlm = FakeVLMService(
        """
        ```json
        {'furniture': [
          {'item': 'bed at x=1.20m', 'target': 'x=1.20m, y=2.0m',
           'placement_hint': 'exactly 0.40 meters from the wall',
           'notes': 'primary anchor', 'coordinates': [1.2, 2.0]},
          {'item': 'two nightstands', 'target': 'beside the bed',
           'placement_hint': 'beside the head side of the bed'},
        ],}
        ```
        """
    )
    cfg = {
        "stage_placement_order": {
            "furniture": {
                "enabled": True,
                "cache": True,
                "fallback_enabled": False,
            }
        }
    }

    reference = _furniture_reference(tmp_path, cfg, vlm)

    assert vlm.calls == 1
    assert "bed" in reference.lower()
    assert "Left nightstand" in reference
    assert "Right nightstand" in reference
    assert "1.20m" not in reference
    assert "0.40 meters" not in reference

    cache_path = tmp_path / "furniture_placement_order.json"
    cache_text = cache_path.read_text(encoding="utf-8")
    assert "coordinates" not in cache_text
    assert "1.20m" not in cache_text
    assert "0.40 meters" not in cache_text

    cached_vlm = FakeVLMService(RuntimeError("cache should avoid this call"))
    assert _furniture_reference(tmp_path, cfg, cached_vlm) == reference
    assert cached_vlm.calls == 0


def test_furniture_vlm_failure_uses_fallback(tmp_path):
    cfg = {
        "stage_placement_order": {
            "furniture": {
                "enabled": True,
                "cache": False,
                "fallback_enabled": True,
            }
        }
    }
    reference = _furniture_reference(
        tmp_path,
        cfg,
        FakeVLMService(RuntimeError("generation failed")),
    )
    assert "Bed" in reference
    assert "Left nightstand" in reference
    assert "Area rug" in reference


def test_furniture_limit_is_applied_to_reference_and_cache(tmp_path):
    cfg = {
        "stage_placement_order": {
            "furniture": {
                "enabled": True,
                "cache": True,
                "fallback_enabled": False,
                "max_items": 1,
            }
        }
    }
    reference = _furniture_reference(
        tmp_path,
        cfg,
        FakeVLMService(
            '{"furniture": [{"item": "bed"}, {"item": "wardrobe"}]}'
        ),
    )
    cache = json.loads(
        (tmp_path / "furniture_placement_order.json").read_text(encoding="utf-8")
    )
    assert len(cache) == 1
    assert reference.count("\n1.") == 1
    assert "\n2." not in reference


def test_manipuland_filters_positions_and_reuses_per_furniture_cache(tmp_path):
    vlm = FakeVLMService(
        """
        ```json
        {'manipuland': [
          {'item': 'left table lamp', 'surface': 'surface S_10',
           'notes': 'place near the left edge', 'coordinates': [0.1, 0.2]},
          {'item': 'alarm clock', 'surface': 'S_10',
           'notes': 'small functional item'},
        ],}
        ```
        """
    )
    cfg = {
        "stage_placement_order": {
            "manipuland": {
                "enabled": True,
                "cache": True,
                "fallback_enabled": True,
                "max_items_per_surface": 8,
            }
        }
    }

    reference = _manipuland_reference(tmp_path, cfg, vlm)

    assert "1. table lamp (surface: S_10)" in reference
    assert "2. alarm clock (surface: S_10)" in reference
    assert "left edge" not in reference
    cache = json.loads(
        (tmp_path / "manipuland_placement_orders.json").read_text(encoding="utf-8")
    )
    cache_text = json.dumps(cache)
    assert list(cache) == ["nightstand_0"]
    assert "coordinates" not in cache_text
    assert "left edge" not in cache_text
    assert "left table lamp" not in cache_text

    cached_vlm = FakeVLMService(RuntimeError("cache should avoid this call"))
    assert _manipuland_reference(tmp_path, cfg, cached_vlm) == reference
    assert cached_vlm.calls == 0


def test_manipuland_vlm_failure_fallback_keeps_required_first(tmp_path):
    cfg = {
        "stage_placement_order": {
            "manipuland": {
                "enabled": True,
                "cache": False,
                "fallback_enabled": True,
            }
        }
    }
    reference = _manipuland_reference(
        tmp_path,
        cfg,
        FakeVLMService(RuntimeError("generation failed")),
        suggested_items="Optional: alarm clock, REQUIRED: table lamp",
    )
    assert reference.index("table lamp") < reference.index("alarm clock")
    assert "Required item" in reference


def test_prompt_injection_is_exactly_noop_when_reference_empty():
    original = "ORIGINAL PROMPT"
    assert append_placement_order_reference(original, "") is original
    injected = append_placement_order_reference(original, "ORDER REFERENCE")
    assert injected == "ORIGINAL PROMPT\n\nORDER REFERENCE"
    assert injected.count("ORDER REFERENCE") == 1
