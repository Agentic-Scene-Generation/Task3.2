from __future__ import annotations

import pytest

from scenesmith.scenebenchmark_critic.config import critic_config_from_any


def _parse(auto_repair: object = None):
    value = {} if auto_repair is None else {"auto_repair": auto_repair}
    return critic_config_from_any({"scenebenchmark_critic": value}).auto_repair


def test_auto_repair_defaults_preserve_legacy_behavior() -> None:
    config = _parse()
    assert config.enabled
    assert all(
        config.should_repair(module)
        for module in (
            "furniture_relations",
            "visual_clearance",
            "storage_accessibility",
            "seating_orientation",
            "window_clearance",
        )
    )


@pytest.mark.parametrize("value, enabled", [(True, True), (False, False)])
def test_auto_repair_boolean_shorthand(value: bool, enabled: bool) -> None:
    assert _parse(value).enabled is enabled


def test_auto_repair_mapping_and_budget_parsing() -> None:
    config = _parse(
        {
            "furniture_relations": False,
            "max_candidate_evaluations": "12",
            "storage_accessibility_budget": 0,
        }
    )
    assert not config.should_repair("furniture_relations")
    assert config.max_candidate_evaluations == 12
    assert config.storage_accessibility_budget == 0


@pytest.mark.parametrize(
    "value",
    ["false", 0, [], {"enabled": "fasle"}, {"max_repairs_per_call": True}, {"max_candidate_evaluations": -1}, {"visual_clearance_budget": 1.5}],
)
def test_auto_repair_rejects_ambiguous_values(value: object) -> None:
    with pytest.raises(ValueError):
        _parse(value)


def test_auto_repair_rejects_unknown_fields_and_modules() -> None:
    with pytest.raises(ValueError, match="Unknown.*bogus"):
        _parse({"bogus": True})
    with pytest.raises(ValueError, match="Unknown auto-repair module"):
        _parse().should_repair("bogus")  # type: ignore[arg-type]
