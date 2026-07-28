from __future__ import annotations

from scenesmith.scenebenchmark_critic.metrics.interaction_clearance.evaluator import (
    build_clearance_checks,
)


def _object(
    object_id: str,
    *,
    category: str,
    object_type: str,
    bbox_min: tuple[float, float, float],
    bbox_max: tuple[float, float, float],
    yaw_deg: float = 0.0,
    clearance: dict[str, object] | None = None,
) -> dict[str, object]:
    metadata: dict[str, object] = {}
    if clearance is not None:
        metadata["clearance"] = clearance
    return {
        "id": object_id,
        "name": object_id,
        "category": category,
        "object_type": object_type,
        "bbox_world": {"min": bbox_min, "max": bbox_max},
        "yaw_deg": yaw_deg,
        "metadata": metadata,
    }


def _monitor() -> dict[str, object]:
    return _object(
        "monitor_0",
        category="monitor",
        object_type="manipuland",
        bbox_min=(-1.1, -1.6, 0.75),
        bbox_max=(-0.9, -1.4, 1.1),
        yaw_deg=90.0,
        clearance={
            "clearance_type": "approach",
            "direction": "front",
            "depth_m": 0.45,
            "height_m": 1.8,
            "confidence": "high",
        },
    )


def _clearance_label(*objects: dict[str, object]) -> str:
    checks = build_clearance_checks({str(obj["id"]): obj for obj in objects})
    return str(checks[0]["clearance_result"]["label"])


def test_high_ceiling_fixture_does_not_block_horizontal_clearance() -> None:
    fixture = _object(
        "ceiling_light_0",
        category="ceiling_light",
        object_type="ceiling_mounted",
        bbox_min=(-1.2, -1.6, 2.36),
        bbox_max=(-0.9, -1.4, 2.7),
    )

    assert _clearance_label(_monitor(), fixture) == "pass"


def test_low_ceiling_fixture_still_blocks_horizontal_clearance() -> None:
    fixture = _object(
        "pendant_0",
        category="pendant_light",
        object_type="ceiling_mounted",
        bbox_min=(-1.2, -1.6, 1.85),
        bbox_max=(-0.9, -1.4, 2.1),
    )

    assert _clearance_label(_monitor(), fixture) == "fail"
