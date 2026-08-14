"""SceneExpert integration boundary for behavior-template planning."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from scenesmith.scene_expert.behavior.assets import merge_behavior_assets
from scenesmith.scene_expert.behavior.planner import build_behavior_spec
from scenesmith.scene_expert.behavior.schemas import BehaviorSpec
from scenesmith.scene_expert.schemas import SceneTaskSpec


def apply_behavior_template(
    prompt: str,
    task_spec: SceneTaskSpec,
    *,
    config: dict[str, Any],
    output_path: Path,
    model: str | None = None,
    api_base_url: str | None = None,
    api_key: str | None = None,
    client: Any | None = None,
) -> tuple[SceneTaskSpec, BehaviorSpec | None]:
    """Plan, persist, and merge behavior assets when the feature is enabled."""
    if not bool(config.get("enabled", False)):
        return task_spec, None
    planner = str(config.get("planner", "template"))
    if planner != "template":
        raise ValueError(
            f"Unsupported scene_expert.behavior.planner={planner!r}. Use 'template'."
        )

    behavior_spec = build_behavior_spec(
        prompt,
        model=model,
        api_base_url=api_base_url,
        api_key=api_key,
        client=client,
        room_type=task_spec.room_type,
        horizon=str(config.get("horizon", "week")),
    )
    merged = merge_behavior_assets(
        task_spec,
        behavior_spec,
        inferred_assets_are_required=bool(
            config.get("inferred_assets_are_required", True)
        ),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        behavior_spec.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )
    return merged, behavior_spec
