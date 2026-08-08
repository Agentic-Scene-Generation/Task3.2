"""Regression tests for ceiling asset retrieval circuit breaking."""

import json

import numpy as np

from pydrake.math import RigidTransform

from scenesmith.agent_utils.asset_manager import (
    AssetGenerationRequest,
    AssetGenerationResult,
    FailedAsset,
)
from scenesmith.agent_utils.room import ObjectType, SceneObject, UniqueID
from scenesmith.ceiling_agents.tools.ceiling_tools import (
    CeilingTools,
    _is_structural_only_ceiling_prompt,
)
from scenesmith.ceiling_agents.tools.ceiling_tools import compute_ceiling_transform
from scenesmith.ceiling_agents.tools.response_dataclasses import CeilingErrorType


class _AlwaysFailingAssetManager:
    def __init__(self) -> None:
        self.calls = 0

    def generate_assets(self, request: object) -> AssetGenerationResult:
        self.calls += 1
        return AssetGenerationResult(
            successful_assets=[],
            failed_assets=[
                FailedAsset(
                    index=0,
                    description="unavailable ceiling decoration",
                    error_message="no compatible HSSD candidate",
                )
            ],
        )


class _RecordingAssetManager:
    def __init__(self, *assets: SceneObject) -> None:
        self.assets = list(assets)
        self.requests: list[AssetGenerationRequest] = []

    def generate_assets(self, request: AssetGenerationRequest) -> AssetGenerationResult:
        self.requests.append(request)
        return AssetGenerationResult(successful_assets=self.assets, failed_assets=[])


class _CeilingScene:
    def __init__(self, *objects: SceneObject) -> None:
        self.objects = {obj.object_id: obj for obj in objects}

    def get_objects_by_type(self, object_type: ObjectType) -> list[SceneObject]:
        return [obj for obj in self.objects.values() if obj.object_type == object_type]

    def get_object(self, object_id: UniqueID) -> SceneObject | None:
        return self.objects.get(object_id)


def _ceiling_object(
    object_id: str, x: float, y: float, *, width: float = 1.0, depth: float = 0.2
) -> SceneObject:
    return SceneObject(
        object_id=UniqueID(object_id),
        object_type=ObjectType.CEILING_MOUNTED,
        name="ceiling_beam",
        description="ceiling beam",
        transform=compute_ceiling_transform(x, y, 0.0, 3.0),
        bbox_min=np.array([-width / 2.0, -depth / 2.0, -0.2]),
        bbox_max=np.array([width / 2.0, depth / 2.0, 0.0]),
    )


def _tools_with_scene(*objects: SceneObject) -> CeilingTools:
    tools = object.__new__(CeilingTools)
    tools.scene = _CeilingScene(*objects)
    tools.room_bounds = (-2.0, -2.0, 2.0, 2.0)
    tools.ceiling_height = 3.0
    return tools


def test_ceiling_asset_generation_stops_after_consecutive_empty_failures() -> None:
    """Semantic prompt rewrites must not cause an unbounded retrieval loop."""
    tools = object.__new__(CeilingTools)
    manager = _AlwaysFailingAssetManager()
    tools.asset_manager = manager
    tools._max_consecutive_asset_generation_failures = 2
    tools._consecutive_asset_generation_failures = 0
    tools._asset_generation_circuit_open = False

    first_result = json.loads(tools._generate_assets_impl(object()))
    second_result = json.loads(tools._generate_assets_impl(object()))
    third_result = json.loads(tools._generate_assets_impl(object()))

    assert manager.calls == 2
    assert first_result["success"] is False
    assert "circuit is open" in second_result["message"]
    assert "Do not call generate_ceiling_assets again" in third_result["message"]


def test_structural_only_prompt_filters_invented_lighting_from_batch() -> None:
    tools = object.__new__(CeilingTools)
    manager = _RecordingAssetManager(_ceiling_object("beam_asset", 0.0, 0.0))
    tools.asset_manager = manager
    tools._max_consecutive_asset_generation_failures = 2
    tools._consecutive_asset_generation_failures = 0
    tools._asset_generation_circuit_open = False
    request = AssetGenerationRequest(
        object_descriptions=["Exposed reclaimed-wood ceiling beam", "Pendant light"],
        short_names=["wooden_beam", "pendant_light"],
        object_type=ObjectType.CEILING_MOUNTED,
        desired_dimensions=[[4.5, 0.2, 0.15], [0.5, 0.5, 0.6]],
        scene_prompt_context="A rustic bedroom with exposed wooden beams on the ceiling.",
        semantic_name_candidates=[["wooden_beam"], ["pendant_light"]],
    )

    response = json.loads(tools._generate_assets_impl(request))

    assert [request.object_descriptions for request in manager.requests] == [
        ["Exposed reclaimed-wood ceiling beam"]
    ]
    assert response["success"] is False
    assert response["successful_count"] == 1
    assert response["failed_count"] == 1
    assert "Rejected unrequested lighting asset" in response["failures"]


def test_prompt_explicitly_requesting_lighting_does_not_filter_it() -> None:
    tools = object.__new__(CeilingTools)
    manager = _RecordingAssetManager(
        _ceiling_object("beam_asset", 0.0, 0.0),
        _ceiling_object("pendant_asset", 1.0, 0.0),
    )
    tools.asset_manager = manager
    tools._max_consecutive_asset_generation_failures = 2
    tools._consecutive_asset_generation_failures = 0
    tools._asset_generation_circuit_open = False
    request = AssetGenerationRequest(
        object_descriptions=["Exposed reclaimed-wood ceiling beam", "Pendant light"],
        short_names=["wooden_beam", "pendant_light"],
        object_type=ObjectType.CEILING_MOUNTED,
        desired_dimensions=[[4.5, 0.2, 0.15], [0.5, 0.5, 0.6]],
        scene_prompt_context=(
            "A rustic bedroom with exposed wooden beams and a pendant light."
        ),
    )

    response = json.loads(tools._generate_assets_impl(request))

    assert [request.object_descriptions for request in manager.requests] == [
        ["Exposed reclaimed-wood ceiling beam", "Pendant light"]
    ]
    assert response["success"] is True


def test_structural_filter_ignores_generated_stage_guidance() -> None:
    prompt = (
        "A rustic bedroom with exposed wooden beams.\n\n"
        "=== SceneExpert Stage Brief: ceiling_mounted ===\n"
        "Consider balanced lighting and avoid inventing fixtures.\n"
        "=== End Stage Brief ==="
    )

    assert _is_structural_only_ceiling_prompt(prompt)


def test_ceiling_placement_rejects_full_footprint_outside_room() -> None:
    tools = _tools_with_scene()
    beam = _ceiling_object("beam_asset", 0.0, 0.0, width=1.0)

    error = tools._ceiling_placement_error(
        beam, compute_ceiling_transform(1.8, 0.0, 0.0, 3.0)
    )

    assert error is not None
    assert error[0] == CeilingErrorType.POSITION_OUT_OF_BOUNDS


def test_ceiling_move_rejects_overlap_without_mutating_scene() -> None:
    moving = _ceiling_object("beam_moving", -1.0, 0.0, width=1.5)
    existing = _ceiling_object("beam_existing", 0.5, 0.0, width=1.5)
    tools = _tools_with_scene(moving, existing)
    before = moving.transform.GetAsMatrix4().copy()

    response = json.loads(
        tools._move_ceiling_object_impl(
            object_id="beam_moving",
            position_x=0.5,
            position_y=0.0,
        )
    )

    assert response["success"] is False
    assert response["error_type"] == CeilingErrorType.OVERLAPS_CEILING_OBJECT.value
    np.testing.assert_allclose(moving.transform.GetAsMatrix4(), before)
