"""Invalid asset batches stop before wall/manipuland policy or native work."""

from __future__ import annotations

import ast
import asyncio
import json
import logging
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from agents import RunContextWrapper, function_tool

from scenesmith.agent_utils.response_datatypes import AssetGenerationResult
from scenesmith.scene_expert.tool_contracts import asset_generation_argument_error


def _build_tool(stage):
    class_name = "WallTools" if stage == "wall" else "ManipulandTools"
    tool_name = f"generate_{stage}_assets"
    source = Path(__file__).parents[2] / (
        f"scenesmith/{stage}_agents/tools/{stage}_tools.py"
    )
    module = ast.parse(source.read_text(encoding="utf-8"))
    cls = next(node for node in module.body if getattr(node, "name", "") == class_name)
    factory = deepcopy(
        next(
            node
            for node in cls.body
            if getattr(node, "name", "") == "_create_tool_closures"
        )
    )
    closure = next(
        node for node in factory.body if getattr(node, "name", "") == tool_name
    )
    factory.body = [closure, ast.Return(ast.Name(id=tool_name, ctx=ast.Load()))]
    namespace = {
        "Any": object,
        "json": json,
        "function_tool": function_tool,
        "console_logger": logging.getLogger(__name__),
        "asset_generation_argument_error": asset_generation_argument_error,
        "AssetGenerationResult": AssetGenerationResult,
        "AssetGenerationResultDTO": AssetGenerationResult,
        "AssetGenerationRequest": SimpleNamespace,
        "ObjectType": SimpleNamespace(WALL_MOUNTED="wall", MANIPULAND="manipuland"),
        "semantic_name_candidates_for_request": Mock(return_value=[]),
        "existing_floor_covering_ids": Mock(return_value=[]),
        "redundant_floor_covering_request_indices": Mock(return_value=[]),
    }
    tree = ast.fix_missing_locations(ast.Module(body=[factory], type_ignores=[]))
    exec(compile(tree, str(source), "exec"), namespace)  # noqa: S102
    owner = SimpleNamespace(
        scene=SimpleNamespace(text_description="reading room", scene_dir=Path("room")),
        current_furniture_id="side_table",
        _generate_assets_impl=Mock(return_value='{"success": true}'),
        _filter_fulfilled_floor_requirement_requests=Mock(
            side_effect=lambda **x: (
                x["object_descriptions"], x["short_names"], x["desired_dimensions"], []
            )
        ),
        _budget_dining_centerpiece_dimensions=Mock(
            side_effect=lambda **x: x["desired_dimensions"]
        ),
    )
    return namespace["_create_tool_closures"](owner), owner, namespace


def _invoke(tool, descriptions, names, dimensions):
    return asyncio.run(
        tool.on_invoke_tool(
            RunContextWrapper(context=None),
            json.dumps(
                {
                    "object_descriptions": descriptions,
                    "short_names": names,
                    "desired_dimensions": dimensions,
                    "style_context": "cozy",
                }
            ),
        )
    )


@pytest.mark.parametrize("stage", ["wall", "manipuland"])
@pytest.mark.parametrize(
    "descriptions,names,dimensions,expected",
    [
        (["Book"], ["book"], [[-2.25, -2, 2.25, 2]], "exactly three"),
        (["Book"], ["book"], [[0.2, -0.1, 0.03]], "strictly positive"),
        (["Book"], ["book"], [], "same nonzero length"),
        (["Book", "Mug"], ["book"], [[0.2, 0.1, 0.03]], "same nonzero length"),
        ([], [], [], "at least one"),
    ],
)
def test_invalid_stage_asset_batches_precede_all_policy_and_backend_calls(
    stage, descriptions, names, dimensions, expected
):
    tool, owner, namespace = _build_tool(stage)
    result = json.loads(_invoke(tool, descriptions, names, dimensions))
    assert result["success"] is False
    assert result["assets"] == []
    assert result["failed_count"] == len(descriptions)
    assert expected in result["message"]
    assert "Correct the arguments and retry" in result["message"]
    owner._generate_assets_impl.assert_not_called()
    owner._filter_fulfilled_floor_requirement_requests.assert_not_called()
    owner._budget_dining_centerpiece_dimensions.assert_not_called()
    for name in [
        "semantic_name_candidates_for_request",
        "existing_floor_covering_ids",
        "redundant_floor_covering_request_indices",
    ]:
        namespace[name].assert_not_called()


@pytest.mark.parametrize("stage", ["wall", "manipuland"])
def test_valid_stage_requests_preserve_existing_policy_flow_and_native_values(stage):
    tool, owner, namespace = _build_tool(stage)
    dimensions = [[0.2, 0.1, 0.03]]
    assert json.loads(_invoke(tool, ["Book"], ["book"], dimensions))["success"]
    owner._generate_assets_impl.assert_called_once()
    namespace["semantic_name_candidates_for_request"].assert_called_once()
    request = owner._generate_assets_impl.call_args.args[0]
    assert request.desired_dimensions == dimensions
    assert request.object_type == stage
    assert request.scene_prompt_context == "reading room"
    if stage == "manipuland":
        owner._filter_fulfilled_floor_requirement_requests.assert_called_once()
        owner._budget_dining_centerpiece_dimensions.assert_called_once()
        namespace["existing_floor_covering_ids"].assert_called_once()
        namespace["redundant_floor_covering_request_indices"].assert_called_once()
