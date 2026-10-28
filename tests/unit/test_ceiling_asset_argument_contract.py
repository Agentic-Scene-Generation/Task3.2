"""Malformed ceiling dimensions must stop before native asset side effects."""

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


def _build_ceiling_generation_tool():
    """Execute the actual SDK closure with native asset dependencies replaced."""
    source = Path(__file__).parents[2] / (
        "scenesmith/ceiling_agents/tools/ceiling_tools.py"
    )
    module = ast.parse(source.read_text(encoding="utf-8"))
    cls = next(
        node for node in module.body if getattr(node, "name", "") == "CeilingTools"
    )
    factory = deepcopy(
        next(
            node
            for node in cls.body
            if getattr(node, "name", "") == "_create_tool_closures"
        )
    )
    closure = next(
        node
        for node in factory.body
        if getattr(node, "name", "") == "generate_ceiling_assets"
    )
    factory.body = [
        closure,
        ast.Return(value=ast.Name(id="generate_ceiling_assets", ctx=ast.Load())),
    ]
    namespace = {
        "Any": object,
        "function_tool": function_tool,
        "console_logger": logging.getLogger(__name__),
        "asset_generation_argument_error": asset_generation_argument_error,
        "AssetGenerationResultDTO": AssetGenerationResult,
        "AssetGenerationRequest": SimpleNamespace,
        "ObjectType": SimpleNamespace(CEILING_MOUNTED="ceiling_mounted"),
        "semantic_name_candidates_for_request": Mock(return_value=[]),
    }
    tree = ast.fix_missing_locations(ast.Module(body=[factory], type_ignores=[]))
    exec(compile(tree, str(source), "exec"), namespace)  # noqa: S102
    owner = SimpleNamespace(
        scene=SimpleNamespace(text_description="reading room", scene_dir=Path("room")),
        _generate_assets_impl=Mock(return_value='{"success": true}'),
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
                    "style_context": "warm lighting",
                }
            ),
        )
    )


@pytest.mark.parametrize(
    "descriptions,names,dimensions,expected",
    [
        (["String lights"], ["lights"], [[-2.25, -2.0, 2.25, 2.0]], "exactly three"),
        (["String lights"], ["lights"], [[1.8, -0.2, 0.08]], "strictly positive"),
        (["String lights"], ["lights"], [], "same nonzero length"),
        ([], [], [], "at least one"),
    ],
)
def test_invalid_ceiling_requests_do_not_reach_native_asset_work(
    descriptions, names, dimensions, expected
):
    tool, owner, namespace = _build_ceiling_generation_tool()
    result = json.loads(_invoke(tool, descriptions, names, dimensions))
    assert result["success"] is False
    assert result["assets"] == []
    assert result["successful_count"] == 0
    assert result["failed_count"] == len(descriptions)
    assert expected in result["message"]
    assert "Correct the arguments and retry" in result["message"]
    owner._generate_assets_impl.assert_not_called()
    namespace["semantic_name_candidates_for_request"].assert_not_called()


def test_corrected_ceiling_request_keeps_the_original_native_request():
    tool, owner, namespace = _build_ceiling_generation_tool()
    dimensions = [[1.8, 0.2, 0.08]]
    result = json.loads(_invoke(tool, ["String lights"], ["lights"], dimensions))
    assert result["success"]
    owner._generate_assets_impl.assert_called_once()
    namespace["semantic_name_candidates_for_request"].assert_called_once()
    request = owner._generate_assets_impl.call_args.args[0]
    assert request.object_descriptions == ["String lights"]
    assert request.short_names == ["lights"]
    assert request.desired_dimensions == dimensions
    assert request.object_type == "ceiling_mounted"
    assert request.style_context == "warm lighting"
    assert request.scene_prompt_context == "reading room"
