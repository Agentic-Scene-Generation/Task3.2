"""Model argument errors remain recoverable before native asset side effects.

The SDK tests execute the actual furniture closure from the source AST, with
native asset dependencies replaced by spies; they require no Drake or GPU.
"""

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
from scenesmith.scene_expert.slow_memory.paired import validate_tool_execution
from scenesmith.scene_expert.tool_contracts import asset_generation_argument_error


@pytest.mark.parametrize(
    "descriptions,names,dimensions,expected",
    [
        ([], [], [], "at least one"),
        (["TV"], ["tv"], [], "same nonzero length"),
        (["TV"], [], [[1, 0.3, 0.8]], "same nonzero length"),
        (["TV", "sofa"], ["tv", "sofa"], [[1, 0.3, 0.8]], "same nonzero length"),
        (["TV"], ["tv"], [[], [1, 0.3, 0.8]], "same nonzero length"),
        (["TV"], ["tv"], [[]], "exactly three"),
        (["TV"], ["tv"], [[1, 0.3]], "exactly three"),
        (["TV"], ["tv"], [[1, 0.3, 0.8, 1]], "exactly three"),
        (["TV"], ["tv"], [[1, 0, 0.8]], "strictly positive"),
        (["TV"], ["tv"], [[1, -0.3, 0.8]], "strictly positive"),
        (["TV"], ["tv"], [[1, float("nan"), 0.8]], "finite"),
        (["TV"], ["tv"], [[1, float("inf"), 0.8]], "finite"),
        (["TV"], ["tv"], [[1, True, 0.8]], "numbers"),
        (["TV"], ["tv"], [[1, "wide", 0.8]], "numbers"),
        ([" "], ["tv"], [[1, 0.3, 0.8]], "object_descriptions[0]"),
        (["TV"], [""], [[1, 0.3, 0.8]], "short_names[0]"),
    ],
)
def test_invalid_arguments_have_actionable_feedback(
    descriptions, names, dimensions, expected
):
    assert expected in asset_generation_argument_error(descriptions, names, dimensions)


def test_valid_batch_preserves_every_requested_value():
    args = (["TV", "sofa"], ["tv", "sofa"], [[1, 0.3, 0.8], [2, 1, 1]])
    before = deepcopy(args)
    assert asset_generation_argument_error(*args) is None
    assert args == before


def _build_generation_tool():
    """Load the real decorated closure without importing native asset servers."""
    source = Path(__file__).parents[2] / (
        "scenesmith/furniture_agents/tools/furniture_tools.py"
    )
    module = ast.parse(source.read_text(encoding="utf-8"))
    cls = next(
        node for node in module.body if getattr(node, "name", "") == "FurnitureTools"
    )
    factory = deepcopy(
        next(
            node
            for node in cls.body
            if getattr(node, "name", "") == "_create_tool_closures"
        )
    )
    closure = next(
        node for node in factory.body if getattr(node, "name", "") == "generate_assets"
    )
    factory.body = [
        closure,
        ast.Return(value=ast.Name(id="generate_assets", ctx=ast.Load())),
    ]
    ast_module = ast.fix_missing_locations(ast.Module(body=[factory], type_ignores=[]))
    policy = Mock(
        side_effect=lambda **kwargs: SimpleNamespace(
            object_descriptions=kwargs["object_descriptions"],
            short_names=kwargs["short_names"],
            desired_dimensions=kwargs["desired_dimensions"],
            notes=[],
        )
    )
    namespace = {
        "Any": object,
        "function_tool": function_tool,
        "console_logger": logging.getLogger(__name__),
        "asset_generation_argument_error": asset_generation_argument_error,
        "AssetGenerationResult": AssetGenerationResult,
        "apply_bedroom_asset_size_policy": policy,
        "AssetGenerationRequest": SimpleNamespace,
        "ObjectType": SimpleNamespace(FURNITURE="furniture"),
        "semantic_name_candidates_for_request": Mock(return_value=[]),
        "forbidden_semantic_components_for_request": Mock(return_value=[]),
    }
    # Only the trusted repository closure is compiled, never model input.
    exec(compile(ast_module, str(source), "exec"), namespace)  # noqa: S102
    owner = SimpleNamespace(
        _safety_denial_generate_assets=Mock(return_value=None),
        _generate_assets_impl=Mock(return_value='{"success": true}'),
        cfg=SimpleNamespace(),
        scene=SimpleNamespace(text_description="living room", scene_dir=Path("room")),
    )
    return namespace["_create_tool_closures"](owner), owner, policy


def _invoke(tool, dimensions):
    return asyncio.run(
        tool.on_invoke_tool(
            RunContextWrapper(context=None),
            json.dumps(
                {
                    "object_descriptions": ["Flat screen television"],
                    "short_names": ["television"],
                    "desired_dimensions": dimensions,
                    "style_context": None,
                }
            ),
        )
    )


def test_observed_empty_dimensions_and_retry_do_not_consume_asset_allowance():
    tool, owner, policy = _build_generation_tool()
    outputs = [_invoke(tool, []), _invoke(tool, [[]])]
    for output in outputs:
        result = json.loads(output)
        assert result["success"] is False
        assert result["assets"] == []
        assert result["successful_count"] == 0
        assert "Correct the arguments and retry" in result["message"]
    owner._safety_denial_generate_assets.assert_not_called()
    owner._generate_assets_impl.assert_not_called()
    policy.assert_not_called()
    assert json.loads(_invoke(tool, [[1.25, 0.35, 0.8]]))["success"] is True
    owner._safety_denial_generate_assets.assert_called_once_with()
    policy.assert_called_once()
    owner._generate_assets_impl.assert_called_once()
    request = owner._generate_assets_impl.call_args.args[0]
    assert request.desired_dimensions == [[1.25, 0.35, 0.8]]
    assert request.object_descriptions == ["Flat screen television"]
    # Recoverable feedback stays in the recorded trajectory; it is not a
    # transport exception and does not itself determine a preference label.
    validate_tool_execution({"tool_results": [{"output": x} for x in outputs]})


def test_valid_requests_still_obey_the_existing_safety_denial():
    tool, owner, policy = _build_generation_tool()
    owner._safety_denial_generate_assets.return_value = (
        "Safety controller blocked generate_assets"
    )
    assert (
        _invoke(tool, [[1.25, 0.35, 0.8]])
        == "Safety controller blocked generate_assets"
    )
    owner._safety_denial_generate_assets.assert_called_once_with()
    owner._generate_assets_impl.assert_not_called()
    policy.assert_not_called()


def test_unhandled_backend_failures_still_quarantine_the_candidate():
    tool, owner, _ = _build_generation_tool()
    owner._generate_assets_impl.side_effect = ConnectionError(
        "asset server disconnected"
    )
    output = _invoke(tool, [[1.25, 0.35, 0.8]])
    with pytest.raises(ValueError, match="infrastructure or unhandled-tool failure"):
        validate_tool_execution({"tool_results": [{"output": output}]})
