"""Regression tests for ceiling asset retrieval circuit breaking."""

import json

from scenesmith.agent_utils.asset_manager import AssetGenerationResult, FailedAsset
from scenesmith.ceiling_agents.tools.ceiling_tools import CeilingTools


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
