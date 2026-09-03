import inspect

from scenesmith.experiments.indoor_scene_generation import (
    IndoorSceneGenerationExperiment,
)
from scenesmith.floor_plan_agents.stateful_floor_plan_agent import (
    StatefulFloorPlanAgent,
)
from scenesmith.floor_plan_agents.tools.floor_plan_tools import FloorPlanTools


def test_floor_plan_agent_accepts_polygon_and_reservation_contracts() -> None:
    parameters = inspect.signature(StatefulFloorPlanAgent.__init__).parameters

    assert "reservation_manifest" in parameters
    assert hasattr(StatefulFloorPlanAgent, "_create_polygon_validation_config")


def test_floor_plan_tools_accepts_polygon_and_reservation_contracts() -> None:
    parameters = inspect.signature(FloorPlanTools.__init__).parameters

    assert "polygon_config" in parameters
    assert "reservation_manifest" in parameters


def test_promptgen_polygon_profile_is_registered_to_stateful_agent() -> None:
    compatible_agents = IndoorSceneGenerationExperiment.compatible_floor_plan_agents

    assert compatible_agents["polygon_promptgen_v3_4"] is StatefulFloorPlanAgent
