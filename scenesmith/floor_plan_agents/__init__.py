"""Floor plan agents for designing and generating house layouts.

Agent classes are loaded lazily so lightweight geometry helpers can be imported
by shared validation code without initializing the complete agent/physics stack.
"""

__all__ = [
    "BaseFloorPlanAgent",
    "StatefulFloorPlanAgent",
]


def __getattr__(name: str):
    """Preserve public imports while avoiding eager agent initialization."""
    if name == "BaseFloorPlanAgent":
        from scenesmith.floor_plan_agents.base_floor_plan_agent import (
            BaseFloorPlanAgent,
        )

        return BaseFloorPlanAgent
    if name == "StatefulFloorPlanAgent":
        from scenesmith.floor_plan_agents.stateful_floor_plan_agent import (
            StatefulFloorPlanAgent,
        )

        return StatefulFloorPlanAgent
    raise AttributeError(name)
