"""Offline slow-memory data curation and DPO training support.

The online SceneSmith pipeline remains authoritative.  This package only
observes its persisted designer, critic, and deterministic-repair evidence and
turns verified observations into auditable offline training artifacts.
"""

from scenesmith.scene_expert.slow_memory.schemas import (
    DPOPreferencePair,
    PreferenceEvidence,
    TrajectoryOutcome,
    TrajectoryRecord,
)
from scenesmith.scene_expert.slow_memory.trajectory import TrajectoryCollector

__all__ = [
    "DPOPreferencePair",
    "PreferenceEvidence",
    "TrajectoryCollector",
    "TrajectoryOutcome",
    "TrajectoryRecord",
]
