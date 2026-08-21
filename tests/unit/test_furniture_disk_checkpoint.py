import copy
import json

import pytest

from types import SimpleNamespace
from unittest.mock import Mock

from scenesmith.agent_utils.furniture_safety import HardStateEvaluation
from scenesmith.furniture_agents.stateful_furniture_agent import StatefulFurnitureAgent


class _CheckpointScene:
    def __init__(self, *, has_loveseat: bool) -> None:
        self.text_description = "A living room requiring one loveseat."
        self.state = self._state(has_loveseat)

    @staticmethod
    def _state(has_loveseat: bool) -> dict:
        return {
            "room_geometry": None,
            "objects": {"loveseat_0": {"id": "loveseat_0"}} if has_loveseat else {},
            "text_description": "A living room requiring one loveseat.",
            "metadata": {},
        }

    @property
    def has_loveseat(self) -> bool:
        return "loveseat_0" in self.state["objects"]

    def to_state_dict(self) -> dict:
        return copy.deepcopy(self.state)

    def restore_from_state_dict(self, state: dict) -> None:
        self.state = copy.deepcopy(state)

    def content_hash(self) -> str:
        return json.dumps(self.state, sort_keys=True)


def _agent(tmp_path, *, has_loveseat: bool) -> StatefulFurnitureAgent:
    agent = StatefulFurnitureAgent.__new__(StatefulFurnitureAgent)
    agent.logger = SimpleNamespace(output_dir=tmp_path)
    agent.scene = _CheckpointScene(has_loveseat=has_loveseat)
    agent.rendering_manager = SimpleNamespace(clear_cache=Mock())
    agent.furniture_safety_controller = SimpleNamespace(
        enabled=True,
        required_counts={"loveseat": 1},
        best_scene_state=None,
        best_scores=None,
        best_render_dir=None,
        best_weighted_score=-1.0,
        best_reasons=[],
    )
    agent._evaluate_current_hard_state = lambda: HardStateEvaluation(
        hard_valid=agent.scene.has_loveseat,
        hard_reasons=(
            [] if agent.scene.has_loveseat else ["missing required loveseat"]
        ),
    )
    agent._checkpoint_eligible_furniture_hard_state = lambda state: state
    return agent


@pytest.mark.parametrize(
    "reason",
    [
        "physics hard violation: collisions",
        "physics hard violation: door clearance violations",
        "unresolved prompt-core furniture relation: surround:table_0",
    ],
)
def test_degraded_policy_records_invalid_checkpoint_without_aborting(
    tmp_path,
    reason: str,
) -> None:
    agent = _agent(tmp_path, has_loveseat=False)
    agent.cfg = {"fail_stage_on_unresolved_hard_constraints": False}
    agent._evaluate_current_hard_state = lambda: HardStateEvaluation(
        hard_valid=False,
        hard_reasons=[reason],
    )

    agent._ensure_furniture_checkpoint_integrity(source="inventory convergence")

    report = json.loads(
        agent._furniture_checkpoint_report_path().read_text(encoding="utf-8")
    )
    assert report["events"][-1]["event"] == "rejected"
    assert report["events"][-1]["reason"] == reason


def test_strict_policy_still_aborts_without_hard_valid_checkpoint(tmp_path) -> None:
    agent = _agent(tmp_path, has_loveseat=False)
    agent.cfg = {"fail_stage_on_unresolved_hard_constraints": True}

    with pytest.raises(RuntimeError, match="missing required loveseat"):
        agent._ensure_furniture_checkpoint_integrity(source="inventory convergence")


def test_new_agent_restores_fresh_verified_disk_checkpoint_with_required_loveseat(
    tmp_path,
) -> None:
    writer = _agent(tmp_path, has_loveseat=True)
    assert writer._persist_furniture_hard_valid_checkpoint(
        scene_state=writer.scene.to_state_dict(),
        source="accepted_critique",
    )

    reader = _agent(tmp_path, has_loveseat=False)
    loaded = reader._load_furniture_hard_valid_checkpoint(
        restore_when_current_invalid=True
    )

    assert loaded
    assert reader.scene.has_loveseat
    assert reader.furniture_safety_controller.best_scene_state is not None
    report = json.loads(
        reader._furniture_checkpoint_report_path().read_text(encoding="utf-8")
    )
    assert [event["event"] for event in report["events"]][-2:] == [
        "saved",
        "restored",
    ]


def test_hard_invalid_candidate_cannot_overwrite_disk_checkpoint(tmp_path) -> None:
    agent = _agent(tmp_path, has_loveseat=True)
    assert agent._persist_furniture_hard_valid_checkpoint(
        scene_state=agent.scene.to_state_dict(),
        source="valid",
    )
    checkpoint_path = agent._furniture_disk_checkpoint_path()
    saved = checkpoint_path.read_text(encoding="utf-8")
    agent.scene.state = agent.scene._state(False)

    persisted = agent._persist_furniture_hard_valid_checkpoint(
        scene_state=agent.scene.to_state_dict(),
        source="hard_invalid_retry",
    )

    assert not persisted
    assert checkpoint_path.read_text(encoding="utf-8") == saved
