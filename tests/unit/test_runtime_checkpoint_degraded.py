import json

from pathlib import Path
from types import SimpleNamespace

from scenesmith.agent_utils.base_stateful_agent import BaseStatefulAgent
from scenesmith.agent_utils.room import AgentType
from scenesmith.experiments.indoor_scene_generation import (
    _checkpoint_degraded_continuation,
)


class _CheckpointLogger:
    def __init__(self, root: Path) -> None:
        self.root = root

    def log_scene(self, _scene, *, name: str) -> Path:
        checkpoint_dir = self.root / "scene_states" / name
        checkpoint_dir.mkdir(parents=True)
        (checkpoint_dir / "scene_state.json").write_text(
            json.dumps({"objects": {"chair_0": {}}}), encoding="utf-8"
        )
        return checkpoint_dir


class _Agent(BaseStatefulAgent):
    @property
    def agent_type(self):
        return AgentType.FURNITURE

    def _get_initial_design_prompt_enum(self):
        return None

    def _get_design_change_prompt_enum(self):
        return None

    def _get_critique_prompt_enum(self):
        return None

    def _get_initial_design_prompt_kwargs(self):
        return {}

    def _get_final_scores_directory(self):
        return ""

    def _set_placement_noise_profile(self, _mode):
        return None


def _scene(path: Path, *, scene_hash: str = "scene-hash"):
    return SimpleNamespace(
        metadata={
            "scenesmith_runtime_failure": {
                "child_agent": "designer",
                "operation": "request_design_change",
                "error_type": "APITimeoutError",
                "error": "timed out",
                "recovered": False,
                "checkpoint": {
                    "path": str(path),
                    "scene_hash": scene_hash,
                    "validation": "passed",
                    "persisted": True,
                },
            }
        },
        content_hash=lambda: scene_hash,
    )


def test_checkpoint_degraded_requires_transient_persisted_hash_consistent_state(
    tmp_path: Path,
) -> None:
    state = tmp_path / "scene_state.json"
    state.write_text(json.dumps({"objects": {"chair_0": {}}}), encoding="utf-8")
    scene = _scene(state)

    provenance = _checkpoint_degraded_continuation(
        scene=scene,
        stage="furniture",
        cfg_dict={"experiment": {"runtime_failure_policy": "checkpoint_degraded"}},
    )

    assert provenance is not None
    assert provenance["continuation_policy"] == "checkpoint_degraded"
    assert provenance["failure"]["recovered"] is True
    assert scene.metadata["scenesmith_runtime_degraded"] == provenance


def test_runtime_checkpoint_policy_keeps_strict_and_invalid_checkpoint_blocking(
    tmp_path: Path,
) -> None:
    state = tmp_path / "scene_state.json"
    state.write_text(json.dumps({"objects": {"chair_0": {}}}), encoding="utf-8")

    assert (
        _checkpoint_degraded_continuation(
            scene=_scene(state),
            stage="furniture",
            cfg_dict={"experiment": {"runtime_failure_policy": "strict"}},
        )
        is None
    )
    inconsistent_scene = _scene(state)
    inconsistent_scene.metadata["scenesmith_runtime_failure"]["checkpoint"][
        "scene_hash"
    ] = "different"
    assert (
        _checkpoint_degraded_continuation(
            scene=inconsistent_scene,
            stage="furniture",
            cfg_dict={"experiment": {"runtime_failure_policy": "checkpoint_degraded"}},
        )
        is None
    )


def test_agent_only_persists_runtime_checkpoint_after_real_designer_mutation(
    tmp_path: Path,
) -> None:
    agent = object.__new__(_Agent)
    agent._planner_successful_designer_mutations = 0
    agent.scene = SimpleNamespace(content_hash=lambda: "scene-hash", metadata={})
    agent.logger = _CheckpointLogger(tmp_path)
    failure = {"error_type": "APITimeoutError"}

    agent._persist_runtime_failure_checkpoint(failure)
    assert "checkpoint" not in failure

    agent._planner_successful_designer_mutations = 1
    agent._persist_runtime_failure_checkpoint(failure)
    assert failure["checkpoint"]["validation"] == "passed"
    assert (
        agent.scene.metadata["scenesmith_runtime_failure"]["checkpoint"]["scene_hash"]
        == "scene-hash"
    )
