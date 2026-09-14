"""Raw scoring must observe final geometry rather than the last tool cache."""

from __future__ import annotations

import copy
import sys
from types import ModuleType, SimpleNamespace

import pytest

from scenesmith.scene_expert.slow_memory.paired import digest
from scenesmith.scene_expert.slow_memory.paired_scoring import score_raw_candidate


@pytest.mark.parametrize("cached_collisions,fresh_collisions", [(3, 0), (0, 1)])
def test_scoring_replaces_stale_physics_in_detached_case_pack(
    monkeypatch: pytest.MonkeyPatch, cached_collisions: int, fresh_collisions: int
) -> None:
    scene, cfg, calls = _boundaries(monkeypatch, cached_collisions, fresh_collisions)
    before = copy.deepcopy(scene.state)
    report, proof = score_raw_candidate(scene, cfg)
    assert report["summary"]["scene_summary"]["fail"] == fresh_collisions
    assert (
        report["case_pack"]["physics_evidence"]["source_phase"]
        == "sceneexpert_raw_candidate"
    )
    assert proof["raw_state_sha256"] == digest(before)
    assert scene.state == before
    assert calls == ["physics", "adapter", "checks", "critic"]


@pytest.mark.parametrize(
    "failure", ["physics_error", "mutate_geometry", "mutate_metadata"]
)
def test_scoring_rejects_unavailable_or_mutating_evaluation(
    monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    scene, cfg, _ = _boundaries(monkeypatch, 0, 0, failure=failure)
    with pytest.raises((ValueError, RuntimeError)):
        score_raw_candidate(scene, cfg)


def _boundaries(
    monkeypatch: pytest.MonkeyPatch, cached: int, fresh: int, *, failure: str = ""
):
    class Scene:
        room_id = "bedroom"

        def __init__(self) -> None:
            self.state = {
                "objects": {"bed": {"position": 1}},
                "metadata": {"physics": {"collisions": list(range(cached))}},
            }

        def to_state_dict(self):
            return self.state

        def content_hash(self):
            return digest(self.state["objects"])

    scene = Scene()
    calls = []
    cfg = SimpleNamespace(
        physics_validation=SimpleNamespace(
            object_penetration_threshold_m=0.001,
            floor_penetration_tolerance_m=0.05,
            manipuland_furniture_tolerance_m=0.02,
        )
    )
    physics = ModuleType("scenesmith.agent_utils.physics_validation")

    def compute(**kwargs):
        calls.append("physics")
        assert kwargs["scene"] is scene
        assert kwargs["penetration_threshold"] == 0.001
        if failure == "physics_error":
            raise RuntimeError("Drake unavailable")
        if failure == "mutate_geometry":
            scene.state["objects"]["bed"]["position"] = 2
        if failure == "mutate_metadata":
            scene.state["metadata"]["corrupt"] = True
        return [
            SimpleNamespace(
                object_a_id="bed",
                object_b_id="wall",
                object_a_name="bed",
                object_b_name="wall",
                penetration_depth=0.1,
            )
            for _ in range(fresh)
        ]

    physics.compute_scene_collisions = compute
    adapter = ModuleType("scenesmith.scenebenchmark_critic.adapter")

    def pack(*args, **kwargs):
        calls.append("adapter")
        return {"physics_evidence": copy.deepcopy(scene.state["metadata"]["physics"])}

    adapter.room_scene_to_case_pack = pack
    config = ModuleType("scenesmith.scenebenchmark_critic.config")
    config.critic_config_from_any = lambda cfg: SimpleNamespace(
        enabled=True, metrics=["physics_collision"]
    )
    evaluator = ModuleType("scenesmith.scenebenchmark_critic.evaluator")

    def build(case_pack, **kwargs):
        calls.append("checks")
        return list(case_pack["physics_evidence"]["collisions"])

    evaluator.build_all_checks = build

    def evaluate(case_pack, **kwargs):
        calls.append("critic")
        assert case_pack["checks"] == case_pack["physics_evidence"]["collisions"]
        return case_pack["checks"]

    evaluator.run_case_pack_checks = evaluate
    reports = ModuleType("scenesmith.scenebenchmark_critic.reports")
    reports.build_evaluation_payload = lambda **kw: {
        "case_pack": kw["case_pack"],
        "summary": {"scene_summary": {"fail": len(kw["results"])}},
    }
    for module in (physics, adapter, config, evaluator, reports):
        monkeypatch.setitem(sys.modules, module.__name__, module)
    return scene, cfg, calls
