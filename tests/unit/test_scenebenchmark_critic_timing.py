"""Timing audit coverage for the embedded SceneBenchmark critic API."""

import json

from types import SimpleNamespace

from scenesmith.scenebenchmark_critic import api


def test_evaluate_room_scene_writes_timing_record(tmp_path, monkeypatch) -> None:
    scene = SimpleNamespace(scene_dir=tmp_path, room_id="bedroom_01")
    config = SimpleNamespace(metrics=("functional_dependency",))
    case_pack = {"checks": [{"check_id": "check_1"}]}

    monkeypatch.setattr(api, "_coerce_config", lambda _config: config)
    monkeypatch.setattr(
        api,
        "room_scene_to_case_pack",
        lambda _scene, *, stage, metrics: case_pack,
    )
    monkeypatch.setattr(api, "constraint_mode", lambda _config: "contract")

    def _run_checks(_case_pack, *, config, timing):
        timing["run_case_pack_checks_sec"] = 0.001
        return [{"check_id": "check_1"}]

    monkeypatch.setattr(api, "run_case_pack_checks", _run_checks)
    monkeypatch.setattr(
        api,
        "build_evaluation_payload",
        lambda **kwargs: {"results": kwargs["results"]},
    )

    payload = api.evaluate_room_scene(scene, config=config, stage="furniture")

    assert payload["results"] == [{"check_id": "check_1"}]
    timing_path = (
        tmp_path / "scene_expert" / "timing" / "scenebenchmark_critic_timing.jsonl"
    )
    record = json.loads(timing_path.read_text(encoding="utf-8").splitlines()[0])
    assert record["schema_version"] == "scenesmith.scenebenchmark_critic.timing.v1"
    assert record["status"] == "ok"
    assert record["scope"] == "room:bedroom_01"
    assert record["stage"] == "furniture"
    assert record["details"]["case_pack_check_count"] == 1
    assert "case_pack_build_sec" in record["steps"]
    assert "check_execution_wall_sec" in record["steps"]
