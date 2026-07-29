import json
import sqlite3
from datetime import datetime
from pathlib import Path
from time import sleep

from PIL import Image

from tools.critic_probe_web import _command_runs_this_script, create_app


def write(path: Path, contents: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(contents, encoding="utf-8")


def test_identifies_only_the_same_web_server_script() -> None:
    project_root = Path(__file__).resolve().parents[1]

    assert _command_runs_this_script(
        [".venv/bin/python", "tools/critic_probe_web.py"], project_root
    )
    assert not _command_runs_this_script(
        [".venv/bin/python", "tools/another_service.py"], project_root
    )


def test_indexes_scene_and_rejects_paths_outside_probe_root(tmp_path: Path) -> None:
    room = (
        tmp_path
        / "run_a"
        / "critic_on"
        / "batch_001"
        / "hydra"
        / "scene_000"
        / "room_bedroom"
    )
    write(room / "action_log.json", "[]")
    write(room / "timing_stats.jsonl", '{"stage":"furniture","event":"critic"}\n')
    write(room / "scene_states" / "final_scene" / "scene_state.json", "{}")
    app = create_app(tmp_path)
    client = app.test_client()

    runs = client.get("/api/runs").get_json()["runs"]
    scenes = client.get("/api/runs/run_a/scenes").get_json()["scenes"]

    assert runs[0]["id"] == "run_a"
    assert scenes[0]["room"] == "bedroom"
    assert scenes[0]["status"] == "complete"
    assert client.get("/api/image?path=../../etc/passwd").status_code == 404


def test_compares_scene_object_changes(tmp_path: Path) -> None:
    room = (
        tmp_path
        / "run_a"
        / "critic_on"
        / "batch_001"
        / "hydra"
        / "scene_000"
        / "room_bedroom"
    )
    write(room / "action_log.json", "[]")
    before = room / "scene_renders" / "furniture" / "renders_001" / "scene_state.json"
    after = room / "scene_renders" / "furniture" / "renders_002" / "scene_state.json"
    write(
        before,
        '{"objects":[{"object_id":"chair_0","transform":{"translation":[0,0,0]}}]}',
    )
    write(
        after,
        '{"objects":[{"object_id":"chair_0","transform":{"translation":[1,0,0]}},{"object_id":"desk_0"}]}',
    )
    app = create_app(tmp_path)
    client = app.test_client()
    response = client.get(
        "/api/diff",
        query_string={
            "before": str(before.relative_to(tmp_path)),
            "after": str(after.relative_to(tmp_path)),
        },
    )

    payload = response.get_json()
    assert response.status_code == 200
    assert payload["changed"][0]["object_id"] == "chair_0"
    assert payload["added"][0]["object_id"] == "desk_0"


def test_returns_newest_render_first(tmp_path: Path) -> None:
    room = (
        tmp_path
        / "run_a"
        / "critic_on"
        / "batch_001"
        / "hydra"
        / "scene_000"
        / "room_bedroom"
    )
    write(room / "action_log.json", "[]")
    first = room / "scene_renders" / "furniture" / "renders_001" / "scene_state.json"
    second = room / "scene_renders" / "furniture" / "renders_002" / "scene_state.json"
    write(first, "{}")
    sleep(0.01)
    write(second, "{}")
    app = create_app(tmp_path)

    payload = (
        app.test_client()
        .get("/api/scene", query_string={"path": str(room.relative_to(tmp_path))})
        .get_json()
    )

    assert payload["renders"][0]["id"].endswith("renders_002")


def test_normalizes_naive_action_timestamps_to_utc(tmp_path: Path) -> None:
    room = (
        tmp_path
        / "run_a"
        / "critic_on"
        / "batch_001"
        / "hydra"
        / "scene_000"
        / "room_bedroom"
    )
    write(
        room / "action_log.json",
        json.dumps(
            [
                {
                    "step_number": 1,
                    "timestamp": "2026-07-28T23:54:43.157472",
                    "tool_name": "_add_furniture_to_scene_impl",
                    "arguments": {"asset_id": "bed_0"},
                }
            ]
        ),
    )
    payload = (
        create_app(tmp_path)
        .test_client()
        .get("/api/scene", query_string={"path": str(room.relative_to(tmp_path))})
        .get_json()
    )

    timestamp = payload["actions"][0]["timestamp"]
    assert datetime.fromisoformat(timestamp).tzinfo is not None
    assert timestamp.endswith("+00:00")


def test_exposes_full_llm_audit_from_scene_trace(tmp_path: Path) -> None:
    room = (
        tmp_path
        / "run_a"
        / "critic_on"
        / "batch_001"
        / "hydra"
        / "scene_000"
        / "room_bedroom"
    )
    write(room / "action_log.json", "[]")
    write(
        room / "timing_stats.jsonl",
        json.dumps(
            {
                "created_at": "2026-07-28T12:00:08Z",
                "stage": "furniture",
                "module": "critic",
                "event": "review_layout",
                "elapsed_sec": 8.0,
            }
        )
        + "\n",
    )
    write(
        room.parent / "scene_expert" / "timing" / "llm_calls.jsonl",
        json.dumps(
            {
                "created_at": "2026-07-28T12:00:08Z",
                "stage": "furniture",
                "agent_role": "critic",
                "event": "review_layout",
                "prompt_chars": 24,
                "prompt_excerpt": "truncated prompt",
                "output_chars": 20,
                "output_excerpt": "truncated output",
                "token_usage": {"total_tokens": 44},
            }
        )
        + "\n",
    )
    write(
        room / "scene_expert" / "timing" / "scenebenchmark_critic_timing.jsonl",
        json.dumps(
            {
                "created_at": "2026-07-28T12:00:09Z",
                "stage": "furniture_relation_repair",
                "status": "ok",
                "elapsed_sec": 0.1,
                "steps": {"rule_evaluator_sec": 0.08},
                "details": {"result_count": 3},
            }
        )
        + "\n",
    )
    database = room / "critic.db"
    connection = sqlite3.connect(database)
    connection.execute(
        "CREATE TABLE agent_messages (id INTEGER PRIMARY KEY, message_data TEXT, created_at TEXT)"
    )
    messages = [
        (
            1,
            {"role": "user", "content": "FULL PROMPT: inspect all clearance paths"},
            "2026-07-28 12:00:01",
        ),
        (
            2,
            {
                "type": "reasoning",
                "summary": [{"text": "I should inspect the bed first."}],
            },
            "2026-07-28 12:00:03",
        ),
        (
            3,
            {
                "type": "function_call",
                "call_id": "call_1",
                "name": "check_physics",
                "arguments": "{}",
            },
            "2026-07-28 12:00:04",
        ),
        (
            4,
            {
                "type": "function_call_output",
                "call_id": "call_1",
                "output": [
                    {"image_url": "data:image/png;base64,abcdef"},
                    {"status": "clear"},
                ],
            },
            "2026-07-28 12:00:05",
        ),
        (
            5,
            {
                "type": "message",
                "content": [{"text": "FULL RESPONSE: layout is clear."}],
            },
            "2026-07-28 12:00:07",
        ),
    ]
    connection.executemany(
        "INSERT INTO agent_messages (id, message_data, created_at) VALUES (?, ?, ?)",
        [
            (message_id, json.dumps(payload), timestamp)
            for message_id, payload, timestamp in messages
        ],
    )
    connection.commit()
    connection.close()

    app = create_app(tmp_path)
    client = app.test_client()
    path = str(room.relative_to(tmp_path))
    scene = client.get("/api/scene", query_string={"path": path}).get_json()
    llm_event = next(event for event in scene["audit_events"] if event["kind"] == "llm")
    benchmark_event = next(
        event for event in scene["audit_events"] if event["kind"] == "benchmark"
    )
    response = client.get(
        "/api/audit-event", query_string={"path": path, "event_id": llm_event["id"]}
    )
    payload = response.get_json()

    assert response.status_code == 200
    assert llm_event["actor"] == "critic"
    assert llm_event["started_at"] == "2026-07-28T12:00:00+00:00"
    assert benchmark_event["title"] == "SceneBenchmark critic"
    assert payload["provenance"] == "sqlite_session_trace"
    assert payload["input"] == "FULL PROMPT: inspect all clearance paths"
    assert payload["output"] == "FULL RESPONSE: layout is clear."
    assert payload["messages"][0]["agent"] == "critic"
    assert payload["messages"][0]["direction"] == "input"
    assert payload["tool_calls"][0]["name"] == "check_physics"
    assert payload["tool_calls"][0]["output"][0]["image_url"]["kind"] == "image_payload"


def test_exposes_planner_orchestration_and_full_session_trace(tmp_path: Path) -> None:
    room = (
        tmp_path
        / "run_a"
        / "critic_on"
        / "batch_001"
        / "hydra"
        / "scene_000"
        / "room_bedroom"
    )
    write(room / "action_log.json", "[]")
    write(
        room / "timing_stats.jsonl",
        json.dumps(
            {
                "created_at": "2026-07-28T12:00:10Z",
                "stage": "furniture",
                "module": "planner",
                "event": "coordinate_stage",
                "elapsed_sec": 10.0,
            }
        )
        + "\n",
    )
    scene_root = room.parent
    write(
        scene_root / "scene_expert" / "timing" / "planner_orchestration.jsonl",
        "\n".join(
            [
                json.dumps(
                    {
                        "created_at": "2026-07-28T12:00:01Z",
                        "stage": "furniture",
                        "actor": "planner",
                        "call_id": "furniture:request_initial_design:001",
                        "phase": "dispatch",
                        "operation": "request_initial_design",
                        "child_agent": "designer",
                        "status": "started",
                    }
                ),
                json.dumps(
                    {
                        "created_at": "2026-07-28T12:00:08Z",
                        "stage": "furniture",
                        "actor": "planner",
                        "call_id": "furniture:request_initial_design:001",
                        "phase": "resume",
                        "operation": "request_initial_design",
                        "child_agent": "designer",
                        "status": "completed",
                    }
                ),
            ]
        )
        + "\n",
    )
    payload_ref = "scene_expert/audit/llm_payloads/planner.json"
    write(
        scene_root / "scene_expert" / "timing" / "llm_calls.jsonl",
        json.dumps(
            {
                "created_at": "2026-07-28T12:00:10Z",
                "stage": "furniture",
                "agent_role": "planner",
                "event": "coordinate_stage",
                "payload_ref": payload_ref,
                "token_usage": {"total_tokens": 123},
            }
        )
        + "\n",
    )
    write(
        scene_root / payload_ref,
        json.dumps(
            {
                "prompt": {
                    "instructions": "Coordinate designer and critic.",
                    "runner_input": "Begin the furniture workflow.",
                },
                "output": "Furniture workflow complete.",
            }
        ),
    )
    database = room / "planner.db"
    connection = sqlite3.connect(database)
    connection.execute(
        "CREATE TABLE agent_messages (id INTEGER PRIMARY KEY, message_data TEXT, created_at TEXT)"
    )
    messages = [
        (
            1,
            {"role": "user", "content": "Begin the furniture workflow."},
            "2026-07-28 12:00:00",
        ),
        (
            2,
            {
                "type": "function_call",
                "call_id": "planner_call_1",
                "name": "request_initial_design",
                "arguments": "{}",
            },
            "2026-07-28 12:00:01",
        ),
        (
            3,
            {
                "type": "function_call_output",
                "call_id": "planner_call_1",
                "output": "Designer completed the room.",
            },
            "2026-07-28 12:00:08",
        ),
        (
            4,
            {"type": "message", "content": "Furniture workflow complete."},
            "2026-07-28 12:00:09",
        ),
    ]
    connection.executemany(
        "INSERT INTO agent_messages (id, message_data, created_at) VALUES (?, ?, ?)",
        [
            (message_id, json.dumps(message), created_at)
            for message_id, message, created_at in messages
        ],
    )
    connection.commit()
    connection.close()

    client = create_app(tmp_path).test_client()
    path = str(room.relative_to(tmp_path))
    scene = client.get("/api/scene", query_string={"path": path}).get_json()
    orchestration = [
        event for event in scene["audit_events"] if event["kind"] == "orchestration"
    ]
    planner_event = next(
        event
        for event in scene["audit_events"]
        if event["kind"] == "llm" and event["actor"] == "planner"
    )
    detail = client.get(
        "/api/audit-event",
        query_string={"path": path, "event_id": planner_event["id"]},
    ).get_json()

    assert [event["title"] for event in orchestration] == [
        "Delegate to designer",
        "Resume from designer",
    ]
    assert orchestration[1]["elapsed_sec"] == 7.0
    assert detail["provenance"] == "full_payload_file"
    assert detail["input"]["runner_input"] == "Begin the furniture workflow."
    assert detail["tool_calls"][0]["name"] == "request_initial_design"
    assert detail["tool_calls"][0]["output"] == "Designer completed the room."
    assert detail["session_databases"] == ["planner.db"]
    assert all(message["agent"] == "planner" for message in detail["messages"])


def test_exposes_hssd_retrieval_and_vlm_trace_for_add_furniture(tmp_path: Path) -> None:
    room = (
        tmp_path
        / "run_a"
        / "critic_on"
        / "batch_001"
        / "hydra"
        / "scene_000"
        / "room_bedroom"
    )
    write(
        room / "action_log.json",
        json.dumps(
            [
                {
                    "step_number": 1,
                    "timestamp": "2026-07-28T12:00:08Z",
                    "tool_name": "_add_furniture_to_scene_impl",
                    "arguments": {"asset_id": "chair_0"},
                }
            ]
        ),
    )
    write(
        room / "generated_assets" / "furniture" / "asset_registry.json",
        json.dumps(
            {
                "chair_0": {
                    "object_id": "chair_0",
                    "name": "chair",
                    "description": "oak dining chair",
                    "metadata": {
                        "asset_source": "hssd",
                        "asset_short_name": "chair",
                        "hssd_mesh_id": "candidate-b",
                    },
                }
            }
        ),
    )
    write(
        room.parents[1] / "asset_choice_audit.jsonl",
        json.dumps(
            {
                "schema_version": "hssd_rendered_choice_audit.v2",
                "object_description": "oak dining chair",
                "object_short_name": "chair",
                "requested_dimensions": [0.5, 0.5, 0.9],
                "candidates": [
                    {"hssd_id": "candidate-a", "similarity_score": 0.83},
                    {"hssd_id": "candidate-b", "similarity_score": 0.81},
                ],
                "status": "selected",
                "selected_hssd_id": "candidate-b",
                "selected_index": 2,
                "reason": "matches the requested wood dining style",
                "raw_response": '{"selected_index": 2}',
            }
        )
        + "\n",
    )

    payload = (
        create_app(tmp_path)
        .test_client()
        .get(
            "/api/audit-event",
            query_string={
                "path": str(room.relative_to(tmp_path)),
                "event_id": "tool-action:1",
            },
        )
        .get_json()
    )

    assert payload["selection_trace"]["status"] == "recorded"
    assert payload["selection_trace"]["retrieval"]["backend"] == "embedding"
    assert (
        payload["selection_trace"]["retrieval"]["candidates"][1]["hssd_id"]
        == "candidate-b"
    )
    assert payload["selection_trace"]["vlm_selection"]["selected_index"] == 2


def test_serves_cached_hssd_asset_thumbnail_only(tmp_path: Path) -> None:
    asset_path = tmp_path / "hssd_rendered_assets" / ("a" * 40) / "iso.png"
    asset_path.parent.mkdir(parents=True)
    Image.new("RGB", (640, 400), color=(30, 100, 70)).save(asset_path)
    client = create_app(tmp_path).test_client()

    response = client.get(
        "/api/asset-thumbnail",
        query_string={"path": str(asset_path), "width": 180},
    )

    assert response.status_code == 200
    assert response.content_type == "image/jpeg"
    assert "max-age=86400" in response.headers["Cache-Control"]
    assert (
        client.get(
            "/api/asset-thumbnail", query_string={"path": "/etc/passwd"}
        ).status_code
        == 404
    )
