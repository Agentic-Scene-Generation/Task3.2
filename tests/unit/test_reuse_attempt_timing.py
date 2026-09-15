"""CPU regression of actual status/copy functions without importing Blender."""

from __future__ import annotations

import ast
import json
import logging
import os
import shutil

from datetime import datetime
from pathlib import Path

import pytest

from scenesmith.scene_expert.attempt_timing import scene_attempt_start
from scenesmith.scene_expert.evaluation_costs import collect_attempt_costs
from scenesmith.scene_expert.trace_logger import TraceLogger


@pytest.fixture
def native_functions():
    """Execute unmodified, dependency-light native functions from their AST."""
    source = (
        Path(__file__).resolve().parents[2]
        / "scenesmith/experiments/indoor_scene_generation.py"
    )
    tree = ast.parse(source.read_text(encoding="utf-8"))
    selected = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name in {"_write_scene_status", "_copy_checkpoint_for_stage"}
    ]
    namespace = {
        "Path": Path,
        "json": json,
        "os": os,
        "datetime": datetime,
        "shutil": shutil,
        "console_logger": logging.getLogger(__name__),
        "_SCENE_STATUS_FILENAME": "scene_status.json",
        "_SCENE_STATUS_SCHEMA_VERSION": "test",
        "_resolve_furniture_render_resume_mode": lambda *a, **kw: None,
        "STAGE_CHECKPOINTS": {"furniture": "scene_after_floor_plan"},
        "STAGE_ASSET_DIRS": {"furniture": []},
    }
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            *selected,
        ],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(source), "exec"), namespace)
    return namespace


@pytest.mark.parametrize("reuse", [False, True])
@pytest.mark.parametrize(
    "terminal", ["completed", "completed_with_quality_issues", "failed"]
)
def test_start_survives_native_reuse_and_terminal_status(
    tmp_path, native_functions, reuse, terminal
):
    start = "2026-09-08T00:00:00+00:00"
    output = tmp_path / "run"
    scene = output / "scene_000"
    status = native_functions["_write_scene_status"]
    kwargs = dict(
        output_dir=output, scene_id=0, prompt="Office", attempt=1, run_id="test-run"
    )
    status(**kwargs, status="running", started_at=start)
    if reuse:
        source = tmp_path / "shared_base/scene_000"
        (source / "room_geometry").mkdir(parents=True)
        (source / "floor_plans").mkdir()
        (source / "house_layout.json").write_text("{}")
        # The native function recursively replaces only this validated fixture.
        assert scene.resolve().is_relative_to(tmp_path.resolve())
        native_functions["_copy_checkpoint_for_stage"](source, scene, "furniture")
        assert not (scene / "scene_status.json").exists()
    logger = TraceLogger(
        str(output), scene_index=0, prompt="Office", scene_started_at=start
    )
    trace = json.loads(logger.save_partial().read_text(encoding="utf-8"))
    assert trace["runtime_identity"]["scene_started_at"] == start
    status(**kwargs, status=terminal)
    payload = json.loads((scene / "scene_status.json").read_text())
    assert payload["started_at"] == start
    cost = collect_attempt_costs(scene, 0)
    assert cost["all_attempt_cost_complete"]
    assert cost["attempts"][0]["start_time_source"] == "scene_status"
    assert cost["all_attempt_time_sec"] > 0


def test_crash_before_hooks_still_has_cost(tmp_path, native_functions):
    kwargs = dict(
        output_dir=tmp_path, scene_id=0, prompt="Office", attempt=1, run_id="test-run"
    )
    native_functions["_write_scene_status"](
        **kwargs, status="running", started_at="2026-09-08T00:00:00Z"
    )
    (tmp_path / "scene_000/scene_status.json").unlink()
    native_functions["_write_scene_status"](
        **kwargs, status="failed", error="Checkpoint copy failed"
    )
    result = collect_attempt_costs(tmp_path / "scene_000", 0)
    assert result["all_attempt_cost_complete"]
    assert result["final_attempt_trace_time_sec"] is None


def test_explicit_trace_start_does_not_use_stale_status(tmp_path):
    scene = tmp_path / "scene_000"
    scene.mkdir()
    (scene / "scene_status.json").write_text(
        json.dumps({"status": "running", "updated_at": "2020-01-01T00:00:00Z"})
    )
    trace = TraceLogger(
        str(tmp_path),
        prompt="Office",
        scene_index=0,
        scene_started_at="2026-09-08T00:00:00Z",
    )
    assert (
        json.loads(trace.save_partial().read_text())["runtime_identity"][
            "scene_started_at"
        ]
        == "2026-09-08T00:00:00Z"
    )


@pytest.mark.parametrize(
    "change", [{"run_id": "other"}, {"prompt": "Other task"}, {"attempt": 2}]
)
def test_unrelated_attempt_start_is_never_reused(tmp_path, change):
    kwargs = dict(
        output_dir=tmp_path, scene_id=0, attempt=1, prompt="Office", run_id="test"
    )
    scene_attempt_start(**kwargs, started_at="2026-09-08T00:00:00Z")
    assert scene_attempt_start(**(kwargs | change)) == ""


def test_missing_or_naive_start_does_not_become_zero_cost(tmp_path):
    kwargs = dict(
        output_dir=tmp_path, scene_id=0, attempt=1, prompt="Office", run_id="test"
    )
    assert scene_attempt_start(**kwargs, started_at="2026-09-08T00:00:00") == ""
    assert scene_attempt_start(**kwargs) == ""


def test_worker_passes_same_start_to_status_and_hooks():
    root = Path(__file__).resolve().parents[2]
    tree = ast.parse(
        (root / "scenesmith/experiments/indoor_scene_generation.py").read_text(
            encoding="utf-8"
        )
    )
    for function, keyword in [
        ("_write_scene_status", "started_at"),
        ("build_hook_runner", "scene_started_at"),
    ]:
        calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == function
        ]
        assert any(
            any(
                kw.arg == keyword
                and isinstance(kw.value, ast.Name)
                and kw.value.id == "scene_started_at"
                for kw in call.keywords
            )
            for call in calls
        )


@pytest.mark.parametrize("bad_start", ["2026-09-08T00:00:30Z", "invalid"])
def test_conflicting_or_malformed_status_start_blocks_cost_claim(tmp_path, bad_start):
    from tests.unit.test_memory_evaluation import cost_attempt, write

    current = tmp_path / "scene_000"
    cost_attempt(current, 1, "2026-09-08T00:00:00Z", "2026-09-08T00:01:00Z", 50)
    status = json.loads((current / "scene_status.json").read_text())
    write(current / "scene_status.json", status | {"started_at": bad_start})
    result = collect_attempt_costs(current, 0)
    assert not result["all_attempt_cost_complete"]
    assert result["all_attempt_time_sec"] is None


def test_status_only_retry_cost_includes_failure_and_retry_wait(tmp_path):
    from tests.unit.test_memory_evaluation import write

    old = tmp_path / "failed_attempts/scene_000_attempt_01_stamp"
    current = tmp_path / "scene_000"
    write(
        old / "scene_status.json",
        {
            "attempt": 1,
            "status": "failed",
            "started_at": "2026-09-08T00:00:00Z",
            "updated_at": "2026-09-08T00:01:00Z",
        },
    )
    write(
        current / "scene_status.json",
        {
            "attempt": 2,
            "status": "completed",
            "started_at": "2026-09-08T00:01:10Z",
            "updated_at": "2026-09-08T00:02:00Z",
        },
    )
    result = collect_attempt_costs(current, 0)
    assert result["all_attempt_cost_complete"]
    assert result["all_attempt_time_sec"] == 120
    assert result["attempt_service_time_sec"] == 110
    assert result["all_attempt_trace_time_sec"] is None
