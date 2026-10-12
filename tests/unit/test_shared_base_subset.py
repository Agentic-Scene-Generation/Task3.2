"""A failed base must not strand unrelated complete collection cases."""

from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from scenesmith.scene_expert.slow_memory.shared_base_subset import partition_shared_base

ROOT = Path(__file__).resolve().parents[2]


def checkpoint(root: Path, index: int, status: str = "completed") -> Path:
    scene = root / f"shared_base/batch_{index+1:03d}/hydra/scene_{index:03d}"
    scene.mkdir(parents=True)
    (scene / "scene_status.json").write_text(
        json.dumps(
            {
                "scene_id": index,
                "prompt": f"task {index}",
                "status": status,
            }
        )
    )
    (scene / "_SUCCESS").touch()
    (scene / "geometry").mkdir()
    (scene / "geometry/room.sdf").write_text(
        '<sdf version="1.7"><model name="room"/></sdf>'
    )
    (scene / "geometry/floor.gltf").write_text('{"asset":{"version":"2.0"}}')
    (scene / "house_layout.json").write_text(
        json.dumps(
            {
                "rooms": [{"id": "room"}],
                "room_geometries": {
                    "room": {
                        "sdf_path": "geometry/room.sdf",
                        "floor": {"geometry_path": "geometry/floor.gltf"},
                    }
                },
            }
        )
    )
    return scene


def assignment(root: Path, indices: list[int]) -> None:
    with (root / "assigned_cases.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["batch_id", "scene_index", "prompt", "case_id", "critic_goal"],
        )
        writer.writeheader()
        for index in indices:
            writer.writerow(
                {
                    "batch_id": f"batch_{index+1:03d}",
                    "scene_index": index,
                    "prompt": f"task {index}",
                    "case_id": str(index),
                    "critic_goal": "test",
                }
            )


def test_only_complete_cases_keep_original_indices_and_hashes(tmp_path):
    assignment(tmp_path, [127, 129, 360, 371])
    for index in [127, 129, 360, 371]:
        checkpoint(tmp_path, index, "failed" if index == 129 else "completed")
    report = partition_shared_base(tmp_path, tmp_path / "shared_base")
    assert [r["scene_index"] for r in report["eligible"]] == [127, 360, 371]
    assert [r["case_id"] for r in report["excluded"]] == ["129"]
    assert report["complete"] is False
    assert report["failed_cases_become_preferences"] is False
    assert report["eligible"][1]["batch_id"] == "batch_361"
    assert "geometry/room.sdf" in report["eligible"][0]["evidence"]["artifact_sha256"]


@pytest.mark.parametrize(
    "defect",
    ["status", "marker", "layout", "geometry", "sdf", "identity", "prompt", "external"],
)
def test_incomplete_or_mismatched_checkpoint_is_excluded(tmp_path, defect):
    assignment(tmp_path, [127])
    scene = checkpoint(tmp_path, 127)
    if defect == "status":
        (scene / "scene_status.json").write_text("{")
    elif defect == "marker":
        (scene / "_SUCCESS").unlink()
    elif defect == "layout":
        (scene / "house_layout.json").write_text('{"rooms":[]}')
    elif defect == "geometry":
        (scene / "geometry/floor.gltf").unlink()
    elif defect == "sdf":
        (scene / "geometry/room.sdf").write_text("not XML")
    elif defect in {"identity", "prompt"}:
        value = json.loads((scene / "scene_status.json").read_text())
        value["scene_id" if defect == "identity" else "prompt"] = "wrong"
        (scene / "scene_status.json").write_text(json.dumps(value))
    else:
        foreign = tmp_path / "foreign.sdf"
        foreign.write_text('<sdf version="1.7"/>')
        layout = json.loads((scene / "house_layout.json").read_text())
        layout["room_geometries"]["room"]["sdf_path"] = str(foreign)
        (scene / "house_layout.json").write_text(json.dumps(layout))
    report = partition_shared_base(tmp_path, tmp_path / "shared_base")
    assert not report["eligible"] and len(report["excluded"]) == 1


@pytest.mark.skipif(
    sys.platform == "win32" or not shutil.which("bash"), reason="native Bash runtime"
)
@pytest.mark.parametrize(
    "isolated,base_exit,branch_exit,all_failed,expected_calls,expected_exit",
    [
        (False, 1, 0, False, False, 1),
        (True, 1, 0, False, True, 1),
        (True, 0, 0, False, True, 1),
        (True, 1, 0, True, False, 1),
        (True, 0, 7, False, True, 1),
    ],
)
def test_production_driver_preserves_failure_but_runs_valid_subset(
    tmp_path,
    isolated,
    base_exit,
    branch_exit,
    all_failed,
    expected_calls,
    expected_exit,
):
    assignment(tmp_path, [127, 129, 360, 371])
    for index in [127, 129, 360, 371]:
        checkpoint(
            tmp_path, index, "failed" if all_failed or index == 129 else "completed"
        )
    source = (ROOT / "scripts/run_parallel_critic_on.sh").read_text()
    driver = source[source.index('run_exit_code=0\nif [ "$GENERATE_SHARED_BASE"') :]
    driver = driver[: driver.index("# Metrics are generated")]
    script = (
        """set -euo pipefail
GENERATE_SHARED_BASE=true
DRY_RUN=false
SCENE_SELECTION=127,129,360,371
SCENE_SELECTION_EXPLICIT=true
MAX_CASES=4
run_shared_base_with_recovery() { return "$BASE_EXIT"; }
run_batches() { printf '%s|%s\\n' "$1" "$SCENE_SELECTION" >> "$OUTPUT_ROOT/calls.txt"; return "$BRANCH_EXIT"; }
"""
        + driver
        + '\n[ "$SCENE_SELECTION" = \'127,129,360,371\' ] || exit 99\nexit "$run_exit_code"\n'
    )
    env = {
        **os.environ,
        "PYTHONPATH": str(ROOT),
        "PYTHON_BIN": sys.executable,
        "OUTPUT_ROOT": str(tmp_path),
        "SHARED_BASE_ROOT": str(tmp_path / "shared_base"),
        "SCENEEXPERT_CONTINUE_PARTIAL_SHARED_BASE": str(isolated).lower(),
        "BASE_EXIT": str(base_exit),
        "BRANCH_EXIT": str(branch_exit),
    }
    result = subprocess.run(
        ["bash", "-c", script], env=env, capture_output=True, text=True, timeout=30
    )
    assert result.returncode == expected_exit, result.stdout + result.stderr
    calls = tmp_path / "calls.txt"
    assert calls.exists() is expected_calls
    if expected_calls:
        assert calls.read_text().strip() == "critic_on|127,360,371"
