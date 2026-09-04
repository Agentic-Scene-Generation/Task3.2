"""Focused checks for SceneEval single-room registry filtering."""

from __future__ import annotations

import csv
import os
import subprocess

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ANNOTATIONS = PROJECT_ROOT / "scripts" / "assets" / "annotations.csv"
RUNNER = PROJECT_ROOT / "scripts" / "run_parallel_critic_on.sh"


def _run_runner(tmp_path: Path, *args: str, annotations: Path = ANNOTATIONS):
    env = os.environ.copy()
    env.update(
        {
            "CRITIC_PROBE_ALLOW_HIGH_MEMORY_START": "1",
            "CRITIC_PROBE_PARALLEL": "false",
            "DRY_RUN": "true",
            "OUTPUT_ROOT": str(tmp_path / "output"),
            "PYTHON_BIN": str(PROJECT_ROOT / ".venv" / "bin" / "python"),
            "SCENEEVAL_ANNOTATIONS": str(annotations),
        }
    )
    return subprocess.run(
        ["bash", str(RUNNER), *args],
        cwd=PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def test_annotations_mark_only_known_multi_room_cases() -> None:
    with ANNOTATIONS.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == 500
    assert [int(row["ID"]) for row in rows] == list(range(500))
    assert {row["SceneScope"] for row in rows} == {"single_room", "multi_room"}
    assert {int(row["ID"]) for row in rows if row["SceneScope"] == "multi_room"} == {
        48,
        49,
        97,
        98,
        99,
    }


def test_runner_filters_scope_without_renumbering_batches(tmp_path: Path) -> None:
    result = _run_runner(
        tmp_path,
        "--case-set",
        "sceneeval100",
        "--difficulty",
        "hard",
        "--scenes",
        "43,45,91",
    )

    assert result.returncode == 0, result.stderr
    assert (
        "SceneEval scope: single_room only (95 supported, 5 filtered)" in result.stdout
    )
    assert "SceneEval filtered multi_room IDs: 48,49,97,98,99" in result.stdout
    assert "[critic_on/batch_044 slot=1]" in result.stdout
    assert "[critic_on/batch_046 slot=1]" in result.stdout
    assert "[critic_on/batch_092 slot=1]" in result.stdout
    assert "experiment.scene_failure_policy=record" in result.stdout
    assert "SceneEval no-VLM geometry after run: true" in result.stdout


def test_runner_rejects_invalid_scene_failure_policy(tmp_path: Path) -> None:
    env = os.environ.copy()
    env.update(
        {
            "CRITIC_PROBE_ALLOW_HIGH_MEMORY_START": "1",
            "DRY_RUN": "true",
            "OUTPUT_ROOT": str(tmp_path / "output"),
            "PYTHON_BIN": str(PROJECT_ROOT / ".venv" / "bin" / "python"),
            "SCENE_FAILURE_POLICY": "ignore",
        }
    )

    result = subprocess.run(
        ["bash", str(RUNNER), "--case-set", "new3", "--scenes", "bedroom"],
        cwd=PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 1
    assert "SCENE_FAILURE_POLICY must be strict or record" in result.stderr


def test_runner_rejects_explicit_multi_room_before_output_creation(
    tmp_path: Path,
) -> None:
    result = _run_runner(
        tmp_path,
        "--case-set",
        "sceneeval100",
        "--scenes",
        "49",
    )

    assert result.returncode == 2
    assert "scene ID '49' is marked multi_room" in result.stderr
    assert not (tmp_path / "output").exists()


def test_runner_rejects_invalid_scene_scope(tmp_path: Path) -> None:
    invalid_annotations = tmp_path / "annotations.csv"
    with ANNOTATIONS.open(newline="", encoding="utf-8-sig") as source:
        reader = csv.DictReader(source)
        rows = list(reader)
        fieldnames = reader.fieldnames
    assert fieldnames is not None
    rows[0]["SceneScope"] = "whole_house"
    with invalid_annotations.open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    result = _run_runner(
        tmp_path,
        "--case-set",
        "sceneeval100",
        "--scenes",
        "0",
        annotations=invalid_annotations,
    )

    assert result.returncode == 2
    assert "SceneEval ID 0 has invalid SceneScope 'whole_house'" in result.stderr
