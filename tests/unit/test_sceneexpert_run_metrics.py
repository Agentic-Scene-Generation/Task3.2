from __future__ import annotations

import csv
import json

from pathlib import Path

from scenesmith.scene_expert.run_metrics import collect_run_metrics, write_run_metrics


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_manifest(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=("scene_index", "prompt", "case_id", "critic_goal"),
        )
        writer.writeheader()
        writer.writerow(
            {
                "scene_index": 0,
                "prompt": "bedroom",
                "case_id": "bedroom",
                "critic_goal": "quality",
            }
        )
        writer.writerow(
            {
                "scene_index": 1,
                "prompt": "living room",
                "case_id": "long_living_room",
                "critic_goal": "quality",
            }
        )


def test_metrics_survive_partial_failure_and_attribute_repairs(tmp_path) -> None:
    output_root = tmp_path / "run_001"
    batch = output_root / "critic_on" / "batch_001"
    hydra = batch / "hydra"
    _write_manifest(batch / "batch_cases.csv")

    scene0 = hydra / "scene_000"
    _write_json(
        scene0 / "scene_status.json",
        {"status": "completed", "attempt": 1, "prompt": "bedroom"},
    )
    _write_json(
        hydra / "traces" / "trace_000000.json",
        {
            "status": "completed",
            "total_time_sec": 120.0,
            "component_flags": {
                "fast_memory_retrieval": True,
                "memory_writer": True,
            },
            "component_status": {
                "memory_writer": {
                    "write_status": "promoted",
                    "promoted_count": 1,
                    "fallback_written": False,
                    "store_apply": {"added": 1, "merged": 0},
                }
            },
            "stages": [
                {
                    "stage": "furniture",
                    "memory_pack": {
                        "success_case_ids": ["success-1"],
                        "failure_case_ids": [],
                        "skill_names": [],
                    },
                    "execution_evidence": {
                        "injected_memory_hash": "abc",
                        "designer_prompt_contains_memory": True,
                    },
                    "repair_actions": [
                        {
                            "repair_owner": "scene_expert_repair_controller",
                            "repair_type": "local_repair",
                            "failure_type": "missing_object",
                            "repair_action": "Add the missing object.",
                            "execution_status": "planned",
                        }
                    ],
                },
                {
                    "stage": "furniture",
                    "memory_pack": {
                        "success_case_ids": ["success-1"],
                        "failure_case_ids": [],
                        "skill_names": [],
                    },
                    "execution_evidence": {
                        "injected_memory_hash": "abc",
                        "designer_prompt_contains_memory": True,
                    },
                    "repair_actions": [
                        {
                            "repair_owner": "scene_expert_repair_controller",
                            "repair_type": "local_repair",
                            "failure_type": "missing_object",
                            "repair_action": "Add the missing object.",
                            "execution_status": "executed",
                        }
                    ],
                },
            ],
            "final_report": {"overall_score": 0.8, "pass_scene": True},
        },
    )
    _write_json(
        scene0
        / "room_main"
        / "scenebenchmark_critic"
        / "final_scene"
        / "scenebenchmark_critic.json",
        {
            "summary": {
                "scene_summary": {
                    "effective_checks": 4,
                    "pass": 4,
                    "degraded": 0,
                    "fail": 0,
                    "unknown": 0,
                    "score": 1.0,
                }
            }
        },
    )

    scene1 = hydra / "scene_001"
    _write_json(
        scene1 / "scene_status.json",
        {
            "status": "failed",
            "attempt": 1,
            "prompt": "living room",
            "error": "physics hard violation: collisions",
        },
    )
    _write_json(
        scene1 / "scene_expert" / "trace" / "trace_000001_partial.json",
        {
            "status": "failed",
            "error": "physics hard violation: collisions",
            "total_time_sec": 45.0,
            "component_flags": {"fast_memory_retrieval": True},
            "stages": [],
        },
    )
    _write_json(
        scene1 / "scene_expert" / "stages" / "000_furniture_pre.json",
        {
            "stage": "furniture",
            "memory_pack": {
                "success_case_ids": [],
                "failure_case_ids": [],
                "skill_names": [],
            },
            "execution_evidence": {},
        },
    )
    timing_dir = scene1 / "scene_expert" / "timing"
    timing_dir.mkdir(parents=True, exist_ok=True)
    (timing_dir / "memory_retrieval.jsonl").write_text(
        json.dumps(
            {
                "retriever_type": "hybrid",
                "stage": "furniture",
                "total_sec": 0.5,
                "zero_result_reason": "no_structurally_compatible_memory",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (timing_dir / "repair_events.jsonl").write_text(
        json.dumps({"schema_version": "scenesmith.repair_event.v1"}) + "\n",
        encoding="utf-8",
    )

    metrics = collect_run_metrics(output_root, process_exit_code=1)

    assert metrics["quality_comparison_ready"] is False
    assert metrics["summary"]["expected_scenes"] == 2
    assert metrics["summary"]["completed_scenes"] == 1
    assert metrics["summary"]["failed_scenes"] == 1
    assert metrics["summary"]["critic_mean_score"] == 1.0
    assert metrics["summary"]["memory_injection_delivery_rate"] == 1.0
    assert metrics["summary"]["memory_writer_fallback_writes"] == 0
    assert metrics["summary"]["scenesmith_repair_events"] == 1
    assert metrics["summary"]["sceneexpert_repair_plans"] == 1
    assert metrics["summary"]["sceneexpert_repairs_executed"] == 1
    assert metrics["summary"]["memory_zero_result_reasons"] == [
        "no_structurally_compatible_memory"
    ]

    paths = write_run_metrics(metrics)
    assert all(Path(path).is_file() for path in paths.values())
    rows = list(
        csv.DictReader(Path(paths["scene_csv"]).open("r", encoding="utf-8"))
    )
    assert [row["status"] for row in rows] == ["completed", "failed"]


def test_malformed_jsonl_is_a_warning_not_a_metrics_failure(tmp_path) -> None:
    output_root = tmp_path / "run_002"
    batch = output_root / "critic_on" / "batch_001"
    _write_manifest(batch / "batch_cases.csv")
    timing = (
        batch
        / "hydra"
        / "scene_000"
        / "scene_expert"
        / "timing"
        / "memory_retrieval.jsonl"
    )
    timing.parent.mkdir(parents=True, exist_ok=True)
    timing.write_text("{not-json}\n", encoding="utf-8")

    metrics = collect_run_metrics(output_root)

    assert any(
        warning.startswith("malformed_jsonl:")
        for warning in metrics["data_quality_warnings"]
    )
