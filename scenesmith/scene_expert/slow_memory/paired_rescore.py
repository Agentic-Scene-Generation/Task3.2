"""Rescore retained raw A/B states into a new collection, without LLM calls."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

from scenesmith.scene_expert.slow_memory.paired import (
    audit_pairs,
    copy_scene_tree,
    deterministic_verdict,
    digest,
    read_json,
    tree_hashes,
    validate_pair_inputs,
    write_json,
)
from scenesmith.scene_expert.slow_memory.paired_provenance import (
    collect_pair_code_provenance,
)
from scenesmith.scene_expert.slow_memory.paired_runtime import json_value
from scenesmith.scene_expert.slow_memory.paired_scoring import (
    SCORING_PROTOCOL,
    save_scoring_proof,
    score_raw_candidate,
)


def restore_raw_scene(directory: Path, snapshot: dict[str, Any]) -> Any:
    """Restore only the persisted raw state and its retained private assets."""
    from scenesmith.agent_utils.house import RoomGeometry
    from scenesmith.agent_utils.room import RoomScene

    state = read_json(directory / "raw_state.json")
    room_dir = directory / "raw_scene" / snapshot["room_relative"]
    scene = RoomScene(
        room_geometry=RoomGeometry.from_dict(
            state["room_geometry"], scene_dir=room_dir
        ),
        scene_dir=room_dir,
        room_id=snapshot["room_id"],
        text_description=state["text_description"],
        action_log_path=room_dir / "action_log.json",
        floor_plan_mode=state["floor_plan_mode"],
        tool_schema_version=state["tool_schema_version"],
    )
    scene.restore_from_state_dict(state)
    for key, value in snapshot["scene_attributes"].items():
        setattr(scene, key, value)
    if json_value(scene.to_state_dict()) != state:
        raise ValueError("restored raw state differs; refuse offline relabeling")
    return scene


def rescore_pairs(source: Path, output: Path) -> dict[str, Any]:
    """Verify original executions, refresh labels in a copy and run the full gate."""
    from omegaconf import OmegaConf

    source, output = source.resolve(), output.resolve()
    if (
        output.exists()
        or output.is_relative_to(source)
        or source.is_relative_to(output)
    ):
        raise ValueError("rescoring requires a fresh output outside the source tree")
    # This does not authorize any old labels for export. It checks the original
    # execution, frozen context, raw assets and continuation before new scoring.
    _, groups, errors = validate_pair_inputs(source, require_fresh_physics=False)
    if errors or not 1 <= len(groups) <= 4:
        raise ValueError(
            f"source execution integrity failed: {errors or 'invalid group count'}"
        )
    output.mkdir(parents=True)
    origin = {
        "schema_version": "sceneexpert.initial_pairs_rescore.v1",
        "source_root": str(source),
        "scoring_protocol": SCORING_PROTOCOL,
        "scoring_code_provenance": collect_pair_code_provenance(),
        "model_calls": 0,
        "candidates": [],
    }
    try:
        for group_info in groups:
            old_group, group = (
                source / group_info["group"],
                output / group_info["group"],
            )
            group.mkdir()
            for name in (
                "snapshot.json",
                "status.json",
                "identity.json",
                "continuation_proof.json",
            ):
                shutil.copy2(old_group / name, group / name)
            copy_scene_tree(old_group / "input_scene", group / "input_scene")
            snapshot = read_json(group / "snapshot.json")
            for name in ("A", "B"):
                old_dir, directory = old_group / name, group / name
                directory.mkdir()
                for file_name in (
                    "raw_state.json",
                    "returned_state.json",
                    "safety.json",
                    "first_request.json",
                    "candidate_payload.json",
                ):
                    if (old_dir / file_name).exists():
                        shutil.copy2(old_dir / file_name, directory / file_name)
                copy_scene_tree(old_dir / "raw_scene", directory / "raw_scene")
                copy_scene_tree(old_dir / "slow_memory", directory / "slow_memory")
                old_result = read_json(old_dir / "result.json")
                report, proof = score_raw_candidate(
                    restore_raw_scene(directory, snapshot),
                    OmegaConf.create(snapshot["cfg"]),
                )
                if proof["raw_state_sha256"] != old_result["raw_state_hash"]:
                    raise ValueError(
                        "rescored state differs from retained raw candidate"
                    )
                files = tree_hashes(directory / "raw_scene")
                if files != old_result["raw_files"]:
                    raise ValueError("rescoring changed retained raw assets")
                verdict, score, failures = deterministic_verdict(report)
                # Keep original reports alongside revised evidence for inspection.
                write_json(
                    directory / "rescore_origin.json",
                    {
                        "result": old_result,
                        "report": read_json(old_dir / "report.json"),
                        "source_candidate": str(old_dir),
                    },
                )
                write_json(directory / "report.json", report)
                save_scoring_proof(directory, report, proof, files)
                trajectory_path = directory / "slow_memory/trajectories.jsonl"
                record = read_json(
                    trajectory_path
                )  # Validated one record per candidate.
                old_id = record["trajectory_id"]
                record["trajectory_id"] = (
                    "trajectory_" + digest([old_id, SCORING_PROTOCOL, proof])[0:24]
                )
                record["run_id"] = f"{output.parent.parent.name}/{group.name}/{name}"
                record["evidence"].update(
                    verdict=verdict,
                    quality_score=score,
                    evidence_id=record["trajectory_id"],
                )
                record["outcome"].update(
                    hard_passed=failures == 0,
                    hard_violation_count=failures,
                    deterministic_score=score,
                )
                record["provenance"]["rescoring"] = {
                    "original_trajectory_id": old_id,
                    "original_verdict": old_result["verdict"],
                    "source_candidate": str(old_dir),
                    "scoring_protocol": SCORING_PROTOCOL,
                    "scoring_code_provenance": origin["scoring_code_provenance"],
                }
                # Single-line JSONL preserves the existing loader contract.
                trajectory_path.write_text(
                    json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8"
                )
                result = {
                    **old_result,
                    "scoring_protocol": SCORING_PROTOCOL,
                    "verdict": verdict,
                    "score": score,
                    "trajectory_sha256": hashlib.sha256(
                        trajectory_path.read_bytes()
                    ).hexdigest(),
                }
                write_json(directory / "result.json", result)
                origin["candidates"].append(
                    {
                        "group": group.name,
                        "candidate": name,
                        "old_verdict": old_result["verdict"],
                        "old_score": old_result["score"],
                        "new_verdict": verdict,
                        "new_score": score,
                    }
                )
        _, _, after_errors = validate_pair_inputs(source, require_fresh_physics=False)
        if after_errors:
            raise ValueError(f"source changed during rescoring: {after_errors}")
        if collect_pair_code_provenance() != origin["scoring_code_provenance"]:
            raise ValueError("scoring source changed during offline evaluation")
        audit = audit_pairs(output, expected_groups=len(groups))
        write_json(output / "rescore_manifest.json", origin)
        write_json(
            output / "rescore_status.json",
            {"status": "completed", "gate_passed": audit["gate_passed"]},
        )
        return audit
    except Exception as exc:
        write_json(
            output / "rescore_status.json",
            {"status": "failed", "error_type": type(exc).__name__, "error": str(exc)},
        )
        raise
