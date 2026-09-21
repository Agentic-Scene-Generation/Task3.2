"""Rescore retained raw A/B states into a new collection, without LLM calls."""

from __future__ import annotations

import hashlib
import json
import logging
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
from scenesmith.scene_expert.slow_memory.paired_restoration import (
    relocate_raw_assets,
    save_restoration_proof,
)
from scenesmith.scene_expert.slow_memory.paired_runtime import json_value
from scenesmith.scene_expert.slow_memory.paired_scoring import (
    SCORING_PROTOCOL,
    save_scoring_proof,
    score_raw_candidate,
)

LOGGER = logging.getLogger(__name__)


def restore_raw_scene(
    directory: Path, snapshot: dict[str, Any], *, source_candidate: Path
) -> Any:
    """Restore only the persisted raw state and its retained private assets."""
    from scenesmith.agent_utils.house import RoomGeometry
    from scenesmith.agent_utils.room import RoomScene

    state = read_json(directory / "raw_state.json")
    mapped, bindings, unavailable_images = relocate_raw_assets(
        state,
        directory,
        snapshot,
        source_candidate,
        tree_hashes(directory / "raw_scene"),
    )
    room_dir = directory / "raw_scene" / snapshot["room_relative"]
    scene = RoomScene(
        room_geometry=RoomGeometry.from_dict(
            mapped["room_geometry"], scene_dir=room_dir
        ),
        scene_dir=room_dir,
        room_id=snapshot["room_id"],
        text_description=state["text_description"],
        action_log_path=room_dir / "action_log.json",
        floor_plan_mode=state["floor_plan_mode"],
        tool_schema_version=state["tool_schema_version"],
    )
    scene.restore_from_state_dict(mapped)
    for key, value in snapshot["scene_attributes"].items():
        setattr(scene, key, value)
    save_restoration_proof(
        directory,
        state,
        mapped,
        json_value(scene.to_state_dict()),
        bindings,
        unavailable_images,
    )
    return scene


def _select_valid_groups(
    groups: list[dict[str, Any]], group_names: list[str] | None
) -> list[dict[str, Any]]:
    available = {row["group"] for row in groups}
    requested = available if group_names is None else set(group_names)
    if (
        not requested
        or not requested <= available
        or (group_names is not None and len(requested) != len(group_names))
    ):
        raise ValueError("rescore groups must be unique existing source groups")
    selected = [row for row in groups if row["group"] in requested]
    if not 1 <= len(selected) <= 4 or any(not row["valid"] for row in selected):
        raise ValueError(
            f"source execution integrity failed for selected groups: {selected}"
        )
    return selected


def rescore_pairs(
    source: Path, output: Path, *, group_names: list[str] | None = None
) -> dict[str, Any]:
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
    _, source_groups, errors = validate_pair_inputs(source, require_fresh_physics=False)
    groups = _select_valid_groups(source_groups, group_names)
    output.mkdir(parents=True)
    origin = {
        "schema_version": "sceneexpert.initial_pairs_rescore.v1",
        "source_root": str(source),
        "scoring_protocol": SCORING_PROTOCOL,
        "scoring_code_provenance": collect_pair_code_provenance(),
        "model_calls": 0,
        "selected_groups": [row["group"] for row in groups],
        "excluded_groups": [row for row in source_groups if row not in groups],
        "source_audit_errors": errors,
        "candidates": [],
    }
    phase, active_candidate = "prepare", ""
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
                active_candidate = f"{group.name}/{name}"
                LOGGER.info("Rescoring retained candidate %s", active_candidate)
                phase = "copy_candidate"
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
                phase = "restore_raw_state"
                scene = restore_raw_scene(directory, snapshot, source_candidate=old_dir)
                restoration = read_json(directory / "restoration_proof.json")
                LOGGER.info(
                    "Verified restoration of %s: %d rotation roundoffs, %d asset relocations",
                    active_candidate,
                    len(restoration["rotation_roundoff"]),
                    len(restoration["asset_bindings"]),
                )
                phase = "score_raw_state"
                report, proof = score_raw_candidate(
                    scene, OmegaConf.create(snapshot["cfg"])
                )
                if (
                    restoration["source_raw_state_sha256"]
                    != old_result["raw_state_hash"]
                    or proof["raw_state_sha256"] != restoration["restored_state_sha256"]
                ):
                    raise ValueError(
                        "rescored state differs from retained raw candidate"
                    )
                proof["raw_state_sha256"] = old_result["raw_state_hash"]
                proof["restoration_proof_sha256"] = digest(restoration)
                files = tree_hashes(directory / "raw_scene")
                if files != old_result["raw_files"]:
                    raise ValueError("rescoring changed retained raw assets")
                verdict, score, failures = deterministic_verdict(report)
                LOGGER.info(
                    "Fresh result for %s: verdict=%s score=%.6f failures=%d",
                    active_candidate,
                    verdict,
                    score,
                    failures,
                )
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
                record["evidence"]["details"].update(
                    evaluation_state_sha256=proof["evaluation_state_sha256"],
                    restoration_proof_sha256=proof["restoration_proof_sha256"],
                )
                from scenesmith.scene_expert.slow_memory.relative import report_profile

                record["evidence"]["details"]["relative_profile"] = report_profile(
                    report
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
                    "restoration_proof_sha256": proof["restoration_proof_sha256"],
                    "scoring_code_provenance": origin["scoring_code_provenance"],
                }
                # Single-line JSONL preserves the existing loader contract.
                trajectory_path.write_text(
                    json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8"
                )
                result = {
                    **old_result,
                    "scoring_protocol": SCORING_PROTOCOL,
                    "evaluation_state_hash": proof["evaluation_state_sha256"],
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
        phase = "final_audit"
        _, after_groups, _ = validate_pair_inputs(source, require_fresh_physics=False)
        _select_valid_groups(after_groups, [row["group"] for row in groups])
        if collect_pair_code_provenance() != origin["scoring_code_provenance"]:
            raise ValueError("scoring source changed during offline evaluation")
        audit = audit_pairs(output, expected_groups=len(groups))
        write_json(output / "rescore_manifest.json", origin)
        write_json(
            output / "rescore_status.json",
            {
                "status": (
                    "completed" if audit["execution_integrity_passed"] else "failed"
                ),
                "outcome": audit["status"],
                "operation_succeeded": audit["execution_integrity_passed"],
                "preference_gate_passed": audit["preference_gate_passed"],
                "candidate_count": audit["candidate_count"],
                "eligible_pair_count": audit["eligible_pair_count"],
                "gate_passed": audit["gate_passed"],
            },
        )
        return audit
    except Exception as exc:
        write_json(
            output / "rescore_status.json",
            {
                "status": "failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "phase": phase,
                "candidate": active_candidate,
            },
        )
        raise
