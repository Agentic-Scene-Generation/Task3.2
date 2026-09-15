"""Fresh, detached evidence for raw candidates; no native repair or model calls."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

from scenesmith.scene_expert.slow_memory.paired import digest, read_json, write_json

SCORING_PROTOCOL = "sceneexpert.raw_candidate_scoring.v3"


def score_raw_candidate(scene: Any, cfg: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    """Recompute physics for the exact raw state without changing its metadata."""
    from scenesmith.agent_utils.physics_validation import compute_scene_collisions
    from scenesmith.scene_expert.slow_memory.paired_runtime import json_value
    from scenesmith.scenebenchmark_critic.adapter import room_scene_to_case_pack
    from scenesmith.scenebenchmark_critic.config import critic_config_from_any
    from scenesmith.scenebenchmark_critic.evaluator import (
        build_all_checks,
        run_case_pack_checks,
    )
    from scenesmith.scenebenchmark_critic.reports import build_evaluation_payload

    raw_hash = digest(json_value(scene.to_state_dict()))
    geometry_hash = scene.content_hash()
    config = critic_config_from_any(cfg)
    if not config.enabled or "physics_collision" not in config.metrics:
        raise ValueError("raw candidate scoring requires the Main physics metric")
    settings = cfg.physics_validation
    collisions = compute_scene_collisions(
        scene=scene,
        penetration_threshold=settings.object_penetration_threshold_m,
        floor_penetration_tolerance=settings.floor_penetration_tolerance_m,
        manipuland_furniture_tolerance_m=settings.manipuland_furniture_tolerance_m,
    )
    physics = {
        "schema_version": "scenesmith.physics_evidence.v1",
        "available": True,
        "source_phase": "sceneexpert_raw_candidate",
        "scene_hash": geometry_hash,
        "raw_state_sha256": raw_hash,
        "penetration_tolerance_m": settings.object_penetration_threshold_m,
        "collisions": [
            {
                "object_a_id": str(c.object_a_id),
                "object_b_id": str(c.object_b_id),
                "object_a_name": str(c.object_a_name),
                "object_b_name": str(c.object_b_name),
                "penetration_depth_m": float(c.penetration_depth),
                "classification": "hard",
            }
            for c in collisions
        ],
    }
    # Main's adapter normally consumes a cached observation. Replace only the
    # detached case pack's evidence; do not refresh live scene metadata or budgets.
    case_pack = deepcopy(
        room_scene_to_case_pack(scene, stage="furniture", metrics=list(config.metrics))
    )
    case_pack["physics_evidence"] = physics
    case_pack["checks"] = build_all_checks(case_pack, metrics=list(config.metrics))
    report = build_evaluation_payload(
        case_pack=case_pack,
        results=run_case_pack_checks(case_pack, config=config),
        stage="furniture",
        scope=f"room:{scene.room_id}",
        config=config,
    )
    after = digest(json_value(scene.to_state_dict()))
    if after != raw_hash or scene.content_hash() != geometry_hash:
        raise ValueError("raw candidate evaluator changed scene state or asset content")
    proof = {
        "schema_version": SCORING_PROTOCOL,
        "raw_state_sha256": raw_hash,
        "evaluation_state_sha256": after,
        "scene_content_hash": geometry_hash,
        "physics_sha256": digest(physics),
        "report_sha256": digest(report),
    }
    return report, proof


def save_scoring_proof(
    directory: Path,
    report: dict[str, Any],
    proof: dict[str, Any],
    raw_files: dict[str, str],
) -> None:
    """Bind fresh measurements to the exact retained raw asset bytes."""
    write_json(directory / "physics.json", report["case_pack"]["physics_evidence"])
    write_json(
        directory / "evaluation_proof.json",
        {**proof, "raw_files_sha256": digest(raw_files)},
    )


def validate_scoring_proof(directory: Path, candidate: dict[str, Any]) -> None:
    """Reject historical cached-physics labels and missing or altered fresh proofs."""
    proof = read_json(directory / "evaluation_proof.json")
    physics = read_json(directory / "physics.json")
    report = read_json(directory / "report.json")
    if proof.get("restoration_proof_sha256"):
        from scenesmith.scene_expert.slow_memory.paired_restoration import (
            validate_restoration_proof,
        )

        validate_restoration_proof(directory, candidate, proof)
    elif candidate.get("evaluation_state_hash") != candidate.get("raw_state_hash"):
        raise ValueError("raw evaluation state changed without verified restoration")
    if (
        candidate.get("scoring_protocol") != SCORING_PROTOCOL
        or proof.get("schema_version") != SCORING_PROTOCOL
        or proof.get("raw_state_sha256") != candidate.get("raw_state_hash")
        or proof.get("evaluation_state_sha256")
        != candidate.get("evaluation_state_hash")
        or physics.get("raw_state_sha256") != candidate.get("evaluation_state_hash")
        or proof.get("physics_sha256") != digest(physics)
        or proof.get("report_sha256") != digest(report)
        or proof.get("raw_files_sha256") != digest(candidate.get("raw_files"))
        or physics.get("available") is not True
        or physics.get("source_phase") != "sceneexpert_raw_candidate"
        or not proof.get("scene_content_hash")
        or physics.get("scene_hash") != proof.get("scene_content_hash")
        or report.get("case_pack", {}).get("physics_evidence") != physics
    ):
        raise ValueError("fresh raw-state physics proof mismatch")
