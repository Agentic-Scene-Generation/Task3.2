"""Portable bookkeeping for the opt-in furniture initial-decision pilot.

No simulator imports: archive validation and export work on the review machine.
Native execution lives in paired_runtime.py; a group never selects a winner for
the online policy. A is always the canonical continuation.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
from pathlib import Path
from typing import Any

from scenesmith.scene_expert.slow_memory.dpo import (
    export_dpo_dataset,
    load_trajectories,
)

EXPECTED_MODEL = "unsloth/Qwen3.8-27B-GGUF"


def digest(value: Any) -> str:
    """Hash complete JSON, without dropping state or history fields."""
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def write_json(path: Path, value: Any) -> None:
    """Replace one completed JSON artifact atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    temporary.replace(path)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def tree_hashes(root: Path) -> dict[str, str]:
    """Fingerprint every copied input file; reject links and excessive snapshots."""
    result: dict[str, str] = {}
    total = 0
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"snapshot contains a symlink: {path}")
        if not path.is_file():
            continue
        total += path.stat().st_size
        if total > 2 * 1024**3 or len(result) >= 20000:
            raise ValueError(
                "initial snapshot exceeds the 2 GiB / 20000 file pilot limit"
            )
        with path.open("rb") as stream:
            result[path.relative_to(root).as_posix()] = hashlib.file_digest(
                stream, "sha256"
            ).hexdigest()
    return result


def copy_scene_tree(source: Path, destination: Path) -> dict[str, str]:
    """Copy files, never share writable inodes or live SQLite connections."""
    if destination.exists():
        raise FileExistsError(destination)
    # SQLite sessions are checked empty at the supported decision boundary.
    # Old floor-plan sessions, logs and caches are not part of that decision.
    ignore = shutil.ignore_patterns("*.db", "*.db-*", "*.log", "__pycache__", "*.lock")
    if destination.resolve().is_relative_to(source.resolve()):
        raise ValueError("snapshot destination must be outside the live scene")
    tree_hashes(source)  # Enforce the bound before copying bytes.
    shutil.copytree(source, destination, ignore=ignore, copy_function=shutil.copy2)
    return tree_hashes(destination)


def rebase(value: Any, source: str, destination: str) -> Any:
    """Rebase serialized scene/runtime paths in a private restoration."""
    if isinstance(value, str):
        return value.replace(source, destination)
    if isinstance(value, list):
        return [rebase(v, source, destination) for v in value]
    if isinstance(value, dict):
        return {k: rebase(v, source, destination) for k, v in value.items()}
    return value


def reserve_group(root: Path, identity: str, limit: int) -> Path | None:
    """Claim a bounded slot without overwriting a previous group."""
    if not 1 <= limit <= 4:
        raise ValueError("initial pilot supports 1 to 4 decision groups")
    root.mkdir(parents=True, exist_ok=True)
    for group in root.glob("group_*/identity.json"):
        if read_json(group).get("identity") == identity:
            return None
    for index in range(limit):
        group = root / f"group_{index:03d}"
        try:
            group.mkdir()
        except FileExistsError:
            continue
        write_json(group / "identity.json", {"identity": identity})
        return group
    return None


def deterministic_verdict(report: dict[str, Any]) -> tuple[str, float, int]:
    """Keep unknown or absent Main evidence out of the preference dataset."""
    summary = (report.get("summary") or {}).get("scene_summary") or {}
    total = int(summary.get("total_checks") or 0)
    unknown = int(summary.get("unknown") or 0)
    score = summary.get("score")
    if (
        total <= 0
        or unknown
        or not isinstance(score, (int, float))
        or not math.isfinite(score)
        or not 0 <= score <= 1
    ):
        raise ValueError("candidate lacks complete deterministic Main evidence")
    failures = int(summary.get("fail") or 0)
    if not 0 <= failures <= total:
        raise ValueError("invalid deterministic failure count")
    return ("accepted" if failures == 0 else "rejected", float(score), failures)


def validate_tool_execution(
    trace: dict[str, Any], fatal_asset_error: Any = None
) -> None:
    """Quarantine transport/unhandled-tool failures, never label them as poor design."""
    if fatal_asset_error:
        raise ValueError(f"candidate asset infrastructure failed: {fatal_asset_error}")
    failures = re.compile(
        r"an error occurred while running the tool|connection(?:error| refused| reset)|"
        r"connecttimeout|readtimeout|timed out|out of memory|cuda error|"
        r"no available ports|server (?:unavailable|disconnected)|http[^\n]{0,30}\b50[234]\b",
        re.IGNORECASE,
    )
    for result in trace.get("tool_results") or []:
        output = result.get("output", "")
        text = output if isinstance(output, str) else json.dumps(output)
        if failures.search(text):
            raise ValueError(
                "candidate tool trace contains an infrastructure or unhandled-tool failure"
            )


def validate_pair_inputs(
    root: Path, *, require_fresh_physics: bool = True
) -> tuple[list[Path], list[dict[str, Any]], list[str]]:
    """Validate immutable execution artifacts; legacy mode is for rescoring only."""
    sources: list[Path] = []
    groups: list[dict[str, Any]] = []
    errors: list[str] = []
    for group in sorted(root.glob("group_*")):
        reasons: list[str] = []
        try:
            status = read_json(group / "status.json")
            if status.get("status") != "completed":
                reasons.append("group_execution_incomplete")
                if status.get("error"):
                    reasons.append(
                        f"group_failure[{status.get('phase', 'unknown')}]: {status['error']}"
                    )
            snapshot = read_json(group / "snapshot.json")
            proof = read_json(group / "continuation_proof.json")
            if (
                proof["canonical_before"] != proof["canonical_after"]
                or proof["assets_before"] != proof["assets_after"]
                or proof["memory_before"] != proof["memory_after"]
            ):
                reasons.append("shadow_changed_canonical_state_or_memory")
            if tree_hashes(group / "input_scene") != snapshot["files"]:
                reasons.append("snapshot_files_changed")
            a = read_json(group / "A/result.json")
            b = read_json(group / "B/result.json")
            request_a = read_json(group / "A/first_request.json")
            request_b = read_json(group / "B/first_request.json")
            if request_a != request_b:
                reasons.append("effective_request_mismatch")
            if request_a.get("model") != EXPECTED_MODEL:
                reasons.append("unexpected_model")
            for name, candidate in (("A", a), ("B", b)):
                if require_fresh_physics:
                    from scenesmith.scene_expert.slow_memory.paired_scoring import (
                        validate_scoring_proof,
                    )

                    validate_scoring_proof(group / name, candidate)
                if candidate.get("status") != "completed":
                    reasons.append(f"{name}_execution_failed")
                if digest(
                    read_json(group / name / "returned_state.json")
                ) != candidate.get("returned_state_hash") or digest(
                    read_json(group / name / "safety.json")
                ) != candidate.get(
                    "safety_hash"
                ):
                    reasons.append(f"{name}_safety_proof_changed")
                if candidate.get("snapshot_hash") != digest(snapshot):
                    reasons.append(f"{name}_snapshot_mismatch")
                if candidate.get("evaluation_state_hash") != candidate.get(
                    "raw_state_hash"
                ):
                    reasons.append(f"{name}_evaluation_state_changed")
                if digest(read_json(group / name / "raw_state.json")) != candidate.get(
                    "raw_state_hash"
                ):
                    reasons.append(f"{name}_raw_state_changed")
                if tree_hashes(group / name / "raw_scene") != candidate.get(
                    "raw_files"
                ):
                    reasons.append(f"{name}_raw_assets_changed")
                verdict, score, _ = deterministic_verdict(
                    read_json(group / name / "report.json")
                )
                if verdict != candidate.get("verdict") or score != candidate.get(
                    "score"
                ):
                    reasons.append(f"{name}_evidence_mismatch")
                if candidate.get("model") != EXPECTED_MODEL:
                    reasons.append(f"{name}_model_mismatch")
                if not (group / name / "slow_memory/trajectories.jsonl").is_file():
                    reasons.append(f"{name}_trajectory_missing")
                if hashlib.sha256(
                    (group / name / "slow_memory/trajectories.jsonl").read_bytes()
                ).hexdigest() != candidate.get("trajectory_sha256"):
                    reasons.append(f"{name}_trajectory_bytes_changed")
                candidate_records, candidate_errors = load_trajectories(
                    [group / name / "slow_memory/trajectories.jsonl"]
                )
                if candidate_errors or len(candidate_records) != 1:
                    reasons.append(f"{name}_invalid_candidate_capture")
                for record in candidate_records:
                    if (
                        record.task_type != "designer_initial"
                        or record.stage != "furniture"
                        or record.model_id != EXPECTED_MODEL
                        or not record.prompt_complete
                        or not record.response_complete
                    ):
                        reasons.append(f"{name}_invalid_candidate_scope")
                    if record.spatial_context.get("initial_snapshot_sha256") != digest(
                        snapshot
                    ):
                        reasons.append(f"{name}_incomplete_pairing_context")
                    if (
                        record.evidence.details.get("raw_state_sha256")
                        != candidate.get("raw_state_hash")
                        or record.evidence.verdict != verdict
                        or record.evidence.quality_score != score
                        or record.evidence.source
                        != "main_raw_candidate_deterministic_checks"
                    ):
                        reasons.append(f"{name}_trajectory_evidence_mismatch")
                    for reference in [
                        *record.image_refs,
                        *record.provenance.get("tool_media_refs", []),
                    ]:
                        media_path = Path(reference["path"])
                        if not media_path.is_absolute():
                            media_path = group / name / "slow_memory" / media_path
                        with media_path.open("rb") as stream:
                            actual = hashlib.file_digest(stream, "sha256").hexdigest()
                        if actual != reference.get("sha256"):
                            reasons.append(f"{name}_media_hash_mismatch")
            if not reasons:
                sources.extend(
                    [
                        group / "A/slow_memory/trajectories.jsonl",
                        group / "B/slow_memory/trajectories.jsonl",
                    ]
                )
        except (OSError, ValueError, KeyError, TypeError) as exc:
            reasons.append(f"incomplete_group: {exc}")
        groups.append({"group": group.name, "valid": not reasons, "errors": reasons})
        errors.extend(f"{group.name}: {reason}" for reason in reasons)
    return sources, groups, errors


def audit_pairs(
    root: Path, *, min_pairs: int = 1, expected_groups: int = 1
) -> dict[str, Any]:
    """Export only independent groups with fresh raw-state evidence."""
    sources, groups, errors = validate_pair_inputs(root)
    if len(groups) != expected_groups:
        errors.append(f"group_count_mismatch: {len(groups)} != {expected_groups}")
    manifest = export_dpo_dataset(trajectory_sources=sources, output_dir=root / "dpo")
    records, diagnostics = load_trajectories(sources)
    if diagnostics:
        errors.append("trajectory_load_failed")
    execution_integrity_passed = not errors
    count = manifest["stats"]["eligible_pair_count"]
    preference_gate_passed = count >= min_pairs and (
        min_pairs == 0 or manifest["validation"]["valid"]
    )
    if count < min_pairs or (min_pairs > 0 and not manifest["validation"]["valid"]):
        errors.append("minimum_pair_gate_failed")
    result = {
        "schema_version": "sceneexpert.initial_pairs_audit.v1",
        "groups": groups,
        "errors": errors,
        "gate_passed": not errors,
        "execution_integrity_passed": execution_integrity_passed,
        "preference_gate_passed": preference_gate_passed,
        "valid_group_count": sum(group["valid"] for group in groups),
        "candidate_count": len(records),
        "eligible_pair_count": count,
        "min_pairs": min_pairs,
        "expected_groups": expected_groups,
        "scoring": "raw_candidate_main_deterministic_checks",
        "canonical_candidate": "A",
        "training_preflight_status": "not_run",
        "dpo_stats": manifest["stats"],
        "dpo_validation": manifest["validation"],
    }
    write_json(root / "pair_audit.json", result)
    return result
