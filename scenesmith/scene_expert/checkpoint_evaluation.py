"""Register fixed diagnostic inputs and audit balanced frozen-pair results.

This is an evaluation entrypoint, not a replacement for native scene generation.
It never executes instructions/tools in checkpoint or feedback files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math

from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from scenesmith.scene_expert.evaluation_costs import timestamp
from scenesmith.scene_expert.evaluation_io import read_object
from scenesmith.scene_expert.memory.evidence import evidence_hash

REQUIRED_INPUTS = {
    "checkpoint",
    "task_spec",
    "intent_contract",
    "model_config",
    "tool_config",
}
REQUIRED_IDENTITIES = {
    "shared_base_fingerprint",
    "task_spec_fingerprint",
    "intent_contract_fingerprint",
    "decision_state_fingerprint",
}


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def register(spec: dict, root: Path) -> dict:
    """Require 6–10 predetermined independent cases and complete fixed inputs."""
    problems = spec.get("problems") or []
    if not 6 <= len(problems) <= 10:
        raise ValueError("Register 6–10 fixed independent problems before running them")
    ids = [str(row.get("case_id") or "") for row in problems]
    tasks = [str(row.get("task_id") or "") for row in problems]
    if (
        len(set(ids)) != len(ids)
        or "" in ids
        or len(set(tasks)) != len(tasks)
        or "" in tasks
    ):
        raise ValueError("case_id and task_id must be unique and nonempty")
    prepared = []
    for row in problems:
        inputs = row.get("inputs") or {}
        if not REQUIRED_INPUTS <= inputs.keys():
            raise ValueError(f"Missing fixed input files for {row['case_id']}")
        if row.get("stage") not in {
            "furniture",
            "wall_mounted",
            "ceiling_mounted",
            "manipuland",
        }:
            raise ValueError("Diagnostic checkpoint must be a native RoomScene stage")
        expected_identity = row.get("expected_identity") or {}
        if any(not expected_identity.get(key) for key in REQUIRED_IDENTITIES):
            raise ValueError(
                "Each fixed problem needs expected runtime input fingerprints"
            )
        files = {}
        for name, value in inputs.items():
            path = (root / str(value)).resolve()
            if not path.is_file():
                raise ValueError(f"Missing input: {path}")
            files[name] = {"path": str(path), "sha256": _file_hash(path)}
        reviewed = []
        for value in row.get("reviewed_memory_sources", []):
            path = (root / str(value)).resolve()
            warnings: list[str] = []
            source = read_object(path, warnings)
            source_tasks = source.get("source_task_ids") or []
            if warnings or not source_tasks or row["task_id"] in source_tasks:
                raise ValueError(
                    "Reviewed memory must have independent training-task provenance"
                )
            reviewed.append(
                {
                    "path": str(path),
                    "sha256": _file_hash(path),
                    "source_task_ids": source_tasks,
                }
            )
        prepared.append(
            {
                "case_id": row["case_id"],
                "task_id": row["task_id"],
                "stage": row["stage"],
                "expected_identity": expected_identity,
                "inputs": files,
                "reviewed_memory_sources": reviewed,
                "target_constraint_ids": list(row.get("target_constraint_ids") or []),
            }
        )
    policy = spec.get("acceptance") or {}
    for key in ("max_mean_quality_loss", "min_cost_reduction_fraction"):
        value = policy.get(key)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or not 0 <= value <= 1
        ):
            raise ValueError(f"Explicit [0,1] acceptance threshold required: {key}")
    result = {
        "schema_version": "memory-checkpoint-plan.v1",
        "registered_at": datetime.now(timezone.utc).isoformat(),
        "problems": prepared,
        "acceptance": policy,
        "minimum_balanced_pairs": 2,
    }
    return {**result, "manifest_hash": evidence_hash(result)}


def verify(manifest: dict) -> list[str]:
    """Reject edited plans and inputs; the local timestamp is not external attestation."""
    errors = []
    problems = manifest.get("problems") or []
    if (
        manifest.get("schema_version") != "memory-checkpoint-plan.v1"
        or not 6 <= len(problems) <= 10
    ):
        errors.append("invalid_checkpoint_plan_schema_or_case_count")
    if len({row.get("case_id") for row in problems}) != len(problems) or len(
        {row.get("task_id") for row in problems}
    ) != len(problems):
        errors.append("nonindependent_checkpoint_problems")
    if manifest.get("manifest_hash") != evidence_hash(
        {key: value for key, value in manifest.items() if key != "manifest_hash"}
    ):
        errors.append("manifest_hash_mismatch")
    for row in manifest.get("problems", []):
        if (
            not REQUIRED_INPUTS <= (row.get("inputs") or {}).keys()
            or not REQUIRED_IDENTITIES <= (row.get("expected_identity") or {}).keys()
        ):
            errors.append("incomplete_fixed_inputs")
        for record in [
            *(row.get("inputs") or {}).values(),
            *row.get("reviewed_memory_sources", []),
        ]:
            try:
                matches = _file_hash(Path(record["path"])) == record["sha256"]
            except (OSError, KeyError, TypeError):
                matches = False
            if not matches:
                errors.append(
                    f"checkpoint_input_changed:{row.get('case_id')}:{record.get('path')}"
                )
    return errors


def assess(manifest: dict, comparisons: list[dict]) -> dict:
    """Use every preregistered case and both arm orders; do not cherry-pick wins."""
    errors = verify(manifest)
    expected = {row["case_id"] for row in manifest.get("problems", [])}
    expected_rows = {row["case_id"]: row for row in manifest.get("problems", [])}
    registered = timestamp(manifest.get("registered_at", ""))
    orders = set()
    order_counts = Counter()
    identities = set()
    run_ids = []
    for result in comparisons:
        if (
            not result.get("comparison_ready")
            or not result.get("speed_comparison_ready")
            or not result.get("quality_delta_ready")
        ):
            errors.append("comparison_not_ready")
        if {row.get("case_id") for row in result.get("pairs", [])} != expected:
            errors.append("preregistered_case_set_mismatch")
        binding = result.get("checkpoint_plan_hash")
        if binding != manifest.get("manifest_hash"):
            errors.append("run_not_bound_to_checkpoint_plan")
        for pair in result.get("pairs", []):
            expected_row = expected_rows.get(pair.get("case_id"), {})
            if pair.get("task_id") != expected_row.get("task_id"):
                errors.append("preregistered_task_mismatch")
            for key in REQUIRED_IDENTITIES:
                if pair.get("fixed_input_identity", {}).get(key) != expected_row.get(
                    "expected_identity", {}
                ).get(key):
                    errors.append(f"preregistered_input_mismatch:{key}")
        started = timestamp(result.get("earliest_scene_started_at", ""))
        if registered is None or started is None or registered >= started:
            errors.append("plan_not_registered_before_run")
        orders.update(result.get("arm_orders") or [])
        if len(result.get("arm_orders") or []) != 1:
            errors.append("mixed_or_missing_arm_order_within_pair")
        order_counts.update(result.get("arm_orders") or [])
        identities.add((result.get("baseline_run_id"), result.get("treatment_run_id")))
        run_ids.extend((result.get("baseline_run_id"), result.get("treatment_run_id")))
    if (
        len(identities) < 2
        or orders != {"off_on", "on_off"}
        or order_counts["off_on"] != order_counts["on_off"]
        or len(set(run_ids)) != len(run_ids)
    ):
        errors.append("need_two_independent_balanced_pairs")
    deltas = [
        row.get("critic_score_delta")
        for result in comparisons
        for row in result.get("pairs", [])
    ]
    times = [
        (row.get("baseline_time_sec"), row.get("treatment_time_sec"))
        for result in comparisons
        for row in result.get("pairs", [])
    ]
    if (
        not deltas
        or any(value is None for value in deltas)
        or any(a is None or b is None for a, b in times)
    ):
        errors.append("incomplete_all_assigned_quality_or_cost")
    if errors:
        return {
            "decision": "inconclusive",
            "reasons": sorted(set(errors)),
            "causal_proof": False,
        }
    policy = manifest["acceptance"]
    mean_quality = sum(deltas) / len(deltas)
    before, after = sum(a for a, _ in times), sum(b for _, b in times)
    reduction = (before - after) / before if before > 0 else None
    no_regression = all(
        row.get("outcome_transition") != "regressed"
        for result in comparisons
        for row in result["pairs"]
    )
    meets = (
        no_regression
        and mean_quality >= -policy["max_mean_quality_loss"]
        and reduction is not None
        and reduction >= policy["min_cost_reduction_fraction"]
    )
    return {
        "decision": (
            "meets_preregistered_engineering_threshold"
            if meets
            else "does_not_meet_preregistered_engineering_threshold"
        ),
        "mean_quality_delta": mean_quality,
        "all_assigned_cost_reduction_fraction": reduction,
        "completion_nonregression": no_regression,
        "causal_proof": False,
        "hardware_equivalence_verified": False,
        "limitations": "Small paired diagnostics are not population-level or continual-memory-learning proof. Local registration is auditable, not externally timestamp-attested. Cost conclusions assume operator-controlled equal resources, backend deployment and competing load; these hardware conditions are not automatically verified.",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("register", "verify", "assess"))
    parser.add_argument("--spec", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--comparison", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.action == "register":
        if not args.spec or not args.output:
            parser.error("register requires --spec and --output")
        payload = register(
            json.loads(args.spec.read_text(encoding="utf-8")), args.spec.parent
        )
    else:
        if not args.manifest:
            parser.error("--manifest is required")
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        payload = (
            {"errors": verify(manifest)}
            if args.action == "verify"
            else assess(
                manifest,
                [
                    json.loads(path.read_text(encoding="utf-8"))
                    for path in args.comparison
                ],
            )
        )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    else:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    return (
        1 if payload.get("errors") or payload.get("decision") == "inconclusive" else 0
    )


if __name__ == "__main__":
    raise SystemExit(main())
