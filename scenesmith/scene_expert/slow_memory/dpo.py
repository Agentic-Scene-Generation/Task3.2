"""Strict DPO pair curation, validation, diagnostics, and leakage-safe splits."""

from __future__ import annotations

import hashlib
import json
import time

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from pydantic import ValidationError

from scenesmith.scene_expert.slow_memory.schemas import (
    DPOPreferencePair,
    TrajectoryRecord,
)


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _hash(value: str, length: int = 24) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:length]


def _append_diagnostic(
    diagnostics: list[dict[str, Any]],
    *,
    reason: str,
    detail: str,
    trajectories: Iterable[TrajectoryRecord] = (),
) -> None:
    records = list(trajectories)
    diagnostics.append(
        {
            "reason": reason,
            "detail": detail,
            "trajectory_ids": [record.trajectory_id for record in records],
            "context_hashes": sorted({record.context_hash for record in records}),
            "task_ids": sorted({record.task_id for record in records}),
        }
    )


def load_trajectories(
    paths: Iterable[Path],
) -> tuple[list[TrajectoryRecord], list[dict[str, Any]]]:
    """Load and deduplicate versioned trajectory rows."""
    trajectories: dict[str, TrajectoryRecord] = {}
    diagnostics: list[dict[str, Any]] = []
    for source in paths:
        source = Path(source)
        if source.is_dir():
            candidates = sorted(source.rglob("trajectories.jsonl"))
            if not candidates:
                diagnostics.append(
                    {
                        "reason": "no_trajectory_files",
                        "detail": str(source),
                        "trajectory_ids": [],
                        "context_hashes": [],
                        "task_ids": [],
                    }
                )
        else:
            candidates = [source]
        for path in candidates:
            if not path.exists():
                diagnostics.append(
                    {
                        "reason": "missing_trajectory_source",
                        "detail": str(path),
                        "trajectory_ids": [],
                        "context_hashes": [],
                        "task_ids": [],
                    }
                )
                continue
            for line_number, line in enumerate(
                path.read_text(encoding="utf-8").splitlines(), start=1
            ):
                if not line.strip():
                    continue
                try:
                    record = TrajectoryRecord.model_validate_json(line)
                except (ValidationError, ValueError) as exc:
                    diagnostics.append(
                        {
                            "reason": "invalid_trajectory",
                            "detail": f"{path}:{line_number}: {exc}",
                            "trajectory_ids": [],
                            "context_hashes": [],
                            "task_ids": [],
                        }
                    )
                    continue
                existing = trajectories.get(record.trajectory_id)
                if existing is not None and existing != record:
                    _append_diagnostic(
                        diagnostics,
                        reason="trajectory_id_collision",
                        detail="same trajectory_id has different payloads",
                        trajectories=[existing, record],
                    )
                    continue
                trajectories[record.trajectory_id] = record
    return list(trajectories.values()), diagnostics


def build_preference_pairs(
    trajectories: Iterable[TrajectoryRecord],
    *,
    min_quality_margin: float = 0.05,
) -> tuple[list[DPOPreferencePair], list[dict[str, Any]]]:
    """Build at most one best-vs-worst pair for each exact model context.

    Context equality is deliberately strict: prompt hash, task, stage, role,
    and event must all match.  Near matches are diagnostics, never training rows.
    """
    groups: dict[tuple[str, str, str, str, str], list[TrajectoryRecord]] = defaultdict(
        list
    )
    diagnostics: list[dict[str, Any]] = []
    for record in trajectories:
        if not record.prompt_complete or not record.response_complete:
            _append_diagnostic(
                diagnostics,
                reason="truncated_payload",
                detail="truncated prompt or response cannot be used for DPO",
                trajectories=[record],
            )
            continue
        if not record.evidence.authoritative:
            _append_diagnostic(
                diagnostics,
                reason="non_authoritative_evidence",
                detail="record retained for audit but not preference training",
                trajectories=[record],
            )
            continue
        if record.evidence.verdict not in {"accepted", "rejected"}:
            _append_diagnostic(
                diagnostics,
                reason="unlabeled_trajectory",
                detail="no candidate-level accepted/rejected verdict",
                trajectories=[record],
            )
            continue
        key = (
            record.context_hash,
            record.task_id,
            record.stage,
            record.agent_role,
            record.event,
        )
        groups[key].append(record)

    pairs: list[DPOPreferencePair] = []
    for key, records in groups.items():
        accepted = [r for r in records if r.evidence.verdict == "accepted"]
        rejected = [r for r in records if r.evidence.verdict == "rejected"]
        if not accepted:
            _append_diagnostic(
                diagnostics,
                reason="missing_exact_context_counterpart",
                detail=("exact context has no authoritative accepted response"),
                trajectories=records,
            )
            continue
        chosen = max(
            accepted,
            key=lambda record: (
                (
                    record.evidence.quality_score
                    if record.evidence.quality_score is not None
                    else -1.0
                ),
                record.trajectory_id,
            ),
        )
        preference_basis = "accepted_vs_rejected_outcome"
        negative_evidence = None
        if rejected:
            negative = min(
                rejected,
                key=lambda record: (
                    (
                        record.evidence.quality_score
                        if record.evidence.quality_score is not None
                        else 0.0
                    ),
                    record.trajectory_id,
                ),
            )
        else:
            critic_ranked = [
                record
                for record in accepted
                if record.evidence.kind in {"critic", "critic_and_deterministic"}
                and record.evidence.quality_score is not None
            ]
            if len(critic_ranked) < 2:
                _append_diagnostic(
                    diagnostics,
                    reason="missing_exact_context_counterpart",
                    detail=(
                        "exact context has neither a rejected outcome nor two "
                        "critic-scored accepted responses"
                    ),
                    trajectories=records,
                )
                continue
            chosen = max(
                critic_ranked,
                key=lambda record: (
                    float(record.evidence.quality_score or 0.0),
                    record.trajectory_id,
                ),
            )
            negative = min(
                critic_ranked,
                key=lambda record: (
                    float(record.evidence.quality_score or 0.0),
                    record.trajectory_id,
                ),
            )
            preference_basis = "relative_main_critic_ranking"
            negative_evidence = negative.evidence.model_copy(
                update={
                    "verdict": "rejected",
                    "source": negative.evidence.source + ":relative_preference",
                    "details": {
                        **negative.evidence.details,
                        "observed_runtime_verdict": "accepted",
                        "preference_derivation": (
                            "lower main-critic quality under exact same context"
                        ),
                    },
                },
                deep=True,
            )
        if chosen.prompt != negative.prompt:
            _append_diagnostic(
                diagnostics,
                reason="context_hash_prompt_mismatch",
                detail=(
                    "matching context_hash values contain different prompts; "
                    "the pair was rejected as corrupt or externally fabricated"
                ),
                trajectories=[chosen, negative],
            )
            continue
        if chosen.evidence.kind == "none" or negative.evidence.kind == "none":
            _append_diagnostic(
                diagnostics,
                reason="unsupported_evidence_kind",
                detail="both sides require critic or deterministic evidence",
                trajectories=[chosen, negative],
            )
            continue
        if chosen.response_hash == negative.response_hash:
            _append_diagnostic(
                diagnostics,
                reason="identical_responses",
                detail="chosen and rejected evidence points to identical output",
                trajectories=[chosen, negative],
            )
            continue
        chosen_score = chosen.evidence.quality_score
        rejected_score = negative.evidence.quality_score
        if chosen_score is None or rejected_score is None:
            _append_diagnostic(
                diagnostics,
                reason="missing_quality_score",
                detail="both preference sides require an evidence-derived score",
                trajectories=[chosen, negative],
            )
            continue
        margin = float(chosen_score - rejected_score)
        if margin < min_quality_margin:
            _append_diagnostic(
                diagnostics,
                reason="insufficient_quality_margin",
                detail=f"quality margin {margin:.4f} < {min_quality_margin:.4f}",
                trajectories=[chosen, negative],
            )
            continue
        pair_id = "dpo_" + _hash(
            "|".join([*key, chosen.trajectory_id, negative.trajectory_id]), 32
        )
        pairs.append(
            DPOPreferencePair(
                pair_id=pair_id,
                prompt=[{"role": "user", "content": chosen.prompt}],
                chosen=[{"role": "assistant", "content": chosen.response}],
                rejected=[{"role": "assistant", "content": negative.response}],
                context_hash=chosen.context_hash,
                leakage_group=chosen.task_id,
                task_id=chosen.task_id,
                stage=chosen.stage,
                agent_role=chosen.agent_role,
                chosen_trajectory_id=chosen.trajectory_id,
                rejected_trajectory_id=negative.trajectory_id,
                chosen_evidence=chosen.evidence,
                rejected_evidence=negative_evidence or negative.evidence,
                quality_margin=margin,
                provenance={
                    "chosen_run_id": chosen.run_id,
                    "rejected_run_id": negative.run_id,
                    "chosen_source_refs": chosen.source_refs,
                    "rejected_source_refs": negative.source_refs,
                    "chosen_model_id": chosen.model_id,
                    "rejected_model_id": negative.model_id,
                    "chosen_config_hash": chosen.config_hash,
                    "rejected_config_hash": negative.config_hash,
                    "exact_prompt_match": chosen.prompt == negative.prompt,
                    "preference_basis": preference_basis,
                    "rejected_response_is_observed": True,
                },
            )
        )
    return sorted(pairs, key=lambda pair: pair.pair_id), diagnostics


def _split_groups(
    pairs: list[DPOPreferencePair],
    *,
    validation_ratio: float,
    test_ratio: float,
    seed: int,
) -> dict[str, list[DPOPreferencePair]]:
    if validation_ratio < 0 or test_ratio < 0 or validation_ratio + test_ratio >= 1:
        raise ValueError("validation_ratio + test_ratio must be in [0, 1)")
    by_group: dict[str, list[DPOPreferencePair]] = defaultdict(list)
    for pair in pairs:
        by_group[pair.leakage_group].append(pair)
    groups = sorted(by_group, key=lambda group: _hash(f"{seed}:{group}", 64))
    count = len(groups)
    validation_count = round(count * validation_ratio)
    test_count = round(count * test_ratio)
    if count >= 3 and validation_ratio > 0:
        validation_count = max(1, validation_count)
    if count >= 3 and test_ratio > 0:
        test_count = max(1, test_count)
    while validation_count + test_count >= count and test_count > 0:
        test_count -= 1
    while validation_count + test_count >= count and validation_count > 0:
        validation_count -= 1
    train_count = count - validation_count - test_count
    assignments: dict[str, str] = {}
    for group in groups[:train_count]:
        assignments[group] = "train"
    for group in groups[train_count : train_count + validation_count]:
        assignments[group] = "validation"
    for group in groups[train_count + validation_count :]:
        assignments[group] = "test"
    splits = {"train": [], "validation": [], "test": []}
    for group, group_pairs in by_group.items():
        splits[assignments[group]].extend(group_pairs)
    return splits


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def validate_dataset_dir(dataset_dir: Path) -> dict[str, Any]:
    """Validate schema, exact prompt equality, and cross-split group isolation."""
    dataset_dir = Path(dataset_dir)
    errors: list[str] = []
    split_groups: dict[str, set[str]] = {}
    split_counts: dict[str, int] = {}
    pair_ids: set[str] = set()
    for split in ("train", "validation", "test"):
        path = dataset_dir / f"{split}.jsonl"
        groups: set[str] = set()
        count = 0
        if not path.exists():
            errors.append(f"missing split file: {path.name}")
            split_groups[split] = groups
            split_counts[split] = 0
            continue
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not line.strip():
                continue
            try:
                pair = DPOPreferencePair.model_validate_json(line)
            except (ValidationError, ValueError) as exc:
                errors.append(f"{path.name}:{line_number}: {exc}")
                continue
            if pair.pair_id in pair_ids:
                errors.append(f"duplicate pair_id: {pair.pair_id}")
            pair_ids.add(pair.pair_id)
            if not pair.provenance.get("exact_prompt_match"):
                errors.append(f"pair does not prove exact prompt match: {pair.pair_id}")
            if pair.provenance.get("rejected_response_is_observed") is not True:
                errors.append(
                    f"pair does not prove rejected response provenance: {pair.pair_id}"
                )
            groups.add(pair.leakage_group)
            count += 1
        split_groups[split] = groups
        split_counts[split] = count
    for left, right in (
        ("train", "validation"),
        ("train", "test"),
        ("validation", "test"),
    ):
        overlap = split_groups[left] & split_groups[right]
        if overlap:
            errors.append(
                f"leakage groups overlap between {left} and {right}: {sorted(overlap)}"
            )
    if split_counts.get("train", 0) == 0:
        errors.append("training split is empty")
    return {
        "schema_version": "sceneexpert.dpo_validation.v1",
        "validated_at": _utc_now(),
        "valid": not errors,
        "errors": errors,
        "split_counts": split_counts,
        "split_leakage_groups": {
            split: sorted(groups) for split, groups in split_groups.items()
        },
        "pair_count": len(pair_ids),
    }


def export_dpo_dataset(
    *,
    trajectory_sources: Iterable[Path],
    output_dir: Path,
    validation_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
    min_quality_margin: float = 0.05,
) -> dict[str, Any]:
    """Materialize a complete, auditable DPO dataset package."""
    trajectory_sources = [Path(path) for path in trajectory_sources]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    trajectories, load_diagnostics = load_trajectories(trajectory_sources)
    pairs, pair_diagnostics = build_preference_pairs(
        trajectories, min_quality_margin=min_quality_margin
    )
    diagnostics = load_diagnostics + pair_diagnostics
    splits = _split_groups(
        pairs,
        validation_ratio=validation_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )
    for split, split_pairs in splits.items():
        _write_jsonl(
            output_dir / f"{split}.jsonl",
            (pair.model_dump(mode="json") for pair in split_pairs),
        )
    _write_jsonl(
        output_dir / "all.jsonl",
        (pair.model_dump(mode="json") for pair in pairs),
    )
    _write_jsonl(output_dir / "rejected_pair_diagnostics.jsonl", diagnostics)

    reason_counts = Counter(item.get("reason", "unknown") for item in diagnostics)
    stage_counts = Counter(pair.stage for pair in pairs)
    role_counts = Counter(pair.agent_role for pair in pairs)
    evidence_counts = Counter(pair.chosen_evidence.kind for pair in pairs)
    preference_basis_counts = Counter(
        str(pair.provenance.get("preference_basis") or "unknown") for pair in pairs
    )
    stats = {
        "schema_version": "sceneexpert.dpo_stats.v1",
        "generated_at": _utc_now(),
        "trajectory_count": len(trajectories),
        "eligible_pair_count": len(pairs),
        "diagnostic_count": len(diagnostics),
        "diagnostic_reasons": dict(sorted(reason_counts.items())),
        "pairs_by_stage": dict(sorted(stage_counts.items())),
        "pairs_by_role": dict(sorted(role_counts.items())),
        "pairs_by_evidence": dict(sorted(evidence_counts.items())),
        "pairs_by_preference_basis": dict(sorted(preference_basis_counts.items())),
        "split_counts": {split: len(rows) for split, rows in splits.items()},
        "unique_leakage_groups": len({pair.leakage_group for pair in pairs}),
        "quality_margin": {
            "minimum_required": min_quality_margin,
            "minimum_observed": min(
                (pair.quality_margin for pair in pairs), default=None
            ),
            "maximum_observed": max(
                (pair.quality_margin for pair in pairs), default=None
            ),
        },
        "text_chars": {
            "max_prompt": max(
                (len(pair.prompt[0]["content"]) for pair in pairs), default=0
            ),
            "max_chosen": max(
                (len(pair.chosen[0]["content"]) for pair in pairs), default=0
            ),
            "max_rejected": max(
                (len(pair.rejected[0]["content"]) for pair in pairs), default=0
            ),
        },
    }
    (output_dir / "stats.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False),
        encoding="utf-8",
        newline="\n",
    )
    validation = validate_dataset_dir(output_dir)
    (output_dir / "validation.json").write_text(
        json.dumps(validation, indent=2, ensure_ascii=False),
        encoding="utf-8",
        newline="\n",
    )
    manifest = {
        "schema_version": "sceneexpert.dpo_dataset_manifest.v1",
        "generated_at": _utc_now(),
        "source_paths": [str(path) for path in trajectory_sources],
        "seed": seed,
        "validation_ratio": validation_ratio,
        "test_ratio": test_ratio,
        "min_quality_margin": min_quality_margin,
        "pairing_policy": {
            "same_context_required": True,
            "authoritative_evidence_required": True,
            "synthetic_rejected_responses_allowed": False,
            "relative_critic_ranking_allowed": True,
            "truncated_payloads_allowed": False,
            "one_best_worst_pair_per_context": True,
        },
        "files": {
            "train": "train.jsonl",
            "validation": "validation.jsonl",
            "test": "test.jsonl",
            "all": "all.jsonl",
            "diagnostics": "rejected_pair_diagnostics.jsonl",
            "statistics": "stats.json",
            "validation_report": "validation.json",
        },
        "stats": stats,
        "validation": validation,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
        newline="\n",
    )
    return manifest
