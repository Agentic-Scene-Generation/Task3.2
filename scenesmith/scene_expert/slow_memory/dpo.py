"""Hard-grounded DPO curation, validation, and leakage-safe splits."""

from __future__ import annotations

import hashlib
import json
import shutil
import time

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from pydantic import ValidationError

from scenesmith.scene_expert.slow_memory.schemas import (
    DPOPreferencePair,
    PreferenceTaskType,
    TrajectoryRecord,
)

DEFAULT_TRAINING_TASK_TYPES: frozenset[PreferenceTaskType] = frozenset(
    {"designer_initial", "designer_repair"}
)
_VOLATILE_CONTEXT_KEYS = frozenset(
    {
        "created_at",
        "updated_at",
        "run_id",
        "scene_id",
        "trace_id",
        "request_id",
        "elapsed_sec",
        "latency_sec",
        "queue_wait_sec",
        "ttft_sec",
        "decode_sec",
    }
)


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _hash(value: str, length: int = 24) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:length]


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def _canonical_context(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): _canonical_context(item)
            for key, item in value.items()
            if str(key) not in _VOLATILE_CONTEXT_KEYS
        }
    if isinstance(value, list):
        return [_canonical_context(item) for item in value]
    return value


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
            "task_types": sorted({record.task_type for record in records}),
        }
    )


def load_trajectories(
    paths: Iterable[Path],
) -> tuple[list[TrajectoryRecord], list[dict[str, Any]]]:
    """Load and deduplicate both legacy and v2 trajectory rows."""

    trajectories: dict[str, TrajectoryRecord] = {}
    diagnostics: list[dict[str, Any]] = []
    for source in paths:
        source = Path(source)
        if source.is_dir():
            candidates = sorted(source.rglob("trajectories*.jsonl"))
            if not candidates:
                diagnostics.append(
                    {
                        "reason": "no_trajectory_files",
                        "detail": str(source),
                        "trajectory_ids": [],
                        "context_hashes": [],
                        "task_ids": [],
                        "task_types": [],
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
                        "task_types": [],
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
                            "task_types": [],
                        }
                    )
                    continue
                resolved_refs: list[dict[str, Any]] = []
                for reference in record.image_refs:
                    normalized = dict(reference)
                    path_value = str(normalized.get("path") or "").strip()
                    image_path = Path(path_value)
                    if path_value and not image_path.is_absolute():
                        normalized["path"] = str((path.parent / image_path).resolve())
                    resolved_refs.append(normalized)
                record = record.model_copy(
                    update={"image_refs": resolved_refs}, deep=True
                )
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


def _prompt_messages(record: TrajectoryRecord) -> list[dict[str, Any]]:
    return record.messages or [{"role": "user", "content": record.prompt}]


def _completion_messages(record: TrajectoryRecord) -> list[dict[str, Any]]:
    return record.completion_messages or [
        {"role": "assistant", "content": record.response}
    ]


def _image_identity(record: TrajectoryRecord) -> list[str]:
    return [
        str(reference.get("sha256") or reference.get("path") or "")
        for reference in record.image_refs
    ]


def _image_paths(record: TrajectoryRecord) -> list[str]:
    return [
        str(reference.get("path") or "")
        for reference in record.image_refs
        if reference.get("path")
    ]


def _candidate_rank(record: TrajectoryRecord) -> tuple[Any, ...]:
    return (
        record.outcome.preference_key(),
        (
            float(record.evidence.quality_score)
            if record.evidence.quality_score is not None
            else -1.0
        ),
        record.trajectory_id,
    )


def _context_mismatch(
    chosen: TrajectoryRecord, rejected: TrajectoryRecord
) -> tuple[str, str] | None:
    checks = (
        (
            "context_hash_prompt_mismatch",
            _prompt_messages(chosen),
            _prompt_messages(rejected),
            "matching context hashes contain different model-visible messages",
        ),
        (
            "context_hash_tool_mismatch",
            chosen.tools,
            rejected.tools,
            "matching context hashes contain different tool schemas",
        ),
        (
            "context_hash_media_mismatch",
            _image_identity(chosen),
            _image_identity(rejected),
            "matching context hashes contain different visual inputs",
        ),
        (
            "context_hash_spatial_mismatch",
            _canonical_context(chosen.spatial_context),
            _canonical_context(rejected.spatial_context),
            "matching context hashes contain different spatial state",
        ),
    )
    for reason, left, right, detail in checks:
        if _canonical(left) != _canonical(right):
            return reason, detail
    if any(not value for value in _image_identity(chosen)):
        return (
            "unverifiable_media_context",
            "a visual context reference is missing its content hash",
        )
    return None


def _preference_margin(
    chosen: TrajectoryRecord,
    rejected: TrajectoryRecord,
    *,
    min_quality_margin: float,
) -> tuple[float | None, dict[str, Any]]:
    chosen_score = chosen.evidence.quality_score
    rejected_score = rejected.evidence.quality_score
    if chosen_score is None or rejected_score is None:
        return None, {"reason": "missing_quality_score"}
    raw_margin = float(chosen_score - rejected_score)
    chosen_key = chosen.outcome.preference_key()
    rejected_key = rejected.outcome.preference_key()
    if chosen_key < rejected_key:
        return None, {
            "reason": "outcome_order_conflict",
            "raw_quality_margin": raw_margin,
            "chosen_outcome_key": chosen_key,
            "rejected_outcome_key": rejected_key,
        }
    hard_first_dominance = chosen_key > rejected_key
    if raw_margin < min_quality_margin and not hard_first_dominance:
        return None, {
            "reason": "insufficient_quality_margin",
            "raw_quality_margin": raw_margin,
        }
    effective_margin = (
        raw_margin if raw_margin >= min_quality_margin else float(min_quality_margin)
    )
    return effective_margin, {
        "raw_quality_margin": raw_margin,
        "hard_first_dominance": hard_first_dominance,
        "chosen_outcome_key": chosen_key,
        "rejected_outcome_key": rejected_key,
    }


def build_preference_pairs(
    trajectories: Iterable[TrajectoryRecord],
    *,
    min_quality_margin: float = 0.05,
    include_task_types: set[str] | frozenset[str] | None = None,
) -> tuple[list[DPOPreferencePair], list[dict[str, Any]]]:
    """Build one observed best-vs-worst pair per exact decision context.

    Transport validity, deterministic hard constraints, and spatial relation
    satisfaction rank ahead of critic aesthetics. Critic advice is eligible only
    when independent downstream execution established its causal outcome.
    """

    groups: dict[tuple[str, str, str, str, str, str], list[TrajectoryRecord]] = (
        defaultdict(list)
    )
    diagnostics: list[dict[str, Any]] = []
    for record in trajectories:
        if (
            include_task_types is not None
            and record.task_type not in include_task_types
        ):
            _append_diagnostic(
                diagnostics,
                reason="excluded_task_type",
                detail=f"task type {record.task_type!r} is not enabled for this export",
                trajectories=[record],
            )
            continue
        if not record.prompt_complete or not record.response_complete:
            _append_diagnostic(
                diagnostics,
                reason="truncated_payload",
                detail="truncated model messages cannot be used for DPO",
                trajectories=[record],
            )
            continue
        if record.agent_role == "critic" and not record.outcome.causal_link_verified:
            _append_diagnostic(
                diagnostics,
                reason="critic_causal_link_unverified",
                detail=(
                    "critic advice needs independent downstream designer/execution "
                    "evidence; self-scoring is not a preference label"
                ),
                trajectories=[record],
            )
            continue
        if not record.evidence.authoritative:
            _append_diagnostic(
                diagnostics,
                reason="non_authoritative_evidence",
                detail="record remains audit data, not preference training data",
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
            record.task_type,
        )
        groups[key].append(record)

    pairs: list[DPOPreferencePair] = []
    for key, records in groups.items():
        accepted = [
            record for record in records if record.evidence.verdict == "accepted"
        ]
        rejected = [
            record for record in records if record.evidence.verdict == "rejected"
        ]
        if not accepted:
            _append_diagnostic(
                diagnostics,
                reason="missing_exact_context_counterpart",
                detail="exact context has no authoritative accepted response",
                trajectories=records,
            )
            continue
        chosen = max(accepted, key=_candidate_rank)
        preference_basis = "accepted_vs_rejected_outcome"
        negative_evidence = None
        if rejected:
            negative = min(rejected, key=_candidate_rank)
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
                        "independently critic-scored observed responses"
                    ),
                    trajectories=records,
                )
                continue
            chosen = max(critic_ranked, key=_candidate_rank)
            negative = min(critic_ranked, key=_candidate_rank)
            preference_basis = "relative_main_critic_ranking"
            negative_evidence = negative.evidence.model_copy(
                update={
                    "verdict": "rejected",
                    "source": negative.evidence.source + ":relative_preference",
                    "details": {
                        **negative.evidence.details,
                        "observed_runtime_verdict": "accepted",
                        "preference_derivation": (
                            "lower hard-first outcome under the exact same context"
                        ),
                    },
                },
                deep=True,
            )
        mismatch = _context_mismatch(chosen, negative)
        if mismatch is not None:
            _append_diagnostic(
                diagnostics,
                reason=mismatch[0],
                detail=mismatch[1],
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
        chosen_completion = _completion_messages(chosen)
        rejected_completion = _completion_messages(negative)
        if _canonical(chosen_completion) == _canonical(rejected_completion):
            _append_diagnostic(
                diagnostics,
                reason="identical_responses",
                detail="chosen and rejected evidence point to identical output",
                trajectories=[chosen, negative],
            )
            continue
        margin, margin_evidence = _preference_margin(
            chosen,
            negative,
            min_quality_margin=min_quality_margin,
        )
        if margin is None:
            reason = str(margin_evidence.get("reason") or "invalid_preference_margin")
            _append_diagnostic(
                diagnostics,
                reason=reason,
                detail=_canonical(margin_evidence),
                trajectories=[chosen, negative],
            )
            continue
        pair_id = "dpo_" + _hash(
            "|".join([*key, chosen.trajectory_id, negative.trajectory_id]), 32
        )
        pairs.append(
            DPOPreferencePair(
                pair_id=pair_id,
                prompt=_prompt_messages(chosen),
                chosen=chosen_completion,
                rejected=rejected_completion,
                tools=chosen.tools,
                images=_image_paths(chosen),
                context_hash=chosen.context_hash,
                leakage_group=chosen.scenario_family_id or chosen.task_id,
                task_id=chosen.task_id,
                stage=chosen.stage,
                agent_role=chosen.agent_role,
                task_type=chosen.task_type,
                chosen_trajectory_id=chosen.trajectory_id,
                rejected_trajectory_id=negative.trajectory_id,
                chosen_evidence=chosen.evidence,
                rejected_evidence=negative_evidence or negative.evidence,
                chosen_outcome=chosen.outcome,
                rejected_outcome=negative.outcome,
                quality_margin=margin,
                spatial_context=chosen.spatial_context,
                provenance={
                    "chosen_run_id": chosen.run_id,
                    "rejected_run_id": negative.run_id,
                    "chosen_source_refs": chosen.source_refs,
                    "rejected_source_refs": negative.source_refs,
                    "chosen_model_id": chosen.model_id,
                    "rejected_model_id": negative.model_id,
                    "chosen_config_hash": chosen.config_hash,
                    "rejected_config_hash": negative.config_hash,
                    "exact_prompt_match": True,
                    "exact_tool_schema_match": True,
                    "exact_media_context_match": True,
                    "exact_spatial_context_match": True,
                    "preference_basis": preference_basis,
                    "preference_order": "execution_then_hard_then_relation_then_visual",
                    "preference_margin_evidence": margin_evidence,
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
    assignments = {
        group: (
            "train"
            if index < train_count
            else "validation" if index < train_count + validation_count else "test"
        )
        for index, group in enumerate(groups)
    }
    splits = {"train": [], "validation": [], "test": []}
    for group, group_pairs in by_group.items():
        splits[assignments[group]].extend(group_pairs)
    return splits


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _vision_messages(
    messages: list[dict[str, Any]],
    *,
    image_count: int = 0,
) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    placeholders = 0
    for raw in messages:
        message = dict(raw)
        content = message.get("content", "")
        if isinstance(content, list):
            items = [dict(item) for item in content if isinstance(item, dict)]
        elif content in (None, "") and message.get("tool_calls"):
            items = []
        else:
            items = [{"type": "text", "text": str(content)}]
        placeholders += sum(item.get("type") == "image" for item in items)
        message["content"] = items
        normalized.append(message)
    missing = max(0, image_count - placeholders)
    if missing:
        target = next(
            (
                message
                for message in reversed(normalized)
                if message.get("role") == "user"
            ),
            None,
        )
        if target is None:
            raise ValueError("multimodal preference prompt has no user message")
        target["content"] = [
            *[{"type": "image"} for _ in range(missing)],
            *target.get("content", []),
        ]
    if placeholders > image_count:
        raise ValueError("prompt has more image placeholders than packaged images")
    return normalized


def _materialize_pair_media(
    pair: DPOPreferencePair,
    *,
    output_dir: Path,
) -> tuple[DPOPreferencePair | None, str]:
    if not pair.images:
        return pair, ""
    relative_images: list[str] = []
    for source_value in pair.images:
        source = Path(source_value)
        if not source.is_file():
            return None, f"missing visual input: {source}"
        digest = _file_sha256(source)
        suffix = source.suffix.lower() or ".bin"
        destination = output_dir / "images" / f"{digest}{suffix}"
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            shutil.copy2(source, destination)
        relative_images.append(destination.relative_to(output_dir).as_posix())
    try:
        prompt = _vision_messages(pair.prompt, image_count=len(relative_images))
        chosen = _vision_messages(pair.chosen)
        rejected = _vision_messages(pair.rejected)
    except ValueError as exc:
        return None, str(exc)
    return (
        pair.model_copy(
            update={
                "images": relative_images,
                "prompt": prompt,
                "chosen": chosen,
                "rejected": rejected,
            },
            deep=True,
        ),
        "",
    )


def validate_dataset_dir(dataset_dir: Path) -> dict[str, Any]:
    """Validate schemas, preference proofs, media, and split isolation."""

    dataset_dir = Path(dataset_dir)
    errors: list[str] = []
    split_groups: dict[str, set[str]] = {}
    split_counts: dict[str, int] = {}
    pair_ids: set[str] = set()
    role_counts: Counter[str] = Counter()
    task_type_counts: Counter[str] = Counter()
    multimodal_count = 0
    tool_pair_count = 0
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
            proofs = (
                "exact_prompt_match",
                "exact_tool_schema_match",
                "exact_media_context_match",
                "exact_spatial_context_match",
            )
            for proof in proofs:
                if (
                    pair.schema_version.endswith(".v2")
                    and pair.provenance.get(proof) is not True
                ):
                    errors.append(f"pair does not prove {proof}: {pair.pair_id}")
            if pair.provenance.get("rejected_response_is_observed") is not True:
                errors.append(
                    f"pair does not prove rejected response provenance: {pair.pair_id}"
                )
            if pair.agent_role == "critic" and not (
                pair.chosen_outcome.causal_link_verified
                and pair.rejected_outcome.causal_link_verified
            ):
                errors.append(f"critic pair lacks causal evidence: {pair.pair_id}")
            for image in pair.images:
                image_path = Path(image)
                resolved = (dataset_dir / image_path).resolve()
                try:
                    resolved.relative_to(dataset_dir.resolve())
                except ValueError:
                    errors.append(f"image escapes dataset directory: {pair.pair_id}")
                    continue
                if image_path.is_absolute() or not resolved.is_file():
                    errors.append(f"missing packaged image {image!r}: {pair.pair_id}")
            groups.add(pair.leakage_group)
            role_counts[pair.agent_role] += 1
            task_type_counts[pair.task_type] += 1
            multimodal_count += int(bool(pair.images))
            tool_pair_count += int(bool(pair.tools))
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
        "schema_version": "sceneexpert.dpo_validation.v2",
        "validated_at": _utc_now(),
        "valid": not errors,
        "errors": errors,
        "split_counts": split_counts,
        "split_leakage_groups": {
            split: sorted(groups) for split, groups in split_groups.items()
        },
        "pair_count": len(pair_ids),
        "pairs_by_role": dict(sorted(role_counts.items())),
        "pairs_by_task_type": dict(sorted(task_type_counts.items())),
        "multimodal_pair_count": multimodal_count,
        "tool_pair_count": tool_pair_count,
    }


def export_dpo_dataset(
    *,
    trajectory_sources: Iterable[Path],
    output_dir: Path,
    validation_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
    min_quality_margin: float = 0.05,
    include_task_types: set[str] | frozenset[str] | None = DEFAULT_TRAINING_TASK_TYPES,
) -> dict[str, Any]:
    """Materialize a portable, auditable tool/VLM DPO dataset package."""

    trajectory_sources = [Path(path) for path in trajectory_sources]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    trajectories, load_diagnostics = load_trajectories(trajectory_sources)
    pairs, pair_diagnostics = build_preference_pairs(
        trajectories,
        min_quality_margin=min_quality_margin,
        include_task_types=include_task_types,
    )
    diagnostics = load_diagnostics + pair_diagnostics
    portable_pairs: list[DPOPreferencePair] = []
    for pair in pairs:
        portable, error = _materialize_pair_media(pair, output_dir=output_dir)
        if portable is None:
            diagnostics.append(
                {
                    "reason": "media_materialization_failed",
                    "detail": error,
                    "trajectory_ids": [
                        pair.chosen_trajectory_id,
                        pair.rejected_trajectory_id,
                    ],
                    "context_hashes": [pair.context_hash],
                    "task_ids": [pair.task_id],
                    "task_types": [pair.task_type],
                }
            )
            continue
        portable_pairs.append(portable)
    pairs = portable_pairs
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
    task_type_counts = Counter(pair.task_type for pair in pairs)
    evidence_counts = Counter(pair.chosen_evidence.kind for pair in pairs)
    preference_basis_counts = Counter(
        str(pair.provenance.get("preference_basis") or "unknown") for pair in pairs
    )
    stats = {
        "schema_version": "sceneexpert.dpo_stats.v2",
        "generated_at": _utc_now(),
        "trajectory_count": len(trajectories),
        "eligible_pair_count": len(pairs),
        "diagnostic_count": len(diagnostics),
        "diagnostic_reasons": dict(sorted(reason_counts.items())),
        "pairs_by_stage": dict(sorted(stage_counts.items())),
        "pairs_by_role": dict(sorted(role_counts.items())),
        "pairs_by_task_type": dict(sorted(task_type_counts.items())),
        "pairs_by_evidence": dict(sorted(evidence_counts.items())),
        "pairs_by_preference_basis": dict(sorted(preference_basis_counts.items())),
        "tool_pair_count": sum(bool(pair.tools) for pair in pairs),
        "multimodal_pair_count": sum(bool(pair.images) for pair in pairs),
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
        "serialized_chars": {
            "max_prompt": max(
                (len(_canonical(pair.prompt)) for pair in pairs), default=0
            ),
            "max_chosen": max(
                (len(_canonical(pair.chosen)) for pair in pairs), default=0
            ),
            "max_rejected": max(
                (len(_canonical(pair.rejected)) for pair in pairs), default=0
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
        "schema_version": "sceneexpert.dpo_dataset_manifest.v2",
        "generated_at": _utc_now(),
        "source_paths": [str(path) for path in trajectory_sources],
        "seed": seed,
        "validation_ratio": validation_ratio,
        "test_ratio": test_ratio,
        "min_quality_margin": min_quality_margin,
        "included_task_types": sorted(include_task_types or []),
        "pairing_policy": {
            "same_context_required": True,
            "same_tool_schema_required": True,
            "same_visual_input_required": True,
            "same_spatial_context_required": True,
            "authoritative_evidence_required": True,
            "critic_causal_evidence_required": True,
            "synthetic_rejected_responses_allowed": False,
            "relative_critic_ranking_allowed": True,
            "truncated_payloads_allowed": False,
            "one_best_worst_pair_per_context": True,
            "preference_order": (
                "execution_then_hard_then_relation_then_deterministic_then_visual"
            ),
        },
        "files": {
            "train": "train.jsonl",
            "validation": "validation.jsonl",
            "test": "test.jsonl",
            "all": "all.jsonl",
            "images": "images/",
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
