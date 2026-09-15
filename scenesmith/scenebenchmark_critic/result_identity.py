"""Canonical semantic identity for critic checks and evaluator results."""

from __future__ import annotations

import hashlib
import json

from typing import Any, Iterable


_LABEL_RANK = {"pass": 0, "unknown": 1, "degraded": 2, "fail": 3}


def semantic_signature(payload: dict[str, Any]) -> tuple[Any, ...]:
    """Return a stable signature without producer-specific IDs."""
    evidence = (
        payload.get("evidence") if isinstance(payload.get("evidence"), dict) else {}
    )
    intent = (
        evidence.get("intent_constraint")
        if isinstance(evidence.get("intent_constraint"), dict)
        else {}
    )
    dependency = (
        evidence.get("dependency")
        if isinstance(evidence.get("dependency"), dict)
        else {}
    )
    diagnostics = (
        payload.get("diagnostics")
        if isinstance(payload.get("diagnostics"), dict)
        else {}
    )
    relation = (
        str(
            payload.get("relation_type")
            or payload.get("relation")
            or intent.get("relation")
            or ""
        )
        .strip()
        .lower()
    )
    metric = str(payload.get("metric") or "").strip().lower()
    subject = str(
        payload.get("subject_id")
        or payload.get("primary_object")
        or payload.get("subject")
        or ""
    )
    targets = payload.get("target_ids")
    if targets is None:
        targets = (
            payload.get("related_objects")
            or payload.get("selected_related_objects")
            or []
        )
    normalized_targets = tuple(sorted(str(item) for item in targets if str(item)))
    params = _semantic_parameters(payload, intent, dependency, diagnostics)
    return (
        metric,
        relation,
        subject,
        normalized_targets,
        str(payload.get("cohort") or intent.get("cohort") or ""),
        str(payload.get("stage") or intent.get("stage") or ""),
        str(payload.get("scoring_tier") or "core"),
        str(payload.get("authority") or ""),
        params,
    )


def canonical_id(signature: tuple[Any, ...]) -> str:
    encoded = json.dumps(
        signature,
        ensure_ascii=True,
        sort_keys=True,
        default=str,
        separators=(",", ":"),
    )
    return "semantic_" + hashlib.sha1(encoded.encode("utf-8")).hexdigest()[:14]


def deduplicate_checks(checks: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: dict[tuple[Any, ...], dict[str, Any]] = {}
    for original in checks:
        if not isinstance(original, dict):
            continue
        check = dict(original)
        signature = semantic_signature(check)
        canonical = canonical_id(signature)
        producer_id = str(check.get("check_id") or "")
        check.setdefault("canonical_check_id", canonical)
        check.setdefault("producer_check_ids", [producer_id] if producer_id else [])
        previous = merged.get(signature)
        if previous is None:
            merged[signature] = check
            continue
        _merge_provenance(previous, check)
    return list(merged.values())


def deduplicate_results(results: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: dict[tuple[Any, ...], dict[str, Any]] = {}
    for original in results:
        if not isinstance(original, dict):
            continue
        result = dict(original)
        signature = semantic_signature(result)
        canonical = canonical_id(signature)
        producer_id = str(result.get("check_id") or "")
        result.setdefault("canonical_check_id", canonical)
        result.setdefault("producer_check_ids", [producer_id] if producer_id else [])
        previous = merged.get(signature)
        if previous is None:
            merged[signature] = result
            continue
        _merge_result(previous, result)
    return list(merged.values())


def _semantic_parameters(
    payload: dict[str, Any],
    intent: dict[str, Any],
    dependency: dict[str, Any],
    diagnostics: dict[str, Any],
) -> tuple[tuple[str, str], ...]:
    values: dict[str, Any] = {}
    for source in (payload, intent, dependency, diagnostics):
        for key in (
            "affordance",
            "relation_type",
            "threshold",
            "max_gap_m",
            "max_angle_deg",
            "direction",
            "dependency_binding",
            "requires_external_adjacency",
            "support_target",
        ):
            if key in source and source[key] not in (None, "", []):
                values[key] = source[key]
    return tuple(
        sorted((str(key), _normalize_value(value)) for key, value in values.items())
    )


def _normalize_value(value: Any) -> str:
    if isinstance(value, (dict, list, tuple, set)):
        return json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            default=str,
            separators=(",", ":"),
        )
    return str(value)


def _merge_provenance(target: dict[str, Any], source: dict[str, Any]) -> None:
    producer_ids = list(target.get("producer_check_ids") or [])
    producer_ids.extend(source.get("producer_check_ids") or [])
    producer_ids.extend(
        value for value in (target.get("check_id"), source.get("check_id")) if value
    )
    target["producer_check_ids"] = sorted(
        {str(value) for value in producer_ids if str(value)}
    )

    target_evidence = target.get("evidence")
    source_evidence = source.get("evidence")
    evidence_rows: list[dict[str, Any]] = []
    for evidence in (target_evidence, source_evidence):
        if not isinstance(evidence, dict):
            continue
        constraint = evidence.get("intent_constraint")
        if isinstance(constraint, dict):
            evidence_rows.append(constraint)
        evidence_rows.extend(
            row
            for row in evidence.get("intent_constraints") or []
            if isinstance(row, dict)
        )

    source_constraint_ids = list(target.get("source_constraint_ids") or [])
    source_constraint_ids.extend(source.get("source_constraint_ids") or [])
    source_constraint_ids.extend(
        str(row.get("constraint_id") or "")
        for row in evidence_rows
        if row.get("constraint_id")
    )
    target["source_constraint_ids"] = sorted(
        {str(value) for value in source_constraint_ids if str(value)}
    )

    evidence_refs = list(target.get("evidence_refs") or [])
    evidence_refs.extend(source.get("evidence_refs") or [])
    target["evidence_refs"] = sorted(
        {str(value) for value in evidence_refs if str(value)}
    )

    if isinstance(target_evidence, dict) and evidence_rows:
        target_evidence["intent_constraints"] = _unique_json_rows(evidence_rows)


def _merge_result(target: dict[str, Any], source: dict[str, Any]) -> None:
    _merge_provenance(target, source)
    labels = [
        str(target.get("label") or "unknown"),
        str(source.get("label") or "unknown"),
    ]
    target["label"] = max(labels, key=lambda label: _LABEL_RANK.get(label, 1))
    if target.get("scoring_tier") != "core" and source.get("scoring_tier") == "core":
        target["scoring_tier"] = "core"
    if labels[0] != labels[1]:
        diagnostics = target.setdefault("diagnostics", {})
        disagreements = list(diagnostics.get("label_disagreements") or [])
        disagreements.extend(labels)
        diagnostics["label_disagreements"] = sorted(set(disagreements))
    blocking = set(str(item) for item in target.get("blocking_objects") or [])
    blocking.update(str(item) for item in source.get("blocking_objects") or [])
    target["blocking_objects"] = sorted(item for item in blocking if item)


def _unique_json_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    result: list[dict[str, Any]] = []
    for row in rows:
        encoded = json.dumps(
            row, ensure_ascii=True, sort_keys=True, default=str, separators=(",", ":")
        )
        if encoded not in seen:
            seen.add(encoded)
            result.append(row)
    return result
