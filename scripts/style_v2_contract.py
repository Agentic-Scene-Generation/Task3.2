"""Validation helpers for the hierarchical asset-style v2 overlay."""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "asset_style@2.0"
ONTOLOGY_VERSION = "bonn_furniture_styles_hierarchical@1.0"
STYLE_STATUSES = {"source_metadata", "visual_reviewed", "unlabeled", "needs_review"}
SOURCE_EVIDENCE = "source_metadata"
VISUAL_EVIDENCE = "multiview_visual_review"


def load_style_ontology(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema_version") != ONTOLOGY_VERSION:
        raise ValueError(f"unsupported style ontology: {value.get('schema_version')}")
    return value


def _valid_sha256(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    return all(char in "0123456789abcdef" for char in value.lower())


def validate_style_annotation(annotation: dict[str, Any], ontology: dict[str, Any]) -> list[str]:
    """Return stable error codes; an empty list means the record is valid."""
    errors: list[str] = []
    if annotation.get("schema_version") != SCHEMA_VERSION:
        errors.append("invalid_schema_version")

    status = annotation.get("style_status")
    if status not in STYLE_STATUSES:
        errors.append("invalid_style_status")

    styles = annotation.get("styles")
    if not isinstance(styles, list):
        errors.append("styles_not_list")
        styles = []
    if len(styles) > 3:
        errors.append("too_many_style_paths")
    if status == "unlabeled" and styles:
        errors.append("unlabeled_has_styles")
    if status in {"source_metadata", "visual_reviewed"} and not styles:
        errors.append("labeled_status_has_no_styles")

    level_1 = ontology.get("level_1", {})
    level_2 = ontology.get("level_2", {})
    seen: set[tuple[Any, Any]] = set()
    for item in styles:
        if not isinstance(item, dict):
            errors.append("style_path_not_object")
            continue
        first = item.get("level_1_id")
        second = item.get("level_2_id")
        path = (first, second)
        if path in seen:
            errors.append("duplicate_style_path")
        seen.add(path)
        if first not in level_1:
            errors.append("unknown_level_1_id")
        if second is not None:
            concept = level_2.get(second)
            if concept is None:
                errors.append("unknown_level_2_id")
            elif concept.get("parent_level_1_id") != first:
                errors.append("level_2_parent_mismatch")

        evidence = item.get("evidence_type")
        if status == "visual_reviewed" or evidence == VISUAL_EVIDENCE:
            if second is not None:
                errors.append("visual_style_must_not_have_level_2")
            if evidence != VISUAL_EVIDENCE:
                errors.append("invalid_visual_evidence_type")
            if not item.get("visual_evidence_id"):
                errors.append("missing_visual_evidence_id")
        if status == "source_metadata" and evidence != SOURCE_EVIDENCE:
            errors.append("invalid_source_evidence_type")

        confidence = item.get("confidence")
        if not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1:
            errors.append("invalid_style_confidence")

    review = annotation.get("visual_review")
    if status == "visual_reviewed":
        valid_review = (
            isinstance(review, dict)
            and bool(review.get("evidence_id"))
            and bool(review.get("model_id"))
            and _valid_sha256(review.get("prompt_sha256"))
            and review.get("adjudication_status") == "agreed"
            and all(
                item.get("visual_evidence_id") == review.get("evidence_id")
                for item in styles
                if isinstance(item, dict)
            )
        )
        if not valid_review:
            errors.append("invalid_visual_review")
    elif status == "unlabeled":
        valid_insufficient_review = (
            isinstance(review, dict)
            and bool(review.get("evidence_id"))
            and bool(review.get("model_id"))
            and _valid_sha256(review.get("prompt_sha256"))
            and review.get("adjudication_status") == "agreed_insufficient"
        )
        if not valid_insufficient_review:
            errors.append("invalid_unlabeled_visual_review")
    elif review is not None and status != "needs_review":
        errors.append("unexpected_visual_review")

    if not isinstance(annotation.get("source_style_labels", []), list):
        errors.append("source_style_labels_not_list")
    if not isinstance(annotation.get("review_flags", []), list):
        errors.append("review_flags_not_list")
    provenance = annotation.get("provenance")
    if not isinstance(provenance, dict) or provenance.get("ontology_id") != ONTOLOGY_VERSION:
        errors.append("invalid_provenance")
    return list(dict.fromkeys(errors))


def normalize_style_annotation(
    annotation: dict[str, Any], ontology: dict[str, Any]
) -> dict[str, Any]:
    """Return a deterministic deep copy after validating the input record."""
    errors = validate_style_annotation(annotation, ontology)
    if errors:
        raise ValueError("invalid asset_style@2.0 record: " + ", ".join(errors))
    value = copy.deepcopy(annotation)
    value["styles"] = sorted(
        value.get("styles", []),
        key=lambda item: (item["level_1_id"], item.get("level_2_id") or ""),
    )
    value["source_style_labels"] = sorted(dict.fromkeys(value.get("source_style_labels", [])))
    value["review_flags"] = sorted(dict.fromkeys(value.get("review_flags", [])))
    return value
