"""Prompt-grounded label calibration for detached SceneExpert candidate scoring.

This never changes the live scene, designer input or cached online contract.
The existing deterministic parser and Main evaluators remain the semantic source.
"""

from __future__ import annotations

import re
from copy import deepcopy
from typing import Any

from scenesmith.scene_expert.slow_memory.paired import digest

CALIBRATION_PROTOCOL = "sceneexpert.candidate_contract_calibration.v2"


def _selector_key(selector: dict[str, Any] | None) -> tuple[Any, ...]:
    value = selector or {}
    return (
        value.get("category"),
        value.get("count", 1),
        value.get("cohort") if value.get("cohort") not in (None, "", "all") else "",
        value.get("stage") or "",
        value.get("support_target") or "",
    )


def _endpoints(row: dict[str, Any]) -> tuple[Any, ...]:
    return (_selector_key(row.get("subjects")), _selector_key(row.get("targets")))


def _grounded_constraints(prompt: str) -> list[dict[str, Any]]:
    """Parse explicit relations, including the positive noun phrase floor plants."""
    from scenesmith.scenebenchmark_critic.intent_contract import build_intent_contract

    rows = list(build_intent_contract(prompt).get("constraints") or [])
    # Do not resolve mixed/contradictory support instructions by dropping one.
    if any(
        row.get("relation") in {"on_top_of", "object_on_support"}
        and (row.get("subjects") or {}).get("category") == "plant"
        and (row.get("targets") or {}).get("category") != "floor"
        for row in rows
    ):
        return rows
    for sentence in re.split(r"[.!?;]", prompt):
        if not re.search(r"\bfloor[ -]+plants?\b", sentence, re.IGNORECASE):
            continue
        if re.search(r"\b(?:no|not|without|avoid)\b", sentence, re.IGNORECASE):
            continue
        normalized = re.sub(
            r"\bfloor[ -]+(plants?)\b",
            r"\1 on the floor",
            sentence,
            flags=re.IGNORECASE,
        )
        for row in build_intent_contract(normalized).get("constraints") or []:
            if (
                row.get("relation") == "on_top_of"
                and (row.get("subjects") or {}).get("category") == "plant"
                and (row.get("targets") or {}).get("category") == "floor"
                and not any(
                    old.get("relation") == "on_top_of"
                    and _endpoints(old) == _endpoints(row)
                    for old in rows
                )
            ):
                row = deepcopy(row)
                row["evidence_span"] = sentence.strip()
                row["inference_reason"] = (
                    "Explicit floor-plant noun phrase normalization"
                )
                rows.append(row)
    return rows


def _covered_edge_rows(
    rows: list[dict[str, Any]], candidate: dict[str, Any]
) -> list[dict[str, Any]]:
    """Find exact topological parts covered by a complete prompt-derived layout."""
    expected = {group["edge_class"]: group for group in candidate.get("groups") or []}
    covered: list[dict[str, Any]] = []
    for row in rows:
        groups = row.get("groups") or []
        if (
            row.get("relation") != "edge_distribution"
            or (row.get("subjects") or {}).get("category")
            != (candidate.get("subjects") or {}).get("category")
            or _selector_key(row.get("targets"))
            != _selector_key(candidate.get("targets"))
            or row.get("edge_frame") != candidate.get("edge_frame")
            or row.get("orientation")
            not in {"unconstrained", candidate.get("orientation")}
            or not groups
            or len({group.get("edge_class") for group in groups}) != len(groups)
            or sum(sum(group.get("counts_per_edge") or []) for group in groups)
            != int((row.get("subjects") or {}).get("count") or 0)
        ):
            continue
        if all(
            group.get("edge_class") in expected
            and sorted(group.get("counts_per_edge") or [])
            == sorted(expected[group["edge_class"]].get("counts_per_edge") or [])
            and group.get("spacing")
            in {"unconstrained", expected[group["edge_class"]].get("spacing")}
            for group in groups
        ):
            covered.append(row)
    # Require evidence for the whole topology, never select a geometric subset.
    if {group["edge_class"] for row in covered for group in row["groups"]} != set(
        expected
    ):
        return []
    return covered


def _covered_short_side_centering(
    row: dict[str, Any], candidate: dict[str, Any]
) -> bool:
    """Recognize inferred wall-centering errors already enforced by a table slot."""
    return bool(
        row.get("source") == "model_inferred"
        and row.get("relation") == "centered_on_wall"
        and (row.get("subjects") or {}).get("category")
        == (candidate.get("subjects") or {}).get("category")
        and (row.get("subjects") or {}).get("count") == 1
        and (row.get("subjects") or {}).get("cohort") in (None, "", "all")
        and _selector_key(row.get("targets")) == _selector_key(candidate.get("targets"))
        and re.search(
            r"\bcentered\b[^.]{0,80}\bshort (?:side|edge)\b",
            str(row.get("inference_reason") or ""),
            re.IGNORECASE,
        )
        and any(
            group.get("edge_class") == "short"
            and sorted(group.get("counts_per_edge") or []) == [0, 1]
            and group.get("spacing") == "equal_segments"
            for group in candidate.get("groups") or []
        )
    )


def calibrate_contract(case_pack: dict[str, Any]) -> None:
    """Reconcile narrowly supported contract errors against the original prompt."""
    contract = deepcopy(case_pack.get("intent_contract") or {})
    calibration: dict[str, Any] = {
        "schema_version": CALIBRATION_PROTOCOL,
        "original_contract_sha256": digest(contract),
        "changes": [],
        "superseded_asset_checks": [],
    }
    case_pack["sceneexpert_contract_calibration"] = calibration
    prompt = str(case_pack.get("original_task_instruction") or "").strip()
    if not prompt:
        calibration["calibrated_contract_sha256"] = digest(contract)
        return

    grounded = _grounded_constraints(prompt)
    rows = list(contract.get("constraints") or [])
    for candidate in grounded:
        relation = candidate.get("relation")
        endpoints = _endpoints(candidate)
        if (
            relation == "on_top_of"
            and (candidate.get("targets") or {}).get("category") == "floor"
        ):
            if not any(
                row.get("relation") == relation and _endpoints(row) == endpoints
                for row in rows
            ):
                rows.append(deepcopy(candidate))
                calibration["changes"].append(
                    {"kind": "restore_explicit_floor_support", "constraint": candidate}
                )
        elif (
            relation == "aligned_with"
            and "tucked" in str(candidate.get("evidence_span") or "").lower()
        ):
            # Preserve a separately grounded front/rear requirement for the same
            # endpoints rather than resolving an ambiguous prompt by preference.
            if any(
                row.get("relation") in {"in_front_of", "behind"}
                and _endpoints(row) == endpoints
                for row in grounded
            ):
                continue
            replaced = [
                row
                for row in rows
                if row.get("relation") in {"in_front_of", "behind"}
                and _endpoints(row) == endpoints
            ]
            if replaced:
                rows = [row for row in rows if row not in replaced]
                if not any(
                    row.get("relation") == relation and _endpoints(row) == endpoints
                    for row in rows
                ):
                    rows.append(deepcopy(candidate))
                calibration["changes"].append(
                    {
                        "kind": "restore_tucked_seat_alignment",
                        "removed_constraints": replaced,
                        "constraint": candidate,
                    }
                )
        elif relation == "edge_distribution":
            # A complete row plus an inferred partial duplicate is just as
            # contradictory as separate cohorts. Restore the full prompt rule,
            # including its centering/facing checks, without looking at poses.
            parts = _covered_edge_rows(rows, candidate)
            if not parts:
                continue
            parts += [
                row for row in rows if _covered_short_side_centering(row, candidate)
            ]
            if parts == [candidate]:
                continue
            rows = [row for row in rows if row not in parts]
            rows.append(deepcopy(candidate))
            calibration["changes"].append(
                {
                    "kind": "restore_complete_edge_partition",
                    "removed_constraints": parts,
                    "constraint": candidate,
                }
            )
    contract["constraints"] = rows
    case_pack["intent_contract"] = contract
    calibration["calibrated_contract_sha256"] = digest(contract)


def calibrate_asset_checks(case_pack: dict[str, Any]) -> None:
    """Keep generic support priors as auxiliary when explicit floor support owns it."""
    prompt = str(case_pack.get("original_task_instruction") or "").strip()
    if not prompt:
        return
    from scenesmith.scenebenchmark_critic.intent_contract import (
        augment_contract_checks,
        bound_ids,
    )

    # Main normally materializes these during evaluation. Materialize them
    # idempotently now so a replacement core check must exist before demotion.
    augment_contract_checks(case_pack)
    objects = (case_pack.get("scene_geometry") or {}).get("objects") or []
    authorized: dict[str, str] = {}
    for row in _grounded_constraints(prompt):
        if (
            row.get("relation") != "on_top_of"
            or (row.get("targets") or {}).get("category") != "floor"
        ):
            continue
        if not bound_ids(row.get("targets"), objects):
            continue
        for subject in bound_ids(row.get("subjects"), objects):
            authorized[subject] = row.get("constraint_id", "")
    checks = case_pack.get("checks") or []
    enforced_floor_subjects = {
        check.get("subject_id")
        for check in checks
        if check.get("relation_type") == "object_on_floor"
        and (check.get("scoring_tier") or "core") == "core"
    }
    for check in checks:
        subject = check.get("subject_id")
        if (
            subject in authorized
            and subject in enforced_floor_subjects
            and check.get("check_source") == "asset_explicit_target_relation"
            and check.get("relation_type") == "object_on_support"
        ):
            check["scoring_tier"] = "auxiliary"
            check.setdefault("evidence", {})["superseded_by_prompt_floor_support"] = (
                authorized[subject]
            )
            case_pack["sceneexpert_contract_calibration"][
                "superseded_asset_checks"
            ].append(check["check_id"])
