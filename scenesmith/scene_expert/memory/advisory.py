"""Observe memory-specific spatial outcomes without changing native scoring."""

from __future__ import annotations

from types import SimpleNamespace

from scenesmith.scene_expert.memory.evidence import resolve_constraint_evidence
from scenesmith.scene_expert.memory.placement import (
    check_object_ids,
    object_role,
    pair_measurements,
    valid_episode,
    valid_state,
)
from scenesmith.scene_expert.memory.schemas import PlacementEpisode
from scenesmith.scene_expert.memory.placement_scope import source_metric
from scenesmith.scene_expert.memory.skill_policy import _category_compatible, _relation


def advice_check_reasons(source: dict, choice: dict, context: dict) -> list[str]:
    """Bind checks to exact selected source episodes; never accept invented metrics."""
    experience = source.get("placement_experience") or {}
    if not experience:
        return (
            ["advice_checks_without_spatial_experience"]
            if choice.get("advice_checks")
            else []
        )
    reasons = []
    episodes = {}
    for raw in experience.get("episodes", []):
        episode = PlacementEpisode.model_validate(raw)
        if not valid_episode(episode):
            reasons.append("invalid_placement_episode")
        episodes[episode.episode_id] = episode
    indices = choice.get("source_relation_indices")
    relations = source.get("spatial_relations", [])
    scoped = (
        relations
        if indices is None
        else [relations[i] for i in indices if 0 <= i < len(relations)]
    )
    allowed = {row.get("evidence_ref") for row in scoped}
    steps = experience.get("method_steps", [])
    selected = choice.get("source_method_step_indices")
    selected = list(range(len(steps))) if selected is None else selected
    selected_steps = [
        steps[i] for i in selected if type(i) is int and 0 <= i < len(steps)
    ]
    # Source-stage context is not an invitation to measure unrelated geometry.
    if steps:
        allowed &= {i for s in selected_steps for i in s.get("episode_ids", [])}
    expected = {
        (r["episode_id"], r["metric"])
        for s in selected_steps
        for r in s.get("relation_bindings", [])
        if r["metric"] != "pair_observation"
    }
    checks = choice.get("advice_checks") or []
    if not checks and (allowed or not selected_steps):
        reasons.append("missing_advice_observation")
    if expected - {(c.get("source_episode_id"), c.get("metric")) for c in checks}:
        reasons.append("missing_method_metric_observation")
    for check in checks:
        episode = episodes.get(check.get("source_episode_id"))
        if episode is None or episode.episode_id not in allowed:
            reasons.append("unknown_or_unselected_episode")
            continue
        if check.get("subject_role") != object_role(episode.subject) or check.get(
            "anchor_role"
        ) != object_role(episode.anchor):
            reasons.append("advice_role_mismatch")
        for role in (check.get("subject_role"), check.get("anchor_role")):
            if not any(
                b.get("source_role") == role for b in choice.get("bindings", [])
            ):
                reasons.append("missing_advice_role_binding")
        if check.get("metric") == "native_constraint":
            current = next(
                (
                    row
                    for row in context.get("hard_constraints", [])
                    if row.get("constraint_id") == check.get("constraint_id")
                ),
                None,
            )
            if not current or not any(
                _relation(
                    c["constraint"].get("relation")
                    or c["constraint"].get("relation_type")
                )
                == _relation(current.get("relation") or current.get("relation_type"))
                for c in episode.native_checks
            ):
                reasons.append("unmatched_native_advice_predicate")
        elif source_metric(episode, check.get("metric")) is None:
            reasons.append("unobserved_source_metric")
        elif check.get("constraint_id"):
            reasons.append("geometry_observation_cannot_borrow_constraint_pass")
    return list(dict.fromkeys(reasons))


def _bound_object(role: str, choice: dict, state: dict) -> dict | None:
    bindings = [b for b in choice.get("bindings", []) if b.get("source_role") == role]
    if len(bindings) != 1:
        return None
    binding = bindings[0]
    ids = binding.get("object_ids") or []
    rows = [
        row
        for row in state.get("objects", [])
        if (
            row.get("object_id") in ids
            if ids
            else _category_compatible(binding.get("current_role", ""), object_role(row))
        )
    ]
    # Do not select a convenient object out of multiple chairs/nightstands.
    return rows[0] if len(rows) == 1 and len(ids) <= 1 else None


def observe_advice(item: dict, stage: str, entry: dict) -> list[dict]:
    """Return measurements/unknowns, with native labels only for exact checks.

    Matching a source pose is never treated as quality or causal improvement.
    Geometry observations have no pass threshold; alternative layouts are valid.
    """
    source, choice = item.get("source") or {}, item.get("adaptation") or {}
    checks = choice.get("advice_checks") or []
    context = entry.get("relation_context") or {}
    reasons = advice_check_reasons(source, choice, context)
    if reasons:
        return (
            [{"status": "unknown", "reasons": reasons}]
            if checks or source.get("placement_experience")
            else []
        )
    episodes = {
        e["episode_id"]: e
        for e in (source.get("placement_experience") or {}).get("episodes", [])
    }
    post = entry.get("post_scene_state") or {}
    pre = (entry.get("injection") or {}).get("current_scene_state") or {}
    result = []
    if not checks and source.get("placement_experience"):
        return [
            {
                "status": "unknown",
                "reason": "critic_advice_only_no_geometric_verdict",
                "quality_gain": None,
                "causal_gain": None,
            }
        ]
    for check in checks:
        row = {**check, "status": "unknown", "quality_gain": None, "causal_gain": None}
        result.append(row)
        if not valid_state(post):
            row["reason"] = "missing_valid_post_state"
            continue
        subject = _bound_object(check["subject_role"], choice, post)
        anchor = _bound_object(check["anchor_role"], choice, post)
        if not subject or not anchor or subject["object_id"] == anchor["object_id"]:
            row["reason"] = "missing_or_ambiguous_object_binding"
            continue
        row["object_ids"] = [subject["object_id"], anchor["object_id"]]
        row["state_fingerprint"] = post["fingerprint"]
        episode = episodes[check["source_episode_id"]]
        metric = check["metric"]
        if metric == "native_constraint":
            constraint = next(
                c
                for c in context["hard_constraints"]
                if c.get("constraint_id") == check["constraint_id"]
            )
            status, evidence = resolve_constraint_evidence(
                constraint, {**entry, "stage": stage}
            )
            matched = [
                e
                for e in evidence
                if set(row["object_ids"]) <= check_object_ids(e.observations)[1]
                and e.metric in {c["check"]["metric"] for c in episode["native_checks"]}
                and bool(e.metric)
                and check_object_ids(e.observations)[0] == subject["object_id"]
                and e.label in {"pass", "fail"}
                and e.scoring_tier == "core"
                and e.evaluation_state not in {"unknown", "deferred", "not_applicable"}
            ]
            if matched and status in {"verified_pass", "verified_fail"}:
                row.update(
                    status=(
                        "verified_fail"
                        if any(e.label == "fail" for e in matched)
                        else "verified_pass"
                    ),
                    evidence=[e.model_dump(mode="json") for e in matched],
                )
            else:
                row["reason"] = "native_predicate_or_object_evidence_unavailable"
            continue
        measured = pair_measurements(subject, anchor)
        if metric == "bbox_center_distance_m":
            measured[metric] = source_metric(
                SimpleNamespace(subject=subject, anchor=anchor), metric
            )
        if metric not in measured or measured[metric] is None:
            row["reason"] = "metric_unavailable"
            continue
        row.update(
            status="measured",
            source_value=source_metric(
                PlacementEpisode.model_validate(episode), metric
            ),
            after_value=measured[metric],
            before_value=None,
            before_checkpoint="pre_stage",
            delta=None,
            interpretation="Observation only; not navigable clearance, semantic front, or an improvement label.",
        )
        if valid_state(pre):
            before_objects = {o["object_id"]: o for o in pre["objects"]}
            if (
                subject["object_id"] in before_objects
                and anchor["object_id"] in before_objects
            ):
                before = pair_measurements(
                    before_objects[subject["object_id"]],
                    before_objects[anchor["object_id"]],
                ).get(metric)
                if metric == "bbox_center_distance_m":
                    before = source_metric(
                        SimpleNamespace(
                            subject=before_objects[subject["object_id"]],
                            anchor=before_objects[anchor["object_id"]],
                        ),
                        metric,
                    )
                row["before_value"] = before
                if before is not None:
                    after = measured[metric]
                    delta = (
                        [round(a - b, 4) for a, b in zip(after, before)]
                        if isinstance(after, list)
                        else round(after - before, 4)
                    )
                    row["delta"] = (
                        (delta + 180) % 360 - 180
                        if metric == "relative_yaw_deg"
                        else delta
                    )
    return result
