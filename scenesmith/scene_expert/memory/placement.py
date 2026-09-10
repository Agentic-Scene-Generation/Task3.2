"""Read-only spatial experience extraction from bounded scene/audit evidence.

This module neither places assets nor judges native scene quality. Distances
between AABBs are explicitly NOT navigable clearance. Transform yaw is NOT an
asset's semantic front. Incomplete evidence never manufactures a repair story.
"""

from __future__ import annotations

import json
import math
import re

from pathlib import Path
from typing import Any

from scenesmith.scene_expert.evaluation_costs import timestamp
from scenesmith.scene_expert.evaluation_io import (
    local_reference,
    read_object,
    read_rows,
)
from scenesmith.scene_expert.memory.evidence import resolve_constraint_evidence
from scenesmith.scene_expert.memory.schemas import PlacementEpisode, PlacementExperience
from scenesmith.scene_expert.memory.state import state_hash


def valid_state(state: dict) -> bool:
    """Reject missing, mutated, partial-error or ambiguous object observations."""
    ids = [row.get("object_id") for row in state.get("objects", [])]
    return bool(
        state.get("fingerprint")
        and not state.get("observation_error")
        and all(ids)
        and len(ids) == len(set(ids))
        and state["fingerprint"]
        == state_hash(
            {key: value for key, value in state.items() if key != "fingerprint"}
        )
    )


def _vector(row: dict, key: str) -> list[float] | None:
    value = row.get(key)
    if not isinstance(value, list) or len(value) != 3:
        return None
    if not all(type(v) in (int, float) and math.isfinite(v) for v in value):
        return None
    return value


def pair_measurements(subject: dict, anchor: dict) -> dict[str, Any]:
    """Measure a directed pair in the anchor transform frame, not semantic axes."""
    result: dict[str, Any] = {}
    p, q = _vector(subject, "translation"), _vector(anchor, "translation")
    yaw, anchor_yaw = subject.get("yaw_deg"), anchor.get("yaw_deg")
    finite = lambda v: type(v) in (int, float) and math.isfinite(v)
    if p and q and finite(anchor_yaw):
        dx, dy, dz = [p[i] - q[i] for i in range(3)]
        angle = math.radians(anchor_yaw)
        result["anchor_local_offset_m"] = [
            round(v, 4)
            for v in (
                dx * math.cos(angle) + dy * math.sin(angle),
                -dx * math.sin(angle) + dy * math.cos(angle),
                dz,
            )
        ]
    if finite(yaw) and finite(anchor_yaw):
        result["relative_yaw_deg"] = round((yaw - anchor_yaw + 180) % 360 - 180, 3)
    lo1, hi1 = _vector(subject, "bbox_min"), _vector(subject, "bbox_max")
    lo2, hi2 = _vector(anchor, "bbox_min"), _vector(anchor, "bbox_max")
    if all(v is not None for v in (lo1, hi1, lo2, hi2)) and all(
        lo1[i] <= hi1[i] and lo2[i] <= hi2[i] for i in range(3)
    ):
        gaps = [max(0.0, lo1[i] - hi2[i], lo2[i] - hi1[i]) for i in range(3)]
        result["aabb_separation_m"] = round(math.sqrt(sum(v * v for v in gaps)), 4)
    return result


def object_role(row: dict) -> str:
    """Prefer the descriptive asset name over generic categories like furniture."""
    return str(row.get("name") or row.get("category") or "")


def check_object_ids(observations: dict) -> tuple[str, set[str]]:
    """Accept explicit IDs only; object names/prose cannot establish identity."""

    def identifier(value: Any) -> str:
        if isinstance(value, dict):
            value = value.get("object_id") or value.get("id")
        return value if isinstance(value, str) else ""

    primary = identifier(observations.get("primary_object"))
    related = observations.get("related_objects") or []
    related = related if isinstance(related, list) else []
    return primary, ({primary, *[identifier(value) for value in related]} - {""})


def valid_episode(episode: PlacementEpisode) -> bool:
    """Fail closed on edited observations or unbound provenance."""
    return bool(
        episode.episode_id == episode.content_hash()
        and episode.subject.get("object_id") != episode.anchor.get("object_id")
        and object_role(episode.subject)
        and object_role(episode.anchor)
        and episode.state_fingerprint
        and episode.evidence_refs
        and episode.measurements
        and episode.measurements == pair_measurements(episode.subject, episode.anchor)
        and (episode.actions or episode.native_checks)
    )


def _pair_check_index(entry: dict) -> dict[tuple[str, str], list[dict]]:
    """Validate native evidence once, not once for every possible object pair."""
    result: dict[tuple[str, str], list[dict]] = {}
    for constraint in (entry.get("relation_context") or {}).get("hard_constraints", []):
        relation = str(
            constraint.get("relation") or constraint.get("relation_type") or ""
        )
        if relation in {"required_count", "required_coverage", "count", "existence"}:
            continue
        status, checks = resolve_constraint_evidence(constraint, entry)
        if status not in {"verified_pass", "verified_fail"}:
            continue
        for check in checks:
            if (
                check.label not in {"pass", "fail"}
                or check.scoring_tier != "core"
                or check.evaluation_state in {"unknown", "deferred", "not_applicable"}
            ):
                continue
            observation = check.observations
            # Never infer object bindings from a prose mention or category name.
            primary, bound = check_object_ids(observation)
            if not primary:
                continue
            for anchor_id in sorted(bound - {primary}):
                # The relation may quantify several chairs. Its aggregate label
                # must not overwrite this exact pair's own observation.
                result.setdefault((primary, anchor_id), []).append(
                    {
                        "constraint": constraint,
                        "status": (
                            "verified_pass"
                            if check.label == "pass"
                            else "verified_fail"
                        ),
                        "check": check.model_dump(mode="json"),
                    }
                )
    return result


def collect_placement_episodes(scene_dir: Path, stage: str, entry: dict) -> dict:
    """Collect bounded final-state episodes and exact successful tool references.

    Audit calls must belong to this stage's prepared/finished interval. Before
    observations are from the earliest captured designer call, NOT a claimed
    immediate pre-repair state. Final critic evidence applies only to the final
    stage snapshot. No intermediate repair verification is synthesized.
    """
    from scenesmith.scene_expert.memory.usage import related_mutations

    warnings: list[str] = []
    post = entry.get("post_scene_state") or {}
    if not valid_state(post):
        return {"episodes": [], "warnings": ["missing_valid_post_scene_state"]}
    start, end = entry.get("prepared_at_epoch"), entry.get("finished_at_epoch")
    calls = []
    if start is not None and end is not None:
        for call in read_rows(
            scene_dir / "scene_expert/timing/llm_calls.jsonl", warnings
        ):
            at = timestamp(call.get("created_at", ""))
            if (
                call.get("stage") != stage
                or call.get("agent_role") != "designer"
                or call.get("event")
                not in {"request_initial_design", "request_design_change"}
                or at is None
                or not start <= at <= end
            ):
                continue
            path = local_reference(scene_dir, str(call.get("payload_ref") or ""))
            if path is None or not path.is_file() or path.stat().st_size > 32_000_000:
                warnings.append("missing_or_oversized_designer_audit")
                continue
            payload = read_object(path, warnings)
            if (
                payload.get("stage") != stage
                or payload.get("agent_role") != "designer"
                or payload.get("event") != call.get("event")
                or payload.get("error")
                or call.get("error")
            ):
                continue
            calls.append((at, str(path.relative_to(scene_dir.resolve())), payload))
    calls.sort(key=lambda row: (row[0], row[1]))
    if not any((p.get("agent_trace") or {}).get("tool_calls") for _, _, p in calls):
        warnings.append("designer_tool_trace_unavailable; native_pair_evidence_only")
    before = (
        ((calls[0][2].get("context_snapshot") or {}).get("decision_state") or {})
        if calls
        else {}
    )
    before_objects = (
        {r["object_id"]: r for r in before.get("objects", [])}
        if valid_state(before)
        else {}
    )
    objects = [r for r in post["objects"] if object_role(r)]
    # Bound quadratic work; the underlying observer already caps objects at 96.
    objects = objects[:96]
    pair_checks = _pair_check_index({**entry, "stage": stage})
    proposals = []
    for subject in objects:
        candidates = []
        item = {"adaptation": {"bindings": [{"object_ids": [subject["object_id"]]}]}}
        actions = []
        for _, reference, payload in calls:
            for action in related_mutations(payload, item, entry):
                original = next(
                    c
                    for c in payload["agent_trace"]["tool_calls"]
                    if c.get("id") == action["tool_call_id"]
                )
                arguments = original["function"].get("arguments")
                if isinstance(arguments, str):
                    try:
                        arguments = json.loads(arguments)
                    except ValueError:
                        arguments = None
                encoded = json.dumps(arguments, ensure_ascii=False, default=str)
                actions.append(
                    {
                        **action,
                        "payload_ref": reference,
                        "arguments": arguments if len(encoded) <= 2048 else None,
                        "arguments_omitted": len(encoded) > 2048,
                    }
                )
        for anchor in objects:
            if subject["object_id"] == anchor["object_id"]:
                continue
            measured = pair_measurements(subject, anchor)
            if not measured:
                continue
            checks = pair_checks.get((subject["object_id"], anchor["object_id"]), [])
            if (
                str(anchor.get("object_type") or "").casefold() in {"floor", "ceiling"}
                and not checks
                and subject.get("support_object_id") != anchor["object_id"]
            ):
                continue  # Huge room-boundary AABBs must not crowd out asset pairs.
            if not actions and not checks:
                continue
            candidates.append(
                (
                    not bool(checks),
                    measured.get("aabb_separation_m", float("inf")),
                    anchor["object_id"],
                    anchor,
                    measured,
                    checks,
                )
            )
        # Keep the nearest two relevant anchors, prioritizing exact native checks.
        for _, _, _, anchor, measured, checks in sorted(
            candidates, key=lambda row: row[:3]
        )[:2]:
            previous = (
                pair_measurements(
                    before_objects[subject["object_id"]],
                    before_objects[anchor["object_id"]],
                )
                if subject["object_id"] in before_objects
                and anchor["object_id"] in before_objects
                else {}
            )
            episode = PlacementEpisode(
                episode_id="",
                stage=stage,
                state_fingerprint=post["fingerprint"],
                subject=subject,
                anchor=anchor,
                measurements=measured,
                before_measurements=previous,
                actions=actions[:12],
                native_checks=checks,
                stage_passed=(entry.get("verify_report") or {}).get("pass_stage"),
                stage_scores=(entry.get("verify_report") or {}).get("visual_scores")
                or (entry.get("verify_report") or {}).get("scores")
                or {},
                evidence_refs=list(
                    dict.fromkeys(
                        [
                            str(entry.get("scene_state_path") or ""),
                            *[action["payload_ref"] for action in actions[:12]],
                            *([calls[0][1]] if previous else []),
                        ]
                    )
                ),
            )
            episode.episode_id = episode.content_hash()
            proposals.append(episode)
    proposals.sort(
        key=lambda e: (
            not bool(e.native_checks),
            not bool(e.before_measurements and e.before_measurements != e.measurements),
            e.episode_id,
        )
    )
    return {
        "schema_version": "placement-episodes.v1",
        "episodes": [e.model_dump(mode="json") for e in proposals[:32]],
        "omitted_episode_count": max(0, len(proposals) - 32),
        "warnings": sorted(set(warnings)),
    }


def experience_text(experience: PlacementExperience) -> str:
    """Render transferable procedure and measured precedents, without source IDs."""
    from scenesmith.scene_expert.memory.placement_methods import methods_valid

    lines = ["Spatial placement experience (advisory, not task requirements):"]
    lines.extend("Applicable when: " + value for value in experience.applicability)
    if (
        experience.method_steps
        or experience.schema_version == "placement-experience.v2"
    ):
        if not methods_valid(experience):
            return "Spatial method unavailable: invalid per-step source contract."
        by_id = {e.episode_id: e for e in experience.episodes}
        quote_labels = {
            q.evidence_id: i for i, q in enumerate(experience.critic_advice, 1)
        }
        lines.append(
            "Transfer hypotheses (not verified outcomes or fixed target values):"
        )
        for i, step in enumerate(experience.method_steps, 1):
            sources = [
                object_role(by_id[e].subject)
                + " relative to "
                + object_role(by_id[e].anchor)
                for e in step.episode_ids
            ] + [f"critic excerpt {quote_labels[q]}" for q in step.critic_refs]
            lines.append(
                f"{i}. {step.instruction} [sources: {'; '.join(sources)}; {step.binding}]"
            )
    else:
        lines.append(
            "Legacy unverified hypotheses: step-level source binding unavailable; do not copy source constants."
        )
        lines.extend(f"{i}. {step}" for i, step in enumerate(experience.procedure, 1))
    for episode in experience.episodes:
        lines.append(
            f"Source observation only — {object_role(episode.subject)} relative to {object_role(episode.anchor)}: "
            + json.dumps(episode.measurements, sort_keys=True)
        )
        labels = sorted({row["status"] for row in episode.native_checks})
        lines.append(
            "Native checks for this source pair: "
            + (", ".join(labels) or "unavailable; observed layout only")
        )
    for i, quote in enumerate(experience.critic_advice, 1):
        lines.append(
            f"Critic excerpt {i} (source-scene opinion; numeric values are NOT transfer thresholds):\n{quote.quote}"
        )
    lines.extend("Limit: " + value for value in experience.limitations)
    return "\n".join(lines)


def experience_priority(record: Any) -> int:
    """Prefer valid procedural evidence without making legacy records unreadable."""
    experience = record.placement_experience
    if (
        experience
        and experience.procedure
        and experience.episodes
        and all(valid_episode(e) for e in experience.episodes)
    ):
        from scenesmith.scene_expert.memory.placement_methods import methods_valid

        if methods_valid(experience):
            return (
                2
                if all(s.binding == "explicit" for s in experience.method_steps)
                else 1
            )
        return (
            1
            if not experience.method_steps
            and experience.schema_version != "placement-experience.v2"
            else 0
        )
    return 0


def inventory_only(steps: list[str]) -> bool:
    """Detect only obvious inventory restatements, not all shared task goals."""
    if not steps:
        return False
    spatial = re.compile(
        r"\b(rotate|align|offset|axis|axes|yaw|anchor|clearance|edge|front|behind|perpendicular|parallel|distance|spacing|support|collision)\b|朝向|间距|相对|净空|旋转|支撑|对齐",
        re.I,
    )
    inventory = re.compile(
        r"\b(required|inventory|count|all objects|all assets|specified objects)\b|必需|清单|数量|齐全",
        re.I,
    )
    return all(inventory.search(step) and not spatial.search(step) for step in steps)
