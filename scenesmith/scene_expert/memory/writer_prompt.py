"""Bounded model projection of an immutable, independently archived writer input.

Budgets use serialized UTF-8 bytes, not the unsafe English chars/4 heuristic.
This is a conservative local envelope, not a claim to implement a server's
tokenizer. Server context errors remain authoritative and trigger reduction.
"""

from __future__ import annotations

import json

from collections import defaultdict
from typing import Any

from scenesmith.scene_expert.memory.evidence import evidence_hash
from scenesmith.scene_expert.memory.placement import valid_episode
from scenesmith.scene_expert.memory.placement_methods import critic_catalog
from scenesmith.scene_expert.memory.placement_outcomes import (
    episode_catalog,
    success_scope,
)
from scenesmith.scene_expert.memory.schemas import PlacementEpisode

INSTRUCTION = (
    "Extract concise reusable spatial methods from these observations. "
    "Return at most two candidates total. Bind each method_steps item to visible "
    "episode_ids and/or critic_refs. Keep numeric values only in source evidence, "
    "not in transferred instructions. Select only visible episode_ids. "
    "Declare relations with exact episode_id, subject_id, anchor_id and metric; "
    "never attach fixture-wall geometry to a fixture-fixture spacing method. "
    "Critic-only advice uses critic_refs and empty step episode_ids/relations; "
    "root episode_ids supply context only, or code binds same-snapshot context. "
    "Stage reports and episode-specific outcomes are authoritative, not task "
    "inventories or retrieved advice. Omitted evidence is unknown, not passing. "
    "Local pair passes do not certify a failed stage. Include a supported placement action. "
    "Final geometry does not prove an optimal layout or a successful repair. "
    "If no supported method exists, return empty arrays and explain noop_reason.\n"
)


class WriterPromptBudgetError(ValueError):
    """The minimum complete writer input cannot fit the configured envelope."""


def encoded(value: Any) -> str:
    """Canonical compact JSON, also used for deterministic size accounting."""
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)


def byte_size(value: Any) -> int:
    """Measure serialized bytes without estimating language-specific tokens."""
    return len(encoded(value).encode("utf-8"))


def _small(value: Any, limit: int = 2048) -> Any:
    # Never slice JSON, a constraint, a negation or a procedure mid-sentence.
    if byte_size(value) <= limit:
        return value
    return {"omitted": "field_byte_budget", "sha256": evidence_hash(value)}


def _episode_view(episode: PlacementEpisode) -> dict:
    row = episode.model_dump(mode="json")
    view = {
        key: row[key]
        for key in (
            "episode_id",
            "stage",
            "measurements",
            "before_measurements",
            "observation_scope",
            "repair_verified",
            "stage_passed",
        )
    }
    for role in ("subject", "anchor"):
        view[role] = {k: row[role].get(k) for k in ("object_id", "name", "category")}
    # Keep whole operations, including arguments, and the final adjustments.
    actions = row["actions"]
    selected = actions if len(actions) <= 3 else [actions[0], *actions[-2:]]
    view["actions"] = selected
    view["actions_omitted_count"] = len(actions) - len(selected)
    view["native_checks"] = []
    for item in row["native_checks"]:
        check = item["check"]
        observations = check.get("observations") or {}
        diagnostics = observations.get("diagnostics") or {}
        evidence = observations.get("evidence") or {}
        # Keep the critic's reason/measurements, not duplicated full contracts.
        view["native_checks"].append(
            {
                "constraint_id": check.get("constraint_id"),
                "metric": check.get("metric"),
                "status": item["status"],
                "relation": (item.get("constraint") or {}).get("relation", "unknown"),
                "source_requirement": _small(
                    (item.get("constraint") or {}).get("evidence_span", ""), 1024
                ),
                "observations": {
                    "primary_object": observations.get("primary_object"),
                    "related_objects": observations.get("related_objects", []),
                    "evaluation_source": observations.get("evaluation_source"),
                    "diagnostics": {
                        k: _small(v, 1024)
                        for k, v in diagnostics.items()
                        if k
                        not in {
                            "intent_contract",
                            "intent_constraints",
                            "intent_constraint",
                        }
                    },
                    "evidence": {
                        k: _small(v, 1024)
                        for k, v in evidence.items()
                        if k
                        not in {
                            "intent_contract",
                            "intent_constraints",
                            "intent_constraint",
                        }
                    },
                },
            }
        )
    return view


def build_writer_prompt(
    *,
    evidence: dict,
    final_report: dict,
    trace_summary: str,
    related_old_memory: str,
    max_user_bytes: int,
) -> tuple[str, dict]:
    """Pack complete evidence units fairly across stages, with omission reasons.

    The complete source remains the authority for binding and validation. Only
    hash-valid passing/failing episodes compete for model space; unknown stages
    must not consume a slot intended for learnable spatial evidence.
    """
    payload = {
        "final_report": _small(final_report, 4096),
        "evidence": {
            "trace_id": _small(evidence.get("trace_id", "")),
            "prompt": _small(evidence.get("prompt", ""), 4096),
            "task_spec": _small(evidence.get("task_spec", {}), 4096),
            "stages": [],
        },
    }
    projected = payload["evidence"]
    has_catalog = "placement_experience_catalog" in evidence
    if has_catalog:
        projected["placement_experience_catalog"] = {"episodes": []}

    def size() -> int:
        return len((INSTRUCTION + encoded(payload)).encode("utf-8"))

    if size() > max_user_bytes:
        raise WriterPromptBudgetError(
            "Writer task/report envelope exceeds input budget"
        )

    omitted: list[dict] = []
    stage_details = []
    for index, stage in enumerate(evidence.get("stages") or []):
        report = stage.get("verify_report") or {}
        compact = {
            "stage": stage.get("stage"),
            "attempt_index": index,
            "verify_report": {
                k: _small(report.get(k))
                for k in ("pass_stage", "score_source", "scores", "visual_scores")
            },
        }
        projected["stages"].append(compact)
        if size() > max_user_bytes:
            projected["stages"].pop()
            omitted.append({"stage": stage.get("stage"), "reason": "stage_budget"})
        else:
            stage_details.append((compact["verify_report"], report))

    groups: dict[tuple[str, str], list[PlacementEpisode]] = defaultdict(list)
    catalog = episode_catalog(evidence)
    seen = set()
    for raw in (evidence.get("placement_experience_catalog") or {}).get("episodes", []):
        eid = str(raw.get("episode_id") or "")
        try:
            episode = PlacementEpisode.model_validate(raw)
            valid = valid_episode(episode)
        except (ValueError, TypeError, KeyError):
            valid = False
        if not valid:
            omitted.append({"episode_id": eid, "reason": "invalid_episode"})
            continue
        if eid in seen:
            continue
        seen.add(eid)
        failing = any(c["status"] == "verified_fail" for c in episode.native_checks)
        if not failing and success_scope(episode, catalog) is None:
            omitted.append({"episode_id": eid, "reason": "no_verified_outcome"})
            continue
        groups[(episode.stage, "failure" if failing else "success")].append(episode)
    eligible_ids = [e.episode_id for group in groups.values() for e in group]
    selected_ids = []
    counts: dict[str, int] = defaultdict(int)
    # Round-robin prevents a large furniture stage starving all other stages.
    for index in range(max((len(g) for g in groups.values()), default=0)):
        for group in groups.values():
            if index >= len(group):
                continue
            episode = group[index]
            reason = "stage_episode_limit"
            if counts[episode.stage] < 4:
                episodes = projected["placement_experience_catalog"]["episodes"]
                episodes.append(_episode_view(episode))
                if size() <= max_user_bytes:
                    selected_ids.append(episode.episode_id)
                    counts[episode.stage] += 1
                    continue
                episodes.pop()
                reason = "input_byte_budget"
            omitted.append({"episode_id": episode.episode_id, "reason": reason})

    # Quote references are bound to original full reports, not copied model prose.
    selected_critic_ids = []
    if has_catalog:
        projected["critic_evidence_catalog"] = []
        quote_groups: dict[str, list] = defaultdict(list)
        for quote in critic_catalog(evidence):
            if counts[quote.stage]:
                quote_groups[quote.stage].append(quote)
        # Give later stages a chance to contribute advice, just like geometry.
        quotes = [
            group[i]
            for i in range(max(map(len, quote_groups.values()), default=0))
            for group in quote_groups.values()
            if i < len(group)
        ]
        for quote in quotes:
            projected["critic_evidence_catalog"].append(quote.model_dump(mode="json"))
            if size() <= max_user_bytes:
                selected_critic_ids.append(quote.evidence_id)
            else:
                projected["critic_evidence_catalog"].pop()
                omitted.append(
                    {"critic_ref": quote.evidence_id, "reason": "input_byte_budget"}
                )
        if not selected_critic_ids:
            del projected["critic_evidence_catalog"]
    # Secondary narrative cannot crowd out exact spatial evidence. Whitelist:
    # no source-file hash list, full memory packs, planner prompts or raw states.
    for destination, report in stage_details:
        for key in ("critique_summary", "issues", "repair_suggestions"):
            if has_catalog and key == "critique_summary":
                continue  # Exact, indexed excerpts already accompany spatial input.
            if key not in report:
                continue
            destination[key] = _small(report[key], 4096)
            if size() > max_user_bytes:
                del destination[key]
    for key, value in (
        ("trace_summary", trace_summary),
        ("related_existing_memory", related_old_memory),
    ):
        if value:
            payload[key] = _small(value, 2048)
            if size() > max_user_bytes:
                del payload[key]
    metadata = {
        "schema_version": "memory-writer-prompt.v2",
        "source_evidence_sha256": evidence_hash(evidence),
        "source_evidence_bytes": byte_size(evidence),
        "max_user_bytes": max_user_bytes,
        "user_bytes": size(),
        "eligible_episode_ids": eligible_ids,
        "selected_episode_ids": selected_ids,
        "selected_critic_ids": selected_critic_ids,
        "omissions": omitted,
        "has_catalog": has_catalog,
    }
    return INSTRUCTION + encoded(payload), metadata
