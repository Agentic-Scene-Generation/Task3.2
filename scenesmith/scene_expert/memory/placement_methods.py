"""Source-bound spatial methods, separate from observations and critic opinions.

This is a provenance/representation guard, not a natural-language theorem
prover. Every proposed method remains a transfer hypothesis. No scene decisions,
critic verdicts, or measured source values are rewritten here.
"""

from __future__ import annotations

import re

from collections import Counter
from typing import Any

from scenesmith.scene_expert.memory.evidence import evidence_hash
from scenesmith.scene_expert.memory.placement_scope import (
    aliases as _scope_aliases,
    mentions as _scope_mentions,
    intended_metric,
    pair_scope_matches,
    source_metric,
)
from scenesmith.scene_expert.memory.schemas import (
    CriticAdviceEvidence,
    PlacementEpisode,
    PlacementMethodCandidate,
    PlacementMethodStep,
    PlacementRelationEvidence,
)


def critic_catalog(evidence: dict) -> list[CriticAdviceEvidence]:
    """Index complete paragraphs, never synthesized or partially cut quotes.

    Ambiguous repeated-stage reports/snapshots cannot certify a same-attempt join.
    They remain in the full audit but do not become method-level quote references.
    """
    stages = evidence.get("stages") or []
    counts = Counter(s.get("stage") for s in stages)
    catalog = (evidence.get("placement_experience_catalog") or {}).get("episodes", [])
    result = []
    for index, entry in enumerate(stages):
        stage = entry.get("stage")
        report = entry.get("verify_report") or {}
        text = report.get("critique_summary")
        source = report.get("score_source")
        states = {
            e.get("state_fingerprint") for e in catalog if e.get("stage") == stage
        } - {"", None}
        if (
            counts[stage] != 1
            or len(states) != 1
            or not isinstance(text, str)
            or not source
            or source == "unknown"
            or not evidence.get("trace_id")
        ):
            continue
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text)]
        # Whole paragraphs preserve negations, qualifications and list items.
        for quote in list(dict.fromkeys(p for p in paragraphs if 30 <= len(p) <= 1600))[
            :8
        ]:
            item = CriticAdviceEvidence(
                evidence_id="",
                stage=stage,
                stage_entry_index=index,
                trace_id=str(evidence["trace_id"]),
                report_hash=evidence_hash(report),
                source=str(source),
                quote=quote,
                state_fingerprint=next(iter(states)),
                object_roles=sorted(
                    {
                        role
                        for raw in catalog
                        for obj in (raw.get("subject", {}), raw.get("anchor", {}))
                        for role in _mentions(quote, _aliases(obj))
                    }
                ),
            )
            item.evidence_id = item.content_hash()
            result.append(item)
    return result


def instruction_reasons(text: str) -> list[str]:
    """Catch explicit constant/guarantee leakage, not all possible semantic errors.

    Numbers remain available in code-owned observations and verbatim critic quotes.
    Methods must say how to recompute them, not invent a hard target for a new room.
    """
    reasons = []
    if not text.strip() or re.fullmatch(r"[0-9a-f]{64}", text.strip(), re.I):
        reasons.append("not_readable_method")
    numeric_text = re.sub(r"\b[23][dD]\b", "", text)
    if re.search(
        r"\d|[<>≤≥]|\b(one|two|three|four|five|ten)\s*(m|cm|mm|meters?|metres?|degrees?)\b",
        numeric_text,
        re.I,
    ):
        reasons.append("numeric_target_in_method")
    if re.search(
        r"\b(guarantee\w*|always|optimal|universally)\b|\b(will prevent|proven improvement|verified repair)\b|必然|最优",
        text,
        re.I,
    ):
        reasons.append("unproven_transfer_guarantee")
    return reasons


def _aliases(obj: dict) -> set[str]:
    return _scope_aliases(obj)


def _mentions(text: str, aliases: set[str]) -> set[str]:
    return _scope_mentions(text, aliases)


def has_placement_action(instructions: list[str]) -> bool:
    """Do not promote a checklist after its actual placement step was rejected."""
    return any(
        re.search(
            r"(?:^|[.;:]\s*|\band\s+|\bthen\s+)(?:place|position|move|reposition|"
            r"rotate|orient|align|adjust|arrange|center|centre|offset|ensure|maintain|"
            r"distribute|assign|mount|hang|keep|remove|replace|avoid)\b|^(?:摆放|放置|调整|旋转|对齐)",
            text.strip(),
            re.I,
        )
        for text in instructions
    )


def bind_method_steps(
    candidate: Any,
    episodes: dict[str, PlacementEpisode],
    quotes: dict[str, CriticAdviceEvidence],
    *,
    visible_episodes: set[str] | None,
    visible_quotes: set[str] | None,
) -> tuple[list[PlacementMethodStep], list[dict]]:
    """Bind each step independently; rejected prose cannot survive in other fields.

    Legacy response objects are still readable, but shared references are explicitly
    labelled and subjected to the same constant, scope and source checks. Runtime
    prompts request explicit per-step references for all newly authored methods.
    """
    explicit = bool(candidate.method_steps)
    proposals = candidate.method_steps or [
        PlacementMethodCandidate(instruction=t, episode_ids=candidate.episode_ids)
        for t in candidate.procedure
        if t.strip() and len(t) <= 600
    ]
    known_aliases = (
        set().union(
            *[
                _aliases(o)
                for e in episodes.values()
                if e.stage == candidate.stage
                for o in (e.subject, e.anchor)
            ]
        )
        if episodes
        else set()
    )
    accepted, decisions, seen = [], [], set()
    for index, proposal in enumerate(proposals):
        text = proposal.instruction.strip()
        ids, refs = proposal.episode_ids, proposal.critic_refs
        bound = [episodes[i] for i in ids if i in episodes]
        cited = [quotes[i] for i in refs if i in quotes]
        reasons = instruction_reasons(text)
        if not ids and not refs:
            reasons.append("missing_step_source")
        if len(set(ids)) != len(ids) or len(bound) != len(ids):
            reasons.append("invalid_step_episode")
        if len(set(refs)) != len(refs) or len(cited) != len(refs):
            reasons.append("invalid_step_critic_reference")
        if visible_episodes is not None and any(i not in visible_episodes for i in ids):
            reasons.append("step_episode_not_in_model_input")
        if visible_quotes is not None and any(i not in visible_quotes for i in refs):
            reasons.append("step_critic_not_in_model_input")
        if any(e.stage != candidate.stage for e in [*bound, *cited]):
            reasons.append("step_stage_mismatch")
        if any(q.evidence_id != q.content_hash() for q in cited):
            reasons.append("changed_critic_quote")
        states = {e.state_fingerprint for e in bound} | {
            q.state_fingerprint for q in cited
        }
        if len(states) > 1:
            reasons.append("step_snapshot_mismatch")
        supported_aliases = (
            set().union(*[_aliases(o) for e in bound for o in (e.subject, e.anchor)])
            if bound
            else set()
        )
        for q in cited:
            supported_aliases |= _mentions(q.quote, known_aliases)
        missing = _mentions(text, known_aliases) - supported_aliases
        if missing:
            reasons.append("uncited_object_roles:" + ",".join(sorted(missing)))
        if text.casefold() in seen:
            reasons.append("duplicate_method_step")
        # Invalid identities remain hard failures. Valid but unrelated geometry
        # may be omitted when an exact critic quote supports an advisory step.
        warnings, relations = [], []
        declarations = getattr(proposal, "relations", [])
        if declarations and {r.episode_id for r in declarations} != set(ids):
            reasons.append("relation_episode_union_mismatch")
        for episode in bound:
            declared = [r for r in declarations if r.episode_id == episode.episode_id]
            metric = intended_metric(text)
            if declarations:
                if len(declared) != 1:
                    reasons.append("missing_or_duplicate_relation_binding")
                    continue
                relation = declared[0]
                if (relation.subject_id, relation.anchor_id) != (
                    episode.subject.get("object_id"),
                    episode.anchor.get("object_id"),
                ):
                    reasons.append("relation_instance_mismatch")
                    continue
                if metric != "pair_observation" and relation.metric != metric:
                    reasons.append("relation_metric_mismatch")
                    continue
                if metric == "pair_observation" and relation.metric != metric:
                    warnings.append("unsupported_step_metric:" + episode.episode_id)
                    continue
                metric = relation.metric
            if not pair_scope_matches(text, episode):
                warnings.append("unrelated_step_pair:" + episode.episode_id)
                continue
            value = source_metric(episode, metric)
            if metric != "pair_observation" and value is None:
                warnings.append("unavailable_step_metric:" + episode.episode_id)
                continue
            relations.append(
                PlacementRelationEvidence(
                    episode_id=episode.episode_id,
                    subject_id=episode.subject["object_id"],
                    anchor_id=episode.anchor["object_id"],
                    metric=metric,
                    value=value,
                )
            )
        kept_ids = [r.episode_id for r in relations]
        if ids and not kept_ids and not cited:
            reasons.append("no_supported_step_relation_or_advice")
        decisions.append(
            {
                "step_index": index,
                "instruction": text,
                "episode_ids": ids,
                "critic_refs": refs,
                "accepted_episode_ids": kept_ids,
                "relation_bindings": [r.model_dump(mode="json") for r in relations],
                "warnings": warnings,
                "evidence_scope": (
                    "source_observation" if kept_ids else "critic_advice_only"
                ),
                "decision": "rejected" if reasons else "bound_hypothesis",
                "reasons": reasons,
            }
        )
        if reasons:
            continue
        seen.add(text.casefold())
        accepted.append(
            PlacementMethodStep(
                instruction=text,
                episode_ids=kept_ids,
                critic_refs=refs,
                binding="explicit" if explicit else "legacy_shared_refs",
                evidence_kinds=(["source_observation"] if kept_ids else [])
                + (["critic_advice"] if refs else []),
                relation_binding_version=1,
                relation_bindings=relations,
            )
        )
    return accepted, decisions


def methods_valid(experience: Any) -> bool:
    """Validate persisted per-step references and immutable quote contents."""
    if not experience.method_steps or not has_placement_action(experience.procedure):
        return False
    episodes = {e.episode_id: e for e in experience.episodes}
    quotes = {q.evidence_id: q for q in experience.critic_advice}
    from scenesmith.scene_expert.memory.placement import valid_episode

    if len(episodes) != len(experience.episodes) or len(quotes) != len(
        experience.critic_advice
    ):
        return False
    if not episodes or any(not valid_episode(e) for e in episodes.values()):
        return False
    if (
        len({e.stage for e in episodes.values()} | {q.stage for q in quotes.values()})
        != 1
    ):
        return False
    if (
        len(
            {e.state_fingerprint for e in episodes.values()}
            | {q.state_fingerprint for q in quotes.values()}
        )
        != 1
    ):
        return False
    if any(q.evidence_id != q.content_hash() for q in quotes.values()):
        return False
    context_ids = experience.source_context_episode_ids
    method_ids = {i for s in experience.method_steps for i in s.episode_ids}
    if (
        len(context_ids) != len(set(context_ids))
        or set(context_ids) - episodes.keys()
        or set(context_ids) & method_ids
    ):
        return False
    for step in experience.method_steps:
        if any(
            not pair_scope_matches(step.instruction, episodes[i])
            for i in step.episode_ids
            if i in episodes
        ):
            return False
        if step.relation_binding_version == 1:
            if [r.episode_id for r in step.relation_bindings] != step.episode_ids:
                return False
            for r in step.relation_bindings:
                e = episodes.get(r.episode_id)
                if e is None or (r.subject_id, r.anchor_id) != (
                    e.subject.get("object_id"),
                    e.anchor.get("object_id"),
                ):
                    return False
                metric = intended_metric(step.instruction)
                if metric != r.metric:
                    return False
                value = source_metric(e, r.metric)
                if (
                    r.metric != "pair_observation" and value is None
                ) or r.value != value:
                    return False
        elif step.relation_bindings:
            return False
    if any(
        len(s.episode_ids) != len(set(s.episode_ids))
        or len(s.critic_refs) != len(set(s.critic_refs))
        or s.evidence_kinds
        != (["source_observation"] if s.episode_ids else [])
        + (["critic_advice"] if s.critic_refs else [])
        for s in experience.method_steps
    ):
        return False
    return experience.procedure == [
        s.instruction for s in experience.method_steps
    ] and all(
        not instruction_reasons(s.instruction)
        and (s.episode_ids or s.critic_refs)
        and set(s.episode_ids) <= episodes.keys()
        and set(s.critic_refs) <= quotes.keys()
        for s in experience.method_steps
    )
