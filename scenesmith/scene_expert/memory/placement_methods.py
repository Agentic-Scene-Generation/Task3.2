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
from scenesmith.scene_expert.memory.schemas import (
    CriticAdviceEvidence,
    PlacementEpisode,
    PlacementMethodCandidate,
    PlacementMethodStep,
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
        r"\b(guarantee\w*|always|optimal|universally)\b|\b(to ensure|will prevent|proven improvement|verified repair)\b|保证|必然|最优",
        text,
        re.I,
    ):
        reasons.append("unproven_transfer_guarantee")
    return reasons


def _aliases(obj: dict) -> set[str]:
    name = re.sub(
        r"_\d+$", "", str(obj.get("name") or obj.get("category") or "")
    ).lower()
    name = re.sub(r"[_\-]+", " ", name).strip()
    aliases = {name} - {""}
    # Stable generic endpoints; this is a small scope guard, not a new taxonomy.
    for word in (
        "wall",
        "bed",
        "nightstand",
        "chair",
        "table",
        "shelf",
        "bookshelf",
        "lamp",
        "mirror",
    ):
        if word in name.split():
            aliases.add(word)
    if obj.get("object_type") == "ceiling_mounted":
        aliases.update(("fixture", "light"))
    return aliases


def _mentions(text: str, aliases: set[str]) -> set[str]:
    normalized = text.lower().replace("_", " ")
    return {
        a
        for a in aliases
        if re.search(r"(?<!\w)" + re.escape(a) + r"(?:s)?(?!\w)", normalized)
    }


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
        decisions.append(
            {
                "step_index": index,
                "instruction": text,
                "episode_ids": ids,
                "critic_refs": refs,
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
                episode_ids=ids,
                critic_refs=refs,
                binding="explicit" if explicit else "legacy_shared_refs",
                evidence_kinds=(["source_observation"] if ids else [])
                + (["critic_advice"] if refs else []),
            )
        )
    return accepted, decisions


def methods_valid(experience: Any) -> bool:
    """Validate persisted per-step references and immutable quote contents."""
    if not experience.method_steps:
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
