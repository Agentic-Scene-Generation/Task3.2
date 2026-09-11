"""One source-local eligibility rule for projection, binding and persistence.

Local passes never certify a stage or a transferred procedure. A conflicting
native failure on the same pair/snapshot wins, including reversed pair storage.
"""

from __future__ import annotations

from collections.abc import Iterable
import re

from scenesmith.scene_expert.memory.placement import check_object_ids, valid_episode
from scenesmith.scene_expert.memory.schemas import PlacementEpisode, PlacementExperience


def episode_catalog(evidence: dict) -> list[PlacementEpisode]:
    """Read immutable, hash-checked episodes without modifying their outcomes."""
    result = {}
    for raw in (evidence.get("placement_experience_catalog") or {}).get("episodes", []):
        try:
            episode = PlacementEpisode.model_validate(raw)
            if valid_episode(episode):
                result[episode.episode_id] = episode
        except (ValueError, TypeError, KeyError):
            continue
    return list(result.values())


def local_passes(episode: PlacementEpisode) -> list[dict]:
    """Only explicit native pair checks, not aggregate scores or prose."""
    result = []
    pair = {episode.subject.get("object_id"), episode.anchor.get("object_id")}
    for row in episode.native_checks:
        check = row.get("check") or {}
        primary, related = check_object_ids(check.get("observations") or {})
        relation = (row.get("constraint") or {}).get("relation")
        if (
            row.get("status") == "verified_pass"
            and check.get("label") == "pass"
            and check.get("scoring_tier") == "core"
            and check.get("stage") == episode.stage
            and isinstance(relation, str)
            and relation not in (None, "", "count", "exists", "existence")
            and primary in pair
            and pair - {primary} <= related
        ):
            result.append(row)
    return result


def success_scope(
    episode: PlacementEpisode, catalog: Iterable[PlacementEpisode]
) -> str | None:
    """Return stage/relation support, or None; never manufacture scene success."""
    catalog = list(catalog)
    if not valid_episode(episode) or episode.episode_id not in {
        e.episode_id for e in catalog
    }:
        return None
    pair = {episode.subject.get("object_id"), episode.anchor.get("object_id")}
    for other in catalog:
        if (
            other.stage == episode.stage
            and other.state_fingerprint == episode.state_fingerprint
            and {other.subject.get("object_id"), other.anchor.get("object_id")} == pair
            and any(c.get("status") == "verified_fail" for c in other.native_checks)
        ):
            return None
    if episode.stage_passed is True:
        return "stage"
    return "relation" if local_passes(episode) else None


def relation_method_supported(
    experience: PlacementExperience, catalog: list[PlacementEpisode]
) -> bool:
    """A local pass cannot sponsor an unrelated action or critic-only method.

    This small vocabulary follows native relation names. Unknown paraphrases abstain;
    we do not attempt semantic entailment or turn transform yaw into functional front.
    """
    from scenesmith.scene_expert.memory.placement_methods import has_placement_action

    patterns = {
        "near": r"\b(near|adjacent|next to|proximity)\b",
        "next_to": r"\b(near|adjacent|next to|beside)\b",
        "against_wall": r"\b(against|along|backed|wall[- ]aligned)\b",
        "faces": r"\b(fac(?:e|es|ing)|towards?|orient)\b",
        "flanking": r"\b(flank\w*|lateral sides|either side|opposite sides)\b",
        "surround": r"\b(surround\w*|around|encircl\w*)\b",
    }
    by_id = {e.episode_id: e for e in experience.episodes}
    actions = [
        s for s in experience.method_steps if has_placement_action([s.instruction])
    ]
    has_local_action = False
    for step in actions:
        if not step.episode_ids:
            return False
        for eid in step.episode_ids:
            episode = by_id[eid]
            scope = success_scope(episode, catalog)
            if scope == "stage":
                continue
            if scope != "relation":
                return False
            supported = False
            for row in local_passes(episode):
                relation = row["constraint"]["relation"]
                pattern = patterns.get(
                    relation, r"\b" + re.escape(relation.replace("_", " ")) + r"\b"
                )
                supported |= bool(re.search(pattern, step.instruction, re.I))
            if not supported:
                return False
            has_local_action = True
    return has_local_action
