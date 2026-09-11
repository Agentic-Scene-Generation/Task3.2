"""Deterministic semantic identity for independently observed Skill records."""

from __future__ import annotations

import hashlib
import json
import re

from typing import Any

from scenesmith.scene_expert.memory.room_taxonomy import canonical_room_type
from scenesmith.scene_expert.memory.schemas import Skill

SKILL_SIGNATURE_VERSION = "sceneexpert.skill_signature.v1"

_RELATION_ALIASES = {
    "away_from": "faces_away_from",
    "beside": "near",
    "centered_in_room": "room_center_alignment",
    "close_to": "near",
    "face": "faces",
    "facing": "faces",
    "facing_away": "faces_away_from",
    "facing_toward": "faces",
    "far": "far_from",
}

_OBJECT_FAMILY_BY_ROLE = {
    **{
        role: "chair"
        for role in (
            "armchair",
            "chair",
            "dining_chair",
            "guest_chair",
            "office_chair",
            "student_chair",
            "stool",
        )
    },
    **{
        role: "desk"
        for role in (
            "desk",
            "reception_desk",
            "student_desk",
            "teacher_desk",
        )
    },
    **{
        role: "table"
        for role in (
            "coffee_table",
            "conference_table",
            "dining_table",
            "side_table",
            "table",
        )
    },
}

_PROCEDURE_FAMILIES: dict[str, frozenset[str]] = {
    "alignment": frozenset({"align", "aligned", "alignment", "axis", "center"}),
    "balance": frozenset({"balance", "balanced", "symmetry", "symmetric"}),
    "clearance": frozenset(
        {"access", "approach", "clearance", "egress", "walkable", "walkway"}
    ),
    "collision": frozenset(
        {"collision", "intersect", "intersection", "overlap", "penetration"}
    ),
    "coverage": frozenset({"count", "coverage", "required", "requirement"}),
    "distribution": frozenset(
        {"distribute", "distributed", "distribution", "edge", "slot", "spacing"}
    ),
    "orientation": frozenset(
        {"face", "faces", "facing", "orient", "orientation", "toward", "yaw"}
    ),
    "style": frozenset({"aesthetic", "material", "palette", "style", "visual"}),
    "support": frozenset(
        {"height", "on", "surface", "support", "supported", "supporting", "top"}
    ),
}


def _normalize(value: Any) -> str:
    return re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "_", str(value or "").casefold()).strip(
        "_"
    )


def _object_role(value: Any) -> str:
    normalized = _normalize(value)
    if normalized.endswith("ies"):
        normalized = normalized[:-3] + "y"
    elif normalized.endswith("ses"):
        normalized = normalized[:-2]
    elif normalized.endswith("s") and not normalized.endswith("ss"):
        normalized = normalized[:-1]
    return _OBJECT_FAMILY_BY_ROLE.get(normalized, normalized)


def _relation(value: Any) -> str:
    normalized = _normalize(value)
    return _RELATION_ALIASES.get(normalized, normalized)


def _procedure_families(skill: Skill) -> list[str]:
    text = " ".join(
        [
            skill.skill_name,
            *skill.preconditions,
            *skill.procedure,
            *skill.failure_avoidance,
            *skill.postconditions,
        ]
    )
    tokens = {
        token
        for token in re.split(r"[^a-z0-9\u4e00-\u9fff]+", text.casefold())
        if token
    }
    families = sorted(
        family
        for family, vocabulary in _PROCEDURE_FAMILIES.items()
        if tokens & vocabulary
    )
    if families:
        return families
    # Unknown procedures still receive a stable, conservative identity. Using
    # the explicit Skill name avoids merging unrelated procedures that merely
    # share a room, stage, and generic object role.
    fallback = _normalize(skill.skill_name)
    return [fallback] if fallback else []


def build_skill_semantic_signature(skill: Skill) -> str:
    """Return a name-independent signature for equivalent procedural evidence.

    Structured stage, room, role, and relation scopes control identity. A coarse
    procedure family prevents unrelated procedures over the same objects from
    merging, while allowing harmless wording/name differences across LLM calls.
    """

    declared_rooms = [
        skill.room_type,
        *skill.room_types,
        *skill.applicability.room_types,
    ]
    rooms = sorted(
        {
            canonical
            for value in declared_rooms
            if (canonical := canonical_room_type(value))
        }
    )
    roles = sorted(
        {
            role
            for value in [
                *skill.required_objects,
                *skill.applicability.required_object_roles,
                *[relation.subject_role for relation in skill.spatial_relations],
                *[relation.target_role for relation in skill.spatial_relations],
            ]
            if (role := _object_role(value))
        }
    )
    relations = sorted(
        {
            relation
            for value in [
                *skill.applicability.required_relation_types,
                *[item.relation_type for item in skill.spatial_relations],
            ]
            if (relation := _relation(value))
        }
    )
    payload = {
        "version": SKILL_SIGNATURE_VERSION,
        "stage": _normalize(skill.stage),
        "rooms": rooms or ["generic"],
        "roles": roles,
        "relations": relations,
        "procedure_families": _procedure_families(skill),
    }
    digest = hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return f"{SKILL_SIGNATURE_VERSION}:{digest[:24]}"
