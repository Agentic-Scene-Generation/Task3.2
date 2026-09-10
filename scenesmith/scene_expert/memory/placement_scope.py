"""Conservative source-pair scope checks, not a general language entailment model.

Explicit Writer pair/metric declarations are verified against immutable episodes.
Small lexical guards catch identifiable endpoint contradictions in legacy prose.
Unknown relations remain observations/advice, never fabricated measurements.
"""

from __future__ import annotations

import math
import re
from typing import Any

OBJECT_ROLES = (
    "wall",
    "bed",
    "nightstand",
    "chair",
    "table",
    "desk",
    "shelf",
    "bookshelf",
    "lamp",
    "mirror",
    "sofa",
    "cabinet",
    "wardrobe",
    "sink",
    "toilet",
    "plant",
    "bench",
    "stool",
    "fixture",
    "light",
)


def aliases(obj: dict) -> set[str]:
    name = re.sub(
        r"_\d+$", "", str(obj.get("name") or obj.get("category") or "")
    ).lower()
    name = re.sub(r"[_\-]+", " ", name).strip()
    result = {name} - {""}
    for word in OBJECT_ROLES:
        if word in name.split():
            result.add(word)
    if obj.get("object_type") == "ceiling_mounted":
        result.update(("fixture", "light"))
    return result


def mentions(text: str, names: set[str]) -> set[str]:
    normalized = text.lower().replace("_", " ")
    return {
        a
        for a in names
        if re.search(r"(?<!\w)" + re.escape(a) + r"(?:s)?(?!\w)", normalized)
    }


def pair_scope_matches(text: str, episode: Any) -> bool:
    """Reject named endpoint substitutions; generic anchor procedures stay readable."""
    left, right = aliases(episode.subject), aliases(episode.anchor)
    a, b = mentions(text, left), mentions(text, right)
    # Explicit plural peer relations need TWO distinct instances of that role,
    # not the same fixture paired with two different walls.
    for role in OBJECT_ROLES:
        peer = re.search(
            r"\b(?:between|among)\s+(?:(?:the|ceiling|adjacent)\s+)*"
            + role
            + r"s\b|\binter[- ]"
            + role
            + r"\b",
            text,
            re.I,
        )
        if peer and not re.match(r"\s+(?:and|or)\b", text[peer.end() :], re.I):
            return role in left and role in right
    # Named source identities must be the actual endpoints, never category aliases.
    named_ids = set(re.findall(r"\b[a-z][a-z_]*_\d+\b", text, re.I))
    if named_ids - {episode.subject.get("object_id"), episode.anchor.get("object_id")}:
        return False
    if a and b:
        return True
    # A cited paragraph may mention many roles. It cannot make an unrelated
    # pair support a method involving a different named endpoint.
    if mentions(text, set(OBJECT_ROLES)) - left - right and (a or b):
        return False
    # If an instruction explicitly talks about furniture/fixture endpoints but
    # never a wall, a wall observation is not evidence for that relation.
    if ("wall" in left or "wall" in right) and "wall" not in text.lower():
        if re.search(
            r"\b(align|centers?|distance|spacing|flank|between|relative)\b", text, re.I
        ):
            return False
    return True


def intended_metric(text: str) -> str:
    """Only identifiable measurement kinds; never reinterpret yaw as semantic front."""
    if re.search(r"center[- ]to[- ]center|centre[- ]to[- ]centre", text, re.I):
        return "bbox_center_distance_m"
    if re.search(r"\b(yaw|relative rotation|transform orientation)\b", text, re.I):
        return "relative_yaw_deg"
    if re.search(r"\b(offset|local coordinates)\b", text, re.I):
        return "anchor_local_offset_m"
    if re.search(r"\b(aabb|gap|separation)\b", text, re.I):
        return "aabb_separation_m"
    return "pair_observation"


def source_metric(episode: Any, metric: str) -> float | list[float] | None:
    """Compute bbox-center distance separately; preserve old episode hashes."""
    if metric == "pair_observation":
        return None
    if metric != "bbox_center_distance_m":
        return episode.measurements.get(metric)
    boxes = [
        obj.get(k)
        for obj in (episode.subject, episode.anchor)
        for k in ("bbox_min", "bbox_max")
    ]
    if any(
        not isinstance(v, list)
        or len(v) != 3
        or any(type(x) not in (int, float) or not math.isfinite(x) for x in v)
        for v in boxes
    ):
        return None
    lo, hi, alo, ahi = boxes
    if any(lo[i] > hi[i] or alo[i] > ahi[i] for i in range(3)):
        return None
    return round(
        math.sqrt(sum(((lo[i] + hi[i] - alo[i] - ahi[i]) / 2) ** 2 for i in range(3))),
        4,
    )
