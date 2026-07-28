"""Deterministic recovery of prompt-explicit manipuland target furniture.

The VLM still decides optional surface decoration.  This module only recovers
furniture whose small-object obligations are explicit in the user prompt, so a
selection miss cannot silently drop required table settings or bedside items.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any


@dataclass(frozen=True)
class ManipulandTargetObligation:
    category: str
    required_items: str
    target_count: int = 1


_CATEGORY_ALIASES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("dining_table", ("dining table",)),
    ("coffee_table", ("coffee table",)),
    ("nightstand", ("nightstand", "bedside table")),
    ("sideboard", ("sideboard", "buffet cabinet")),
    ("bookshelf", ("bookshelf", "bookcase")),
    ("desk", ("desk", "workstation")),
    ("dresser", ("dresser", "chest of drawers")),
    ("table", ("table",)),
)


def classify_manipuland_furniture(obj: Any, object_id: Any = "") -> str | None:
    """Return the most specific support-furniture category for an object."""
    text = (
        " ".join(
            (
                str(object_id),
                str(getattr(obj, "name", "")),
                str(getattr(obj, "description", "")),
            )
        )
        .lower()
        .replace("_", " ")
    )
    for category, aliases in _CATEGORY_ALIASES:
        if any(re.search(rf"\b{re.escape(alias)}s?\b", text) for alias in aliases):
            return category
    return None


def infer_prompt_manipuland_obligations(
    scene_description: str,
) -> list[ManipulandTargetObligation]:
    """Infer only prompt-explicit furniture-to-small-object assignments."""
    text = " ".join(str(scene_description or "").lower().split())
    obligations: list[ManipulandTargetObligation] = []

    if re.search(r"\bdining\s+table\b", text) and re.search(
        r"\b(?:table|place)\s+settings?\b|\bplates?\b|\bcutlery\b|"
        r"\b(?:wine\s+)?glasses?\b|\bcenterpiece\b",
        text,
    ):
        obligations.append(
            ManipulandTargetObligation(
                category="dining_table",
                required_items=(
                    "table settings in the prompt-required quantity, including "
                    "plates, cutlery, and glasses; centerpiece vase/flowers when requested"
                ),
            )
        )

    if re.search(r"\bcoffee\s+table\b", text) and re.search(
        r"\bremote(?:\s+control)?s?\b|\bmagazines?\b|\bbooks?\b|\bvases?\b",
        text,
    ):
        obligations.append(
            ManipulandTargetObligation(
                category="coffee_table",
                required_items="remote controls, magazines, books, or vases explicitly assigned to the coffee table",
            )
        )

    nightstand_items = re.search(
        r"\btable\s+lamps?\b|\balarm\s+clocks?\b|\bbooks?\b|\bphones?\b",
        text,
    )
    if re.search(r"\b(?:nightstands?|bedside\s+tables?)\b", text) and nightstand_items:
        bilateral = bool(
            re.search(
                r"(?:on|at)\s+(?:the\s+)?(?:each|either|both)\s+side(?:s)?\s+of\s+(?:the\s+)?bed\b",
                text,
            )
            or re.search(r"\bone\s+nightstand\b.{0,100}\bthe\s+other\b", text)
        )
        obligations.append(
            ManipulandTargetObligation(
                category="nightstand",
                required_items="table lamps, alarm clocks, books, or phones explicitly assigned to the nightstands",
                target_count=2 if bilateral else 1,
            )
        )

    if re.search(r"\b(?:sideboard|buffet\s+cabinet)\b", text) and re.search(
        r"\bcoasters?\b.{0,80}\b(?:on|atop)\s+(?:the\s+)?(?:sideboard|buffet)\b|"
        r"\b(?:on|atop)\s+(?:the\s+)?(?:sideboard|buffet)\b.{0,80}\bcoasters?\b",
        text,
    ):
        obligations.append(
            ManipulandTargetObligation(
                category="sideboard",
                required_items="the prompt-required set of coasters",
            )
        )

    if re.search(r"\bdesk\b", text) and re.search(
        r"\b(?:monitor|desk\s+lamp|notebook|pen\s+holder|laptop)\b.{0,80}"
        r"\b(?:on|atop)\s+(?:the\s+)?desk\b|"
        r"\b(?:on|atop)\s+(?:the\s+)?desk\b.{0,100}"
        r"\b(?:monitor|desk\s+lamp|notebook|pen\s+holder|laptop)\b",
        text,
    ):
        obligations.append(
            ManipulandTargetObligation(
                category="desk",
                required_items="monitor, desk lamp, notebook, pen holder, or laptop explicitly assigned to the desk",
            )
        )

    return obligations
