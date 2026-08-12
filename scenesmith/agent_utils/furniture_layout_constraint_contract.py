"""Shared visual-layout constraints for image generation and furniture agents."""

from __future__ import annotations

from typing import Any


def build_furniture_layout_constraint_contract(
    scene: Any, safety_controller: Any | None
) -> dict[str, Any]:
    """Compile one prioritized rule set from authoritative scene/safety state."""
    room_geometry = getattr(scene, "room_geometry", None)
    polygon_room = isinstance(
        getattr(room_geometry, "footprint_vertices", None), (list, tuple)
    )
    openings = list(getattr(room_geometry, "openings", []) or [])
    opening_types = sorted(
        {str(getattr(opening, "opening_type", "opening")) for opening in openings}
    )
    required_counts = dict(getattr(safety_controller, "required_counts", {}) or {})
    return {
        "schema_version": 1,
        "room": {
            "geometry_mode": "exact_polygon" if polygon_room else "rectangle",
            "opening_types": opening_types,
        },
        "inventory": [
            {"category": category, "required_count": int(count), "source": "prompt"}
            for category, count in sorted(required_counts.items())
        ],
        "hard_constraints": [
            "keep every furniture footprint inside the authoritative room boundary",
            "keep doors and open connections clear and preserve primary circulation",
            "avoid furniture-furniture collisions and keep floor furniture supported",
            "preserve explicitly required inventory and functional relationships",
            "orient chairs toward tables/desks and storage fronts toward usable "
            "room space",
        ],
        "conditional_constraints": [
            "prefer the bed headboard on a solid wall without an opening when "
            "available",
            "keep paired nightstands beside the bed headboard and aligned with the bed",
            "do not place tall storage across windows",
            "keep service-counter staff workspace clear",
        ],
        "soft_preferences": [
            "preserve natural light, coherent grouping, realistic spacing, and "
            "visual balance"
        ],
        "conflict_priority": [
            "authoritative_geometry",
            "explicit_prompt",
            "safety_and_function",
            "reference_layout",
            "soft_plausibility",
        ],
    }


def format_furniture_layout_constraint_contract(contract: dict[str, Any]) -> str:
    """Render a compact shared view without tool or workflow instructions."""
    inventory = contract.get("inventory") or []
    inventory_text = (
        ", ".join(f"{item['category']} x{item['required_count']}" for item in inventory)
        or "follow the scene request"
    )
    core_rules = "; ".join(contract["hard_constraints"])
    conditional_rules = "; ".join(contract["conditional_constraints"])
    preferences = "; ".join(contract.get("soft_preferences") or [])
    return (
        f"Layout contract: {contract['room']['geometry_mode']} room; requested "
        f"inventory: {inventory_text}. Required: {core_rules}. When applicable: "
        f"{conditional_rules}. Prefer: {preferences}. Resolve conflicts in this "
        f"order: {' > '.join(contract['conflict_priority'])}."
    )
