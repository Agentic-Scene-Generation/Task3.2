"""Canonical registry for prompt-originated functional relations.

The registry is intentionally data-only.  TaskCompiler validation, contract
binding, evaluators, prompt formatting, and repair dispatch all consume the
same finite relation vocabulary instead of maintaining parallel if/elif
allowlists.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping


STAGE_ORDER = (
    "floor_plan",
    "furniture",
    "wall_mounted",
    "ceiling_mounted",
    "manipuland",
    "final",
)

# These labels describe a wall relative to the room, not a generated object
# category.  TaskCompiler can preserve them as an intent target, but binding
# must resolve them through the scene's real wall objects.
ROOM_RELATIVE_WALL_CATEGORIES = frozenset(
    {
        "back_wall",
        "front_wall",
        "side_wall",
        "main_wall",
        "opposite_wall",
        "adjacent_wall",
    }
)

# Canonical categories with a structural owner outside the furniture stage.
# They are shared by TaskCompiler inventory normalization and contract staging
# so a relation cannot be evaluated before either endpoint can exist.
WALL_MOUNTED_CATEGORIES = frozenset(
    {
        "instructional_surface",
        "painting",
        "mirror",
        "wall_shelf",
        "wall_light",
    }
)
CEILING_MOUNTED_CATEGORIES = frozenset(
    {"ceiling_light", "ceiling_fan", "chandelier", "pendant_light"}
)
MANIPULAND_CATEGORIES = frozenset(
    {
        "alarm_clock",
        "bedside_lamp",
        "book",
        "bottle",
        "bowl",
        "coaster",
        "computer_display",
        "computer_monitor",
        "cup",
        "cutlery",
        "desk_lamp",
        "flower",
        "glass",
        "keyboard",
        "laptop",
        "magazine",
        "mug",
        "monitor",
        "notebook",
        "plate",
        "pen",
        "pen_holder",
        "remote",
        "table_lamp",
        "table_setting",
        "trash_can",
        "vase",
        "wastebasket",
    }
)


@dataclass(frozen=True)
class RelationSpec:
    name: str
    target_arity: int
    earliest_stage: str
    evaluator: str
    prompt_description: str
    dependencies: tuple[str, ...] = ()
    dependency_binding: str = "same_endpoints"
    thresholds: Mapping[str, float] = field(default_factory=dict)
    repair_strategy: str | None = None
    repair_relation_types: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.target_arity not in {0, 1, 2}:
            raise ValueError(f"Invalid target arity for {self.name!r}")
        if self.earliest_stage not in STAGE_ORDER:
            raise ValueError(f"Invalid earliest stage for {self.name!r}")
        if not self.evaluator:
            raise ValueError(f"Relation {self.name!r} has no evaluator")
        if self.dependency_binding not in {
            "same_endpoints",
            "subject",
            "any_endpoint",
        }:
            raise ValueError(
                f"Invalid dependency binding for {self.name!r}: "
                f"{self.dependency_binding!r}"
            )
        if any(
            not relation_type.strip() for relation_type in self.repair_relation_types
        ):
            raise ValueError(f"Relation {self.name!r} has an empty repair relation")


def _relation(
    name: str,
    target_arity: int,
    earliest_stage: str,
    evaluator: str,
    prompt_description: str,
    *,
    dependencies: tuple[str, ...] = (),
    dependency_binding: str = "same_endpoints",
    thresholds: Mapping[str, float] | None = None,
    repair_strategy: str | None = None,
    repair_relation_types: tuple[str, ...] = (),
) -> RelationSpec:
    return RelationSpec(
        name=name,
        target_arity=target_arity,
        earliest_stage=earliest_stage,
        evaluator=evaluator,
        prompt_description=prompt_description,
        dependencies=dependencies,
        dependency_binding=dependency_binding,
        thresholds=MappingProxyType(dict(thresholds or {})),
        repair_strategy=repair_strategy,
        repair_relation_types=repair_relation_types,
    )


_RELATIONS = (
    _relation(
        "required_count", 0, "furniture", "required_count", "required object count"
    ),
    _relation(
        "against_wall",
        1,
        "furniture",
        "back_against_wall",
        "object backed against a wall",
        repair_strategy="furniture_relation",
        repair_relation_types=(
            "back_against_wall",
            "wall_backed_storage_alignment",
        ),
    ),
    _relation(
        "centered_on_wall",
        1,
        "furniture",
        "centered_on_wall",
        "object centered on a wall",
        dependencies=("against_wall",),
        dependency_binding="subject",
        repair_strategy="furniture_relation",
    ),
    _relation(
        "centered_in_room",
        1,
        "furniture",
        "centered_in_room",
        "object centered in the room",
        repair_strategy="furniture_relation",
        repair_relation_types=("room_center_alignment",),
    ),
    _relation(
        "centered_between",
        2,
        "furniture",
        "centered_between",
        "subject centered between two anchors",
        repair_strategy="furniture_relation",
        repair_relation_types=("centered_between_alignment",),
    ),
    _relation(
        "between",
        2,
        "furniture",
        "between",
        "subject lies between two anchors",
        repair_strategy="furniture_relation",
        repair_relation_types=("between_alignment",),
    ),
    _relation(
        "in_front_of",
        1,
        "furniture",
        "in_front_of",
        "subject in front of target",
        repair_strategy="furniture_relation",
        repair_relation_types=("front_axis_alignment",),
    ),
    _relation(
        "behind",
        1,
        "furniture",
        "behind",
        "subject behind target along the target's rear axis",
        repair_strategy="furniture_relation",
        repair_relation_types=("rear_axis_alignment",),
    ),
    _relation(
        "faces",
        1,
        "furniture",
        "faces",
        "subject faces target",
        dependencies=("near",),
        thresholds={"max_angle_deg": 60.0},
        repair_strategy="furniture_relation",
        repair_relation_types=(
            "faces",
            "furniture_faces_furniture",
            "front_axis_alignment",
            "seating_to_work_surface",
            "workstation_focal_alignment",
        ),
    ),
    _relation(
        "on_top_of",
        1,
        "furniture",
        "object_on_support",
        "subject supported on target",
        repair_strategy="support_relation",
    ),
    _relation(
        "near",
        1,
        "furniture",
        "generic_near_relation",
        "subject near target",
        thresholds={"max_gap_m": 1.5},
        repair_strategy="furniture_relation",
        repair_relation_types=("generic_near_relation",),
    ),
    _relation(
        "next_to",
        1,
        "furniture",
        "generic_near_relation",
        "subject next to target",
        thresholds={"max_gap_m": 0.8},
        repair_strategy="furniture_relation",
        repair_relation_types=("generic_near_relation",),
    ),
    _relation(
        "across_from",
        1,
        "furniture",
        "across_from",
        "subject across from and facing target",
    ),
    _relation(
        "aligned_with",
        1,
        "furniture",
        "aligned_with",
        "subject aligned with target",
        repair_strategy="furniture_relation",
        repair_relation_types=(
            "furniture_faces_furniture",
            "seating_to_work_surface",
        ),
    ),
    _relation(
        "flanking",
        1,
        "furniture",
        "flanking",
        "subjects flank target",
        repair_strategy="furniture_relation",
    ),
    _relation("surround", 1, "furniture", "surround", "subjects surround target"),
    _relation(
        "paired_with",
        1,
        "furniture",
        "paired_with",
        "one-to-one functional pairing",
        dependencies=("required_count",),
        dependency_binding="any_endpoint",
        repair_strategy="furniture_relation",
        repair_relation_types=(
            "furniture_faces_furniture",
            "seating_to_work_surface",
        ),
    ),
    _relation(
        "distributed_evenly",
        1,
        "furniture",
        "distributed_evenly",
        "subjects distributed evenly",
        dependencies=("required_count",),
        dependency_binding="any_endpoint",
    ),
    _relation(
        "edge_distribution",
        1,
        "furniture",
        "edge_distribution",
        "subjects distributed across the finite long/short edges of a rectangular target",
        dependencies=("required_count",),
        dependency_binding="any_endpoint",
        repair_strategy="edge_distribution",
        repair_relation_types=("edge_distribution",),
    ),
    _relation(
        "corner_of_room",
        1,
        "furniture",
        "corner_of_room",
        "subject placed in a room corner",
        repair_strategy="furniture_relation",
    ),
    _relation(
        "on_wall", 1, "wall_mounted", "mounted_to_wall", "subject mounted on a wall"
    ),
    _relation(
        "hang_from_ceiling",
        1,
        "ceiling_mounted",
        "mounted_to_ceiling",
        "subject attached to the ceiling",
    ),
    _relation(
        "clear_access",
        1,
        "furniture",
        "clear_access",
        "subject has an unobstructed interaction zone",
        dependencies=("required_count",),
        dependency_binding="any_endpoint",
        thresholds={"min_clearance_m": 0.8},
        repair_strategy="furniture_relation",
    ),
    _relation(
        "operation_zone_at_wall",
        1,
        "furniture",
        "operation_zone_at_wall",
        "workstation operation zone faces inward from wall",
        repair_strategy="furniture_relation",
    ),
    _relation(
        "instructional_surface_alignment",
        1,
        "wall_mounted",
        "instructional_surface_alignment",
        "instructional surface aligns with presenter workstation",
        dependencies=("operation_zone_at_wall",),
        dependency_binding="any_endpoint",
        repair_strategy="furniture_relation",
    ),
)

RELATION_REGISTRY: Mapping[str, RelationSpec] = MappingProxyType(
    {relation.name: relation for relation in _RELATIONS}
)
PUBLIC_RELATIONS = frozenset(RELATION_REGISTRY)


def relation_spec(name: str) -> RelationSpec:
    try:
        return RELATION_REGISTRY[str(name)]
    except KeyError as exc:
        raise ValueError(
            f"Unknown intent relation {name!r}; choose from {sorted(PUBLIC_RELATIONS)}"
        ) from exc


def repair_relation_types(
    *, strategies: tuple[str, ...] | None = None
) -> frozenset[str]:
    """Return evaluator result types owned by the requested repair strategies."""
    selected = set(strategies) if strategies is not None else None
    return frozenset(
        relation_type
        for spec in RELATION_REGISTRY.values()
        if spec.repair_strategy is not None
        and (selected is None or spec.repair_strategy in selected)
        for relation_type in (spec.repair_relation_types or (spec.evaluator,))
    )


def validate_relation_registry() -> None:
    """Fail fast when registry metadata is incomplete or inconsistent."""
    for name, spec in RELATION_REGISTRY.items():
        if name != spec.name:
            raise ValueError(f"Relation registry key mismatch: {name!r}")
        for dependency in spec.dependencies:
            if dependency not in RELATION_REGISTRY:
                raise ValueError(
                    f"Relation {name!r} depends on unknown relation {dependency!r}"
                )
        if spec.repair_strategy is not None and not spec.repair_strategy.strip():
            raise ValueError(f"Relation {name!r} has an empty repair strategy")


def relation_registry_payload() -> dict[str, dict[str, Any]]:
    """Return an audit-safe JSON representation of the registry."""
    return {
        name: {
            "target_arity": spec.target_arity,
            "earliest_stage": spec.earliest_stage,
            "evaluator": spec.evaluator,
            "prompt_description": spec.prompt_description,
            "dependencies": list(spec.dependencies),
            "dependency_binding": spec.dependency_binding,
            "thresholds": dict(spec.thresholds),
            "repair_strategy": spec.repair_strategy,
            "repair_relation_types": list(spec.repair_relation_types),
        }
        for name, spec in RELATION_REGISTRY.items()
    }


validate_relation_registry()
