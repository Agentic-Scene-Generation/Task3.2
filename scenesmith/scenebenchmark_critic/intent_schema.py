"""Typed v5 schema for the independent SceneBenchmark intent contract.

The contract is deliberately separate from :class:`SceneTaskSpec`.  The task
compiler describes inventory and stage ownership; this module describes hard
functional relations that the critic is allowed to enforce.
"""

from __future__ import annotations

import hashlib
import json
import re
from copy import deepcopy
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from scenesmith.scenebenchmark_critic.object_taxonomy import (
    canonical_object_category,
    execution_owner,
)
from scenesmith.scenebenchmark_critic.relation_registry import (
    RELATION_REGISTRY,
    STAGE_ORDER,
    relation_spec,
    relations_are_exclusive,
)


_GENERIC_SELECTOR_FAMILIES = {
    "chair": frozenset(
        {
            "chair",
            "office_chair",
            "guest_chair",
            "student_chair",
            "dining_chair",
            "armchair",
            "stool",
        }
    ),
    "table": frozenset(
        {
            "table",
            "conference_table",
            "dining_table",
            "coffee_table",
            "side_table",
            "desk",
            "dressing_table",
        }
    ),
    "desk": frozenset({"desk", "student_desk", "teacher_desk", "reception_desk"}),
    "plant": frozenset({"plant", "large_plant", "potted_plant"}),
    "sofa": frozenset({"sofa", "two_seater_sofa", "loveseat", "sectional_sofa"}),
    "speaker": frozenset({"speaker", "floor_speaker"}),
}


def canonical_selector_category(value: Any) -> str:
    """Normalize compiler selector spellings to stable semantic categories."""
    return canonical_object_category(value)


def selector_categories_overlap(first: str, second: str) -> bool:
    """Return whether two selector categories can denote the same object."""
    if first == second:
        return True
    return any(
        (first == generic and second in members)
        or (second == generic and first in members)
        for generic, members in _GENERIC_SELECTOR_FAMILIES.items()
    )


# Keep the private spelling for existing schema validation call sites.
_selector_categories_overlap = selector_categories_overlap


INTENT_CONTRACT_SCHEMA_VERSION = "scenesmith.intent_contract.v5"
INTENT_COMPILER_SPEC_VERSION = "scenesmith.intent_compiler.v13"

_WALL_QUALIFIED_DIRECTION_PATTERN = re.compile(
    r"(?P<subject>[^,.;!?]{1,100}?)\s+against\s+"
    r"(?:the\s+|a\s+|an\s+)?(?:[a-z]+\s+){0,3}?wall\s+"
    r"(?:behind|in\s+front\s+of)\s+(?:the\s+|a\s+|an\s+)?"
    r"(?P<target>[^,.;!?]{1,100})",
    re.IGNORECASE,
)


def _selector_category_mentioned(text: str, category: str) -> bool:
    """Match a selector's concrete noun in a compact prompt clause."""
    normalized = canonical_selector_category(category).replace("_", " ")
    candidates = [normalized]
    if " " in normalized:
        candidates.append(normalized.rsplit(" ", 1)[-1])
    return any(
        re.search(
            rf"(?<![a-z0-9]){re.escape(candidate)}(?:s|es)?(?![a-z0-9])",
            text,
            re.IGNORECASE,
        )
        is not None
        for candidate in candidates
    )


def _is_wall_qualified_directional_relation(
    prompt: str, relation: "IntentRelation"
) -> bool:
    """Whether a directional row mistakes a wall qualifier for object layout."""
    if relation.relation not in {"behind", "in_front_of"} or not relation.targets:
        return False
    for match in _WALL_QUALIFIED_DIRECTION_PATTERN.finditer(prompt):
        if _selector_category_mentioned(
            match.group("subject"), relation.subjects.category
        ) and _selector_category_mentioned(
            match.group("target"), relation.targets.category
        ):
            return True
    return False


class IntentSelector(BaseModel):
    """Semantic selector used by a relation endpoint."""

    model_config = ConfigDict(extra="forbid")

    category: str = Field(min_length=1)
    count: int | None = Field(default=None, ge=1)
    quantifier: Literal["all", "exactly", "at_least", "minimum"] = "all"
    role: str = ""
    secondary_category: str = ""
    secondary_count: int | None = Field(default=None, ge=1)
    secondary_role: str = ""

    @field_validator("category", "secondary_category", mode="before")
    @classmethod
    def _normalize_category(cls, value: Any) -> str:
        return canonical_selector_category(value)

    @field_validator("role", "secondary_role", mode="before")
    @classmethod
    def _normalize_role(cls, value: Any) -> str:
        return "_".join(str(value or "").strip().lower().split())


class EdgeDistributionGroup(BaseModel):
    """Counts for the two unoriented edges in one edge class."""

    model_config = ConfigDict(extra="forbid")

    edge_class: Literal["long", "short"]
    counts_per_edge: list[int] = Field(min_length=2, max_length=2)
    spacing: Literal["equal_segments", "unconstrained"] = "equal_segments"

    @field_validator("counts_per_edge", mode="before")
    @classmethod
    def _normalize_counts(cls, value: Any) -> list[int]:
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            return value
        counts = [int(item) for item in value]
        if any(item < 0 for item in counts):
            raise ValueError("counts_per_edge values must be non-negative")
        return sorted(counts, reverse=True)


class IntentRelation(BaseModel):
    """One schema-validated hard relation."""

    model_config = ConfigDict(extra="forbid")

    relation: str
    constraint_id: str = ""
    stage: str = ""
    strength: Literal["hard"] = "hard"
    subjects: IntentSelector
    targets: IntentSelector | None = None
    edge_frame: Literal["target_local_rectangle"] | None = None
    groups: list[EdgeDistributionGroup] = Field(default_factory=list)
    orientation: (
        Literal[
            "toward_target",
            "away_from_target",
            "parallel_to_edge",
            "unconstrained",
        ]
        | None
    ) = None
    source: Literal[
        "explicit_prompt",
        "task_compiler_inventory",
        "model_inferred",
        "room_ontology",
        "deterministic_fallback",
    ]
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    evidence_span: str = ""
    inference_reason: str = ""
    reconciliation_reason: Literal["disjoint_support_cohort_minimum"] | None = None

    @field_validator("relation", mode="before")
    @classmethod
    def _normalize_relation(cls, value: Any) -> str:
        relation = str(value or "").strip().lower()
        if relation == "one_per_side":
            raise ValueError("one_per_side was removed; use edge_distribution")
        relation_spec(relation)
        return relation

    @model_validator(mode="after")
    def _validate_relation_shape(self) -> "IntentRelation":
        if self.source == "explicit_prompt" and not self.evidence_span.strip():
            raise ValueError("explicit_prompt relations require evidence_span")
        if self.source == "model_inferred" and not self.inference_reason.strip():
            raise ValueError("model_inferred relations require inference_reason")
        if (
            self.source == "task_compiler_inventory"
            and not self.inference_reason.strip()
        ):
            raise ValueError(
                "task_compiler_inventory relations require inventory provenance"
            )
        if self.stage and self.stage not in STAGE_ORDER:
            raise ValueError(f"Unknown intent contract stage: {self.stage!r}")

        # A singular target can still be existential when the prompt says
        # "one of two chairs" or "beside one armchair". Keeping the default
        # ``all`` quantifier here makes a valid relation fail as soon as the
        # room contains multiple candidates, even though any one candidate
        # can satisfy the relation. Subjects retain their declared
        # cardinality; this normalization only applies to the target endpoint.
        if (
            self.targets is not None
            and self.targets.count == 1
            and self.targets.quantifier == "all"
            and _evidence_names_one_target(self.evidence_span, self.targets.category)
        ):
            self.targets.quantifier = "minimum"

        spec = relation_spec(self.relation)
        if self.relation == "edge_distribution":
            if (
                self.targets is None
                or self.targets.count != 1
                or self.targets.secondary_category
                or self.targets.secondary_count is not None
                or self.targets.secondary_role
            ):
                raise ValueError(
                    "edge_distribution target must select exactly one object"
                )
            if self.edge_frame != "target_local_rectangle":
                raise ValueError("edge_distribution requires target_local_rectangle")
            if self.orientation is None:
                raise ValueError("edge_distribution requires orientation")
            if not self.groups:
                raise ValueError("edge_distribution requires at least one edge group")
            classes = [group.edge_class for group in self.groups]
            if len(classes) != len(set(classes)):
                raise ValueError("edge_distribution edge_class may appear only once")
            total = sum(sum(group.counts_per_edge) for group in self.groups)
            if self.subjects.count != total:
                raise ValueError(
                    "edge_distribution subjects.count must equal edge counts sum"
                )
        else:
            if (
                self.groups
                or self.edge_frame is not None
                or self.orientation is not None
            ):
                raise ValueError(
                    f"relation {self.relation!r} cannot contain edge distribution fields"
                )
            target_arity = 0
            if self.targets is not None:
                target_arity = 2 if self.targets.secondary_category else 1
            if target_arity != spec.target_arity:
                raise ValueError(
                    f"Relation {self.relation!r} requires {spec.target_arity} target(s), "
                    f"got {target_arity}"
                )
        return self


def _evidence_names_one_target(evidence: str, category: str) -> bool:
    """Return whether evidence selects one member of a target category."""
    normalized_category = " ".join(str(category or "").replace("_", " ").split())
    if not evidence or not normalized_category:
        return False
    noun = normalized_category.rsplit(" ", 1)[-1]
    return (
        re.search(
            rf"\b(?:one|any|either|another|other)\s+"
            rf"(?:[a-z0-9_-]+\s+){{0,3}}{re.escape(noun)}(?:s|es)?\b",
            evidence.lower(),
        )
        is not None
    )


class IntentContract(BaseModel):
    """Complete independent intent compiler output."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[INTENT_CONTRACT_SCHEMA_VERSION] = (
        INTENT_CONTRACT_SCHEMA_VERSION
    )
    prompt: str = ""
    prompt_sha256: str = ""
    intent_compiler_spec_version: str = INTENT_COMPILER_SPEC_VERSION
    room_type: str = ""
    constraints: list[IntentRelation] = Field(default_factory=list)
    retry_count: int = Field(default=0, ge=0, le=1)
    warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_cross_relation_invariants(self) -> "IntentContract":
        for relation in self.constraints:
            if _is_wall_qualified_directional_relation(self.prompt, relation):
                raise ValueError(
                    "wall-relative directional relation is invalid: in 'X against "
                    "the wall behind/in front of Y', the direction qualifies the "
                    "wall rather than the X-to-Y layout"
                )

        required_counts: dict[str, int] = {}
        for relation in self.constraints:
            if relation.relation != "required_count":
                continue
            category = relation.subjects.category
            count = relation.subjects.count
            if count is None:
                raise ValueError(
                    f"required_count must declare a positive count for {category!r}"
                )
            previous = required_counts.get(category)
            if previous is not None and previous != count:
                raise ValueError(
                    "conflicting required_count relations for "
                    f"{category!r}: {previous} and {count}"
                )
            required_counts[category] = count

        for relation in self.constraints:
            if relation.relation != "edge_distribution":
                continue
            expected = required_counts.get(relation.subjects.category)
            if expected is not None and expected != relation.subjects.count:
                raise ValueError(
                    "edge_distribution subjects.count must match required_count "
                    f"for {relation.subjects.category!r}"
                )

        for relation in self.constraints:
            if relation.relation == "one_per_support":
                if relation.targets is None:
                    continue
                subject_count = relation.subjects.count
                target_count = relation.targets.count
                if (
                    subject_count is None
                    or target_count is None
                    or subject_count != target_count
                ):
                    raise ValueError(
                        "one_per_support requires equal explicit subject and target counts"
                    )
            elif relation.relation == "corner_distribution":
                if (
                    relation.targets is None
                    or relation.targets.category != "room"
                    or (relation.subjects.count or 0) < 2
                ):
                    raise ValueError(
                        "corner_distribution requires at least two subjects and a room target"
                    )

        for index, first in enumerate(self.constraints):
            for second in self.constraints[index + 1 :]:
                if not relations_are_exclusive(first.relation, second.relation):
                    continue
                if not _selector_categories_overlap(
                    first.subjects.category, second.subjects.category
                ):
                    continue
                first_role = first.subjects.role
                second_role = second.subjects.role
                if first_role and second_role and first_role != second_role:
                    continue
                raise ValueError(
                    "conflicting hard relations for overlapping subject selector "
                    f"{first.subjects.category!r}: {first.relation!r} and "
                    f"{second.relation!r}"
                )

        edge_pairs = {
            (
                relation.subjects.category,
                relation.targets.category if relation.targets else "",
            )
            for relation in self.constraints
            if relation.relation == "edge_distribution"
            and relation.orientation == "toward_target"
        }
        for relation in self.constraints:
            if relation.relation != "faces" or relation.targets is None:
                continue
            if any(
                _selector_categories_overlap(
                    relation.subjects.category,
                    edge_subject,
                )
                and _selector_categories_overlap(
                    relation.targets.category,
                    edge_target,
                )
                for edge_subject, edge_target in edge_pairs
            ):
                raise ValueError(
                    "edge_distribution.toward_target cannot duplicate faces for "
                    f"{relation.subjects.category!r}->{relation.targets.category!r}"
                )
        return self


def validate_intent_contract(payload: dict[str, Any]) -> dict[str, Any]:
    """Validate and normalize a wire payload, returning JSON-compatible data."""

    contract = IntentContract.model_validate(payload)
    result = contract.model_dump(mode="json", exclude_none=True)
    constraints = result.get("constraints") or []
    seen: dict[str, int] = {}
    for constraint in constraints:
        relation = str(constraint.get("relation") or "")
        stage = str(constraint.get("stage") or "")
        stage = (
            stage if stage in STAGE_ORDER else relation_spec(relation).earliest_stage
        )
        subjects = constraint.get("subjects") or {}
        targets = constraint.get("targets") or {}
        endpoint_stages = [
            execution_owner(
                subjects.get("category"), relation=relation, endpoint="subject"
            ),
            execution_owner(
                targets.get("category"), relation=relation, endpoint="target"
            ),
            execution_owner(
                targets.get("secondary_category"),
                relation=relation,
                endpoint="target",
            ),
        ]
        constraint["stage"] = max(
            [stage, *(value for value in endpoint_stages if value)],
            key=STAGE_ORDER.index,
        )
        identity = dict(constraint)
        identity.pop("constraint_id", None)
        identity.pop("stage", None)
        digest = hashlib.sha1(
            json.dumps(
                identity,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                default=str,
            ).encode("utf-8")
        ).hexdigest()[:12]
        occurrence = seen.get(digest, 0) + 1
        seen[digest] = occurrence
        constraint["constraint_id"] = (
            f"intent_{digest}" if occurrence == 1 else f"intent_{digest}_{occurrence}"
        )
    return result


def intent_contract_json_schema() -> dict[str, Any]:
    """Return an LLM schema that enforces relation-specific target arity.

    Pydantic's ``IntentRelation`` must keep ``targets`` optional because count
    relations have no endpoint.  A plain model schema consequently lets an LLM
    omit ``targets`` for every relation, including ``flanking`` and
    ``corner_of_room``.  Encode the registry's target arity in JSON Schema as
    well, so decoders that honor JSON Schema conditionals require an object
    target whenever the selected relation has an endpoint.  The compiler keeps
    a separate post-parse safeguard for llama.cpp's more limited grammar.
    """

    schema = deepcopy(IntentContract.model_json_schema())
    schema.setdefault("required", [])
    if "constraints" not in schema["required"]:
        schema["required"].append("constraints")
    relation_schema = schema["$defs"]["IntentRelation"]
    relation_properties = relation_schema["properties"]
    relation_names_by_arity = {
        arity: sorted(
            name
            for name, spec in RELATION_REGISTRY.items()
            if spec.target_arity == arity
        )
        for arity in (0, 1, 2)
    }
    relation_properties["relation"] = {"enum": sorted(RELATION_REGISTRY)}
    # llama.cpp's grammar converter ignores JSON Schema if/then branches, but
    # it does honor a top-level required list. Requiring the key for every row
    # prevents the local model from silently omitting relation endpoints;
    # zero-arity relations use an explicit null value.
    relation_schema.setdefault("required", [])
    if "targets" not in relation_schema["required"]:
        relation_schema["required"].append("targets")

    endpoint_conditions: list[dict[str, Any]] = [
        {
            "if": {
                "properties": {"relation": {"enum": relation_names_by_arity[0]}},
                "required": ["relation"],
            },
            "then": {"properties": {"targets": {"type": "null"}}},
        }
    ]
    for arity in (1, 2):
        relation_names = relation_names_by_arity[arity]
        target_schema: dict[str, Any] = {"$ref": "#/$defs/IntentSelector"}
        if arity == 2:
            target_schema = {
                "allOf": [
                    target_schema,
                    {"required": ["secondary_category"]},
                ]
            }
        endpoint_conditions.append(
            {
                "if": {
                    "properties": {"relation": {"enum": relation_names}},
                    "required": ["relation"],
                },
                "then": {
                    "required": ["targets"],
                    "properties": {"targets": target_schema},
                },
            }
        )
    relation_schema["allOf"] = endpoint_conditions
    return schema


def intent_compiler_wire_json_schema() -> dict[str, Any]:
    """Return the compact, provenance-free schema generated by the LLM.

    Long evidence and inference strings are deterministic compiler metadata.
    Keeping them out of constrained generation prevents a malformed string from
    consuming the entire completion budget before the JSON object can close.
    """

    full_schema = intent_contract_json_schema()
    selector_schema = deepcopy(full_schema["$defs"]["IntentSelector"])
    for field in ("category", "role", "secondary_category", "secondary_role"):
        selector_schema["properties"][field]["maxLength"] = 64

    group_schema = deepcopy(full_schema["$defs"]["EdgeDistributionGroup"])
    relation_schema = deepcopy(full_schema["$defs"]["IntentRelation"])
    retained = {
        "relation",
        "subjects",
        "targets",
        "edge_frame",
        "groups",
        "orientation",
    }
    relation_schema["properties"] = {
        key: value
        for key, value in relation_schema["properties"].items()
        if key in retained
    }
    relation_schema["properties"]["grounding"] = {
        "type": "string",
        "pattern": r"^(prompt|interaction|aesthetic):[0-9]+$",
        "maxLength": 32,
    }
    relation_schema["required"] = ["relation", "subjects", "targets", "grounding"]
    relation_schema["additionalProperties"] = False

    allowed_relations = sorted(set(RELATION_REGISTRY) - {"required_count"})
    relation_schema["properties"]["relation"] = {"enum": allowed_relations}
    for condition in relation_schema.get("allOf") or []:
        relation_enum = (
            condition.get("if", {})
            .get("properties", {})
            .get("relation", {})
            .get("enum")
        )
        if relation_enum is not None:
            relation_enum[:] = [
                name for name in relation_enum if name != "required_count"
            ]
    relation_schema["allOf"] = [
        condition
        for condition in relation_schema.get("allOf") or []
        if condition.get("if", {}).get("properties", {}).get("relation", {}).get("enum")
    ]

    return {
        "$defs": {
            "EdgeDistributionGroup": group_schema,
            "IntentSelector": selector_schema,
            "IntentWireRelation": relation_schema,
        },
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "constraints": {
                "type": "array",
                "items": {"$ref": "#/$defs/IntentWireRelation"},
            }
        },
        "required": ["constraints"],
    }
