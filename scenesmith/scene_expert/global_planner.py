"""GlobalPlanner: Qwen3 global_planner role that generates a StageBrief for each stage.

Takes the HarnessContext (task spec + memory pack + scene state summary)
and produces expert planning hints to inject into SceneSmith's stage prompt.

The planner does NOT place objects or modify the scene — it only generates
a structured text brief for the SceneSmith designer agent.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time

from pathlib import Path
from typing import Any

from scenesmith.agent_utils.thinking import (
    chat_template_kwargs_from_effort,
    openrouter_extra_body,
    prepend_text_thinking_directive,
    thinking_directive_from_effort,
)
from scenesmith.scene_expert.context_bundle import build_llm_call_debug_record
from scenesmith.scene_expert.schemas import (
    HarnessContext,
    MemoryPack,
    SceneTaskSpec,
    StageBrief,
)
from scenesmith.utils.openai_strict_schema import make_openai_strict_json_schema

console_logger = logging.getLogger(__name__)


def _append_llm_debug(record: dict) -> None:
    path = os.environ.get("SCENEEXPERT_LLM_DEBUG_PATH", "")
    if not path:
        return
    try:
        debug_path = Path(path)
        debug_path.parent.mkdir(parents=True, exist_ok=True)
        with debug_path.open("a", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    except Exception as e:
        console_logger.warning("GlobalPlanner failed to write LLM debug record: %s", e)


_SYSTEM_PROMPT = """\
/no_think
You are the global_planner for SceneExpert, a 3D indoor scene generation system.
Your job is to generate a StageBrief — expert planning guidance for one stage of
a 3D scene generation pipeline powered by a downstream AI designer agent.

Stages in order: floor_plan → furniture → wall_mounted → ceiling_mounted → manipuland

You MUST output valid JSON matching this exact schema:
{
  "stage": "string — current stage name",
  "stage_policy": "auto or required_only — echo the supplied policy",
  "optional_assets_allowed": true,
  "required_objects": ["copy the prompt-explicit hard minimum for this stage"],
  "optional_asset_recommendations": [
    {
      "name": "same-stage asset category",
      "count": 1,
      "priority": "low, medium, or high",
      "rationale": "why this improves this specific room",
      "placement_guidance": "capacity-aware placement guidance"
    }
  ],
  "stage_objective": "string — one clear sentence describing the goal for this stage",
  "recommended_skills": ["list of skill names from memory to apply, can be empty"],
  "constraints_for_designer": [
    "list of concrete placement/arrangement rules for the designer",
    "be specific: use object names, spatial relations, measurements"
  ],
  "checks_for_critic": [
    "list of things the critic should verify after this stage"
  ],
  "failure_patterns_to_avoid": [
    "list of known failure patterns from memory — explicitly tell designer to avoid these"
  ]
}

Guidelines:
- Be specific and actionable. Vague guidance is useless for small models.
- Derive constraints from: the task spec, the current scene state, AND the retrieved memory.
- The Authoritative Stage Intent section contains the exact hard contract rows
  for this stage. Cover every constraint_id and never rewrite, weaken, or
  replace one with a convention.
- At floor_plan, Architectural Surface Reservations are hard prerequisites for
  later-stage wall anchors. Reserve a continuous wall segment for every listed
  object; doors and windows must not intersect a reserved segment. Do not move
  a later object away from its explicit required wall merely to keep an opening.
- After floor_plan, Resolved Opening Reservations contain authoritative room
  geometry. Keep furniture out of hard door/open clearances. Place tall or focal
  wall-backed furniture only on a continuous usable segment that does not cross
  a window; prefer a fully opening-free wall when it satisfies the task relation.
- Functional zones named only through later furniture (for example, living,
  dining, work, or storage areas) are not architectural partitions. At
  floor_plan, make those later arrangements feasible through dimensions,
  circulation, and opening-free wall length. Do not invent rooms, partitions,
  or other structural markers unless the Immutable User Task explicitly asks
  for structural separation.
- Prioritize failure patterns from memory — they encode hard-won lessons.
- When an Immutable User Task is supplied, its explicit object, topology, and
  facing relations are authoritative. Memory and current-scene observations may
  refine only non-conflicting details. Omit a suggestion instead of replacing an
  explicit user relation with a design convention.
- The Task Specification owns objects by stage. Never create, move, or
  reinterpret an object assigned to another stage. A non-empty "Required objects
  for this stage" inventory is a minimum, not an exhaustive list: same-stage
  supporting objects may be suggested when they improve function or completeness,
  unless the Immutable User Task explicitly forbids them.
- Under stage_policy=auto, always preserve every required object and independently
  recommend 0-6 useful OPTIONAL same-stage assets based on room type, style, usable
  capacity, openings, circulation, support surfaces, and already placed objects.
  An empty required inventory means there is no hard minimum; it never disables
  the stage and never makes the brief observation-only.
- Under stage_policy=required_only, return no optional recommendations. The native
  stage still executes and verifies the required inventory, even when it is empty.
- Optional recommendations are soft. Never phrase them as required, never let them
  displace a required asset, and omit them when capacity or constraints are unsafe.
- Never turn a non-empty required inventory into an "only these objects" rule
  unless that exclusivity appears explicitly in the Immutable User Task.
- Keep constraints_for_designer to 3-6 items max. More is not better.
- The designer will read this brief directly — write for it, not for humans.
- Output ONLY the JSON object, no other text.
"""

_STAGE_DESCRIPTIONS = {
    "floor_plan": "Generate room geometry: walls, doors, windows, and room dimensions.",
    "furniture": "Place large furniture (beds, sofas, tables, wardrobes) in the room.",
    "wall_mounted": "Place wall-mounted objects (paintings, mirrors, shelves, lights) on walls.",
    "ceiling_mounted": "Place ceiling-mounted objects (lights, fans) on the ceiling.",
    "manipuland": "Place small manipulable objects (books, cups, plants) on furniture surfaces.",
}

_STAGE_CRITIC_CHECKS = {
    "floor_plan": [
        "Verify room footprint, dimensions, wall geometry, and floor layout",
        "Verify doors/openings provide valid access and room connectivity",
        "Verify windows/openings have architecturally reasonable placement and daylight",
        "Do not evaluate furniture, wall decor, ceiling fixtures, or manipulands",
    ],
    "furniture": [
        "Verify prompt-required furniture is present with correct semantic assets",
        "Verify furniture poses, orientation, functional relationships, and clearance",
        "Verify furniture is collision-free, reachable, and inside the room",
        "Do not require wall decor, ceiling fixtures, or manipulands",
    ],
    "wall_mounted": [
        "Verify stage-required wall-mounted objects are present and correctly mounted",
        "Verify mounting height, spacing, opening clearance, and visual balance",
        "Do not redesign furniture or require ceiling/manipuland objects",
    ],
    "ceiling_mounted": [
        "Verify stage-required ceiling objects are present and correctly mounted",
        "Verify coverage, spacing, clearance, and visual balance",
        "Do not redesign upstream stages or require manipulands",
    ],
    "manipuland": [
        "Verify stage-required small objects are present on valid support surfaces",
        "Verify support, usability, local spacing, and collision-free placement",
        "Treat all upstream architecture and furniture as fixed context",
    ],
}

_COMMON_DOWNSTREAM_OBJECT_TERMS = {
    "furniture",
    "wall decor",
    "ceiling fixture",
    "manipuland",
    "bed",
    "nightstand",
    "wardrobe",
    "dresser",
    "sofa",
    "couch",
    "table",
    "chair",
    "desk",
    "rug",
    "plant",
    "cabinet",
    "shelf",
    "books",
    "cup",
}


def _contains_term(text: str, term: str) -> bool:
    normalized_term = str(term or "").casefold().strip()
    if not normalized_term:
        return False
    return bool(
        re.search(
            rf"(?<![a-z0-9_]){re.escape(normalized_term)}(?![a-z0-9_])",
            text,
        )
    )


def _floor_plan_rule_mentions_downstream_placement(
    text: str,
    task_spec: SceneTaskSpec,
) -> bool:
    """Reject floor-plan hints that attempt downstream object placement."""
    lowered = str(text or "").casefold()
    object_terms = {
        *_COMMON_DOWNSTREAM_OBJECT_TERMS,
        *(
            str(value).casefold()
            for value in (
                task_spec.required_large_objects
                + task_spec.required_wall_objects
                + task_spec.required_ceiling_objects
                + task_spec.required_small_objects
            )
            if str(value).strip()
        ),
    }
    if not any(_contains_term(lowered, term) for term in object_terms):
        return False
    capacity_only = any(
        marker in lowered
        for marker in ("reserve", "capacity", "accommodate", "plan space")
    ) and any(
        marker in lowered for marker in ("do not place", "downstream", "later stage")
    )
    return not capacity_only


def enforce_stage_brief_scope(
    stage_brief: StageBrief,
    task_spec: SceneTaskSpec,
) -> StageBrief:
    """Keep planner output within deterministic pipeline-stage ownership."""
    stage = stage_brief.stage
    constraints = list(stage_brief.constraints_for_designer)
    failures = list(stage_brief.failure_patterns_to_avoid)
    recommended_skills = list(stage_brief.recommended_skills)
    stage_objective = stage_brief.stage_objective
    if stage == "floor_plan":
        stage_objective = (
            f"Complete the floor_plan stage for a {task_spec.room_type}: "
            f"{_STAGE_DESCRIPTIONS['floor_plan']}"
        )
        constraints = [
            value
            for value in constraints
            if not _floor_plan_rule_mentions_downstream_placement(value, task_spec)
        ]
        recommended_skills = [
            value
            for value in recommended_skills
            if not _floor_plan_rule_mentions_downstream_placement(value, task_spec)
        ]
        failures = [
            value
            for value in failures
            if not _floor_plan_rule_mentions_downstream_placement(value, task_spec)
        ]
    return stage_brief.model_copy(
        update={
            "stage_objective": stage_objective,
            "recommended_skills": recommended_skills,
            "constraints_for_designer": constraints,
            "checks_for_critic": list(
                _STAGE_CRITIC_CHECKS.get(stage, stage_brief.checks_for_critic)
            ),
            "failure_patterns_to_avoid": failures,
        }
    )


_INVENTORY_CLOSING_PATTERNS = (
    re.compile(
        r"\b(?:do not|don't|must not|never)\s+"
        r"(?:add|create|include|place|generate)\s+"
        r"(?:any\s+)?(?:other|additional|extra)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bavoid\s+(?:adding|creating|including|placing|generating)\s+"
        r"(?:any\s+)?(?:other|additional|extra)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:no|without)\s+(?:other|additional|extra)\s+"
        r"(?:furniture|objects?|items?|fixtures?|decor(?:ation)?s?)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bonly\s+(?:the\s+)?"
        r"(?!(?:furniture|objects?|items?|fixtures?|decorations?)\b)"
        r"[a-z0-9][a-z0-9 _-]{0,48}\s+(?:is|are)\s+"
        r"(?:required|needed)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:add|create|include|place|generate)\s+only\s+(?:the\s+)?"
        r"(?!(?:furniture|objects?|items?|fixtures?|decorations?)\b)"
        r"[a-z0-9][a-z0-9 _-]{0,48}",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:limit|restrict)\s+(?:this\s+stage|the\s+stage|placements?|inventory)"
        r"\s+to\s+(?:the\s+)?required\b",
        re.IGNORECASE,
    ),
)

_EXPLICITLY_CLOSED_TASK_PATTERNS = (
    re.compile(
        r"\b(?:no|without)\s+(?:other|additional|extra)\s+"
        r"(?:furniture|objects?|items?|fixtures?|decor(?:ation)?s?)\b",
        re.IGNORECASE,
    ),
    re.compile(r"\bnothing\s+else\b", re.IGNORECASE),
)

_ONLY_INVENTORY_CLAUSE_PATTERNS = (
    re.compile(
        r"\b(?:with|containing|contains|including|includes|featuring|features)"
        r"\s+only\b(?P<inventory>[^.\n]{0,100})",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:furnished|decorated|equipped)\s+(?:only|solely|exclusively)\s+with\b"
        r"(?P<inventory>[^.\n]{0,100})",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:consists?|consisting)\s+(?:only|solely|exclusively)\s+of\b"
        r"(?P<inventory>[^.\n]{0,100})",
        re.IGNORECASE,
    ),
)

_STRUCTURAL_CEILING_TERMS = frozenset(
    {"beam", "beams", "joist", "joists", "rafter", "rafters"}
)
_LIGHTING_TERMS = frozenset(
    {
        "candelabra",
        "chandelier",
        "chandeliers",
        "downlight",
        "downlights",
        "fixture",
        "fixtures",
        "lamp",
        "lamps",
        "light",
        "lights",
        "lighting",
        "pendant",
        "pendants",
        "recessed",
        "sconce",
        "sconces",
    }
)

_STRUCTURAL_ZONE_PATTERNS = (
    re.compile(r"\b(?:separate|distinct)\s+(?:rooms?|enclosed\s+areas?)\b", re.I),
    re.compile(r"\b(?:interior\s+)?(?:wall|partition)s?\b", re.I),
    re.compile(r"\b(?:divide|split)\s+(?:the\s+)?room\b", re.I),
)

_FURNITURE_ZONE_WORDING = re.compile(
    r"\b(?:physically|architecturally)\s+(?:separate|define)|"
    r"\b(?:spatially\s+)?distinct\s+(?:living|dining|sleeping|working|storage|"
    r"seating|media|functional)\s+zones?\b|"
    r"\b(?:living|dining|sleeping|working|storage|seating|media|functional)\s+"
    r"zones?\s+(?:are|should\s+be|remain)\s+(?:spatially\s+)?distinct\b|"
    r"\b(?:define|create)\s+(?:the\s+)?(?:living|dining|sleeping|working|"
    r"storage|seating|media|functional)\s+zones?\b",
    re.I,
)


def _text_mentions_lighting(text: str) -> bool:
    """Return whether a brief statement introduces a lighting fixture."""
    terms = set(re.findall(r"[a-z0-9]+", str(text or "").lower()))
    return bool(terms & _LIGHTING_TERMS)


def _is_structural_only_ceiling_task(original_task: str) -> bool:
    """Identify prompts that name structural spans but no ceiling lighting.

    A beam, joist, or rafter is architectural decoration rather than a request
    to complete the room's lighting design.  This narrow prompt-grounded case
    prevents the planner from turning an optional fixture into a missing
    ceiling-stage requirement, while prompts that explicitly request lighting
    retain the normal same-stage supporting-object policy.
    """
    terms = set(re.findall(r"[a-z0-9]+", str(original_task or "").lower()))
    return bool(terms & _STRUCTURAL_CEILING_TERMS) and not bool(terms & _LIGHTING_TERMS)


def _task_requests_structural_zoning(original_task: str) -> bool:
    """Return whether the user explicitly asks for a structural room division."""
    normalized = " ".join(str(original_task or "").split())
    return any(pattern.search(normalized) for pattern in _STRUCTURAL_ZONE_PATTERNS)


def _reconcile_floor_plan_zone_guidance(
    brief: StageBrief,
    *,
    original_task: str,
    functional_zones: list[str],
) -> StageBrief:
    """Keep later furniture zones from becoming imaginary floor-plan work.

    Functional zones such as living, dining, or work areas normally describe the
    later furniture arrangement.  A floor-plan agent can make them feasible by
    providing adequate room proportions, circulation, and usable wall segments,
    but it cannot represent them as labelled architectural state.  Requiring a
    partition where the user did not request one makes the critic repeatedly
    mutate doors and windows without improving the downstream scene.
    """
    if (
        brief.stage != "floor_plan"
        or not functional_zones
        or _task_requests_structural_zoning(original_task)
    ):
        return brief

    capacity_guidance = (
        "Treat the named functional zones as later furniture zones: provide room "
        "capacity, unobstructed circulation, and usable wall segments for them, "
        "but do not add rooms, partitions, or architectural markers unless the "
        "immutable user task explicitly requests structural separation."
    )
    capacity_check = (
        "Evaluate functional zones through room capacity, circulation, and usable "
        "opening-free wall length; do not score absent furniture-zone markers or "
        "partitions as a floor-plan failure."
    )

    def reconcile_values(values: list[str], replacement: str) -> list[str]:
        reconciled = [
            replacement if _FURNITURE_ZONE_WORDING.search(value) else value
            for value in values
        ]
        return list(dict.fromkeys(reconciled))

    constraints = reconcile_values(
        list(brief.constraints_for_designer), capacity_guidance
    )
    checks = reconcile_values(list(brief.checks_for_critic), capacity_check)
    if capacity_guidance not in constraints:
        constraints.append(capacity_guidance)
    if capacity_check not in checks:
        checks.append(capacity_check)
    return brief.model_copy(
        update={
            "constraints_for_designer": constraints[:6],
            "checks_for_critic": checks,
        }
    )


def _text_closes_stage_inventory(text: str) -> bool:
    """Return whether planner text makes a stage's required inventory exclusive."""
    normalized = " ".join(str(text or "").split())
    return any(pattern.search(normalized) for pattern in _INVENTORY_CLOSING_PATTERNS)


def _task_explicitly_closes_inventory(
    original_task: str,
    required_objects: list[str],
) -> bool:
    """Recognize user-authored requests for an intentionally sparse inventory."""
    normalized = " ".join(str(original_task or "").split())
    if any(pattern.search(normalized) for pattern in _EXPLICITLY_CLOSED_TASK_PATTERNS):
        return True

    required_terms = {
        " ".join(re.sub(r"[^a-z0-9]+", " ", value.lower()).split())
        for value in required_objects
        if str(value).strip()
    }
    for pattern in _ONLY_INVENTORY_CLAUSE_PATTERNS:
        match = pattern.search(normalized)
        if not match:
            continue
        inventory = " ".join(
            re.sub(r"[^a-z0-9]+", " ", match.group("inventory").lower()).split()
        )
        if any(
            re.search(rf"\b{re.escape(term)}\b", inventory) for term in required_terms
        ):
            return True
    return False


def _reconcile_stage_brief(
    brief: StageBrief,
    *,
    original_task: str,
    room_type: str,
    required_objects: list[str],
) -> StageBrief:
    """Remove planner-invented exclusivity that can deadlock designer and critic."""
    if _task_explicitly_closes_inventory(original_task, required_objects):
        return brief

    structural_only_ceiling = (
        brief.stage == "ceiling_mounted"
        and _is_structural_only_ceiling_task(original_task)
    )

    fields = (
        "constraints_for_designer",
        "checks_for_critic",
        "failure_patterns_to_avoid",
    )
    updates: dict[str, object] = {}
    removed = 0
    for field in fields:
        values = list(getattr(brief, field))
        kept = [
            value
            for value in values
            if not _text_closes_stage_inventory(value)
            and not (structural_only_ceiling and _text_mentions_lighting(value))
        ]
        removed += len(values) - len(kept)
        if len(kept) != len(values):
            updates[field] = kept

    if structural_only_ceiling and _text_mentions_lighting(brief.stage_objective):
        updates["stage_objective"] = (
            "Install the prompt-requested structural ceiling spans while "
            "maintaining clear vertical clearance"
        )
        removed += 1
    elif _text_closes_stage_inventory(brief.stage_objective):
        updates["stage_objective"] = (
            f"Complete the {brief.stage} stage for a {room_type}"
        )
        removed += 1

    if not removed:
        return brief

    constraints = list(
        updates.get("constraints_for_designer", brief.constraints_for_designer)
    )
    constraints = constraints[:5]
    if structural_only_ceiling:
        constraints.append(
            "The prompt requests structural ceiling spans only; do not add or "
            "treat lighting fixtures as required unless the user explicitly asks."
        )
        checks = list(updates.get("checks_for_critic", brief.checks_for_critic))
        checks.append(
            "Evaluate the requested structural spans without treating absent "
            "lighting as a ceiling-stage failure."
        )
        updates["checks_for_critic"] = checks
    else:
        constraints.append(
            "Treat required objects as a minimum; additional objects owned by this "
            "stage are allowed when they improve room function or completeness."
        )
    updates["constraints_for_designer"] = constraints
    console_logger.warning(
        "GlobalPlanner removed %d invented inventory-closing instruction(s) from "
        "the %s brief",
        removed,
        brief.stage,
    )
    return brief.model_copy(update=updates)


def _add_floor_plan_reservation_guidance(
    brief: StageBrief,
    context: HarnessContext,
) -> StageBrief:
    """Make later explicit wall anchors actionable before openings are created."""
    if context.stage != "floor_plan" or context.relation_context is None:
        return brief
    reservations = context.relation_context.floor_plan_reservations
    manifest = context.relation_context.floor_plan_manifest
    if not reservations and not (manifest and manifest.enabled):
        return brief

    anchors: list[str] = []
    for reservation in reservations:
        subject = reservation.get("subjects") or {}
        target = reservation.get("targets") or {}
        category = str(subject.get("category") or "object").replace("_", " ")
        role = str(target.get("role") or "").strip().lower()
        relation = str(reservation.get("relation") or "")
        reservation_kind = str(reservation.get("reservation_kind") or "")
        if reservation_kind == "opening_adjacency":
            anchors.append(f"{category} next to a window")
        elif relation == "centered_on_wall" and role:
            anchors.append(f"{category} centered on the {role} wall")
        elif role:
            anchors.append(f"{category} on a {role} wall")
        else:
            anchors.append(f"{category} on a wall")

    details: list[str] = []
    if anchors:
        details.append("later hard anchors: " + "; ".join(anchors))
    if manifest and manifest.enabled:
        pair_count = sum(
            item.kind == "opposed_anchor_pair" for item in manifest.reservations
        )
        zone_area = sum(
            item.min_zone_area_m2
            for item in manifest.reservations
            if item.kind == "functional_zone"
        )
        if pair_count:
            details.append(
                "aligned opening-free spans on opposed walls for each media-viewing pair"
            )
        opening_adjacency = [
            item for item in manifest.reservations if item.kind == "opening_adjacency"
        ]
        if opening_adjacency:
            details.append(
                "a separate continuous opening-free wall segment on at least one "
                "side of a matching window for every strict next-to reservation; "
                "doors and other openings cannot share that capacity"
            )
        if zone_area and (
            not manifest.explicit_geometry.detected
            or manifest.explicit_geometry.functional_zone_area_policy != "advisory"
        ):
            details.append(
                f"at least {zone_area:g} m2 total usable area for later functional zones"
            )
        elif zone_area:
            details.append(
                "the exact user-specified footprint is immutable; use functional "
                "zones as a downstream layout advisory rather than enlarging it"
            )
        details.append(
            "an adaptive implicit-window budget of 1 for rooms up to 25 m2, "
            "2 for rooms up to 50 m2, and 3 above 50 m2, normally no more "
            "than one implicit window per wall"
        )
        if manifest.explicit_window_required:
            details.append(
                f"all {manifest.explicit_window_count} explicitly required window(s)"
            )
    guidance = (
        "Before adding doors or windows, reserve continuous opening-free usable "
        "wall segments and capacity for "
        + "; ".join(details)
        + ". Size each segment for the named object and its required alignment; "
        "never place an implicit opening through a reserved segment. Keep an "
        "exterior entrance and its route available. A reservation is satisfied "
        "by usable capacity; do not create a partition solely to represent a "
        "later furniture zone."
    )
    check = (
        "Verify every reserved wall-anchor segment remains large enough and "
        "unobstructed by doors or windows, and every reserved window-side segment "
        "remains large enough and unobstructed by doors or other openings before "
        "handing the room to furniture."
    )
    constraints = list(brief.constraints_for_designer)
    if guidance not in constraints:
        constraints = [*constraints[:5], guidance]
    checks = list(brief.checks_for_critic)
    if check not in checks:
        checks.append(check)
    return brief.model_copy(
        update={
            "constraints_for_designer": constraints,
            "checks_for_critic": checks,
        }
    )


def _format_memory_for_prompt(memory_pack: MemoryPack) -> str:
    """Format memory pack into a compact text block."""
    parts: list[str] = []
    if memory_pack.success_hints:
        parts.append("Success patterns from similar scenes:")
        parts.extend(f"  {i+1}. {h}" for i, h in enumerate(memory_pack.success_hints))
    if memory_pack.failure_hints:
        parts.append("Known failure patterns to avoid:")
        parts.extend(f"  {i+1}. {h}" for i, h in enumerate(memory_pack.failure_hints))
    if memory_pack.skill_texts:
        parts.append("Applicable skills:")
        for skill_text in memory_pack.skill_texts:
            parts.append(skill_text)
    return "\n".join(parts) if parts else "No relevant memory retrieved for this stage."


def _stage_required_objects(task_spec: SceneTaskSpec, stage: str) -> list[str]:
    """Return the TaskCompiler-owned inventory for one pipeline stage."""
    return list(
        {
            "floor_plan": task_spec.required_large_objects,
            "furniture": task_spec.required_large_objects,
            "wall_mounted": task_spec.required_wall_objects,
            "ceiling_mounted": task_spec.required_ceiling_objects,
            "manipuland": task_spec.required_small_objects,
        }.get(stage, [])
    )


def _brief_required_objects(task_spec: SceneTaskSpec, stage: str) -> list[str]:
    """Return assets actually created by ``stage``, not downstream capacity."""
    if stage == "floor_plan":
        return list(task_spec.required_architectural_features)
    return _stage_required_objects(task_spec, stage)


def _apply_stage_policy(
    brief: StageBrief,
    context: HarnessContext,
    *,
    original_task: str = "",
) -> StageBrief:
    """Make policy semantics deterministic after the model-generated proposal."""
    policy = context.stage_policy
    required = _brief_required_objects(context.task_spec, context.stage)
    inventory_closed = _task_explicitly_closes_inventory(original_task, required)
    optional = (
        list(brief.optional_asset_recommendations)
        if policy == "auto" and not inventory_closed
        else []
    )
    constraints = list(brief.constraints_for_designer)
    if inventory_closed:
        policy_rule = (
            "The immutable user task explicitly closes this stage inventory. "
            "Complete its required assets without optional additions, but still "
            "execute and verify the native stage."
        )
    elif policy == "auto":
        policy_rule = (
            "Treat prompt-required assets as the non-negotiable minimum, then use "
            "native designer judgment for useful same-stage optional assets that fit "
            "the room type, style, usable capacity, circulation, and existing scene."
        )
    else:
        policy_rule = (
            "This required_only ablation disables optional asset proposals, but the "
            "native stage must still execute and verify every prompt-required asset."
        )
    if policy_rule not in constraints:
        constraints = [*constraints[:5], policy_rule]
    return brief.model_copy(
        update={
            "stage_policy": policy,
            "optional_assets_allowed": bool(policy == "auto" and not inventory_closed),
            "required_objects": required,
            "optional_asset_recommendations": optional,
            "constraints_for_designer": constraints,
        }
    )


def _format_task_spec(task_spec: SceneTaskSpec, stage: str) -> str:
    """Format task spec focusing on stage-relevant requirements."""
    lines = [
        f"Room type: {task_spec.room_type}",
        f"Style: {task_spec.style}",
    ]

    required = _stage_required_objects(task_spec, stage)
    if stage == "floor_plan" and task_spec.required_large_objects:
        lines.append(
            "Downstream furniture capacity requirements (plan space only; do not "
            "place these objects in floor_plan): "
            + ", ".join(task_spec.required_large_objects)
        )
    elif required:
        lines.append(
            "Required objects for this stage (minimum, not exhaustive): "
            f"{', '.join(required)}"
        )
    else:
        lines.extend(
            (
                "Required objects for this stage: none.",
                "Stage ownership: do not create, move, or reinterpret objects "
                "assigned to another stage.",
            )
        )

    if task_spec.functional_zones:
        lines.append(f"Functional zones: {', '.join(task_spec.functional_zones)}")

    if task_spec.interaction_constraints:
        lines.append("Interaction constraints:")
        lines.extend(f"  - {c}" for c in task_spec.interaction_constraints)

    if task_spec.aesthetic_constraints:
        lines.append("Aesthetic constraints:")
        lines.extend(f"  - {c}" for c in task_spec.aesthetic_constraints)

    return "\n".join(lines)


def _extract_json_from_text(text: str) -> dict:
    """Extract JSON from model output, handling markdown code fences."""
    if not text:
        raise ValueError("Empty response text")
    text = text.strip()
    fence_match = re.search(r"```(?:json)?\s*([\s\S]+?)\s*```", text)
    if fence_match:
        text = fence_match.group(1)
    brace_match = re.search(r"\{[\s\S]+\}", text)
    if brace_match:
        text = brace_match.group(0)
    return json.loads(text)


class GlobalPlanner:
    """Generates a StageBrief for each stage using Qwen3."""

    def __init__(
        self,
        model: str,
        api_base_url: str | None = None,
        api_key: str | None = None,
        max_tokens: int = 2048,
        temperature: float = 0.2,
        llm_client: Any | None = None,
    ) -> None:
        self._model = model
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._structured_llm = llm_client
        self._client = None
        if self._structured_llm is None:
            from openai import OpenAI

            self._client = OpenAI(
                base_url=api_base_url
                or os.environ.get("OPENAI_BASE_URL", "http://localhost:8000/v1"),
                api_key=api_key or os.environ.get("OPENAI_API_KEY", "dummy"),
            )
        self.last_trace: dict = {}

    def generate_stage_brief(
        self,
        context: HarnessContext,
        scene_state_summary: str = "",
        original_task: str = "",
    ) -> StageBrief:
        """Generate expert planning hints for a single stage.

        Args:
            context: Harness context with task spec, memory pack, and budget.
            scene_state_summary: Text summary of current SceneSmith scene state
                (present objects, their categories, support surfaces).

        Returns:
            StageBrief to inject into the SceneSmith stage prompt.
        """
        stage = context.stage
        console_logger.info(f"GlobalPlanner: generating StageBrief for stage '{stage}'")

        user_message = self._build_user_message(
            context,
            scene_state_summary,
            original_task=original_task,
        )

        structured_llm = getattr(self, "_structured_llm", None)
        if structured_llm is not None:
            result = structured_llm.complete(
                role="global_planner",
                stage=stage,
                event="generate_stage_brief",
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
                response_model=StageBrief,
            )
            attempts = [
                attempt.model_dump() | {"attempt": max(0, int(attempt.attempt) - 1)}
                for attempt in result.attempts
            ]
            failure_reason = f"{result.final_error_kind}: {result.final_error}".strip(
                ": "
            )
            if result.value is not None and result.value.stage == stage:
                brief = _reconcile_stage_brief(
                    result.value,
                    original_task=original_task,
                    room_type=context.task_spec.room_type,
                    required_objects=_stage_required_objects(context.task_spec, stage),
                )
                brief = _reconcile_floor_plan_zone_guidance(
                    brief,
                    original_task=original_task,
                    functional_zones=context.task_spec.functional_zones,
                )
                brief = enforce_stage_brief_scope(brief, context.task_spec)
                brief = _apply_stage_policy(
                    brief,
                    context,
                    original_task=original_task,
                )
                brief = _add_floor_plan_reservation_guidance(brief, context)
                self.last_trace = {
                    "status": "ok",
                    "stage": stage,
                    "stage_policy": context.stage_policy,
                    "required_objects": brief.required_objects,
                    "optional_asset_recommendations": [
                        item.model_dump(mode="json")
                        for item in brief.optional_asset_recommendations
                    ],
                    "attempts": attempts,
                    "hard_constraint_ids": (
                        context.relation_context.hard_constraint_ids
                        if context.relation_context is not None
                        else []
                    ),
                    "hard_constraint_coverage": 1.0,
                }
                return brief

            if result.value is not None:
                failure_reason = (
                    f"StageBrief.stage must be {stage!r}, "
                    f"got {result.value.stage!r}"
                )
            self.last_trace = {
                "status": "fallback",
                "stage": stage,
                "stage_policy": context.stage_policy,
                "attempts": attempts,
                "failure_reason": failure_reason,
                "hard_constraint_ids": (
                    context.relation_context.hard_constraint_ids
                    if context.relation_context is not None
                    else []
                ),
                "hard_constraint_coverage": 1.0,
            }
            console_logger.warning(
                "Structured GlobalPlanner failed for %s; using main-compatible "
                "fallback brief: %s",
                stage,
                failure_reason,
            )
            return _add_floor_plan_reservation_guidance(
                self._fallback_brief(context, original_task=original_task), context
            )

        attempts: list[dict] = []
        previous_output = ""
        validation_error = ""
        for attempt in range(2):
            messages = [
                {
                    "role": "system",
                    "content": prepend_text_thinking_directive(
                        _SYSTEM_PROMPT,
                        thinking_directive_from_effort("none", model=self._model),
                    ),
                },
                {"role": "user", "content": user_message},
            ]
            if attempt:
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "The previous candidate failed validation. Return a corrected "
                            "JSON object only.\nValidation error: "
                            f"{validation_error}\nPrevious candidate:\n{previous_output}"
                        ),
                    }
                )
            started_at = time.perf_counter()
            raw = ""
            response = None
            response_elapsed_sec: float | None = None
            try:
                response = self._client.chat.completions.create(
                    model=self._model,
                    messages=messages,
                    temperature=self._temperature,
                    max_tokens=self._max_tokens,
                    response_format={
                        "type": "json_schema",
                        "json_schema": {
                            "name": "stage_brief",
                            "strict": True,
                            "schema": make_openai_strict_json_schema(
                                StageBrief.model_json_schema()
                            ),
                        },
                    },
                    extra_body=openrouter_extra_body(
                        chat_template_kwargs_from_effort("none", model=self._model)
                    ),
                )
                response_elapsed_sec = round(time.perf_counter() - started_at, 6)
                message = response.choices[0].message
                raw = message.content
                if not raw:
                    raw = getattr(message, "reasoning_content", None)
                if not raw:
                    extra = getattr(message, "model_extra", None)
                    if isinstance(extra, dict):
                        raw = extra.get("reasoning_content")
                data = _extract_json_from_text(raw)
                brief = StageBrief.model_validate(data)
                if brief.stage != stage:
                    raise ValueError(
                        f"StageBrief.stage must be {stage!r}, got {brief.stage!r}"
                    )
                brief = _reconcile_stage_brief(
                    brief,
                    original_task=original_task,
                    room_type=context.task_spec.room_type,
                    required_objects=_stage_required_objects(context.task_spec, stage),
                )
                brief = _reconcile_floor_plan_zone_guidance(
                    brief,
                    original_task=original_task,
                    functional_zones=context.task_spec.functional_zones,
                )
                brief = enforce_stage_brief_scope(brief, context.task_spec)
                brief = _apply_stage_policy(
                    brief,
                    context,
                    original_task=original_task,
                )
                brief = _add_floor_plan_reservation_guidance(brief, context)
                attempts.append(
                    {
                        "attempt": attempt,
                        "status": "ok",
                        "elapsed_sec": (
                            response_elapsed_sec
                            if response_elapsed_sec is not None
                            else round(time.perf_counter() - started_at, 6)
                        ),
                    }
                )
                self.last_trace = {
                    "status": "ok",
                    "stage": stage,
                    "stage_policy": context.stage_policy,
                    "required_objects": brief.required_objects,
                    "optional_asset_recommendations": [
                        item.model_dump(mode="json")
                        for item in brief.optional_asset_recommendations
                    ],
                    "attempts": attempts,
                    "hard_constraint_ids": (
                        context.relation_context.hard_constraint_ids
                        if context.relation_context is not None
                        else []
                    ),
                    "hard_constraint_coverage": 1.0,
                }
                _append_llm_debug(
                    build_llm_call_debug_record(
                        stage=stage,
                        agent_role="global_planner",
                        event="generate_stage_brief",
                        prompt=messages,
                        output=raw or "",
                        raw_response=response,
                    ).model_dump()
                    | {
                        "attempt": attempt,
                        "status": "ok",
                        "elapsed_sec": response_elapsed_sec,
                    }
                )
                return brief
            except Exception as exc:
                validation_error = f"{type(exc).__name__}: {exc}"
                previous_output = raw
                attempts.append(
                    {
                        "attempt": attempt,
                        "status": "error",
                        "error": validation_error,
                        "elapsed_sec": round(time.perf_counter() - started_at, 6),
                    }
                )
                _append_llm_debug(
                    build_llm_call_debug_record(
                        stage=stage,
                        agent_role="global_planner",
                        event="generate_stage_brief",
                        prompt=messages,
                        output=raw,
                        raw_response=response,
                        error=validation_error,
                    ).model_dump()
                    | {
                        "attempt": attempt,
                        "status": "error",
                        "elapsed_sec": round(time.perf_counter() - started_at, 6),
                    }
                )

        attempts.append(
            {"attempt": 2, "status": "minimal_fallback", "elapsed_sec": 0.0}
        )
        self.last_trace = {
            "status": "fallback",
            "stage": stage,
            "stage_policy": context.stage_policy,
            "attempts": attempts,
            "failure_reason": validation_error,
            "hard_constraint_ids": (
                context.relation_context.hard_constraint_ids
                if context.relation_context is not None
                else []
            ),
            "hard_constraint_coverage": 1.0,
        }
        console_logger.warning(
            "GlobalPlanner failed twice for stage %s, using minimal fallback brief: %s",
            stage,
            validation_error,
        )
        return _add_floor_plan_reservation_guidance(
            self._fallback_brief(context, original_task=original_task), context
        )

    def _build_user_message(
        self,
        context: HarnessContext,
        scene_state_summary: str,
        *,
        original_task: str = "",
    ) -> str:
        stage_desc = _STAGE_DESCRIPTIONS.get(context.stage, "")
        task_spec_text = _format_task_spec(context.task_spec, context.stage)
        memory_text = _format_memory_for_prompt(context.memory_pack)

        parts = [
            f"## Current Stage: {context.stage}",
            f"Stage description: {stage_desc}",
            f"Stage execution policy: {context.stage_policy}",
            "",
            "## Task Specification",
            task_spec_text,
        ]

        if original_task.strip():
            parts += [
                "",
                "## Immutable User Task",
                original_task.strip(),
                (
                    "Every explicit spatial relation in this task takes priority over "
                    "memory, current-scene conventions, and new planner suggestions."
                ),
            ]

        if scene_state_summary:
            parts += [
                "",
                "## Current Scene State (already placed objects)",
                scene_state_summary,
            ]

        if context.relation_context is not None:
            parts += [
                "",
                "## Authoritative Stage Intent (exact JSON; hard)",
                json.dumps(
                    context.relation_context.hard_constraints,
                    ensure_ascii=False,
                    sort_keys=True,
                ),
            ]
            if (
                context.stage == "floor_plan"
                and context.relation_context.floor_plan_reservations
            ):
                parts += [
                    "",
                    "## Architectural Surface Reservations (hard)",
                    json.dumps(
                        context.relation_context.floor_plan_reservations,
                        ensure_ascii=False,
                        sort_keys=True,
                    ),
                ]
            if context.relation_context.resolved_opening_reservations:
                parts += [
                    "",
                    "## Resolved Opening Reservations (authoritative geometry)",
                    json.dumps(
                        context.relation_context.resolved_opening_reservations,
                        ensure_ascii=False,
                        sort_keys=True,
                    ),
                ]

        parts += ["", "## Retrieved Memory", memory_text]
        if (
            context.stage_budget.max_designer_iterations > 0
            or context.stage_budget.max_repair_steps > 0
        ):
            parts += [
                "",
                f"## Budget: max_designer_iterations={context.stage_budget.max_designer_iterations}, "
                f"max_repair_steps={context.stage_budget.max_repair_steps}",
            ]
        parts += ["", "Generate the StageBrief JSON for the designer agent."]

        return "\n".join(parts)

    def _fallback_brief(
        self,
        context: HarnessContext,
        *,
        original_task: str = "",
    ) -> StageBrief:
        """Minimal safe StageBrief used when the model call fails."""
        stage = context.stage
        required = _stage_required_objects(context.task_spec, stage)

        constraints = []
        if stage == "floor_plan" and context.task_spec.required_large_objects:
            constraints.append(
                "Reserve adequate floor area and circulation for downstream "
                "furniture: "
                + ", ".join(context.task_spec.required_large_objects)
                + ". Do not place furniture during floor_plan."
            )
        elif required:
            constraints.append(
                f"Ensure these objects are present: {', '.join(required)}"
            )
            constraints.append(f"Follow {context.task_spec.style} aesthetic style")
            constraints.append("Maintain clear walking paths and avoid overcrowding")
        elif context.stage_policy == "auto":
            constraints.append(
                "No asset is explicitly required for this stage. Still execute the "
                "native stage and autonomously add useful same-stage assets when they "
                "fit the room type, style, usable capacity, circulation, support "
                "surfaces, and existing scene."
            )
        else:
            constraints.append(
                "No asset is explicitly required and optional planning is disabled by "
                "required_only. Still execute the native stage and preserve stage "
                "ownership."
            )

        brief = enforce_stage_brief_scope(
            StageBrief(
                stage=stage,
                stage_policy=context.stage_policy,
                required_objects=_brief_required_objects(context.task_spec, stage),
                stage_objective=f"Complete the {stage} stage for a {context.task_spec.room_type}",
                recommended_skills=[],
                constraints_for_designer=constraints,
                checks_for_critic=[
                    (
                        "Verify all required objects are present"
                        if required
                        else "Verify stage-appropriate completeness and ownership"
                    ),
                    (
                        "Check for collisions"
                        if required
                        else "Check optional additions for capacity and collisions"
                    ),
                ],
                failure_patterns_to_avoid=[],
            ),
            context.task_spec,
        )
        return _apply_stage_policy(
            brief,
            context,
            original_task=original_task,
        )
