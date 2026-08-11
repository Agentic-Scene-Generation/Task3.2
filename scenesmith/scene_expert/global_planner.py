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

from openai import OpenAI

from scenesmith.scene_expert.context_bundle import build_llm_call_debug_record
from scenesmith.agent_utils.thinking import chat_template_kwargs_from_effort
from scenesmith.scene_expert.schemas import (
    HarnessContext,
    MemoryPack,
    SceneTaskSpec,
    StageBrief,
)

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
  replace one with a convention. HSSD priors are advisory only.
- At floor_plan, Architectural Surface Reservations are hard prerequisites for
  later-stage wall anchors. Reserve a continuous wall segment for every listed
  object; doors and windows must not intersect a reserved segment. Do not move
  a later object away from its explicit required wall merely to keep an opening.
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
  unless the Immutable User Task explicitly forbids them. When that inventory is
  empty, emit an observation-only brief with no object-placement instruction.
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
    if not reservations:
        return brief

    anchors: list[str] = []
    for reservation in reservations:
        subject = reservation.get("subjects") or {}
        target = reservation.get("targets") or {}
        category = str(subject.get("category") or "object").replace("_", " ")
        role = str(target.get("role") or "").strip().lower()
        relation = str(reservation.get("relation") or "")
        if relation == "centered_on_wall" and role:
            anchors.append(f"{category} centered on the {role} wall")
        elif role:
            anchors.append(f"{category} on a {role} wall")
        else:
            anchors.append(f"{category} on a wall")

    if not anchors:
        return brief
    guidance = (
        "Before adding doors or windows, reserve continuous opening-free usable "
        "wall segments for these later hard anchors: "
        + "; ".join(anchors)
        + ". Size each segment for the named object and its required alignment; "
        "never place an opening through a reserved segment. A reservation is "
        "satisfied by usable opening-free wall length; do not create a partition "
        "or other architectural marker solely to represent a later furniture zone."
    )
    check = (
        "Verify every reserved wall-anchor segment remains large enough and "
        "unobstructed by doors or windows before handing the room to furniture."
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


def _format_task_spec(task_spec: SceneTaskSpec, stage: str) -> str:
    """Format task spec focusing on stage-relevant requirements."""
    lines = [
        f"Room type: {task_spec.room_type}",
        f"Style: {task_spec.style}",
    ]

    required = _stage_required_objects(task_spec, stage)
    if required:
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
    ) -> None:
        self._model = model
        self._max_tokens = max_tokens
        self._temperature = temperature
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

        # Empty inventories must be a true no-op.  Letting an LLM read the
        # original prompt here can otherwise recreate a manipuland or furniture
        # object in the wall stage, contradicting the compiled contract.
        hard_constraints = (
            context.relation_context.hard_constraints
            if context.relation_context is not None
            else []
        )
        if (
            not _stage_required_objects(context.task_spec, stage)
            and not hard_constraints
        ):
            console_logger.info(
                "GlobalPlanner: %s has no TaskCompiler-owned objects; using no-op brief",
                stage,
            )
            self.last_trace = {
                "status": "no_op",
                "stage": stage,
                "attempts": [],
                "hard_constraint_ids": [],
                "hard_constraint_coverage": 1.0,
            }
            return _add_floor_plan_reservation_guidance(
                self._fallback_brief(context), context
            )

        user_message = self._build_user_message(
            context,
            scene_state_summary,
            original_task=original_task,
        )

        attempts: list[dict] = []
        previous_output = ""
        validation_error = ""
        for attempt in range(2):
            messages = [
                {"role": "system", "content": _SYSTEM_PROMPT},
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
                            "schema": StageBrief.model_json_schema(),
                        },
                    },
                    extra_body=chat_template_kwargs_from_effort("none"),
                )
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
                brief = _add_floor_plan_reservation_guidance(brief, context)
                attempts.append(
                    {
                        "attempt": attempt,
                        "status": "ok",
                        "elapsed_sec": round(time.perf_counter() - started_at, 6),
                    }
                )
                self.last_trace = {
                    "status": "ok",
                    "stage": stage,
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
                    | {"attempt": attempt, "status": "ok"}
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
                        error=validation_error,
                    ).model_dump()
                    | {"attempt": attempt, "status": "error"}
                )

        attempts.append(
            {"attempt": 2, "status": "minimal_fallback", "elapsed_sec": 0.0}
        )
        self.last_trace = {
            "status": "fallback",
            "stage": stage,
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
            self._fallback_brief(context), context
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
                "",
                "## Advisory HSSD Relation Priors (soft; never override hard intent)",
                json.dumps(
                    [
                        item.model_dump(mode="json")
                        for item in context.relation_context.advisory_hssd_priors
                    ],
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

        parts += [
            "",
            "## Retrieved Memory",
            memory_text,
            "",
            f"## Budget: max_designer_iterations={context.stage_budget.max_designer_iterations}, "
            f"max_repair_steps={context.stage_budget.max_repair_steps}",
            "",
            "Generate the StageBrief JSON for the designer agent.",
        ]

        return "\n".join(parts)

    def _fallback_brief(self, context: HarnessContext) -> StageBrief:
        """Minimal safe StageBrief used when the model call fails."""
        stage = context.stage
        required = _stage_required_objects(context.task_spec, stage)

        constraints = []
        if required:
            constraints.append(
                f"Ensure these objects are present: {', '.join(required)}"
            )
            constraints.append(f"Follow {context.task_spec.style} aesthetic style")
            constraints.append("Maintain clear walking paths and avoid overcrowding")
        else:
            constraints.append(
                "No objects are allocated to this stage; do not create, move, or "
                "reinterpret objects owned by another stage."
            )

        return StageBrief(
            stage=stage,
            stage_objective=(
                f"Complete the {stage} stage for a {context.task_spec.room_type}"
                if required
                else f"Preserve the existing scene during the empty {stage} stage"
            ),
            recommended_skills=[],
            constraints_for_designer=constraints,
            checks_for_critic=[
                (
                    "Verify all required objects are present"
                    if required
                    else "Verify no cross-stage object was created or moved"
                ),
                "Check for collisions" if required else "Preserve existing geometry",
            ],
            failure_patterns_to_avoid=[],
        )
