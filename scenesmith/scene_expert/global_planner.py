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

from openai import OpenAI

from scenesmith.scene_expert.schemas import (
    HarnessContext,
    MemoryPack,
    SceneTaskSpec,
    StageBrief,
)

console_logger = logging.getLogger(__name__)

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
- Prioritize failure patterns from memory — they encode hard-won lessons.
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


def _format_task_spec(task_spec: SceneTaskSpec, stage: str) -> str:
    """Format task spec focusing on stage-relevant requirements."""
    lines = [
        f"Room type: {task_spec.room_type}",
        f"Style: {task_spec.style}",
    ]

    stage_objects = {
        "floor_plan": task_spec.required_large_objects,
        "furniture": task_spec.required_large_objects,
        "wall_mounted": task_spec.required_wall_objects,
        "ceiling_mounted": task_spec.required_ceiling_objects,
        "manipuland": task_spec.required_small_objects,
    }
    required = stage_objects.get(stage, [])
    if required:
        lines.append(f"Required objects for this stage: {', '.join(required)}")

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
            base_url=api_base_url or os.environ.get("OPENAI_BASE_URL", "http://localhost:8000/v1"),
            api_key=api_key or os.environ.get("OPENAI_API_KEY", "dummy"),
        )

    def generate_stage_brief(
        self,
        context: HarnessContext,
        scene_state_summary: str = "",
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

        user_message = self._build_user_message(context, scene_state_summary)

        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
                temperature=self._temperature,
                max_tokens=self._max_tokens,
            )
            raw = response.choices[0].message.content
            # Qwen3 with --reasoning-parser may put output in reasoning_content
            if raw is None:
                raw = getattr(response.choices[0].message, "reasoning_content", None)
            console_logger.debug(f"GlobalPlanner raw response: {raw}")
            data = _extract_json_from_text(raw)
            # Ensure stage field is set correctly
            data["stage"] = stage
            brief = StageBrief.model_validate(data)
            console_logger.info(
                f"GlobalPlanner: brief for {stage}: {len(brief.constraints_for_designer)} constraints, "
                f"{len(brief.failure_patterns_to_avoid)} failure patterns"
            )
            return brief
        except Exception as e:
            console_logger.warning(
                f"GlobalPlanner failed for stage {stage}, using minimal fallback brief: {e}"
            )
            return self._fallback_brief(context)

    def _build_user_message(self, context: HarnessContext, scene_state_summary: str) -> str:
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

        if scene_state_summary:
            parts += ["", "## Current Scene State (already placed objects)", scene_state_summary]

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
        required = {
            "floor_plan": context.task_spec.required_large_objects,
            "furniture": context.task_spec.required_large_objects,
            "wall_mounted": context.task_spec.required_wall_objects,
            "ceiling_mounted": context.task_spec.required_ceiling_objects,
            "manipuland": context.task_spec.required_small_objects,
        }.get(stage, [])

        constraints = []
        if required:
            constraints.append(f"Ensure these objects are present: {', '.join(required)}")
        constraints.append(f"Follow {context.task_spec.style} aesthetic style")
        constraints.append("Maintain clear walking paths and avoid overcrowding")

        return StageBrief(
            stage=stage,
            stage_objective=f"Complete the {stage} stage for a {context.task_spec.room_type}",
            recommended_skills=[],
            constraints_for_designer=constraints,
            checks_for_critic=["Verify all required objects are present", "Check for collisions"],
            failure_patterns_to_avoid=[],
        )
