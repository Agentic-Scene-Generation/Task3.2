"""MemoryWriter: Qwen3 memory_writer role that updates fast memory after each run.

Takes a trace summary + final verifier report and produces structured memory
update operations (ADD/UPDATE/NOOP) for the three memory banks.

MVP only uses ADD, UPDATE, NOOP — DELETE is intentionally not implemented
to avoid accidentally removing useful experience.
"""

from __future__ import annotations

import json
import logging
import os

from openai import OpenAI

from scenesmith.scene_expert.memory.schemas import MemoryUpdateOp
from scenesmith.scene_expert.memory.store import FastMemoryStore
from scenesmith.scene_expert.schemas import FullVerifyReport

console_logger = logging.getLogger(__name__)
_SYSTEM_PROMPT = """\
/no_think
You are the memory_writer for SceneExpert, a 3D scene generation system.
Your job is to analyze a completed scene generation trace and extract reusable knowledge
to update the long-term memory system.

You MUST output valid JSON in this exact format:
{
  "updates": [
    {
      "op": "ADD" | "UPDATE" | "NOOP",
      "memory_type": "success_case" | "failure_case" | "skill",
      "target_id": "<case_id or skill_name — only required for UPDATE>",
      "content": { ... }
    }
  ]
}

Rules:
- Use "ADD" to add new memory entries.
- Use "UPDATE" to update existing entries (must provide target_id).
- Use "NOOP" if nothing useful to save.
- Do NOT use "DELETE".
- For success_case content, include: case_id, room_type, style, stage, task_signature, successful_pattern, scores.
- For failure_case content, include: failure_id, room_type, stage, object, failure_type, bad_pattern, failure_reason, repair_action, repair_verified.
- For skill content, include: skill_name, stage, room_types, preconditions, procedure, failure_avoidance, postconditions.
- Focus on patterns that generalize to other rooms of the same type, not one-off details.
- Extract one memory entry per distinct lesson learned. Avoid redundancy with existing memory.
"""


class MemoryWriter:
    """Calls Qwen3 to generate memory update operations from a completed trace."""

    def __init__(
        self,
        model: str,
        api_base_url: str | None = None,
        api_key: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.1,
    ) -> None:
        self._model = model
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._client = OpenAI(
            base_url=api_base_url or os.environ.get("OPENAI_BASE_URL", "http://localhost:8000/v1"),
            api_key=api_key or os.environ.get("OPENAI_API_KEY", "dummy"),
        )

    def write(
        self,
        trace_summary: str,
        full_report: FullVerifyReport,
        related_old_memory: str = "",
    ) -> list[MemoryUpdateOp]:
        """Generate memory update operations for a completed scene run.

        Args:
            trace_summary: Human-readable summary of the full trace.
            full_report: Final verifier report.
            related_old_memory: Relevant existing memory entries (for deduplication context).

        Returns:
            List of MemoryUpdateOp to apply to the store.
        """
        user_message = self._build_user_message(trace_summary, full_report, related_old_memory)

        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
                temperature=self._temperature,
                max_tokens=self._max_tokens,
                response_format={"type": "json_object"},
            )
            raw = response.choices[0].message.content
            if raw is None:
                raw = getattr(response.choices[0].message, "reasoning_content", None)
            console_logger.debug(f"MemoryWriter raw response: {raw}")
            data = json.loads(raw)
            ops = [MemoryUpdateOp.model_validate(op) for op in data.get("updates", [])]
            console_logger.info(f"MemoryWriter: {len(ops)} update ops generated")
            return ops
        except Exception as e:
            console_logger.warning(f"MemoryWriter failed (will skip memory update): {e}")
            return []

    def _build_user_message(
        self,
        trace_summary: str,
        full_report: FullVerifyReport,
        related_old_memory: str,
    ) -> str:
        score_str = (
            f"overall={full_report.overall_score:.2f}, "
            f"semantic={full_report.semantic_score:.2f}, "
            f"aesthetic={full_report.aesthetic_score:.2f}, "
            f"reachability={full_report.reachability_score:.2f}, "
            f"physics={full_report.collision_free_rate:.2f}"
        )
        parts = [
            "## Scene Generation Trace Summary",
            trace_summary,
            "",
            f"## Final Verifier Scores\n{score_str}",
            f"## Pass: {'YES' if full_report.pass_scene else 'NO'}",
        ]
        if related_old_memory:
            parts += ["", "## Related Existing Memory (avoid duplicating these)", related_old_memory]
        parts += ["", "Please generate memory update operations as specified."]
        return "\n".join(parts)
