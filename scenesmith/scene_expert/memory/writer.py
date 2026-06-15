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

from scenesmith.scene_expert.memory.schemas import (
    FailureCase,
    MemoryUpdateOp,
    Skill,
    SuccessCase,
)
from scenesmith.scene_expert.memory.text_builder import build_embedding_text
from scenesmith.scene_expert.schemas import FullVerifyReport

console_logger = logging.getLogger(__name__)
SUCCESS_MEMORY_MIN_OVERALL_SCORE = 0.75
_DETERMINISTIC_FAILURE_KEYWORDS = (
    "deterministic",
    "missing mesh",
    "missing file",
    "file missing",
    "hssd",
    "openclip",
    "clip weight",
    "checkpoint missing",
    "degenerate mesh",
    "invalid mesh",
    "mesh file",
    "asset file",
    "candidate file",
    "geometry failure",
)
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
- For success_case content, include: case_id, room_type, style, stage,
  task_signature, required_objects, functional_zones, scene_summary,
  successful_pattern, positive_guidance, scores, quality_score, confidence,
  embedding_text, trace_ref.
- Only add success_case entries when the final scene is clearly good. If the
  final overall score is below 0.75, do not add success_case entries.
- For failure_case content, include: failure_id, room_type, stage, object,
  failure_type, bad_pattern, failure_reason, repair_action, repair_verified,
  scope, is_deterministic, repeat_count, negative_constraint, critic_check,
  quality_score, confidence, embedding_text, trace_ref.
- Only add failure_case entries when a repair was verified OR the failure is
  deterministic/repeatable, such as missing mesh, degenerate mesh, OpenCLIP
  missing, HSSD file missing, or repeated geometry/asset loading failure.
- For skill content, include: skill_name, stage, room_type, room_types, style,
  required_objects, functional_zones, scene_summary, preconditions, procedure,
  failure_avoidance, postconditions, success_rate, quality_score, confidence,
  embedding_text, trace_ref.
- Do not create a new skill unless the trace shows a reusable multi-step
  procedure. Prefer NOOP over inventing a vague skill.
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
        from openai import OpenAI

        self._model = model
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._client = OpenAI(
            base_url=api_base_url
            or os.environ.get("OPENAI_BASE_URL", "http://localhost:8000/v1"),
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
        user_message = self._build_user_message(
            trace_summary, full_report, related_old_memory
        )

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
            ops = self._gate_and_enrich_ops(ops, full_report)
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
            f"plausibility={full_report.plausibility_score:.2f}, "
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
            parts += [
                "",
                "## Related Existing Memory (avoid duplicating these)",
                related_old_memory,
            ]
        parts += ["", "Please generate memory update operations as specified."]
        return "\n".join(parts)

    def _gate_and_enrich_ops(
        self,
        ops: list[MemoryUpdateOp],
        full_report: FullVerifyReport,
    ) -> list[MemoryUpdateOp]:
        """Apply deterministic quality gates and fill missing retrieval text."""
        filtered: list[MemoryUpdateOp] = []
        for op in ops:
            if op.op == "NOOP":
                filtered.append(op)
                continue
            if op.op not in ("ADD", "UPDATE"):
                console_logger.info(f"MemoryWriter: dropped unsupported op {op.op!r}")
                continue

            if op.memory_type == "success_case":
                if full_report.overall_score < SUCCESS_MEMORY_MIN_OVERALL_SCORE:
                    console_logger.info(
                        "MemoryWriter: dropped success_case below quality gate "
                        f"(overall={full_report.overall_score:.2f})"
                    )
                    continue
                enriched = self._enrich_success_content(op.content, full_report)
                if enriched is not None:
                    filtered.append(op.model_copy(update={"content": enriched}))
                continue

            if op.memory_type == "failure_case":
                enriched = self._enrich_failure_content(op.content)
                if enriched is None:
                    continue
                repair_verified = bool(enriched.get("repair_verified", False))
                deterministic = bool(enriched.get("is_deterministic", False))
                if not repair_verified and not deterministic:
                    console_logger.info(
                        "MemoryWriter: dropped failure_case that is neither "
                        "verified nor deterministic"
                    )
                    continue
                filtered.append(op.model_copy(update={"content": enriched}))
                continue

            if op.memory_type == "skill":
                if op.op == "ADD" and not self._looks_like_reusable_skill(op.content):
                    console_logger.info(
                        "MemoryWriter: dropped vague skill ADD without reusable "
                        "multi-step procedure"
                    )
                    continue
                enriched = self._enrich_skill_content(op.content)
                if enriched is not None:
                    filtered.append(op.model_copy(update={"content": enriched}))
                continue

            console_logger.info(
                f"MemoryWriter: dropped unknown memory_type {op.memory_type!r}"
            )

        return filtered

    def _enrich_success_content(
        self,
        content: dict,
        full_report: FullVerifyReport,
    ) -> dict | None:
        enriched = dict(content)
        enriched.setdefault("quality_score", full_report.overall_score)
        enriched.setdefault("confidence", 0.7 if full_report.pass_scene else 0.5)
        try:
            record = SuccessCase.model_validate(enriched)
        except Exception as e:
            console_logger.info(f"MemoryWriter: dropped invalid success_case: {e}")
            return None
        if not record.embedding_text:
            record = record.model_copy(
                update={"embedding_text": build_embedding_text(record)}
            )
        return record.model_dump()

    def _enrich_failure_content(self, content: dict) -> dict | None:
        enriched = dict(content)
        deterministic = self._detect_deterministic_failure(enriched)
        if deterministic:
            enriched["is_deterministic"] = True
            if enriched.get("scope", "object") == "object":
                enriched["scope"] = "stage"
        try:
            record = FailureCase.model_validate(enriched)
        except Exception as e:
            console_logger.info(f"MemoryWriter: dropped invalid failure_case: {e}")
            return None
        if not record.embedding_text:
            record = record.model_copy(
                update={"embedding_text": build_embedding_text(record)}
            )
        return record.model_dump()

    def _enrich_skill_content(self, content: dict) -> dict | None:
        try:
            record = Skill.model_validate(content)
        except Exception as e:
            console_logger.info(f"MemoryWriter: dropped invalid skill: {e}")
            return None
        if not record.embedding_text:
            record = record.model_copy(
                update={"embedding_text": build_embedding_text(record)}
            )
        return record.model_dump()

    def _detect_deterministic_failure(self, content: dict) -> bool:
        if bool(content.get("is_deterministic", False)):
            return True
        text = " ".join(
            str(content.get(key, ""))
            for key in (
                "failure_type",
                "bad_pattern",
                "failure_reason",
                "repair_action",
                "negative_constraint",
                "critic_check",
            )
        ).lower()
        return any(keyword in text for keyword in _DETERMINISTIC_FAILURE_KEYWORDS)

    def _looks_like_reusable_skill(self, content: dict) -> bool:
        procedure = content.get("procedure") or []
        if (
            not isinstance(procedure, list)
            or len([x for x in procedure if str(x).strip()]) < 2
        ):
            return False
        support_fields = (
            content.get("preconditions") or [],
            content.get("failure_avoidance") or [],
            content.get("postconditions") or [],
        )
        return any(
            isinstance(items, list) and any(str(x).strip() for x in items)
            for items in support_fields
        )
