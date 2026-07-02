"""MemoryWriter: Qwen3 memory_writer role that updates fast memory after each run.

Takes a trace summary + final verifier report and produces structured memory
update operations (ADD/UPDATE/NOOP) for the three memory banks.

MVP only uses ADD, UPDATE, NOOP — DELETE is intentionally not implemented
to avoid accidentally removing useful experience.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any

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
/think
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
        max_tokens: int = 3072,
        temperature: float = 0.1,
        debug_dir: str | Path | None = None,
    ) -> None:
        from openai import OpenAI

        self._model = model
        self._max_tokens = int(
            os.environ.get("SCENEEXPERT_MEMORY_WRITER_MAX_TOKENS", max_tokens)
        )
        self._temperature = temperature
        debug_dir = debug_dir or os.environ.get("SCENEEXPERT_MEMORY_WRITER_DEBUG_DIR")
        self._debug_dir = Path(debug_dir) if debug_dir else None
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

        attempt_logs: list[dict[str, Any]] = []
        attempts = (
            ("json_mode", True),
            ("plain_json_retry", False),
        )
        for label, use_response_format in attempts:
            attempt_log = {
                "label": label,
                "use_response_format": use_response_format,
            }
            try:
                response = self._request_completion(
                    user_message=user_message,
                    use_response_format=use_response_format,
                )
                raw = self._extract_response_text(response)
                attempt_log.update(
                    {
                        "finish_reason": self._response_finish_reason(response),
                        "raw_present": bool(raw),
                        "raw_excerpt": self._compact_text(raw, 2000),
                        "response_snapshot": self._response_snapshot(response),
                    }
                )
                if not raw:
                    raise ValueError(
                        "Qwen/vLLM returned an empty assistant message content. "
                        f"finish_reason={attempt_log['finish_reason']!r}"
                    )

                data = self._parse_json_payload(raw)
                attempt_log["parsed_keys"] = sorted(data.keys())
                attempt_logs.append(attempt_log)
                console_logger.debug("MemoryWriter raw response: %s", raw)
            except Exception as e:
                attempt_log["error"] = f"{type(e).__name__}: {e}"
                attempt_logs.append(attempt_log)
                console_logger.warning("MemoryWriter attempt %s failed: %s", label, e)
                continue

            try:
                ops = [
                    MemoryUpdateOp.model_validate(op)
                    for op in data.get("updates", [])
                ]
                ops = self._gate_and_enrich_ops(ops, full_report)
            except Exception as e:
                attempt_log["error"] = f"{type(e).__name__}: {e}"
                console_logger.warning(
                    "MemoryWriter attempt %s returned invalid update ops: %s",
                    label,
                    e,
                )
                continue

            if self._has_mutating_ops(ops) or not self._should_build_fallback(
                full_report
            ):
                console_logger.info(
                    "MemoryWriter: %d update ops generated via %s",
                    len(ops),
                    label,
                )
                return ops

            fallback_ops = self._fallback_success_ops(trace_summary, full_report)
            fallback_ops = self._gate_and_enrich_ops(fallback_ops, full_report)
            if self._has_mutating_ops(fallback_ops):
                self._save_debug_payload(
                    status="fallback_after_empty_ops",
                    attempts=attempt_logs,
                    trace_summary=trace_summary,
                    full_report=full_report,
                    fallback_ops=fallback_ops,
                )
                console_logger.warning(
                    "MemoryWriter produced no mutating ops for a passed scene; "
                    "using %d conservative fallback success ops.",
                    len(fallback_ops),
                )
                return fallback_ops

            console_logger.info(
                "MemoryWriter: %d non-mutating update ops generated via %s",
                len(ops),
                label,
            )
            return ops

        fallback_ops = self._fallback_success_ops(trace_summary, full_report)
        fallback_ops = self._gate_and_enrich_ops(fallback_ops, full_report)
        self._save_debug_payload(
            status="fallback_after_failed_attempts",
            attempts=attempt_logs,
            trace_summary=trace_summary,
            full_report=full_report,
            fallback_ops=fallback_ops,
        )
        if self._has_mutating_ops(fallback_ops):
            console_logger.warning(
                "MemoryWriter model output was unusable; using %d conservative "
                "fallback success ops.",
                len(fallback_ops),
            )
            return fallback_ops

        console_logger.warning(
            "MemoryWriter failed and no fallback memory passed quality gates; "
            "skipping memory update."
        )
        return []

    def _request_completion(self, user_message: str, use_response_format: bool):
        """Call the OpenAI-compatible server with a Qwen-tolerant retry mode."""
        if use_response_format:
            system_prompt = _SYSTEM_PROMPT
            prompt = user_message
        else:
            system_prompt = (
                _SYSTEM_PROMPT
                + "\nReturn ONLY one JSON object. Do not include markdown fences, "
                "reasoning text, comments, or XML/tool tags."
            )
            prompt = (
                user_message
                + "\n\nReturn ONLY this JSON shape now:\n"
                '{"updates":[{"op":"ADD|UPDATE|NOOP","memory_type":'
                '"success_case|failure_case|skill","target_id":"","content":{}}]}'
            )

        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            "temperature": self._temperature,
            "max_tokens": self._max_tokens,
        }
        if use_response_format:
            kwargs["response_format"] = {"type": "json_object"}
        return self._client.chat.completions.create(**kwargs)

    def _extract_response_text(self, response: Any) -> str:
        """Extract content from OpenAI/vLLM/Qwen response variants."""
        choices = getattr(response, "choices", None) or []
        if not choices:
            return ""

        message = getattr(choices[0], "message", None)
        candidates: list[Any] = []
        if message is not None:
            candidates.extend(
                [
                    getattr(message, "content", None),
                    getattr(message, "reasoning_content", None),
                    getattr(message, "text", None),
                    getattr(message, "refusal", None),
                ]
            )
            dump = self._model_dump(message)
            if isinstance(dump, dict):
                candidates.extend(
                    [
                        dump.get("content"),
                        dump.get("reasoning_content"),
                        dump.get("text"),
                        dump.get("refusal"),
                    ]
                )
                extra = dump.get("model_extra")
                if isinstance(extra, dict):
                    candidates.extend(
                        [
                            extra.get("content"),
                            extra.get("reasoning_content"),
                            extra.get("text"),
                        ]
                    )

        for candidate in candidates:
            text = self._stringify_content(candidate)
            if text:
                return text
        return ""

    def _stringify_content(self, value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value.strip()
        if isinstance(value, list):
            parts: list[str] = []
            for item in value:
                if isinstance(item, dict):
                    text = item.get("text") or item.get("content")
                    if text:
                        parts.append(str(text))
                elif item is not None:
                    parts.append(str(item))
            return "\n".join(part for part in parts if part).strip()
        if isinstance(value, dict):
            for key in ("text", "content", "reasoning_content"):
                if value.get(key):
                    return str(value[key]).strip()
        return str(value).strip()

    def _parse_json_payload(self, raw: str) -> dict:
        """Parse JSON even when a local model wraps it in prose or fences."""
        text = raw.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
            text = re.sub(r"\s*```$", "", text)

        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            parsed = json.loads(self._extract_first_json_object(text))

        if not isinstance(parsed, dict):
            raise ValueError(f"MemoryWriter expected JSON object, got {type(parsed)}")
        if "updates" not in parsed:
            raise ValueError("MemoryWriter JSON object is missing 'updates'")
        if not isinstance(parsed["updates"], list):
            raise ValueError("MemoryWriter JSON 'updates' must be a list")
        return parsed

    def _extract_first_json_object(self, text: str) -> str:
        start = text.find("{")
        if start < 0:
            raise ValueError("No JSON object start found in model output")

        depth = 0
        in_string = False
        escape = False
        for index in range(start, len(text)):
            char = text[index]
            if in_string:
                if escape:
                    escape = False
                elif char == "\\":
                    escape = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    return text[start : index + 1]
        raise ValueError("No complete JSON object found in model output")

    def _has_mutating_ops(self, ops: list[MemoryUpdateOp]) -> bool:
        return any(op.op in ("ADD", "UPDATE") for op in ops)

    def _should_build_fallback(self, full_report: FullVerifyReport) -> bool:
        return (
            bool(full_report.pass_scene)
            and full_report.overall_score >= SUCCESS_MEMORY_MIN_OVERALL_SCORE
        )

    def _fallback_success_ops(
        self,
        trace_summary: str,
        full_report: FullVerifyReport,
    ) -> list[MemoryUpdateOp]:
        """Build conservative success cases when the model response is unusable."""
        if not self._should_build_fallback(full_report):
            return []

        trace_id = self._extract_trace_id(trace_summary)
        prompt = self._extract_prompt(trace_summary)
        room_type = self._infer_room_type(trace_summary)
        required_objects = self._infer_required_objects(trace_summary)
        stages = self._extract_passed_stage_scores(trace_summary)
        if not stages:
            stages = [("furniture", {"overall": full_report.overall_score})]

        ops: list[MemoryUpdateOp] = []
        created_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        for stage, scores in stages:
            digest = hashlib.sha1(
                f"{trace_id}|{stage}|{prompt}|{scores}".encode("utf-8")
            ).hexdigest()[:12]
            content = {
                "case_id": f"success_{room_type}_{stage}_{digest}",
                "room_type": room_type,
                "style": "standard",
                "stage": stage,
                "task_signature": required_objects or [room_type, stage],
                "required_objects": required_objects,
                "functional_zones": [],
                "scene_summary": (
                    "Conservative fallback memory generated from a completed "
                    "SceneExpert trace because the LLM memory-writer response was "
                    "not parseable."
                ),
                "successful_pattern": [
                    f"{stage} passed SceneExpert verifier in trace {trace_id}.",
                    (
                        "Use this only as a weak positive prior; still verify "
                        "collisions, walkability, and plausibility in the new scene."
                    ),
                ],
                "positive_guidance": [
                    (
                        f"For a matching {room_type} task, preserve the verifier-"
                        f"passing {stage} strategy and adapt it to current geometry."
                    ),
                    (
                        "Do not copy coordinates blindly; re-check object sizes, "
                        "door/window constraints, and local support surfaces."
                    ),
                ],
                "scores": scores,
                "trace_ref": trace_id,
                "quality_score": full_report.overall_score,
                "confidence": 0.35,
                "created_at": created_at,
            }
            ops.append(
                MemoryUpdateOp(
                    op="ADD",
                    memory_type="success_case",
                    content=content,
                )
            )
        return ops

    def _extract_passed_stage_scores(
        self, trace_summary: str
    ) -> list[tuple[str, dict[str, float]]]:
        stages: list[tuple[str, dict[str, float]]] = []
        pattern = re.compile(
            r"^\s*\[(?P<stage>[^\]]+)\].*?verify=PASS\s+scores=\((?P<scores>[^)]*)\)",
            re.MULTILINE,
        )
        for match in pattern.finditer(trace_summary):
            stage = match.group("stage").strip()
            scores = self._parse_score_list(match.group("scores"))
            stages.append((stage, scores))
        return stages

    def _parse_score_list(self, score_text: str) -> dict[str, float]:
        scores: dict[str, float] = {}
        for item in score_text.split(","):
            if "=" not in item:
                continue
            key, value = item.split("=", 1)
            try:
                scores[key.strip()] = float(value.strip())
            except ValueError:
                continue
        return scores

    def _extract_trace_id(self, trace_summary: str) -> str:
        match = re.search(r"^Trace:\s*(\S+)", trace_summary, flags=re.MULTILINE)
        return match.group(1) if match else "trace_unknown"

    def _extract_prompt(self, trace_summary: str) -> str:
        match = re.search(r"^Prompt:\s*(.+)$", trace_summary, flags=re.MULTILINE)
        return match.group(1).strip() if match else ""

    def _infer_room_type(self, trace_summary: str) -> str:
        text = trace_summary.lower()
        for room_type in (
            "bedroom",
            "living_room",
            "kitchen",
            "dining_room",
            "office",
            "bathroom",
        ):
            if room_type.replace("_", " ") in text or room_type in text:
                return room_type
        return "room"

    def _infer_required_objects(self, trace_summary: str) -> list[str]:
        text = trace_summary.lower()
        aliases = {
            "bed": ("bed",),
            "nightstand": ("nightstand", "nightstands", "bedside table"),
            "wardrobe": ("wardrobe", "closet"),
            "sofa": ("sofa", "couch"),
            "table": ("table", "desk"),
            "chair": ("chair",),
            "lamp": ("lamp", "light"),
            "painting": ("painting", "artwork", "wall art"),
            "shelf": ("shelf", "shelves"),
        }
        objects = [
            canonical
            for canonical, terms in aliases.items()
            if any(term in text for term in terms)
        ]
        return objects

    def _save_debug_payload(
        self,
        *,
        status: str,
        attempts: list[dict[str, Any]],
        trace_summary: str,
        full_report: FullVerifyReport,
        fallback_ops: list[MemoryUpdateOp],
    ) -> None:
        if self._debug_dir is None:
            return
        self._debug_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": "1.0",
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "status": status,
            "model": self._model,
            "max_tokens": self._max_tokens,
            "full_report": full_report.model_dump(),
            "trace_summary_excerpt": self._compact_text(trace_summary, 6000),
            "attempts": attempts,
            "fallback_ops": [op.model_dump() for op in fallback_ops],
        }
        debug_path = self._debug_dir / "memory_writer_debug.json"
        debug_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
        jsonl_path = self._debug_dir / "memory_writer_debug.jsonl"
        with jsonl_path.open("a", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")

    def _response_finish_reason(self, response: Any) -> str:
        choices = getattr(response, "choices", None) or []
        if not choices:
            return ""
        return str(getattr(choices[0], "finish_reason", "") or "")

    def _response_snapshot(self, response: Any) -> dict[str, Any]:
        dumped = self._model_dump(response)
        if isinstance(dumped, dict):
            return dumped
        return {"repr": self._compact_text(repr(response), 4000)}

    def _model_dump(self, value: Any) -> Any:
        if hasattr(value, "model_dump"):
            try:
                return value.model_dump()
            except Exception:
                return None
        if hasattr(value, "dict"):
            try:
                return value.dict()
            except Exception:
                return None
        return None

    def _compact_text(self, text: Any, max_chars: int) -> str:
        value = "" if text is None else str(text)
        if len(value) <= max_chars:
            return value
        return value[: max_chars - 3] + "..."

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
