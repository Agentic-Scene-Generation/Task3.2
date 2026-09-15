"""Per-stage online working memory for render/design/critic loops.

This is intentionally separate from SceneExpert long-term fast memory.  It is a
local scratchpad for the current scene/stage: every render can leave a compact
record, the critic can enrich that record with scores, and the next designer
call can retrieve recent lessons without waiting for end-of-scene MemoryWriter.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time

from dataclasses import asdict, is_dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, Mapping

from scenesmith.agent_utils.furniture_safety import furniture_category_matches
from scenesmith.agent_utils.scoring import compute_total_score, scores_to_dict
from scenesmith.scene_expert.context_bundle import (
    StageContextBundle,
    build_llm_call_debug_record,
)

console_logger = logging.getLogger(__name__)

_SECRET_PATTERNS = (
    re.compile(r"\bsk-[A-Za-z0-9_-]{8,}\b"),
    re.compile(r"(?i)(bearer\s+)[A-Za-z0-9._~+/-]{12,}"),
    re.compile(r"(?i)((?:api[_-]?key|hf_token|token)\s*[=:]\s*)[^\s,;]+"),
)


def _jsonable(value: Any, *, depth: int = 0) -> Any:
    """Convert SDK/Pydantic/dataclass values into bounded JSON-safe evidence."""

    if depth > 12:
        return "[MAX_DEPTH]"
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {
            str(key): _jsonable(item, depth=depth + 1) for key, item in value.items()
        }
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item, depth=depth + 1) for item in value]
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        try:
            return _jsonable(model_dump(mode="json"), depth=depth + 1)
        except TypeError:
            try:
                return _jsonable(model_dump(), depth=depth + 1)
            except Exception:
                pass
        except Exception:
            pass
    if is_dataclass(value):
        try:
            return _jsonable(asdict(value), depth=depth + 1)
        except Exception:
            pass
    if hasattr(value, "__dict__"):
        try:
            return _jsonable(vars(value), depth=depth + 1)
        except Exception:
            pass
    return str(value)


def _redact_secrets(value: Any) -> Any:
    if isinstance(value, str):
        redacted = value
        for pattern in _SECRET_PATTERNS:
            if pattern.groups:
                redacted = pattern.sub(
                    lambda match: match.group(1) + "[REDACTED]", redacted
                )
            else:
                redacted = pattern.sub("[REDACTED]", redacted)
        return redacted
    if isinstance(value, list):
        return [_redact_secrets(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _redact_secrets(item) for key, item in value.items()}
    return value


def _parse_tool_arguments(value: Any) -> Any:
    if not isinstance(value, str):
        return _jsonable(value)
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return value
    return _jsonable(parsed)


def _find_tool_calls(value: Any) -> list[dict[str, Any]]:
    """Find OpenAI Agents SDK function-call payloads without importing the SDK."""

    calls: list[dict[str, Any]] = []

    def visit(item: Any) -> None:
        if isinstance(item, list):
            for child in item:
                visit(child)
            return
        if not isinstance(item, dict):
            return
        item_type = str(item.get("type") or "").lower()
        function = (
            item.get("function") if isinstance(item.get("function"), dict) else {}
        )
        name = item.get("name") or function.get("name")
        arguments = item.get("arguments", function.get("arguments"))
        if name and (
            "function" in item_type or "tool_call" in item_type or arguments is not None
        ):
            calls.append(
                {
                    "type": "function",
                    "id": str(item.get("call_id") or item.get("id") or ""),
                    "function": {
                        "name": str(name),
                        "arguments": _parse_tool_arguments(arguments or {}),
                    },
                }
            )
        for key, child in item.items():
            if key not in {"arguments", "function"}:
                visit(child)

    visit(_jsonable(value))
    unique: list[dict[str, Any]] = []
    seen: set[str] = set()
    for call in calls:
        signature = json.dumps(call, ensure_ascii=False, sort_keys=True, default=str)
        if signature not in seen:
            seen.add(signature)
            unique.append(call)
    return unique


def _extract_agent_result_trace(result: Any, output: Any) -> dict[str, Any]:
    """Persist observable assistant/tool events needed for preference training."""

    if result is None:
        return {
            "assistant_messages": [{"role": "assistant", "content": _jsonable(output)}],
            "tool_calls": [],
            "tool_results": [],
            "new_items": [],
            "raw_responses": [],
            "run_input": [],
            "replay_items": [],
        }
    new_items = _jsonable(getattr(result, "new_items", []) or [])
    raw_responses = _jsonable(getattr(result, "raw_responses", []) or [])
    tool_calls = _find_tool_calls(new_items)
    if not tool_calls:
        tool_calls = _find_tool_calls(raw_responses)
    tool_results: list[dict[str, Any]] = []
    for item in new_items if isinstance(new_items, list) else []:
        item_type = str(item.get("type") or item.get("__class__") or "").lower()
        raw_item = item.get("raw_item") if isinstance(item, dict) else None
        if "toolcalloutput" in item_type or (
            isinstance(item, dict) and "output" in item and raw_item is not None
        ):
            tool_results.append(
                {
                    "tool_call_id": str(
                        item.get("tool_call_id")
                        or item.get("call_id")
                        or (raw_item or {}).get("call_id", "")
                    ),
                    "output": _jsonable(item.get("output")),
                }
            )
    assistant_message: dict[str, Any] = {
        "role": "assistant",
        "content": _jsonable(output),
    }
    if tool_calls:
        assistant_message["tool_calls"] = tool_calls
    replay_items: Any = []
    to_input_list = getattr(result, "to_input_list", None)
    if callable(to_input_list):
        try:
            replay_items = _jsonable(to_input_list())
        except Exception:
            replay_items = []
    return {
        "assistant_messages": [assistant_message],
        "tool_calls": tool_calls,
        "tool_results": tool_results,
        "new_items": new_items,
        "raw_responses": raw_responses,
        "run_input": _jsonable(getattr(result, "input", [])),
        "replay_items": replay_items,
    }


def _serialize_tool(tool: Any) -> dict[str, Any]:
    payload = _jsonable(tool)
    if (
        isinstance(payload, dict)
        and payload.get("type") == "function"
        and isinstance(payload.get("function"), dict)
    ):
        return payload
    if not isinstance(payload, dict):
        return {"type": "function", "function": {"name": str(payload)}}
    name = payload.get("name") or payload.get("tool_name") or type(tool).__name__
    parameters = (
        payload.get("params_json_schema")
        or payload.get("parameters")
        or payload.get("input_schema")
        or {}
    )
    return {
        "type": "function",
        "function": {
            "name": str(name),
            "description": str(payload.get("description") or ""),
            "parameters": _jsonable(parameters),
        },
    }


def _contains_image_input(value: Any) -> bool:
    payload = _jsonable(value)
    if isinstance(payload, list):
        return any(_contains_image_input(item) for item in payload)
    if not isinstance(payload, dict):
        return False
    item_type = str(payload.get("type") or "").lower()
    if item_type in {"input_image", "image", "image_url"}:
        return True
    return any(_contains_image_input(item) for item in payload.values())


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _now_precise() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _compact(text: str, max_chars: int = 700) -> str:
    value = " ".join(str(text or "").split())
    if len(value) <= max_chars:
        return value
    return value[: max_chars - 3] + "..."


def _object_names(scene: Any) -> list[str]:
    try:
        return [
            str(obj.name)
            for obj in scene.objects.values()
            if getattr(obj, "name", None)
        ]
    except Exception:
        return []


def _count_required_categories(
    object_names: list[str], required_categories: list[str]
) -> dict[str, int]:
    return {
        category: sum(
            furniture_category_matches(name, category) for name in object_names
        )
        for category in required_categories
    }


def _extract_grade(scores: dict[str, Any], *name_parts: str) -> float | None:
    for key, value in scores.items():
        key_lower = str(key).lower().replace("_", " ")
        if not all(part.lower().replace("_", " ") in key_lower for part in name_parts):
            continue
        if isinstance(value, dict):
            grade = value.get("grade") or value.get("score")
            if isinstance(grade, (int, float)):
                return float(grade)
        if isinstance(value, (int, float)):
            return float(value)
    return None


def _deterministic_quality(
    *,
    object_names: list[str],
    required_counts: dict[str, int],
    scores: dict[str, Any],
    critique: str,
) -> dict[str, Any]:
    required_counts = {
        str(key).lower(): int(value)
        for key, value in (required_counts or {}).items()
        if int(value) > 0
    }
    observed_counts = _count_required_categories(object_names, list(required_counts))
    missing: list[str] = []
    for category, required in required_counts.items():
        observed = observed_counts.get(category, 0)
        if observed < required:
            missing.extend([category] * (required - observed))

    prompt_following = _extract_grade(scores, "prompt", "following")
    critique_lower = str(critique or "").lower()
    claims_complete = any(
        term in critique_lower
        for term in (
            "all required",
            "all furniture quantities match",
            "bed - present",
            "bed, two nightstands",
            "all required furniture",
        )
    )
    inconsistent = bool(missing) and (
        claims_complete or (prompt_following is not None and prompt_following >= 8)
    )
    hard_valid = not missing
    note = ""
    if missing:
        note = (
            "Deterministic state check: missing required furniture "
            + ", ".join(missing)
            + f"; observed_counts={observed_counts}."
        )
        if inconsistent:
            note += " Ignore contradictory critic/designer text that claims completion."

    return {
        "required_counts": required_counts,
        "observed_counts": observed_counts,
        "missing_required_objects": missing,
        "hard_valid": hard_valid,
        "critic_inconsistent_with_state": inconsistent,
        "deterministic_note": note,
    }


def _scene_hash(scene: Any) -> str:
    try:
        return str(scene.content_hash())
    except Exception:
        return ""


def _score_dict(scores: Any | None) -> dict[str, Any]:
    if scores is None:
        return {}
    try:
        return scores_to_dict(scores)
    except Exception:
        if isinstance(scores, dict):
            return dict(scores)
    return {}


def _score_total(scores: Any | None) -> float | None:
    if scores is None:
        return None
    try:
        return float(compute_total_score(scores))
    except Exception:
        return None


def _canonical_stage(stage: str) -> str:
    if stage == "wall":
        return "wall_mounted"
    if stage == "ceiling":
        return "ceiling_mounted"
    if stage.startswith("manipulands_"):
        return "manipuland"
    return stage


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=str)


class StageWorkingMemory:
    """Scene-local working memory bank keyed by placement stage."""

    def __init__(self, root_dir: Path, stage: str, enabled: bool = True) -> None:
        self.root_dir = Path(root_dir)
        self.stage = stage
        self.enabled = enabled
        self.memory_dir = self.root_dir / "stage_working_memory" / stage
        self.memory_path = self.memory_dir / "memory.jsonl"
        self.timing_path = self.root_dir / "timing_stats.jsonl"
        self.scene_root_dir = (
            self.root_dir.parent
            if self.root_dir.name.startswith("room_")
            else self.root_dir
        )
        self.debug_memory_dir = (
            self.scene_root_dir / "scene_expert" / "working_memory" / stage
        )
        self.debug_memory_path = self.debug_memory_dir / "memory.jsonl"
        self.debug_timing_path = (
            self.scene_root_dir
            / "scene_expert"
            / "timing"
            / "stage_working_timing.jsonl"
        )
        self.debug_llm_path = (
            self.scene_root_dir / "scene_expert" / "timing" / "llm_calls.jsonl"
        )
        self.debug_orchestration_path = (
            self.scene_root_dir
            / "scene_expert"
            / "timing"
            / "planner_orchestration.jsonl"
        )
        self.debug_repair_path = (
            self.scene_root_dir / "scene_expert" / "timing" / "repair_events.jsonl"
        )
        self.debug_llm_payload_dir = (
            self.scene_root_dir / "scene_expert" / "audit" / "llm_payloads"
        )
        self.debug_context_dir = (
            self.scene_root_dir / "scene_expert" / "context_bundles" / stage
        )
        public_dir = os.environ.get("SCENEEXPERT_ACTIVE_MEMORY_BANK_DIR", "")
        public_bank_read_only = os.environ.get(
            "SCENEEXPERT_ACTIVE_MEMORY_BANK_READ_ONLY", ""
        ).strip().casefold() in {"1", "true", "yes", "on"}
        self.public_memory_dir = Path(public_dir) if public_dir else None
        self.public_events_path = (
            self.public_memory_dir / "events.jsonl"
            if self.public_memory_dir and not public_bank_read_only
            else None
        )
        self.required_counts: dict[str, int] = {}
        if enabled:
            self.memory_dir.mkdir(parents=True, exist_ok=True)
            self.memory_path.touch(exist_ok=True)
            self.timing_path.touch(exist_ok=True)
            self.debug_memory_dir.mkdir(parents=True, exist_ok=True)
            self.debug_memory_path.touch(exist_ok=True)
            self.debug_timing_path.parent.mkdir(parents=True, exist_ok=True)
            self.debug_timing_path.touch(exist_ok=True)
            self.debug_llm_path.parent.mkdir(parents=True, exist_ok=True)
            self.debug_llm_path.touch(exist_ok=True)
            self.debug_orchestration_path.touch(exist_ok=True)
            self.debug_repair_path.touch(exist_ok=True)
            self.debug_llm_payload_dir.mkdir(parents=True, exist_ok=True)
            self.debug_context_dir.mkdir(parents=True, exist_ok=True)
            if self.public_events_path is not None:
                self.public_events_path.parent.mkdir(parents=True, exist_ok=True)
                self.public_events_path.touch(exist_ok=True)

    def set_required_counts(self, required_counts: dict[str, int] | None) -> None:
        """Set deterministic required-object counts for this stage."""
        self.required_counts = {
            str(key).lower(): int(value)
            for key, value in (required_counts or {}).items()
            if int(value) > 0
        }

    def save_render_record(
        self,
        *,
        render_dir: Path,
        role: str,
        event: str,
        scene: Any,
        text: str = "",
        scores: Any | None = None,
        critique: str = "",
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Save a compact record for one render or scored render."""
        if not self.enabled:
            return {}

        render_dir = Path(render_dir)
        images = sorted(str(path) for path in render_dir.glob("*.png"))
        score_data = _score_dict(scores)
        object_names = _object_names(scene)
        deterministic_quality = _deterministic_quality(
            object_names=object_names,
            required_counts=self.required_counts,
            scores=score_data,
            critique=critique or text,
        )
        record = {
            "schema_version": "1.0",
            "created_at": _now(),
            "stage": self.stage,
            "role": role,
            "event": event,
            "render_dir": str(render_dir),
            "images": images,
            "scores_path": (
                str(render_dir / "scores.yaml")
                if (render_dir / "scores.yaml").exists()
                else ""
            ),
            "scores": score_data,
            "score_total": _score_total(scores),
            "critique": _compact(critique, max_chars=900),
            "text": _compact(text, max_chars=900),
            "scene_hash": _scene_hash(scene),
            "object_names": object_names,
            "object_count": len(object_names),
            "deterministic_quality": deterministic_quality,
            "extra": extra or {},
        }
        _write_json(render_dir / "render_memory.json", record)
        _append_jsonl(self.memory_path, record)
        _append_jsonl(self.debug_memory_path, record)
        self._commit_public_stage_event(record)
        console_logger.info(
            "[StageWorkingMemory] saved stage=%s role=%s event=%s render=%s "
            "scores=%s objects=%d",
            self.stage,
            role,
            event,
            render_dir,
            bool(score_data),
            record["object_count"],
        )
        return record

    def save_context_bundle(self, bundle: StageContextBundle) -> None:
        """Persist the structured context used before an LLM call."""
        if not self.enabled:
            return
        safe_event = (
            "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in bundle.event)
            or "context"
        )
        path = self.debug_context_dir / f"{int(time.time() * 1000)}_{safe_event}.json"
        try:
            bundle.save(path)
        except Exception as e:
            console_logger.warning("Failed to save StageContextBundle: %s", e)

    def record_llm_call(
        self,
        *,
        agent_role: str,
        event: str,
        prompt: Any,
        output: Any = "",
        result: Any = None,
        raw_response: Any = None,
        error: str = "",
        event_kind: Literal["llm", "system"] = "llm",
        tools: list[Any] | None = None,
        system_instructions: Any = "",
        context_snapshot: dict[str, Any] | None = None,
        image_refs: list[str] | None = None,
        capture_replay: bool = False,
        requested_max_tokens: int | None = None,
        stage_execution_attempt: int | None = None,
        client_cancelled: bool | None = None,
        elapsed_sec: float | None = None,
    ) -> None:
        """Persist legacy debug output or gated replay-ready Slow Memory evidence."""
        if not self.enabled:
            return
        record = build_llm_call_debug_record(
            stage=self.stage,
            agent_role=agent_role,
            event=event,
            prompt=prompt,
            output=output,
            result=result,
            raw_response=raw_response,
            error=error,
            event_kind=event_kind,
            requested_max_tokens=requested_max_tokens,
            stage_execution_attempt=stage_execution_attempt,
            client_cancelled=client_cancelled,
            elapsed_sec=elapsed_sec,
        )
        payload = record.model_dump()
        safe_name = "".join(
            character if character.isalnum() or character in ("-", "_") else "_"
            for character in f"{agent_role}_{event}"
        )
        payload_path = self.debug_llm_payload_dir / (
            f"{int(time.time() * 1000)}_{safe_name}.json"
        )
        try:
            if capture_replay:
                result_trace = _extract_agent_result_trace(result, output)
                tool_names = {
                    str((call.get("function") or {}).get("name") or "")
                    for call in result_trace.get("tool_calls", [])
                    if isinstance(call, dict)
                }
                visual_context_used = (
                    agent_role == "critic"
                    or _contains_image_input(prompt)
                    or "observe_scene" in tool_names
                )
                full_payload = {
                    "schema_version": "2.0",
                    "created_at": _now(),
                    "stage": self.stage,
                    "agent_role": agent_role,
                    "event": event,
                    "prompt": _jsonable(prompt),
                    "output": _jsonable(output),
                    "raw_response": _jsonable(raw_response),
                    "error": error,
                    "system_instructions": _jsonable(system_instructions),
                    "tools": [_serialize_tool(tool) for tool in (tools or [])],
                    "image_refs": (
                        [str(path) for path in (image_refs or [])]
                        if visual_context_used
                        else []
                    ),
                    "context_snapshot": _jsonable(context_snapshot or {}),
                    "agent_trace": result_trace,
                }
            else:
                full_payload = {
                    "schema_version": "1.0",
                    "created_at": _now(),
                    "stage": self.stage,
                    "agent_role": agent_role,
                    "event": event,
                    "prompt": prompt,
                    "output": output,
                    "raw_response": raw_response,
                    "error": error,
                }
            _write_json(
                payload_path,
                _redact_secrets(full_payload) if capture_replay else full_payload,
            )
            payload["payload_ref"] = str(payload_path.relative_to(self.scene_root_dir))
        except Exception as exc:
            console_logger.warning("Failed to persist full LLM audit payload: %s", exc)
        _append_jsonl(self.debug_llm_path, payload)
        if self.public_events_path is not None:
            event_payload = {
                "schema_version": "1.0",
                "created_at": _now(),
                "event_type": "llm_call",
                "stage": self.stage,
                "payload": payload,
            }
            _append_jsonl(self.public_events_path, event_payload)

    def record_planner_orchestration(
        self,
        *,
        call_id: str,
        phase: str,
        operation: str,
        child_agent: str,
        status: str,
        detail: dict[str, Any] | None = None,
    ) -> None:
        """Persist a Planner-to-child dispatch or child-to-Planner resume."""
        if not self.enabled:
            return
        record = {
            "schema_version": "1.0",
            "created_at": _now_precise(),
            "stage": self.stage,
            "actor": "planner",
            "call_id": call_id,
            "phase": phase,
            "operation": operation,
            "child_agent": child_agent,
            "status": status,
            "detail": detail or {},
        }
        _append_jsonl(self.debug_orchestration_path, record)
        if self.public_events_path is not None:
            _append_jsonl(
                self.public_events_path,
                {
                    "schema_version": "1.0",
                    "created_at": record["created_at"],
                    "event_type": "planner_orchestration",
                    "stage": self.stage,
                    "payload": record,
                },
            )

    def record_repair_event(
        self,
        *,
        source: str,
        strategy: str,
        status: str,
        repair_owner: str = "scenesmith_core",
        attempt: int | None = None,
        trigger_reasons: list[str] | None = None,
        actions: list[str] | None = None,
        affected_objects: list[dict[str, Any]] | None = None,
        detail: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Persist one structured automatic-repair decision for audit."""
        if not self.enabled:
            return {}
        record = {
            "schema_version": "scenesmith.repair_event.v1",
            "created_at": _now_precise(),
            "stage": self.stage,
            "room": (
                self.root_dir.name if self.root_dir.name.startswith("room_") else ""
            ),
            "repair_owner": repair_owner,
            "source": source,
            "strategy": strategy,
            "status": status,
            "attempt": attempt,
            "trigger_reasons": trigger_reasons or [],
            "actions": actions or [],
            "affected_objects": affected_objects or [],
            "detail": detail or {},
        }
        _append_jsonl(self.debug_repair_path, record)
        return record

    def retrieve_for_designer(
        self,
        *,
        query: str = "",
        max_items: int = 3,
    ) -> str:
        """Retrieve compact recent/scored lessons for the next designer turn."""
        if not self.enabled or not self.memory_path.exists():
            return ""

        records: list[dict[str, Any]] = []
        with self.memory_path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except Exception:
                    continue

        if not records:
            console_logger.info(
                "[StageWorkingMemory] retrieve stage=%s query=%r -> 0 records",
                self.stage,
                _compact(query, 80),
            )
            return ""

        query_tokens = {token.lower() for token in query.replace(",", " ").split()}

        def rank(record: dict[str, Any]) -> tuple[float, float]:
            quality = record.get("deterministic_quality") or {}
            text = " ".join(
                [
                    str(record.get("text", "")),
                    str(record.get("critique", "")),
                    str(quality.get("deterministic_note", "")),
                    " ".join(record.get("object_names", [])),
                ]
            ).lower()
            overlap = sum(1 for token in query_tokens if token and token in text)
            has_scores = 1.0 if record.get("scores") else 0.0
            is_critic = 1.0 if record.get("role") == "critic" else 0.0
            invalid_penalty = (
                4.0 if quality.get("critic_inconsistent_with_state") else 0.0
            )
            hard_valid_bonus = 0.5 if quality.get("hard_valid", True) else 0.0
            # Invalid records with high hallucinated scores must not outrank
            # deterministic failure notes.
            score_total = (
                0.0
                if quality.get("critic_inconsistent_with_state")
                else (record.get("score_total") or 0.0)
            )
            return (
                overlap + has_scores + is_critic + hard_valid_bonus - invalid_penalty,
                score_total,
            )

        selected = sorted(records, key=rank, reverse=True)[:max_items]
        console_logger.info(
            "[StageWorkingMemory] retrieve stage=%s query=%r -> %d/%d records",
            self.stage,
            _compact(query, 80),
            len(selected),
            len(records),
        )
        lines = [
            f"=== Stage Working Memory: {self.stage} ===",
            "Use these recent render/critic notes to preserve what worked and avoid repeating failed changes.",
        ]
        for index, record in enumerate(selected, start=1):
            score_total = record.get("score_total")
            score_text = (
                f", total_score={score_total:.1f}"
                if isinstance(score_total, (int, float))
                else ""
            )
            lines.append(
                f"{index}. [{record.get('role')}/{record.get('event')}{score_text}] "
                f"objects={record.get('object_names', [])}"
            )
            quality = record.get("deterministic_quality") or {}
            if quality.get("deterministic_note"):
                lines.append(
                    f"   deterministic: {_compact(quality['deterministic_note'], 320)}"
                )
            if record.get("critique") and not quality.get(
                "critic_inconsistent_with_state"
            ):
                lines.append(f"   critic: {_compact(record['critique'], 260)}")
            elif record.get("text"):
                lines.append(f"   note: {_compact(record['text'], 260)}")
            if record.get("render_dir"):
                lines.append(f"   render_dir: {record['render_dir']}")
        lines.append("=== End Stage Working Memory ===")
        return "\n".join(lines)

    def record_timing(
        self,
        *,
        module: str,
        event: str,
        elapsed_sec: float,
        extra: dict[str, Any] | None = None,
    ) -> None:
        """Append a timing event for later optimization analysis."""
        if not self.enabled:
            return
        record = {
            "schema_version": "1.0",
            "created_at": _now(),
            "stage": self.stage,
            "module": module,
            "event": event,
            "elapsed_sec": round(float(elapsed_sec), 3),
            "extra": extra or {},
        }
        _append_jsonl(self.timing_path, record)
        _append_jsonl(self.debug_timing_path, record)
        console_logger.info(
            "[Timing] stage=%s module=%s event=%s elapsed=%.3fs",
            self.stage,
            module,
            event,
            elapsed_sec,
        )

    def _commit_public_stage_event(self, record: dict[str, Any]) -> None:
        """Append critic evidence without bypassing final memory promotion.

        StageWorkingMemory is an execution journal. Long-term success/failure
        records are promoted only after full-scene verification by the strict
        SceneExpert MemoryWriter.
        """
        if self.public_events_path is None or self.public_memory_dir is None:
            return
        event_payload = {
            "schema_version": "1.0",
            "created_at": _now(),
            "event_type": "stage_working_memory",
            "stage": self.stage,
            "role": record.get("role", ""),
            "event": record.get("event", ""),
            "render_dir": record.get("render_dir", ""),
            "scene_hash": record.get("scene_hash", ""),
            "promotion_status": "evidence_only",
            "payload": record,
        }
        _append_jsonl(self.public_events_path, event_payload)


def save_generic_render_memory(
    *,
    root_dir: Path,
    stage: str,
    render_dir: Path,
    scene: Any,
    rendering_mode: str,
    render_name: str | None,
    elapsed_sec: float,
) -> None:
    """Save a render-only record from RenderingManager."""
    stage = _canonical_stage(stage)
    memory = StageWorkingMemory(root_dir=root_dir, stage=stage, enabled=True)
    memory.save_render_record(
        render_dir=render_dir,
        role="render",
        event=render_name or rendering_mode,
        scene=scene,
        text=f"Rendered stage={stage}, mode={rendering_mode}, render_name={render_name or ''}",
        extra={"rendering_mode": rendering_mode, "render_elapsed_sec": elapsed_sec},
    )
    memory.record_timing(
        module="rendering_manager",
        event=render_name or rendering_mode,
        elapsed_sec=elapsed_sec,
        extra={"render_dir": str(render_dir)},
    )
