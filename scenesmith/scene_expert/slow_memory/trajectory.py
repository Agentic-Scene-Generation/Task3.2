"""Read-only capture of replayable SceneSmith trajectories for preference learning."""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import logging
import re
import time

from pathlib import Path
from typing import Any, Iterable

from scenesmith.scene_expert.schemas import (
    RepairResult,
    SceneTaskSpec,
    StageVerifyReport,
)
from scenesmith.scene_expert.slow_memory.schemas import (
    PreferenceEvidence,
    TrajectoryOutcome,
    TrajectoryRecord,
)

console_logger = logging.getLogger(__name__)

_DESIGNER_EVENTS = frozenset({"request_initial_design", "request_design_change"})
_CRITIC_EVENTS = frozenset({"score_scene", "score_scene_transient_fallback"})
_VOLATILE_CONTEXT_KEYS = frozenset(
    {
        "created_at",
        "updated_at",
        "run_id",
        "scene_id",
        "trace_id",
        "request_id",
        "elapsed_sec",
        "latency_sec",
        "queue_wait_sec",
        "ttft_sec",
        "decode_sec",
    }
)
_SECRET_PATTERNS = (
    re.compile(r"\bsk-[A-Za-z0-9_-]{8,}\b"),
    re.compile(r"(?i)(bearer\s+)[A-Za-z0-9._~+/-]{12,}"),
    re.compile(r"(?i)((?:api[_-]?key|hf_token|token)\s*[=:]\s*)[^\s,;]+"),
)
_DATA_IMAGE_URI = re.compile(
    r"data:(?P<mime>image/[A-Za-z0-9.+-]+);base64,(?P<data>[A-Za-z0-9+/=]+)"
)


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _stable_hash(value: Any, length: int = 24) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:length]


def _stringify(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def redact_sensitive_text(value: Any) -> str:
    """Remove common credential forms before a payload becomes training data."""

    text = _stringify(value)
    for pattern in _SECRET_PATTERNS:
        if pattern.groups:
            text = pattern.sub(lambda match: match.group(1) + "[REDACTED]", text)
        else:
            text = pattern.sub("[REDACTED]", text)
    return text


def _redact_value(value: Any) -> Any:
    if isinstance(value, str):
        return redact_sensitive_text(value)
    if isinstance(value, list):
        return [_redact_value(item) for item in value]
    if isinstance(value, tuple):
        return [_redact_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _redact_value(item) for key, item in value.items()}
    return value


def _bounded_text(value: Any, max_chars: int) -> tuple[str, bool]:
    text = redact_sensitive_text(value)
    if len(text) <= max_chars:
        return text, True
    return text[:max_chars], False


def _bounded_messages(
    value: list[dict[str, Any]],
    max_chars: int,
    *,
    fallback_role: str,
) -> tuple[list[dict[str, Any]], bool]:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    if len(encoded) <= max_chars:
        return value, True
    digest = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
    header = f"[TRUNCATED structured payload sha256={digest}]\n"
    return (
        [
            {
                "role": fallback_role,
                "content": header + encoded[: max(0, max_chars - len(header))],
            }
        ],
        False,
    )


def _bounded_structure(value: Any, max_chars: int, *, empty: Any) -> tuple[Any, bool]:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    if len(encoded) <= max_chars:
        return value, True
    digest = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
    if isinstance(empty, dict):
        return {
            "_truncated": True,
            "sha256": digest,
            "prefix": encoded[:max_chars],
        }, False
    return empty, False


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _load_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return records
    for line in lines:
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            records.append(payload)
    return records


def _mean_score(values: dict[str, Any]) -> float | None:
    numbers = [
        float(value) for value in values.values() if isinstance(value, (int, float))
    ]
    return sum(numbers) / len(numbers) if numbers else None


def _canonical_context(value: Any) -> Any:
    """Strip volatile run identity while retaining model-visible conditions."""

    if isinstance(value, dict):
        return {
            str(key): _canonical_context(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
            if str(key) not in _VOLATILE_CONTEXT_KEYS
        }
    if isinstance(value, list):
        return [_canonical_context(item) for item in value]
    return value


def _replay_completion_messages(
    trace: dict[str, Any],
    *,
    fallback: str,
) -> list[dict[str, Any]]:
    replay = trace.get("replay_items")
    run_input = trace.get("run_input")
    if not isinstance(replay, list) or not isinstance(run_input, list):
        return []
    if replay[: len(run_input)] != run_input:
        return []
    items = replay[len(run_input) :]
    messages: list[dict[str, Any]] = []
    call_names: dict[str, str] = {}
    for wrapped in items:
        if not isinstance(wrapped, dict):
            continue
        item = (
            wrapped.get("raw_item")
            if isinstance(wrapped.get("raw_item"), dict)
            else wrapped
        )
        item_type = str(item.get("type") or wrapped.get("type") or "").lower()
        if item_type in {"function_call", "tool_call", "tool_call_item"}:
            call_id = str(item.get("call_id") or item.get("id") or "")
            function = (
                item.get("function") if isinstance(item.get("function"), dict) else {}
            )
            name = str(item.get("name") or function.get("name") or "tool")
            arguments = item.get("arguments", function.get("arguments", {}))
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    pass
            call_names[call_id] = name
            call = {
                "type": "function",
                "id": call_id,
                "function": {
                    "name": name,
                    "arguments": _redact_value(arguments),
                },
            }
            if (
                messages
                and messages[-1].get("role") == "assistant"
                and messages[-1].get("content") == ""
                and messages[-1].get("tool_calls")
            ):
                messages[-1]["tool_calls"].append(call)
            else:
                messages.append(
                    {"role": "assistant", "content": "", "tool_calls": [call]}
                )
            continue
        if item_type in {
            "function_call_output",
            "tool_call_output",
            "tool_call_output_item",
        }:
            call_id = str(
                item.get("call_id")
                or item.get("tool_call_id")
                or wrapped.get("call_id")
                or wrapped.get("tool_call_id")
                or ""
            )
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "name": call_names.get(call_id, "tool"),
                    "content": redact_sensitive_text(
                        item.get("output", wrapped.get("output", ""))
                    ),
                }
            )
            continue
        role = str(item.get("role") or "")
        if role == "assistant":
            messages.append(
                {
                    "role": "assistant",
                    "content": _trl_content(item.get("content", "")),
                }
            )
    if not any(message.get("role") == "assistant" for message in messages):
        return []
    if fallback.strip() and not any(
        message.get("role") == "assistant"
        and (message.get("content") not in ("", []) or bool(message.get("tool_calls")))
        for message in messages
    ):
        messages.append({"role": "assistant", "content": fallback})
    return messages


def _assistant_messages(payload: dict[str, Any], fallback: str) -> list[dict[str, Any]]:
    trace = (
        payload.get("agent_trace")
        if isinstance(payload.get("agent_trace"), dict)
        else {}
    )
    replay_messages = _replay_completion_messages(trace, fallback=fallback)
    if replay_messages:
        return replay_messages
    calls = trace.get("tool_calls") if isinstance(trace.get("tool_calls"), list) else []
    results = (
        trace.get("tool_results") if isinstance(trace.get("tool_results"), list) else []
    )
    if calls:
        normalized: list[dict[str, Any]] = [
            {"role": "assistant", "content": "", "tool_calls": _redact_value(calls)}
        ]
        names = {
            str(call.get("id") or ""): str(
                (call.get("function") or {}).get("name") or "tool"
            )
            for call in calls
            if isinstance(call, dict)
        }
        for result in results:
            if not isinstance(result, dict):
                continue
            call_id = str(result.get("tool_call_id") or "")
            normalized.append(
                {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "name": names.get(call_id, "tool"),
                    "content": redact_sensitive_text(result.get("output")),
                }
            )
        if fallback.strip():
            normalized.append({"role": "assistant", "content": fallback})
        return normalized
    messages = trace.get("assistant_messages")
    if not isinstance(messages, list):
        messages = []
    normalized: list[dict[str, Any]] = []
    for raw in messages:
        if not isinstance(raw, dict):
            continue
        message = _redact_value(raw)
        message["role"] = "assistant"
        content = message.get("content", "")
        if not isinstance(content, (str, list)):
            message["content"] = _stringify(content)
        normalized.append(message)
    if not normalized:
        normalized = [{"role": "assistant", "content": fallback}]
    return normalized


def _trl_content(value: Any) -> Any:
    """Normalize OpenAI input content into TRL's text/image item schema."""

    if isinstance(value, str):
        return redact_sensitive_text(value)
    if not isinstance(value, list):
        return redact_sensitive_text(value)
    content: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict):
            content.append({"type": "text", "text": redact_sensitive_text(item)})
            continue
        item_type = str(item.get("type") or "").lower()
        if item_type in {"input_image", "image", "image_url"}:
            content.append({"type": "image"})
        elif item_type in {"input_text", "output_text", "text"}:
            content.append(
                {
                    "type": "text",
                    "text": redact_sensitive_text(
                        item.get("text") or item.get("content") or ""
                    ),
                }
            )
    return content or redact_sensitive_text(value)


def _trl_message(value: dict[str, Any]) -> dict[str, Any] | None:
    role = str(value.get("role") or "")
    if role not in {"system", "user", "assistant", "tool"}:
        return None
    message: dict[str, Any] = {
        "role": role,
        "content": _trl_content(value.get("content", "")),
    }
    for key in ("name", "tool_call_id", "tool_calls"):
        if key in value:
            message[key] = _redact_value(value[key])
    return message


def _message_prompt(payload: dict[str, Any], prompt: str) -> list[dict[str, Any]]:
    conversation = payload.get("conversation_messages")
    if isinstance(conversation, list) and conversation:
        return [
            _redact_value(message)
            for message in conversation
            if isinstance(message, dict)
        ]
    messages: list[dict[str, Any]] = []
    instructions = payload.get("system_instructions")
    if instructions not in (None, "", {}):
        messages.append(
            {"role": "system", "content": redact_sensitive_text(instructions)}
        )
    run_input = (
        payload.get("agent_trace", {}).get("run_input")
        if isinstance(payload.get("agent_trace"), dict)
        else None
    )
    if isinstance(run_input, list) and run_input:
        input_count = len(messages)
        for item in run_input:
            if isinstance(item, dict):
                normalized = _trl_message(item)
                if normalized is not None:
                    messages.append(normalized)
        if len(messages) == input_count:
            messages.append({"role": "user", "content": prompt})
    else:
        messages.append({"role": "user", "content": prompt})
    return messages


def _action_trace(payload: dict[str, Any]) -> list[dict[str, Any]]:
    trace = (
        payload.get("agent_trace")
        if isinstance(payload.get("agent_trace"), dict)
        else {}
    )
    calls = trace.get("tool_calls") if isinstance(trace.get("tool_calls"), list) else []
    results = (
        trace.get("tool_results") if isinstance(trace.get("tool_results"), list) else []
    )
    results_by_id = {
        str(item.get("tool_call_id") or ""): item
        for item in results
        if isinstance(item, dict)
    }
    actions: list[dict[str, Any]] = []
    for index, call in enumerate(calls):
        if not isinstance(call, dict):
            continue
        call_id = str(call.get("id") or call.get("call_id") or "")
        actions.append(
            {
                "sequence_index": index,
                "tool_call": _redact_value(call),
                "tool_result": _redact_value(results_by_id.get(call_id, {})),
            }
        )
    return actions


def _action_trace_from_messages(
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Reconstruct ordered calls/results when SDK wrapper extraction is sparse."""

    actions: list[dict[str, Any]] = []
    by_id: dict[str, dict[str, Any]] = {}
    for message in messages:
        if not isinstance(message, dict):
            continue
        if message.get("role") == "assistant":
            for call in message.get("tool_calls") or []:
                if not isinstance(call, dict):
                    continue
                normalized = _redact_value(call)
                call_id = str(call.get("id") or call.get("call_id") or "")
                action = {
                    "sequence_index": len(actions),
                    "tool_call": normalized,
                    "tool_result": {},
                }
                actions.append(action)
                if call_id:
                    by_id[call_id] = action
        elif message.get("role") == "tool":
            call_id = str(message.get("tool_call_id") or "")
            action = by_id.get(call_id)
            if action is not None:
                action["tool_result"] = {
                    "tool_call_id": call_id,
                    "name": str(message.get("name") or "tool"),
                    "output": _redact_value(message.get("content", "")),
                }
    return actions


def _task_type(agent_role: str, event: str) -> str:
    if agent_role == "designer" and event == "request_initial_design":
        return "designer_initial"
    if agent_role == "designer" and event == "request_design_change":
        return "designer_repair"
    if agent_role == "critic":
        return "critic_advice"
    if agent_role == "repair":
        return "deterministic_repair"
    return "legacy"


def _outcome_from_report(
    report: StageVerifyReport | None,
    *,
    action_count: int | None = None,
    evidence_ref: str = "",
) -> TrajectoryOutcome:
    if report is None:
        return TrajectoryOutcome(tool_call_count=action_count)
    issues = list(report.issues)
    issue_ids = [
        str(issue.constraint_id or issue.issue_type)
        for issue in issues
        if str(issue.constraint_id or issue.issue_type)
    ]
    deterministic_score = _mean_score(report.rule_scores)
    visual_score = _mean_score(report.visual_scores or report.scores)
    relation_satisfaction = None
    hard_report = report.hard_check_report or {}
    resolved_ids = [
        str(value)
        for value in (
            hard_report.get("resolved_constraint_ids")
            or hard_report.get("resolved_constraints")
            or []
        )
    ]
    introduced_ids = [
        str(value)
        for value in (
            hard_report.get("introduced_constraint_ids")
            or hard_report.get("new_constraint_ids")
            or []
        )
    ]
    new_hard_count = hard_report.get("new_hard_violation_count")
    if not isinstance(new_hard_count, int):
        new_hard_count = len(introduced_ids) if introduced_ids else None
    hard_passed = hard_report.get("hard_passed")
    if not isinstance(hard_passed, bool):
        hard_passed = report.pass_stage and not issues
    for key in (
        "relation_satisfaction",
        "constraint_satisfaction_rate",
    ):
        value = hard_report.get(key)
        if isinstance(value, (int, float)):
            relation_satisfaction = max(0.0, min(1.0, float(value)))
            break
    return TrajectoryOutcome(
        execution_complete=True,
        stage_passed=report.pass_stage,
        hard_passed=hard_passed,
        hard_violation_count=len(issues),
        new_hard_violation_count=new_hard_count,
        resolved_constraint_ids=resolved_ids,
        introduced_constraint_ids=introduced_ids,
        relation_satisfaction=relation_satisfaction,
        deterministic_score=deterministic_score,
        visual_score=visual_score,
        tool_call_count=action_count,
        score_vector={
            **{
                f"rule.{key}": float(value)
                for key, value in report.rule_scores.items()
                if isinstance(value, (int, float))
            },
            **{
                f"visual.{key}": float(value)
                for key, value in (report.visual_scores or report.scores).items()
                if isinstance(value, (int, float))
            },
        },
        issue_ids=issue_ids,
        evidence_refs=[evidence_ref] if evidence_ref else [],
        causal_link_verified=True,
    )


class TrajectoryCollector:
    """Capture evidence after Main has persisted its authoritative decisions.

    The collector never calls a model, changes a scene, executes a tool, selects
    a checkpoint, or feeds labels back into the online generation path.
    """

    def __init__(
        self,
        *,
        scene_debug_dir: Path,
        prompt: str,
        scene_id: str,
        run_id: str,
        task_spec: SceneTaskSpec,
        experiment_signature: str = "",
        config_hash: str = "",
        model_id: str = "",
        capture_mode: str = "full",
        component_flags: dict[str, bool] | None = None,
        code_provenance: dict[str, Any] | None = None,
        max_prompt_chars: int = 131072,
        max_response_chars: int = 1048576,
    ) -> None:
        self.scene_debug_dir = Path(scene_debug_dir)
        self.output_dir = self.scene_debug_dir / "slow_memory"
        self.trajectory_path = self.output_dir / "trajectories.jsonl"
        self.manifest_path = self.output_dir / "capture_manifest.json"
        self.prompt = str(prompt)
        self.scene_id = str(scene_id)
        self.run_id = str(run_id)
        self.task_id = "task_" + _stable_hash(" ".join(self.prompt.split()), 16)
        self.task_spec = task_spec
        self.experiment_signature = experiment_signature
        self.config_hash = config_hash
        self.model_id = model_id
        self.capture_mode = str(capture_mode)
        self.component_flags = dict(component_flags or {})
        self.code_provenance = _redact_value(code_provenance or {})
        self.max_prompt_chars = max(1024, int(max_prompt_chars))
        self.max_response_chars = max(1024, int(max_response_chars))
        self._seen_ids = {
            str(record.get("trajectory_id"))
            for record in _load_jsonl(self.trajectory_path)
            if record.get("trajectory_id")
        }
        self._counts = {"designer": 0, "critic": 0, "repair": 0, "unlabeled": 0}
        self._invalid_media_count = 0

    @staticmethod
    def _media_suffix(data: bytes, declared_mime: str = "") -> tuple[str, str]:
        if data.startswith(b"\x89PNG\r\n\x1a\n"):
            return ".png", "image/png"
        if data.startswith(b"\xff\xd8\xff"):
            return ".jpg", "image/jpeg"
        if data.startswith((b"GIF87a", b"GIF89a")):
            return ".gif", "image/gif"
        if data.startswith(b"RIFF") and data[8:12] == b"WEBP":
            return ".webp", "image/webp"
        suffixes = {
            "image/png": ".png",
            "image/jpeg": ".jpg",
            "image/jpg": ".jpg",
            "image/webp": ".webp",
            "image/gif": ".gif",
        }
        return suffixes.get(declared_mime.casefold(), ".bin"), (
            declared_mime or "application/octet-stream"
        )

    def _store_media(
        self,
        data: bytes,
        *,
        declared_mime: str = "",
        source_kind: str,
    ) -> dict[str, Any]:
        digest = hashlib.sha256(data).hexdigest()
        suffix, mime_type = self._media_suffix(data, declared_mime)
        relative = Path("media") / f"{digest}{suffix}"
        destination = self.output_dir / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            destination.write_bytes(data)
        return {
            "path": relative.as_posix(),
            "sha256": digest,
            "size_bytes": len(data),
            "mime_type": mime_type,
            "source_kind": source_kind,
        }

    def _externalize_media_value(
        self,
        value: Any,
        *,
        source_kind: str,
    ) -> tuple[Any, list[dict[str, Any]]]:
        """Replace embedded image bytes with stable local URI references."""

        refs: list[dict[str, Any]] = []
        if isinstance(value, dict):
            output: dict[str, Any] = {}
            for key, item in value.items():
                sanitized, nested = self._externalize_media_value(
                    item, source_kind=source_kind
                )
                output[str(key)] = sanitized
                refs.extend(nested)
            return output, refs
        if isinstance(value, (list, tuple)):
            output_list: list[Any] = []
            for item in value:
                sanitized, nested = self._externalize_media_value(
                    item, source_kind=source_kind
                )
                output_list.append(sanitized)
                refs.extend(nested)
            return output_list, refs
        if not isinstance(value, str) or "data:image/" not in value:
            return _redact_value(value), refs

        def replace(match: re.Match[str]) -> str:
            try:
                data = base64.b64decode(match.group("data"), validate=True)
            except (binascii.Error, ValueError):
                self._invalid_media_count += 1
                return "sceneexpert-media://invalid-base64"
            reference = self._store_media(
                data,
                declared_mime=match.group("mime"),
                source_kind=source_kind,
            )
            refs.append(reference)
            return f"sceneexpert-media://sha256/{reference['sha256']}"

        return _DATA_IMAGE_URI.sub(replace, redact_sensitive_text(value)), refs

    def _materialize_input_images(
        self, payload: dict[str, Any]
    ) -> list[dict[str, Any]]:
        references: list[dict[str, Any]] = []
        for raw in payload.get("image_refs") or []:
            source = Path(str(raw))
            try:
                if not source.is_file():
                    self._invalid_media_count += 1
                    references.append(
                        {
                            "path": "",
                            "sha256": "",
                            "size_bytes": 0,
                            "source_kind": "model_input",
                            "source_path_hash": _stable_hash(str(source), 64),
                            "missing": True,
                        }
                    )
                    continue
                reference = self._store_media(
                    source.read_bytes(),
                    declared_mime="",
                    source_kind="model_input",
                )
                reference["source_path_hash"] = _stable_hash(str(source.resolve()), 64)
                references.append(reference)
            except OSError:
                self._invalid_media_count += 1
                continue
        unique: dict[str, dict[str, Any]] = {}
        for reference in references:
            unique[str(reference["sha256"])] = reference
        return list(unique.values())

    def _relative_ref(self, path: Path) -> str:
        try:
            return path.resolve().relative_to(self.scene_debug_dir.resolve()).as_posix()
        except ValueError:
            return path.name

    def _append(self, record: TrajectoryRecord) -> bool:
        if record.trajectory_id in self._seen_ids:
            return False
        self.output_dir.mkdir(parents=True, exist_ok=True)
        with self.trajectory_path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(record.model_dump_json() + "\n")
        self._seen_ids.add(record.trajectory_id)
        self._counts[record.agent_role] = self._counts.get(record.agent_role, 0) + 1
        if record.evidence.verdict == "unlabeled":
            self._counts["unlabeled"] += 1
        return True

    def _stage_evidence(
        self,
        stage: str,
        report: StageVerifyReport,
        report_ref: str,
    ) -> PreferenceEvidence:
        has_critic = bool(
            report.vlm_scoring_performed
            and report.score_source == "scenebenchmark_critic"
        )
        kind = "critic_and_deterministic" if has_critic else "deterministic"
        visual_score = _mean_score(report.visual_scores or report.scores)
        deterministic_score = _mean_score(report.rule_scores)
        base_score = visual_score if visual_score is not None else deterministic_score
        normalized_score = max(0.0, min(1.0, float(base_score or 0.0)))
        quality_score = (1.0 if report.pass_stage else 0.0) + normalized_score
        evidence_payload = {
            "stage": stage,
            "pass_stage": report.pass_stage,
            "score_source": report.score_source,
            "visual_scores": report.visual_scores,
            "rule_scores": report.rule_scores,
            "issues": [issue.model_dump() for issue in report.issues],
            "informational_issues": [
                issue.model_dump() for issue in report.informational_issues
            ],
            "critique_summary": report.critique_summary,
            "repair_suggestions": report.repair_suggestions,
            "hard_check_report": report.hard_check_report,
            "runtime_repair_events": report.runtime_repair_events,
        }
        return PreferenceEvidence(
            evidence_id="evidence_" + _stable_hash(evidence_payload),
            kind=kind,
            verdict="accepted" if report.pass_stage else "rejected",
            source=(
                "main_scenebenchmark_critic_plus_stage_rules"
                if has_critic
                else "deterministic_stage_verifier"
            ),
            authoritative=True,
            quality_score=quality_score,
            report_ref=report_ref,
            details=evidence_payload,
        )

    def _llm_payloads(self, stage: str) -> list[tuple[Path, dict[str, Any]]]:
        payload_dir = self.scene_debug_dir / "audit" / "llm_payloads"
        records: list[tuple[Path, dict[str, Any]]] = []
        critic_protocol: list[tuple[Path, dict[str, Any]]] = []
        for path in sorted(payload_dir.glob("*.json")):
            payload = _load_json(path)
            if not payload or payload.get("stage") != stage:
                continue
            role = str(payload.get("agent_role") or "")
            event = str(payload.get("event") or "")
            if role == "designer":
                critic_protocol = []
                if event not in _DESIGNER_EVENTS:
                    continue
            elif role == "critic":
                if event not in _CRITIC_EVENTS:
                    critic_protocol.append((path, payload))
                    continue
                payload = self._with_critic_protocol_context(
                    payload,
                    protocol=critic_protocol,
                )
                critic_protocol = []
            else:
                continue
            trace = (
                payload.get("agent_trace")
                if isinstance(payload.get("agent_trace"), dict)
                else {}
            )
            has_actions = bool(trace.get("tool_calls"))
            if role == "designer" and not has_actions:
                replay_messages = _replay_completion_messages(trace, fallback="")
                has_actions = bool(_action_trace_from_messages(replay_messages))
            if role == "designer" and payload.get("error") and not has_actions:
                continue
            if not str(payload.get("output") or "").strip() and not has_actions:
                continue
            records.append((path, payload))
        return records

    def _with_critic_protocol_context(
        self,
        payload: dict[str, Any],
        *,
        protocol: list[tuple[Path, dict[str, Any]]],
    ) -> dict[str, Any]:
        """Reconstruct the observable context of Main's three-step critic call."""

        enriched = dict(payload)
        messages: list[dict[str, Any]] = []
        instructions = payload.get("system_instructions")
        if instructions not in (None, "", {}):
            messages.append(
                {"role": "system", "content": redact_sensitive_text(instructions)}
            )
        image_refs: list[str] = []
        protocol_refs: list[str] = []
        for protocol_path, protocol_payload in protocol:
            protocol_refs.append(self._relative_ref(protocol_path))
            messages.append(
                {
                    "role": "user",
                    "content": redact_sensitive_text(protocol_payload.get("prompt")),
                }
            )
            protocol_output = redact_sensitive_text(protocol_payload.get("output"))
            messages.extend(_assistant_messages(protocol_payload, protocol_output))
            raw_refs = protocol_payload.get("image_refs")
            if isinstance(raw_refs, list):
                image_refs.extend(str(value) for value in raw_refs)
        messages.append(
            {
                "role": "user",
                "content": redact_sensitive_text(payload.get("prompt")),
            }
        )
        own_refs = payload.get("image_refs")
        if isinstance(own_refs, list):
            image_refs.extend(str(value) for value in own_refs)
        enriched["conversation_messages"] = messages
        enriched["critic_protocol_refs"] = protocol_refs
        enriched["image_refs"] = list(dict.fromkeys(image_refs))
        return enriched

    def _capture_llm_record(
        self,
        *,
        path: Path,
        payload: dict[str, Any],
        evidence: PreferenceEvidence,
        outcome: TrajectoryOutcome,
        report_ref: str,
        downstream_ref: str = "",
        final_scene_context: dict[str, Any] | None = None,
        scene_state_path: str = "",
    ) -> bool:
        role = str(payload.get("agent_role") or "designer")
        event = str(payload.get("event") or role)
        prompt_value, prompt_media_refs = self._externalize_media_value(
            payload.get("prompt"), source_kind="prompt_embedded"
        )
        prompt, prompt_complete = _bounded_text(prompt_value, self.max_prompt_chars)
        raw_output = payload.get("output")
        sanitized_output, output_media_refs = self._externalize_media_value(
            raw_output, source_kind="sdk_output"
        )
        output_text, output_complete = _bounded_text(
            sanitized_output, self.max_response_chars
        )
        completion_messages = _assistant_messages(payload, output_text)
        completion_messages, completion_media_refs = self._externalize_media_value(
            completion_messages, source_kind="tool_observation"
        )
        action_trace = _action_trace(payload)
        if not action_trace:
            action_trace = _action_trace_from_messages(completion_messages)
        action_trace, action_media_refs = self._externalize_media_value(
            action_trace, source_kind="tool_observation"
        )
        response_value: Any = completion_messages
        response, response_complete = _bounded_text(
            response_value, self.max_response_chars
        )
        messages = _message_prompt(payload, prompt)
        messages, message_media_refs = self._externalize_media_value(
            messages, source_kind="prompt_embedded"
        )
        tools, tool_schema_media_refs = self._externalize_media_value(
            payload.get("tools") or [], source_kind="tool_schema"
        )
        image_refs = self._materialize_input_images(payload)
        spatial_context, spatial_media_refs = self._externalize_media_value(
            payload.get("context_snapshot") or {}, source_kind="spatial_context"
        )
        canonical_context = {
            "role": role,
            "event": event,
            "task_type": _task_type(role, event),
            "messages": messages,
            "tools": tools,
            "image_hashes": [
                item.get("sha256") or item.get("path") for item in image_refs
            ],
            "spatial_context": spatial_context,
        }
        context_hash = _stable_hash(_canonical_context(canonical_context), 64)
        raw_completion_messages = completion_messages
        raw_action_trace = action_trace
        messages, messages_complete = _bounded_messages(
            messages,
            self.max_prompt_chars,
            fallback_role="user",
        )
        completion_messages, completion_complete = _bounded_messages(
            completion_messages,
            self.max_response_chars,
            fallback_role="assistant",
        )
        tools, tools_complete = _bounded_structure(
            tools,
            self.max_prompt_chars,
            empty=[],
        )
        spatial_context, spatial_complete = _bounded_structure(
            spatial_context,
            self.max_prompt_chars,
            empty={},
        )
        action_trace, action_complete = _bounded_structure(
            action_trace,
            self.max_response_chars,
            empty=[],
        )
        prompt_complete = (
            prompt_complete
            and messages_complete
            and tools_complete
            and spatial_complete
        )
        response_complete = (
            response_complete and completion_complete and action_complete
        )
        payload_ref = self._relative_ref(path)
        protocol_refs = [
            str(ref) for ref in (payload.get("critic_protocol_refs") or []) if str(ref)
        ]
        trajectory_id = "trajectory_" + _stable_hash(
            [
                self.run_id,
                self.scene_id,
                payload_ref,
                context_hash,
                raw_completion_messages,
            ]
        )
        trace = (
            payload.get("agent_trace")
            if isinstance(payload.get("agent_trace"), dict)
            else {}
        )
        bounded_final_context, final_context_complete = _bounded_structure(
            _redact_value(final_scene_context or {}),
            self.max_prompt_chars,
            empty={},
        )
        raw_items = trace.get("new_items") or []
        raw_responses = trace.get("raw_responses") or []
        raw_items_hash = _stable_hash(_redact_value(raw_items), 64)
        raw_responses_hash = _stable_hash(_redact_value(raw_responses), 64)
        embedded_media_refs = [
            *prompt_media_refs,
            *output_media_refs,
            *completion_media_refs,
            *action_media_refs,
            *message_media_refs,
            *tool_schema_media_refs,
            *spatial_media_refs,
        ]
        unique_embedded_media = {
            str(reference.get("sha256") or ""): reference
            for reference in embedded_media_refs
            if reference.get("sha256")
        }
        record = TrajectoryRecord(
            trajectory_id=trajectory_id,
            created_at=str(payload.get("created_at") or _utc_now()),
            run_id=self.run_id,
            scene_id=self.scene_id,
            task_id=self.task_id,
            scenario_family_id=self.task_id,
            experiment_signature=self.experiment_signature,
            config_hash=self.config_hash,
            model_id=self.model_id,
            stage=str(payload.get("stage") or ""),
            agent_role=role,
            event=event,
            task_type=_task_type(role, event),
            context_hash=context_hash,
            prompt=prompt,
            response=response,
            response_hash=_stable_hash(raw_completion_messages, 64),
            prompt_complete=prompt_complete,
            response_complete=response_complete,
            evidence=evidence,
            source_refs=[
                ref for ref in (payload_ref, *protocol_refs, report_ref) if ref
            ],
            messages=messages,
            completion_messages=completion_messages,
            tools=tools,
            image_refs=image_refs,
            spatial_context=spatial_context,
            action_trace=action_trace,
            outcome=outcome.model_copy(
                update={
                    "tool_call_valid": not bool(payload.get("error")),
                    "tool_call_count": len(raw_action_trace),
                }
            ),
            provenance={
                "audit_payload_ref": payload_ref,
                "critic_protocol_refs": protocol_refs,
                "scene_state_path": scene_state_path,
                "experiment_signature": self.experiment_signature,
                "config_hash": self.config_hash,
                "model_id": self.model_id,
                "tool_schema_hash": _stable_hash(tools, 64) if tools else "",
                "image_hashes": [item.get("sha256", "") for item in image_refs],
                "tool_media_refs": list(unique_embedded_media.values()),
            },
            metadata={
                "task_spec": self.task_spec.model_dump(mode="json"),
                "capture_policy": (
                    "only_last_designer_call_receives_stage_outcome"
                    if role == "designer"
                    else "critic_advice_requires_downstream_causal_verification"
                ),
                "downstream_designer_payload_ref": downstream_ref,
                "final_scene_context": bounded_final_context,
                "raw_agent_items_sha256": raw_items_hash,
                "raw_provider_responses_sha256": raw_responses_hash,
                "raw_agent_item_count": (
                    len(raw_items) if isinstance(raw_items, list) else 0
                ),
                "raw_provider_response_count": (
                    len(raw_responses) if isinstance(raw_responses, list) else 0
                ),
                "duplicate_raw_sdk_payload_omitted": True,
                "audit_metadata_complete": bool(final_context_complete),
                "raw_output_complete": output_complete,
            },
        )
        return self._append(record)

    def capture_stage(
        self,
        *,
        stage: str,
        verify_report: StageVerifyReport | None,
        repair_actions: list[RepairResult] | None = None,
        final_scene_context: dict[str, Any] | None = None,
        scene_state_path: str = "",
    ) -> dict[str, int]:
        """Capture newly persisted designer, critic, and repair observations."""

        added = {"designer": 0, "critic": 0, "repair": 0, "unlabeled": 0}
        payloads = self._llm_payloads(stage)
        repair_events = [
            event
            for event in _load_jsonl(
                self.scene_debug_dir / "timing" / "repair_events.jsonl"
            )
            if event.get("stage") == stage
        ]
        if not payloads and not repair_events and not repair_actions:
            self._write_manifest()
            return added
        designer_indices = [
            index
            for index, (_, payload) in enumerate(payloads)
            if payload.get("agent_role") == "designer"
        ]
        final_designer_index = designer_indices[-1] if designer_indices else -1
        report_ref = ""
        if verify_report is not None:
            report_payload = verify_report.model_dump(mode="json")
            evidence_path = (
                self.output_dir
                / "evidence"
                / f"{stage}_{_stable_hash(report_payload, 16)}.json"
            )
            evidence_path.parent.mkdir(parents=True, exist_ok=True)
            evidence_path.write_text(
                json.dumps(report_payload, indent=2, ensure_ascii=False),
                encoding="utf-8",
                newline="\n",
            )
            report_ref = self._relative_ref(evidence_path)
        stage_evidence = (
            self._stage_evidence(stage, verify_report, report_ref)
            if verify_report is not None
            else None
        )
        stage_outcome = _outcome_from_report(verify_report, evidence_ref=report_ref)

        for index, (path, payload) in enumerate(payloads):
            role = str(payload.get("agent_role") or "")
            downstream_ref = ""
            if role == "critic":
                for next_path, next_payload in payloads[index + 1 :]:
                    if (
                        next_payload.get("agent_role") == "designer"
                        and next_payload.get("event") == "request_design_change"
                    ):
                        downstream_ref = self._relative_ref(next_path)
                        break
                evidence = PreferenceEvidence(
                    evidence_id="evidence_"
                    + _stable_hash(
                        [self.run_id, self.scene_id, str(path), "critic_unlabeled"]
                    ),
                    kind="none",
                    verdict="unlabeled",
                    source="critic_advice_requires_downstream_causal_verification",
                    authoritative=False,
                    report_ref=report_ref,
                    details={
                        "candidate_scene_report_ref": report_ref,
                        "downstream_designer_payload_ref": downstream_ref,
                        "causal_link_verified": False,
                    },
                )
                outcome = stage_outcome.model_copy(
                    update={"causal_link_verified": False}
                )
            elif index == final_designer_index and stage_evidence is not None:
                evidence = stage_evidence.model_copy(deep=True)
                outcome = stage_outcome
            elif role == "designer":
                next_designer_ref = ""
                for next_path, next_payload in payloads[index + 1 :]:
                    if next_payload.get("agent_role") == "designer":
                        if next_payload.get("event") == "request_design_change":
                            next_designer_ref = self._relative_ref(next_path)
                        break
                if next_designer_ref:
                    evidence = PreferenceEvidence(
                        evidence_id="evidence_"
                        + _stable_hash(
                            [self.run_id, self.scene_id, str(path), "revised"]
                        ),
                        kind="critic",
                        verdict="rejected",
                        source="main_revision_request",
                        authoritative=True,
                        quality_score=0.0,
                        report_ref=next_designer_ref,
                        details={
                            "downstream_designer_payload_ref": next_designer_ref,
                            "candidate_was_revised_by_main": True,
                        },
                    )
                    outcome = TrajectoryOutcome(
                        execution_complete=True,
                        causal_link_verified=True,
                        evidence_refs=[next_designer_ref],
                    )
                    downstream_ref = next_designer_ref
                else:
                    evidence = PreferenceEvidence(
                        evidence_id="evidence_"
                        + _stable_hash(
                            [self.run_id, self.scene_id, str(path), "unlabeled"]
                        ),
                        source="insufficient_candidate_level_evidence",
                        verdict="unlabeled",
                        kind="none",
                        authoritative=False,
                        report_ref=report_ref if verify_report is not None else "",
                    )
                    outcome = TrajectoryOutcome()
            else:
                evidence = PreferenceEvidence(
                    evidence_id="evidence_"
                    + _stable_hash(
                        [self.run_id, self.scene_id, str(path), "unlabeled"]
                    ),
                    source="insufficient_candidate_level_evidence",
                    verdict="unlabeled",
                    kind="none",
                    authoritative=False,
                    report_ref=report_ref if verify_report is not None else "",
                )
                outcome = TrajectoryOutcome()
            if self._capture_llm_record(
                path=path,
                payload=payload,
                evidence=evidence,
                outcome=outcome,
                report_ref=report_ref,
                downstream_ref=downstream_ref,
                final_scene_context=final_scene_context,
                scene_state_path=scene_state_path,
            ):
                added[role] += 1
                if evidence.verdict == "unlabeled":
                    added["unlabeled"] += 1

        for event in repair_events:
            self._capture_repair_event(event, added)

        for action in repair_actions or []:
            self._capture_scene_expert_repair(stage, action, added)

        self._write_manifest()
        return added

    def _capture_repair_event(
        self, event: dict[str, Any], added: dict[str, int]
    ) -> None:
        detail = event.get("detail") if isinstance(event.get("detail"), dict) else {}
        resolved = detail.get("resolved")
        status = str(event.get("status") or "")
        verdict = (
            "accepted" if status == "accepted" and resolved is True else "rejected"
        )
        prompt_payload = {
            "stage": event.get("stage"),
            "source": event.get("source"),
            "strategy": event.get("strategy"),
            "trigger_reasons": event.get("trigger_reasons") or [],
        }
        response_payload = {
            "actions": event.get("actions") or [],
            "affected_objects": event.get("affected_objects") or [],
        }
        prompt, prompt_complete = _bounded_text(prompt_payload, self.max_prompt_chars)
        response, response_complete = _bounded_text(
            response_payload, self.max_response_chars
        )
        event_signature = _stable_hash(event, 64)
        evidence = PreferenceEvidence(
            evidence_id="evidence_" + _stable_hash([event_signature, verdict]),
            kind="deterministic",
            verdict=verdict,
            source=str(event.get("repair_owner") or "scenesmith_core"),
            authoritative=True,
            quality_score=(1.0 if verdict == "accepted" else 0.0),
            report_ref="timing/repair_events.jsonl",
            details={"status": status, "resolved": resolved, "detail": detail},
        )
        completion = [{"role": "assistant", "content": response}]
        record = TrajectoryRecord(
            trajectory_id="trajectory_"
            + _stable_hash([self.run_id, self.scene_id, event_signature]),
            created_at=str(event.get("created_at") or _utc_now()),
            run_id=self.run_id,
            scene_id=self.scene_id,
            task_id=self.task_id,
            scenario_family_id=self.task_id,
            experiment_signature=self.experiment_signature,
            config_hash=self.config_hash,
            model_id=self.model_id,
            stage=str(event.get("stage") or ""),
            agent_role="repair",
            event=str(event.get("strategy") or "deterministic_repair"),
            task_type="deterministic_repair",
            context_hash=_stable_hash(prompt_payload, 64),
            prompt=prompt,
            response=response,
            response_hash=_stable_hash(completion, 64),
            prompt_complete=prompt_complete,
            response_complete=response_complete,
            evidence=evidence,
            source_refs=["timing/repair_events.jsonl"],
            messages=[{"role": "user", "content": prompt}],
            completion_messages=completion,
            action_trace=[
                {"sequence_index": index, "deterministic_action": action}
                for index, action in enumerate(event.get("actions") or [])
            ],
            outcome=TrajectoryOutcome(
                execution_complete=status in {"accepted", "rejected", "completed"},
                tool_call_valid=True,
                hard_passed=resolved if isinstance(resolved, bool) else None,
                hard_violation_count=0 if resolved is True else None,
                tool_call_count=len(event.get("actions") or []),
                causal_link_verified=resolved is not None,
            ),
            metadata={"repair_owner": event.get("repair_owner")},
        )
        if self._append(record):
            added["repair"] += 1

    def _capture_scene_expert_repair(
        self, stage: str, action: RepairResult, added: dict[str, int]
    ) -> None:
        prompt_payload = {
            "stage": stage,
            "repair_type": action.repair_type,
            "failure_type": action.failure_type,
        }
        response_payload = {
            "repair_action": action.repair_action,
            "execution_status": action.execution_status,
        }
        prompt, prompt_complete = _bounded_text(prompt_payload, self.max_prompt_chars)
        response, response_complete = _bounded_text(
            response_payload, self.max_response_chars
        )
        if not response.strip() or response in {
            "{}",
            '{"repair_action": "", "execution_status": ""}',
        }:
            return
        verdict = "accepted" if action.repair_verified else "rejected"
        payload = action.model_dump(mode="json")
        evidence_path = (
            self.output_dir
            / "evidence"
            / f"{stage}_scene_expert_repair_{_stable_hash(payload, 16)}.json"
        )
        evidence_path.parent.mkdir(parents=True, exist_ok=True)
        evidence_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
            newline="\n",
        )
        evidence_ref = self._relative_ref(evidence_path)
        evidence = PreferenceEvidence(
            evidence_id="evidence_" + _stable_hash([payload, verdict]),
            kind="deterministic",
            verdict=verdict,
            source=action.repair_owner,
            authoritative=bool(action.execution_status != "planned"),
            quality_score=(1.0 if action.repair_verified else 0.0),
            report_ref=evidence_ref,
            details=payload,
        )
        completion = [{"role": "assistant", "content": response}]
        record = TrajectoryRecord(
            trajectory_id="trajectory_"
            + _stable_hash([self.run_id, self.scene_id, stage, payload]),
            created_at=_utc_now(),
            run_id=self.run_id,
            scene_id=self.scene_id,
            task_id=self.task_id,
            scenario_family_id=self.task_id,
            experiment_signature=self.experiment_signature,
            config_hash=self.config_hash,
            model_id=self.model_id,
            stage=stage,
            agent_role="repair",
            event=action.repair_type,
            task_type="deterministic_repair",
            context_hash=_stable_hash(prompt_payload, 64),
            prompt=prompt,
            response=response,
            response_hash=_stable_hash(completion, 64),
            prompt_complete=prompt_complete,
            response_complete=response_complete,
            evidence=evidence,
            source_refs=[evidence_ref],
            messages=[{"role": "user", "content": prompt}],
            completion_messages=completion,
            outcome=TrajectoryOutcome(
                execution_complete=action.execution_status != "planned",
                tool_call_valid=action.execution_status != "planned",
                hard_passed=action.repair_verified,
                hard_violation_count=0 if action.repair_verified else None,
                causal_link_verified=action.execution_status != "planned",
                evidence_refs=[evidence_ref],
            ),
        )
        if self._append(record):
            added["repair"] += 1

    def _write_manifest(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        raw_records = list(_load_jsonl(self.trajectory_path))
        records: list[TrajectoryRecord] = []
        invalid_record_count = 0
        for raw_record in raw_records:
            try:
                records.append(TrajectoryRecord.model_validate(raw_record))
            except (TypeError, ValueError):
                invalid_record_count += 1
        task_type_counts: dict[str, int] = {}
        role_counts: dict[str, int] = {}
        complete_count = 0
        authoritative_count = 0
        default_training_candidate_count = 0
        incomplete_designer_count = 0
        missing_media_count = 0
        external_media_path_count = 0
        media_hash_mismatch_count = 0
        for record in records:
            task_type_counts[record.task_type] = (
                task_type_counts.get(record.task_type, 0) + 1
            )
            role_counts[record.agent_role] = role_counts.get(record.agent_role, 0) + 1
            complete_count += int(record.prompt_complete and record.response_complete)
            authoritative_count += int(record.evidence.authoritative)
            default_training_candidate_count += int(
                record.task_type in {"designer_initial", "designer_repair"}
                and record.prompt_complete
                and record.response_complete
                and record.evidence.authoritative
                and record.evidence.verdict in {"accepted", "rejected"}
            )
            if record.task_type in {"designer_initial", "designer_repair"} and not (
                record.prompt_complete and record.response_complete
            ):
                incomplete_designer_count += 1
            media_references = [
                *record.image_refs,
                *[
                    reference
                    for reference in (record.provenance.get("tool_media_refs") or [])
                    if isinstance(reference, dict)
                ],
            ]
            for reference in media_references:
                relative_path = Path(str(reference.get("path") or ""))
                path_value = str(reference.get("path") or "").strip()
                if relative_path.is_absolute():
                    external_media_path_count += 1
                    media_path = relative_path
                else:
                    media_path = self.output_dir / relative_path
                if not path_value or not media_path.is_file():
                    missing_media_count += 1
                    continue
                expected_hash = str(reference.get("sha256") or "")
                if expected_hash:
                    try:
                        observed_hash = hashlib.sha256(
                            media_path.read_bytes()
                        ).hexdigest()
                    except OSError:
                        missing_media_count += 1
                    else:
                        media_hash_mismatch_count += int(observed_hash != expected_hash)
        media_count = len(list((self.output_dir / "media").glob("*")))
        serialized_rows = "\n".join(
            json.dumps(record, ensure_ascii=False, default=str)
            for record in raw_records
        )
        embedded_data_uri_count = serialized_rows.count("data:image/")
        default_training_payload_valid = not any(
            (
                invalid_record_count,
                incomplete_designer_count,
                missing_media_count,
                external_media_path_count,
                media_hash_mismatch_count,
                embedded_data_uri_count,
                self._invalid_media_count,
            )
        )
        payload = {
            "schema_version": "sceneexpert.trajectory_manifest.v3",
            "updated_at": _utc_now(),
            "run_id": self.run_id,
            "scene_id": self.scene_id,
            "task_id": self.task_id,
            "experiment_signature": self.experiment_signature,
            "config_hash": self.config_hash,
            "model_id": self.model_id,
            "resolved_mode": self.capture_mode,
            "component_flags": self.component_flags,
            "code_provenance": self.code_provenance,
            "trajectory_path": self.trajectory_path.name,
            "record_count": len(self._seen_ids),
            "records_added_this_process": self._counts,
            "capture_is_observer_only": True,
            "quality_summary": {
                "complete_record_count": complete_count,
                "incomplete_record_count": len(records) - complete_count,
                "invalid_record_count": invalid_record_count,
                "incomplete_designer_record_count": incomplete_designer_count,
                "missing_materialized_media_count": missing_media_count,
                "external_media_path_count": external_media_path_count,
                "media_hash_mismatch_count": media_hash_mismatch_count,
                "invalid_embedded_media_count": self._invalid_media_count,
                "embedded_data_uri_count": embedded_data_uri_count,
                "authoritative_record_count": authoritative_count,
                "default_training_candidate_count": default_training_candidate_count,
                "has_default_training_candidates": bool(
                    default_training_candidate_count
                ),
                "role_counts": role_counts,
                "task_type_counts": task_type_counts,
                "materialized_media_count": media_count,
                "default_trainable_task_types": [
                    "designer_initial",
                    "designer_repair",
                ],
                "critic_records_are_audit_only": True,
                "default_training_payload_valid": default_training_payload_valid,
            },
            "captures": {
                "exact_messages": True,
                "tool_calls_and_results": True,
                "spatial_context": True,
                "critic_advice_audit_only": True,
                "media_hashes": True,
                "outcome_is_label_only": True,
            },
            "pairing_policy": (
                "offline exporter requires exact decision context plus independent "
                "authoritative accepted/rejected evidence; default training is limited "
                "to designer_initial and designer_repair while critic advice remains audit-only"
            ),
        }
        self.manifest_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
            newline="\n",
        )
