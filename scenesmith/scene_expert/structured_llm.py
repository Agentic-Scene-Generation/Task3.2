"""Reliable structured calls to the local Qwen/vLLM endpoint.

SceneExpert's direct LLM roles need a stricter contract than tool-using agents:
the final assistant ``content`` must contain schema-valid JSON.  This module
centralizes that contract, explicit retries, Qwen thinking controls, and debug
records without changing the public outputs of TaskCompiler/GlobalPlanner.
"""

from __future__ import annotations

import copy
import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Generic, Mapping, TypeVar

from pydantic import BaseModel, Field, ValidationError

from scenesmith.agent_utils.thinking import prepend_text_thinking_directive
from scenesmith.scene_expert.context_bundle import (
    LLMCallDebugRecord,
    compact_text,
    stable_hash,
    utc_now,
)

console_logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


@dataclass(frozen=True)
class StructuredLLMProfile:
    """Per-role request and recovery limits."""

    thinking_mode: str = "none"
    max_tokens: int = 1024
    retry_max_tokens: int | None = None
    timeout_seconds: float = 60.0
    temperature: float = 0.1
    max_attempts: int = 2
    response_format: str = "json_schema"

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any] | None,
        default: "StructuredLLMProfile | None" = None,
    ) -> "StructuredLLMProfile":
        base = default or cls()
        data = dict(value or {})
        retry_tokens = data.get("retry_max_tokens", base.retry_max_tokens)
        return cls(
            thinking_mode=str(data.get("thinking_mode", base.thinking_mode)),
            max_tokens=int(data.get("max_tokens", base.max_tokens)),
            retry_max_tokens=(
                int(retry_tokens) if retry_tokens not in (None, "") else None
            ),
            timeout_seconds=float(
                data.get("timeout_seconds", base.timeout_seconds)
            ),
            temperature=float(data.get("temperature", base.temperature)),
            max_attempts=max(1, int(data.get("max_attempts", base.max_attempts))),
            response_format=str(
                data.get("response_format", base.response_format)
            ),
        )

    @property
    def thinking_enabled(self) -> bool:
        return self.thinking_mode.strip().lower() not in {
            "",
            "none",
            "minimal",
            "off",
            "false",
            "0",
            "no_think",
            "nothink",
        }


class StructuredLLMAttempt(BaseModel):
    attempt: int
    status: str
    error_kind: str = ""
    error: str = ""
    retry_strategy: str = ""
    thinking_mode: str = "none"
    response_format: str = ""
    timeout_seconds: float = 0.0
    max_tokens: int = 0
    elapsed_sec: float = 0.0
    request_id: str = ""
    finish_reason: str = ""
    prompt_chars: int = 0
    output_chars: int = 0
    reasoning_chars: int = 0
    token_usage: dict[str, int] = Field(default_factory=dict)


@dataclass
class StructuredLLMResult(Generic[T]):
    """Outcome returned to a role wrapper before role-specific fallback."""

    value: T | None = None
    content: str = ""
    reasoning_content: str = ""
    attempts: list[StructuredLLMAttempt] = field(default_factory=list)
    final_error_kind: str = ""
    final_error: str = ""

    @property
    def success(self) -> bool:
        return self.value is not None

    @property
    def source(self) -> str:
        return "llm" if self.success else "fallback"

    def status_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "source": self.source,
            "degraded": not self.success,
            "attempt_count": len(self.attempts),
            "final_error_kind": self.final_error_kind,
            "final_error": self.final_error,
            "attempts": [attempt.model_dump() for attempt in self.attempts],
        }


class _StructuredFailure(Exception):
    def __init__(
        self,
        kind: str,
        message: str,
        *,
        content: str = "",
        reasoning: str = "",
        response: Any = None,
    ) -> None:
        super().__init__(message)
        self.kind = kind
        self.content = content
        self.reasoning = reasoning
        self.response = response


class SceneExpertStructuredLLMClient:
    """One structured-output client shared by all SceneExpert direct roles."""

    def __init__(
        self,
        *,
        model: str,
        api_base_url: str | None = None,
        api_key: str | None = None,
        profiles: Mapping[str, StructuredLLMProfile | Mapping[str, Any]] | None = None,
        client: Any = None,
        debug_path: str | Path | None = None,
    ) -> None:
        if client is None:
            from openai import OpenAI

            client = OpenAI(
                base_url=api_base_url
                or os.environ.get("OPENAI_BASE_URL", "http://localhost:8000/v1"),
                api_key=api_key or os.environ.get("OPENAI_API_KEY", "dummy"),
            )
        self._client = client
        self.model = model
        self._debug_path = Path(debug_path) if debug_path else None
        self._profiles: dict[str, StructuredLLMProfile] = {}
        for role, profile in (profiles or {}).items():
            self._profiles[str(role)] = (
                profile
                if isinstance(profile, StructuredLLMProfile)
                else StructuredLLMProfile.from_mapping(profile)
            )

    def profile_for(
        self,
        role: str,
        default: StructuredLLMProfile | None = None,
    ) -> StructuredLLMProfile:
        return self._profiles.get(role, default or StructuredLLMProfile())

    def complete(
        self,
        *,
        role: str,
        stage: str,
        event: str,
        messages: list[dict[str, Any]],
        response_model: type[T],
        profile: StructuredLLMProfile | None = None,
    ) -> StructuredLLMResult[T]:
        """Return validated content or a typed failure for role-level fallback."""
        active_profile = profile or self.profile_for(role)
        attempts: list[StructuredLLMAttempt] = []
        base_messages = copy.deepcopy(messages)
        retry_messages = copy.deepcopy(base_messages)
        previous_kind = ""
        previous_error = ""
        previous_content = ""
        last_reasoning = ""

        for attempt_number in range(1, active_profile.max_attempts + 1):
            thinking_enabled = active_profile.thinking_enabled
            response_format = active_profile.response_format
            max_tokens = active_profile.max_tokens
            retry_strategy = "initial"

            if attempt_number > 1:
                retry_strategy = self._retry_strategy(previous_kind)
                if previous_kind in {
                    "length",
                    "reasoning_only",
                    "empty_response",
                    "invalid_json",
                    "schema_validation",
                }:
                    thinking_enabled = False
                if previous_kind == "bad_request" and response_format == "json_schema":
                    response_format = "json_object"
                if active_profile.retry_max_tokens is not None:
                    max_tokens = active_profile.retry_max_tokens
                retry_messages = self._build_retry_messages(
                    base_messages,
                    previous_kind,
                    previous_error,
                    previous_content,
                )

            thinking_mode = "think" if thinking_enabled else "none"
            request_messages = self._apply_thinking_mode(
                retry_messages,
                thinking_enabled=thinking_enabled,
            )
            prompt_text = json.dumps(
                request_messages, ensure_ascii=False, default=str
            )
            started = time.perf_counter()
            response = None
            content = ""
            reasoning = ""
            try:
                response = self._request(
                    messages=request_messages,
                    response_model=response_model,
                    profile=active_profile,
                    response_format=response_format,
                    max_tokens=max_tokens,
                    thinking_enabled=thinking_enabled,
                )
                content, reasoning = self._response_text(response)
                last_reasoning = reasoning
                finish_reason = self._finish_reason(response)
                if finish_reason == "length":
                    raise _StructuredFailure(
                        "length",
                        "Model exhausted max_tokens before completing structured output",
                        content=content,
                        reasoning=reasoning,
                        response=response,
                    )
                if not content:
                    kind = "reasoning_only" if reasoning else "empty_response"
                    raise _StructuredFailure(
                        kind,
                        "Assistant content is empty",
                        content=content,
                        reasoning=reasoning,
                        response=response,
                    )
                try:
                    payload = extract_json_object(content)
                except Exception as exc:
                    raise _StructuredFailure(
                        "invalid_json",
                        f"Invalid JSON: {exc}",
                        content=content,
                        reasoning=reasoning,
                        response=response,
                    ) from exc
                try:
                    value = response_model.model_validate(payload)
                except ValidationError as exc:
                    raise _StructuredFailure(
                        "schema_validation",
                        compact_text(exc, 1200),
                        content=content,
                        reasoning=reasoning,
                        response=response,
                    ) from exc

                elapsed = time.perf_counter() - started
                attempt = self._attempt_record(
                    attempt=attempt_number,
                    status="success",
                    retry_strategy=retry_strategy,
                    thinking_mode=thinking_mode,
                    response_format=response_format,
                    timeout_seconds=active_profile.timeout_seconds,
                    max_tokens=max_tokens,
                    elapsed_sec=elapsed,
                    prompt_chars=len(prompt_text),
                    content=content,
                    reasoning=reasoning,
                    response=response,
                )
                attempts.append(attempt)
                self._append_debug(
                    stage=stage,
                    role=role,
                    event=event,
                    prompt_text=prompt_text,
                    content=content,
                    reasoning=reasoning,
                    response=response,
                    attempt=attempt,
                )
                return StructuredLLMResult(
                    value=value,
                    content=content,
                    reasoning_content=reasoning,
                    attempts=attempts,
                )
            except Exception as exc:
                failure = self._normalize_failure(exc, response=response)
                content = failure.content or content
                reasoning = failure.reasoning or reasoning
                last_reasoning = reasoning or last_reasoning
                elapsed = time.perf_counter() - started
                attempt = self._attempt_record(
                    attempt=attempt_number,
                    status="failed",
                    error_kind=failure.kind,
                    error=str(failure),
                    retry_strategy=retry_strategy,
                    thinking_mode=thinking_mode,
                    response_format=response_format,
                    timeout_seconds=active_profile.timeout_seconds,
                    max_tokens=max_tokens,
                    elapsed_sec=elapsed,
                    prompt_chars=len(prompt_text),
                    content=content,
                    reasoning=reasoning,
                    response=failure.response or response,
                )
                attempts.append(attempt)
                self._append_debug(
                    stage=stage,
                    role=role,
                    event=event,
                    prompt_text=prompt_text,
                    content=content,
                    reasoning=reasoning,
                    response=failure.response or response,
                    attempt=attempt,
                )
                previous_kind = failure.kind
                previous_error = str(failure)
                previous_content = content
                if attempt_number >= active_profile.max_attempts:
                    break

        return StructuredLLMResult(
            content=previous_content,
            reasoning_content=last_reasoning,
            attempts=attempts,
            final_error_kind=previous_kind,
            final_error=previous_error,
        )

    def _request(
        self,
        *,
        messages: list[dict[str, Any]],
        response_model: type[BaseModel],
        profile: StructuredLLMProfile,
        response_format: str,
        max_tokens: int,
        thinking_enabled: bool,
    ) -> Any:
        client = self._client
        if hasattr(client, "with_options"):
            client = client.with_options(
                timeout=profile.timeout_seconds,
                max_retries=0,
            )
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": profile.temperature,
            "max_tokens": max_tokens,
            "extra_body": {
                "chat_template_kwargs": {"enable_thinking": thinking_enabled}
            },
        }
        if response_format == "json_schema":
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": response_model.__name__,
                    "schema": response_model.model_json_schema(),
                    "strict": True,
                },
            }
        elif response_format == "json_object":
            kwargs["response_format"] = {"type": "json_object"}
        return client.chat.completions.create(**kwargs)

    @staticmethod
    def _apply_thinking_mode(
        messages: list[dict[str, Any]], *, thinking_enabled: bool
    ) -> list[dict[str, Any]]:
        updated = copy.deepcopy(messages)
        directive = "/think" if thinking_enabled else "/no_think"
        for message in reversed(updated):
            if message.get("role") != "user":
                continue
            content = message.get("content")
            if isinstance(content, str):
                message["content"] = prepend_text_thinking_directive(
                    content, directive
                )
                break
            if isinstance(content, list):
                for item in content:
                    if not isinstance(item, dict):
                        continue
                    if item.get("type") in {"text", "input_text"} and isinstance(
                        item.get("text"), str
                    ):
                        item["text"] = prepend_text_thinking_directive(
                            item["text"], directive
                        )
                        return updated
        return updated

    @staticmethod
    def _build_retry_messages(
        messages: list[dict[str, Any]],
        error_kind: str,
        error: str,
        previous_content: str,
    ) -> list[dict[str, Any]]:
        updated = copy.deepcopy(messages)
        feedback = (
            "The previous structured response failed validation. "
            f"Failure type: {error_kind}. Return the final JSON object immediately "
            "with no analysis, markdown, comments, or tool calls."
        )
        if error_kind in {"invalid_json", "schema_validation"} and previous_content:
            feedback += " Previous output excerpt: " + compact_text(
                previous_content, 1000
            )
        if error:
            feedback += " Validation detail: " + compact_text(error, 500)
        updated.append({"role": "user", "content": feedback})
        return updated

    @staticmethod
    def _response_text(response: Any) -> tuple[str, str]:
        choices = getattr(response, "choices", None) or []
        if not choices:
            return "", ""
        message = getattr(choices[0], "message", None)
        if message is None:
            return "", ""
        content = _stringify_content(getattr(message, "content", None))
        reasoning = _stringify_content(
            getattr(message, "reasoning_content", None)
            or getattr(message, "reasoning", None)
        )
        if not reasoning:
            extra = getattr(message, "model_extra", None)
            if isinstance(extra, dict):
                reasoning = _stringify_content(
                    extra.get("reasoning_content") or extra.get("reasoning")
                )
        return content, reasoning

    @staticmethod
    def _finish_reason(response: Any) -> str:
        choices = getattr(response, "choices", None) or []
        return str(getattr(choices[0], "finish_reason", "") or "") if choices else ""

    @staticmethod
    def _normalize_failure(exc: Exception, *, response: Any = None) -> _StructuredFailure:
        if isinstance(exc, _StructuredFailure):
            return exc
        name = type(exc).__name__
        message = str(exc)
        lowered = message.lower()
        if name in {"APITimeoutError", "TimeoutException", "ReadTimeout"} or (
            "timed out" in lowered or "timeout" in lowered
        ):
            kind = "timeout"
        elif name in {"APIConnectionError", "ConnectError", "ConnectionError"}:
            kind = "transport"
        elif name in {"BadRequestError", "UnprocessableEntityError"}:
            kind = "bad_request"
        elif name in {"RateLimitError"}:
            kind = "rate_limit"
        elif name in {"InternalServerError", "APIStatusError"}:
            kind = "server_error"
        else:
            kind = "unexpected"
        return _StructuredFailure(kind, f"{name}: {message}", response=response)

    @staticmethod
    def _retry_strategy(error_kind: str) -> str:
        return {
            "length": "force_no_think_and_retry",
            "reasoning_only": "force_no_think_and_retry",
            "empty_response": "force_no_think_and_retry",
            "invalid_json": "json_validation_feedback",
            "schema_validation": "schema_validation_feedback",
            "bad_request": "downgrade_response_format",
            "timeout": "single_transport_retry",
            "transport": "single_transport_retry",
            "rate_limit": "single_transport_retry",
            "server_error": "single_transport_retry",
        }.get(error_kind, "single_bounded_retry")

    def _attempt_record(
        self,
        *,
        attempt: int,
        status: str,
        retry_strategy: str,
        thinking_mode: str,
        response_format: str,
        timeout_seconds: float,
        max_tokens: int,
        elapsed_sec: float,
        prompt_chars: int,
        content: str,
        reasoning: str,
        response: Any,
        error_kind: str = "",
        error: str = "",
    ) -> StructuredLLMAttempt:
        return StructuredLLMAttempt(
            attempt=attempt,
            status=status,
            error_kind=error_kind,
            error=error,
            retry_strategy=retry_strategy,
            thinking_mode=thinking_mode,
            response_format=response_format,
            timeout_seconds=timeout_seconds,
            max_tokens=max_tokens,
            elapsed_sec=round(elapsed_sec, 4),
            request_id=str(getattr(response, "id", "") or ""),
            finish_reason=self._finish_reason(response),
            prompt_chars=prompt_chars,
            output_chars=len(content),
            reasoning_chars=len(reasoning),
            token_usage=_raw_token_usage(response),
        )

    def _append_debug(
        self,
        *,
        stage: str,
        role: str,
        event: str,
        prompt_text: str,
        content: str,
        reasoning: str,
        response: Any,
        attempt: StructuredLLMAttempt,
    ) -> None:
        path_value = self._debug_path or os.environ.get(
            "SCENEEXPERT_LLM_DEBUG_PATH", ""
        )
        if not path_value:
            return
        try:
            path = Path(path_value)
            path.parent.mkdir(parents=True, exist_ok=True)
            record = LLMCallDebugRecord(
                schema_version="1.1",
                created_at=utc_now(),
                stage=stage,
                agent_role=role,
                event=event,
                prompt_chars=len(prompt_text),
                prompt_hash=stable_hash(prompt_text),
                prompt_excerpt=compact_text(prompt_text, 1800),
                output_chars=len(content),
                output_excerpt=compact_text(content, 1800),
                finish_reasons=(
                    [attempt.finish_reason] if attempt.finish_reason else []
                ),
                token_usage=attempt.token_usage,
                raw_response_excerpt=compact_text(response, 1200),
                error=attempt.error,
                request_id=attempt.request_id,
                attempt=attempt.attempt,
                status=attempt.status,
                error_kind=attempt.error_kind,
                retry_strategy=attempt.retry_strategy,
                thinking_mode=attempt.thinking_mode,
                response_format=attempt.response_format,
                elapsed_sec=attempt.elapsed_sec,
                timeout_sec=attempt.timeout_seconds,
                reasoning_chars=len(reasoning),
                queue_wait_sec=None,
                ttft_sec=None,
                decode_sec=None,
            )
            with path.open("a", encoding="utf-8", newline="\n") as file:
                file.write(json.dumps(record.model_dump(), ensure_ascii=False) + "\n")
        except Exception as exc:
            console_logger.warning("Failed to write structured LLM debug record: %s", exc)


def extract_json_object(text: str) -> dict[str, Any]:
    """Extract one JSON object without accepting reasoning prose as output."""
    if not text or not text.strip():
        raise ValueError("Empty response text")
    stripped = text.strip()
    fence = re.search(r"```(?:json)?\s*([\s\S]+?)\s*```", stripped)
    if fence:
        stripped = fence.group(1).strip()
    try:
        value = json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        if start < 0:
            raise
        decoder = json.JSONDecoder()
        value, _ = decoder.raw_decode(stripped[start:])
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object, got {type(value).__name__}")
    return value


def _stringify_content(value: Any) -> str:
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
        return "\n".join(parts).strip()
    return str(value).strip()


def _raw_token_usage(response: Any) -> dict[str, int]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return {}
    values = {
        "input_tokens": getattr(usage, "prompt_tokens", None)
        or getattr(usage, "input_tokens", None),
        "output_tokens": getattr(usage, "completion_tokens", None)
        or getattr(usage, "output_tokens", None),
        "total_tokens": getattr(usage, "total_tokens", None),
    }
    details = getattr(usage, "completion_tokens_details", None) or getattr(
        usage, "output_tokens_details", None
    )
    if details is not None:
        values["reasoning_tokens"] = getattr(details, "reasoning_tokens", None)
    return {key: int(value) for key, value in values.items() if isinstance(value, int)}
