import asyncio
import base64
import contextvars
import json
import logging
import re
import sqlite3
import threading

from contextlib import asynccontextmanager
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, AsyncIterator

import numpy as np

from openai import AsyncOpenAI, OpenAI
from PIL import Image

console_logger = logging.getLogger(__name__)

_VALID_PERSISTENCE_PROVIDERS = {
    "qwen",
    "openrouter",
    "openai",
    "auto",
    "disabled",
}
_reasoning_persistence_enabled = False
_reasoning_persistence_provider = "disabled"


def _resolve_persistence_provider(
    provider: str,
    *,
    model_id: str | None,
    base_url: str | None,
) -> str:
    """Resolve the response format used for passive reasoning persistence."""
    normalized = str(provider or "disabled").strip().lower()
    if normalized not in _VALID_PERSISTENCE_PROVIDERS:
        console_logger.warning(
            "Unknown reasoning persistence provider %r; disabling persistence",
            provider,
        )
        return "disabled"
    if normalized != "auto":
        return normalized

    url = str(base_url or "").strip().lower()
    model = str(model_id or "").strip().lower()
    if "openrouter.ai" in url:
        return "openrouter"
    if "api.openai.com" in url:
        return "openai"
    if "qwen" in model:
        return "qwen"
    model_leaf = model.rsplit("/", 1)[-1]
    if model_leaf.startswith("gpt-") or re.match(r"^o\d", model_leaf):
        return "openai"

    console_logger.warning(
        "Could not unambiguously infer reasoning persistence provider "
        "(model=%r, base_url=%r); disabling persistence",
        model_id,
        base_url,
    )
    return "disabled"


def configure_reasoning_persistence(
    *,
    enabled: bool,
    provider: str,
    model_id: str | None = None,
    base_url: str | None = None,
) -> str:
    """Configure process-wide passive reasoning persistence.

    This setting only selects how already-returned responses are parsed and
    which SQLite table receives them. It never changes model request arguments.

    Returns:
        The resolved provider name.
    """
    global _reasoning_persistence_enabled
    global _reasoning_persistence_provider

    resolved = _resolve_persistence_provider(
        provider,
        model_id=model_id,
        base_url=base_url,
    )
    _reasoning_persistence_enabled = bool(enabled) and resolved != "disabled"
    _reasoning_persistence_provider = (
        resolved if _reasoning_persistence_enabled else "disabled"
    )
    console_logger.info(
        "Reasoning persistence: enabled=%s provider=%s",
        _reasoning_persistence_enabled,
        _reasoning_persistence_provider,
    )
    return _reasoning_persistence_provider


def reasoning_persistence_enabled() -> bool:
    return _reasoning_persistence_enabled


def reasoning_persistence_provider() -> str:
    return _reasoning_persistence_provider


@dataclass(frozen=True)
class ReasoningPersistenceContext:
    """Active Agent session whose response artifacts should be persisted."""

    session_id: str
    db_path: Path


@dataclass(frozen=True)
class QwenThinkingRecord:
    thinking: str
    content_preview: str | None


@dataclass(frozen=True)
class OnlineReasoningRecord:
    provider: str
    source_type: str
    model: str | None
    response_id: str | None
    summary: str | None
    raw_json: str
    content_preview: str | None
    metadata_json: str | None


_reasoning_persistence_ctx: contextvars.ContextVar[
    ReasoningPersistenceContext | None
] = contextvars.ContextVar("_reasoning_persistence_ctx", default=None)

_initialized_schema_keys: set[tuple[Path, str]] = set()
_initialized_schema_keys_lock = threading.Lock()
_SQLITE_TIMEOUT_SECONDS = 1.0

_QWEN_THINKING_SCHEMA = """
CREATE TABLE IF NOT EXISTS agent_thinking (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id      TEXT    NOT NULL,
    thinking        TEXT    NOT NULL,
    content_preview TEXT,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
"""

_ONLINE_REASONING_SCHEMA = """
CREATE TABLE IF NOT EXISTS agent_reasoning_artifacts (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id      TEXT    NOT NULL,
    provider        TEXT    NOT NULL,
    source_type     TEXT    NOT NULL,
    model           TEXT,
    response_id     TEXT,
    summary         TEXT,
    raw_json        TEXT,
    content_preview TEXT,
    metadata_json   TEXT,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
"""


def _connect_sqlite(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path), timeout=_SQLITE_TIMEOUT_SECONDS)
    conn.execute(f"PRAGMA busy_timeout={int(_SQLITE_TIMEOUT_SECONDS * 1000)}")
    return conn


def _ensure_reasoning_schema(db_path: Path, provider: str) -> None:
    """Best-effort idempotent schema creation; never raises."""
    schema_provider = "qwen" if provider == "qwen" else "online"
    key = (db_path, schema_provider)
    with _initialized_schema_keys_lock:
        if key in _initialized_schema_keys:
            return
        try:
            with _connect_sqlite(db_path) as conn:
                conn.execute("PRAGMA journal_mode=WAL")
                conn.execute(
                    _QWEN_THINKING_SCHEMA
                    if provider == "qwen"
                    else _ONLINE_REASONING_SCHEMA
                )
                conn.commit()
            _initialized_schema_keys.add(key)
        except Exception as exc:
            console_logger.warning(
                "Failed to initialize reasoning persistence schema (%s): %s",
                db_path,
                exc,
            )


@asynccontextmanager
async def reasoning_persistence_context(
    session_id: str,
    db_path: str | Path | None,
) -> AsyncIterator[None]:
    """Scope passive response persistence to one Agent session.

    ContextVar values are isolated per asyncio Task and copied into
    ``asyncio.to_thread`` calls. Missing or in-memory paths are no-ops.
    """
    provider = reasoning_persistence_provider()
    if (
        not reasoning_persistence_enabled()
        or provider == "disabled"
        or db_path is None
        or str(db_path) == ":memory:"
    ):
        yield
        return

    path = Path(db_path)
    await asyncio.to_thread(_ensure_reasoning_schema, path, provider)
    token = _reasoning_persistence_ctx.set(
        ReasoningPersistenceContext(session_id=session_id, db_path=path)
    )
    try:
        yield
    finally:
        _reasoning_persistence_ctx.reset(token)


def _read_field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def _to_jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, list):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return _to_jsonable(model_dump())
    if hasattr(value, "__dict__"):
        return {
            key: _to_jsonable(item)
            for key, item in vars(value).items()
            if not key.startswith("_")
        }
    return str(value)


def _json_dumps(value: Any) -> str:
    return json.dumps(_to_jsonable(value), ensure_ascii=False, sort_keys=True)


def _extract_readable_text(value: Any) -> str | None:
    """Best-effort extraction of readable text from reasoning detail objects."""
    jsonable = _to_jsonable(value)
    parts: list[str] = []

    def visit(item: Any) -> None:
        if isinstance(item, dict):
            for key in ("summary", "text", "content"):
                child = item.get(key)
                if isinstance(child, str) and child.strip():
                    parts.append(child.strip())
                elif isinstance(child, (dict, list)):
                    visit(child)
            for key, child in item.items():
                if key not in {"summary", "text", "content"} and isinstance(
                    child, (dict, list)
                ):
                    visit(child)
        elif isinstance(item, list):
            for child in item:
                visit(child)
        elif isinstance(item, str) and item.strip():
            parts.append(item.strip())

    visit(jsonable)
    if not parts:
        return None
    return "\n".join(dict.fromkeys(parts))


def _has_meaningful_artifact(value: Any) -> bool:
    """Return whether a provider field contains a non-empty artifact."""
    jsonable = _to_jsonable(value)
    if jsonable is None:
        return False
    if isinstance(jsonable, str):
        return bool(jsonable.strip())
    if isinstance(jsonable, (int, float, bool)):
        return True
    if isinstance(jsonable, list):
        return any(_has_meaningful_artifact(item) for item in jsonable)
    if isinstance(jsonable, dict):
        return any(_has_meaningful_artifact(item) for item in jsonable.values())
    return False


def _content_preview(content: Any, limit: int = 200) -> str | None:
    if content is None:
        return None
    if isinstance(content, str):
        return content[:limit]
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            text = _read_field(item, "text")
            if not isinstance(text, str):
                text = _read_field(item, "content")
            if isinstance(text, str):
                parts.append(text)
        if parts:
            return " ".join(parts)[:limit]
    try:
        return _json_dumps(content)[:limit]
    except Exception:
        return None


def extract_qwen_thinking(message: Any) -> str | None:
    """Extract Qwen thinking without mutating the response message."""
    for field in ("reasoning", "reasoning_content"):
        value = _read_field(message, field)
        if isinstance(value, str) and value.strip():
            return value.strip()

    content = _read_field(message, "content")
    if not isinstance(content, str) or "</think>" not in content:
        return None
    head = content.split("</think>", 1)[0]
    cleaned = re.sub(r"^\s*<think>\s*", "", head, flags=re.DOTALL).strip()
    return cleaned or None


def _extract_qwen_chat_records(response: Any) -> list[QwenThinkingRecord]:
    records: list[QwenThinkingRecord] = []
    for choice in _read_field(response, "choices", []) or []:
        message = _read_field(choice, "message")
        if message is None:
            continue
        thinking = extract_qwen_thinking(message)
        if not thinking:
            continue
        records.append(
            QwenThinkingRecord(
                thinking=thinking,
                content_preview=_content_preview(_read_field(message, "content")),
            )
        )
    return records


def _extract_online_chat_records(
    response: Any,
    provider: str,
) -> list[OnlineReasoningRecord]:
    records: list[OnlineReasoningRecord] = []
    response_id = _read_field(response, "id")
    model = _read_field(response, "model")
    for fallback_index, choice in enumerate(_read_field(response, "choices", []) or []):
        message = _read_field(choice, "message")
        if message is None:
            continue
        reasoning = _read_field(message, "reasoning")
        reasoning_details = _read_field(message, "reasoning_details")
        if not _has_meaningful_artifact(reasoning) and not _has_meaningful_artifact(
            reasoning_details
        ):
            continue

        summary = (
            reasoning.strip()
            if isinstance(reasoning, str) and reasoning.strip()
            else _extract_readable_text(reasoning_details)
        )
        raw_json = _json_dumps(
            {
                "reasoning": reasoning,
                "reasoning_details": reasoning_details,
            }
        )
        if summary is None and raw_json in ("", "null", "{}"):
            continue
        message_jsonable = _to_jsonable(message)
        metadata_json = _json_dumps(
            {
                "choice_index": _read_field(choice, "index", fallback_index),
                "finish_reason": _read_field(choice, "finish_reason"),
                "native_finish_reason": _read_field(choice, "native_finish_reason"),
                "message_keys": (
                    sorted(message_jsonable.keys())
                    if isinstance(message_jsonable, dict)
                    else []
                ),
            }
        )
        records.append(
            OnlineReasoningRecord(
                provider=provider,
                source_type=(
                    f"{provider}_reasoning_details"
                    if reasoning_details is not None
                    else f"{provider}_reasoning"
                ),
                model=model,
                response_id=response_id,
                summary=summary,
                raw_json=raw_json,
                content_preview=_content_preview(_read_field(message, "content")),
                metadata_json=metadata_json,
            )
        )
    return records


def _extract_online_responses_records(
    response: Any,
    provider: str,
) -> list[OnlineReasoningRecord]:
    records: list[OnlineReasoningRecord] = []
    response_id = _read_field(response, "id")
    model = _read_field(response, "model")
    for fallback_index, item in enumerate(_read_field(response, "output", []) or []):
        item_type = str(_read_field(item, "type", "") or "").lower()
        if "reasoning" not in item_type:
            continue
        summary_value = _read_field(item, "summary")
        if summary_value is None:
            summary_value = _read_field(item, "summary_text")
        summary = _extract_readable_text(summary_value)
        raw_json = _json_dumps(item)
        if summary is None and raw_json in ("", "null", "{}"):
            continue
        records.append(
            OnlineReasoningRecord(
                provider=provider,
                source_type=f"{provider}_responses_reasoning",
                model=model,
                response_id=response_id,
                summary=summary,
                raw_json=raw_json,
                content_preview=_content_preview(_read_field(response, "output_text")),
                metadata_json=_json_dumps(
                    {
                        "output_index": fallback_index,
                        "item_type": item_type,
                    }
                ),
            )
        )
    return records


def _extract_records(
    response: Any,
    *,
    api_kind: str,
    capture_online: bool,
) -> tuple[list[QwenThinkingRecord], list[OnlineReasoningRecord]]:
    provider = reasoning_persistence_provider()
    if provider == "qwen":
        if api_kind != "chat":
            return [], []
        return _extract_qwen_chat_records(response), []
    if provider in {"openrouter", "openai"} and capture_online:
        if api_kind == "responses":
            return [], _extract_online_responses_records(response, provider)
        return [], _extract_online_chat_records(response, provider)
    return [], []


def _write_qwen_records(
    db_path: Path,
    session_id: str,
    records: list[QwenThinkingRecord],
) -> None:
    if not records:
        return
    try:
        _ensure_reasoning_schema(db_path, "qwen")
        with _connect_sqlite(db_path) as conn:
            conn.executemany(
                "INSERT INTO agent_thinking "
                "(session_id, thinking, content_preview) VALUES (?, ?, ?)",
                [
                    (session_id, record.thinking, record.content_preview)
                    for record in records
                    if record.thinking.strip()
                ],
            )
            conn.commit()
    except Exception as exc:
        console_logger.warning("Failed to write Qwen thinking artifact: %s", exc)


def _write_online_records(
    db_path: Path,
    session_id: str,
    records: list[OnlineReasoningRecord],
) -> None:
    if not records:
        return
    try:
        _ensure_reasoning_schema(db_path, "online")
        with _connect_sqlite(db_path) as conn:
            conn.executemany(
                """
                INSERT INTO agent_reasoning_artifacts (
                    session_id, provider, source_type, model, response_id,
                    summary, raw_json, content_preview, metadata_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        session_id,
                        record.provider,
                        record.source_type,
                        record.model,
                        record.response_id,
                        record.summary,
                        record.raw_json,
                        record.content_preview,
                        record.metadata_json,
                    )
                    for record in records
                    if record.summary is not None
                    or record.raw_json not in ("", "null", "{}")
                ],
            )
            conn.commit()
    except Exception as exc:
        console_logger.warning("Failed to write online reasoning artifact: %s", exc)


def _persist_response_sync(
    response: Any,
    *,
    api_kind: str,
    session_id_override: str | None,
    capture_online: bool,
) -> None:
    """Best-effort synchronous response persistence; never raises."""
    try:
        ctx = _reasoning_persistence_ctx.get()
        if (
            ctx is None
            or not reasoning_persistence_enabled()
            or reasoning_persistence_provider() == "disabled"
        ):
            return
        qwen_records, online_records = _extract_records(
            response,
            api_kind=api_kind,
            capture_online=capture_online,
        )
        session_id = session_id_override or ctx.session_id
        if qwen_records:
            _write_qwen_records(ctx.db_path, session_id, qwen_records)
        if online_records:
            _write_online_records(ctx.db_path, session_id, online_records)
    except Exception as exc:
        console_logger.warning("Reasoning persistence hook failed: %s", exc)


async def _persist_response_async(
    response: Any,
    *,
    api_kind: str,
    session_id_override: str | None,
    capture_online: bool,
) -> None:
    """Best-effort async response persistence without blocking the event loop."""
    try:
        ctx = _reasoning_persistence_ctx.get()
        if (
            ctx is None
            or not reasoning_persistence_enabled()
            or reasoning_persistence_provider() == "disabled"
        ):
            return
        qwen_records, online_records = _extract_records(
            response,
            api_kind=api_kind,
            capture_online=capture_online,
        )
        session_id = session_id_override or ctx.session_id
        if qwen_records:
            await asyncio.to_thread(
                _write_qwen_records,
                ctx.db_path,
                session_id,
                qwen_records,
            )
        if online_records:
            await asyncio.to_thread(
                _write_online_records,
                ctx.db_path,
                session_id,
                online_records,
            )
    except Exception as exc:
        console_logger.warning("Reasoning persistence hook failed: %s", exc)


class _SyncCompletionsWrapper:
    def __init__(
        self,
        completions: Any,
        *,
        session_id_override: str | None,
        capture_online: bool,
    ):
        self._completions = completions
        self._session_id_override = session_id_override
        self._capture_online = capture_online

    def create(self, *args: Any, **kwargs: Any) -> Any:
        response = self._completions.create(*args, **kwargs)
        _persist_response_sync(
            response,
            api_kind="chat",
            session_id_override=self._session_id_override,
            capture_online=self._capture_online,
        )
        return response

    def __getattr__(self, name: str) -> Any:
        return getattr(self._completions, name)


class _AsyncCompletionsWrapper:
    def __init__(
        self,
        completions: Any,
        *,
        session_id_override: str | None,
        capture_online: bool,
    ):
        self._completions = completions
        self._session_id_override = session_id_override
        self._capture_online = capture_online

    async def create(self, *args: Any, **kwargs: Any) -> Any:
        response = await self._completions.create(*args, **kwargs)
        await _persist_response_async(
            response,
            api_kind="chat",
            session_id_override=self._session_id_override,
            capture_online=self._capture_online,
        )
        return response

    def __getattr__(self, name: str) -> Any:
        return getattr(self._completions, name)


class _ChatWrapper:
    def __init__(
        self,
        chat: Any,
        *,
        async_mode: bool,
        session_id_override: str | None,
        capture_online: bool,
    ):
        self._chat = chat
        wrapper_cls = (
            _AsyncCompletionsWrapper if async_mode else _SyncCompletionsWrapper
        )
        self.completions = wrapper_cls(
            chat.completions,
            session_id_override=session_id_override,
            capture_online=capture_online,
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self._chat, name)


class _SyncResponsesWrapper:
    def __init__(
        self,
        responses: Any,
        *,
        session_id_override: str | None,
        capture_online: bool,
    ):
        self._responses = responses
        self._session_id_override = session_id_override
        self._capture_online = capture_online

    def create(self, *args: Any, **kwargs: Any) -> Any:
        response = self._responses.create(*args, **kwargs)
        _persist_response_sync(
            response,
            api_kind="responses",
            session_id_override=self._session_id_override,
            capture_online=self._capture_online,
        )
        return response

    def __getattr__(self, name: str) -> Any:
        return getattr(self._responses, name)


class _AsyncResponsesWrapper:
    def __init__(
        self,
        responses: Any,
        *,
        session_id_override: str | None,
        capture_online: bool,
    ):
        self._responses = responses
        self._session_id_override = session_id_override
        self._capture_online = capture_online

    async def create(self, *args: Any, **kwargs: Any) -> Any:
        response = await self._responses.create(*args, **kwargs)
        await _persist_response_async(
            response,
            api_kind="responses",
            session_id_override=self._session_id_override,
            capture_online=self._capture_online,
        )
        return response

    def __getattr__(self, name: str) -> Any:
        return getattr(self._responses, name)


class ReasoningPersistenceOpenAIClient:
    """Transparent sync OpenAI client with passive response persistence hooks."""

    def __init__(
        self,
        client: OpenAI | None = None,
        *,
        session_id_override: str | None = None,
        capture_online: bool = True,
        **kwargs: Any,
    ):
        self._session_id_override = session_id_override
        self._capture_online = capture_online
        self._client = client or OpenAI(**kwargs)
        self._wrap_namespaces()

    def _wrap_namespaces(self) -> None:
        self.chat = _ChatWrapper(
            self._client.chat,
            async_mode=False,
            session_id_override=self._session_id_override,
            capture_online=self._capture_online,
        )
        self.responses = _SyncResponsesWrapper(
            self._client.responses,
            session_id_override=self._session_id_override,
            capture_online=self._capture_online,
        )

    def with_options(self, **kwargs: Any) -> "ReasoningPersistenceOpenAIClient":
        return self.__class__._from_client(
            self._client.with_options(**kwargs),
            session_id_override=self._session_id_override,
            capture_online=self._capture_online,
        )

    @classmethod
    def _from_client(
        cls,
        client: OpenAI,
        *,
        session_id_override: str | None,
        capture_online: bool,
    ) -> "ReasoningPersistenceOpenAIClient":
        instance = cls.__new__(cls)
        instance._session_id_override = session_id_override
        instance._capture_online = capture_online
        instance._client = client
        instance._wrap_namespaces()
        return instance

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)


class ReasoningPersistenceAsyncOpenAIClient:
    """Transparent async OpenAI client with passive response persistence hooks."""

    def __init__(
        self,
        client: AsyncOpenAI | None = None,
        *,
        session_id_override: str | None = None,
        capture_online: bool = True,
        **kwargs: Any,
    ):
        self._session_id_override = session_id_override
        self._capture_online = capture_online
        self._client = client or AsyncOpenAI(**kwargs)
        self._wrap_namespaces()

    def _wrap_namespaces(self) -> None:
        self.chat = _ChatWrapper(
            self._client.chat,
            async_mode=True,
            session_id_override=self._session_id_override,
            capture_online=self._capture_online,
        )
        self.responses = _AsyncResponsesWrapper(
            self._client.responses,
            session_id_override=self._session_id_override,
            capture_online=self._capture_online,
        )

    def with_options(self, **kwargs: Any) -> "ReasoningPersistenceAsyncOpenAIClient":
        return self.__class__._from_client(
            self._client.with_options(**kwargs),
            session_id_override=self._session_id_override,
            capture_online=self._capture_online,
        )

    @classmethod
    def _from_client(
        cls,
        client: AsyncOpenAI,
        *,
        session_id_override: str | None,
        capture_online: bool,
    ) -> "ReasoningPersistenceAsyncOpenAIClient":
        instance = cls.__new__(cls)
        instance._session_id_override = session_id_override
        instance._capture_online = capture_online
        instance._client = client
        instance._wrap_namespaces()
        return instance

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)


def encode_image_to_base64(image: np.ndarray | str | Path) -> str:
    """Encodes an image to a base64 string.

    Args:
        image: Either a numpy array of shape (H, W, 3) in RGB format, a path string,
            or a Path object to an image file.

    Returns:
        str: The base64 encoded image string.
    """
    if isinstance(image, (str, Path)):
        # Read image directly from path.
        with Image.open(image) as img:
            # Convert to RGB in case it's not.
            img = img.convert("RGB")
            # Save to bytes.
            buffer = BytesIO()
            img.save(buffer, format="JPEG")
            return base64.b64encode(buffer.getvalue()).decode("utf-8")
    else:
        # Convert numpy array to PIL Image.
        img = Image.fromarray(image)
        buffer = BytesIO()
        img.save(buffer, format="JPEG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")
