import asyncio
import json
import sqlite3

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from openai import omit
from openai.types.chat import ChatCompletion

import scenesmith.utils.openai as openai_utils

from scenesmith.utils.openai import (
    ReasoningPersistenceAsyncOpenAIClient,
    ReasoningPersistenceOpenAIClient,
    configure_reasoning_persistence,
    extract_qwen_thinking,
    reasoning_persistence_context,
)


def _chat_response(
    *,
    content: str = "visible",
    reasoning: str | None = None,
    reasoning_content: str | None = None,
    reasoning_details=None,
    model: str = "Qwen/Qwen3.6-35B-A3B",
):
    message = SimpleNamespace(content=content)
    if reasoning is not None:
        message.reasoning = reasoning
    if reasoning_content is not None:
        message.reasoning_content = reasoning_content
    if reasoning_details is not None:
        message.reasoning_details = reasoning_details
    return SimpleNamespace(
        id="response-test",
        model=model,
        choices=[
            SimpleNamespace(
                index=0,
                finish_reason="stop",
                message=message,
            )
        ],
    )


def _sync_client(response):
    client = Mock()
    client.chat.completions.create.return_value = response
    client.responses.create.return_value = response
    return client


def _async_client(response):
    client = Mock()
    client.chat.completions.create = AsyncMock(return_value=response)
    client.responses.create = AsyncMock(return_value=response)
    return client


class _AsyncChunkStream:
    def __init__(self, chunks):
        self._chunks = chunks

    def __aiter__(self):
        return self._iterate()

    async def _iterate(self):
        for chunk in self._chunks:
            yield chunk


class _InterruptedAsyncChunkStream:
    def __aiter__(self):
        return self._iterate()

    async def _iterate(self):
        yield {
            "id": "chatcmpl-interrupted",
            "created": 123,
            "model": "openai/gpt-5.6-luna-pro",
            "choices": [
                {
                    "index": 0,
                    "delta": {"reasoning": "partial"},
                    "finish_reason": None,
                }
            ],
        }
        remote_protocol_error = type("RemoteProtocolError", (Exception,), {})
        raise remote_protocol_error(
            "peer closed connection without sending complete message body "
            "(incomplete chunked read)"
        )


def _function_tool(name: str):
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": f"Call {name}",
            "parameters": {"type": "object", "properties": {}},
        },
    }


def _qwen_stream_chunks():
    return [
        {
            "id": "chatcmpl-qwen-stream",
            "created": 456,
            "model": "Qwen/Qwen3.8-27B",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "role": "assistant",
                        "reasoning_content": "private ",
                        "content": "visible ",
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call-observe",
                                "type": "function",
                                "function": {
                                    "name": "observe_scene",
                                    "arguments": '{"camera":',
                                },
                            }
                        ],
                    },
                    "finish_reason": None,
                }
            ],
        },
        {
            "id": "chatcmpl-qwen-stream",
            "created": 456,
            "model": "Qwen/Qwen3.8-27B",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "reasoning_content": "reasoning",
                        "content": "answer",
                        "tool_calls": [
                            {
                                "index": 0,
                                "function": {"arguments": '"top"}'},
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
        },
        {
            "id": "chatcmpl-qwen-stream",
            "created": 456,
            "model": "Qwen/Qwen3.8-27B",
            "choices": [],
            "usage": {
                "prompt_tokens": 11,
                "completion_tokens": 7,
                "total_tokens": 18,
            },
        },
    ]


def _table_count(db_path, table: str) -> int:
    with sqlite3.connect(db_path) as conn:
        return int(conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])


def teardown_function():
    configure_reasoning_persistence(enabled=False, provider="disabled")


def test_async_reasoning_stream_retries_interrupted_body(monkeypatch):
    monkeypatch.setenv("SCENEEXPERT_REASONING_STREAM", "true")
    monkeypatch.setenv("SCENEEXPERT_OPENAI_TRANSIENT_RETRY_DELAYS", "0")
    configure_reasoning_persistence(enabled=True, provider="openrouter")

    completed_stream = _AsyncChunkStream(
        [
            {
                "id": "chatcmpl-retried",
                "created": 124,
                "model": "openai/gpt-5.6-luna-pro",
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "reasoning": "complete reasoning",
                            "content": "done",
                        },
                        "finish_reason": "stop",
                    }
                ],
            }
        ]
    )
    raw_client = Mock()
    raw_client.chat.completions.create = AsyncMock(
        side_effect=[_InterruptedAsyncChunkStream(), completed_stream]
    )
    raw_client.responses.create = AsyncMock()
    client = ReasoningPersistenceAsyncOpenAIClient(client=raw_client)

    response = asyncio.run(
        client.chat.completions.create(
            model="openai/gpt-5.6-luna-pro",
            messages=[{"role": "user", "content": "make a room"}],
        )
    )

    assert raw_client.chat.completions.create.await_count == 2
    assert response.id == "chatcmpl-retried"
    assert response.choices[0].message.content == "done"
    assert response.choices[0].message.reasoning == "complete reasoning"


def test_extract_qwen_reasoning_fields_and_inline_content_without_mutation():
    message = SimpleNamespace(
        content="<think>\ninline secret\n</think>\nvisible answer",
        reasoning_content="structured secret",
    )
    original_content = message.content

    assert extract_qwen_thinking(message) == "structured secret"
    assert message.content == original_content

    inline_only = SimpleNamespace(content=original_content)
    assert extract_qwen_thinking(inline_only) == "inline secret"
    assert inline_only.content == original_content


def test_qwen_sync_wrapper_preserves_request_response_and_writes_thinking(tmp_path):
    configure_reasoning_persistence(enabled=True, provider="qwen")
    response = _chat_response(reasoning_content="private reasoning")
    raw_client = _sync_client(response)
    client = ReasoningPersistenceOpenAIClient(client=raw_client)
    db_path = tmp_path / "designer.db"
    extra_body = {"chat_template_kwargs": {"enable_thinking": True}}

    async def run():
        async with reasoning_persistence_context("designer", db_path):
            result = client.chat.completions.create(
                model="Qwen/Qwen3.6-35B-A3B",
                messages=[{"role": "user", "content": "hello"}],
                extra_body=extra_body,
            )
            assert result is response

    asyncio.run(run())

    kwargs = raw_client.chat.completions.create.call_args.kwargs
    assert kwargs["extra_body"] is extra_body
    assert kwargs["messages"] == [{"role": "user", "content": "hello"}]
    with sqlite3.connect(db_path) as conn:
        row = conn.execute(
            "SELECT session_id, thinking, content_preview FROM agent_thinking"
        ).fetchone()
    assert row == ("designer", "private reasoning", "visible")


def test_qwen_vlm_override_writes_to_active_agent_db(tmp_path):
    configure_reasoning_persistence(enabled=True, provider="qwen")
    response = _chat_response(reasoning="tool reasoning")
    raw_client = _sync_client(response)
    client = ReasoningPersistenceOpenAIClient(
        client=raw_client,
        session_id_override="vlm",
        capture_online=False,
    )
    db_path = tmp_path / "designer.db"

    async def run():
        async with reasoning_persistence_context("designer", db_path):
            assert client.chat.completions.create(model="qwen", messages=[]) is response

    asyncio.run(run())

    with sqlite3.connect(db_path) as conn:
        row = conn.execute("SELECT session_id, thinking FROM agent_thinking").fetchone()
    assert row == ("vlm", "tool reasoning")


def test_qwen_blank_thinking_does_not_write_empty_row(tmp_path):
    configure_reasoning_persistence(enabled=True, provider="qwen")
    response = _chat_response(reasoning_content="   ")
    raw_client = _sync_client(response)
    client = ReasoningPersistenceOpenAIClient(client=raw_client)
    db_path = tmp_path / "designer.db"

    async def run():
        async with reasoning_persistence_context("designer", db_path):
            client.chat.completions.create(model="qwen", messages=[])

    asyncio.run(run())

    assert _table_count(db_path, "agent_thinking") == 0


def test_openrouter_chat_writes_summary_and_raw_json_only_to_online_table(tmp_path):
    configure_reasoning_persistence(enabled=True, provider="openrouter")
    response = _chat_response(
        reasoning="Readable summary",
        reasoning_details=[
            {"type": "reasoning.summary", "summary": "detail summary"},
            {"type": "reasoning.encrypted", "data": "opaque"},
        ],
        model="openai/gpt-5.2",
    )
    raw_client = _sync_client(response)
    client = ReasoningPersistenceOpenAIClient(client=raw_client)
    db_path = tmp_path / "critic.db"

    async def run():
        async with reasoning_persistence_context("critic", db_path):
            assert client.chat.completions.create(model="openai/gpt-5.2") is response

    asyncio.run(run())

    with sqlite3.connect(db_path) as conn:
        row = conn.execute(
            """
            SELECT session_id, provider, source_type, summary, raw_json
            FROM agent_reasoning_artifacts
            """
        ).fetchone()
        qwen_table = conn.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type='table' AND name='agent_thinking'"
        ).fetchone()
    assert row[:4] == (
        "critic",
        "openrouter",
        "openrouter_reasoning_details",
        "Readable summary",
    )
    assert json.loads(row[4])["reasoning_details"][1]["data"] == "opaque"
    assert qwen_table is None


def test_openrouter_details_can_persist_raw_json_with_null_summary(tmp_path):
    configure_reasoning_persistence(enabled=True, provider="openrouter")
    response = _chat_response(
        reasoning_details=[{"type": "reasoning.encrypted", "data": "opaque"}],
        model="openai/gpt-5.2",
    )
    raw_client = _sync_client(response)
    client = ReasoningPersistenceOpenAIClient(client=raw_client)
    db_path = tmp_path / "designer.db"

    async def run():
        async with reasoning_persistence_context("designer", db_path):
            client.chat.completions.create(model="openai/gpt-5.2")

    asyncio.run(run())

    with sqlite3.connect(db_path) as conn:
        summary, raw_json = conn.execute(
            "SELECT summary, raw_json FROM agent_reasoning_artifacts"
        ).fetchone()
    assert summary is None
    assert json.loads(raw_json)["reasoning_details"][0]["data"] == "opaque"


def test_openrouter_reasoning_content_alias_is_persisted(tmp_path):
    configure_reasoning_persistence(enabled=True, provider="openrouter")
    response = _chat_response(
        reasoning_content="Readable alias summary",
        model="openai/gpt-5.2",
    )
    raw_client = _sync_client(response)
    client = ReasoningPersistenceOpenAIClient(client=raw_client)
    db_path = tmp_path / "designer.db"

    async def run():
        async with reasoning_persistence_context("designer", db_path):
            client.chat.completions.create(model="openai/gpt-5.2")

    asyncio.run(run())

    with sqlite3.connect(db_path) as conn:
        source_type, summary, raw_json = conn.execute(
            "SELECT source_type, summary, raw_json " "FROM agent_reasoning_artifacts"
        ).fetchone()
    assert source_type == "openrouter_reasoning_content"
    assert summary == "Readable alias summary"
    assert json.loads(raw_json)["reasoning_content"] == "Readable alias summary"


def test_openai_responses_reasoning_summary_is_persisted(tmp_path):
    configure_reasoning_persistence(enabled=True, provider="openai")
    response = SimpleNamespace(
        id="resp-1",
        model="gpt-5",
        output_text="visible",
        output=[
            SimpleNamespace(
                type="reasoning",
                summary=[
                    {
                        "type": "summary_text",
                        "text": "A concise reasoning summary",
                    }
                ],
            )
        ],
    )
    raw_client = _sync_client(response)
    client = ReasoningPersistenceOpenAIClient(client=raw_client)
    db_path = tmp_path / "designer.db"

    async def run():
        async with reasoning_persistence_context("designer", db_path):
            assert client.responses.create(model="gpt-5", input="hello") is response

    asyncio.run(run())

    with sqlite3.connect(db_path) as conn:
        row = conn.execute(
            "SELECT provider, source_type, summary FROM agent_reasoning_artifacts"
        ).fetchone()
    assert row == (
        "openai",
        "openai_responses_reasoning",
        "A concise reasoning summary",
    )


def test_with_options_returns_wrapped_client_and_keeps_both_hooks(tmp_path):
    configure_reasoning_persistence(enabled=True, provider="qwen")
    response = _chat_response(reasoning="kept hook")
    raw_client = _sync_client(response)
    optioned_raw_client = _sync_client(response)
    raw_client.with_options.return_value = optioned_raw_client
    client = ReasoningPersistenceOpenAIClient(client=raw_client)

    optioned = client.with_options(timeout=12.0)

    assert isinstance(optioned, ReasoningPersistenceOpenAIClient)
    raw_client.with_options.assert_called_once_with(timeout=12.0)
    db_path = tmp_path / "designer.db"

    async def run():
        async with reasoning_persistence_context("designer", db_path):
            assert optioned.chat.completions.create(model="qwen") is response

    asyncio.run(run())
    assert _table_count(db_path, "agent_thinking") == 1
    assert optioned.responses._responses is optioned_raw_client.responses


def test_with_options_keeps_responses_persistence_hook(tmp_path):
    configure_reasoning_persistence(enabled=True, provider="openai")
    response = SimpleNamespace(
        id="resp-with-options",
        model="gpt-5",
        output=[
            {
                "type": "reasoning",
                "summary": [{"type": "summary_text", "text": "kept response hook"}],
            }
        ],
        output_text="visible",
    )
    raw_client = _sync_client(response)
    optioned_raw_client = _sync_client(response)
    raw_client.with_options.return_value = optioned_raw_client
    optioned = ReasoningPersistenceOpenAIClient(client=raw_client).with_options(
        timeout=12.0
    )
    db_path = tmp_path / "designer.db"

    async def run():
        async with reasoning_persistence_context("designer", db_path):
            assert optioned.responses.create(model="gpt-5", input="hello") is response

    asyncio.run(run())

    with sqlite3.connect(db_path) as conn:
        row = conn.execute(
            "SELECT source_type, summary FROM agent_reasoning_artifacts"
        ).fetchone()
    assert row == ("openai_responses_reasoning", "kept response hook")


def test_async_with_options_keeps_wrapper_and_returns_same_response(tmp_path):
    configure_reasoning_persistence(enabled=True, provider="qwen")
    response = _chat_response(reasoning="async hook")
    raw_client = _async_client(response)
    optioned_raw_client = _async_client(response)
    raw_client.with_options.return_value = optioned_raw_client
    client = ReasoningPersistenceAsyncOpenAIClient(client=raw_client)
    optioned = client.with_options(timeout=8.0)

    assert isinstance(optioned, ReasoningPersistenceAsyncOpenAIClient)

    async def run():
        db_path = tmp_path / "critic.db"
        async with reasoning_persistence_context("critic", db_path):
            result = await optioned.chat.completions.create(model="qwen")
            assert result is response
        return db_path

    db_path = asyncio.run(run())
    assert _table_count(db_path, "agent_thinking") == 1
    assert optioned.responses._responses is optioned_raw_client.responses


def test_two_concurrent_contexts_do_not_cross_session_or_database(tmp_path):
    configure_reasoning_persistence(enabled=True, provider="qwen")

    async def run_one(session_id: str, db_path, thinking: str):
        response = _chat_response(reasoning=thinking)
        client = ReasoningPersistenceAsyncOpenAIClient(client=_async_client(response))
        async with reasoning_persistence_context(session_id, db_path):
            await asyncio.sleep(0)
            await client.chat.completions.create(model="qwen")
            await asyncio.sleep(0)

    first_db = tmp_path / "first.db"
    second_db = tmp_path / "second.db"

    async def run_both():
        await asyncio.gather(
            run_one("designer_a", first_db, "thinking a"),
            run_one("designer_b", second_db, "thinking b"),
        )

    asyncio.run(run_both())

    with sqlite3.connect(first_db) as conn:
        first = conn.execute(
            "SELECT session_id, thinking FROM agent_thinking"
        ).fetchall()
    with sqlite3.connect(second_db) as conn:
        second = conn.execute(
            "SELECT session_id, thinking FROM agent_thinking"
        ).fetchall()
    assert first == [("designer_a", "thinking a")]
    assert second == [("designer_b", "thinking b")]


def test_missing_and_in_memory_db_paths_are_noops():
    configure_reasoning_persistence(enabled=True, provider="qwen")
    response = _chat_response(reasoning="not persisted")
    client = ReasoningPersistenceOpenAIClient(client=_sync_client(response))

    async def run():
        async with reasoning_persistence_context("designer", None):
            assert client.chat.completions.create(model="qwen") is response
        async with reasoning_persistence_context("designer", ":memory:"):
            assert client.chat.completions.create(model="qwen") is response

    asyncio.run(run())


def test_schema_creation_is_idempotent_and_plain_response_creates_no_row(tmp_path):
    configure_reasoning_persistence(enabled=True, provider="qwen")
    db_path = tmp_path / "designer.db"
    plain_response = _chat_response(content="plain response")
    plain_client = ReasoningPersistenceOpenAIClient(client=_sync_client(plain_response))

    async def run():
        for _ in range(2):
            async with reasoning_persistence_context("designer", db_path):
                assert (
                    plain_client.chat.completions.create(model="qwen") is plain_response
                )

    asyncio.run(run())
    assert _table_count(db_path, "agent_thinking") == 0


def test_empty_online_reasoning_fields_do_not_create_artifact(tmp_path):
    configure_reasoning_persistence(enabled=True, provider="openrouter")
    response = _chat_response(reasoning="", reasoning_details=[])
    client = ReasoningPersistenceOpenAIClient(client=_sync_client(response))
    db_path = tmp_path / "designer.db"

    async def run():
        async with reasoning_persistence_context("designer", db_path):
            assert client.chat.completions.create(model="gpt-5") is response

    asyncio.run(run())
    assert _table_count(db_path, "agent_reasoning_artifacts") == 0


def test_sqlite_writer_failure_is_fail_open(monkeypatch, tmp_path):
    configure_reasoning_persistence(enabled=True, provider="qwen")
    response = _chat_response(reasoning="must not break response")
    client = ReasoningPersistenceAsyncOpenAIClient(client=_async_client(response))

    def fail_write(*args, **kwargs):
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(openai_utils, "_write_qwen_records", fail_write)

    async def run():
        async with reasoning_persistence_context("designer", tmp_path / "designer.db"):
            return await client.chat.completions.create(model="qwen")

    assert asyncio.run(run()) is response


def test_auto_provider_prioritizes_endpoint_and_fails_closed_when_unknown():
    assert (
        configure_reasoning_persistence(
            enabled=True,
            provider="auto",
            model_id="Qwen/Qwen3.6",
            base_url="https://openrouter.ai/api/v1",
        )
        == "openrouter"
    )
    assert (
        configure_reasoning_persistence(
            enabled=True,
            provider="auto",
            model_id="custom-model",
            base_url="https://unknown.example/v1",
        )
        == "disabled"
    )


def test_async_openrouter_reasoning_stream_returns_standard_chat_completion(
    monkeypatch,
):
    configure_reasoning_persistence(
        enabled=True,
        provider="openrouter",
        model_id="openai/gpt-5.6-luna-pro",
        base_url="https://openrouter.ai/api/v1",
    )
    monkeypatch.setenv("SCENEEXPERT_REASONING_STREAM", "true")
    chunks = [
        {
            "id": "chatcmpl-stream-test",
            "created": 123,
            "model": "openai/gpt-5.6-luna-pro",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "role": "assistant",
                        "reasoning": "private ",
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call-1",
                                "type": "function",
                                "function": {
                                    "name": "create_room",
                                    "arguments": '{"width":',
                                },
                            }
                        ],
                    },
                    "finish_reason": None,
                }
            ],
        },
        {
            "id": "chatcmpl-stream-test",
            "created": 123,
            "model": "openai/gpt-5.6-luna-pro",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "reasoning": "reasoning",
                        "tool_calls": [
                            {
                                "index": 0,
                                "function": {"arguments": "2.5}"},
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
                "completion_tokens_details": {"reasoning_tokens": 3},
            },
        },
    ]
    raw_client = Mock()
    raw_client.chat.completions.create = AsyncMock(
        return_value=_AsyncChunkStream(chunks)
    )
    raw_client.responses.create = AsyncMock()
    client = ReasoningPersistenceAsyncOpenAIClient(client=raw_client)

    response = asyncio.run(
        client.chat.completions.create(
            model="openai/gpt-5.6-luna-pro",
            messages=[{"role": "user", "content": "make a room"}],
        )
    )

    assert isinstance(response, ChatCompletion)
    assert response.created == 123
    assert response.choices[0].finish_reason == "tool_calls"
    assert response.choices[0].message.reasoning == "private reasoning"
    tool_call = response.choices[0].message.tool_calls[0]
    assert tool_call.id == "call-1"
    assert tool_call.function.name == "create_room"
    assert tool_call.function.arguments == '{"width":2.5}'
    assert response.usage.completion_tokens_details.reasoning_tokens == 3
    request_kwargs = raw_client.chat.completions.create.call_args.kwargs
    assert request_kwargs["stream"] is True
    assert request_kwargs["stream_options"] == {"include_usage": True}


def test_sync_generic_stream_assembles_qwen_chat_completion(monkeypatch, tmp_path):
    monkeypatch.setenv("SCENEEXPERT_CHAT_COMPLETIONS_STREAM", "true")
    configure_reasoning_persistence(enabled=True, provider="qwen")
    raw_client = Mock()
    raw_client.chat.completions.create.return_value = iter(_qwen_stream_chunks())
    raw_client.responses.create = Mock()
    client = ReasoningPersistenceOpenAIClient(client=raw_client)
    db_path = tmp_path / "designer.db"

    async def run():
        async with reasoning_persistence_context("designer", db_path):
            return client.chat.completions.create(
                model="Qwen/Qwen3.8-27B",
                messages=[{"role": "user", "content": "inspect the scene"}],
                stream=omit,
                stream_options={"continuous_usage_stats": True},
            )

    response = asyncio.run(run())

    assert isinstance(response, ChatCompletion)
    assert response.choices[0].message.content == "visible answer"
    assert response.choices[0].message.reasoning_content == "private reasoning"
    tool_call = response.choices[0].message.tool_calls[0]
    assert tool_call.id == "call-observe"
    assert tool_call.function.name == "observe_scene"
    assert tool_call.function.arguments == '{"camera":"top"}'
    assert response.choices[0].finish_reason == "tool_calls"
    assert response.usage.total_tokens == 18
    request_kwargs = raw_client.chat.completions.create.call_args.kwargs
    assert request_kwargs["stream"] is True
    assert request_kwargs["stream_options"] == {
        "continuous_usage_stats": True,
        "include_usage": True,
    }
    with sqlite3.connect(db_path) as conn:
        persisted = conn.execute(
            "SELECT session_id, thinking FROM agent_thinking"
        ).fetchone()
    assert persisted == ("designer", "private reasoning")


def test_async_generic_stream_normalizes_named_tool_choice(monkeypatch):
    monkeypatch.setenv("SCENEEXPERT_CHAT_COMPLETIONS_STREAM", "true")
    tools = [_function_tool("observe_scene"), _function_tool("move_object")]
    original_tools = json.loads(json.dumps(tools))
    raw_client = Mock()
    raw_client.chat.completions.create = AsyncMock(
        return_value=_AsyncChunkStream(_qwen_stream_chunks())
    )
    raw_client.responses.create = AsyncMock()
    client = ReasoningPersistenceAsyncOpenAIClient(client=raw_client)

    response = asyncio.run(
        client.chat.completions.create(
            model="Qwen/Qwen3.8-27B",
            messages=[],
            tools=tools,
            tool_choice={
                "type": "function",
                "function": {"name": "observe_scene"},
            },
        )
    )

    assert isinstance(response, ChatCompletion)
    request_kwargs = raw_client.chat.completions.create.call_args.kwargs
    assert request_kwargs["tool_choice"] == "required"
    assert request_kwargs["tools"] == [_function_tool("observe_scene")]
    assert request_kwargs["stream"] is True
    assert tools == original_tools


def test_generic_stream_policy_defaults_off_and_can_be_disabled(monkeypatch):
    response = _chat_response()
    raw_client = _sync_client(response)
    client = ReasoningPersistenceOpenAIClient(client=raw_client)

    monkeypatch.delenv("SCENEEXPERT_CHAT_COMPLETIONS_STREAM", raising=False)
    assert client.chat.completions.create(model="qwen") is response
    assert "stream" not in raw_client.chat.completions.create.call_args.kwargs

    monkeypatch.setenv("SCENEEXPERT_CHAT_COMPLETIONS_STREAM", "false")
    assert client.chat.completions.create(model="qwen", stream=False) is response
    assert raw_client.chat.completions.create.call_args.kwargs["stream"] is False


def test_generic_stream_policy_rejects_invalid_value_before_request(monkeypatch):
    monkeypatch.setenv("SCENEEXPERT_CHAT_COMPLETIONS_STREAM", "sometimes")
    raw_client = _sync_client(_chat_response())
    client = ReasoningPersistenceOpenAIClient(client=raw_client)

    with pytest.raises(
        ValueError,
        match="SCENEEXPERT_CHAT_COMPLETIONS_STREAM must be true or false",
    ):
        client.chat.completions.create(model="qwen")

    raw_client.chat.completions.create.assert_not_called()


def test_explicit_raw_stream_contract_is_not_assembled(monkeypatch):
    monkeypatch.setenv("SCENEEXPERT_CHAT_COMPLETIONS_STREAM", "true")
    raw_stream = iter(_qwen_stream_chunks())
    raw_client = Mock()
    raw_client.chat.completions.create.return_value = raw_stream
    raw_client.responses.create = Mock()
    client = ReasoningPersistenceOpenAIClient(client=raw_client)

    response = client.chat.completions.create(model="qwen", stream=True)

    assert response is raw_stream
    assert raw_client.chat.completions.create.call_args.kwargs["stream"] is True


def test_generic_stream_rejects_clean_eof_without_finish_reason(monkeypatch):
    monkeypatch.setenv("SCENEEXPERT_CHAT_COMPLETIONS_STREAM", "true")
    incomplete_chunks = _qwen_stream_chunks()[:1]

    sync_client = Mock()
    sync_client.chat.completions.create.return_value = iter(incomplete_chunks)
    sync_client.responses.create = Mock()
    sync_wrapper = ReasoningPersistenceOpenAIClient(client=sync_client)

    with pytest.raises(RuntimeError, match="before a finish_reason was received"):
        sync_wrapper.chat.completions.create(model="qwen")

    async_client = Mock()
    async_client.chat.completions.create = AsyncMock(
        return_value=_AsyncChunkStream(incomplete_chunks)
    )
    async_client.responses.create = AsyncMock()
    async_wrapper = ReasoningPersistenceAsyncOpenAIClient(client=async_client)

    with pytest.raises(RuntimeError, match="before a finish_reason was received"):
        asyncio.run(async_wrapper.chat.completions.create(model="qwen"))


def test_named_tool_choice_does_not_narrow_later_requests():
    response = _chat_response()
    raw_client = _sync_client(response)
    client = ReasoningPersistenceOpenAIClient(client=raw_client)
    tools = [_function_tool("observe_scene"), _function_tool("move_object")]

    client.chat.completions.create(
        model="qwen",
        tools=tools,
        tool_choice={
            "type": "function",
            "function": {"name": "observe_scene"},
        },
    )
    first_kwargs = raw_client.chat.completions.create.call_args.kwargs
    client.chat.completions.create(model="qwen", tools=tools, tool_choice=omit)
    second_kwargs = raw_client.chat.completions.create.call_args.kwargs

    assert first_kwargs["tool_choice"] == "required"
    assert first_kwargs["tools"] == [_function_tool("observe_scene")]
    assert second_kwargs["tools"] is tools
    assert second_kwargs["tool_choice"] is omit
    assert len(tools) == 2


def test_string_tool_choices_pass_through_unchanged():
    response = _chat_response()
    raw_client = _sync_client(response)
    client = ReasoningPersistenceOpenAIClient(client=raw_client)
    tools = [_function_tool("observe_scene")]

    for choice in ("auto", "required", "none"):
        assert (
            client.chat.completions.create(
                model="qwen",
                tools=tools,
                tool_choice=choice,
            )
            is response
        )
        request_kwargs = raw_client.chat.completions.create.call_args.kwargs
        assert request_kwargs["tool_choice"] == choice
        assert request_kwargs["tools"] is tools


@pytest.mark.parametrize(
    ("tool_choice", "tools", "error_match"),
    [
        (
            {"type": "function", "function": {}},
            [_function_tool("observe_scene")],
            "exactly one non-empty 'name' field",
        ),
        (
            {
                "type": "function",
                "function": {"name": "observe_scene"},
            },
            [],
            "found 0",
        ),
        (
            {
                "type": "function",
                "function": {"name": "observe_scene"},
            },
            [_function_tool("observe_scene"), _function_tool("observe_scene")],
            "found 2",
        ),
        (
            {"type": "custom", "name": "observe_scene"},
            [_function_tool("observe_scene")],
            "exactly 'type' and 'function' fields",
        ),
    ],
)
def test_invalid_named_tool_choices_fail_before_request(
    tool_choice,
    tools,
    error_match,
):
    raw_client = _sync_client(_chat_response())
    client = ReasoningPersistenceOpenAIClient(client=raw_client)

    with pytest.raises(ValueError, match=error_match):
        client.chat.completions.create(
            model="qwen",
            tools=tools,
            tool_choice=tool_choice,
        )

    raw_client.chat.completions.create.assert_not_called()
