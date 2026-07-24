import asyncio
import json
import sqlite3

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

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


def _table_count(db_path, table: str) -> int:
    with sqlite3.connect(db_path) as conn:
        return int(conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])


def teardown_function():
    configure_reasoning_persistence(enabled=False, provider="disabled")


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
