"""Task-local capture at the actual HTTP request boundary, after SDK filtering."""

from __future__ import annotations

import json
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Any, Iterator

from scenesmith.scene_expert.slow_memory.paired import EXPECTED_MODEL, write_json

_target: ContextVar[Path | None] = ContextVar("sceneexpert_pair_wire", default=None)


@contextmanager
def capture_wire(candidate_dir: Path | None) -> Iterator[None]:
    token = _target.set(candidate_dir)
    try:
        yield
    finally:
        _target.reset(token)


def client_options() -> dict[str, Any]:
    """Keep the normal SDK client unchanged when collection is disabled."""
    destination = _target.get()
    if destination is None:
        return {}
    from openai import DefaultAsyncHttpxClient

    async def on_request(request: Any) -> None:
        if not request.url.path.endswith("/chat/completions"):
            return
        payload = json.loads(request.content)
        if payload.get("model") != EXPECTED_MODEL:
            raise ValueError("paired collection observed a different request model")
        path = destination / "first_request.json"
        if not path.exists():
            # Request body only; credentials in HTTP headers are never persisted.
            write_json(path, payload)

    return {
        "http_client": DefaultAsyncHttpxClient(event_hooks={"request": [on_request]})
    }
