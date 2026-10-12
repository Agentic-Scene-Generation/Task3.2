"""Task-local capture at the actual HTTP request boundary, after SDK filtering."""

from __future__ import annotations

import json
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Any, Iterator

from scenesmith.scene_expert.slow_memory.paired import EXPECTED_MODEL, write_json

_target: ContextVar[Path | None] = ContextVar("sceneexpert_pair_wire", default=None)
_observer: ContextVar[Any] = ContextVar("sceneexpert_pair_observer", default=None)
_sampler: ContextVar[Any] = ContextVar("sceneexpert_pair_scope_sampler", default=None)
_expected: ContextVar[dict[str, Any] | None] = ContextVar(
    "sceneexpert_pair_expected", default=None
)


@contextmanager
def capture_wire(
    candidate_dir: Path | None,
    *,
    observer: Any = None,
    expected_request: dict[str, Any] | None = None,
    scope_sampler: Any = None,
) -> Iterator[None]:
    token = _target.set(candidate_dir)
    observer_token = _observer.set(observer)
    expected_token = _expected.set(expected_request)
    sampler_token = _sampler.set(scope_sampler)
    try:
        yield
    finally:
        _sampler.reset(sampler_token)
        _expected.reset(expected_token)
        _observer.reset(observer_token)
        _target.reset(token)


def wrap_pair_provider(provider: Any) -> Any:
    """Observe native model results only inside a direct-decision collection."""
    sampler = _sampler.get()
    if sampler is not None:
        from scenesmith.scene_expert.slow_memory.scope_sampling import ScopeSamplingProvider
        return ScopeSamplingProvider(provider, sampler)
    observer = _observer.get()
    if not callable(getattr(observer, "observe_response", None)):
        return provider
    from scenesmith.scene_expert.slow_memory.decision_pair import DecisionProvider

    return DecisionProvider(provider, observer)


def client_options() -> dict[str, Any]:
    """Keep the normal SDK client unchanged when collection is disabled."""
    destination = _target.get()
    if destination is None:
        return {}
    from openai import DefaultAsyncHttpxClient

    observer = _observer.get()
    expected = _expected.get()
    sampler = _sampler.get()

    async def on_request(request: Any) -> None:
        if not request.url.path.endswith("/chat/completions"):
            return
        payload = json.loads(request.content)
        if payload.get("model") != EXPECTED_MODEL:
            raise ValueError("paired collection observed a different request model")
        if callable(getattr(observer, "observe_request", None)):
            observer.observe_request(payload)
            return
        if sampler is not None:
            sampler.observe_request(payload)
        path = destination / "first_request.json"
        if not path.exists():
            if expected is not None and payload != expected:
                raise ValueError("shadow decision request differs before execution")
            # Request body only; credentials in HTTP headers are never persisted.
            write_json(path, payload)

    return {
        "http_client": DefaultAsyncHttpxClient(event_hooks={"request": [on_request]})
    }
