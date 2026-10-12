"""Opt-in, bounded first-response scope sampling with unchanged requests.

Only scope is inspected before execution. Rejected draws execute no tools and
remain archived; scene quality is never used to choose which draw to execute.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

from agents.models.interface import Model, ModelProvider

from scenesmith.scene_expert.slow_memory.decision_scope import DESIGN_TOOLS
from scenesmith.scene_expert.slow_memory.paired import digest, read_json, write_json

SAMPLING_PROTOCOL = "sceneexpert.shadow_direct_action_sampling.v1"


def sampling_policy(max_samples: int) -> dict[str, Any]:
    """Declare an explicit scope-conditioned protocol independently of labels."""
    if not 1 <= max_samples <= 3:
        raise ValueError("shadow scope sampling budget must be 1..3")
    return {"protocol": SAMPLING_PROTOCOL, "max_samples": max_samples,
            "selection": "first_design_response_or_last_draw_at_budget",
            "forced_tool_choice": False, "quality_conditioning": False}


def has_design(items: list[dict[str, Any]]) -> bool:
    """Recognize native first-response design calls, without looking ahead."""
    return any(item.get("type") == "function_call" and item.get("name") in DESIGN_TOOLS
               for item in items)


class ShadowScopeSampler:
    """Retain every first-response draw and its actual, identical wire inputs."""
    def __init__(self, directory: Path, expected_request: dict, policy: dict):
        if policy != sampling_policy(policy["max_samples"]):
            raise ValueError("unsupported shadow scope sampling policy")
        self.directory = directory
        self.expected = deepcopy(expected_request)
        self.policy = deepcopy(policy)
        self.completed = False
        self.active = False
        self.requests: list[dict] = []
        self.attempts: list[dict] = []

    def observe_request(self, payload: dict) -> None:
        """Reject drift on every draw and HTTP retry, before transport or tools."""
        if not self.active:
            return
        if payload != self.expected:
            raise ValueError("scope sampling request differs before execution")
        self.requests.append(deepcopy(payload))

    async def sample(self, model: Model, args: tuple, kwargs: dict) -> Any:
        from scenesmith.scene_expert.slow_memory.paired_runtime import json_value

        if self.completed:
            return await model.get_response(*args, **kwargs)
        if self.active or self.attempts:
            raise ValueError("cannot restart a partial shadow sampling invocation")
        for ordinal in range(self.policy["max_samples"]):
            self.requests = []
            self.active = True
            try:
                response = await model.get_response(*args, **kwargs)
            finally:
                self.active = False
            if not self.requests:
                raise ValueError("scope sample has no actual wire request evidence")
            items = json_value(response.output)
            in_scope = has_design(items)
            record = {"ordinal": ordinal, "requests": self.requests,
                      "response_items": items, "usage": json_value(getattr(response, "usage", None)),
                      "has_direct_design": in_scope, "tools_executed_before_selection": False}
            name = f"scope_samples/sample_{ordinal:03d}.json"
            write_json(self.directory / name, record)
            self.attempts.append({"path": name, "sha256": digest(record)})
            if in_scope or ordinal + 1 == self.policy["max_samples"]:
                self.completed = True
                write_json(self.directory / "scope_sampling.json", {
                    **self.policy, "request_sha256": digest(self.expected),
                    "attempts": self.attempts, "selected_ordinal": ordinal,
                    "selected_has_direct_design": in_scope,
                    "selected_response_sha256": digest(items),
                })
                return response
        raise AssertionError("bounded scope sampling did not return a response")


class ScopeSamplingModel(Model):
    """Sample only the first response, then keep native continuation unchanged."""
    def __init__(self, model: Model, sampler: ShadowScopeSampler):
        self.model, self.sampler = model, sampler

    async def get_response(self, *args: Any, **kwargs: Any) -> Any:
        return await self.sampler.sample(self.model, args, kwargs)

    def stream_response(self, *args: Any, **kwargs: Any) -> Any:
        raise ValueError("streaming is unsupported for audited scope sampling")


class ScopeSamplingProvider(ModelProvider):
    """Apply the sampler only within the existing task-local capture context."""
    def __init__(self, provider: ModelProvider, sampler: ShadowScopeSampler):
        self.provider, self.sampler = provider, sampler

    def get_model(self, model_name: str | None) -> Model:
        return ScopeSamplingModel(self.provider.get_model(model_name), self.sampler)


def validate_sampling_proof(directory: Path, policy: dict, expected: dict,
                            selected_items: list[dict] | None = None) -> str:
    """Bind the chosen first response to all retained, quality-blind scope draws."""
    summary = read_json(directory / "scope_sampling.json")
    if policy != sampling_policy(policy["max_samples"]) or any(
            summary.get(k) != v for k, v in policy.items()):
        raise ValueError("scope sampling policy differs from frozen snapshot")
    attempts = summary.get("attempts") or []
    if not 1 <= len(attempts) <= policy["max_samples"]:
        raise ValueError("scope sampling budget evidence is invalid")
    if summary.get("request_sha256") != digest(expected):
        raise ValueError("scope sampling request proof mismatch")
    for ordinal, entry in enumerate(attempts):
        name = f"scope_samples/sample_{ordinal:03d}.json"
        if entry.get("path") != name:
            raise ValueError("scope sampling ordinal/path mismatch")
        row = read_json(directory / name)
        requests = row.get("requests") or []
        direct = has_design(row["response_items"])
        if (entry.get("sha256") != digest(row) or row.get("ordinal") != ordinal
                or not requests or any(request != expected for request in requests)
                or row.get("has_direct_design") != direct
                or row.get("tools_executed_before_selection") is not False):
            raise ValueError("scope sampling draw proof mismatch")
        if ordinal + 1 < len(attempts) and direct:
            raise ValueError("scope sampling skipped an eligible earlier draw")
    selected = len(attempts) - 1
    if (summary.get("selected_ordinal") != selected
            or summary.get("selected_has_direct_design") != direct
            or summary.get("selected_response_sha256") != digest(row["response_items"])
            or (not direct and len(attempts) != policy["max_samples"])):
        raise ValueError("scope sampling selected response proof mismatch")
    if selected_items is not None and selected_items != row["response_items"]:
        raise ValueError("executed first response differs from selected scope sample")
    return digest(summary)
