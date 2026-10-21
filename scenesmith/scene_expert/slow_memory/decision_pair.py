"""Opt-in fork at the first natural scene action, after a shared read-only prefix.

A's response identifies the boundary before its tools execute. By default B
samples once from that exact request and native SDK input. A separately frozen,
opt-in scope protocol can retain up to three draws before executing the first
design response. No tool choice or scene-quality conditioning is applied.
Evidence evaluates each remaining rollout, rather than immediate action reward.
"""

from __future__ import annotations

import logging
from copy import deepcopy
from pathlib import Path
import os
from types import SimpleNamespace
from typing import Any

from agents.models.interface import Model, ModelProvider

from scenesmith.scene_expert.slow_memory.paired import (
    copy_scene_tree,
    digest,
    write_json,
)
from scenesmith.scene_expert.slow_memory.paired_runtime import (
    InitialPair,
    json_value,
    random_state,
)
from scenesmith.scene_expert.slow_memory.decision_scope import DESIGN_TOOLS

LOGGER = logging.getLogger(__name__)
PREFIX_TOOLS = frozenset(
    {
        "observe_scene",
        "list_available_assets",
        "get_current_scene_state",
        "designer_todo_manager",
    }
)


def workflow_owner(tools: list[Any]) -> Any:
    """Find the known WorkflowTools closure without serializing opaque clients."""
    from scenesmith.agent_utils.workflow_tools import WorkflowTools

    tool = next((tool for tool in tools if tool.name == "designer_todo_manager"), None)
    if tool is None:
        raise ValueError("direct decision snapshot requires native workflow tool")
    pending, seen, owners = [tool.on_invoke_tool], set(), []
    while pending:
        value = pending.pop()
        if id(value) in seen:
            continue
        seen.add(id(value))
        if isinstance(value, WorkflowTools):
            owners.append(value)
        for cell in getattr(value, "__closure__", None) or []:
            try:
                pending.append(cell.cell_contents)
            except ValueError:
                pass
        if len(seen) > 100:
            raise ValueError("unsupported workflow closure graph")
    if len(owners) != 1:
        raise ValueError("native workflow closure owner is not unique")
    return owners[0]


def restore_workflow(tools: list[Any], todos: list[dict[str, Any]]) -> None:
    """Restore the only mutable prefix tool state using its native dataclass."""
    from scenesmith.agent_utils.workflow_tools import TodoItem

    owner = workflow_owner(tools)
    owner._designer_todos = [TodoItem(**deepcopy(item)) for item in todos]
    if json_value(owner._designer_todos) != todos:
        raise ValueError("restored shared workflow state differs")


def decision_result(result: Any, snapshot: dict[str, Any]) -> Any:
    """Retain SDK items starting with the fork response, excluding shared actions."""
    expected = snapshot["fork_response_items"]
    start = next(
        (
            i
            for i, item in enumerate(result.new_items)
            if json_value(item.raw_item) in expected
        ),
        None,
    )
    if start is None:
        raise ValueError("fork response not found in canonical SDK result")
    tail = result.new_items[start:]
    return SimpleNamespace(
        final_output=result.final_output,
        input=[],
        new_items=tail,
        raw_responses=result.raw_responses[snapshot["prefix_model_turns"] :],
        to_input_list=lambda: [item.to_input_item() for item in tail],
    )


class DecisionPair(InitialPair):
    """Freeze read-only prefix state before native tools commit the first action."""

    def __init__(self, group: Path, snapshot: dict[str, Any], agent: Any):
        super().__init__(group, snapshot)
        self.agent = agent
        self.turns = 0
        self.prefix_tools: list[str] = []
        self.request: dict[str, Any] | None = None
        self.frozen = False
        self.disabled = False

    def observe_request(self, request: dict[str, Any]) -> None:
        """Keep the last effective body only until the decision is frozen."""
        if not self.frozen:
            self.request = deepcopy(request)

    async def observe_response(self, response: Any, args: tuple, kwargs: dict) -> None:
        """Snapshot synchronously before Runner receives a mutating response."""
        if self.frozen or self.disabled:
            return
        output = json_value(response.output)
        names = [item["name"] for item in output if item.get("type") == "function_call"]
        try:
            if any(name in DESIGN_TOOLS for name in names):
                self._freeze(output, args, kwargs)
            elif not names or any(name not in PREFIX_TOOLS for name in names):
                raise ValueError(
                    "unsupported or terminal shared prefix before a design action"
                )
            else:
                self.prefix_tools.extend(names)
                self.turns += 1
        except Exception as exc:
            self.disabled = True
            write_json(
                self.group / "decision_boundary_failure.json",
                {
                    "error": str(exc),
                    "type": type(exc).__name__,
                    "prefix_tools": self.prefix_tools,
                },
            )
            self.fail("decision_boundary", exc)
            LOGGER.exception(
                "Direct decision capture failed; canonical response retained"
            )

    def _freeze(self, output: list[dict], args: tuple, kwargs: dict) -> None:
        agent = self.agent
        system = kwargs.get("system_instructions", args[0] if args else None)
        inputs = kwargs.get("input", args[1] if len(args) > 1 else None)
        if not isinstance(inputs, list) or not self.request:
            raise ValueError(
                "direct decision requires complete native SDK input and wire request"
            )
        if any(
            kwargs.get(key) is not None
            for key in ("previous_response_id", "conversation_id", "prompt")
        ):
            raise ValueError(
                "stored model conversations are unsupported for direct decisions"
            )
        if json_value(agent.scene.to_state_dict()) != self.snapshot["state"]:
            raise ValueError(
                "shared prefix changed scene state; unsupported snapshot boundary"
            )
        remaining = int(agent.cfg.agents.designer_agent.max_turns) - self.turns
        if remaining < 1:
            raise ValueError("no remaining native designer budget")
        snapshot = deepcopy(self.snapshot)
        snapshot.update(
            decision_boundary="first_mutation",
            input=json_value(inputs),
            system=system,
            prefix_model_turns=self.turns,
            prefix_tools=list(self.prefix_tools),
            remaining_max_turns=remaining,
            fork_response_items=output,
            boundary_selection="first_natural_A_design_response_before_tools;B_sampled_once",
            workflow_todos=json_value(
                workflow_owner(agent.designer.tools)._designer_todos
            ),
            safety_controller=json_value(vars(agent.furniture_safety_controller)),
            renderer={
                name: json_value(getattr(agent.rendering_manager, name))
                for name in snapshot["renderer"]
            },
            random_state=random_state(),
        )
        max_samples = int(os.environ.get("SCENEEXPERT_SHADOW_SCOPE_SAMPLES", "1"))
        from scenesmith.scene_expert.slow_memory.scope_sampling import sampling_policy
        policy = sampling_policy(max_samples)
        if max_samples > 1:
            snapshot["shadow_sampling"] = policy
            snapshot["boundary_selection"] = (
                "first_natural_A_design_response_before_tools;B_bounded_scope_sampling_v1"
            )
        count = int(os.environ.get("SCENEEXPERT_POOL_CANDIDATES", "2"))
        if count != 2:
            from scenesmith.scene_expert.slow_memory.candidate_pool import pool_policy

            completion_policy = os.environ.get("SCENEEXPERT_POOL_COMPLETION_POLICY", "all_four_v1")
            snapshot["candidate_pool"] = pool_policy(count, completion_policy)
            snapshot["boundary_selection"] += (
                ";fixed_four_candidate_pool_pair_local_v2" if completion_policy == "pair_local_v2"
                else ";fixed_four_candidate_pool_v1"
            )
        elif os.environ.get("SCENEEXPERT_POOL_COMPLETION_POLICY", "all_four_v1") != "all_four_v1":
            raise ValueError("pair-local completion requires four frozen candidates")
        snapshot["files"] = copy_scene_tree(
            Path(snapshot["source_scene_root"]), self.group / "input_scene"
        )
        write_json(self.group / "snapshot.json", snapshot)
        write_json(self.canonical_dir / "first_request.json", self.request)
        write_json(
            self.group / "decision_boundary.json",
            {
                "sdk_input_sha256": digest(snapshot["input"]),
                "wire_request_sha256": digest(self.request),
                "shared_tools": self.prefix_tools,
                "prefix_model_turns": self.turns,
                "selection": snapshot["boundary_selection"],
                "forced_tool_choice": False,
            },
        )
        self.snapshot, self.frozen = snapshot, True
        write_json(
            self.group / "status.json",
            {"status": "canonical_running", "canonical": "A"},
        )

    def capture_raw(self, agent: Any, result: Any, *, candidate: str = "A") -> None:
        """A missing fork cannot silently become a first-observation training row."""
        if not self.frozen:
            raise ValueError("no supported natural design decision captured")
        super().capture_raw(agent, result, candidate=candidate)


class DecisionModel(Model):
    """Delegate model execution unchanged and observe its native returned items."""

    def __init__(self, model: Model, observer: DecisionPair):
        self.model, self.observer = model, observer

    async def get_response(self, *args: Any, **kwargs: Any) -> Any:
        response = await self.model.get_response(*args, **kwargs)
        await self.observer.observe_response(response, args, kwargs)
        return response

    def stream_response(self, *args: Any, **kwargs: Any) -> Any:
        self.observer.disabled = True
        self.observer.fail(
            "decision_boundary", ValueError("streamed Designer capture is unsupported")
        )
        return self.model.stream_response(*args, **kwargs)


class DecisionProvider(ModelProvider):
    """Wrap only the provider scoped to this opt-in Designer call."""

    def __init__(self, provider: ModelProvider, observer: DecisionPair):
        self.provider, self.observer = provider, observer

    def get_model(self, model_name: str | None) -> Model:
        return DecisionModel(self.provider.get_model(model_name), self.observer)
