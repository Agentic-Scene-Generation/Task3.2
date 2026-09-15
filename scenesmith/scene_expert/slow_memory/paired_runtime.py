"""One canonical initial Designer call and one isolated shadow execution.

This adapter supports only the first furniture decision with empty Designer and
Critic sessions. It scores raw outputs with the read-only Main deterministic
critic; native repair/rollback and all canonical continuation remain untouched.
"""

from __future__ import annotations

import asyncio
import dataclasses
import hashlib
import json
import logging
import os
import random
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from scenesmith.scene_expert.slow_memory.paired import (
    EXPECTED_MODEL,
    copy_scene_tree,
    deterministic_verdict,
    digest,
    read_json,
    rebase,
    reserve_group,
    tree_hashes,
    validate_tool_execution,
    write_json,
)
from scenesmith.scene_expert.slow_memory.paired_provenance import (
    verify_pair_code_provenance,
)
from scenesmith.scene_expert.slow_memory.paired_wire import capture_wire

LOGGER = logging.getLogger(__name__)
REPO = Path(__file__).resolve().parents[3]


def json_value(value: Any, *, field_path: str = "$") -> Any:
    """Serialize complete runtime data, never truncate unknown state silently."""
    from httpx import Timeout
    from omegaconf import OmegaConf

    if OmegaConf.is_config(value):
        return json_value(
            OmegaConf.to_container(value, resolve=True), field_path=field_path
        )
    if isinstance(value, Timeout):
        # Transport settings are part of the contract, not a model-request body.
        # B rebuilds its real Timeout via the native agent, then compares this
        # representation before execution. Preserve disabled (None) dimensions.
        return {
            "__type__": "httpx.Timeout",
            "values": json_value(value.as_dict(), field_path=field_path),
        }
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "model_dump"):
        return json_value(value.model_dump(mode="python"), field_path=field_path)
    if dataclasses.is_dataclass(value):
        return {
            field.name: json_value(
                getattr(value, field.name), field_path=f"{field_path}.{field.name}"
            )
            for field in dataclasses.fields(value)
        }
    if isinstance(value, dict):
        return {
            str(k): json_value(v, field_path=f"{field_path}.{k}")
            for k, v in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [
            json_value(v, field_path=f"{field_path}[{i}]") for i, v in enumerate(value)
        ]
    if isinstance(value, set):
        return sorted(json_value(v, field_path=field_path) for v in value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(
        f"unsupported initial snapshot value at {field_path}: {type(value).__name__}"
    )


def snapshot_codec_preflight() -> dict[str, Any]:
    """Check the installed SDK's snapshot values without models, services or GPU."""
    from importlib.metadata import version

    from agents import ModelSettings
    from agents.model_settings import Reasoning
    from httpx import Timeout

    settings = ModelSettings(
        extra_args={"timeout": Timeout(connect=10.0, read=600, write=600, pool=None)},
        reasoning=Reasoning(effort="high"),
        max_tokens=1024,
    )
    serialized = json_value(settings, field_path="model_settings")
    if json.loads(json.dumps(serialized)) != serialized:
        raise ValueError("model settings snapshot did not round-trip through JSON")
    timeout = serialized["extra_args"]["timeout"]
    if timeout != {
        "__type__": "httpx.Timeout",
        "values": settings.extra_args["timeout"].as_dict(),
    }:
        raise ValueError("model settings timeout contract was not preserved")
    return {
        "schema_version": "sceneexpert.snapshot_codec_preflight.v1",
        "status": "passed",
        "openai_agents_version": version("openai-agents"),
        "httpx_version": version("httpx"),
        "timeout_contract": timeout,
        "scope": "snapshot_serialization_only",
    }


def validate_model_settings(settings: Any, expected: Any) -> None:
    """Reject changes to sampling or transport settings before the shadow runs."""
    if json_value(settings, field_path="model_settings") != expected:
        raise ValueError("reconstructed model settings differ")


def tool_schemas(tools: list[Any]) -> list[dict[str, Any]]:
    """Capture declarative tool contracts without serializing closures/clients."""
    return [
        {
            "name": tool.name,
            "description": tool.description,
            "parameters": json_value(tool.params_json_schema),
            "strict": tool.strict_json_schema,
        }
        for tool in tools
    ]


def random_state() -> dict[str, Any]:
    """Freeze local tool randomness separately from remote model sampling."""
    import numpy as np

    algorithm, keys, position, has_gauss, cached = np.random.get_state()
    return {
        "python": json_value(random.getstate()),
        "numpy": [algorithm, keys.tolist(), position, has_gauss, cached],
    }


def restore_random_state(state: dict[str, Any]) -> None:
    import numpy as np

    version, keys, gaussian = state["python"]
    random.setstate((version, tuple(keys), gaussian))
    algorithm, keys, position, has_gauss, cached = state["numpy"]
    np.random.set_state(
        (algorithm, np.array(keys, dtype=np.uint32), position, has_gauss, cached)
    )


async def open_initial_pair(agent: Any, input_message: Any) -> "InitialPair | None":
    """Snapshot a supported first decision before canonical execution starts."""
    root_value = os.environ.get("SCENEEXPERT_INITIAL_PAIRS_DIR", "")
    if not root_value or agent.agent_type.value != "furniture":
        return None
    if getattr(agent, "_initial_pair_seen", False):
        return None
    agent._initial_pair_seen = True
    if os.environ.get("SCENEEXPERT_INITIAL_PAIR_SHADOW") == "1":
        return None
    if str(agent.cfg.openai.model) != EXPECTED_MODEL:
        raise ValueError("initial pairs require the pinned Qwen3.8 model")
    if not getattr(agent.scene, "scene_expert_slow_memory_capture_enabled", False):
        raise ValueError("initial pairs require Full trajectory capture")
    if (
        await agent.designer_session.get_items()
        or await agent.critic_session.get_items()
    ):
        raise ValueError(
            "initial pilot does not support nonempty Designer/Critic history"
        )
    if agent._stage_execution_attempt != 1:
        raise ValueError("initial pilot does not support stage retries")
    # Reconstructing arbitrary pre-used tool caches is deliberately unsupported.
    if any(
        getattr(obj.object_type, "value", "") == "furniture"
        for obj in agent.scene.objects.values()
    ):
        raise ValueError("initial pilot requires an unfurnished starting scene")
    scene_dir = Path(agent.scene.scene_dir).resolve()
    scene_root = scene_dir.parent if scene_dir.name.startswith("room_") else scene_dir
    root = Path(root_value).resolve()
    if root.is_relative_to(scene_root):
        raise ValueError("pair root must be outside the canonical scene")
    group = reserve_group(
        root, str(scene_dir), int(os.environ.get("SCENEEXPERT_PAIR_MAX_GROUPS", "2"))
    )
    if group is None:
        return None
    write_json(group / "status.json", {"status": "preparing", "canonical": "A"})
    phase = "snapshot_serialization"
    try:
        from scenesmith.utils.openai import (
            reasoning_persistence_enabled,
            reasoning_persistence_provider,
        )

        attrs = {
            key: json_value(value)
            for key, value in vars(agent.scene).items()
            if key.startswith("scene_expert_")
            or key == "scenebenchmark_intent_contract"
        }
        ctor = {
            name: getattr(agent, name)
            for name in (
                "geometry_server_host",
                "geometry_server_port",
                "hssd_server_host",
                "hssd_server_port",
            )
        }
        ctor["render_gpu_id"] = getattr(agent.blender_server, "_gpu_id", None)
        for prefix, attribute in (
            ("articulated", "articulated_client"),
            ("materials", "materials_client"),
        ):
            client = getattr(agent.asset_manager, attribute, None)
            if client is not None:
                parsed = urlsplit(client.base_url)
                ctor[f"{prefix}_server_host"] = parsed.hostname
                ctor[f"{prefix}_server_port"] = parsed.port
        snapshot = {
            "schema_version": "sceneexpert.initial_snapshot.v2",
            "code_provenance": verify_pair_code_provenance(repo_root=REPO),
            "model": EXPECTED_MODEL,
            "source_scene_root": str(scene_root),
            "room_relative": scene_dir.relative_to(scene_root).as_posix(),
            "room_id": agent.scene.room_id,
            "state": json_value(agent.scene.to_state_dict()),
            "scene_attributes": attrs,
            "cfg": json_value(agent.cfg),
            "constructor": ctor,
            "input": json_value(input_message),
            "system": json_value(agent.designer.instructions),
            "model_settings": json_value(
                agent.designer.model_settings, field_path="model_settings"
            ),
            "tools": tool_schemas(agent.designer.tools),
            "designer_history": [],
            "critic_history": [],
            "active_noise_profile": json_value(
                agent.furniture_tools.active_noise_profile
            ),
            "safety_controller": json_value(vars(agent.furniture_safety_controller)),
            "renderer": {
                name: json_value(getattr(agent.rendering_manager, name))
                for name in (
                    "_render_counter",
                    "_render_cache",
                    "_last_render_dir",
                    "_active_render_profile",
                )
            },
            "random_state": random_state(),
            "reasoning_persistence": {
                "enabled": reasoning_persistence_enabled(),
                "provider": reasoning_persistence_provider(),
            },
            "placement_order_reference": agent._placement_order_reference,
            "placement_style": agent.placement_style,
            "context_image_path": (
                str(agent.context_image_path) if agent.context_image_path else None
            ),
            "house_layout": (
                agent.house_layout.to_dict(scene_dir=scene_root)
                if agent.house_layout
                else None
            ),
        }
        # Validate all runtime values before spending time copying native assets.
        snapshot = json_value(snapshot, field_path="snapshot")
        phase = "snapshot_assets"
        snapshot["files"] = copy_scene_tree(scene_root, group / "input_scene")
        phase = "snapshot_commit"
        write_json(group / "snapshot.json", snapshot)
        (group / "A").mkdir()
        write_json(
            group / "status.json", {"status": "canonical_running", "canonical": "A"}
        )
        return InitialPair(group, snapshot)
    except Exception as exc:
        write_json(
            group / "status.json",
            {
                "status": "failed",
                "phase": phase,
                "error_type": type(exc).__name__,
                "error": str(exc),
            },
        )
        raise


class InitialPair:
    """Own one snapshot and candidate-specific artifacts, never policy selection."""

    def __init__(self, group: Path, snapshot: dict[str, Any]):
        self.group, self.snapshot = group, snapshot

    @property
    def canonical_dir(self) -> Path:
        return self.group / "A"

    def fail(self, phase: str, error: BaseException) -> None:
        """Persist a failed collection attempt without altering canonical policy."""
        try:
            write_json(
                self.group / "status.json",
                {
                    "status": "failed",
                    "phase": phase,
                    "error": f"{type(error).__name__}: {error}",
                    "canonical": "A",
                },
            )
        except OSError:
            LOGGER.exception(
                "Could not persist pair failure; incomplete group remains ineligible"
            )

    def capture_returned(self, agent: Any, message: str, candidate: str) -> None:
        """Commit completion only after native safety and its returned state exist."""
        verify_pair_code_provenance(
            self.snapshot.get("code_provenance", {}), repo_root=REPO
        )
        directory = self.group / candidate
        state = json_value(agent.scene.to_state_dict())
        write_json(directory / "returned_state.json", state)
        write_json(directory / "safety.json", {"message": message})
        result = read_json(directory / "result.json")
        result.update(
            status="completed",
            returned_state_hash=digest(state),
            safety_hash=digest({"message": message}),
        )
        write_json(directory / "result.json", result)

    def capture_raw(self, agent: Any, result: Any, *, candidate: str = "A") -> None:
        """Score exactly the post-Runner state before native end-of-call safety."""
        verify_pair_code_provenance(
            self.snapshot.get("code_provenance", {}), repo_root=REPO
        )
        from scenesmith.agent_utils.stage_working_memory import (
            _extract_agent_result_trace,
        )
        from scenesmith.scene_expert.slow_memory.paired_scoring import (
            SCORING_PROTOCOL,
            save_scoring_proof,
            score_raw_candidate,
        )
        from scenesmith.scene_expert.schemas import SceneTaskSpec
        from scenesmith.scene_expert.slow_memory.schemas import (
            PreferenceEvidence,
            TrajectoryOutcome,
        )
        from scenesmith.scene_expert.slow_memory.trajectory import TrajectoryCollector

        directory = self.group / candidate
        agent_trace = _extract_agent_result_trace(result, result.final_output or "")
        validate_tool_execution(
            agent_trace,
            getattr(agent.asset_manager, "_fatal_asset_error", None),
            failure_path=directory / "tool_execution_failure.json",
        )
        state = json_value(agent.scene.to_state_dict())
        raw_hash = digest(state)
        write_json(directory / "raw_state.json", state)
        report, scoring_proof = score_raw_candidate(agent.scene, agent.cfg)
        evaluation_hash = digest(json_value(agent.scene.to_state_dict()))
        if raw_hash != evaluation_hash:
            raise ValueError("candidate evaluator changed the state being scored")
        write_json(directory / "report.json", report)
        verdict, score, failures = deterministic_verdict(report)
        wire = read_json(directory / "first_request.json")
        payload = {
            "schema_version": "2.0",
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "stage": "furniture",
            "agent_role": "designer",
            "event": "request_initial_design",
            "prompt": self.snapshot["input"],
            "conversation_messages": wire["messages"],
            "tools": wire.get("tools", []),
            "image_refs": [],
            "context_snapshot": {"initial_snapshot_sha256": digest(self.snapshot)},
            "output": result.final_output or "",
            "error": "",
            "agent_trace": agent_trace,
            "capture_policy": "independently_scored_raw_initial_candidate",
        }
        attrs = self.snapshot["scene_attributes"]
        task_spec = attrs.get("scene_expert_task_spec") or self.snapshot["state"].get(
            "metadata", {}
        ).get("scene_expert_task_spec", {})
        prompt = (
            attrs.get("scene_expert_original_description")
            or self.snapshot["state"]["text_description"]
        )
        collector = TrajectoryCollector(
            scene_debug_dir=directory,
            prompt=prompt,
            scene_id=str(self.snapshot["room_id"]),
            run_id=f"{self.group.name}/{candidate}",
            task_spec=SceneTaskSpec.model_validate(task_spec),
            model_id=EXPECTED_MODEL,
            config_hash=digest(self.snapshot["cfg"]),
            code_provenance=self.snapshot["code_provenance"],
            max_prompt_chars=4 * 1024**2,
            max_response_chars=4 * 1024**2,
        )
        collector.capture_candidate(
            payload=payload,
            evidence=PreferenceEvidence(
                evidence_id=f"{self.group.name}_{candidate}",
                kind="deterministic",
                verdict=verdict,
                source="main_raw_candidate_deterministic_checks",
                authoritative=True,
                quality_score=score,
                report_ref="report.json",
                details={"raw_state_sha256": raw_hash, "candidate": candidate},
            ),
            outcome=TrajectoryOutcome(
                execution_complete=True,
                tool_call_valid=True,
                hard_passed=failures == 0,
                hard_violation_count=failures,
                deterministic_score=score,
                causal_link_verified=True,
                evidence_refs=["report.json", "raw_state.json"],
            ),
            scene_state_path="raw_state.json",
        )
        # Retain raw asset bytes for both sides before native safety can mutate
        # either candidate. The snapshot is independently portable for review.
        source_root = (
            Path(self.snapshot["source_scene_root"])
            if candidate == "A"
            else directory / "scene"
        )
        raw_files = copy_scene_tree(source_root, directory / "raw_scene")
        save_scoring_proof(directory, report, scoring_proof, raw_files)
        write_json(
            directory / "result.json",
            {
                "status": "raw_captured",
                "scoring_protocol": SCORING_PROTOCOL,
                "candidate": candidate,
                "model": wire["model"],
                "snapshot_hash": digest(self.snapshot),
                "raw_state_hash": raw_hash,
                "evaluation_state_hash": evaluation_hash,
                "verdict": verdict,
                "score": score,
                "raw_files": raw_files,
                "trajectory_sha256": hashlib.sha256(
                    collector.trajectory_path.read_bytes()
                ).hexdigest(),
            },
        )

    async def finish(self, agent: Any, safety_message: str) -> None:
        """Journal canonical safety effects, then execute B without touching A."""
        self.capture_returned(agent, safety_message, "A")
        before = digest(json_value(agent.scene.to_state_dict()))
        files_before = canonical_asset_hashes(Path(self.snapshot["source_scene_root"]))
        bank = Path(
            os.environ.get(
                "SCENEEXPERT_ACTIVE_MEMORY_BANK_DIR", str(self.group / "no_bank")
            )
        )
        bank_before = tree_hashes(bank) if bank.is_dir() else {}
        write_json(
            self.group / "status.json", {"status": "shadow_running", "canonical": "A"}
        )
        with (self.group / "shadow.log").open("w", encoding="utf-8") as log:
            process = await asyncio.create_subprocess_exec(
                sys.executable,
                str(REPO / "scripts/collect_sceneexpert_initial_pairs.py"),
                "--worker-group",
                str(self.group),
                cwd=REPO,
                env=shadow_environment(self.group),
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=(os.name == "posix"),
            )
            try:
                code = await asyncio.wait_for(
                    process.wait(),
                    timeout=int(os.environ.get("SCENEEXPERT_PAIR_TIMEOUT", "3600")),
                )
            except BaseException:
                if process.returncode is None:
                    process.terminate()
                    try:
                        await asyncio.wait_for(process.wait(), timeout=30)
                    except asyncio.TimeoutError:
                        process.kill()
                        await process.wait()
                # Only the private worker's process group, never shared services.
                if os.name == "posix":
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                raise
        after = digest(json_value(agent.scene.to_state_dict()))
        files_after = canonical_asset_hashes(Path(self.snapshot["source_scene_root"]))
        bank_after = tree_hashes(bank) if bank.is_dir() else {}
        write_json(
            self.group / "continuation_proof.json",
            {
                "canonical_before": before,
                "canonical_after": after,
                "assets_before": digest(files_before),
                "assets_after": digest(files_after),
                "memory_before": digest(bank_before),
                "memory_after": digest(bank_after),
            },
        )
        if code and os.name == "posix":
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        if after != before or bank_after != bank_before or files_after != files_before:
            raise ValueError("shadow execution changed canonical scene")
        write_json(
            self.group / "status.json",
            {
                "status": "completed" if code == 0 else "shadow_failed",
                "shadow_exit": code,
                "canonical": "A",
            },
        )
        if code:
            LOGGER.error(
                "Initial-pair shadow failed; canonical scene continues; see %s",
                self.group / "shadow.log",
            )


def canonical_asset_hashes(root: Path) -> dict[str, str]:
    """Fingerprint writable scene artifacts, excluding live logging/session files."""
    return {
        name: value
        for name, value in tree_hashes(root).items()
        if not (
            name.endswith((".log", ".db", ".db-wal", ".db-shm", ".lock"))
            or "__pycache__" in Path(name).parts
        )
    }


def shadow_environment(group: Path) -> dict[str, str]:
    """Disable public bank writes and external audit paths in the child only."""
    env = dict(os.environ)
    for key in list(env):
        if key.startswith("SCENEEXPERT_") and (
            "DEBUG" in key or "TIMING" in key or "MEMORY" in key
        ):
            env.pop(key)
    env.pop("SCENEBENCHMARK_CRITIC_TIMING_PATH", None)
    env["SCENEEXPERT_INITIAL_PAIR_SHADOW"] = "1"
    env["SCENEEXPERT_ACTIVE_MEMORY_BANK_DIR"] = str(group / "B/private_memory")
    env["SCENEEXPERT_ACTIVE_MEMORY_BANK_READ_ONLY"] = "true"
    env["SCENEEXPERT_COMPONENT_MEMORY_WRITER_ENABLED"] = "false"
    return env


async def run_shadow(group: Path) -> None:
    """Reconstruct fresh clients, tools, renderer and empty sessions for B."""
    snapshot = read_json(group / "snapshot.json")
    verify_pair_code_provenance(snapshot.get("code_provenance", {}), repo_root=REPO)
    if tree_hashes(group / "input_scene") != snapshot["files"]:
        raise ValueError("snapshot files changed; refuse shadow replay")

    from agents import Runner
    from omegaconf import OmegaConf
    from scenesmith.agent_utils.house import HouseLayout, RoomGeometry
    from scenesmith.agent_utils.room import RoomScene
    from scenesmith.furniture_agents.stateful_furniture_agent import (
        StatefulFurnitureAgent,
    )
    from scenesmith.utils.logging import ConsoleLogger
    from scenesmith.utils.openai import configure_reasoning_persistence

    destination = group / "B"
    if (destination / "result.json").exists():
        if (
            read_json(destination / "result.json").get("status") == "completed"
            and (destination / "returned_state.json").exists()
            and (destination / "safety.json").exists()
        ):
            return  # Never execute an already completed candidate twice.
    destination.mkdir(exist_ok=True)
    private_root = destination / "scene"
    # Preserve any interrupted execution; a fresh group is required for retry.
    copy_scene_tree(group / "input_scene", private_root)
    mapped = rebase(snapshot, snapshot["source_scene_root"], str(private_root))
    room_dir = private_root / snapshot["room_relative"]
    room_dir.mkdir(parents=True, exist_ok=True)
    state = mapped["state"]
    scene = RoomScene(
        room_geometry=RoomGeometry.from_dict(
            state["room_geometry"], scene_dir=room_dir
        ),
        scene_dir=room_dir,
        room_id=snapshot["room_id"],
        text_description=state["text_description"],
        action_log_path=room_dir / "action_log.json",
        floor_plan_mode=state["floor_plan_mode"],
        tool_schema_version=state["tool_schema_version"],
    )
    scene.restore_from_state_dict(state)
    for key, value in mapped["scene_attributes"].items():
        setattr(scene, key, value)
    # Whole executable state is included in pairing, rather than a bounded summary.
    restored = rebase(
        json_value(scene.to_state_dict()),
        str(private_root),
        snapshot["source_scene_root"],
    )
    if restored != snapshot["state"]:
        raise ValueError("restored scene differs from initial snapshot")
    house = (
        HouseLayout.from_dict(mapped["house_layout"], house_dir=private_root)
        if mapped["house_layout"]
        else None
    )
    configure_reasoning_persistence(**snapshot["reasoning_persistence"])
    agent = StatefulFurnitureAgent(
        cfg=OmegaConf.create(mapped["cfg"]),
        logger=ConsoleLogger(room_dir),
        house_layout=house,
        **snapshot["constructor"],
    )
    try:
        agent.scene = scene
        agent._configure_furniture_safety_for_scene(
            getattr(scene, "scene_expert_original_description", scene.text_description)
        )
        agent._synchronize_task_required_counts()
        agent._placement_order_reference = snapshot["placement_order_reference"]
        agent.placement_style = snapshot["placement_style"]
        agent.context_image_path = (
            Path(mapped["context_image_path"]) if mapped["context_image_path"] else None
        )
        agent.designer = agent._create_designer_agent(
            tools=agent._create_designer_tools()
        )
        agent.furniture_tools.active_noise_profile = OmegaConf.create(
            snapshot["active_noise_profile"]
        )
        if tool_schemas(agent.designer.tools) != snapshot["tools"]:
            raise ValueError("reconstructed tool schemas differ")
        validate_model_settings(
            agent.designer.model_settings, snapshot["model_settings"]
        )
        agent.designer.instructions = snapshot["system"]
        for name, value in mapped["renderer"].items():
            if name == "_render_cache":
                value = {key: Path(path) for key, path in value.items()}
            elif name == "_last_render_dir" and value:
                value = Path(value)
            setattr(agent.rendering_manager, name, value)
        if (
            await agent.designer_session.get_items()
            or await agent.critic_session.get_items()
        ):
            raise ValueError("shadow sessions must be empty")
        transaction = agent._begin_furniture_design_transaction(call_kind="initial")
        restored_controller = rebase(
            json_value(vars(agent.furniture_safety_controller)),
            str(private_root),
            snapshot["source_scene_root"],
        )
        if restored_controller != snapshot["safety_controller"]:
            raise ValueError("restored safety-controller state differs")
        restore_random_state(snapshot["random_state"])
        with capture_wire(destination):
            async with agent._reasoning_persistence_context_for_session(
                agent.designer_session
            ):
                result = await Runner.run(
                    starting_agent=agent.designer,
                    input=snapshot["input"],
                    session=agent.designer_session,
                    max_turns=agent.cfg.agents.designer_agent.max_turns,
                    run_config=agent._create_run_config(),
                )
        pair = InitialPair(group, snapshot)
        pair.capture_raw(agent, result, candidate="B")
        message = agent._end_furniture_design_transaction(transaction)
        pair.capture_returned(agent, message, "B")
    finally:
        agent.cleanup()
