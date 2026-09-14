"""Real SDK settings at the initial snapshot boundary, without scene servers."""

from __future__ import annotations

import ast
import asyncio
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest
from agents import ModelSettings
from httpx import Timeout
from omegaconf import OmegaConf

from scenesmith.agent_utils.thinking import chat_template_kwargs_from_effort
from scenesmith.scene_expert.slow_memory import paired_runtime
from scenesmith.scene_expert.slow_memory.paired import (
    EXPECTED_MODEL,
    audit_pairs,
    read_json,
)

ROOT = Path(__file__).resolve().parents[2]


def _native_settings(cfg: Any) -> ModelSettings:
    # Execute the repository's real factory with real SDK/HTTPX classes. Only
    # constructing the full native agent would require Drake and GPU services.
    source = ast.parse(
        (ROOT / "scenesmith/agent_utils/base_stateful_agent.py").read_text(
            encoding="utf-8"
        )
    )
    method = next(
        n
        for n in ast.walk(source)
        if isinstance(n, ast.FunctionDef) and n.name == "_get_model_settings"
    )
    namespace = {
        "ModelSettings": ModelSettings,
        "Timeout": Timeout,
        "chat_template_kwargs_from_effort": chat_template_kwargs_from_effort,
    }
    exec(  # noqa: S102 - Compile only the repository's production method for this test.
        compile(ast.Module(body=[method], type_ignores=[]), "native_settings", "exec"),
        namespace,
    )
    agent = SimpleNamespace(
        cfg=cfg,
        _reasoning_request_provider=lambda: "qwen",
        _role_max_output_tokens=lambda key: 16384,
    )
    return namespace["_get_model_settings"](agent, settings_key="designer")


def _agent(tmp_path: Path) -> SimpleNamespace:
    defaults = OmegaConf.load(
        ROOT / "configurations/furniture_agent/base_furniture_agent.yaml"
    )
    # Use the native timeout defaults; unrelated asset-path interpolations belong
    # to Hydra's full project config and are not needed by this server-free fixture.
    cfg = OmegaConf.create(
        {
            "api_timeout": OmegaConf.to_container(defaults.api_timeout),
            "openai": {
                "model": EXPECTED_MODEL,
                "reasoning_effort": {"designer": "high"},
            },
        }
    )
    source = tmp_path / "canonical"
    source.mkdir()
    (source / "geometry.sdf").write_text("room geometry", encoding="utf-8")
    scene = SimpleNamespace(
        scene_dir=source,
        room_id="bedroom",
        objects={},
        scene_expert_slow_memory_capture_enabled=True,
        to_state_dict=lambda: {"text_description": "Place a bed", "objects": {}},
    )
    return SimpleNamespace(
        cfg=cfg,
        scene=scene,
        agent_type=SimpleNamespace(value="furniture"),
        designer_session=SimpleNamespace(get_items=AsyncMock(return_value=[])),
        critic_session=SimpleNamespace(get_items=AsyncMock(return_value=[])),
        _stage_execution_attempt=1,
        geometry_server_host="localhost",
        geometry_server_port=8000,
        hssd_server_host="localhost",
        hssd_server_port=8001,
        blender_server=SimpleNamespace(_gpu_id=0),
        asset_manager=SimpleNamespace(),
        designer=SimpleNamespace(
            instructions="Build the room",
            model_settings=_native_settings(cfg),
            tools=[],
        ),
        furniture_tools=SimpleNamespace(
            active_noise_profile=OmegaConf.create({"noise": 0.1})
        ),
        furniture_safety_controller=SimpleNamespace(
            required_terms={"bed"}, required_counts={"bed": 1}
        ),
        rendering_manager=SimpleNamespace(
            _render_counter=0,
            _render_cache={},
            _last_render_dir=None,
            _active_render_profile="final",
        ),
        _placement_order_reference=[],
        placement_style="natural",
        context_image_path=None,
        house_layout=None,
    )


def test_native_sdk_model_settings_commit_a_complete_initial_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "pairs"
    monkeypatch.setenv("SCENEEXPERT_INITIAL_PAIRS_DIR", str(root))
    monkeypatch.delenv("SCENEEXPERT_INITIAL_PAIR_SHADOW", raising=False)
    monkeypatch.delenv("SCENEEXPERT_PAIR_SOURCE_HASH", raising=False)
    agent = _agent(tmp_path)
    original = agent.designer.model_settings.extra_args["timeout"]
    pair = asyncio.run(paired_runtime.open_initial_pair(agent, "Place a bed"))
    snapshot = read_json(pair.group / "snapshot.json")
    assert snapshot["model_settings"]["extra_args"]["timeout"] == {
        "__type__": "httpx.Timeout",
        "values": {"connect": 10.0, "read": 600, "write": 600, "pool": 600},
    }
    assert (pair.group / "input_scene/geometry.sdf").read_text() == "room geometry"
    assert (pair.group / "A").is_dir()
    assert read_json(pair.group / "status.json")["status"] == "canonical_running"
    paired_runtime.validate_model_settings(
        _native_settings(OmegaConf.create(snapshot["cfg"])), snapshot["model_settings"]
    )
    assert (
        isinstance(original, Timeout)
        and agent.designer.model_settings.extra_args["timeout"] is original
    )


@pytest.mark.parametrize("dimension", ["connect", "read", "write", "pool"])
def test_shadow_gate_rejects_changed_timeout_dimensions(dimension: str) -> None:
    settings = ModelSettings(
        extra_args={"timeout": Timeout(connect=10, read=600, write=600, pool=None)}
    )
    expected = json.loads(json.dumps(paired_runtime.json_value(settings)))
    assert expected["extra_args"]["timeout"]["values"]["pool"] is None
    paired_runtime.validate_model_settings(settings, expected)
    setattr(settings.extra_args["timeout"], dimension, 7)
    with pytest.raises(ValueError, match="reconstructed model settings differ"):
        paired_runtime.validate_model_settings(settings, expected)


def test_unknown_sdk_value_fails_before_asset_copy_and_audit_keeps_root_cause(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "pairs"
    monkeypatch.setenv("SCENEEXPERT_INITIAL_PAIRS_DIR", str(root))
    monkeypatch.delenv("SCENEEXPERT_INITIAL_PAIR_SHADOW", raising=False)
    monkeypatch.delenv("SCENEEXPERT_PAIR_SOURCE_HASH", raising=False)
    agent = _agent(tmp_path)
    agent.designer.model_settings.extra_args["unsupported_client"] = object()
    with pytest.raises(TypeError, match="model_settings.extra_args.unsupported_client"):
        asyncio.run(paired_runtime.open_initial_pair(agent, "Place a bed"))
    group = root / "group_000"
    status = read_json(group / "status.json")
    assert status["phase"] == "snapshot_serialization"
    assert status["error_type"] == "TypeError"
    assert not (group / "input_scene").exists()
    assert not (group / "snapshot.json").exists()
    audit = audit_pairs(root, expected_groups=1)
    assert audit["candidate_count"] == 0 and not audit["gate_passed"]
    assert any(
        "unsupported_client" in error and "snapshot_serialization" in error
        for error in audit["errors"]
    )


def test_codec_preflight_cli_creates_report_without_native_imports(
    tmp_path: Path,
) -> None:
    report = tmp_path / "preflight.json"
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/collect_sceneexpert_initial_pairs.py"),
            "--preflight",
            "--preflight-report",
            str(report),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert read_json(report)["status"] == "passed"
    assert read_json(report)["timeout_contract"]["values"]["pool"] is None
