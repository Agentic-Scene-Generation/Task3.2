"""Failed candidates retain bounded diagnostics without becoming preference negatives."""

from __future__ import annotations

import hashlib
import sys
from types import ModuleType, SimpleNamespace

import pytest

from scenesmith.scene_expert.slow_memory.paired import (
    read_json,
    validate_tool_execution,
)


@pytest.mark.parametrize("long_output", [False, True])
def test_tool_failure_records_the_exact_match_and_call_identity(tmp_path, long_output):
    prefix = "x" * 10000 if long_output else ""
    suffix = "y" * 10000 if long_output else ""
    output = prefix + "ReadTimeout: request timed out" + suffix
    trace = {
        "tool_results": [
            {"tool_call_id": "valid", "output": "success"},
            {"tool_call_id": "failed-call", "output": output},
        ]
    }
    path = tmp_path / "B/tool_execution_failure.json"
    with pytest.raises(ValueError, match="infrastructure or unhandled-tool failure"):
        validate_tool_execution(trace, failure_path=path)
    row = read_json(path)
    assert row["tool_result_index"] == 1
    assert row["tool_call_id"] == "failed-call"
    assert row["matched_text"] == "ReadTimeout"
    assert row["output_sha256"] == hashlib.sha256(output.encode()).hexdigest()
    start, end = row["excerpt_char_range"]
    assert row["output_excerpt"] == output[start:end]
    assert row["output_char_count"] == len(output)
    assert len(row["output_excerpt"]) <= 4096
    assert row["excerpt_truncated"] is long_output
    assert row["preference_eligible"] is False


def test_asset_failure_is_recorded_before_quarantine(tmp_path):
    path = tmp_path / "A/tool_execution_failure.json"
    with pytest.raises(ValueError, match="asset infrastructure failed"):
        validate_tool_execution({}, "asset service unavailable", failure_path=path)
    row = read_json(path)
    assert row["reason"] == "fatal_asset_infrastructure"
    assert row["tool_result_index"] is None
    assert row["output_excerpt"] == "asset service unavailable"
    assert row["preference_eligible"] is False


def test_recoverable_design_failure_does_not_create_infrastructure_diagnostic(tmp_path):
    path = tmp_path / "tool_execution_failure.json"
    validate_tool_execution(
        {
            "tool_results": [
                {
                    "output": {
                        "success": False,
                        "message": "object would exceed room bounds",
                    }
                }
            ]
        },
        failure_path=path,
    )
    assert not path.exists()


@pytest.mark.parametrize("candidate", ["A", "B"])
def test_runtime_persists_failure_before_scoring_or_training_capture(
    tmp_path, monkeypatch, candidate
):
    from scenesmith.scene_expert.slow_memory import paired_runtime

    module = ModuleType("scenesmith.agent_utils.stage_working_memory")
    module._extract_agent_result_trace = lambda *args: {
        "tool_results": [{"tool_call_id": "bad", "output": "ConnectionError"}]
    }
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(
        paired_runtime, "verify_pair_code_provenance", lambda *a, **k: {}
    )
    pair = paired_runtime.InitialPair(tmp_path, {})
    # No scene state exists on this stub: the failure must be saved and raised
    # before raw-state scoring, trajectory capture or native safety can run.
    agent = SimpleNamespace(asset_manager=SimpleNamespace(_fatal_asset_error=None))
    with pytest.raises(ValueError, match="infrastructure or unhandled-tool failure"):
        pair.capture_raw(
            agent, SimpleNamespace(final_output="failed"), candidate=candidate
        )
    assert (
        read_json(tmp_path / candidate / "tool_execution_failure.json")["tool_call_id"]
        == "bad"
    )
    assert not (tmp_path / candidate / "raw_state.json").exists()
    assert not (tmp_path / candidate / "result.json").exists()
