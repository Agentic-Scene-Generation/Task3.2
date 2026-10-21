"""Production admission and runtime regressions for the default-off v2 pool."""

from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scenesmith.scene_expert.slow_memory.candidate_pool import (
    is_pair_local,
    pool_policy,
)
from scenesmith.scene_expert.slow_memory.paired import (
    audit_pairs,
    digest,
    read_json,
    validate_pair_inputs,
    write_json,
)
from scenesmith.scene_expert.slow_memory.paired_runtime import InitialPair
from tests.unit.test_fixed_candidate_pool import four_group


def local_group(root: Path, states: dict[str, str] | None = None) -> Path:
    group = four_group(root)
    snapshot = read_json(group / "snapshot.json")
    source_hashes = {"fixture.py": "mechanism-only"}
    provenance = {
        "identity_kind": "source_bundle_sha256",
        "source_hashes": source_hashes,
        "source_bundle_hash": hashlib.sha256(
            json.dumps(source_hashes, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
    }
    snapshot.update(
        candidate_pool=pool_policy(4, "pair_local_v2"),
        input=[{"role": "user", "content": "Place a bed"}],
        prefix_model_turns=0,
        remaining_max_turns=40,
        cfg={"agents": {"designer_agent": {"max_turns": 40}}},
        boundary_selection="fixture-first-design",
        code_provenance=provenance,
    )
    write_json(group / "snapshot.json", snapshot)
    request = read_json(group / "A/first_request.json")
    write_json(
        group / "decision_boundary.json",
        {
            "wire_request_sha256": digest(request),
            "sdk_input_sha256": digest(snapshot["input"]),
            "prefix_model_turns": 0,
            "forced_tool_choice": False,
            "selection": snapshot["boundary_selection"],
        },
    )
    states = states or dict.fromkeys("ABCD", "completed")
    slots, exits = {}, {}
    for name in "ABCD":
        code = None if name == "A" else (0 if states[name] == "completed" else 2)
        slots[name] = {"state": states[name], "exit_code": code}
        if states[name] != "completed":
            slots[name]["reason"] = (
                "native candidate could not complete; never a negative"
            )
        if name != "A":
            exits[name] = code
            write_json(
                group / name / "worker_exit.json",
                {"exit_code": code, "process_started": True, "error": None},
            )
        path = group / name / "slow_memory/trajectories.jsonl"
        row = json.loads(path.read_bytes())
        row["context_hash"] = digest(snapshot)
        row["spatial_context"]["initial_snapshot_sha256"] = digest(snapshot)
        row["provenance"]["code_provenance"] = provenance
        row["completion_messages"] = [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": name,
                        "type": "function",
                        "function": {
                            "name": "generate_assets",
                            "arguments": json.dumps({"candidate": name}),
                        },
                    }
                ],
            }
        ]
        path.write_text(json.dumps(row) + "\n", encoding="utf-8")
        result = read_json(group / name / "result.json")
        result.update(
            snapshot_hash=digest(snapshot),
            trajectory_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        )
        write_json(group / name / "result.json", result)
    complete = all(value == "completed" for value in states.values())
    write_json(
        group / "candidate_pool_execution.json",
        {
            "policy": snapshot["candidate_pool"],
            "scheduled": ["B", "C", "D"],
            "worker_exits": exits,
            "slots": slots,
            "terminal": True,
            "pool_complete": complete,
        },
    )
    write_json(
        group / "status.json",
        {
            "status": "completed" if complete else "pool_failed",
            "pool_complete": complete,
        },
    )
    return group


def test_default_policy_remains_v1():
    assert pool_policy(4)["protocol"] == "sceneexpert.fixed_candidate_pool.v1"
    assert not is_pair_local({})
    assert not is_pair_local(
        {"decision_boundary": "first_mutation", "candidate_pool": pool_policy(4)}
    )


def test_partial_A_preserves_failure_and_exports_only_complete_direct_candidates(
    tmp_path,
):
    group = local_group(
        tmp_path, {"A": "failed", "B": "completed", "C": "completed", "D": "completed"}
    )
    (group / "A/result.json").unlink()
    (group / "A/slow_memory/trajectories.jsonl").unlink()
    sources, groups, errors = validate_pair_inputs(tmp_path)
    assert not errors and {path.parent.parent.name for path in sources} == {
        "B",
        "C",
        "D",
    }
    assert groups[0]["pool_complete"] is False
    assert groups[0]["pair_candidate_pool_eligible"] is True
    result = audit_pairs(tmp_path)
    assert result["eligible_pair_count"] == 1
    assert result["execution_integrity_passed"] is False
    assert result["gate_passed"] is False
    assert "candidate_pool_execution_failed" in result["errors"]


@pytest.mark.parametrize(
    "change",
    [
        "request_failed",
        "budget",
        "snapshot",
        "proof_failed",
        "slot_exit",
        "slot_omitted",
        "trajectory_source",
        "unknown_complete",
        "source_hash",
    ],
)
def test_shared_drift_or_invalid_completed_evidence_rejects_entire_group(
    tmp_path, change
):
    group = local_group(
        tmp_path, {"A": "completed", "B": "completed", "C": "failed", "D": "completed"}
    )
    if change == "request_failed":
        write_json(group / "C/first_request.json", {"model": "different"})
    elif change in {"budget", "source_hash"}:
        value = read_json(group / "snapshot.json")
        if change == "budget":
            value["remaining_max_turns"] += 1
        else:
            value["code_provenance"]["source_bundle_hash"] = "changed"
        write_json(group / "snapshot.json", value)
    elif change == "snapshot":
        (group / "input_scene/geometry.sdf").write_text("changed")
    elif change == "proof_failed":
        proof = read_json(group / "continuation_proof_C.json")
        proof["assets_after"] = "changed"
        write_json(group / "continuation_proof_C.json", proof)
    elif change in {"slot_exit", "slot_omitted"}:
        execution = read_json(group / "candidate_pool_execution.json")
        if change == "slot_exit":
            execution["worker_exits"]["C"] = 0
        else:
            del execution["slots"]["C"]
        write_json(group / "candidate_pool_execution.json", execution)
    elif change == "trajectory_source":
        path = group / "B/slow_memory/trajectories.jsonl"
        row = json.loads(path.read_bytes())
        row["provenance"]["code_provenance"] = {}
        path.write_text(json.dumps(row) + "\n", encoding="utf-8")
        result = read_json(group / "B/result.json")
        result["trajectory_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        write_json(group / "B/result.json", result)
    else:
        report = read_json(group / "B/report.json")
        report["summary"]["scene_summary"]["unknown"] = 1
        write_json(group / "B/report.json", report)
    sources, groups, errors = validate_pair_inputs(tmp_path)
    assert not sources and errors and not groups[0]["valid"]


def test_non_direct_candidate_is_excluded_before_native_pair_selection(tmp_path):
    group = local_group(tmp_path)
    path = group / "B/slow_memory/trajectories.jsonl"
    row = json.loads(path.read_bytes())
    row["completion_messages"] = [{"role": "assistant", "content": "observe only"}]
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    result = read_json(group / "B/result.json")
    result["trajectory_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    write_json(group / "B/result.json", result)
    sources, groups, errors = validate_pair_inputs(tmp_path)
    assert not errors and {path.parent.parent.name for path in sources} == {
        "A",
        "C",
        "D",
    }
    assert "B" in groups[0]["excluded_slots"]


def test_one_complete_candidate_cannot_create_a_negative(tmp_path):
    local_group(
        tmp_path, {"A": "failed", "B": "completed", "C": "failed", "D": "failed"}
    )
    sources, _, errors = validate_pair_inputs(tmp_path)
    assert not sources and any("fewer_than_two" in error for error in errors)


def test_canonical_completion_failure_does_not_skip_frozen_shadows(
    tmp_path, monkeypatch
):
    group = local_group(tmp_path)
    (group / "candidate_pool_execution.json").unlink()
    for name in "BCD":
        (group / name / "worker_exit.json").unlink()
    pair = InitialPair(group, read_json(group / "snapshot.json"))
    from scenesmith.scene_expert.slow_memory import paired_provenance

    monkeypatch.setattr(
        paired_provenance,
        "verify_pair_code_provenance",
        lambda *a, **k: pair.snapshot["code_provenance"],
    )
    monkeypatch.setattr(
        pair,
        "capture_returned",
        lambda *a: (_ for _ in ()).throw(ValueError("unknown Main")),
    )
    calls = []

    async def worker(agent, name):
        calls.append(name)
        code = 2 if name == "C" else 0
        write_json(
            group / name / "worker_exit.json",
            {"exit_code": code, "process_started": True, "error": None},
        )
        return code

    monkeypatch.setattr(pair, "_finish_shadow", worker)
    asyncio.run(pair.finish(object(), "native safety"))
    execution = read_json(group / "candidate_pool_execution.json")
    assert calls == ["B", "C", "D"]
    assert execution["slots"]["A"]["exit_code"] is None
    assert execution["slots"]["A"]["state"] == "failed"
    assert execution["worker_exits"] == {"B": 0, "C": 2, "D": 0}
    assert read_json(group / "status.json")["status"] == "pool_failed"


def test_shared_input_failure_starts_no_worker(tmp_path, monkeypatch):
    group = local_group(tmp_path)
    (group / "candidate_pool_execution.json").unlink()
    for name in "BCD":
        (group / name / "worker_exit.json").unlink()
    pair = InitialPair(group, read_json(group / "snapshot.json"))
    from scenesmith.scene_expert.slow_memory import paired_provenance

    monkeypatch.setattr(
        paired_provenance, "verify_pair_code_provenance", lambda *a, **k: None
    )
    (group / "input_scene/geometry.sdf").write_text("changed")
    with pytest.raises(ValueError, match="snapshot_files_changed"):
        asyncio.run(pair.finish(object(), "safety"))
    execution = read_json(group / "candidate_pool_execution.json")
    assert execution["global_error"]
    assert execution["worker_exits"] == dict.fromkeys("BCD")


def test_completed_schedule_is_never_reexecuted(tmp_path):
    group = local_group(tmp_path)
    pair = InitialPair(group, read_json(group / "snapshot.json"))
    before = (group / "candidate_pool_execution.json").read_bytes()
    with pytest.raises(ValueError, match="second execution"):
        asyncio.run(pair.finish(object(), "safety"))
    assert (group / "candidate_pool_execution.json").read_bytes() == before


def test_real_CPU_worker_exits_and_canonical_proofs_survive_partial_pool(
    tmp_path, monkeypatch
):
    """Real child processes exercise dispatch; no simulator/model/native yield."""
    from scenesmith.scene_expert.slow_memory import paired_provenance, paired_runtime

    group = local_group(tmp_path / "pool")
    (group / "candidate_pool_execution.json").unlink()
    for name in "BCD":
        (group / name / "worker_exit.json").unlink()
    fake_repo = tmp_path / "cpu-worker"
    script = fake_repo / "scripts/collect_sceneexpert_initial_pairs.py"
    script.parent.mkdir(parents=True)
    script.write_text(
        "import sys\nn=sys.argv[sys.argv.index('--worker-candidate')+1]\nprint(n)\nsys.exit(2 if n=='C' else 0)\n"
    )
    canonical = tmp_path / "canonical"
    canonical.mkdir()
    (canonical / "geometry.txt").write_text("unchanged")
    snapshot = read_json(group / "snapshot.json")
    snapshot["source_scene_root"] = str(canonical)
    pair = InitialPair(group, snapshot)
    monkeypatch.setattr(paired_runtime, "REPO", fake_repo)
    monkeypatch.setattr(
        paired_provenance,
        "verify_pair_code_provenance",
        lambda *a, **k: snapshot["code_provenance"],
    )
    monkeypatch.setattr(
        pair,
        "capture_returned",
        lambda *a: (_ for _ in ()).throw(ValueError("unknown Main")),
    )
    agent = SimpleNamespace(
        scene=SimpleNamespace(to_state_dict=lambda: {"stable": True})
    )
    asyncio.run(pair.finish(agent, "safety"))
    execution = read_json(group / "candidate_pool_execution.json")
    assert execution["worker_exits"] == {"B": 0, "C": 2, "D": 0}
    assert execution["slots"]["A"]["exit_code"] is None
    for name in "BCD":
        assert read_json(group / name / "worker_exit.json")["process_started"] is True
        proof = read_json(group / f"continuation_proof_{name}.json")
        assert all(
            proof[f"{key}_before"] == proof[f"{key}_after"]
            for key in ("canonical", "assets", "memory")
        )


def test_timeout_records_observed_exit_instead_of_inventing_one(tmp_path, monkeypatch):
    from scenesmith.scene_expert.slow_memory import paired_runtime
    from scenesmith.scene_expert.slow_memory.pair_local_runtime import finish_shadow

    repo = tmp_path / "cpu-worker"
    script = repo / "scripts/collect_sceneexpert_initial_pairs.py"
    script.parent.mkdir(parents=True)
    script.write_text("import time\ntime.sleep(20)\n")
    canonical = tmp_path / "canonical"
    canonical.mkdir()
    group = tmp_path / "group"
    group.mkdir()
    pair = InitialPair(group, {"source_scene_root": str(canonical)})
    monkeypatch.setattr(paired_runtime, "REPO", repo)
    monkeypatch.setenv("SCENEEXPERT_PAIR_TIMEOUT", "0")
    agent = SimpleNamespace(
        scene=SimpleNamespace(to_state_dict=lambda: {"stable": True})
    )
    code = asyncio.run(finish_shadow(pair, agent, "B"))
    actual = read_json(group / "B/worker_exit.json")
    assert type(code) is int and code != 0 and actual["exit_code"] == code
    assert "TimeoutError" in actual["error"]
    assert (
        read_json(group / "continuation_proof_B.json")["memory_before"]
        == read_json(group / "continuation_proof_B.json")["memory_after"]
    )
