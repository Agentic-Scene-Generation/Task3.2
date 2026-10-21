"""Opt-in slot accounting around unchanged native candidate workers."""

from __future__ import annotations

import asyncio
import os
import signal
import subprocess
import sys
from pathlib import Path
from typing import Any

from scenesmith.scene_expert.slow_memory.paired import (
    digest,
    read_json,
    tree_hashes,
    write_json,
)


async def finish_pool(pair: Any, agent: Any, safety_message: str) -> None:
    """Continue the frozen schedule only while every shared invariant holds."""
    from scenesmith.scene_expert.slow_memory.pair_local import validate_frozen_inputs
    from scenesmith.scene_expert.slow_memory.paired_provenance import (
        verify_pair_code_provenance,
    )
    from scenesmith.scene_expert.slow_memory.paired_runtime import REPO

    names = ("A", "B", "C", "D")
    path = pair.group / "candidate_pool_execution.json"
    if path.exists():
        raise ValueError(
            "pair-local schedule already recorded; refusing a second execution"
        )
    execution = {
        "policy": pair.snapshot["candidate_pool"],
        "scheduled": list(names[1:]),
        "worker_exits": dict.fromkeys(names[1:]),
        "terminal": False,
        "slots": {
            name: {
                "state": "unstarted",
                "exit_code": None,
                "reason": "frozen schedule has not reached this slot",
            }
            for name in names
        },
    }
    execution["slots"]["A"] = {
        "state": "failed",
        "exit_code": None,
        "reason": "canonical execution occurred; frozen input validation or completion is pending",
    }
    write_json(path, execution)
    try:
        verify_pair_code_provenance(
            pair.snapshot.get("code_provenance", {}), repo_root=REPO
        )
        validate_frozen_inputs(pair.group, pair.snapshot)
        if getattr(pair, "_shared_evaluation_failure", None):
            raise ValueError(pair._shared_evaluation_failure)
        try:
            pair.capture_returned(agent, safety_message, "A")
        except (OSError, ValueError, KeyError, TypeError) as exc:
            # An absent native result/unknown Main is local, never a negative.
            # Re-check shared bytes before allowing any independent worker.
            verify_pair_code_provenance(
                pair.snapshot.get("code_provenance", {}), repo_root=REPO
            )
            validate_frozen_inputs(pair.group, pair.snapshot)
            failure = {
                "state": "failed",
                "exit_code": None,
                "reason": f"{type(exc).__name__}: {exc}",
            }
            execution["slots"]["A"] = failure
            write_json(pair.group / "A/completion_failure.json", failure)
        else:
            execution["slots"]["A"] = {"state": "completed", "exit_code": None}
        write_json(path, execution)
        for name in names[1:]:
            verify_pair_code_provenance(
                pair.snapshot.get("code_provenance", {}), repo_root=REPO
            )
            validate_frozen_inputs(pair.group, pair.snapshot)
            code = await pair._finish_shadow(agent, name)
            execution["worker_exits"][name] = code
            actual = read_json(pair.group / name / "worker_exit.json")
            success = code == 0 and not actual.get("error")
            slot = {"state": "completed" if success else "failed", "exit_code": code}
            if not success:
                slot["reason"] = (
                    actual.get("error")
                    or "native worker returned a nonzero process exit"
                )
            execution["slots"][name] = slot
            write_json(path, execution)
    except BaseException as exc:
        execution["global_error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        # A shared-invariance exception can occur after a worker really ran.
        # Preserve its observed exit instead of calling that slot unstarted.
        for name in names[1:]:
            exit_path = pair.group / name / "worker_exit.json"
            if execution["slots"][name]["state"] == "unstarted" and exit_path.exists():
                actual = read_json(exit_path)
                execution["worker_exits"][name] = actual["exit_code"]
                execution["slots"][name] = {
                    "state": "failed",
                    "exit_code": actual["exit_code"],
                    "reason": actual.get("error")
                    or execution.get("global_error")
                    or "worker invariance was not established",
                }
        execution["terminal"] = True
        execution["pool_complete"] = all(
            slot["state"] == "completed" for slot in execution["slots"].values()
        )
        write_json(path, execution)
        write_json(
            pair.group / "status.json",
            {
                "status": "completed"
                if execution["pool_complete"] and not execution.get("global_error")
                else "pool_failed",
                "canonical": "A",
                "pool_complete": execution["pool_complete"],
                "worker_exits": execution["worker_exits"],
                "global_error": execution.get("global_error"),
            },
        )


async def finish_shadow(pair: Any, agent: Any, candidate: str) -> int | None:
    """Record actual worker exits and invariance even after a worker timeout."""
    from scenesmith.scene_expert.slow_memory.paired_runtime import (
        REPO,
        canonical_asset_hashes,
        json_value,
        shadow_environment,
    )

    before = digest(json_value(agent.scene.to_state_dict()))
    assets = canonical_asset_hashes(Path(pair.snapshot["source_scene_root"]))
    bank = Path(
        os.environ.get(
            "SCENEEXPERT_ACTIVE_MEMORY_BANK_DIR", str(pair.group / "no_bank")
        )
    )
    memory = tree_hashes(bank) if bank.is_dir() else {}
    process, error, code = None, None, None
    log_name = "shadow.log" if candidate == "B" else f"shadow_{candidate}.log"
    try:
        with (pair.group / log_name).open("w", encoding="utf-8") as log:
            process = await asyncio.create_subprocess_exec(
                sys.executable,
                str(REPO / "scripts/collect_sceneexpert_initial_pairs.py"),
                "--worker-group",
                str(pair.group),
                "--worker-candidate",
                candidate,
                cwd=REPO,
                env=shadow_environment(pair.group, candidate),
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=(os.name == "posix"),
            )
            try:
                code = await asyncio.wait_for(
                    process.wait(),
                    timeout=int(os.environ.get("SCENEEXPERT_PAIR_TIMEOUT", "3600")),
                )
            except BaseException as exc:  # noqa: BLE001 - reap the private worker before propagating errors
                error = exc
                if process.returncode is None:
                    process.terminate()
                    try:
                        await asyncio.wait_for(process.wait(), timeout=30)
                    except TimeoutError:
                        process.kill()
                        await process.wait()
                code = process.returncode
    except OSError as exc:
        error = exc
    finally:
        if process is not None and (code or error) and os.name == "posix":
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        proof = {
            "canonical_before": before,
            "canonical_after": digest(json_value(agent.scene.to_state_dict())),
            "assets_before": digest(assets),
            "assets_after": digest(
                canonical_asset_hashes(Path(pair.snapshot["source_scene_root"]))
            ),
            "memory_before": digest(memory),
            "memory_after": digest(tree_hashes(bank) if bank.is_dir() else {}),
        }
        write_json(pair.group / f"continuation_proof_{candidate}.json", proof)
        write_json(pair.group / "continuation_proof.json", proof)
        write_json(
            pair.group / candidate / "worker_exit.json",
            {
                "exit_code": code,
                "process_started": process is not None,
                "error": f"{type(error).__name__}: {error}" if error else None,
            },
        )
    if any(
        proof[f"{key}_before"] != proof[f"{key}_after"]
        for key in ("canonical", "assets", "memory")
    ):
        raise ValueError("shadow execution changed canonical scene or memory")
    failure_path = pair.group / candidate / "failure.json"
    if (
        failure_path.exists()
        and read_json(failure_path).get("failure_scope") == "global"
    ):
        raise ValueError(f"{candidate} violated a native shared-input guard")
    if isinstance(error, (asyncio.CancelledError, KeyboardInterrupt, SystemExit)):
        raise error
    if error is not None and not isinstance(error, (TimeoutError, OSError)):
        raise error
    # A timeout/spawn failure has true evidence and is local; the next slot runs.
    return code
