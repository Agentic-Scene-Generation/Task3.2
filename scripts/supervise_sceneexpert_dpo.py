#!/usr/bin/env python3
"""Supervise one DPO invocation; exit zero only after verifying final artifacts."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scenesmith.scene_expert.slow_memory.training_lifecycle import check_training_completion, write_json


def resource_sample(pid: int) -> dict:
    """Sample child RSS and job cgroup limits/events without probing other jobs."""
    result = {"time_unix": time.time(), "pid": pid}
    try:
        result["process_memory"] = [line for line in Path(f"/proc/{pid}/status").read_text().splitlines()
                                    if line.startswith(("VmRSS:", "VmHWM:", "VmSize:"))]
        cgroup = Path(f"/proc/{pid}/cgroup").read_text().splitlines()
        result["cgroup"] = cgroup
        relative = next((line.split(":", 2)[2] for line in cgroup if line.startswith("0::")), "/")
        roots = [Path("/sys/fs/cgroup") / relative.lstrip("/"), Path("/sys/fs/cgroup")]
        for root in roots:
            if (root / "memory.current").is_file():
                result["memory"] = {name: (root / name).read_text().strip() for name in
                                    ("memory.current", "memory.peak", "memory.max", "memory.events") if (root / name).is_file()}
                result["cgroup_memory_path"] = str(root)
                break
    except OSError as exc:
        result["resource_read_error"] = str(exc)
    return result


def emit(payload: dict) -> None:
    """Do not terminate training if the platform closes its log consumer."""
    try:
        print(json.dumps(payload), flush=True)
    except (BrokenPipeError, OSError):
        pass


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    output = args.output_dir
    if args.check:
        try:
            status = json.loads((output / "training_supervisor.json").read_text())
            check = check_training_completion(output, status["execution_id"])
            if status.get("state") != "completed" or status.get("process_returncode") != 0:
                check["completed"] = False
                check["errors"].append("supervisor did not confirm normal completion")
        except (OSError, ValueError, KeyError) as exc:
            check = {"completed": False, "errors": [str(exc)]}
        emit(check)
        return 0 if check["completed"] else 4
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("a training command is required")
    output.mkdir(parents=True, exist_ok=True)
    execution_id = uuid.uuid4().hex
    state = {"execution_id": execution_id, "state": "starting", "started_at": time.time(), "received_signal": None}
    write_json(output / "training_supervisor.json", state)
    child = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                             env={**os.environ, "PYTHONUNBUFFERED": "1", "SCENEEXPERT_TRAIN_EXECUTION_ID": execution_id},
                             start_new_session=os.name != "nt")
    state.update(state="running", pid=child.pid)
    write_json(output / "training_supervisor.json", state)

    def terminate(signum: int, frame: object) -> None:
        state.update(state="interrupted", received_signal=signum, signal_time=time.time())
        write_json(output / "training_supervisor.json", state)
        try:
            if os.name == "nt":
                child.terminate()
            else:
                os.killpg(child.pid, signum)
        except ProcessLookupError:
            pass

    for name in ("SIGTERM", "SIGINT", "SIGHUP"):
        if hasattr(signal, name):
            signal.signal(getattr(signal, name), terminate)

    def relay() -> None:
        console = True
        with (output / "train.log").open("a", encoding="utf-8") as log:
            for line in child.stdout:
                log.write(line)
                log.flush()
                if console:
                    try:
                        sys.stdout.write(line)
                        sys.stdout.flush()
                    except (BrokenPipeError, OSError):
                        console = False

    reader = threading.Thread(target=relay, daemon=True)
    reader.start()
    sample = {}
    with (output / "resource_usage.jsonl").open("a", encoding="utf-8") as resources:
        while child.poll() is None:
            sample = resource_sample(child.pid)
            resources.write(json.dumps(sample) + "\n")
            resources.flush()
            emit({"training_heartbeat": sample})
            try:
                child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                if state.get("signal_time") and time.time() - state["signal_time"] > 25:
                    try:
                        if os.name == "nt":
                            child.kill()
                        else:
                            os.killpg(child.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
        resources.write(json.dumps(resource_sample(child.pid)) + "\n")
        # /proc/<pid> is gone after wait; retain final cgroup OOM counters too.
        if sample.get("cgroup_memory_path"):
            root = Path(sample["cgroup_memory_path"])
            resources.write(json.dumps({"time_unix": time.time(), "after_exit": True,
                "memory": {name: (root / name).read_text().strip() for name in
                    ("memory.current", "memory.peak", "memory.max", "memory.events")
                    if (root / name).is_file()}}) + "\n")
    reader.join(timeout=10)
    check = check_training_completion(output, execution_id)
    rc = child.returncode
    exit_code = 128 + state["received_signal"] if state["received_signal"] else (rc if rc >= 0 else 128 - rc)
    if exit_code == 0 and not check["completed"]:
        exit_code = 4
    state.update(state="completed" if exit_code == 0 else "failed", process_returncode=rc,
                 exit_code=exit_code, finished_at=time.time(), completion_check=check)
    write_json(output / "training_supervisor.json", state)
    emit(state)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
