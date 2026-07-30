"""Read-only API for inspecting SceneSmith critic-probe runs.

The service deliberately exposes only files below ``outputs/critic_probe``.  It
is intended for a trusted local/CCI network and never mutates a replay run.
"""

from __future__ import annotations

from functools import lru_cache
from io import BytesIO
import json
import math
import os
import re
import signal
import sqlite3
import time
from collections import Counter
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from flask import Flask, abort, jsonify, request, send_file
from PIL import Image


DEFAULT_PROBE_ROOT = Path(__file__).resolve().parents[1] / "outputs" / "critic_probe"
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}
HSSD_ASSET_ID_PATTERN = re.compile(r"[0-9a-f]{40}")
THUMBNAIL_MAX_WIDTH = 360
SERVER_HOST = "127.0.0.1"
SERVER_PORT = 5055


def _listening_tcp_pids(port: int) -> set[int]:
    """Return local process IDs listening on a TCP port, without extra packages."""
    socket_inodes: set[str] = set()
    for table in (Path("/proc/net/tcp"), Path("/proc/net/tcp6")):
        try:
            rows = table.read_text(encoding="utf-8").splitlines()[1:]
        except OSError:
            continue
        for row in rows:
            fields = row.split()
            if len(fields) < 10 or fields[3] != "0A":
                continue
            try:
                listening_port = int(fields[1].rsplit(":", 1)[1], 16)
            except (IndexError, ValueError):
                continue
            if listening_port == port:
                socket_inodes.add(fields[9])

    if not socket_inodes:
        return set()

    pids: set[int] = set()
    for process_dir in Path("/proc").iterdir():
        if not process_dir.name.isdigit():
            continue
        try:
            for descriptor in (process_dir / "fd").iterdir():
                target = os.readlink(descriptor)
                if target.startswith("socket:[") and target.endswith("]"):
                    if target[8:-1] in socket_inodes:
                        pids.add(int(process_dir.name))
                        break
        except OSError:
            continue
    return pids


def _command_runs_this_script(command: list[str], cwd: Path) -> bool:
    script_path = Path(__file__).resolve()
    module_name = "tools.critic_probe_web"
    for index, argument in enumerate(command[1:], start=1):
        if argument == "-m" and index + 1 < len(command):
            if command[index + 1] == module_name:
                return True
        if argument == "-c" and index + 1 < len(command):
            inline_program = command[index + 1]
            if module_name in inline_program and "create_app" in inline_program:
                return True
    for argument in command[1:]:
        candidate = Path(argument)
        if candidate.name != script_path.name:
            continue
        resolved = (
            candidate.resolve()
            if candidate.is_absolute()
            else (cwd / candidate).resolve()
        )
        if resolved == script_path:
            return True
    return False


def _process_runs_this_script(pid: int) -> bool:
    try:
        command = [
            value.decode("utf-8", errors="replace")
            for value in (Path("/proc") / str(pid) / "cmdline")
            .read_bytes()
            .split(b"\0")
            if value
        ]
        cwd = Path(os.readlink(Path("/proc") / str(pid) / "cwd"))
    except OSError:
        return False
    return _command_runs_this_script(command, cwd)


def _release_previous_probe_server(port: int) -> None:
    listener_pids = _listening_tcp_pids(port)
    previous_pids = {
        pid
        for pid in listener_pids
        if pid != os.getpid() and _process_runs_this_script(pid)
    }
    if not previous_pids:
        if listener_pids:
            raise RuntimeError(
                f"Port {port} is occupied by another program; refusing to stop it."
            )
        return

    print(
        f"Stopping previous critic-probe web server: {sorted(previous_pids)}",
        flush=True,
    )
    for pid in previous_pids:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            continue

    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        if not (_listening_tcp_pids(port) & previous_pids):
            return
        time.sleep(0.05)
    raise RuntimeError(f"Previous critic-probe web server did not release port {port}.")


def _safe_relative_path(root: Path, relative_path: str) -> Path:
    candidate = (root / relative_path).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError:
        abort(404)
    return candidate


def _safe_hssd_rendered_image(image_path: str) -> Path:
    """Limit external image access to individual HSSD rendered-asset views."""
    candidate = Path(image_path).resolve()
    if (
        not candidate.is_file()
        or candidate.suffix.lower() not in IMAGE_SUFFIXES
        or candidate.parent.parent.name != "hssd_rendered_assets"
        or not HSSD_ASSET_ID_PATTERN.fullmatch(candidate.parent.name.lower())
    ):
        abort(404)
    return candidate


@lru_cache(maxsize=256)
def _thumbnail_bytes(path_value: str, mtime_ns: int, width: int) -> bytes:
    """Resize once locally so cloud-backed source images are not repeatedly sent."""
    del mtime_ns  # The value is part of the cache key and invalidates changed assets.
    with Image.open(path_value) as source:
        image = source.convert("RGB")
        image.thumbnail((width, width), Image.Resampling.LANCZOS)
        output = BytesIO()
        image.save(output, format="JPEG", quality=82, optimize=True)
    return output.getvalue()


def _read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    try:
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(item, dict):
                rows.append(item)
    except OSError:
        return []
    return rows


def _iso_mtime(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime, UTC).isoformat()


def _json_safe(value: Any) -> Any:
    """Convert Python-only numeric values before sending browser JSON."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    return value


def _json_response(payload: Any) -> Any:
    return jsonify(_json_safe(payload))


def _room_directories(run_root: Path) -> list[Path]:
    # Probe output has a stable shallow layout.  Avoid ``rglob`` here: each room
    # also contains large generated-asset trees, and polling must stay cheap.
    patterns = (
        "*/*/hydra/scene_*/room_*/action_log.json",
        "*/*/hydra/*/room_*/action_log.json",
        "*/hydra/scene_*/room_*/action_log.json",
        "*/*/hydra/scene_*/room_*/timing_stats.jsonl",
        "*/*/hydra/*/room_*/timing_stats.jsonl",
        "*/hydra/scene_*/room_*/timing_stats.jsonl",
    )
    paths = {marker.parent for pattern in patterns for marker in run_root.glob(pattern)}
    return sorted(paths)


def _is_room_directory(path: Path) -> bool:
    return path.is_dir() and (
        (path / "action_log.json").is_file() or (path / "timing_stats.jsonl").is_file()
    )


def _scene_identity(run_root: Path, room_dir: Path) -> dict[str, str]:
    parts = room_dir.relative_to(run_root).parts
    batch = next((part for part in parts if part.startswith("batch_")), "unbatched")
    scene = next((part for part in parts if part.startswith("scene_")), room_dir.name)
    mode = parts[0] if parts else "default"
    return {
        "batch": batch,
        "scene": scene,
        "room": room_dir.name.removeprefix("room_"),
        "mode": mode,
    }


def _stage_from_render_dir(render_dir: Path) -> str:
    name = render_dir.parent.name
    if name.startswith("manipulands_"):
        return "manipuland"
    return {"wall": "wall_mounted", "ceiling": "ceiling_mounted"}.get(name, name)


def _render_records(probe_root: Path, room_dir: Path) -> list[dict[str, Any]]:
    render_root = room_dir / "scene_renders"
    records: list[dict[str, Any]] = []
    if render_root.is_dir():
        for state_path in sorted(render_root.rglob("scene_state.json")):
            render_dir = state_path.parent
            top_image = next(iter(sorted(render_dir.glob("*top*.png"))), None)
            side_image = next(iter(sorted(render_dir.glob("*side*.png"))), None)
            score_path = render_dir / "scores.yaml"
            records.append(
                {
                    "id": str(render_dir.relative_to(room_dir)),
                    "stage": _stage_from_render_dir(render_dir),
                    "label": f"{_stage_from_render_dir(render_dir)} / {render_dir.name}",
                    "state_path": str(state_path.relative_to(probe_root)),
                    "top_image": (
                        str(top_image.relative_to(probe_root)) if top_image else None
                    ),
                    "side_image": (
                        str(side_image.relative_to(probe_root)) if side_image else None
                    ),
                    "has_scores": score_path.is_file(),
                    # The scene state is written once per snapshot. Directory mtimes
                    # can be changed later by score files or auxiliary renders.
                    "created_at": _iso_mtime(state_path),
                }
            )

    final_view_dir = room_dir.parent / "critic_final_views"
    final_top = final_view_dir / "00_top.png"
    final_side = final_view_dir / "01_side.png"
    if final_top.is_file() or final_side.is_file():
        final_images = [path for path in (final_top, final_side) if path.is_file()]
        records.append(
            {
                "id": "critic_final_views",
                "stage": "final_view",
                "label": "Final view",
                # Final views are rendered from house.blend and have no matching
                # scene_state.json, so they intentionally cannot be diffed.
                "state_path": None,
                "top_image": (
                    str(final_top.relative_to(probe_root))
                    if final_top.is_file()
                    else None
                ),
                "side_image": (
                    str(final_side.relative_to(probe_root))
                    if final_side.is_file()
                    else None
                ),
                "has_scores": False,
                "created_at": _iso_mtime(
                    max(final_images, key=lambda path: path.stat().st_mtime)
                ),
            }
        )
    return sorted(records, key=lambda record: record["created_at"], reverse=True)


def _timing_records(room_dir: Path) -> list[dict[str, Any]]:
    rows = _read_jsonl(room_dir / "timing_stats.jsonl")
    for row in rows:
        row["source"] = "timing"
        row["detail"] = row.get("extra", {})
    return rows


def _llm_records(room_dir: Path) -> list[dict[str, Any]]:
    # Older probe runs stored LLM traces at scene level while newer runs write
    # them in the room directory. Read both, retaining the owning directory so
    # a payload_ref is resolved against the same place that produced it.
    records: list[dict[str, Any]] = []
    seen: set[Path] = set()
    for audit_root in (room_dir, room_dir.parent):
        for filename in ("llm_calls.jsonl", "scene_expert_llm_calls.jsonl"):
            path = audit_root / "scene_expert" / "timing" / filename
            if path in seen:
                continue
            seen.add(path)
            for row in _read_jsonl(path):
                row["source"] = "llm"
                row["_audit_root"] = "." if audit_root == room_dir else ".."
                row["_audit_file"] = filename
                records.append(row)
    return sorted(records, key=lambda row: str(row.get("created_at", "")))


def _orchestration_records(room_dir: Path) -> list[dict[str, Any]]:
    """Read explicit Planner dispatch/resume events from current and legacy roots."""
    records: list[dict[str, Any]] = []
    seen: set[Path] = set()
    for audit_root in (room_dir, room_dir.parent):
        path = audit_root / "scene_expert" / "timing" / "planner_orchestration.jsonl"
        if path in seen:
            continue
        seen.add(path)
        records.extend(_read_jsonl(path))
    return sorted(records, key=lambda row: str(row.get("created_at", "")))


def _benchmark_records(room_dir: Path) -> list[dict[str, Any]]:
    path = room_dir / "scene_expert" / "timing" / "scenebenchmark_critic_timing.jsonl"
    rows = _read_jsonl(path)
    for row in rows:
        row["source"] = "scenebenchmark"
    return rows


def _repair_records(room_dir: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    seen: set[Path] = set()
    for audit_root in (room_dir, room_dir.parent):
        path = audit_root / "scene_expert" / "timing" / "repair_events.jsonl"
        if path in seen:
            continue
        seen.add(path)
        records.extend(
            row
            for row in _read_jsonl(path)
            if not row.get("room") or row.get("room") == room_dir.name
        )
    return sorted(records, key=lambda row: str(row.get("created_at", "")))


def _parse_time(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return (
            parsed.replace(tzinfo=UTC)
            if parsed.tzinfo is None
            else parsed.astimezone(UTC)
        )
    except ValueError:
        return None


def _action_records(room_dir: Path) -> list[dict[str, Any]]:
    raw_actions = _read_json(room_dir / "action_log.json", [])
    if not isinstance(raw_actions, list):
        return []
    local_timezone = datetime.now().astimezone().tzinfo
    actions: list[dict[str, Any]] = []
    for item in raw_actions:
        if not isinstance(item, dict):
            continue
        action = dict(item)
        timestamp = action.get("timestamp")
        if isinstance(timestamp, str):
            try:
                parsed = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            except ValueError:
                parsed = None
            if parsed is not None:
                if parsed.tzinfo is None:
                    parsed = parsed.replace(tzinfo=local_timezone)
                action["timestamp"] = parsed.astimezone(UTC).isoformat()
        actions.append(action)
    return actions


def _asset_record_for_action(
    room_dir: Path, action: dict[str, Any]
) -> dict[str, Any] | None:
    arguments = action.get("arguments", {})
    asset_id = arguments.get("asset_id") if isinstance(arguments, dict) else None
    if not isinstance(asset_id, str) or not asset_id:
        return None
    for registry_path in sorted(
        room_dir.glob("generated_assets/*/asset_registry.json")
    ):
        registry = _read_json(registry_path, {})
        if not isinstance(registry, dict):
            continue
        record = registry.get(asset_id)
        if isinstance(record, dict):
            return record
    return None


def _stage_for_action(action: dict[str, Any]) -> str:
    name = str(action.get("tool_name", "")).lower()
    if "manipuland" in name:
        return "manipuland"
    if "ceiling" in name:
        return "ceiling_mounted"
    if "wall" in name:
        return "wall_mounted"
    return "furniture"


def _choice_audit_records(room_dir: Path) -> list[dict[str, Any]]:
    paths = [
        room_dir / "audit" / "hssd_rendered_choice.jsonl",
        *sorted(room_dir.glob("generated_assets/*/hssd_rendered_choice.jsonl")),
        # Historical Hydra runs record rendered-choice decisions once per batch.
        room_dir.parents[1] / "asset_choice_audit.jsonl",
    ]
    records: list[dict[str, Any]] = []
    for path in paths:
        for row in _read_jsonl(path):
            if (
                isinstance(row, dict)
                and row.get("schema_version") == "hssd_rendered_choice_audit.v2"
            ):
                records.append(row)
    return records


def _retrieval_backend(decision: dict[str, Any]) -> str:
    recorded_backend = decision.get("retrieval_backend")
    if isinstance(recorded_backend, str) and recorded_backend:
        return recorded_backend
    candidates = decision.get("candidates", [])
    if any(
        isinstance(candidate, dict) and "similarity_score" in candidate
        for candidate in candidates
    ):
        return "embedding"
    return "unknown"


def _selection_trace_for_action(
    room_dir: Path, action: dict[str, Any]
) -> dict[str, Any]:
    """Resolve an Add Furniture action to its recorded retrieval decision."""
    if "add_furniture" not in str(action.get("tool_name", "")).lower():
        return {"status": "not_applicable"}
    asset = _asset_record_for_action(room_dir, action)
    if asset is None:
        return {
            "status": "not_recorded",
            "note": "The asset registry does not contain this placement asset.",
        }
    metadata = (
        asset.get("metadata", {}) if isinstance(asset.get("metadata"), dict) else {}
    )
    hssd_id = metadata.get("hssd_mesh_id")
    description = asset.get("description")
    short_name = metadata.get("asset_short_name") or asset.get("name")
    matching = [
        row
        for row in _choice_audit_records(room_dir)
        if row.get("selected_hssd_id") == hssd_id
        or (
            row.get("object_description") == description
            and row.get("object_short_name") == short_name
        )
    ]
    if not matching:
        return {
            "status": "not_recorded",
            "asset": {
                "asset_id": asset.get("object_id"),
                "description": description,
                "hssd_id": hssd_id,
                "asset_source": metadata.get("asset_source"),
            },
            "note": "No retrieval/VLM audit was written for this historical asset.",
        }
    decision = matching[-1]
    return {
        "status": "recorded",
        "asset": {
            "asset_id": asset.get("object_id"),
            "description": description,
            "hssd_id": hssd_id,
            "asset_source": metadata.get("asset_source"),
        },
        "retrieval": {
            "backend": _retrieval_backend(decision),
            "requested_dimensions": decision.get("requested_dimensions"),
            "candidates": decision.get("candidates", []),
        },
        "vlm_selection": {
            "status": decision.get("status"),
            "model": decision.get("model"),
            "selected_index": decision.get("selected_index"),
            "selected_hssd_id": decision.get("selected_hssd_id"),
            "reason": decision.get("reason"),
            "raw_response": _audit_value(decision.get("raw_response")),
            "parsed_response": decision.get("parsed_response"),
            "quality_fallback_used": decision.get("quality_fallback_used"),
            "render_quality_by_hssd_id": decision.get("render_quality_by_hssd_id"),
        },
    }


def _timing_for_llm(
    call: dict[str, Any], timings: list[dict[str, Any]]
) -> dict[str, Any] | None:
    call_time = _parse_time(str(call.get("created_at", "")))
    candidates = [
        row
        for row in timings
        if row.get("stage") == call.get("stage")
        and row.get("module") == call.get("agent_role")
        and row.get("event") == call.get("event")
    ]
    if not candidates:
        return None
    if call_time is None:
        return candidates[-1]
    return min(
        candidates,
        key=lambda row: abs(
            (_parse_time(str(row.get("created_at", ""))) or call_time) - call_time
        ),
    )


def _audit_events(room_dir: Path) -> list[dict[str, Any]]:
    timings = _timing_records(room_dir)
    repair_records = _repair_records(room_dir)
    has_structured_hard_repairs = any(
        row.get("strategy") == "deterministic_hard_state" for row in repair_records
    )
    events: list[dict[str, Any]] = []
    for index, row in enumerate(timings):
        if row.get("module") == "deterministic_repair":
            if has_structured_hard_repairs:
                continue
            extra = row.get("extra", {})
            repair = {
                "source": row.get("event", "hard_state"),
                "strategy": "deterministic_hard_state",
                "status": "accepted" if extra.get("repaired") else "rejected",
                "attempt": extra.get("attempt"),
                "trigger_reasons": extra.get("hard_reasons", []),
                "actions": extra.get("actions", []),
                "affected_objects": [],
                "detail": {
                    "legacy_timing_record": True,
                    "max_attempts": extra.get("max_attempts"),
                },
            }
            events.append(
                {
                    "id": f"legacy-repair:{index}",
                    "kind": "repair",
                    "source": "timing",
                    "created_at": row.get("created_at"),
                    "stage": row.get("stage", "furniture"),
                    "actor": "deterministic repair",
                    "function": row.get("event", "deterministic_repair"),
                    "title": "Automatic furniture repair",
                    "elapsed_sec": row.get("elapsed_sec"),
                    "audit_status": repair["status"],
                    "detail": repair,
                    "repair": repair,
                }
            )
            continue
        events.append(
            {
                "id": f"timing:{index}",
                "kind": "system",
                "source": "timing",
                "created_at": row.get("created_at"),
                "stage": row.get("stage", "system"),
                "actor": row.get("module", "system"),
                "function": row.get("event", "pipeline_event"),
                "title": str(row.get("event", "pipeline event")).replace("_", " "),
                "elapsed_sec": row.get("elapsed_sec"),
                "audit_status": "timing_only",
                "detail": row.get("extra", {}),
            }
        )
    for index, row in enumerate(_benchmark_records(room_dir)):
        function = str(row.get("stage", "scenebenchmark_critic"))
        events.append(
            {
                "id": f"benchmark:{index}",
                "kind": "benchmark",
                "source": "scenebenchmark",
                "created_at": row.get("created_at"),
                "stage": function,
                "actor": "deterministic critic",
                "function": function,
                "title": "SceneBenchmark critic",
                "elapsed_sec": row.get("elapsed_sec"),
                "audit_status": str(row.get("status", "unknown")),
                "detail": row.get("details", {}),
                "metrics": row.get("steps", {}),
                "evaluation": row.get("evaluation", {}),
            }
        )
    for index, row in enumerate(repair_records):
        repair = {
            "source": row.get("source", "automatic_repair"),
            "strategy": row.get("strategy", "automatic_repair"),
            "status": row.get("status", "unknown"),
            "attempt": row.get("attempt"),
            "trigger_reasons": row.get("trigger_reasons", []),
            "actions": row.get("actions", []),
            "affected_objects": row.get("affected_objects", []),
            "detail": row.get("detail", {}),
        }
        events.append(
            {
                "id": f"repair:{index}",
                "kind": "repair",
                "source": "repair_event",
                "created_at": row.get("created_at"),
                "stage": row.get("stage", "furniture"),
                "actor": "deterministic repair",
                "function": repair["strategy"],
                "title": str(repair["strategy"]).replace("_", " "),
                "audit_status": repair["status"],
                "detail": repair,
                "repair": repair,
            }
        )
    dispatch_times: dict[str, datetime] = {}
    for index, row in enumerate(_orchestration_records(room_dir)):
        call_id = str(row.get("call_id", f"call-{index}"))
        phase = str(row.get("phase", "dispatch"))
        child_agent = str(row.get("child_agent", "agent"))
        operation = str(row.get("operation", "delegate"))
        event_time = _parse_time(str(row.get("created_at", "")))
        if phase == "dispatch" and event_time is not None:
            dispatch_times[call_id] = event_time
        started_at = dispatch_times.get(call_id) if phase == "resume" else None
        elapsed = (
            (event_time - started_at).total_seconds()
            if event_time is not None and started_at is not None
            else None
        )
        detail = (
            dict(row.get("detail", {})) if isinstance(row.get("detail"), dict) else {}
        )
        detail.update(
            {
                "call_id": call_id,
                "phase": phase,
                "operation": operation,
                "child_agent": child_agent,
                "status": row.get("status", "unknown"),
            }
        )
        events.append(
            {
                "id": f"orchestration:{index}",
                "kind": "orchestration",
                "source": "planner_orchestration",
                "created_at": row.get("created_at"),
                "started_at": started_at.isoformat() if started_at else None,
                "stage": row.get("stage", "unknown"),
                "actor": row.get("actor", "planner"),
                "function": operation,
                "title": (
                    f"Delegate to {child_agent}"
                    if phase == "dispatch"
                    else f"Resume from {child_agent}"
                ),
                "elapsed_sec": elapsed,
                "audit_status": str(row.get("status", phase)),
                "detail": detail,
                "orchestration": {
                    "call_id": call_id,
                    "phase": phase,
                    "child_agent": child_agent,
                },
            }
        )
    for index, row in enumerate(_llm_records(room_dir)):
        timing = _timing_for_llm(row, timings)
        end_time = _parse_time(str(row.get("created_at", "")))
        elapsed = timing.get("elapsed_sec") if timing else row.get("elapsed_sec")
        started_at = None
        if end_time is not None and isinstance(elapsed, (int, float)):
            started_at = (end_time - timedelta(seconds=float(elapsed))).isoformat()
        events.append(
            {
                "id": f"llm:{index}",
                "kind": "llm",
                "source": "llm",
                "created_at": row.get("created_at"),
                "started_at": started_at,
                "stage": row.get("stage", "unknown"),
                "actor": row.get("agent_role", "LLM"),
                "function": row.get("event", "llm_call"),
                "title": (
                    "TaskCompiler"
                    if row.get("agent_role") == "task_compiler"
                    else str(row.get("event", "llm call")).replace("_", " ")
                ),
                "elapsed_sec": elapsed,
                "audit_status": (
                    "full_payload"
                    if row.get("payload_ref")
                    else (
                        "full_inline_audit"
                        if "input" in row or "output" in row
                        else "session_trace"
                    )
                ),
                "token_usage": row.get("token_usage", {}),
                "prompt_chars": row.get("prompt_chars", 0),
                "output_chars": row.get("output_chars", 0),
                "has_error": bool(row.get("error")),
            }
        )
    return sorted(events, key=lambda event: str(event.get("created_at", "")))


def _audit_value(value: Any) -> Any:
    if isinstance(value, str) and value.startswith("data:image/"):
        return {"kind": "image_payload", "bytes_omitted": len(value)}
    if isinstance(value, list):
        return [_audit_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _audit_value(item) for key, item in value.items()}
    return value


def _message_text(payload: dict[str, Any]) -> str:
    content = payload.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            str(item.get("text", "")) if isinstance(item, dict) else str(item)
            for item in content
        )
    summary = payload.get("summary")
    if isinstance(summary, list):
        return "\n".join(
            str(item.get("text", "")) if isinstance(item, dict) else str(item)
            for item in summary
        )
    return ""


def _agent_from_database(database: Path) -> str:
    """Return a stable role label without discarding the source database."""
    name = database.stem.lower()
    for role in ("designer", "planner", "critic"):
        if role in name:
            return role
    return name.replace("_", " ") or "agent"


def _database_matches_actor(database: Path, actor: str | None) -> bool:
    """Keep an LLM event tied to its own agent session when possible."""
    normalized = str(actor or "").strip().lower().replace("_", " ")
    if not normalized:
        return True
    role = _agent_from_database(database)
    return normalized in role or role in normalized


def _session_trace(
    room_dir: Path,
    started_at: str | None,
    completed_at: str | None,
    actor: str | None = None,
) -> dict[str, Any]:
    start = _parse_time(started_at) or _parse_time(completed_at)
    end = _parse_time(completed_at) or start
    if start is None or end is None:
        return {
            "inputs": [],
            "outputs": [],
            "reasoning": [],
            "tool_calls": [],
            "messages": [],
            "databases": [],
        }
    start -= timedelta(seconds=2)
    end += timedelta(seconds=2)
    inputs: list[str] = []
    outputs: list[str] = []
    reasoning: list[str] = []
    messages: list[dict[str, Any]] = []
    pending_calls: dict[tuple[str, str], dict[str, Any]] = {}
    tool_calls: list[dict[str, Any]] = []
    databases: set[str] = set()
    for database in sorted(room_dir.glob("*.db")):
        if database.name.endswith("_summaries.db"):
            continue
        if actor and not _database_matches_actor(database, actor):
            continue
        try:
            connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
            rows = connection.execute(
                "SELECT id, message_data, created_at FROM agent_messages ORDER BY id"
            ).fetchall()
            connection.close()
        except sqlite3.Error:
            continue
        for message_id, raw_payload, created_at in rows:
            message_time = _parse_time(str(created_at))
            if message_time is None or not start <= message_time <= end:
                continue
            payload = _read_json_from_string(raw_payload)
            if not payload:
                continue
            databases.add(database.name)
            message_type = payload.get("type", payload.get("role", ""))
            record = {
                "database": database.name,
                "agent": _agent_from_database(database),
                "message_id": message_id,
                "created_at": message_time.isoformat(),
            }
            if message_type == "user":
                text = _message_text(payload)
                if text:
                    inputs.append(text)
                    messages.append({**record, "direction": "input", "content": text})
            elif message_type == "message":
                text = _message_text(payload)
                if text:
                    outputs.append(text)
                    messages.append({**record, "direction": "output", "content": text})
            elif message_type == "reasoning":
                text = _message_text(payload)
                if text:
                    reasoning.append(text)
                    messages.append(
                        {**record, "direction": "reasoning", "content": text}
                    )
            elif message_type == "function_call":
                call = {
                    "database": database.name,
                    "agent": _agent_from_database(database),
                    "message_id": message_id,
                    "name": payload.get("name", "function"),
                    "arguments": payload.get("arguments", "{}"),
                    "output": None,
                }
                pending_calls[
                    (database.name, str(payload.get("call_id", message_id)))
                ] = call
                tool_calls.append(call)
                messages.append(
                    {
                        **record,
                        "direction": "tool_call",
                        "content": {
                            "name": call["name"],
                            "arguments": call["arguments"],
                        },
                    }
                )
            elif message_type == "function_call_output":
                output = _audit_value(payload.get("output"))
                call = pending_calls.get(
                    (database.name, str(payload.get("call_id", "")))
                )
                if call is not None:
                    call["output"] = output
                messages.append(
                    {**record, "direction": "tool_output", "content": output}
                )
    messages.sort(
        key=lambda message: (
            message["created_at"],
            message["database"],
            message["message_id"],
        )
    )
    return {
        "inputs": inputs,
        "outputs": outputs,
        "reasoning": reasoning,
        "tool_calls": tool_calls,
        "messages": messages,
        "databases": sorted(databases),
    }


def _read_json_from_string(value: str) -> dict[str, Any]:
    try:
        parsed = json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _payload_for_llm_event(room_dir: Path, event_id: str) -> dict[str, Any] | None:
    try:
        index = int(event_id.removeprefix("llm:"))
    except ValueError:
        return None
    calls = _llm_records(room_dir)
    if index < 0 or index >= len(calls):
        return None
    call = calls[index]
    event = next(
        (item for item in _audit_events(room_dir) if item["id"] == event_id), None
    )
    if event is None:
        return None
    timings = _timing_records(room_dir)
    timing = _timing_for_llm(call, timings)
    completed_at = str(call.get("created_at", ""))
    completed = _parse_time(completed_at)
    elapsed = timing.get("elapsed_sec") if timing else call.get("elapsed_sec")
    started_at = None
    if completed is not None and isinstance(elapsed, (int, float)):
        started_at = (completed - timedelta(seconds=float(elapsed))).isoformat()
    trace = _session_trace(
        room_dir, started_at, completed_at, str(event.get("actor", ""))
    )
    if "input" in call or "output" in call:
        inline_input = _audit_value(call.get("input", call.get("prompt_excerpt", "")))
        inline_output = _audit_value(call.get("output", call.get("output_excerpt", "")))
        messages = [
            {
                "agent": event.get("actor", "LLM"),
                "database": call.get("_audit_file", "inline audit"),
                "message_id": 0,
                "created_at": event.get("started_at") or event.get("created_at"),
                "direction": "input",
                "content": inline_input,
            },
            {
                "agent": event.get("actor", "LLM"),
                "database": call.get("_audit_file", "inline audit"),
                "message_id": 1,
                "created_at": event.get("created_at"),
                "direction": "output",
                "content": inline_output,
            },
        ]
        return {
            "event": event,
            "provenance": "full_inline_audit",
            "input": inline_input,
            "output": inline_output,
            "reasoning": [],
            "tool_calls": [],
            "messages": messages,
            "session_databases": [call.get("_audit_file", "inline audit")],
            "has_full_input": "input" in call,
            "has_full_output": "output" in call,
        }
    payload_ref = call.get("payload_ref")
    if isinstance(payload_ref, str) and payload_ref:
        audit_root = (room_dir / str(call.get("_audit_root", "."))).resolve()
        payload_path = (audit_root / payload_ref).resolve()
        scene_root = room_dir.parent.resolve()
        if payload_path.is_file() and payload_path.is_relative_to(scene_root):
            payload = _read_json(payload_path, {})
            if isinstance(payload, dict):
                messages = trace["messages"] or [
                    {
                        "agent": event.get("actor", "LLM"),
                        "database": "payload file",
                        "message_id": 0,
                        "created_at": event.get("started_at")
                        or event.get("created_at"),
                        "direction": "input",
                        "content": _audit_value(payload.get("prompt", "")),
                    },
                    {
                        "agent": event.get("actor", "LLM"),
                        "database": "payload file",
                        "message_id": 1,
                        "created_at": event.get("created_at"),
                        "direction": "output",
                        "content": _audit_value(payload.get("output", "")),
                    },
                ]
                return {
                    "event": event,
                    "provenance": "full_payload_file",
                    "input": _audit_value(payload.get("prompt", "")),
                    "output": _audit_value(payload.get("output", "")),
                    "raw_response": _audit_value(payload.get("raw_response")),
                    "tool_calls": trace["tool_calls"],
                    "reasoning": trace["reasoning"],
                    "messages": messages,
                    "session_databases": trace["databases"],
                    "has_full_input": True,
                    "has_full_output": True,
                }
    return {
        "event": event,
        "provenance": "sqlite_session_trace",
        "input": "\n\n".join(trace["inputs"]) or call.get("prompt_excerpt", ""),
        "output": "\n\n".join(trace["outputs"]) or call.get("output_excerpt", ""),
        "reasoning": trace["reasoning"],
        "tool_calls": trace["tool_calls"],
        "messages": trace["messages"],
        "session_databases": trace["databases"],
        "has_full_input": bool(trace["inputs"]),
        "has_full_output": bool(trace["outputs"]),
    }


def _score_summary(room_dir: Path) -> dict[str, Any]:
    score_files = sorted(room_dir.glob("scene_renders/**/scores.yaml"))
    if not score_files:
        return {}
    try:
        import yaml

        parsed = yaml.safe_load(score_files[-1].read_text(encoding="utf-8")) or {}
    except (ImportError, OSError):
        return {}
    grades = {
        key: value.get("grade")
        for key, value in parsed.items()
        if isinstance(value, dict) and isinstance(value.get("grade"), (int, float))
    }
    return {"grades": grades, "summary": parsed.get("summary", "")}


def _agent_messages(room_dir: Path, limit: int = 150) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    for database in sorted(room_dir.glob("*_critic.db")) + sorted(
        room_dir.glob("critic.db")
    ):
        try:
            connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
            rows = connection.execute(
                "SELECT id, session_id, message_data, created_at FROM agent_messages ORDER BY id"
            ).fetchall()
            connection.close()
        except sqlite3.Error:
            continue
        for message_id, session_id, message_data, created_at in rows[-limit:]:
            messages.append(
                {
                    "id": f"{database.stem}:{message_id}",
                    "agent": database.stem,
                    "session_id": session_id,
                    "created_at": created_at,
                    "content": message_data,
                }
            )
    return messages[-limit:]


def _object_map(payload: Any) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            object_id = value.get("object_id")
            if isinstance(object_id, str):
                transform = value.get("transform", {})
                result[object_id] = {
                    "position": transform.get("translation", value.get("position")),
                    "rotation": transform.get("rotation_wxyz", value.get("rotation")),
                    "description": value.get(
                        "description", value.get("name", object_id)
                    ),
                }
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(payload)
    return result


def _scene_diff(before: Path, after: Path) -> dict[str, Any]:
    before_objects = _object_map(_read_json(before, {}))
    after_objects = _object_map(_read_json(after, {}))
    before_ids = set(before_objects)
    after_ids = set(after_objects)
    changed = []
    for object_id in sorted(before_ids & after_ids):
        old, new = before_objects[object_id], after_objects[object_id]
        if old["position"] != new["position"] or old["rotation"] != new["rotation"]:
            changed.append({"object_id": object_id, "before": old, "after": new})
    return {
        "added": [
            {"object_id": item, **after_objects[item]}
            for item in sorted(after_ids - before_ids)
        ],
        "removed": [
            {"object_id": item, **before_objects[item]}
            for item in sorted(before_ids - after_ids)
        ],
        "changed": changed,
    }


def create_app(probe_root: Path = DEFAULT_PROBE_ROOT) -> Flask:
    app = Flask(__name__)
    app.config["PROBE_ROOT"] = probe_root.resolve()

    @app.get("/api/runs")
    def runs() -> Any:
        root = app.config["PROBE_ROOT"]
        if not root.is_dir():
            return _json_response({"runs": []})
        records = []
        for child in sorted(
            (item for item in root.iterdir() if item.is_dir()),
            key=lambda item: item.stat().st_mtime,
            reverse=True,
        ):
            rooms = _room_directories(child)
            records.append(
                {
                    "id": child.name,
                    "scene_count": len(rooms),
                    "updated_at": _iso_mtime(child),
                    "status": (
                        "complete"
                        if any(
                            (room / "scene_states" / "final_scene").exists()
                            for room in rooms
                        )
                        else "running"
                    ),
                    "modes": sorted(
                        {_scene_identity(child, room)["mode"] for room in rooms}
                    ),
                }
            )
        return _json_response({"runs": records})

    @app.get("/api/runs/<run_id>/scenes")
    def scenes(run_id: str) -> Any:
        root = app.config["PROBE_ROOT"]
        run_root = _safe_relative_path(root, run_id)
        if not run_root.is_dir() or run_root.parent != root:
            abort(404)
        records = []
        for room in _room_directories(run_root):
            identity = _scene_identity(run_root, room)
            timings = _timing_records(room)
            stages = sorted(
                {str(row.get("stage")) for row in timings if row.get("stage")}
            )
            relative_path = str(room.relative_to(root))
            records.append(
                {
                    "id": relative_path,
                    "path": relative_path,
                    **identity,
                    "status": (
                        "complete"
                        if (room / "scene_states" / "final_scene").exists()
                        else "running"
                    ),
                    "stages": stages,
                    "event_count": len(timings),
                    "updated_at": _iso_mtime(room),
                    "score_summary": _score_summary(room).get("grades", {}),
                }
            )
        return _json_response({"scenes": records})

    @app.get("/api/scene")
    def scene_detail() -> Any:
        root = app.config["PROBE_ROOT"]
        relative_path = request.args.get("path", "")
        room = _safe_relative_path(root, relative_path)
        if not _is_room_directory(room):
            abort(404)
        action_log = _action_records(room)
        timings = _timing_records(room)
        llm_calls = _llm_records(room)
        audit_events = _audit_events(room)
        return _json_response(
            {
                "path": relative_path,
                "actions": action_log,
                "timings": timings,
                "llm_calls": llm_calls,
                "renders": _render_records(root, room),
                "score_summary": _score_summary(room),
                "messages": _agent_messages(room),
                "event_counts": Counter(row.get("stage", "unknown") for row in timings),
                "audit_events": audit_events,
            }
        )

    @app.get("/api/audit-event")
    def audit_event() -> Any:
        root = app.config["PROBE_ROOT"]
        relative_path = request.args.get("path", "")
        event_id = request.args.get("event_id", "")
        room = _safe_relative_path(root, relative_path)
        if not _is_room_directory(room):
            abort(404)
        if event_id.startswith("llm:"):
            payload = _payload_for_llm_event(room, event_id)
            if payload is None:
                abort(404)
            return _json_response(payload)
        if event_id.startswith("tool-action:"):
            try:
                step_number = int(event_id.removeprefix("tool-action:"))
            except ValueError:
                abort(404)
            action = next(
                (
                    item
                    for item in _action_records(room)
                    if item.get("step_number") == step_number
                ),
                None,
            )
            if action is None:
                abort(404)
            event = {
                "id": event_id,
                "kind": "tool",
                "source": "action_log",
                "created_at": action.get("timestamp"),
                "stage": _stage_for_action(action),
                "actor": "designer",
                "function": action.get("tool_name", "tool_action"),
                "title": str(action.get("tool_name", "tool action")).replace("_", " "),
                "audit_status": "tool_action",
                "detail": action,
            }
            return _json_response(
                {
                    "event": event,
                    "provenance": "action_log",
                    "input": "",
                    "output": "",
                    "reasoning": [],
                    "tool_calls": [],
                    "action": action,
                    "selection_trace": _selection_trace_for_action(room, action),
                }
            )
        event = next(
            (item for item in _audit_events(room) if item["id"] == event_id), None
        )
        if event is None:
            abort(404)
        return _json_response(
            {
                "event": event,
                "provenance": event["source"],
                "input": "",
                "output": "",
                "reasoning": [],
                "tool_calls": [],
                "metrics": event.get("metrics", event.get("detail", {})),
            }
        )

    @app.get("/api/diff")
    def diff() -> Any:
        root = app.config["PROBE_ROOT"]
        before = _safe_relative_path(root, request.args.get("before", ""))
        after = _safe_relative_path(root, request.args.get("after", ""))
        if before.name != "scene_state.json" or after.name != "scene_state.json":
            abort(404)
        if not before.is_file() or not after.is_file():
            abort(404)
        return _json_response(_scene_diff(before, after))

    @app.get("/api/image")
    def image() -> Any:
        root = app.config["PROBE_ROOT"]
        file_path = _safe_relative_path(root, request.args.get("path", ""))
        if not file_path.is_file() or file_path.suffix.lower() not in IMAGE_SUFFIXES:
            abort(404)
        return send_file(file_path, conditional=True, max_age=0)

    @app.get("/api/asset-thumbnail")
    def asset_thumbnail() -> Any:
        file_path = _safe_hssd_rendered_image(request.args.get("path", ""))
        try:
            requested_width = int(request.args.get("width", THUMBNAIL_MAX_WIDTH))
        except ValueError:
            requested_width = THUMBNAIL_MAX_WIDTH
        width = min(max(requested_width, 96), THUMBNAIL_MAX_WIDTH)
        try:
            thumbnail = _thumbnail_bytes(
                str(file_path), file_path.stat().st_mtime_ns, width
            )
        except (OSError, ValueError):
            abort(404)
        return send_file(
            BytesIO(thumbnail),
            mimetype="image/jpeg",
            conditional=True,
            max_age=86400,
        )

    return app


if __name__ == "__main__":
    _release_previous_probe_server(SERVER_PORT)
    create_app().run(host=SERVER_HOST, port=SERVER_PORT, debug=False)
