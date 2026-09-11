"""Attempt start evidence that survives native checkpoint directory replacement."""

from __future__ import annotations

import hashlib
import json
import logging
import os

from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


def scene_attempt_start(
    *,
    output_dir: Path,
    scene_id: int,
    attempt: int,
    prompt: str,
    run_id: str | None,
    started_at: str | None = None,
) -> str:
    """Record a worker start or recover it for its terminal status, fail open.

    Records live outside ``scene_NNN``: reuse replaces that directory and native
    retries archive it. Match run, task and attempt; never reuse a stale start or
    infer one from file timestamps. A new worker explicitly replaces its record.
    Timing failures must not change scene-generation decisions or mask errors.
    """
    path = (
        Path(output_dir)
        / "scene_attempt_timing"
        / f"scene_{scene_id:03d}_attempt_{attempt:02d}.json"
    )
    identity = {
        "schema_version": "scene-attempt-start.v1",
        "scene_id": scene_id,
        "attempt": attempt,
        "run_id": run_id or "",
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
    }
    try:
        if started_at is not None:
            parsed = datetime.fromisoformat(started_at.replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                raise ValueError("Attempt start must include a timezone")
            payload = {**identity, "started_at": started_at, "pid": os.getpid()}
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
            temporary.write_text(json.dumps(payload) + "\n", encoding="utf-8")
            temporary.replace(path)
            return started_at
        if not path.is_file():
            return ""
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or any(
            payload.get(key) != value for key, value in identity.items()
        ):
            raise ValueError("Attempt start identity mismatch")
        value = payload.get("started_at", "")
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            raise ValueError("Attempt start must include a timezone")
        return value
    except (OSError, ValueError, TypeError, AttributeError) as exc:
        logger.warning("Attempt timing unavailable for %s: %s", path.name, exc)
        return ""
