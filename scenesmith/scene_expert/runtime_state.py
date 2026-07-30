"""Durable SceneExpert runtime states that cross worker-process boundaries."""

from __future__ import annotations

import hashlib
import json
import os
import re
import time

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

SCENE_PAUSED_MARKER = "[SCENE_PAUSED_RETRYABLE]"
_STAGE_REASON_PATTERN = re.compile(r"^\[(?P<stage>[a-z_]+)\]\s*(?P<reason>.*)$")


class ScenePauseManifest(BaseModel):
    """Checkpoint required to resume one scene at its interrupted decision."""

    schema_version: str = "1.0"
    status: str = "PAUSED_RETRYABLE"
    stage: str
    role: str = "critic"
    reason: str
    resume_action: str = "retry_critic_only"
    candidate_hash: str = ""
    candidate_state_path: str = ""
    render_dir: str = ""
    attempt_count: int = 0
    created_at: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class DegradedSceneManifest(BaseModel):
    """Auditable non-success outcome that remains exportable."""

    schema_version: str = "1.0"
    status: str = "DEGRADED_INCOMPLETE"
    reasons: list[str] = Field(default_factory=list)
    created_at: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class ScenePausedError(RuntimeError):
    """Non-fatal scene outcome used to stop only the affected task."""

    def __init__(
        self,
        stage: str,
        reason: str,
        manifest_path: str = "",
    ) -> None:
        self.stage = stage
        self.reason = reason
        self.manifest_path = manifest_path
        message = f"{SCENE_PAUSED_MARKER} stage={stage}: {reason}"
        if manifest_path:
            message += f" (resume_manifest={manifest_path})"
        super().__init__(message)

    def __reduce__(self):
        return (
            type(self),
            (self.stage, self.reason, self.manifest_path),
        )


def persist_retryable_pause(
    *,
    scene_root_dir: str | Path,
    stage: str,
    reason: str,
    candidate_state: dict[str, Any] | None = None,
    candidate_hash: str = "",
    render_dir: str | Path | None = None,
    attempt_count: int = 0,
    metadata: dict[str, Any] | None = None,
    role: str = "critic",
    resume_action: str = "retry_critic_only",
) -> Path:
    """Atomically persist a stage-specific resume checkpoint for one scene."""

    scene_root = Path(scene_root_dir)
    pause_dir = scene_root / "scene_expert" / "resume"
    pause_dir.mkdir(parents=True, exist_ok=True)

    candidate_state_path = pause_dir / f"{stage}_candidate_state.json"
    if candidate_state is not None:
        _write_json_atomic(candidate_state_path, candidate_state)
    elif candidate_state_path.exists():
        candidate_state_path.unlink()

    manifest = ScenePauseManifest(
        stage=stage,
        role=role,
        reason=reason,
        resume_action=resume_action,
        candidate_hash=candidate_hash,
        candidate_state_path=(
            str(candidate_state_path) if candidate_state is not None else ""
        ),
        render_dir=str(render_dir or ""),
        attempt_count=max(0, int(attempt_count)),
        created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        metadata=dict(metadata or {}),
    )
    manifest_path = pause_dir / "pause_manifest.json"
    _write_json_atomic(manifest_path, manifest.model_dump())
    return manifest_path


def persist_degraded_incomplete(
    *,
    scene_root_dir: str | Path,
    reasons: list[str],
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Persist an exportable incomplete outcome without pausing the task."""
    scene_root = Path(scene_root_dir)
    degraded_dir = scene_root / "scene_expert" / "degraded"
    degraded_dir.mkdir(parents=True, exist_ok=True)
    manifest = DegradedSceneManifest(
        reasons=list(dict.fromkeys(str(reason) for reason in reasons if str(reason))),
        created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        metadata=dict(metadata or {}),
    )
    manifest_path = degraded_dir / "degraded_manifest.json"
    _write_json_atomic(manifest_path, manifest.model_dump())
    return manifest_path


def is_scene_paused_error(value: object) -> bool:
    return SCENE_PAUSED_MARKER in str(value or "")


def split_degraded_stage_reasons(
    reasons: list[object],
    *,
    current_stage: str,
) -> tuple[list[str], list[str], list[str]]:
    """Separate current-stage blockers from upstream diagnostic history.

    Runtime placement failures are persisted on the scene so the final outcome
    remains auditable.  That accumulated history must not be reinterpreted as a
    fresh failure of every downstream stage.
    """

    all_reasons: list[str] = []
    current_reasons: list[str] = []
    upstream_reasons: list[str] = []
    for raw_reason in reasons:
        reason = str(raw_reason or "").strip()
        if not reason or reason in all_reasons:
            continue
        all_reasons.append(reason)
        match = _STAGE_REASON_PATTERN.match(reason)
        if match is None:
            # Legacy/unscoped reasons originate from the active stage.
            current_reasons.append(reason)
            continue
        origin_stage = match.group("stage")
        if not current_stage or origin_stage == current_stage:
            current_reasons.append(reason)
        else:
            upstream_reasons.append(reason)
    return all_reasons, current_reasons, upstream_reasons


def candidate_state_hash(scene: object) -> str:
    """Hash candidate geometry and objects without mutable prompt text."""

    try:
        room_geometry = getattr(scene, "room_geometry")
        hash_objects = getattr(scene, "_hash_objects")
        payload: Any = {
            "room_geometry": room_geometry.content_hash(),
            "objects": hash_objects(),
        }
    except (AttributeError, TypeError, ValueError):
        try:
            payload = dict(getattr(scene, "to_state_dict")())
            payload.pop("text_description", None)
        except (AttributeError, TypeError, ValueError):
            return ""
    serialized = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def mark_retryable_pause_resolved(scene_root_dir: str | Path) -> Path | None:
    """Close a stale active pause after the scene later completes successfully."""

    pause_dir = Path(scene_root_dir) / "scene_expert" / "resume"
    manifest_path = pause_dir / "pause_manifest.json"
    if not manifest_path.exists():
        return None
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        payload = {"status": "PAUSED_RETRYABLE"}
    payload["status"] = "RESOLVED"
    payload["resolved_at"] = time.strftime(
        "%Y-%m-%dT%H:%M:%SZ",
        time.gmtime(),
    )
    resolved_path = pause_dir / "last_resolved_pause.json"
    _write_json_atomic(resolved_path, payload)
    manifest_path.unlink(missing_ok=True)
    return resolved_path


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    temporary.replace(path)
