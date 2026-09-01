"""Stable experiment identity helpers for traces and paired evaluation."""

from __future__ import annotations

import copy
import hashlib
import json

from pathlib import Path
from typing import Any

_VOLATILE_CONFIG_KEYS = frozenset(
    {
        "api_key",
        "attempt",
        "base_url",
        "batch_id",
        "batch_index",
        "case_id",
        "debug_dir",
        "host",
        "log_dir",
        "memory_dir",
        "openai_api_key",
        "openai_base_url",
        "output_dir",
        "output_root",
        "pid",
        "port",
        "port_base",
        "resume_from_path",
        "run_id",
        "scene_id",
        "scene_ids",
        "scene_index",
        "server_url",
        "temp_dir",
        "timing_path",
    }
)

MEMORY_RETRIEVAL_EVALUATION_DIMENSION = "fast_memory_retrieval"
_SUPPORTED_CONTROLLED_DIMENSIONS = frozenset({MEMORY_RETRIEVAL_EVALUATION_DIMENSION})


def stable_config_hash(cfg_dict: dict[str, Any]) -> str:
    """Hash the complete resolved configuration for exact replay identity."""
    try:
        payload = json.dumps(cfg_dict, sort_keys=True, default=str)
    except TypeError:
        payload = repr(cfg_dict)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _volatile_key(key: object, path: tuple[str, ...], value: Any) -> bool:
    normalized = str(key).casefold()
    if normalized in {"prompt", "prompts"}:
        # Scene tasks vary per case and must not fragment a run signature.
        # Agent prompt-template selectors (for example
        # furniture_agent.agents.designer.prompt) remain semantic inputs.
        return not any(
            parent == "agents" or parent.endswith("_agent") for parent in path
        )
    if normalized == "name" and not path:
        return True
    if (
        normalized in {"name", "experiment_name", "run_name"}
        and isinstance(value, str)
        and value.casefold().startswith("critic_on_batch_")
    ):
        # Hydra's per-batch label is an execution instance, not a semantic
        # experiment choice. Agent role names remain in the signature.
        return True
    return normalized in _VOLATILE_CONFIG_KEYS or normalized.endswith(
        ("_dir", "_path", "_port", "_port_range", "_root")
    )


def _sanitize_experiment_config(
    value: Any,
    *,
    path: tuple[str, ...] = (),
) -> Any:
    if isinstance(value, dict):
        return {
            str(key): _sanitize_experiment_config(
                item,
                path=path + (str(key).casefold(),),
            )
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
            if not _volatile_key(key, path, item)
        }
    if isinstance(value, (list, tuple)):
        return [_sanitize_experiment_config(item, path=path) for item in value]
    return value


def stable_experiment_signature(cfg_dict: dict[str, Any]) -> str:
    """Hash experiment semantics while excluding machine/run-specific values."""
    payload = json.dumps(
        _sanitize_experiment_config(cfg_dict),
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def stable_control_signature(
    cfg_dict: dict[str, Any],
    *,
    controlled_dimension: str,
) -> str:
    """Hash every semantic setting except one declared treatment dimension.

    This is deliberately narrower than making arbitrary config differences
    ignorable. A Full-mode memory experiment may vary only Fast Memory
    retrieval/injection. The exact config hash and normal experiment signature
    continue to expose the arm difference, while this signature proves that all
    non-treatment semantics match.
    """
    dimension = str(controlled_dimension or "").strip().casefold()
    if dimension not in _SUPPORTED_CONTROLLED_DIMENSIONS:
        raise ValueError(f"Unsupported controlled dimension: {controlled_dimension!r}")
    controlled = copy.deepcopy(cfg_dict)
    for scene_expert_cfg in (
        controlled.get("scene_expert"),
        (controlled.get("experiment") or {}).get("scene_expert"),
    ):
        if not isinstance(scene_expert_cfg, dict):
            continue
        # Evaluation metadata is audited independently and must not make the
        # two otherwise identical arms appear semantically different.
        scene_expert_cfg.pop("evaluation", None)
        components = scene_expert_cfg.get("components")
        if isinstance(components, dict):
            components.pop(MEMORY_RETRIEVAL_EVALUATION_DIMENSION, None)
    return stable_experiment_signature(controlled)


def shared_base_scene_identity(
    resume_root: str | Path,
    *,
    scene_index: int,
) -> dict[str, Any]:
    """Fingerprint the exact reused scene checkpoint without hashing its path."""
    root = Path(resume_root).resolve()
    scene_dir = root / f"scene_{int(scene_index):03d}"
    if not scene_dir.is_dir():
        return {
            "schema_version": "sceneexpert.shared_base_snapshot.v1",
            "fingerprint": "",
            "file_count": 0,
            "scene_dir": str(scene_dir),
        }
    digest = hashlib.sha256()
    file_count = 0
    for path in sorted(
        (candidate for candidate in scene_dir.rglob("*") if candidate.is_file()),
        key=lambda candidate: candidate.relative_to(scene_dir).as_posix(),
    ):
        relative = path.relative_to(scene_dir).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        try:
            with path.open("rb") as stream:
                while chunk := stream.read(1024 * 1024):
                    digest.update(chunk)
        except OSError:
            return {
                "schema_version": "sceneexpert.shared_base_snapshot.v1",
                "fingerprint": "",
                "file_count": file_count,
                "scene_dir": str(scene_dir),
            }
        digest.update(b"\0")
        file_count += 1
    fingerprint = digest.hexdigest() if file_count else ""
    return {
        "schema_version": "sceneexpert.shared_base_snapshot.v1",
        "fingerprint": (
            f"sceneexpert.shared_base_snapshot.v1:{fingerprint[:24]}"
            if fingerprint
            else ""
        ),
        "file_count": file_count,
        "scene_dir": str(scene_dir),
    }


def stable_source_bundle_hash(source_hashes: dict[str, str]) -> str:
    """Hash loaded source bytes without requiring a Git checkout or revision."""
    canonical = {
        str(path).replace("\\", "/"): str(digest)
        for path, digest in source_hashes.items()
        if str(path) and str(digest)
    }
    if not canonical:
        return ""
    payload = json.dumps(canonical, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
