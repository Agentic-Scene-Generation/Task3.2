"""Stable experiment identity helpers for traces and paired evaluation."""

from __future__ import annotations

import hashlib
import json

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


def stable_config_hash(cfg_dict: dict[str, Any]) -> str:
    """Hash the complete resolved configuration for exact replay identity."""
    try:
        payload = json.dumps(cfg_dict, sort_keys=True, default=str)
    except TypeError:
        payload = repr(cfg_dict)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def stable_experiment_signature(cfg_dict: dict[str, Any]) -> str:
    """Hash experiment semantics while excluding machine/run-specific values."""

    def volatile_key(key: object, path: tuple[str, ...]) -> bool:
        normalized = str(key).casefold()
        if normalized in {"prompt", "prompts"}:
            # Scene tasks vary per case and must not fragment a run signature.
            # Agent prompt-template selectors (for example
            # furniture_agent.agents.designer.prompt) remain semantic inputs.
            return not any(
                parent == "agents" or parent.endswith("_agent") for parent in path
            )
        return normalized in _VOLATILE_CONFIG_KEYS or normalized.endswith(
            ("_dir", "_path", "_port", "_port_range", "_root")
        )

    def sanitize(
        value: Any,
        *,
        root: bool = False,
        path: tuple[str, ...] = (),
    ) -> Any:
        if isinstance(value, dict):
            return {
                str(key): sanitize(item, path=path + (str(key).casefold(),))
                for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
                if not volatile_key(key, path)
                and not (root and str(key).casefold() == "name")
            }
        if isinstance(value, (list, tuple)):
            return [sanitize(item, path=path) for item in value]
        return value

    payload = json.dumps(sanitize(cfg_dict, root=True), sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


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
