"""Stable experiment identity helpers for traces and paired evaluation."""

from __future__ import annotations

import hashlib
import json

from typing import Any

_VOLATILE_CONFIG_KEYS = frozenset(
    {
        "api_key",
        "base_url",
        "debug_dir",
        "host",
        "log_dir",
        "memory_dir",
        "openai_api_key",
        "openai_base_url",
        "output_dir",
        "output_root",
        "port",
        "port_base",
        "resume_from_path",
        "run_id",
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

    def volatile_key(key: object) -> bool:
        normalized = str(key).casefold()
        return normalized in _VOLATILE_CONFIG_KEYS or normalized.endswith(
            ("_dir", "_path", "_port", "_port_range", "_root")
        )

    def sanitize(value: Any, *, root: bool = False) -> Any:
        if isinstance(value, dict):
            return {
                str(key): sanitize(item)
                for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
                if not volatile_key(key)
                and not (root and str(key).casefold() == "name")
            }
        if isinstance(value, (list, tuple)):
            return [sanitize(item) for item in value]
        return value

    payload = json.dumps(sanitize(cfg_dict, root=True), sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
