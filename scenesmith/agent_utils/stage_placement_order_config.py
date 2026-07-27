"""Configuration helpers for stage-specific placement-order references."""

from __future__ import annotations

import logging

from typing import Any

from omegaconf import DictConfig, OmegaConf

console_logger = logging.getLogger(__name__)

_LEGACY_FURNITURE_KEYS = (
    "enabled",
    "cache",
    "fallback_enabled",
    "enable_thinking",
)


def append_placement_order_reference(instruction: str, reference: str) -> str:
    """Append a non-empty soft reference while preserving an exact empty no-op."""
    if not reference:
        return instruction
    return f"{instruction}\n\n{reference}"


def get_stage_placement_order_config(
    cfg: DictConfig | dict[str, Any],
    stage: str,
) -> dict[str, Any]:
    """Return one stage's local config, failing closed on malformed input.

    Legacy flat keys apply to furniture only. Non-null legacy values override
    the nested furniture values so historical Hydra overrides remain effective
    even though the base config now contains nested defaults.
    """
    if stage not in {"furniture", "manipuland"}:
        console_logger.warning(
            "Unknown placement-order stage %r; treating it as disabled", stage
        )
        return {"enabled": False}

    try:
        root = _to_dict(_select_root(cfg))
    except Exception as exc:
        console_logger.warning(
            "Failed to read %s placement-order config; treating it as disabled: %s",
            stage,
            exc,
        )
        return {"enabled": False}

    if not root:
        return {"enabled": False}

    stage_cfg = _to_dict(root.get(stage))
    stage_cfg.setdefault("enabled", False)

    if stage == "furniture":
        _apply_legacy_furniture_overrides(stage_cfg, root)

    return stage_cfg


def _select_root(cfg: DictConfig | dict[str, Any]) -> Any:
    if isinstance(cfg, DictConfig):
        return OmegaConf.select(cfg, "stage_placement_order")
    if isinstance(cfg, dict):
        return cfg.get("stage_placement_order")
    return None


def _to_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, DictConfig):
        converted = OmegaConf.to_container(value, resolve=True)
        return dict(converted) if isinstance(converted, dict) else {}
    return dict(value) if isinstance(value, dict) else {}


def _apply_legacy_furniture_overrides(
    stage_cfg: dict[str, Any],
    root: dict[str, Any],
) -> None:
    mappings = {key: key for key in _LEGACY_FURNITURE_KEYS}
    mappings["max_items_per_stage"] = "max_items"

    for legacy_key, nested_key in mappings.items():
        legacy_value = root.get(legacy_key)
        if legacy_value is None:
            continue
        nested_value = stage_cfg.get(nested_key)
        if nested_value is not None and nested_value != legacy_value:
            console_logger.warning(
                "Conflicting placement-order config for furniture.%s=%r and "
                "legacy %s=%r; using the legacy value",
                nested_key,
                nested_value,
                legacy_key,
                legacy_value,
            )
        stage_cfg[nested_key] = legacy_value
