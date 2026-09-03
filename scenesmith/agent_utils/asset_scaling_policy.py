"""Experiment-level policy for HSSD fitting and agent runtime rescaling."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, TypeVar


_T = TypeVar("_T")


def _config_get(config: Any, key: str, default: _T) -> Any | _T:
    """Read a key from dictionaries, DictConfig objects, or simple namespaces."""
    if config is None:
        return default
    if isinstance(config, Mapping):
        return config.get(key, default)
    get = getattr(config, "get", None)
    if callable(get):
        return get(key, default)
    return getattr(config, key, default)


@dataclass(frozen=True)
class AssetScalingPolicy:
    """Effective asset-scaling behavior after applying the master switch."""

    enabled: bool = True
    hssd_dimension_fit_enabled: bool = True
    agent_rescale_tools_enabled: bool = True

    @classmethod
    def from_experiment_config(cls, experiment_config: Any) -> "AssetScalingPolicy":
        """Resolve the new policy while honoring the legacy HSSD-only switch."""
        legacy_hssd_fit = bool(
            _config_get(
                experiment_config,
                "hssd_scale_to_requested_dimensions",
                True,
            )
        )
        policy_config = _config_get(experiment_config, "asset_scaling", None)
        if policy_config is None:
            return cls(hssd_dimension_fit_enabled=legacy_hssd_fit)

        enabled = bool(_config_get(policy_config, "enabled", True))
        hssd_fit = bool(
            _config_get(
                policy_config,
                "hssd_dimension_fit_enabled",
                legacy_hssd_fit,
            )
        )
        agent_tools = bool(
            _config_get(policy_config, "agent_rescale_tools_enabled", True)
        )
        return cls(
            enabled=enabled,
            hssd_dimension_fit_enabled=enabled and hssd_fit,
            agent_rescale_tools_enabled=enabled and agent_tools,
        )

    @classmethod
    def from_agent_config(cls, agent_config: Any) -> "AssetScalingPolicy":
        """Read the already-resolved policy propagated into an agent subtree."""
        policy_config = _config_get(agent_config, "asset_scaling", None)
        if policy_config is None:
            return cls()
        enabled = bool(_config_get(policy_config, "enabled", True))
        return cls(
            enabled=enabled,
            hssd_dimension_fit_enabled=enabled
            and bool(
                _config_get(policy_config, "hssd_dimension_fit_enabled", True)
            ),
            agent_rescale_tools_enabled=enabled
            and bool(
                _config_get(policy_config, "agent_rescale_tools_enabled", True)
            ),
        )

    def as_agent_config(self) -> dict[str, bool]:
        """Return an auditable configuration subtree with effective values."""
        return {
            "enabled": self.enabled,
            "hssd_dimension_fit_enabled": self.hssd_dimension_fit_enabled,
            "agent_rescale_tools_enabled": self.agent_rescale_tools_enabled,
        }


def agent_rescale_tools_enabled(agent_config: Any) -> bool:
    """Whether an agent may expose and execute runtime rescale tools."""
    return AssetScalingPolicy.from_agent_config(
        agent_config
    ).agent_rescale_tools_enabled


def filter_agent_rescale_tools(
    tools: Mapping[str, _T], agent_config: Any, *, tool_names: set[str] | frozenset[str]
) -> dict[str, _T]:
    """Remove stage-specific runtime rescale tools when policy disables them."""
    if agent_rescale_tools_enabled(agent_config):
        return dict(tools)
    return {name: tool for name, tool in tools.items() if name not in tool_names}
