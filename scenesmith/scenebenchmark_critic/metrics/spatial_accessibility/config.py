from __future__ import annotations

from typing import Any


def _params(config: Any) -> dict[str, float]:
    run = getattr(config, "run", config)
    return {
        "grid_resolution_m": float(
            getattr(run, "accessibility_grid_resolution_m", 0.05) or 0.05
        ),
        "agent_width_m": float(
            getattr(run, "accessibility_agent_width_m", 0.50) or 0.50
        ),
        "height_threshold_m": float(
            getattr(run, "accessibility_obstacle_height_threshold_m", 1.8) or 1.8
        ),
        "access_zone_depth_m": float(
            getattr(run, "accessibility_access_zone_depth_m", 0.55) or 0.55
        ),
        "pass_ratio": float(getattr(run, "accessibility_pass_ratio", 0.60) or 0.60),
        "degraded_ratio": float(
            getattr(run, "accessibility_degraded_ratio", 0.25) or 0.25
        ),
    }


def _agent_profiles(config: Any, params: dict[str, float]) -> list[dict[str, Any]]:
    run = getattr(config, "run", config)
    raw_profiles = getattr(run, "accessibility_agent_profiles", None) or []
    profiles: list[dict[str, Any]] = []
    for raw in raw_profiles:
        payload = raw.model_dump() if hasattr(raw, "model_dump") else dict(raw)
        profile_id = str(payload.get("id") or "default")
        locomotion_mode = str(payload.get("locomotion_mode") or "walk")
        profiles.append(
            {
                "id": profile_id,
                "clearance_width_m": float(
                    payload.get("clearance_width_m") or params["agent_width_m"]
                ),
                "reach_radius_m": float(payload.get("reach_radius_m") or 0.75),
                "arm_origin_height_m": float(
                    payload.get("arm_origin_height_m") or 1.10
                ),
                "locomotion_mode": locomotion_mode,
                "crouch_factor": _crouch_factor(
                    payload.get("crouch_factor"), locomotion_mode
                ),
                "eye_height_m": _optional_float(payload.get("eye_height_m")),
            }
        )
    if profiles:
        return profiles
    return [
        {
            "id": "default",
            "clearance_width_m": params["agent_width_m"],
            "reach_radius_m": 0.75,
            "arm_origin_height_m": 1.10,
            "locomotion_mode": "walk",
            "crouch_factor": 0.40,
            "eye_height_m": None,
        }
    ]


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _crouch_factor(value: Any, locomotion_mode: str) -> float:
    if value is not None:
        return _clamp(float(value), 0.0, 0.9)
    normalized = locomotion_mode.lower()
    if normalized in {"wheelchair", "wheeled"}:
        return 0.10
    return 0.40


def _clamp(value: float, lower: float, upper: float) -> float:
    return min(max(value, lower), upper)
