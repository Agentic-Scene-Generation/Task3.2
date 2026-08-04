"""VLM quality gate for furniture context-image editing."""

import hashlib
import json
import os
import tempfile

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from scenesmith.prompts import prompt_manager
from scenesmith.prompts.manager import PromptManager
from scenesmith.prompts.registry import ImageGenerationPrompts
from scenesmith.utils.openai import encode_image_to_base64

if TYPE_CHECKING:
    from scenesmith.agent_utils.vlm_service import VLMService


_MAX_REGENERATIONS = 10
_MAX_REASONS = 5
_MAX_REASON_CHARS = 300


@dataclass(frozen=True)
class ContextImageQualityGateConfig:
    """Validated controls for optional context-image quality gating."""

    enabled: bool = False
    max_regenerations: int = 2
    min_score: float = 60.0

    @classmethod
    def from_config(cls, cfg: Any | None) -> "ContextImageQualityGateConfig":
        """Read the nested quality_gate config without requiring OmegaConf."""
        if cfg is None:
            return cls()
        enabled = _as_bool(cfg.get("enabled", False), field="enabled")
        raw_max_regenerations = cfg.get("max_regenerations", 2)
        if isinstance(raw_max_regenerations, bool):
            raise ValueError(
                "context_image_generation.quality_gate.max_regenerations must "
                "be an integer"
            )
        if isinstance(raw_max_regenerations, int):
            max_regenerations = raw_max_regenerations
        elif isinstance(raw_max_regenerations, str):
            max_regenerations = int(raw_max_regenerations.strip())
        else:
            raise ValueError(
                "context_image_generation.quality_gate.max_regenerations must "
                "be an integer"
            )
        if not 0 <= max_regenerations <= _MAX_REGENERATIONS:
            raise ValueError(
                "context_image_generation.quality_gate.max_regenerations must "
                f"be between 0 and {_MAX_REGENERATIONS}"
            )
        raw_min_score = cfg.get("min_score", 60)
        if isinstance(raw_min_score, bool):
            raise ValueError(
                "context_image_generation.quality_gate.min_score must be a number"
            )
        if isinstance(raw_min_score, (int, float)):
            min_score = float(raw_min_score)
        elif isinstance(raw_min_score, str):
            min_score = float(raw_min_score.strip())
        else:
            raise ValueError(
                "context_image_generation.quality_gate.min_score must be a number"
            )
        if not 0.0 <= min_score <= 100.0:
            raise ValueError(
                "context_image_generation.quality_gate.min_score must be between "
                "0 and 100"
            )
        return cls(
            enabled=enabled,
            max_regenerations=max_regenerations,
            min_score=min_score,
        )

    @property
    def max_attempts(self) -> int:
        """Return the initial edit plus the permitted regenerations."""
        return 1 + self.max_regenerations


@dataclass(frozen=True)
class ContextImageQualityResult:
    """Normalized result of one original-versus-candidate VLM comparison."""

    passed: bool
    model_passed: bool
    fallback_eligible: bool
    quality_score: float
    doors_windows_preserved: bool
    room_geometry_preserved: bool
    view_preserved: bool
    furniture_inside_room: bool
    openings_clear: bool
    furniture_quality_ok: bool
    reasons: tuple[str, ...]
    raw_response: dict[str, Any]

    @classmethod
    def from_response_text(
        cls,
        response_text: str,
        *,
        min_score: float = 60.0,
    ) -> "ContextImageQualityResult":
        """Parse and normalize a VLM JSON response."""
        payload = json.loads(str(response_text).strip())
        if not isinstance(payload, dict):
            raise ValueError("Context-image quality response must be a JSON object")

        criteria = {
            name: _require_bool(payload, name)
            for name in (
                "doors_windows_preserved",
                "room_geometry_preserved",
                "view_preserved",
                "furniture_inside_room",
                "openings_clear",
                "furniture_quality_ok",
            )
        }
        model_passed = _require_bool(payload, "passed")
        raw_quality_score = payload.get("quality_score")
        if isinstance(raw_quality_score, bool) or not isinstance(
            raw_quality_score, (int, float)
        ):
            raise ValueError(
                "Context-image quality field 'quality_score' must be a number"
            )
        quality_score = float(raw_quality_score)
        if not 0.0 <= quality_score <= 100.0:
            raise ValueError("Context-image quality_score must be between 0 and 100")

        raw_reasons = payload.get("reasons")
        if not isinstance(raw_reasons, list) or not all(
            isinstance(reason, str) for reason in raw_reasons
        ):
            raise ValueError(
                "Context-image quality field 'reasons' must be a list of strings"
            )
        reasons = tuple(
            reason.strip()[:_MAX_REASON_CHARS]
            for reason in raw_reasons[:_MAX_REASONS]
            if reason.strip()
        )

        structurally_eligible = all(
            criteria[name]
            for name in (
                "doors_windows_preserved",
                "room_geometry_preserved",
                "view_preserved",
                "furniture_inside_room",
                "furniture_quality_ok",
            )
        )
        fallback_score_floor = min(40.0, float(min_score))
        fallback_eligible = (
            structurally_eligible and quality_score >= fallback_score_floor
        )
        normalized_passed = (
            structurally_eligible and quality_score >= float(min_score)
        )
        return cls(
            passed=normalized_passed,
            model_passed=model_passed,
            fallback_eligible=fallback_eligible,
            quality_score=quality_score,
            reasons=reasons,
            raw_response=payload,
            **criteria,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        payload = asdict(self)
        payload["reasons"] = list(self.reasons)
        return payload


class ContextImageQualityEvaluator:
    """Compare an original room render and edited candidate with a VLM."""

    def __init__(
        self,
        vlm_service: "VLMService",
        prompts: PromptManager | None = None,
    ) -> None:
        self.vlm_service = vlm_service
        self.prompt_manager = prompts or prompt_manager

    def evaluate(
        self,
        *,
        original_image_path: Path,
        candidate_image_path: Path,
        scene_description: str,
        model: str,
        min_score: float = 60.0,
        reasoning_effort: str = "none",
        verbosity: str = "low",
    ) -> ContextImageQualityResult:
        """Submit both images in a fixed order and normalize the JSON result."""
        original_image_path = Path(original_image_path)
        candidate_image_path = Path(candidate_image_path)
        prompt = self.prompt_manager.get_prompt(
            ImageGenerationPrompts.FURNITURE_CONTEXT_IMAGE_QUALITY,
            scene_description=scene_description,
            min_score=min_score,
        )
        original_base64 = encode_image_to_base64(original_image_path)
        candidate_base64 = encode_image_to_base64(candidate_image_path)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "text", "text": "ORIGINAL EMPTY ROOM:"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{original_base64}"
                        },
                    },
                    {"type": "text", "text": "EDITED CANDIDATE:"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{candidate_base64}"
                        },
                    },
                ],
            }
        ]
        response_text = self.vlm_service.create_completion(
            model=model,
            messages=messages,
            reasoning_effort=reasoning_effort,
            verbosity=verbosity,
            response_format={"type": "json_object"},
            vision_detail="high",
        )
        return ContextImageQualityResult.from_response_text(
            response_text,
            min_score=min_score,
        )


def file_sha256(path: Path) -> str:
    """Return a stable SHA-256 hash for a local artifact."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_context_image_quality_report(path: Path, payload: dict[str, Any]) -> None:
    """Atomically persist the quality-gate report."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as temporary_file:
            json.dump(payload, temporary_file, ensure_ascii=False, indent=2)
            temporary_file.write("\n")
            temporary_path = Path(temporary_file.name)
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _require_bool(payload: dict[str, Any], field: str) -> bool:
    value = payload.get(field)
    if not isinstance(value, bool):
        raise ValueError(f"Context-image quality field {field!r} must be a boolean")
    return value


def _as_bool(value: Any, *, field: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "on"}:
            return True
        if normalized in {"false", "0", "no", "off"}:
            return False
    raise ValueError(f"context_image_generation.quality_gate.{field} must be a boolean")
