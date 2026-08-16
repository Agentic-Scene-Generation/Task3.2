"""VLM quality gate for furniture context-image editing."""

import hashlib
import json
import os
import tempfile

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from PIL import Image

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
    rendering_style_preserved: bool
    furniture_inside_room: bool
    openings_clear: bool
    furniture_quality_ok: bool
    architecture_score: float
    style_score: float
    layout_utility_score: float
    grounding_utility_score: float
    grounding_quality_mode: str
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
                "rendering_style_preserved",
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

        # Furniture completeness/quality is deliberately a utility signal, not a
        # structural invariant.  A sparse but geometrically faithful candidate can
        # still be useful for grounding and must not be discarded here.
        structurally_eligible = all(
            criteria[name]
            for name in (
                "doors_windows_preserved",
                "room_geometry_preserved",
                "view_preserved",
                "rendering_style_preserved",
                "furniture_inside_room",
            )
        )
        architecture_score = _optional_score(
            payload,
            "architecture_score",
            (
                100.0
                if criteria["doors_windows_preserved"]
                and criteria["room_geometry_preserved"]
                and criteria["view_preserved"]
                else 25.0
            ),
        )
        style_score = _optional_score(
            payload,
            "style_score",
            100.0 if criteria["rendering_style_preserved"] else 20.0,
        )
        layout_utility_score = _optional_score(
            payload,
            "layout_utility_score",
            quality_score if criteria["furniture_quality_ok"] else quality_score * 0.6,
        )
        grounding_utility_score = _optional_score(
            payload,
            "grounding_utility_score",
            layout_utility_score,
        )
        grounding_quality_mode = _grounding_quality_mode(
            criteria=criteria,
            grounding_utility_score=grounding_utility_score,
        )
        fallback_score_floor = min(40.0, float(min_score))
        fallback_eligible = (
            structurally_eligible and quality_score >= fallback_score_floor
        )
        normalized_passed = structurally_eligible and quality_score >= float(min_score)
        return cls(
            passed=normalized_passed,
            model_passed=model_passed,
            fallback_eligible=fallback_eligible,
            quality_score=quality_score,
            architecture_score=architecture_score,
            style_score=style_score,
            layout_utility_score=layout_utility_score,
            grounding_utility_score=grounding_utility_score,
            grounding_quality_mode=grounding_quality_mode,
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


def evaluate_context_image_deterministic(
    original_image_path: Path, candidate_image_path: Path
) -> dict[str, Any]:
    """Run cheap fail-closed checks before trusting a visual reference."""
    with Image.open(original_image_path) as original_source:
        original = np.asarray(original_source.convert("RGB"), dtype=np.uint8)
    with Image.open(candidate_image_path) as candidate_source:
        candidate = np.asarray(candidate_source.convert("RGB"), dtype=np.uint8)
    canvas_size_ok = original.shape == candidate.shape
    if not canvas_size_ok:
        return {
            "passed": False,
            "canvas_size_ok": False,
            "content_nonempty": bool(candidate.size),
            "background_preserved": False,
            "dynamic_range_ok": False,
            "reasons": ["candidate canvas size differs from authoritative render"],
        }
    candidate_brightness = np.max(candidate, axis=2)
    original_background = np.max(original, axis=2) <= 8
    content_nonempty = float(np.mean(candidate_brightness > 8)) >= 0.01
    if np.any(original_background):
        background_preserved = (
            float(np.mean(candidate_brightness[original_background] <= 16)) >= 0.90
        )
    else:
        background_preserved = True
    dynamic_range_ok = int(candidate.max()) - int(candidate.min()) >= 3
    reasons = []
    if not content_nonempty:
        reasons.append("candidate is almost entirely black")
    if not background_preserved:
        reasons.append("authoritative black exterior/background was repainted")
    if not dynamic_range_ok:
        reasons.append("candidate is flat or solid-color")
    return {
        "passed": all(
            (canvas_size_ok, content_nonempty, background_preserved, dynamic_range_ok)
        ),
        "canvas_size_ok": canvas_size_ok,
        "content_nonempty": content_nonempty,
        "background_preserved": background_preserved,
        "dynamic_range_ok": dynamic_range_ok,
        "reasons": reasons,
    }


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


def _optional_score(payload: dict[str, Any], field: str, fallback: float) -> float:
    """Parse an optional 0-100 score while accepting legacy judge responses."""
    value = payload.get(field, fallback)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Context-image quality field {field!r} must be a number")
    score = float(value)
    if not 0.0 <= score <= 100.0:
        raise ValueError(
            f"Context-image quality field {field!r} must be between 0 and 100"
        )
    return score


def _grounding_quality_mode(
    *, criteria: dict[str, bool], grounding_utility_score: float
) -> str:
    """Choose the safest information level that may be extracted from a candidate."""
    architecture_reliable = all(
        criteria[name]
        for name in (
            "doors_windows_preserved",
            "room_geometry_preserved",
            "view_preserved",
        )
    )
    if (
        architecture_reliable
        and criteria["rendering_style_preserved"]
        and criteria["furniture_inside_room"]
        and grounding_utility_score >= 40.0
    ):
        return "full_reference"
    if architecture_reliable and criteria["furniture_inside_room"]:
        return "contract_only"
    if grounding_utility_score >= 40.0:
        return "relations_only"
    if grounding_utility_score >= 10.0:
        return "inventory_only"
    return "none"


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
