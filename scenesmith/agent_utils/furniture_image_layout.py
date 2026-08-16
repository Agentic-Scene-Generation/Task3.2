"""Ground an edited furniture image and build a language-only layout contract."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import re
import tempfile
import time

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable

import yaml

from PIL import Image, ImageDraw, ImageFont

from scenesmith.agent_utils.context_image_quality import file_sha256
from scenesmith.agent_utils.grounding_dino_client import (
    GroundingDinoClient,
    GroundingDinoClientConfig,
)
from scenesmith.prompts import prompt_manager
from scenesmith.prompts.manager import PromptManager
from scenesmith.prompts.registry import FurnitureAgentPrompts
from scenesmith.utils.llm_json import parse_llm_json
from scenesmith.utils.openai import encode_image_to_base64

if TYPE_CHECKING:
    from scenesmith.agent_utils.vlm_service import VLMService


console_logger = logging.getLogger(__name__)

LAYOUT_SCHEMA_VERSION = 2
LAYOUT_PROMPT_VERSION = "2.0"
RAW_ARTIFACT_NAME = "context_grounding_raw.json"
ANNOTATED_ARTIFACT_NAME = "context_grounding_annotated.png"
LAYOUT_ARTIFACT_NAME = "context_furniture_layout.json"

_COORDINATE_PATTERN = re.compile(
    r"(?:\b(?:pixel|pixels|px|normalized|coordinate|coordinates)\b)"
    r"|(?:\b[xyz]\s*[:=]\s*[-+]?\d)"
    r"|(?:[-+]?\d+(?:\.\d+)?\s*%)"
    r"|(?:[-+]?\d+(?:\.\d+)?\s*(?:mm|cm|m|meter|meters|metre|metres|"
    r"ft|feet|inch|inches|degree|degrees|°)\b)"
    r"|(?:[\[(]\s*[-+]?\d+(?:\.\d+)?\s*,\s*[-+]?\d+(?:\.\d+)?)",
    flags=re.IGNORECASE,
)
_IMAGE_RELATIVE_PATTERN = re.compile(
    r"\b(?:foreground|background|from the viewer|viewer.?s perspective|image (?:top|bottom|left|right)|"
    r"(?:top|bottom|left|right) (?:edge|side) of (?:the )?image|image frame)\b",
    flags=re.IGNORECASE,
)
_NON_FURNITURE_NAMES = re.compile(
    r"\b(?:person|people|caption|text|wall art|poster|picture|chandelier|"
    r"ceiling lamp|table lamp|book|cup|vase|"
    r"plate|bathtub|toilet|sink|headboard|mattress|bedclothes)\b",
    flags=re.IGNORECASE,
)


_ARCHITECTURE_NAMES = {
    "ceiling",
    "door",
    "floor",
    "opening",
    "wall",
    "window",
}


def _is_non_furniture_name(name: str) -> bool:
    normalized = " ".join(name.strip().lower().split())
    return (
        normalized in _ARCHITECTURE_NAMES
        or normalized.startswith(("ceiling ", "hanging ", "wall "))
        or bool(_NON_FURNITURE_NAMES.search(normalized))
    )


@dataclass(frozen=True)
class GroundedLayoutConfig:
    """Validated orchestration controls for the optional feature."""

    enabled: bool = False
    cache: bool = True
    vocabulary_path: str = (
        "scenesmith/agent_utils/data/hssd_furniture_grounding_vocabulary.yaml"
    )
    same_region_iou_threshold: float = 0.80
    # Deprecated compatibility field; valid detections are no longer truncated.
    max_detections: int = 60
    max_coverage_regrounds: int = 1
    vision_detail: str = "high"
    reasoning_effort: str = "low"
    client: GroundingDinoClientConfig = GroundingDinoClientConfig()

    @classmethod
    def from_config(cls, cfg: Any | None) -> "GroundedLayoutConfig":
        """Read the nested config while keeping absent configurations disabled."""
        if cfg is None:
            return cls()
        enabled = _as_bool(cfg.get("enabled", False), "enabled")
        cache = _as_bool(cfg.get("cache", True), "cache")
        max_detections = _bounded_int(
            cfg.get("max_detections", 60), "max_detections", 1, 500
        )
        max_coverage_regrounds = _bounded_int(
            cfg.get("max_coverage_regrounds", 1),
            "max_coverage_regrounds",
            0,
            1,
        )
        iou = float(cfg.get("same_region_iou_threshold", 0.80))
        if not 0.0 < iou <= 1.0:
            raise ValueError("same_region_iou_threshold must be in (0, 1]")
        vision_detail = str(cfg.get("vision_detail", "high")).lower()
        if vision_detail not in {"low", "high", "auto"}:
            raise ValueError("vision_detail must be low, high, or auto")
        reasoning_effort = str(cfg.get("reasoning_effort", "low")).lower()
        if reasoning_effort not in {"none", "low", "medium", "high"}:
            raise ValueError("reasoning_effort must be none, low, medium, or high")
        return cls(
            enabled=enabled,
            cache=cache,
            vocabulary_path=str(cfg.get("vocabulary_path", cls.vocabulary_path)),
            same_region_iou_threshold=iou,
            max_detections=max_detections,
            max_coverage_regrounds=max_coverage_regrounds,
            vision_detail=vision_detail,
            reasoning_effort=reasoning_effort,
            client=GroundingDinoClientConfig.from_config(cfg),
        )


def load_grounding_vocabulary(path: Path) -> dict[str, Any]:
    """Load and audit the versioned fixed HSSD vocabulary."""
    path = Path(path)
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Grounding vocabulary must be a YAML mapping")
    groups = payload.get("grounding_phrases")
    if not isinstance(groups, dict) or not groups:
        raise ValueError("Grounding vocabulary has no grounding_phrases")
    phrases: list[str] = []
    for raw_group in groups.values():
        if not isinstance(raw_group, list):
            raise ValueError("Every grounding vocabulary group must be a list")
        phrases.extend(str(phrase).strip().lower() for phrase in raw_group)
    if any(not phrase for phrase in phrases) or len(set(phrases)) != len(phrases):
        raise ValueError("Grounding vocabulary phrases must be non-empty and unique")
    expected_count = payload.get("summary", {}).get("grounding_phrases")
    if expected_count is not None and int(expected_count) != len(phrases):
        raise ValueError("Grounding vocabulary summary count does not match contents")
    payload["flattened_phrases"] = phrases
    return payload


def merge_grounding_regions(
    detections: Iterable[dict[str, Any]],
    *,
    image_width: int,
    image_height: int,
    same_region_iou_threshold: float = 0.80,
    max_detections: int | None = None,
) -> list[dict[str, Any]]:
    """Clip, filter, merge duplicate-label boxes, and assign stable IDs.

    ``max_detections`` is retained as a compatibility argument but no longer
    truncates valid detector output.
    """
    valid: list[dict[str, Any]] = []
    for detection in detections:
        normalized = _normalize_detection(detection, image_width, image_height)
        if normalized is not None:
            valid.append(normalized)
    valid.sort(
        key=lambda item: (
            -item["score"],
            item["phrase"],
            *item["box_xyxy"],
        )
    )
    merged: list[dict[str, Any]] = []
    for detection in valid:
        matching_region = next(
            (
                region
                for region in merged
                if _same_region(
                    detection["box_xyxy"],
                    region["box_xyxy"],
                    same_region_iou_threshold,
                )
            ),
            None,
        )
        candidate = {
            "label": detection["phrase"],
            "score": detection["score"],
        }
        if matching_region is None:
            merged.append(
                {
                    "box_xyxy": detection["box_xyxy"],
                    "candidate_labels": [candidate],
                    "top_score": detection["score"],
                }
            )
            continue
        existing_labels = {
            item["label"]: item for item in matching_region["candidate_labels"]
        }
        previous = existing_labels.get(candidate["label"])
        if previous is None or candidate["score"] > previous["score"]:
            existing_labels[candidate["label"]] = candidate
        matching_region["candidate_labels"] = sorted(
            existing_labels.values(), key=lambda item: (-item["score"], item["label"])
        )
        matching_region["top_score"] = matching_region["candidate_labels"][0]["score"]

    merged.sort(
        key=lambda region: (
            (region["box_xyxy"][1] + region["box_xyxy"][3]) / 2.0,
            (region["box_xyxy"][0] + region["box_xyxy"][2]) / 2.0,
            *region["box_xyxy"],
        )
    )
    for index, region in enumerate(merged, start=1):
        region["detection_id"] = f"F{index:03d}"
    return merged


def write_grounding_annotation(
    image_path: Path,
    regions: list[dict[str, Any]],
    output_path: Path,
) -> Path:
    """Draw stable IDs and compact candidates without changing the source image."""
    with Image.open(image_path) as source:
        image = source.convert("RGB")
    draw = ImageDraw.Draw(image)
    font = _annotation_font(image.width, image.height)
    line_width = max(2, round(min(image.size) / 300))
    for region in regions:
        detection_id = region["detection_id"]
        box = tuple(round(value) for value in region["box_xyxy"])
        color = _stable_color(detection_id)
        draw.rectangle(box, outline=color, width=line_width)
        top_candidate = region["candidate_labels"][0]
        label = f"{detection_id} {top_candidate['label']} {top_candidate['score']:.2f}"
        text_box = draw.textbbox((0, 0), label, font=font)
        text_width = text_box[2] - text_box[0]
        text_height = text_box[3] - text_box[1]
        x = max(0, min(box[0], image.width - text_width - 6))
        y = box[1] - text_height - 6
        if y < 0:
            y = min(image.height - text_height - 6, box[1] + 2)
        draw.rectangle((x, y, x + text_width + 6, y + text_height + 6), fill=color)
        draw.text((x + 3, y + 3), label, fill="black", font=font)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, format="PNG")
    return output_path


def normalize_layout_response(
    payload: Any,
    regions: list[dict[str, Any]],
    *,
    max_items: int | None = None,
    quality_mode: str = "full_reference",
) -> dict[str, Any]:
    """Bind every valid VLM item to a detection and apply quality-mode redaction.

    ``max_items`` remains accepted for call-site compatibility but is intentionally
    ignored: every valid detection is retained.
    """
    if not isinstance(payload, dict):
        raise ValueError("Image-grounded layout response must be a JSON object")
    region_by_id = {region["detection_id"]: region for region in regions}
    known_ids = set(region_by_id)
    if quality_mode not in {
        "full_reference",
        "contract_only",
        "relations_only",
        "inventory_only",
    }:
        quality_mode = "inventory_only"
    raw_items = payload.get("items", [])
    if not isinstance(raw_items, list):
        raise ValueError("Image-grounded layout items must be a list")
    accepted: list[tuple[int, dict[str, Any]]] = []
    seen_ids: set[str] = set()
    for source_index, raw_item in enumerate(raw_items):
        if not isinstance(raw_item, dict):
            continue
        detection_id = str(raw_item.get("detection_id", "")).strip().upper()
        if detection_id not in known_ids or detection_id in seen_ids:
            continue
        furniture_name = _safe_relation_text(raw_item.get("furniture_name"), 80, "")
        if not furniture_name or _is_non_furniture_name(furniture_name):
            continue
        role = str(raw_item.get("role", "secondary")).strip().lower()
        if role not in {"primary_anchor", "dependent", "secondary"}:
            role = "secondary"
        approximate_position = _safe_relation_text(
            raw_item.get("approximate_position", raw_item.get("semantic_location")),
            300,
            "no reliable relation",
        )
        wall_relation = _safe_relation_text(raw_item.get("wall_relation"), 240, "")
        facing_relation = _safe_relation_text(
            raw_item.get("facing_relation", raw_item.get("facing")),
            200,
            "no reliable relation",
        )
        nearby_landmarks = _safe_text_list(raw_item.get("nearby_landmarks"), 12, 120)
        group_id = _safe_text(raw_item.get("group_id"), 80)
        notes = _safe_relation_text(raw_item.get("notes"), 240, "")
        relations: list[dict[str, str]] = []
        raw_relations = raw_item.get("relative_relations", [])
        if isinstance(raw_relations, list):
            for raw_relation in raw_relations[:20]:
                if not isinstance(raw_relation, dict):
                    continue
                target_id = str(raw_relation.get("target_id", "")).strip().upper()
                relation = _safe_relation_text(raw_relation.get("relation"), 240, "")
                if target_id in known_ids and target_id != detection_id and relation:
                    relations.append({"target_id": target_id, "relation": relation})
        raw_order = raw_item.get("placement_order")
        try:
            order = int(raw_order)
        except (TypeError, ValueError):
            order = 1_000_000 + source_index
        if quality_mode == "relations_only":
            approximate_position = "no reliable relation"
            wall_relation = ""
            nearby_landmarks = []
        elif quality_mode == "inventory_only":
            approximate_position = "no reliable relation"
            wall_relation = ""
            facing_relation = "no reliable relation"
            nearby_landmarks = []
            relations = []
            group_id = ""
        detection_confidence = float(region_by_id[detection_id].get("top_score", 0.0))
        description_confidence = _bounded_confidence(
            raw_item.get("description_confidence"),
            fallback=0.75 if quality_mode == "full_reference" else 0.45,
        )
        orientation_confidence = _bounded_confidence(
            raw_item.get("orientation_confidence"), fallback=description_confidence
        )
        accepted.append(
            (
                order,
                {
                    "detection_id": detection_id,
                    "furniture_name": furniture_name,
                    "role": role,
                    "group_id": group_id,
                    "approximate_position": approximate_position,
                    # Compatibility aliases for existing audit consumers.
                    "semantic_location": approximate_position,
                    "wall_relation": wall_relation,
                    "facing_relation": facing_relation,
                    "facing": facing_relation,
                    "nearby_landmarks": nearby_landmarks,
                    "relative_relations": relations,
                    "detection_confidence": round(detection_confidence, 4),
                    "description_confidence": description_confidence,
                    "orientation_confidence": orientation_confidence,
                    "source_quality_mode": quality_mode,
                    "notes": notes,
                },
            )
        )
        seen_ids.add(detection_id)
    accepted.sort(key=lambda pair: (pair[0], pair[1]["detection_id"]))
    items = [item for _, item in accepted]
    for placement_order, item in enumerate(items, start=1):
        item["placement_order"] = placement_order

    ignored = _unique_known_ids(payload.get("ignored_detection_ids"), known_ids)
    included_ids = {item["detection_id"] for item in items}
    ignored = [
        detection_id for detection_id in ignored if detection_id not in included_ids
    ]
    unboxed = _safe_text_list(payload.get("unboxed_visible_furniture"), 10, 80)
    unboxed = [name for name in unboxed if not _is_non_furniture_name(name)]
    coverage_notes = _safe_text_list(payload.get("coverage_notes"), 10, 240)
    return {
        "items": items,
        "ignored_detection_ids": ignored,
        "unboxed_visible_furniture": unboxed,
        "coverage_notes": coverage_notes,
    }


def format_layout_contract(layout: dict[str, Any]) -> str:
    """Render the safe language fields and their provenance/confidence."""
    items = layout.get("items", [])
    if not items:
        return ""
    lines = [
        "## Image-Grounded Furniture Layout Contract",
        "The items below are image-derived language guidance, not room coordinates. "
        "A concept image is attached only in full_reference mode. Map the guidance "
        "onto the authoritative room returned by observe_scene, and obey room "
        "boundaries, openings, collisions, and accessibility first.",
    ]
    for item in items:
        details = [item["role"].replace("_", " ")]
        if item.get("group_id"):
            details.append(f"group: {item['group_id']}")
        if item["approximate_position"]:
            details.append(item["approximate_position"])
        if item.get("wall_relation"):
            details.append(f"wall relation: {item['wall_relation']}")
        if item["facing_relation"] != "no reliable relation":
            details.append(f"facing: {item['facing_relation']}")
        if item.get("nearby_landmarks"):
            details.append("near: " + ", ".join(item["nearby_landmarks"]))
        for relation in item["relative_relations"]:
            details.append(
                f"relative to {relation['target_id']}: {relation['relation']}"
            )
        if item["notes"]:
            details.append(item["notes"])
        normalized_details = [detail.rstrip(" .;") for detail in details]
        normalized_details.append(
            "confidence: detection "
            f"{item['detection_confidence']:.2f}, description "
            f"{item['description_confidence']:.2f}; mode "
            f"{item['source_quality_mode']}"
        )
        lines.append(
            f"{item['placement_order']}. {item['detection_id']} "
            f"{item['furniture_name']} — " + "; ".join(normalized_details) + "."
        )
    # Model-authored fields were already rejected above. The fixed safety preamble
    # intentionally uses the word "coordinates" to prohibit them.
    return "\n".join(lines)


def build_grounded_furniture_layout_reference(
    *,
    image_path: Path,
    scene_prompt: str,
    cfg: Any,
    vlm_service: "VLMService",
    model: str,
    artifact_dir: Path | None = None,
    client: GroundingDinoClient | None = None,
    prompts: PromptManager | None = None,
    quality_mode: str = "full_reference",
) -> str:
    """Build the reference fail-open and preserve a complete audit artifact."""
    image_path = Path(image_path)
    artifact_dir = Path(artifact_dir or image_path.parent)
    layout_path = artifact_dir / LAYOUT_ARTIFACT_NAME
    started_at = time.monotonic()
    try:
        config = GroundedLayoutConfig.from_config(cfg)
        if not config.enabled or not image_path.is_file():
            return ""
        artifact_dir.mkdir(parents=True, exist_ok=True)
        vocabulary_path = _resolve_vocabulary_path(config.vocabulary_path)
        vocabulary = load_grounding_vocabulary(vocabulary_path)
        grounding_client = client or GroundingDinoClient(config.client)
        health = grounding_client.health()
        if not bool(health.get("ready", False)):
            raise RuntimeError("GroundingDINO sidecar is not ready")
        cache_key = _cache_key(
            image_path=image_path,
            vocabulary_path=vocabulary_path,
            health=health,
            config=config,
            vlm_model=model,
            scene_prompt=scene_prompt,
            quality_mode=quality_mode,
        )
        cached_contract = _read_cached_contract(layout_path, cache_key, config.cache)
        if cached_contract is not None:
            return cached_contract

        grounding_result = grounding_client.ground_image(
            image_path, vocabulary["flattened_phrases"]
        )
        image_width, image_height = _image_dimensions(image_path, grounding_result)
        raw_detections = list(grounding_result["detections"])
        regions = merge_grounding_regions(
            raw_detections,
            image_width=image_width,
            image_height=image_height,
            same_region_iou_threshold=config.same_region_iou_threshold,
        )
        annotated_path = artifact_dir / ANNOTATED_ARTIFACT_NAME
        raw_path = artifact_dir / RAW_ARTIFACT_NAME
        raw_audit = {
            "schema_version": LAYOUT_SCHEMA_VERSION,
            "image_path": str(image_path),
            "image_sha256": file_sha256(image_path),
            "model_health": health,
            "model": grounding_result.get("model"),
            "vocabulary_path": str(vocabulary_path),
            "vocabulary_sha256": file_sha256(vocabulary_path),
            "thresholds": {
                "box": config.client.box_threshold,
                "text": config.client.text_threshold,
                "same_region_iou": config.same_region_iou_threshold,
            },
            "batches": grounding_result.get("batches", []),
            "raw_detections": raw_detections,
            "regions": regions,
        }
        _write_json_atomic(raw_path, raw_audit)
        if not regions:
            _write_json_atomic(
                layout_path,
                _layout_audit(
                    cache_key=cache_key,
                    image_path=image_path,
                    vocabulary_path=vocabulary_path,
                    health=health,
                    config=config,
                    model=model,
                    raw_path=raw_path,
                    annotated_path=None,
                    regions=[],
                    raw_responses=[],
                    normalized_layout={"items": []},
                    contract="",
                    coverage_history=[],
                    fallback_reason="no_grounding_regions",
                    elapsed_seconds=time.monotonic() - started_at,
                    quality_mode=quality_mode,
                ),
            )
            return ""

        write_grounding_annotation(image_path, regions, annotated_path)
        analyzer = _LayoutAnalyzer(
            vlm_service=vlm_service,
            prompts=prompts or prompt_manager,
            model=model,
            config=config,
        )
        raw_responses: list[dict[str, Any]] = []
        coverage_history: list[dict[str, Any]] = []
        first_payload = analyzer.analyze(
            image_path=image_path,
            annotated_path=annotated_path,
            scene_prompt=scene_prompt,
            regions=regions,
            quality_mode=quality_mode,
        )
        raw_responses.append(first_payload)
        normalized = normalize_layout_response(
            first_payload, regions, quality_mode=quality_mode
        )

        for coverage_attempt in range(config.max_coverage_regrounds):
            missing = normalized["unboxed_visible_furniture"]
            if not missing:
                break
            history: dict[str, Any] = {
                "attempt": coverage_attempt + 1,
                "requested_categories": missing,
            }
            try:
                supplemental = grounding_client.ground_image(image_path, missing)
                supplemental_detections = list(supplemental["detections"])
                history["raw_detection_count"] = len(supplemental_detections)
                history["batches"] = supplemental.get("batches", [])
                if not supplemental_detections:
                    history["status"] = "no_new_detections"
                    coverage_history.append(history)
                    break
                candidate_raw_detections = [
                    *raw_detections,
                    *supplemental_detections,
                ]
                candidate_regions = merge_grounding_regions(
                    candidate_raw_detections,
                    image_width=image_width,
                    image_height=image_height,
                    same_region_iou_threshold=config.same_region_iou_threshold,
                )
                write_grounding_annotation(
                    image_path, candidate_regions, annotated_path
                )
                final_payload = analyzer.analyze(
                    image_path=image_path,
                    annotated_path=annotated_path,
                    scene_prompt=scene_prompt,
                    regions=candidate_regions,
                    quality_mode=quality_mode,
                )
                raw_responses.append(final_payload)
                candidate_normalized = normalize_layout_response(
                    final_payload, candidate_regions, quality_mode=quality_mode
                )
                raw_detections = candidate_raw_detections
                regions = candidate_regions
                normalized = candidate_normalized
                history["status"] = "reanalyzed"
                history["region_count"] = len(regions)
            except Exception as exc:
                try:
                    write_grounding_annotation(image_path, regions, annotated_path)
                except Exception:
                    console_logger.debug("Failed to restore annotation", exc_info=True)
                history["status"] = "error"
                history["error_type"] = type(exc).__name__
                history["error"] = str(exc)[:1000]
            coverage_history.append(history)

        raw_audit["raw_detections"] = raw_detections
        raw_audit["regions"] = regions
        raw_audit["coverage_history"] = coverage_history
        _write_json_atomic(raw_path, raw_audit)
        contract = format_layout_contract(normalized)
        _write_json_atomic(
            layout_path,
            _layout_audit(
                cache_key=cache_key,
                image_path=image_path,
                vocabulary_path=vocabulary_path,
                health=health,
                config=config,
                model=model,
                raw_path=raw_path,
                annotated_path=annotated_path,
                regions=regions,
                raw_responses=raw_responses,
                normalized_layout=normalized,
                contract=contract,
                coverage_history=coverage_history,
                fallback_reason=None if contract else "no_valid_vlm_items",
                elapsed_seconds=time.monotonic() - started_at,
                quality_mode=quality_mode,
            ),
        )
        console_logger.info(
            "Image-grounded layout: %d raw detections, %d regions, %d items in %.2fs",
            len(raw_detections),
            len(regions),
            len(normalized["items"]),
            time.monotonic() - started_at,
        )
        return contract
    except Exception as exc:
        console_logger.warning(
            "Image-grounded furniture layout failed; preserving reference image: %s",
            exc,
        )
        _write_failure_best_effort(
            layout_path,
            image_path=image_path,
            error=exc,
            elapsed_seconds=time.monotonic() - started_at,
        )
        return ""


class _LayoutAnalyzer:
    def __init__(
        self,
        *,
        vlm_service: "VLMService",
        prompts: PromptManager,
        model: str,
        config: GroundedLayoutConfig,
    ) -> None:
        self.vlm_service = vlm_service
        self.prompts = prompts
        self.model = model
        self.config = config

    def analyze(
        self,
        *,
        image_path: Path,
        annotated_path: Path,
        scene_prompt: str,
        regions: list[dict[str, Any]],
        quality_mode: str,
    ) -> dict[str, Any]:
        regions_json = json.dumps(regions, ensure_ascii=False, separators=(",", ":"))
        prompt = self.prompts.get_prompt(
            FurnitureAgentPrompts.IMAGE_GROUNDED_LAYOUT_ANALYSIS,
            scene_requirements=scene_prompt,
            detection_regions_json=regions_json,
            quality_mode=quality_mode,
        )
        original = encode_image_to_base64(image_path)
        annotated = encode_image_to_base64(annotated_path)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "text", "text": "ORIGINAL EDITED REFERENCE IMAGE:"},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{original}"},
                    },
                    {"type": "text", "text": "GROUNDING ANNOTATION IMAGE:"},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{annotated}"},
                    },
                ],
            }
        ]
        response = self.vlm_service.create_completion(
            model=self.model,
            messages=messages,
            reasoning_effort=self.config.reasoning_effort,
            verbosity="low",
            response_format={"type": "json_object"},
            vision_detail=self.config.vision_detail,
            timeout_seconds=self.config.client.timeout_seconds,
            max_retries=self.config.client.max_retries,
        )
        parsed = parse_llm_json(response)
        if not isinstance(parsed, dict):
            raise ValueError("VLM image-grounded layout response is not an object")
        return parsed


def _normalize_detection(
    detection: dict[str, Any], image_width: int, image_height: int
) -> dict[str, Any] | None:
    if not isinstance(detection, dict):
        return None
    phrase = str(detection.get("phrase", "")).strip().lower()
    box = detection.get("box_xyxy")
    try:
        score = float(detection.get("score"))
        coords = [float(value) for value in box]
    except (TypeError, ValueError):
        return None
    if not phrase or len(coords) != 4 or not math.isfinite(score):
        return None
    if not all(math.isfinite(value) for value in coords):
        return None
    x1, y1, x2, y2 = coords
    clipped = [
        max(0.0, min(float(image_width), x1)),
        max(0.0, min(float(image_height), y1)),
        max(0.0, min(float(image_width), x2)),
        max(0.0, min(float(image_height), y2)),
    ]
    if clipped[2] <= clipped[0] or clipped[3] <= clipped[1]:
        return None
    return {"phrase": phrase, "score": score, "box_xyxy": clipped}


def _same_region(first: list[float], second: list[float], threshold: float) -> bool:
    intersection = _intersection_area(first, second)
    if intersection <= 0:
        return False
    first_area = (first[2] - first[0]) * (first[3] - first[1])
    second_area = (second[2] - second[0]) * (second[3] - second[1])
    union = first_area + second_area - intersection
    iou = intersection / union if union > 0 else 0.0
    smaller_coverage = intersection / min(first_area, second_area)
    return iou >= threshold or smaller_coverage >= threshold


def _intersection_area(first: list[float], second: list[float]) -> float:
    width = max(0.0, min(first[2], second[2]) - max(first[0], second[0]))
    height = max(0.0, min(first[3], second[3]) - max(first[1], second[1]))
    return width * height


def _safe_text(value: Any, max_chars: int) -> str:
    if not isinstance(value, str):
        return ""
    return " ".join(value.strip().split())[:max_chars]


def _safe_relation_text(value: Any, max_chars: int, fallback: str) -> str:
    text = _safe_text(value, max_chars)
    if (
        not text
        or _COORDINATE_PATTERN.search(text)
        or _IMAGE_RELATIVE_PATTERN.search(text)
    ):
        return fallback
    return text


def _safe_text_list(value: Any, max_items: int, max_chars: int) -> list[str]:
    if not isinstance(value, list):
        return []
    result: list[str] = []
    for item in value[:max_items]:
        text = _safe_relation_text(item, max_chars, "")
        if text and text not in result:
            result.append(text)
    return result


def _bounded_confidence(value: Any, *, fallback: float) -> float:
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        confidence = fallback
    if not math.isfinite(confidence):
        confidence = fallback
    return round(max(0.0, min(1.0, confidence)), 4)


def _unique_known_ids(value: Any, known_ids: set[str]) -> list[str]:
    if not isinstance(value, list):
        return []
    result: list[str] = []
    for raw_id in value:
        detection_id = str(raw_id).strip().upper()
        if detection_id in known_ids and detection_id not in result:
            result.append(detection_id)
    return result


def _annotation_font(width: int, height: int) -> ImageFont.ImageFont:
    size = max(14, round(min(width, height) / 42))
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    ):
        try:
            return ImageFont.truetype(path, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def _stable_color(identifier: str) -> tuple[int, int, int]:
    digest = hashlib.sha256(identifier.encode("utf-8")).digest()
    return tuple(96 + byte % 144 for byte in digest[:3])


def _resolve_vocabulary_path(raw_path: str) -> Path:
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = Path(__file__).resolve().parents[2] / path
    if not path.is_file():
        raise FileNotFoundError(f"Grounding vocabulary not found: {path}")
    return path


def _image_dimensions(
    image_path: Path, grounding_result: dict[str, Any]
) -> tuple[int, int]:
    with Image.open(image_path) as image:
        actual = image.size
    reported = (
        int(grounding_result.get("image_width")),
        int(grounding_result.get("image_height")),
    )
    if reported != actual:
        raise ValueError(
            f"GroundingDINO image dimensions {reported} do not match {actual}"
        )
    return actual


def _cache_key(
    *,
    image_path: Path,
    vocabulary_path: Path,
    health: dict[str, Any],
    config: GroundedLayoutConfig,
    vlm_model: str,
    scene_prompt: str,
    quality_mode: str,
) -> str:
    payload = {
        "image_sha256": file_sha256(image_path),
        "grounding_model": health.get("model"),
        "vocabulary_sha256": file_sha256(vocabulary_path),
        "box_threshold": config.client.box_threshold,
        "text_threshold": config.client.text_threshold,
        "same_region_iou_threshold": config.same_region_iou_threshold,
        "max_coverage_regrounds": config.max_coverage_regrounds,
        "vision_detail": config.vision_detail,
        "reasoning_effort": config.reasoning_effort,
        "quality_mode": quality_mode,
        "scene_prompt_sha256": hashlib.sha256(scene_prompt.encode()).hexdigest(),
        "vlm_model": vlm_model,
        "layout_prompt_version": LAYOUT_PROMPT_VERSION,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _read_cached_contract(
    path: Path, expected_key: str, cache_enabled: bool
) -> str | None:
    if not cache_enabled or not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("cache_key") != expected_key:
            return None
        contract = payload.get("layout_contract")
        return contract if isinstance(contract, str) else None
    except (OSError, ValueError):
        return None


def _layout_audit(
    *,
    cache_key: str,
    image_path: Path,
    vocabulary_path: Path,
    health: dict[str, Any],
    config: GroundedLayoutConfig,
    model: str,
    raw_path: Path,
    annotated_path: Path | None,
    regions: list[dict[str, Any]],
    raw_responses: list[dict[str, Any]],
    normalized_layout: dict[str, Any],
    contract: str,
    coverage_history: list[dict[str, Any]],
    fallback_reason: str | None,
    elapsed_seconds: float,
    quality_mode: str,
) -> dict[str, Any]:
    return {
        "schema_version": LAYOUT_SCHEMA_VERSION,
        "source_quality_mode": quality_mode,
        "cache_key": cache_key,
        "input_image": {
            "path": str(image_path),
            "sha256": file_sha256(image_path),
        },
        "grounding": {
            "health": health,
            "vocabulary_path": str(vocabulary_path),
            "vocabulary_sha256": file_sha256(vocabulary_path),
            "config": asdict(config.client),
            "raw_artifact_path": str(raw_path),
            "annotation_path": str(annotated_path) if annotated_path else None,
            "regions": regions,
        },
        "vlm": {
            "model": model,
            "prompt_version": LAYOUT_PROMPT_VERSION,
            "raw_responses": raw_responses,
            "normalized": normalized_layout,
        },
        "coverage_history": coverage_history,
        "layout_contract": contract,
        "fallback_reason": fallback_reason,
        "elapsed_seconds": round(elapsed_seconds, 3),
    }


def _write_failure_best_effort(
    path: Path, *, image_path: Path, error: Exception, elapsed_seconds: float
) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        _write_json_atomic(
            path,
            {
                "schema_version": LAYOUT_SCHEMA_VERSION,
                "input_image": {"path": str(image_path)},
                "layout_contract": "",
                "fallback_reason": "analysis_error",
                "error_type": type(error).__name__,
                "error": str(error)[:1000],
                "elapsed_seconds": round(elapsed_seconds, 3),
            },
        )
    except Exception:
        console_logger.debug(
            "Failed to write grounded-layout failure audit", exc_info=True
        )


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as output:
            json.dump(payload, output, indent=2, ensure_ascii=False)
            output.write("\n")
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _as_bool(value: Any, field: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.strip().lower() in {"true", "false"}:
        return value.strip().lower() == "true"
    raise ValueError(f"grounded_layout.{field} must be a boolean")


def _bounded_int(value: Any, field: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool):
        raise ValueError(f"grounded_layout.{field} must be an integer")
    parsed = int(value)
    if not minimum <= parsed <= maximum:
        raise ValueError(
            f"grounded_layout.{field} must be between {minimum} and {maximum}"
        )
    return parsed
