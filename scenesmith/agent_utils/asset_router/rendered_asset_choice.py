"""VLM-assisted choice among rendered HSSD retrieval candidates."""

import base64
import io
import json
import logging
import os
import re
import time
from datetime import datetime, timezone

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from scenesmith.utils.llm_json import parse_llm_json_object
from scenesmith.agent_utils.semantic_names import normalize_semantic_name
from PIL import Image, ImageDraw

if TYPE_CHECKING:
    from scenesmith.agent_utils.hssd_retrieval_server.dataclasses import (
        HssdRetrievalResult,
    )
    from scenesmith.agent_utils.vlm_service import VLMService


console_logger = logging.getLogger(__name__)

_FLOOR_COVERING_TOKENS = {
    "bathmat",
    "carpet",
    "doormat",
    "floor_mat",
    "mat",
    "rug",
    "runner",
}

_ROUND_FOOTPRINT_TOKENS = {"circle", "circular", "oval", "round"}
_SQUARE_FOOTPRINT_MAX_ASPECT_RATIO = 1.30
_HSSD_ID_PATTERN = re.compile(r"[0-9a-f]{40}", re.IGNORECASE)
_AXIS_TO_RENDER_VIEW = {
    "+X": "right",
    "-X": "left",
    "+Y": "front",
    "-Y": "back",
}
_OPPOSITE_HORIZONTAL_AXIS = {
    "+X": "-X",
    "-X": "+X",
    "+Y": "-Y",
    "-Y": "+Y",
}
_COMPONENT_EQUIVALENTS = {
    "television": frozenset(
        {"television", "tv", "display", "screen", "monitor", "flat_screen"}
    ),
}


def _component_matches_forbidden(component: object, forbidden: list[str]) -> bool:
    normalized = normalize_semantic_name(component)
    if not normalized:
        return False
    for raw_forbidden in forbidden:
        normalized_forbidden = normalize_semantic_name(raw_forbidden)
        equivalents = _COMPONENT_EQUIVALENTS.get(
            normalized_forbidden, frozenset({normalized_forbidden})
        )
        if normalized in equivalents:
            return True
    return False


def _forbidden_candidate_ids(
    response: dict[str, object],
    evidence_records: list["_CandidateRenderEvidence"],
    forbidden_components: list[str],
) -> set[str]:
    """Return candidates that are forbidden or lack an explicit safe assessment."""
    if not forbidden_components:
        return set()
    records_by_index = {record.original_index: record for record in evidence_records}
    records_by_id = {record.candidate.hssd_id: record for record in evidence_records}
    rejected: set[str] = set()
    assessed: set[str] = set()
    assessments = response.get("candidate_assessments")
    if not isinstance(assessments, list):
        # A component-exclusion request may only use candidates that the VLM
        # explicitly inspected.  Missing assessments must not bypass the guard.
        return set(records_by_id)
    for assessment in assessments:
        if not isinstance(assessment, dict):
            continue
        record = records_by_id.get(str(assessment.get("hssd_id") or ""))
        if record is None:
            record = records_by_index.get(
                _coerce_selected_index(assessment.get("index")) or -1
            )
        if record is None:
            continue
        components = assessment.get("forbidden_components")
        component_values = components if isinstance(components, list) else []
        contains_forbidden = assessment.get("contains_forbidden_components")
        component_match = any(
            _component_matches_forbidden(value, forbidden_components)
            for value in component_values
        )
        if contains_forbidden is True or component_match:
            assessed.add(record.candidate.hssd_id)
            rejected.add(record.candidate.hssd_id)
        elif contains_forbidden is False:
            assessed.add(record.candidate.hssd_id)
    rejected.update(set(records_by_id) - assessed)
    return rejected


def _normalized_tokens(*values: str | None) -> set[str]:
    return set(
        re.findall(r"[a-z0-9]+", " ".join(str(value or "").lower() for value in values))
    )


def is_floor_covering_request(*values: str | None) -> bool:
    """Recognize semantic floor coverings without matching unrelated substrings."""
    return bool(_normalized_tokens(*values) & _FLOOR_COVERING_TOKENS)


def infer_floor_covering_footprint_shape(*values: str | None) -> str:
    """Infer the requested visual footprint without changing collision semantics."""
    tokens = _normalized_tokens(*values)
    if "square" in tokens:
        return "square"
    if tokens & _ROUND_FOOTPRINT_TOKENS:
        return "circular"
    return "rectangular"


def _planar_aspect_ratio(candidate: "HssdRetrievalResult") -> float:
    """Return the larger-to-smaller ratio of a SceneSmith-frame footprint."""
    try:
        width, depth = (abs(float(axis)) for axis in candidate.size[:2])
    except (TypeError, ValueError):
        return float("inf")
    if width <= 0.0 or depth <= 0.0:
        return float("inf")
    return max(width, depth) / min(width, depth)


def hssd_rendered_choice_options(cfg: object) -> tuple[bool, int, Path]:
    """Read shared config and environment options for rendered HSSD choice."""
    hssd_cfg = cfg.asset_manager.get("hssd", {}) or {}
    choice_cfg = hssd_cfg.get("rendered_asset_choice", {}) or {}

    enabled = bool(choice_cfg.get("enabled", False))
    env_enabled = os.environ.get("HSSD_RENDERED_ASSET_CHOICE")
    if env_enabled is not None:
        enabled = env_enabled.strip().lower() in {"1", "true", "yes", "on"}

    raw_top_n = os.environ.get("HSSD_RENDERED_ASSET_CHOICE_TOP_N")
    if raw_top_n is None:
        raw_top_n = choice_cfg.get("top_n", 4)
    try:
        top_n = max(1, int(raw_top_n))
    except (TypeError, ValueError):
        console_logger.warning(
            "Invalid HSSD rendered_asset_choice top_n=%r; using 4", raw_top_n
        )
        top_n = 4

    rendered_assets_dir = Path(
        os.environ.get("HSSD_RENDERED_ASSETS_DIR")
        or choice_cfg.get("rendered_assets_dir", "data/hssd_rendered_assets")
    )
    return enabled, top_n, rendered_assets_dir


@dataclass(frozen=True)
class RenderedAssetChoice:
    """Result of asking the VLM to choose among rendered HSSD candidates."""

    candidates: list["HssdRetrievalResult"]
    selected_hssd_id: str | None = None
    selected_index: int | None = None
    semantic_name: str | None = None
    reason: str | None = None
    used_image_count: int = 0


@dataclass(frozen=True)
class _CandidateRenderEvidence:
    original_index: int
    candidate: "HssdRetrievalResult"
    views: tuple[tuple[str, Path], ...]


def _render_material_quality(
    evidence: _CandidateRenderEvidence,
) -> dict[str, object]:
    """Return a conservative material-evidence score for an HSSD render."""
    if not evidence.views:
        return {"score": 0.0, "issues": ["missing_render_evidence"]}

    image_path = next(
        (path for label, path in evidence.views if label == "iso"), evidence.views[0][1]
    )
    try:
        with Image.open(image_path) as source:
            image = source.convert("RGB")
            pixels = list(image.getdata())
    except Exception as exc:
        return {
            "score": 0.0,
            "issues": ["unreadable_render_evidence"],
            "error": str(exc),
        }
    if not pixels:
        return {"score": 0.0, "issues": ["empty_render_evidence"]}

    width, height = image.size
    corners = (
        pixels[0],
        pixels[max(0, width - 1)],
        pixels[max(0, (height - 1) * width)],
        pixels[-1],
    )
    background = tuple(
        sum(pixel[channel] for pixel in corners) / len(corners) for channel in range(3)
    )
    foreground = [
        pixel
        for pixel in pixels
        if sum(abs(pixel[channel] - background[channel]) for channel in range(3)) > 30.0
    ]
    coverage = len(foreground) / len(pixels)
    if not foreground:
        return {
            "score": 0.0,
            "issues": ["blank_render"],
            "foreground_coverage": round(coverage, 4),
        }

    luminance = sorted(sum(pixel) / 3.0 for pixel in foreground)
    chroma = sum(max(pixel) - min(pixel) for pixel in foreground) / len(foreground)
    p10 = luminance[len(luminance) // 10]
    p90 = luminance[min(len(luminance) - 1, len(luminance) * 9 // 10)]
    mean_luminance = sum(luminance) / len(luminance)
    issues: list[str] = []
    if coverage < 0.015:
        issues.append("tiny_visible_subject")
    if mean_luminance >= 202.0 and chroma <= 2.5 and p90 - p10 <= 32.0:
        issues.append("low_material_detail")
    score = 1.0
    if "tiny_visible_subject" in issues:
        score -= 0.65
    if "low_material_detail" in issues:
        score -= 0.55
    return {
        "score": round(max(0.0, score), 3),
        "issues": issues,
        "foreground_coverage": round(coverage, 4),
        "mean_luminance": round(mean_luminance, 2),
        "mean_chroma": round(chroma, 2),
        "luminance_range": [round(p10, 2), round(p90, 2)],
    }


def _encode_candidate_views_as_one_image(
    views: tuple[tuple[str, Path], ...],
    identity_label: str | None = None,
) -> str:
    """Stitch one candidate's views into one labelled image for the VLM.

    Sending one multimodal image per asset keeps the candidate boundary
    unambiguous.  The original renders are kept at their native resolution
    (up to a conservative 2048px-wide canvas) and each panel is labelled with
    its verified/fallback view name directly in the pixels.
    """
    if not views:
        raise ValueError("candidate has no rendered views")

    panels: list[tuple[str, Image.Image]] = []
    for label, image_path in views:
        with Image.open(image_path) as source:
            image = source.convert("RGB").copy()
        panels.append((label, image))

    panel_height = max(image.height for _, image in panels)
    panel_header_height = max(28, min(56, panel_height // 8))
    identity_header_height = panel_header_height if identity_label else 0
    gap = 4
    panel_widths = [
        max(1, round(image.width * panel_height / max(1, image.height)))
        for _, image in panels
    ]
    total_width = sum(panel_widths) + gap * (len(panels) - 1)
    max_width = 2048
    if total_width > max_width:
        scale = max_width / total_width
        panel_height = max(1, round(panel_height * scale))
        panel_header_height = max(24, round(panel_header_height * scale))
        identity_header_height = panel_header_height if identity_label else 0
        panel_widths = [max(1, round(width * scale)) for width in panel_widths]
        total_width = sum(panel_widths) + gap * (len(panels) - 1)

    header_height = identity_header_height + panel_header_height
    canvas = Image.new("RGB", (total_width, panel_height + header_height), "white")
    draw = ImageDraw.Draw(canvas)
    if identity_label:
        draw.rectangle(
            (0, 0, total_width - 1, identity_header_height - 1),
            fill=(210, 225, 240),
            outline=(70, 90, 110),
        )
        draw.text(
            (6, max(2, identity_header_height // 5)), identity_label, fill="black"
        )
    x_offset = 0
    for panel_index, ((label, image), width) in enumerate(zip(panels, panel_widths)):
        resized = image.resize((width, panel_height), Image.Resampling.LANCZOS)
        canvas.paste(resized, (x_offset, header_height))
        draw.rectangle(
            (
                x_offset,
                identity_header_height,
                x_offset + width - 1,
                header_height - 1,
            ),
            fill=(235, 235, 235),
            outline=(90, 90, 90),
        )
        draw.text(
            (x_offset + 6, identity_header_height + max(2, panel_header_height // 5)),
            label,
            fill="black",
        )
        x_offset += width
        if panel_index < len(panels) - 1:
            draw.rectangle(
                (x_offset, 0, x_offset + gap - 1, panel_height + header_height),
                fill=(70, 70, 70),
            )
            x_offset += gap

    output = io.BytesIO()
    canvas.save(output, format="PNG", optimize=True)
    return base64.b64encode(output.getvalue()).decode("ascii")


def _write_choice_audit(
    *,
    object_description: str,
    scene_context: str | None,
    object_short_name: str | None,
    requested_dimensions: list[float] | tuple[float, ...] | None,
    requested_shape: str | None,
    candidates: list["HssdRetrievalResult"],
    evidence_records: list[_CandidateRenderEvidence],
    top_n: int,
    model: str,
    reasoning_effort: str,
    verbosity: str,
    vision_detail: str,
    status: str,
    audit_path: Path | None = None,
    **details: object,
) -> None:
    """Append a complete rendered-choice decision to the optional JSONL audit."""
    audit_path_raw = os.environ.get("HSSD_RENDERED_ASSET_CHOICE_AUDIT_PATH", "").strip()
    resolved_audit_path = audit_path or (
        Path(audit_path_raw) if audit_path_raw else None
    )
    if resolved_audit_path is None:
        return

    candidate_records = []
    evidence_by_id = {record.candidate.hssd_id: record for record in evidence_records}
    for index, candidate in enumerate(candidates, start=1):
        evidence = evidence_by_id.get(candidate.hssd_id)
        candidate_records.append(
            {
                "original_index": index,
                "hssd_id": candidate.hssd_id,
                "object_name": candidate.object_name,
                "category": candidate.category,
                "size": [float(axis) for axis in candidate.size],
                "similarity_score": float(candidate.similarity_score),
                "evidence_views": (
                    [
                        {"label": label, "path": str(path)}
                        for label, path in evidence.views
                    ]
                    if evidence is not None
                    else []
                ),
            }
        )

    event = {
        "schema_version": "hssd_rendered_choice_audit.v2",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "pid": os.getpid(),
        "status": status,
        "object_description": object_description,
        "object_short_name": object_short_name,
        "scene_context": scene_context,
        "requested_dimensions": (
            [float(axis) for axis in requested_dimensions]
            if requested_dimensions is not None
            else None
        ),
        "requested_shape": requested_shape,
        "top_n": top_n,
        "model": model,
        "reasoning_effort": reasoning_effort,
        "verbosity": verbosity,
        "vision_detail": vision_detail,
        "image_policy": "one_composite_per_candidate",
        "candidates": candidate_records,
    }
    event.update(details)

    try:
        resolved_audit_path.parent.mkdir(parents=True, exist_ok=True)
        with resolved_audit_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")
            stream.flush()
    except Exception as exc:  # pragma: no cover - audit must never break retrieval
        console_logger.warning(
            "Could not write HSSD choice audit %s: %s", resolved_audit_path, exc
        )


def _normalize_annotation_front_axis(value: object) -> str | None:
    """Convert a SceneBenchmark HSSD front hint to a SceneSmith axis."""
    text = str(value or "").strip().upper().replace(" ", "")
    if text in _AXIS_TO_RENDER_VIEW:
        return text
    if len(text) == 1 and text in {"X", "Y"}:
        return f"+{text}"

    if not isinstance(value, (list, tuple)) or len(value) != 3:
        return None
    try:
        x_hsm, y_hsm, z_hsm = (float(axis) for axis in value)
    except (TypeError, ValueError):
        return None

    # HSM is X-right/Y-up/Z-forward; SceneSmith is X-right/Y-forward/Z-up.
    x_scene = x_hsm
    y_scene = -z_hsm
    z_scene = y_hsm
    axis_name, axis_value = max(
        {"X": x_scene, "Y": y_scene}.items(), key=lambda item: abs(item[1])
    )
    if abs(axis_value) < 1e-6 or abs(z_scene) > abs(axis_value):
        return None
    return f"{'+' if axis_value >= 0 else '-'}{axis_name}"


def _annotation_axis_from_record(record: dict[str, object]) -> str | None:
    canonical = record.get("canonical_front") or {}
    if not isinstance(canonical, dict):
        return None
    axis_value = canonical.get("asset_local_front_axis")
    if axis_value is None:
        axis_value = canonical.get("canonical_orientation_axis")
    if axis_value is None:
        hints = record.get("scenebenchmark_functional_hints") or {}
        if isinstance(hints, dict):
            axis_value = hints.get("asset_local_front_axis")
    return _normalize_annotation_front_axis(axis_value)


def _annotation_render_views(
    *, hssd_id: str, asset_dir: Path
) -> tuple[tuple[str, Path], ...]:
    """Return useful named views selected from SceneBenchmark front metadata."""
    if _HSSD_ID_PATTERN.fullmatch(hssd_id) is None:
        return ()

    try:
        from scenesmith.scenebenchmark_critic.asset_library_annotations import (
            get_hssd_asset_annotations,
        )

        record = get_hssd_asset_annotations(hssd_id)
    except Exception as exc:
        console_logger.warning(
            "Could not load SceneBenchmark HSSD front evidence for %s: %s",
            hssd_id,
            exc,
        )
        return ()
    if not record:
        return ()

    canonical = record.get("canonical_front") or {}
    if not isinstance(canonical, dict):
        return ()
    axis = _annotation_axis_from_record(record)
    if axis is None:
        return ()

    semantic_front = canonical.get("canonical_orientation_is_semantic_front") is True
    strict_front = bool(
        canonical.get("is_strict_front") or canonical.get("is_strict_positive_front")
    )
    if not semantic_front and not strict_front:
        return ()

    axes = [axis]
    if strict_front and not semantic_front:
        # A fallback axis is only a coordinate convention. Include its opposite
        # so the VLM can find doors or controls without trusting the axis sign.
        axes.append(_OPPOSITE_HORIZONTAL_AXIS[axis])

    views: list[tuple[str, Path]] = []
    for view_axis in axes:
        view_name = _AXIS_TO_RENDER_VIEW[view_axis]
        image_path = asset_dir / f"{view_name}.png"
        if not image_path.exists():
            continue
        evidence_kind = (
            "SceneBenchmark semantic-front"
            if semantic_front
            else "SceneBenchmark fallback-axis"
        )
        views.append((f"{evidence_kind} {view_axis} ({view_name})", image_path))
    return tuple(views)


def choose_hssd_candidate_from_iso_renders(
    *,
    candidates: list["HssdRetrievalResult"],
    object_description: str,
    scene_context: str | None,
    vlm_service: "VLMService",
    model: str,
    reasoning_effort: str,
    verbosity: str,
    vision_detail: str,
    rendered_assets_dir: Path,
    top_n: int,
    object_short_name: str | None = None,
    requested_dimensions: list[float] | tuple[float, ...] | None = None,
    requested_shape: str | None = None,
    semantic_name_candidates: list[str] | None = None,
    forbidden_components: list[str] | None = None,
    audit_path: Path | None = None,
    retrieval_backend: str | None = None,
) -> RenderedAssetChoice:
    """Optionally reorder candidates using pre-rendered HSSD visual evidence."""
    fallback_semantic_name = normalize_semantic_name(object_short_name)
    forbidden_components = [
        normalized
        for value in forbidden_components or []
        if (normalized := normalize_semantic_name(value))
    ]
    allowed_semantic_names = []
    for value in semantic_name_candidates or []:
        normalized = normalize_semantic_name(value)
        if normalized and normalized not in allowed_semantic_names:
            allowed_semantic_names.append(normalized)
    if fallback_semantic_name and fallback_semantic_name not in allowed_semantic_names:
        allowed_semantic_names.append(fallback_semantic_name)
    requires_component_assessment = bool(forbidden_components)
    if (top_n <= 1 or len(candidates) <= 1) and not requires_component_assessment:
        _write_choice_audit(
            audit_path=audit_path,
            object_description=object_description,
            scene_context=scene_context,
            object_short_name=object_short_name,
            requested_dimensions=requested_dimensions,
            requested_shape=requested_shape,
            candidates=candidates,
            evidence_records=[],
            top_n=top_n,
            model=model,
            reasoning_effort=reasoning_effort,
            verbosity=verbosity,
            vision_detail=vision_detail,
            status="skipped",
            retrieval_backend=retrieval_backend,
            forbidden_components=forbidden_components,
            reason="top_n_or_candidate_count_too_small",
        )
        return RenderedAssetChoice(
            candidates=candidates, semantic_name=fallback_semantic_name or None
        )

    evidence_records: list[_CandidateRenderEvidence] = []
    candidates_to_assess = (
        candidates if requires_component_assessment else candidates[:top_n]
    )
    for original_index, candidate in enumerate(candidates_to_assess, start=1):
        asset_dir = rendered_assets_dir / candidate.hssd_id
        iso_path = asset_dir / "iso.png"
        if not iso_path.exists():
            continue
        views = [("iso", iso_path)]
        views.extend(
            _annotation_render_views(hssd_id=candidate.hssd_id, asset_dir=asset_dir)
        )
        evidence_records.append(
            _CandidateRenderEvidence(
                original_index=original_index,
                candidate=candidate,
                views=tuple(views),
            )
        )

    render_quality_by_id = {
        record.candidate.hssd_id: _render_material_quality(record)
        for record in evidence_records
    }
    quality_eligible_ids = {
        hssd_id
        for hssd_id, quality in render_quality_by_id.items()
        if float(quality["score"]) >= 0.5
    }

    used_view_count = sum(len(record.views) for record in evidence_records)
    # Keep one multimodal image per asset; multiple views are stitched below.
    used_image_count = len(evidence_records)
    console_logger.info(
        "Prepared rendered HSSD evidence for '%s': %s",
        object_description,
        {
            record.candidate.hssd_id: [label for label, _ in record.views]
            for record in evidence_records
        },
    )
    if requires_component_assessment and len(evidence_records) != len(candidates):
        missing_evidence_ids = sorted(
            candidate.hssd_id
            for candidate in candidates
            if candidate.hssd_id
            not in {record.candidate.hssd_id for record in evidence_records}
        )
        _write_choice_audit(
            audit_path=audit_path,
            object_description=object_description,
            scene_context=scene_context,
            object_short_name=object_short_name,
            requested_dimensions=requested_dimensions,
            requested_shape=requested_shape,
            candidates=candidates,
            evidence_records=evidence_records,
            top_n=top_n,
            model=model,
            reasoning_effort=reasoning_effort,
            verbosity=verbosity,
            vision_detail=vision_detail,
            status="insufficient_forbidden_component_evidence",
            retrieval_backend=retrieval_backend,
            forbidden_components=forbidden_components,
            missing_evidence_candidate_ids=missing_evidence_ids,
            used_image_count=used_image_count,
            used_view_count=used_view_count,
        )
        console_logger.warning(
            "Rejecting rendered HSSD candidates for '%s': missing visual evidence "
            "for forbidden-component assessment",
            object_description,
        )
        return RenderedAssetChoice(
            candidates=[],
            semantic_name=fallback_semantic_name or None,
            reason="missing visual evidence for forbidden-component assessment",
            used_image_count=used_image_count,
        )

    if len(evidence_records) <= 1 and not requires_component_assessment:
        _write_choice_audit(
            audit_path=audit_path,
            object_description=object_description,
            scene_context=scene_context,
            object_short_name=object_short_name,
            requested_dimensions=requested_dimensions,
            requested_shape=requested_shape,
            candidates=candidates,
            evidence_records=evidence_records,
            top_n=top_n,
            model=model,
            reasoning_effort=reasoning_effort,
            verbosity=verbosity,
            vision_detail=vision_detail,
            status="insufficient_evidence",
            retrieval_backend=retrieval_backend,
            forbidden_components=forbidden_components,
            used_image_count=used_image_count,
            used_view_count=used_view_count,
        )
        console_logger.debug(
            "Skipping rendered HSSD choice for '%s': only %d/%d iso renders found",
            object_description,
            len(evidence_records),
            len(candidates_to_assess),
        )
        return RenderedAssetChoice(
            candidates=candidates,
            semantic_name=fallback_semantic_name or None,
            used_image_count=used_image_count,
        )

    candidate_lines = [
        "- index {index}: hssd_id={hssd_id}, name={name}, category={category}, "
        "size_m={size}, embedding_score={score:.4f}, render_quality={quality}".format(
            index=record.original_index,
            hssd_id=record.candidate.hssd_id,
            name=record.candidate.object_name or "(unnamed)",
            category=record.candidate.category or "(unknown)",
            size=tuple(round(float(axis), 3) for axis in record.candidate.size),
            score=float(record.candidate.similarity_score),
            quality=json.dumps(
                render_quality_by_id[record.candidate.hssd_id],
                sort_keys=True,
            ),
        )
        for record in evidence_records
    ]
    floor_covering = is_floor_covering_request(object_description, object_short_name)
    request_details = []
    if object_short_name:
        request_details.append(f"Requested short name: {object_short_name}")
    if requested_dimensions:
        request_details.append(
            "Requested size [width, depth, height] m: "
            + str(tuple(round(float(axis), 3) for axis in requested_dimensions))
        )
    if requested_shape:
        request_details.append(f"Requested footprint shape: {requested_shape}")
    if allowed_semantic_names:
        request_details.append(
            "Allowed semantic names (choose exactly one verbatim): "
            + json.dumps(allowed_semantic_names)
        )
    if forbidden_components:
        request_details.append(
            "Forbidden bundled components (the selected mesh must not contain any): "
            + json.dumps(forbidden_components)
        )
    covering_guidance = (
        " For rugs, carpets, mats, and runners, near-zero visual thickness is "
        "expected and must not be penalized; prioritize semantic appearance, "
        "footprint shape, and planar width-to-depth aspect ratio."
        if floor_covering
        else ""
    )
    prompt = (
        "You are choosing the best HSSD asset for a 3D indoor scene.\n"
        f"Requested object: {object_description}\n"
        + ("\n".join(request_details) + "\n" if request_details else "")
        + (f"Original scene prompt: {scene_context}\n" if scene_context else "")
        + "\nInspect the attached render evidence. Each candidate has exactly one "
        "composite image; its panels are captioned with view labels and the "
        "candidate index/HSSD ID text immediately precedes that image. A "
        "SceneBenchmark semantic-front view is "
        "a verified front hint. A fallback-axis view is only coordinate evidence; "
        "use it together with its opposite view to find doors, drawers, controls, "
        "or other usable surfaces. Named front/back/left/right values describe "
        "camera directions and are not themselves semantic labels.\n"
        + "\n".join(candidate_lines)
        + "\n\nChoose exactly one candidate that best matches the requested "
        "object type and likely has usable proportions. Penalize wrong object "
        "types, bunk/loft beds unless explicitly requested, partial objects, "
        "and assets whose front/usable side is visually unclear. Treat the "
        "reported size as [width, depth, height] and reject visibly incompatible "
        "relative proportions rather than assuming later scaling can fix them. "
        "In particular, an upright dining, desk, or task chair must not be "
        "replaced by a low lounge chair or armchair. Prefer candidates with "
        "material evidence; `low_material_detail` means the render is nearly "
        "monochrome and untextured, unless the request explicitly calls for that "
        "appearance." + covering_guidance + "\n"
        "For every candidate, report whether any forbidden bundled component is "
        "visibly present. A TV stand that already includes a television is invalid "
        "when television is forbidden. Do not select an invalid candidate.\n"
        'Return JSON only: {"candidate_assessments": [{"index": <index>, '
        '"hssd_id": "<hssd_id>", "contains_forbidden_components": <boolean>, '
        '"forbidden_components": ["<component>"]}], '
        '"selected_index": <index number>, '
        '"selected_hssd_id": "<hssd_id>", '
        '"semantic_name": "<one allowed semantic name>", '
        '"reason": "<short reason>"}'
    )

    user_content = [{"type": "text", "text": prompt}]
    for record in evidence_records:
        view_labels = "; ".join(label for label, _ in record.views)
        user_content.append(
            {
                "type": "text",
                "text": (
                    f"CANDIDATE_INDEX={record.original_index} "
                    f"HSSD_ID={record.candidate.hssd_id} PANELS={view_labels}"
                ),
            }
        )
        encoded = _encode_candidate_views_as_one_image(
            record.views,
            identity_label=(
                f"INDEX={record.original_index} " f"HSSD_ID={record.candidate.hssd_id}"
            ),
        )
        user_content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{encoded}"},
            }
        )

    try:
        start_time = time.time()
        response_text = vlm_service.create_completion(
            model=model,
            messages=[{"role": "user", "content": user_content}],
            reasoning_effort=reasoning_effort,
            verbosity=verbosity,
            response_format={"type": "json_object"},
            vision_detail=vision_detail,
        )
        response_json = parse_llm_json_object(response_text)
        console_logger.info(
            "Rendered HSSD choice completed in %.1fs for '%s': %s",
            time.time() - start_time,
            object_description,
            response_json,
        )
    except Exception as exc:
        _write_choice_audit(
            audit_path=audit_path,
            object_description=object_description,
            scene_context=scene_context,
            object_short_name=object_short_name,
            requested_dimensions=requested_dimensions,
            requested_shape=requested_shape,
            candidates=candidates,
            evidence_records=evidence_records,
            top_n=top_n,
            model=model,
            reasoning_effort=reasoning_effort,
            verbosity=verbosity,
            vision_detail=vision_detail,
            status="vlm_error",
            retrieval_backend=retrieval_backend,
            forbidden_components=forbidden_components,
            used_image_count=used_image_count,
            used_view_count=used_view_count,
            error=str(exc),
        )
        if requires_component_assessment:
            console_logger.warning(
                "Rejecting rendered HSSD candidates for '%s': forbidden-component "
                "assessment failed: %s",
                object_description,
                exc,
            )
            return RenderedAssetChoice(
                candidates=[],
                semantic_name=fallback_semantic_name or None,
                reason="forbidden-component assessment failed",
                used_image_count=used_image_count,
            )
        console_logger.warning(
            "Rendered HSSD choice failed for '%s'; keeping retrieval order: %s",
            object_description,
            exc,
        )
        return RenderedAssetChoice(
            candidates=candidates,
            semantic_name=fallback_semantic_name or None,
            used_image_count=used_image_count,
        )

    selected_index = _coerce_selected_index(response_json.get("selected_index"))
    selected_hssd_id = response_json.get("selected_hssd_id")
    raw_semantic_name = normalize_semantic_name(response_json.get("semantic_name"))
    semantic_name = (
        raw_semantic_name
        if raw_semantic_name and raw_semantic_name in allowed_semantic_names
        else fallback_semantic_name
    )
    reason = response_json.get("reason")
    forbidden_ids = _forbidden_candidate_ids(
        response_json, evidence_records, forbidden_components
    )
    eligible_records = [
        record
        for record in evidence_records
        if record.candidate.hssd_id not in forbidden_ids
    ]
    if forbidden_components and not eligible_records:
        _write_choice_audit(
            audit_path=audit_path,
            object_description=object_description,
            scene_context=scene_context,
            object_short_name=object_short_name,
            requested_dimensions=requested_dimensions,
            requested_shape=requested_shape,
            candidates=candidates,
            evidence_records=evidence_records,
            top_n=top_n,
            model=model,
            reasoning_effort=reasoning_effort,
            verbosity=verbosity,
            vision_detail=vision_detail,
            status="rejected_all_forbidden_components",
            retrieval_backend=retrieval_backend,
            forbidden_components=forbidden_components,
            forbidden_candidate_ids=sorted(forbidden_ids),
            raw_response=response_text,
            parsed_response=response_json,
        )
        console_logger.warning(
            "All rendered HSSD candidates for '%s' contain forbidden components: %s",
            object_description,
            forbidden_components,
        )
        return RenderedAssetChoice(
            candidates=[],
            semantic_name=fallback_semantic_name or None,
            reason="all rendered candidates contain forbidden components",
            used_image_count=used_image_count,
        )

    selected_candidate: "HssdRetrievalResult | None" = None
    if isinstance(selected_hssd_id, str) and selected_hssd_id:
        selected_candidate = next(
            (
                record.candidate
                for record in evidence_records
                if record.candidate.hssd_id == selected_hssd_id
            ),
            None,
        )
    if selected_candidate is None and selected_index is not None:
        selected_candidate = next(
            (
                record.candidate
                for record in evidence_records
                if record.original_index == selected_index
            ),
            None,
        )

    if selected_candidate is None and forbidden_components and eligible_records:
        selected_candidate = eligible_records[0].candidate
        reason = (
            "forbidden-component guard selected the first explicitly safe candidate"
        )

    if selected_candidate is None:
        _write_choice_audit(
            audit_path=audit_path,
            object_description=object_description,
            scene_context=scene_context,
            object_short_name=object_short_name,
            requested_dimensions=requested_dimensions,
            requested_shape=requested_shape,
            candidates=candidates,
            evidence_records=evidence_records,
            top_n=top_n,
            model=model,
            reasoning_effort=reasoning_effort,
            verbosity=verbosity,
            vision_detail=vision_detail,
            status="invalid_selection",
            retrieval_backend=retrieval_backend,
            forbidden_components=forbidden_components,
            used_image_count=used_image_count,
            used_view_count=used_view_count,
            raw_response=response_text,
            parsed_response=response_json,
            selected_index=selected_index,
            selected_hssd_id=selected_hssd_id,
            semantic_name=semantic_name or None,
            allowed_semantic_names=allowed_semantic_names,
        )
        console_logger.warning(
            "Rendered HSSD choice for '%s' returned invalid selection %s/%s; "
            "keeping retrieval order",
            object_description,
            selected_index,
            selected_hssd_id,
        )
        return RenderedAssetChoice(
            candidates=candidates,
            semantic_name=fallback_semantic_name or None,
            used_image_count=used_image_count,
        )

    forbidden_override_used = False
    if selected_candidate.hssd_id in forbidden_ids:
        rejected_candidate = selected_candidate
        selected_candidate = eligible_records[0].candidate
        forbidden_override_used = True
        reason = (
            "forbidden-component guard overrode rendered choice "
            f"{rejected_candidate.hssd_id}; selected {selected_candidate.hssd_id}"
        )
        console_logger.warning(
            "Rendered HSSD choice for '%s' selected candidate %s with forbidden "
            "components; using %s",
            object_description,
            rejected_candidate.hssd_id,
            selected_candidate.hssd_id,
        )

    quality_fallback_used = False
    eligible_quality_ids = quality_eligible_ids - forbidden_ids
    if eligible_quality_ids and selected_candidate.hssd_id not in eligible_quality_ids:
        rejected_candidate = selected_candidate
        selected_candidate = next(
            record.candidate
            for record in evidence_records
            if record.candidate.hssd_id in eligible_quality_ids
        )
        quality_fallback_used = True
        reason = (
            "material-evidence guard overrode rendered choice "
            f"{rejected_candidate.hssd_id}; selected {selected_candidate.hssd_id}"
        )
        console_logger.warning(
            "Rendered HSSD choice for '%s' selected low-material-detail asset %s; "
            "using %s instead",
            object_description,
            rejected_candidate.hssd_id,
            selected_candidate.hssd_id,
        )

    if str(requested_shape or "").strip().lower() == "square":
        square_candidates = [
            record.candidate
            for record in evidence_records
            if record.candidate.hssd_id not in forbidden_ids
            if _planar_aspect_ratio(record.candidate)
            <= _SQUARE_FOOTPRINT_MAX_ASPECT_RATIO
        ]
        if square_candidates and selected_candidate not in square_candidates:
            rejected_candidate = selected_candidate
            selected_candidate = square_candidates[0]
            reason = (
                "square footprint guard overrode rendered choice "
                f"{rejected_candidate.hssd_id} "
                f"(aspect={_planar_aspect_ratio(rejected_candidate):.3f})"
            )
            console_logger.warning(
                "Rendered HSSD choice for '%s' selected non-square candidate %s "
                "(aspect=%.3f); using %s (aspect=%.3f)",
                object_description,
                rejected_candidate.hssd_id,
                _planar_aspect_ratio(rejected_candidate),
                selected_candidate.hssd_id,
                _planar_aspect_ratio(selected_candidate),
            )

    original_index = candidates.index(selected_candidate) + 1
    _write_choice_audit(
        audit_path=audit_path,
        object_description=object_description,
        scene_context=scene_context,
        object_short_name=object_short_name,
        requested_dimensions=requested_dimensions,
        requested_shape=requested_shape,
        candidates=candidates,
        evidence_records=evidence_records,
        top_n=top_n,
        model=model,
        reasoning_effort=reasoning_effort,
        verbosity=verbosity,
        vision_detail=vision_detail,
        status="selected",
        retrieval_backend=retrieval_backend,
        used_image_count=used_image_count,
        used_view_count=used_view_count,
        raw_response=response_text,
        parsed_response=response_json,
        selected_index=original_index,
        selected_hssd_id=selected_candidate.hssd_id,
        semantic_name=semantic_name or None,
        allowed_semantic_names=allowed_semantic_names,
        returned_semantic_name=raw_semantic_name or None,
        reason=str(reason) if reason is not None else None,
        render_quality_by_hssd_id=render_quality_by_id,
        quality_fallback_used=quality_fallback_used,
        forbidden_components=forbidden_components,
        forbidden_candidate_ids=sorted(forbidden_ids),
        forbidden_override_used=forbidden_override_used,
    )

    reordered = [selected_candidate] + [
        candidate
        for candidate in candidates
        if candidate.hssd_id != selected_candidate.hssd_id
        and (not forbidden_components or candidate.hssd_id not in forbidden_ids)
    ]
    return RenderedAssetChoice(
        candidates=reordered,
        selected_hssd_id=selected_candidate.hssd_id,
        selected_index=original_index,
        semantic_name=semantic_name or None,
        reason=str(reason) if reason is not None else None,
        used_image_count=used_image_count,
    )


def _coerce_selected_index(value: object) -> int | None:
    """Parse a 1-based candidate index from model output."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
