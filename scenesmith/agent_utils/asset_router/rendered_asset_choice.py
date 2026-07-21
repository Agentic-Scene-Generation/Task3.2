"""VLM-assisted choice among rendered HSSD retrieval candidates."""

import logging
import time

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from scenesmith.utils.llm_json import parse_llm_json_object
from scenesmith.utils.openai import encode_image_to_base64

if TYPE_CHECKING:
    from scenesmith.agent_utils.hssd_retrieval_server.dataclasses import (
        HssdRetrievalResult,
    )
    from scenesmith.agent_utils.vlm_service import VLMService


console_logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RenderedAssetChoice:
    """Result of asking the VLM to choose among rendered HSSD candidates."""

    candidates: list["HssdRetrievalResult"]
    selected_hssd_id: str | None = None
    selected_index: int | None = None
    reason: str | None = None
    used_image_count: int = 0


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
) -> RenderedAssetChoice:
    """Optionally reorder candidates using their pre-rendered iso images."""
    if top_n <= 1 or len(candidates) <= 1:
        return RenderedAssetChoice(candidates=candidates)

    image_records: list[tuple[int, "HssdRetrievalResult", Path]] = []
    for original_index, candidate in enumerate(candidates[:top_n], start=1):
        image_path = rendered_assets_dir / candidate.hssd_id / "iso.png"
        if image_path.exists():
            image_records.append((original_index, candidate, image_path))

    if len(image_records) <= 1:
        console_logger.debug(
            "Skipping rendered HSSD choice for '%s': only %d/%d iso renders found",
            object_description,
            len(image_records),
            min(top_n, len(candidates)),
        )
        return RenderedAssetChoice(
            candidates=candidates, used_image_count=len(image_records)
        )

    candidate_lines = [
        "- index {index}: hssd_id={hssd_id}, name={name}, category={category}, "
        "size_m={size}, embedding_score={score:.4f}".format(
            index=original_index,
            hssd_id=candidate.hssd_id,
            name=candidate.object_name or "(unnamed)",
            category=candidate.category or "(unknown)",
            size=tuple(round(float(axis), 3) for axis in candidate.size),
            score=float(candidate.similarity_score),
        )
        for original_index, candidate, _ in image_records
    ]
    prompt = (
        "You are choosing the best HSSD asset for a 3D indoor scene.\n"
        f"Requested object: {object_description}\n"
        + (f"Original scene prompt: {scene_context}\n" if scene_context else "")
        + "\nInspect the attached iso render images. The images are attached in the "
        "same order as these candidate lines:\n"
        + "\n".join(candidate_lines)
        + "\n\nChoose exactly one candidate that best matches the requested "
        "object type and likely has usable proportions. Penalize wrong object "
        "types, bunk/loft beds unless explicitly requested, partial objects, "
        "and assets whose front/usable side is visually unclear.\n"
        'Return JSON only: {"selected_index": <index number>, '
        '"selected_hssd_id": "<hssd_id>", "reason": "<short reason>"}'
    )

    user_content = [{"type": "text", "text": prompt}]
    for _, _, image_path in image_records:
        encoded = encode_image_to_base64(image_path)
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
        console_logger.warning(
            "Rendered HSSD choice failed for '%s'; keeping retrieval order: %s",
            object_description,
            exc,
        )
        return RenderedAssetChoice(
            candidates=candidates, used_image_count=len(image_records)
        )

    selected_index = _coerce_selected_index(response_json.get("selected_index"))
    selected_hssd_id = response_json.get("selected_hssd_id")
    reason = response_json.get("reason")

    selected_candidate: "HssdRetrievalResult | None" = None
    if isinstance(selected_hssd_id, str) and selected_hssd_id:
        selected_candidate = next(
            (
                candidate
                for _, candidate, _ in image_records
                if candidate.hssd_id == selected_hssd_id
            ),
            None,
        )
    if selected_candidate is None and selected_index is not None:
        selected_candidate = next(
            (
                candidate
                for original_index, candidate, _ in image_records
                if original_index == selected_index
            ),
            None,
        )

    if selected_candidate is None:
        console_logger.warning(
            "Rendered HSSD choice for '%s' returned invalid selection %s/%s; "
            "keeping retrieval order",
            object_description,
            selected_index,
            selected_hssd_id,
        )
        return RenderedAssetChoice(
            candidates=candidates, used_image_count=len(image_records)
        )

    reordered = [selected_candidate] + [
        candidate
        for candidate in candidates
        if candidate.hssd_id != selected_candidate.hssd_id
    ]
    original_index = candidates.index(selected_candidate) + 1
    return RenderedAssetChoice(
        candidates=reordered,
        selected_hssd_id=selected_candidate.hssd_id,
        selected_index=original_index,
        reason=str(reason) if reason is not None else None,
        used_image_count=len(image_records),
    )


def _coerce_selected_index(value: object) -> int | None:
    """Parse a 1-based candidate index from model output."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
