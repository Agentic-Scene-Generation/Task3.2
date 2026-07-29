"""Generate soft per-furniture placement-order references for manipulands."""

from __future__ import annotations

import json
import logging
import re
import tempfile

from pathlib import Path
from typing import Any

from omegaconf import DictConfig

from scenesmith.agent_utils.room import SupportSurface, UniqueID
from scenesmith.agent_utils.stage_placement_order_config import (
    get_stage_placement_order_config,
)
from scenesmith.agent_utils.vlm_service import VLMService
from scenesmith.utils.llm_json import parse_llm_json_object

console_logger = logging.getLogger(__name__)

PLACEMENT_ORDER_CACHE_FILENAME = "manipuland_placement_orders.json"
GENERIC_SURFACE = "any suitable support surface"

_POSITIONAL_PATTERN = re.compile(
    r"\b(?:left|right|front|back|rear|corner|edge|center|centre|middle|near|"
    r"beside|behind|adjacent|above|below|upper|lower|coordinate|coordinates|"
    r"position|x-axis|y-axis|meter|meters|metre|metres)\b"
    r"|(?:\bnext\s+to\b)"
    r"|(?:\b[xyz]\s*[:=]\s*[-+]?\d+(?:\.\d+)?)"
    r"|(?:[-+]?\d+(?:\.\d+)?\s*(?:m|cm|mm|meter|meters|metre|metres|"
    r"centimeter|centimeters|centimetre|centimetres|foot|feet|ft|inch|inches|in)\b)",
    flags=re.IGNORECASE,
)

_LEADING_POSITION_PATTERN = re.compile(
    r"^(?:the\s+)?(?:left|right|front|back|rear|center|centre|middle|upper|lower)"
    r"(?:[-_\s]+(?:side|edge|corner))?[-_\s:]+",
    flags=re.IGNORECASE,
)
_TRAILING_POSITION_PATTERN = re.compile(
    r"\s+(?:on|at|near|beside|behind|adjacent\s+to|next\s+to|in)\s+(?:the\s+)?"
    r"(?:left|right|front|back|rear|center|centre|middle|upper|lower|edge|corner)"
    r"\b.*$",
    flags=re.IGNORECASE,
)


def build_manipuland_placement_order_reference(
    cfg: DictConfig | dict[str, Any],
    scene_prompt: str,
    scene_dir: Path,
    vlm_service: VLMService,
    model: str,
    furniture_id: UniqueID,
    furniture_description: str,
    suggested_items: str,
    prompt_constraints: str,
    style_notes: str,
    support_surfaces: dict[str, SupportSurface],
) -> str:
    """Build one furniture item's reference, failing open to an empty string."""
    try:
        stage_cfg = get_stage_placement_order_config(cfg, stage="manipuland")
        if not bool(stage_cfg.get("enabled", False)):
            return ""

        items = _load_or_generate_order(
            stage_cfg=stage_cfg,
            scene_prompt=scene_prompt,
            scene_dir=Path(scene_dir),
            vlm_service=vlm_service,
            model=model,
            furniture_id=str(furniture_id),
            furniture_description=furniture_description,
            suggested_items=suggested_items,
            prompt_constraints=prompt_constraints,
            style_notes=style_notes,
            support_surfaces=support_surfaces,
        )
        maximum = max(0, int(stage_cfg.get("max_items_per_surface", 8)))
        return _format_reference(_limit_per_surface(items, maximum))
    except Exception as exc:
        console_logger.warning(
            "Failed to build manipuland placement order for %s: %s",
            furniture_id,
            exc,
        )
        return ""


def _load_or_generate_order(
    *,
    stage_cfg: dict[str, Any],
    scene_prompt: str,
    scene_dir: Path,
    vlm_service: VLMService,
    model: str,
    furniture_id: str,
    furniture_description: str,
    suggested_items: str,
    prompt_constraints: str,
    style_notes: str,
    support_surfaces: dict[str, SupportSurface],
) -> list[dict[str, str]]:
    cache_enabled = bool(stage_cfg.get("cache", True))
    fallback_enabled = bool(stage_cfg.get("fallback_enabled", True))
    maximum = max(0, int(stage_cfg.get("max_items_per_surface", 8)))
    cache_path = scene_dir / PLACEMENT_ORDER_CACHE_FILENAME
    surface_ids = list(support_surfaces)
    cache = _read_cache(cache_path) if cache_enabled else {}

    cached = cache.get(furniture_id)
    if isinstance(cached, list):
        items = _normalize_items(cached, surface_ids)
        if not items and fallback_enabled:
            items = _fallback_order(suggested_items, surface_ids)
        items = _limit_per_surface(items, maximum)
        cache[furniture_id] = items
        _write_cache_best_effort(cache_path, cache)
        return items

    raw_data: Any = {}
    try:
        response = vlm_service.create_completion(
            model=model,
            messages=_build_messages(
                scene_prompt=scene_prompt,
                furniture_description=furniture_description,
                suggested_items=suggested_items,
                prompt_constraints=prompt_constraints,
                style_notes=style_notes,
                support_surfaces=support_surfaces,
            ),
            reasoning_effort="low",
            verbosity="low",
            response_format={"type": "json_object"},
            enable_thinking=bool(stage_cfg.get("enable_thinking", False)),
        )
        raw_data = parse_llm_json_object(response)
    except Exception as exc:
        console_logger.warning(
            "Manipuland placement-order generation failed for %s%s: %s",
            furniture_id,
            "; using fallback" if fallback_enabled else "",
            exc,
        )

    items = _normalize_items(raw_data, surface_ids)
    if not items and fallback_enabled:
        items = _fallback_order(suggested_items, surface_ids)
    items = _limit_per_surface(items, maximum)

    if cache_enabled:
        cache[furniture_id] = items
        _write_cache_best_effort(cache_path, cache)
    return items


def _build_messages(
    *,
    scene_prompt: str,
    furniture_description: str,
    suggested_items: str,
    prompt_constraints: str,
    style_notes: str,
    support_surfaces: dict[str, SupportSurface],
) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "Create a concise manipuland placement order for one furniture "
                "item. Return JSON only."
            ),
        },
        {
            "role": "user",
            "content": f"""
Room prompt:
{scene_prompt}

Current furniture:
{furniture_description}

Assigned items:
{suggested_items}

Prompt requirements:
{prompt_constraints}

Style guidance:
{style_notes}

Available support surfaces:
{_format_surfaces(support_surfaces)}

Return:
{{
  "manipuland": [
    {{"item": "...", "surface": "exact surface ID", "notes": "..."}}
  ]
}}

Rules:
- Include only small objects that can rest on this furniture.
- Put REQUIRED and functionally primary objects first.
- Use an exact available support surface ID.
- Notes may explain function or sequence only.
- Never provide coordinates, distances, directions, relative positions, edges, or corners.
- Do not include floor-standing, wall-mounted, or ceiling-mounted assets.
""".strip(),
        },
    ]


def _format_surfaces(support_surfaces: dict[str, SupportSurface]) -> str:
    if not support_surfaces:
        return "No support surfaces available."
    lines: list[str] = []
    for surface_id, surface in support_surfaces.items():
        try:
            dimensions = surface.bounding_box_max - surface.bounding_box_min
            size = f"{float(dimensions[0]):.2f}m x {float(dimensions[1]):.2f}m"
        except Exception:
            size = "size unknown"
        link_name = getattr(surface, "link_name", None)
        link = f", link={link_name}" if link_name else ""
        lines.append(f"- {surface_id}: {size} support area{link}")
    return "\n".join(lines)


def _normalize_items(data: Any, surface_ids: list[str]) -> list[dict[str, str]]:
    result: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for raw in _extract_items(data):
        if isinstance(raw, str):
            name, surface, notes = raw, "", ""
        elif isinstance(raw, dict):
            name = raw.get("item") or raw.get("asset") or raw.get("name") or ""
            surface = (
                raw.get("surface") or raw.get("surface_id") or raw.get("target") or ""
            )
            notes = raw.get("notes") or raw.get("rationale") or raw.get("reason") or ""
        else:
            continue

        safe_name = _sanitize_item_name(str(name))
        if not safe_name:
            continue
        normalized_surface = _normalize_surface(str(surface), surface_ids)
        safe_notes = _sanitize_notes(str(notes))
        key = (safe_name.lower(), normalized_surface.lower())
        if key in seen:
            continue
        seen.add(key)
        result.append(
            {
                "item": safe_name,
                "surface": normalized_surface,
                "notes": safe_notes,
            }
        )
    return result


def _extract_items(data: Any) -> list[Any]:
    if isinstance(data, list):
        return data
    if not isinstance(data, dict):
        return []
    for key in ("manipuland", "manipulands", "items", "placement_order"):
        value = data.get(key)
        if isinstance(value, list):
            return value
        if isinstance(value, dict) and isinstance(value.get("items"), list):
            return value["items"]
    return []


def _sanitize_item_name(name: str) -> str:
    value = name.strip()
    if not value:
        return ""
    value = _LEADING_POSITION_PATTERN.sub("", value)
    value = _TRAILING_POSITION_PATTERN.sub("", value)
    value = value.strip(" -:;,")
    if not value or _POSITIONAL_PATTERN.search(value):
        return ""
    return value


def _sanitize_notes(notes: str) -> str:
    value = notes.strip()
    return "" if _POSITIONAL_PATTERN.search(value) else value


def _normalize_surface(surface: str, surface_ids: list[str]) -> str:
    value = surface.strip().lower()
    for surface_id in surface_ids:
        lowered = surface_id.lower()
        if value == lowered or re.search(
            rf"(?<![a-z0-9_]){re.escape(lowered)}(?![a-z0-9_])",
            value,
        ):
            return surface_id
    return surface_ids[0] if len(surface_ids) == 1 else GENERIC_SURFACE


def _fallback_order(
    suggested_items: str,
    surface_ids: list[str],
) -> list[dict[str, str]]:
    surface = surface_ids[0] if len(surface_ids) == 1 else GENERIC_SURFACE
    candidates = _split_suggested_items(suggested_items)
    candidates.sort(key=_fallback_priority)
    return [
        {
            "item": item,
            "surface": surface,
            "notes": (
                "Required item; sequence by function and size"
                if required
                else "Optional item; sequence by function and size"
            ),
        }
        for item, required in candidates
        if _sanitize_item_name(item)
    ]


def _split_suggested_items(text: str) -> list[tuple[str, bool]]:
    parts = re.split(
        r"\s*(?:,|;|\band\b)\s*",
        str(text or "").replace("\n", ","),
        flags=re.IGNORECASE,
    )
    required = False
    seen: set[str] = set()
    result: list[tuple[str, bool]] = []
    for part in parts:
        cleaned = re.sub(r"^\s*(?:[-*]|\d+[.)])?\s*", "", part)
        label = re.match(
            r"^(required|optional)(?:\s+items?)?\s*:?\s*(.*)$",
            cleaned,
            flags=re.IGNORECASE,
        )
        if label:
            required = label.group(1).lower() == "required"
            cleaned = label.group(2)
        cleaned = re.sub(r"^items?\s*:?\s*", "", cleaned, flags=re.IGNORECASE)
        item = _sanitize_item_name(cleaned.strip(" .:-"))
        if not item or item.lower() in {"none", "n/a", "no specific items"}:
            continue
        if item.lower() not in seen:
            seen.add(item.lower())
            result.append((item, required))
    return result


def _fallback_priority(candidate: tuple[str, bool]) -> tuple[int, int, str]:
    item, required = candidate
    text = item.lower()
    primary = ("lamp", "plant", "vase", "bowl")
    secondary = ("book", "cup", "clock", "phone", "glasses", "remote")
    size_rank = 0 if any(k in text for k in primary) else 1
    if any(k in text for k in secondary):
        size_rank = 2
    return (0 if required else 1, size_rank, text)


def _limit_per_surface(
    items: list[dict[str, str]],
    maximum: int,
) -> list[dict[str, str]]:
    if maximum <= 0:
        return []
    counts: dict[str, int] = {}
    result: list[dict[str, str]] = []
    for item in items:
        surface = item["surface"]
        if counts.get(surface, 0) >= maximum:
            continue
        counts[surface] = counts.get(surface, 0) + 1
        result.append(item)
    return result


def _format_reference(items: list[dict[str, str]]) -> str:
    if not items:
        return ""
    lines = [
        "## Suggested Manipuland Placement Order (planning reference)",
        (
            "This is a soft sequencing guide. It does not prescribe coordinates "
            "or exact positions. The observed scene, support bounds, physics, and "
            "hard constraints remain authoritative."
        ),
        "",
    ]
    for index, item in enumerate(items, start=1):
        line = f"{index}. {item['item']} (surface: {item['surface']})"
        if item["notes"]:
            line += f" - {item['notes']}"
        lines.append(line)
    return "\n".join(lines)


def _read_cache(cache_path: Path) -> dict[str, Any]:
    if not cache_path.exists():
        return {}
    try:
        data = json.loads(cache_path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception as exc:
        console_logger.warning(
            "Ignoring invalid manipuland placement-order cache %s: %s",
            cache_path,
            exc,
        )
        return {}


def _write_cache_best_effort(cache_path: Path, cache: dict[str, Any]) -> None:
    temporary_path: Path | None = None
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=cache_path.parent,
            prefix=f".{cache_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as cache_file:
            json.dump(cache, cache_file, indent=2, ensure_ascii=False)
            cache_file.flush()
            temporary_path = Path(cache_file.name)
        temporary_path.replace(cache_path)
    except Exception as exc:
        console_logger.warning(
            "Failed to write manipuland placement-order cache %s: %s",
            cache_path,
            exc,
        )
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
