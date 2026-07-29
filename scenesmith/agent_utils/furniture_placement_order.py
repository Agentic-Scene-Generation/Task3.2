"""Generate a soft furniture placement-order reference for the designer."""

from __future__ import annotations

import json
import logging
import re
import tempfile

from pathlib import Path
from typing import Any

from omegaconf import DictConfig

from scenesmith.agent_utils.stage_placement_order_config import (
    get_stage_placement_order_config,
)
from scenesmith.agent_utils.vlm_service import VLMService
from scenesmith.utils.llm_json import parse_llm_json

console_logger = logging.getLogger(__name__)

PLACEMENT_ORDER_CACHE_FILENAME = "furniture_placement_order.json"

_PRECISE_POSITION_PATTERN = re.compile(
    r"(?:\b[xyz]\s*[:=]\s*[-+]?\d+(?:\.\d+)?)"
    r"|(?:[\[(]\s*[-+]?\d+(?:\.\d+)?\s*,\s*[-+]?\d+(?:\.\d+)?(?:\s*,"
    r"\s*[-+]?\d+(?:\.\d+)?)?\s*[\])])"
    r"|(?:\b[-+]?\d+(?:\.\d+)?\s*(?:m|cm|mm|meter|meters|metre|metres|"
    r"centimeter|centimeters|centimetre|centimetres|foot|feet|ft|inch|inches|in)\b)"
    r"|(?:\b(?:coordinate|coordinates|exact distance|numeric position)\b)",
    flags=re.IGNORECASE,
)

_FURNITURE_NAMES = (
    "nightstand",
    "bedside table",
    "bed",
    "rug",
    "armchair",
    "chair",
    "sofa",
    "couch",
    "desk",
    "wardrobe",
    "dresser",
    "bookshelf",
    "cabinet",
    "coffee table",
    "dining table",
    "table",
    "plant",
)


def build_furniture_placement_order_reference(
    cfg: DictConfig | dict[str, Any],
    scene_prompt: str,
    scene_dir: Path,
    vlm_service: VLMService,
    model: str,
    room_dimensions: dict[str, float] | None = None,
) -> str:
    """Build the reference, returning an empty string on any unrecoverable error."""
    try:
        stage_cfg = get_stage_placement_order_config(cfg, stage="furniture")
        if not bool(stage_cfg.get("enabled", False)):
            return ""

        items = _load_or_generate_order(
            stage_cfg=stage_cfg,
            scene_prompt=scene_prompt,
            scene_dir=Path(scene_dir),
            vlm_service=vlm_service,
            model=model,
            room_dimensions=room_dimensions,
        )
        max_items = max(0, int(stage_cfg.get("max_items", 20)))
        return _format_reference(items[:max_items])
    except Exception as exc:
        console_logger.warning(
            "Failed to build furniture placement-order reference: %s", exc
        )
        return ""


def _load_or_generate_order(
    *,
    stage_cfg: dict[str, Any],
    scene_prompt: str,
    scene_dir: Path,
    vlm_service: VLMService,
    model: str,
    room_dimensions: dict[str, float] | None,
) -> list[dict[str, str]]:
    cache_enabled = bool(stage_cfg.get("cache", True))
    fallback_enabled = bool(stage_cfg.get("fallback_enabled", True))
    max_items = max(0, int(stage_cfg.get("max_items", 20)))
    cache_path = scene_dir / PLACEMENT_ORDER_CACHE_FILENAME

    if cache_enabled and cache_path.exists():
        try:
            items = _finalize_items(
                json.loads(cache_path.read_text(encoding="utf-8")),
                scene_prompt=scene_prompt,
                fallback_enabled=fallback_enabled,
            )[:max_items]
            _write_cache_best_effort(cache_path, items)
            return items
        except Exception as exc:
            console_logger.warning(
                "Ignoring invalid furniture placement-order cache %s: %s",
                cache_path,
                exc,
            )

    raw_data: Any = {}
    try:
        response = vlm_service.create_completion(
            model=model,
            messages=_build_messages(scene_prompt, room_dimensions),
            reasoning_effort="low",
            verbosity="low",
            response_format={"type": "json_object"},
            enable_thinking=bool(stage_cfg.get("enable_thinking", False)),
        )
        raw_data = parse_llm_json(response)
    except Exception as exc:
        console_logger.warning(
            "Furniture placement-order generation failed%s: %s",
            "; using fallback" if fallback_enabled else "",
            exc,
        )

    items = _finalize_items(
        raw_data,
        scene_prompt=scene_prompt,
        fallback_enabled=fallback_enabled,
    )[:max_items]
    if cache_enabled:
        _write_cache_best_effort(cache_path, items)
    return items


def _build_messages(
    scene_prompt: str,
    room_dimensions: dict[str, float] | None,
) -> list[dict[str, str]]:
    dimensions = "unknown"
    if room_dimensions:
        length = room_dimensions.get("length_m")
        width = room_dimensions.get("width_m")
        if length is not None and width is not None:
            dimensions = f"approximately {length:.2f}m by {width:.2f}m"

    return [
        {
            "role": "system",
            "content": (
                "Create a concise furniture placement order for an indoor scene. "
                "Return JSON only."
            ),
        },
        {
            "role": "user",
            "content": f"""
Room prompt:
{scene_prompt}

Room dimensions:
{dimensions}

Return:
{{
  "furniture": [
    {{"item": "...", "target": "...", "placement_hint": "...", "notes": "..."}}
  ]
}}

Rules:
- Cover rugs and floor-standing furniture only, including bed components when needed.
- Include explicitly requested assets before a few appropriate implied assets.
- Order large anchors before dependent items.
- Split paired items when their placements differ.
- Use broad relationships such as against a wall, beside the bed, or beneath a group.
- Never output coordinates, numeric positions, exact distances, or metric measurements.
- Do not override the observed room, physics results, or hard constraints.
""".strip(),
        },
    ]


def _finalize_items(
    data: Any,
    *,
    scene_prompt: str,
    fallback_enabled: bool,
) -> list[dict[str, str]]:
    items = _dedupe(_expand_pairs(_normalize_items(data)))
    items = _ensure_requested_nightstand_pair(items, scene_prompt)
    if fallback_enabled:
        fallback = _fallback_order(scene_prompt)
        if not items:
            items = fallback
        else:
            existing = {_fallback_identity(item) for item in items}
            for item in fallback:
                identity = _fallback_identity(item)
                if identity[0] and identity not in existing:
                    items.append(item)
                    existing.add(identity)
    return [_ensure_broad_hint(item) for item in _dedupe(items)]


def _extract_items(data: Any) -> list[Any]:
    if isinstance(data, list):
        return data
    if not isinstance(data, dict):
        return []
    for key in ("furniture", "furniture_stage", "items"):
        value = data.get(key)
        if isinstance(value, list):
            return value
        if isinstance(value, dict) and isinstance(value.get("items"), list):
            return value["items"]
    return []


def _normalize_items(data: Any) -> list[dict[str, str]]:
    normalized: list[dict[str, str]] = []
    for raw in _extract_items(data):
        if isinstance(raw, str):
            name, target, hint, notes = raw, "", "", ""
        elif isinstance(raw, dict):
            name = raw.get("item") or raw.get("asset") or raw.get("name") or ""
            target = raw.get("target") or raw.get("surface") or ""
            hint = (
                raw.get("placement_hint")
                or raw.get("placement")
                or raw.get("spatial_relation")
                or ""
            )
            notes = raw.get("notes") or raw.get("rationale") or ""
        else:
            continue

        safe_name = _sanitize_item_name(str(name))
        if not safe_name:
            continue
        normalized.append(
            {
                "item": safe_name,
                "target": _sanitize_optional_text(str(target)),
                "placement_hint": _sanitize_optional_text(str(hint)),
                "notes": _sanitize_optional_text(str(notes)),
            }
        )
    return normalized


def _sanitize_item_name(name: str) -> str:
    value = name.strip()
    if not value:
        return ""
    if not _PRECISE_POSITION_PATTERN.search(value):
        return value
    lower = value.lower()
    for category in _FURNITURE_NAMES:
        if re.search(rf"\b{re.escape(category)}s?\b", lower):
            return category
    return ""


def _sanitize_optional_text(text: str) -> str:
    value = text.strip()
    return "" if _PRECISE_POSITION_PATTERN.search(value) else value


def _expand_pairs(items: list[dict[str, str]]) -> list[dict[str, str]]:
    expanded: list[dict[str, str]] = []
    for item in items:
        text = " ".join(item.values()).lower()
        if "nightstands" in text or ("two" in text and "nightstand" in text):
            notes = item["notes"] or "Paired with the bed"
            expanded.extend(
                [
                    _item(
                        "Left nightstand",
                        "left side of bed",
                        "beside the head side of the bed",
                        notes,
                    ),
                    _item(
                        "Right nightstand",
                        "right side of bed",
                        "beside the head side of the bed",
                        notes,
                    ),
                ]
            )
        else:
            expanded.append(item)
    return expanded


def _ensure_requested_nightstand_pair(
    items: list[dict[str, str]],
    scene_prompt: str,
) -> list[dict[str, str]]:
    prompt = scene_prompt.lower()
    pair_requested = bool(
        re.search(
            r"\b(?:two|pair(?:ed)?(?:\s+of)?)\s+(?:bedside\s+)?nightstands?\b",
            prompt,
        )
        or "nightstands" in prompt
    )
    if not pair_requested:
        return items

    indices = [
        index
        for index, item in enumerate(items)
        if _canonical_category(item["item"]) == "nightstand"
    ]
    if len(indices) != 1:
        return items

    index = indices[0]
    notes = items[index]["notes"] or "Paired with the bed"
    return [
        *items[:index],
        _item(
            "Left nightstand",
            "left side of bed",
            "beside the head side of the bed",
            notes,
        ),
        _item(
            "Right nightstand",
            "right side of bed",
            "beside the head side of the bed",
            notes,
        ),
        *items[index + 1 :],
    ]


def _fallback_order(scene_prompt: str) -> list[dict[str, str]]:
    prompt = scene_prompt.lower()
    items: list[dict[str, str]] = []
    if "rug" in prompt:
        items.append(
            _item(
                "Area rug",
                "main furniture grouping",
                "centered beneath the main grouping",
                "Floor anchor",
            )
        )
    if "bed" in prompt:
        items.append(
            _item(
                "Bed",
                "primary solid wall",
                "against the main solid wall with circulation at the foot",
                "Primary room anchor",
            )
        )
    if "nightstand" in prompt or "bedside" in prompt:
        items.extend(
            [
                _item(
                    "Left nightstand",
                    "left side of bed",
                    "beside the head side of the bed",
                ),
                _item(
                    "Right nightstand",
                    "right side of bed",
                    "beside the head side of the bed",
                ),
            ]
        )
    fallback_specs = (
        ("sofa", "Sofa", "main wall or focal grouping"),
        ("desk", "Desk", "appropriate work wall or work zone"),
        ("wardrobe", "Wardrobe", "secondary wall or open corner"),
        ("dresser", "Dresser", "secondary wall"),
        ("bookshelf", "Bookshelf", "available solid wall"),
        ("coffee table", "Coffee table", "main seating grouping"),
        ("dining table", "Dining table", "dining zone"),
        ("armchair", "Armchair", "quiet corner or seating grouping"),
        ("chair", "Chair", "relevant table or seating grouping"),
        ("plant", "Floor plant", "open corner beside a larger anchor"),
    )
    for keyword, name, target in fallback_specs:
        if keyword in prompt and _canonical_category(name) not in {
            _canonical_category(item["item"]) for item in items
        }:
            items.append(_item(name, target, f"within the {target}"))
    return items


def _ensure_broad_hint(item: dict[str, str]) -> dict[str, str]:
    if item["placement_hint"]:
        return item
    result = dict(item)
    category = _canonical_category(item["item"])
    hints = {
        "rug": "centered beneath the main furniture grouping",
        "bed": "against the main solid wall",
        "nightstand": "beside the head side of the bed",
        "sofa": "against a main wall or facing the focal point",
        "desk": "within the work zone against an appropriate wall",
        "wardrobe": "against a secondary wall or in an open corner",
        "table": "within the relevant seating or functional grouping",
        "chair": "within the relevant seating grouping",
        "plant": "in an open corner beside a larger anchor",
    }
    result["placement_hint"] = hints.get(category, "")
    return result


def _canonical_category(name: str) -> str:
    text = name.lower()
    for category in (
        "nightstand",
        "rug",
        "bed",
        "sofa",
        "desk",
        "wardrobe",
        "dresser",
        "bookshelf",
        "table",
        "chair",
        "plant",
    ):
        if category in text:
            return category
    return re.sub(r"\W+", " ", text).strip()


def _fallback_identity(item: dict[str, str]) -> tuple[str, str]:
    category = _canonical_category(item["item"])
    if category == "nightstand":
        target = item["target"].lower()
        if "left" in target or "left" in item["item"].lower():
            return category, "left"
        if "right" in target or "right" in item["item"].lower():
            return category, "right"
    return category, ""


def _dedupe(items: list[dict[str, str]]) -> list[dict[str, str]]:
    seen: set[tuple[str, str]] = set()
    result: list[dict[str, str]] = []
    for item in items:
        key = (item["item"].lower(), item["target"].lower())
        if key not in seen:
            seen.add(key)
            result.append(item)
    return result


def _item(
    name: str,
    target: str,
    hint: str,
    notes: str = "",
) -> dict[str, str]:
    return {
        "item": name,
        "target": target,
        "placement_hint": hint,
        "notes": notes,
    }


def _format_reference(items: list[dict[str, str]]) -> str:
    if not items:
        return ""
    lines = [
        "## Suggested Furniture Placement Order (planning reference)",
        (
            "This is soft planning guidance, not a hard constraint. The observed "
            "room, physics results, functional needs, and hard constraints remain "
            "authoritative. All directions are broad relationships."
        ),
        "",
    ]
    for index, item in enumerate(items, start=1):
        line = f"{index}. {item['item']}"
        if item["target"]:
            line += f" (target: {item['target']})"
        if item["placement_hint"]:
            line += f" - {item['placement_hint']}"
        if item["notes"]:
            line += f" [{item['notes']}]"
        lines.append(line)
    return "\n".join(lines)


def _write_cache_best_effort(
    cache_path: Path,
    items: list[dict[str, str]],
) -> None:
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
            json.dump(items, cache_file, indent=2, ensure_ascii=False)
            cache_file.flush()
            temporary_path = Path(cache_file.name)
        temporary_path.replace(cache_path)
    except Exception as exc:
        console_logger.warning(
            "Failed to write furniture placement-order cache %s: %s",
            cache_path,
            exc,
        )
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
