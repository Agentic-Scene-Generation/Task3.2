"""Lookup merged asset-library annotations by HSSD id.

The default lookup artifact is the generated HSSD-default asset-library
annotation layer bundled with this critic package.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any

DEFAULT_LOOKUP = (
    Path(__file__).resolve().parent
    / "asset_annotation_data"
    / "hssd_annotation_lookup.json.gz"
)
DEFAULT_CLEARANCE_DIR = (
    Path(__file__).resolve().parent / "metrics" / "interaction_clearance" / "data"
)
DEFAULT_UNIFIED_AFFORDANCE_DIR = Path(
    "/data/share/ud4scenesmith/ud4scenesmith_3_2_final_20260513/"
    "affordance_annotations/unified_layer_v0_1"
)
DEFAULT_OPERATION_SPACE_DIR = Path(
    "/data/share/ud4scenesmith/ud4scenesmith_3_2_final_20260513/"
    "affordance_annotations/operation_space_hssd_official"
)
DEFAULT_NONARTIC_CLEARANCE_V2 = Path(
    "/data/250010098/NONARTIC_CLEARANCE_ANNOTATIONS_V2.json"
)
DEFAULT_OFFICIAL_COMBINED_CLEARANCE = Path(
    "/data/share/ud4scenesmith/clearance_fullrun_20260606/"
    "official_combined_clearance.json"
)
DEFAULT_HSSD_ARTICULATION_CLEARANCE_RUN = Path(
    "/data/250010098/clearance_retrieval_pilot_20260609/"
    "hssd_clearance_run_results.json"
)
DEFAULT_HSSD_CLEARANCE_VOXEL_RESULTS = Path(
    "/data/250010098/clearance_retrieval_pilot_20260609/"
    "hssd_clearance_voxel_results.json"
)


def normalize_hssd_id(value: str) -> str:
    value = str(value or "").strip()
    if value.lower().startswith("hssd:"):
        value = value.split(":", 1)[1]
    return value


_SCENEBENCHMARK_AFFORDANCE_VOCAB = {
    "sittable",
    "sleepable",
    "supportable",
    "openable",
    "containable",
    "toggleable",
    "graspable",
}
_AFFORDANCE_CATEGORY_KEYWORDS: tuple[tuple[tuple[str, ...], tuple[str, ...]], ...] = (
    (
        ("chair", "stool", "sofa", "couch", "bench", "seat", "loveseat"),
        ("sittable",),
    ),
    (("bed", "mattress"), ("sittable", "sleepable", "supportable")),
    (
        (
            "table",
            "desk",
            "shelf",
            "stand",
            "counter",
            "cabinet",
            "dresser",
            "nightstand",
            "buffet",
            "armoire",
            "cart",
            "sideboard",
            "console",
            "island",
            "sink",
            "toilet",
        ),
        ("supportable",),
    ),
    (
        (
            "cabinet",
            "drawer",
            "door",
            "dresser",
            "wardrobe",
            "refrigerator",
            "microwave",
            "oven",
            "storage",
            "tv_stand",
            "tv stand",
            "armoire",
            "buffet",
            "hamper",
            "basket",
            "trunk",
            "washer",
            "dryer",
        ),
        ("containable", "openable"),
    ),
    (("lamp", "light", "switch"), ("toggleable",)),
    (
        (
            "bottle",
            "bowl",
            "book",
            "cup",
            "glass",
            "keyboard",
            "laptop",
            "monitor",
            "mouse",
            "mug",
            "plate",
            "remote",
            "tablet",
            "vase",
        ),
        ("graspable",),
    ),
)
_CATEGORY_GROUP_KEYWORDS: tuple[tuple[tuple[str, ...], str], ...] = (
    (("chair", "stool", "sofa", "bench", "seat", "loveseat"), "seating"),
    (("toilet",), "seating"),
    (("bed", "mattress"), "sleeping"),
    (
        (
            "wall art",
            "wall decor",
            "picture frame",
            "wall sign",
            "wall clock",
            "mirror",
            "rug",
            "mat",
            "throw pillow",
            "pillow",
            "sculpture",
            "plant",
            "flower",
            "wreath",
        ),
        "decor",
    ),
    (
        ("shelf", "stand", "sideboard", "console", "nightstand", "bookcase", "buffet", "cart"),
        "storage_surface",
    ),
    (("television", "tv", "monitor", "screen", "speaker"), "media"),
    (("lamp", "light"), "lighting"),
    (
        ("table", "desk", "counter", "island"),
        "work_surface",
    ),
    (
        (
            "cabinet",
            "drawer",
            "dresser",
            "wardrobe",
            "armoire",
            "refrigerator",
            "microwave",
            "oven",
            "storage",
            "washer",
            "dryer",
            "hamper",
            "basket",
            "trunk",
        ),
        "storage",
    ),
    (
        (
            "bottle",
            "bowl",
            "book",
            "cup",
            "glass",
            "keyboard",
            "laptop",
            "mouse",
            "mug",
            "plate",
            "remote",
            "tablet",
            "vase",
        ),
        "small_object",
    ),
)
_SMALL_OBJECT_HINTS = {
    "bottle",
    "bowl",
    "book",
    "cup",
    "glass",
    "keyboard",
    "laptop",
    "mouse",
    "mug",
    "plate",
    "remote",
    "tablet",
    "vase",
}
_WALL_MOUNTED_HINTS = (
    "wall art",
    "wall decor",
    "wall mirror",
    "wall sign",
    "wall clock",
    "picture frame",
    "curtain",
    "shade",
    "blind",
    "showerhead",
    "towel rack",
    "hook rack",
)


def _category_text(record: dict[str, Any]) -> str:
    raw = (
        record.get("category_key")
        or record.get("category")
        or record.get("asset_uid")
        or ""
    )
    return str(raw).strip().lower().replace("-", "_").replace(" ", "_")


def _category_words(record: dict[str, Any]) -> str:
    return _category_text(record).replace("_", " ")


def _normalize_category_token(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = text.replace("-", "_").replace(" ", "_")
    text = "".join(ch for ch in text if ch.isalnum() or ch == "_")
    while "__" in text:
        text = text.replace("__", "_")
    return text.strip("_")


def _category_contains(category_words: str, needles: tuple[str, ...]) -> bool:
    return any(
        needle in category_words or needle.replace("_", " ") in category_words
        for needle in needles
    )


def _scenebenchmark_affordances(record: dict[str, Any]) -> set[str]:
    category_words = _category_words(record)
    affordances: set[str] = set()
    for needles, labels in _AFFORDANCE_CATEGORY_KEYWORDS:
        if _category_contains(category_words, needles):
            affordances.update(labels)
    return affordances & _SCENEBENCHMARK_AFFORDANCE_VOCAB


def _scenebenchmark_category_group(
    record: dict[str, Any], affordances: set[str]
) -> str:
    category_words = _category_words(record)
    for needles, group in _CATEGORY_GROUP_KEYWORDS:
        if _category_contains(category_words, needles):
            return group
    if "supportable" in affordances:
        return "work_surface"
    if "graspable" in affordances:
        return "small_object"
    return "object"


def _scenebenchmark_scene_object_type(record: dict[str, Any], group: str) -> str:
    policy = record.get("week27_asset_policy") or {}
    raw = str(policy.get("scene_object_type") or "").strip()
    if raw and raw != "unknown":
        return raw
    category_words = _category_words(record)
    if "ceiling" in category_words:
        return "ceiling_mounted"
    if _category_contains(category_words, _WALL_MOUNTED_HINTS):
        return "wall_mounted"
    if group == "small_object" or _category_contains(category_words, tuple(_SMALL_OBJECT_HINTS)):
        return "manipuland"
    if group == "decor":
        return "manipuland"
    if group in {
        "seating",
        "sleeping",
        "work_surface",
        "storage",
        "storage_surface",
        "media",
        "lighting",
    }:
        return "furniture"
    placement_dof = record.get("placement_dof") or {}
    if placement_dof.get("dof") in {1, 2}:
        return "furniture"
    return "manipuland"


def _scenebenchmark_mobility_class(
    record: dict[str, Any], group: str, scene_object_type: str
) -> str:
    policy = record.get("week27_asset_policy") or {}
    raw = str(policy.get("mobility_class") or "").strip()
    if raw and raw != "unknown":
        return raw
    if scene_object_type in {"wall_mounted", "ceiling_mounted"}:
        return "mounted"
    category_words = _category_words(record)
    if group == "small_object":
        return "movable"
    if group == "seating" and _category_contains(category_words, ("chair", "stool")):
        return "movable"
    if group in {"seating", "work_surface", "storage_surface", "media"}:
        return "semi_movable"
    if group in {"sleeping", "storage"}:
        return "fixed"
    return "unknown"


def _scenebenchmark_accessibility_policy(
    record: dict[str, Any],
    affordances: set[str],
    scene_object_type: str,
    group: str,
) -> str:
    policy = record.get("week27_asset_policy") or {}
    if scene_object_type in {"wall_mounted", "ceiling_mounted"} or group == "decor":
        return "ignored"
    raw = str(policy.get("accessibility_policy") or "").strip()
    if raw in {"required", "optional", "ignored"}:
        return raw
    if group == "small_object" or scene_object_type == "manipuland":
        return "ignored"
    return "required" if affordances else "ignored"


def _scenebenchmark_front_hint(record: dict[str, Any], affordances: set[str]) -> str | None:
    policy = record.get("week27_asset_policy") or {}
    for side in policy.get("access_sides") or []:
        side_text = str(side).strip().lower()
        if side_text in {"front", "back", "left", "right", "top", "bottom"}:
            return side_text
    canonical = record.get("canonical_front") or {}
    if canonical.get("canonical_orientation_is_semantic_front") is True:
        return "front"
    if "openable" in affordances or "sittable" in affordances:
        return "front"
    if "supportable" in affordances:
        return "top"
    return None


def _scenebenchmark_access_sides(
    record: dict[str, Any], affordances: set[str], front_hint: str | None
) -> list[str]:
    policy = record.get("week27_asset_policy") or {}
    sides = [
        str(side).strip().lower()
        for side in policy.get("access_sides") or []
        if str(side).strip().lower() in {"front", "back", "left", "right", "top", "bottom"}
    ]
    face = front_hint if front_hint in {"front", "back", "left", "right"} else "front"
    if "openable" in affordances or "sittable" in affordances:
        sides.append(face)
    if "supportable" in affordances:
        sides.append("top")
    if "sleepable" in affordances:
        sides.extend(["left", "right", face])
    return list(dict.fromkeys(sides))


def _interaction_surface_map(affordances: set[str]) -> dict[str, list[str]]:
    mapping: dict[str, list[str]] = {}
    if "sittable" in affordances:
        mapping["sittable"] = ["top", "front"]
    if "sleepable" in affordances:
        mapping["sleepable"] = ["top", "left", "right"]
    if "supportable" in affordances:
        mapping["supportable"] = ["top"]
    if "openable" in affordances:
        mapping["openable"] = ["front"]
    if "containable" in affordances:
        mapping["containable"] = ["front", "inside"]
    if "toggleable" in affordances:
        mapping["toggleable"] = ["front"]
    if "graspable" in affordances:
        mapping["graspable"] = ["top", "side"]
    return mapping


def _interaction_height_m(record: dict[str, Any], affordances: set[str]) -> dict[str, float]:
    category_words = _category_words(record)
    heights: dict[str, float] = {}
    if "sittable" in affordances:
        heights["sittable"] = 0.45
    if "sleepable" in affordances:
        heights["sleepable"] = 0.55
    if "supportable" in affordances:
        if "coffee" in category_words:
            heights["supportable"] = 0.42
        elif "counter" in category_words or "island" in category_words:
            heights["supportable"] = 0.9
        elif "desk" in category_words or "table" in category_words:
            heights["supportable"] = 0.74
        else:
            heights["supportable"] = 0.8
    if "openable" in affordances:
        heights["openable"] = 0.9
    if "containable" in affordances:
        heights["containable"] = heights.get("openable", 0.9)
    if "toggleable" in affordances:
        heights["toggleable"] = 1.2
    if "graspable" in affordances:
        heights["graspable"] = 1.0
    return heights


def _relation_target_categories(record: dict[str, Any]) -> list[str]:
    targets: list[str] = []
    for relation in record.get("relation_priors") or []:
        if not isinstance(relation, dict):
            continue
        if relation.get("target_kind") != "asset_category":
            continue
        target = _normalize_category_token(relation.get("target_category"))
        if target and target not in targets:
            targets.append(target)
    partners = ((record.get("interaction_clearance") or {}).get("functional_partners") or {})
    for partner in partners.get("partners") or []:
        target = _normalize_category_token(partner)
        if target and target not in targets:
            targets.append(target)
    return targets


def _functional_dependencies(record: dict[str, Any]) -> list[dict[str, Any]]:
    dependencies: list[dict[str, Any]] = []
    for relation in record.get("relation_priors") or []:
        if not isinstance(relation, dict):
            continue
        target_kind = relation.get("target_kind")
        if target_kind != "asset_category":
            continue
        target = _normalize_category_token(relation.get("target_category"))
        if not target:
            continue
        relation_type = str(
            relation.get("functional_dependency_relation")
            or relation.get("relation_type")
            or "used_with"
        )
        dependencies.append(
            {
                "relation_type": relation_type.replace("funeval_", ""),
                "target_category": target,
                "target_kind": "object_category",
                "distance_range_m": relation.get("distance_range_m"),
                "relative_facing": relation.get("relative_facing"),
                "relative_position": relation.get("relative_position"),
                "height_relation": relation.get("height_relation"),
                "confidence": relation.get("functional_dependency_confidence")
                or relation.get("confidence"),
                "reason": relation.get("reason")
                or f"HSSD relation prior: {relation_type} -> {target}",
                "source": "hssd_annotations:relation_priors",
            }
        )
    return dependencies


def _attachment_dependencies(record: dict[str, Any]) -> list[dict[str, Any]]:
    dependencies: list[dict[str, Any]] = []
    for relation in record.get("relation_priors") or []:
        if not isinstance(relation, dict):
            continue
        if relation.get("target_kind") != "environment_anchor":
            continue
        anchor = _normalize_category_token(relation.get("environment_anchor"))
        if not anchor:
            continue
        dependencies.append(
            {
                "relation_type": relation.get("relation_type") or "against",
                "target_kind": "environment_anchor",
                "environment_anchor": anchor,
                "distance_range_m": relation.get("distance_range_m"),
                "relative_facing": relation.get("relative_facing"),
                "confidence": relation.get("confidence"),
                "source": "hssd_annotations:relation_priors",
            }
        )
    for anchor in record.get("environment_anchors") or []:
        if not isinstance(anchor, dict):
            continue
        anchor_name = _normalize_category_token(anchor.get("anchor"))
        if not anchor_name:
            continue
        item = {
            "relation_type": anchor.get("relation_type") or "supported_by",
            "target_kind": "environment_anchor",
            "environment_anchor": anchor_name,
            "confidence": anchor.get("confidence"),
            "source": "hssd_annotations:environment_anchors",
        }
        if item not in dependencies:
            dependencies.append(item)
    return dependencies


def _orientation_dependencies(record: dict[str, Any]) -> list[dict[str, Any]]:
    dependencies: list[dict[str, Any]] = []
    policy = record.get("week27_asset_policy") or {}
    for item in policy.get("orientation_dependencies") or []:
        if isinstance(item, dict):
            dep = dict(item)
            if dep.get("target_category"):
                dep["target_category"] = _normalize_category_token(
                    dep.get("target_category")
                )
            dep.setdefault("source", "hssd_annotations:week27_asset_policy")
            dependencies.append(dep)
    for relation in record.get("relation_priors") or []:
        if not isinstance(relation, dict):
            continue
        if relation.get("target_kind") != "asset_category":
            continue
        facing = str(relation.get("relative_facing") or "")
        if "front" not in facing and relation.get("relation_type") not in {"faces", "around"}:
            continue
        target = _normalize_category_token(relation.get("target_category"))
        if not target:
            continue
        dep = {
            "relation_type": "front_faces",
            "target_category": target,
            "target_kind": "object_category",
            "max_distance_m": (relation.get("distance_range_m") or [None, None])[-1],
            "confidence": relation.get("confidence"),
            "source": "hssd_annotations:relation_priors",
        }
        if dep not in dependencies:
            dependencies.append(dep)
    return dependencies


def _metric_relevance(
    affordances: set[str],
    accessibility_policy: str,
    target_relations: list[str],
    has_keep_clear: bool,
) -> dict[str, float]:
    if accessibility_policy == "ignored":
        sa = 0.0
    elif {"openable", "sittable"} & affordances:
        sa = 1.0
    elif {"sleepable", "supportable", "graspable"} & affordances:
        sa = 0.8
    else:
        sa = 0.0
    return {
        "spatial_accessibility": sa,
        "functional_dependency": 1.0 if target_relations else 0.0,
        "interaction_clearance": 0.9 if has_keep_clear else 0.0,
    }


def build_scenebenchmark_annotation(record: dict[str, Any]) -> dict[str, Any]:
    """Return SceneBenchmark-normalized FD/SA fields for one HSSD record."""
    affordances = _scenebenchmark_affordances(record)
    group = _scenebenchmark_category_group(record, affordances)
    scene_object_type = _scenebenchmark_scene_object_type(record, group)
    mobility_class = _scenebenchmark_mobility_class(record, group, scene_object_type)
    accessibility_policy = _scenebenchmark_accessibility_policy(
        record, affordances, scene_object_type, group
    )
    front_hint = _scenebenchmark_front_hint(record, affordances)
    access_sides = _scenebenchmark_access_sides(record, affordances, front_hint)
    canonical_front = record.get("canonical_front") or {}
    target_relations = _relation_target_categories(record)
    functional_dependencies = _functional_dependencies(record)
    attachment_dependencies = _attachment_dependencies(record)
    orientation_dependencies = _orientation_dependencies(record)
    has_keep_clear = bool((record.get("interaction_clearance") or {}).get("has_keep_clear"))
    benchmark_relevance = (
        "functional"
        if affordances or target_relations or accessibility_policy == "required"
        else "decorative"
    )

    hints: dict[str, Any] = {
        "functional_categories": sorted(affordances),
        "candidate_affordances": sorted(affordances),
        "affordance_confidence": 0.78 if affordances else 0.0,
        "affordance_source": "hssd_annotations:category_policy",
        "accessibility_policy": accessibility_policy,
        "scene_object_type": scene_object_type,
        "mobility_class": mobility_class,
        "category_group": group,
        "benchmark_relevance": benchmark_relevance,
        "access_sides": access_sides,
        "target_relation": target_relations,
        "explicit_target_relation": target_relations,
        "functional_dependencies": functional_dependencies,
        "attachment_dependencies": attachment_dependencies,
        "orientation_dependencies": orientation_dependencies,
        "interaction_surface_map": _interaction_surface_map(affordances),
        "interaction_height_m": _interaction_height_m(record, affordances),
        "metric_relevance": _metric_relevance(
            affordances, accessibility_policy, target_relations, has_keep_clear
        ),
        "classification_source": "hssd_annotations",
        "asset_annotation_source": "hssd_annotations",
    }
    if front_hint:
        hints["front_hint"] = front_hint
        hints["front_face"] = front_hint
        hints["access_direction"] = front_hint
    axis = canonical_front.get("asset_local_front_axis") or canonical_front.get(
        "canonical_orientation_axis"
    )
    if axis is not None:
        hints["asset_local_front_axis"] = axis
    if canonical_front:
        hints["front_confidence"] = canonical_front.get(
            "canonical_orientation_confidence",
            canonical_front.get("confidence"),
        )
        hints["canonical_orientation_is_semantic_front"] = canonical_front.get(
            "canonical_orientation_is_semantic_front"
        )

    return {
        "schema_version": "scenebenchmark_hssd_fd_sa@0.1",
        "functional_hints": hints,
        "functional_dependencies": functional_dependencies,
        "support_regions": [],
    }


class AssetLibraryAnnotationStore:
    """In-process search library for asset-library and clearance annotations."""

    def __init__(
        self,
        lookup_path: str | Path = DEFAULT_LOOKUP,
        clearance_dir: str | Path = DEFAULT_CLEARANCE_DIR,
        unified_affordance_dir: str | Path = DEFAULT_UNIFIED_AFFORDANCE_DIR,
        operation_space_dir: str | Path = DEFAULT_OPERATION_SPACE_DIR,
        nonartic_clearance_v2_path: str | Path = DEFAULT_NONARTIC_CLEARANCE_V2,
        official_combined_clearance_path: str | Path = DEFAULT_OFFICIAL_COMBINED_CLEARANCE,
        hssd_articulation_clearance_run_path: str | Path = DEFAULT_HSSD_ARTICULATION_CLEARANCE_RUN,
        hssd_clearance_voxel_results_path: str | Path = DEFAULT_HSSD_CLEARANCE_VOXEL_RESULTS,
    ) -> None:
        self.lookup_path = Path(lookup_path)
        self.clearance_dir = Path(clearance_dir)
        self.unified_affordance_dir = Path(unified_affordance_dir)
        self.operation_space_dir = Path(operation_space_dir)
        self.nonartic_clearance_v2_path = Path(nonartic_clearance_v2_path)
        self.official_combined_clearance_path = Path(official_combined_clearance_path)
        self.hssd_articulation_clearance_run_path = Path(hssd_articulation_clearance_run_path)
        self.hssd_clearance_voxel_results_path = Path(hssd_clearance_voxel_results_path)
        self._records: dict[str, dict[str, Any]] | None = None
        self._nonartic_clearance: dict[str, dict[str, Any]] | None = None
        self._artic_clearance: dict[str, dict[str, Any]] | None = None
        self._functional_partners: dict[str, dict[str, Any]] | None = None
        self._unified_affordance_index: dict[str, Path] | None = None
        self._nonartic_clearance_v2: dict[str, dict[str, Any]] | None = None
        self._official_combined_clearance: dict[str, dict[str, Any]] | None = None
        self._hssd_articulation_clearance_run: dict[str, dict[str, Any]] | None = None
        self._hssd_clearance_voxel_results_cache: dict[str, dict[str, Any]] | None = None

    def _load(self) -> dict[str, dict[str, Any]]:
        if self._records is None:
            opener = gzip.open if self.lookup_path.suffix == ".gz" else open
            with opener(self.lookup_path, "rt", encoding="utf-8") as f:
                self._records = json.load(f)
        return self._records

    def _load_clearance_items(self, filename: str) -> dict[str, dict[str, Any]]:
        path = self.clearance_dir / filename
        if not path.exists():
            return {}
        try:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, ValueError):
            return {}
        return data.get("items", {}) if isinstance(data, dict) else {}

    def _nonartic(self) -> dict[str, dict[str, Any]]:
        if self._nonartic_clearance is None:
            self._nonartic_clearance = self._load_clearance_items(
                "nonartic_clearance_index.json"
            )
        return self._nonartic_clearance

    def _artic(self) -> dict[str, dict[str, Any]]:
        if self._artic_clearance is None:
            self._artic_clearance = self._load_clearance_items(
                "artic_clearance_index.json"
            )
        return self._artic_clearance

    def _partners(self) -> dict[str, dict[str, Any]]:
        if self._functional_partners is None:
            self._functional_partners = self._load_clearance_items(
                "functional_partners_index.json"
            )
        return self._functional_partners

    def _load_json_if_present(self, path: Path) -> Any:
        if not path.exists():
            return None
        try:
            with path.open("r", encoding="utf-8") as f:
                return json.load(f)
        except (OSError, ValueError):
            # 2026-07-07: Optional external enrichment paths may be missing,
            # unreadable, or partially mounted; keep bundled annotations usable.
            return None

    def _unified_index(self) -> dict[str, Path]:
        if self._unified_affordance_index is None:
            index_path = self.unified_affordance_dir / "index.jsonl"
            records: dict[str, Path] = {}
            if index_path.exists():
                try:
                    with index_path.open("r", encoding="utf-8") as f:
                        for line in f:
                            if not line.strip():
                                continue
                            row = json.loads(line)
                            asset_id = normalize_hssd_id(row.get("asset_id", ""))
                            records[asset_id] = self.unified_affordance_dir / row["record"]
                except OSError:
                    # 2026-07-07: External UD4 affordance files are optional
                    # enrichment; permission/mount failures must not break the
                    # self-contained bundled HSSD lookup.
                    records = {}
            self._unified_affordance_index = records
        return self._unified_affordance_index

    def _nonartic_v2(self) -> dict[str, dict[str, Any]]:
        if self._nonartic_clearance_v2 is None:
            data = self._load_json_if_present(self.nonartic_clearance_v2_path) or {}
            annotations = data.get("annotations") or []
            self._nonartic_clearance_v2 = {
                str(item.get("object_id")): item
                for item in annotations
                if isinstance(item, dict) and item.get("object_id")
            }
        return self._nonartic_clearance_v2

    def _official_combined(self) -> dict[str, dict[str, Any]]:
        if self._official_combined_clearance is None:
            data = self._load_json_if_present(self.official_combined_clearance_path) or {}
            self._official_combined_clearance = data if isinstance(data, dict) else {}
        return self._official_combined_clearance

    def _hssd_articulation_run(self) -> dict[str, dict[str, Any]]:
        if self._hssd_articulation_clearance_run is None:
            data = self._load_json_if_present(self.hssd_articulation_clearance_run_path) or []
            records: dict[str, dict[str, Any]] = {}
            if isinstance(data, list):
                for item in data:
                    if not isinstance(item, dict):
                        continue
                    asset_id = normalize_hssd_id(item.get("hssd_id", ""))
                    if asset_id:
                        records[asset_id] = item
            self._hssd_articulation_clearance_run = records
        return self._hssd_articulation_clearance_run

    def _hssd_clearance_voxel_results(self) -> dict[str, dict[str, Any]]:
        if self._hssd_clearance_voxel_results_cache is None:
            data = self._load_json_if_present(self.hssd_clearance_voxel_results_path) or []
            records: dict[str, dict[str, Any]] = {}
            if isinstance(data, list):
                for item in data:
                    if not isinstance(item, dict):
                        continue
                    asset_id = normalize_hssd_id(item.get("hssd_id", ""))
                    if asset_id:
                        records[asset_id] = item
            self._hssd_clearance_voxel_results_cache = records
        return self._hssd_clearance_voxel_results_cache

    def get_unified_affordance_annotations(self, hssd_id: str) -> dict[str, Any]:
        normalized = normalize_hssd_id(hssd_id)
        record_path = self._unified_index().get(normalized)
        if record_path is None or not record_path.exists():
            return {
                "available": False,
                "source_layer": "unified_layer_v0_1",
                "asset_id": normalized,
            }
        record = self._load_json_if_present(record_path)
        return {
            "available": record is not None,
            "source_layer": "unified_layer_v0_1",
            "asset_id": normalized,
            "record_path": str(record_path),
            "record": record,
        }

    def get_operation_space_annotations(self, hssd_id: str) -> dict[str, Any]:
        normalized = normalize_hssd_id(hssd_id)
        record_path = self.operation_space_dir / "records" / f"{normalized}.json"
        record = self._load_json_if_present(record_path)
        summary = self._load_json_if_present(self.operation_space_dir / "SUMMARY.json")
        return {
            "available": record is not None,
            "source_layer": "operation_space_hssd_official",
            "asset_id": normalized,
            "record_path": str(record_path) if record_path.exists() else None,
            "summary": summary,
            "record": record,
        }

    def get_clearance_region_annotations(self, hssd_id: str) -> dict[str, Any]:
        normalized = normalize_hssd_id(hssd_id)
        return {
            "asset_id": normalized,
            "nonartic_clearance_v2": self._nonartic_v2().get(normalized),
            "official_combined_clearance": self._official_combined().get(normalized),
            "hssd_articulation_clearance_run": self._hssd_articulation_run().get(normalized),
            "hssd_clearance_voxel_metrics": self._hssd_clearance_voxel_results().get(normalized),
            "sources": {
                "nonartic_clearance_v2": str(self.nonartic_clearance_v2_path),
                "official_combined_clearance": str(self.official_combined_clearance_path),
                "hssd_articulation_clearance_run": str(self.hssd_articulation_clearance_run_path),
                "hssd_clearance_voxel_metrics": str(self.hssd_clearance_voxel_results_path),
            },
        }

    def get_clearance_annotations(self, hssd_id: str) -> dict[str, Any]:
        """Return clearance-session annotations keyed by HSSD id.

        This is the asset-level lookup side of the interaction_clearance work:
        human-anchored non-artic keep-clear, articulated swept-volume envelopes,
        and functional-dependency partner categories used for exclusion.
        """
        normalized = normalize_hssd_id(hssd_id)
        nonartic = self._nonartic().get(normalized)
        artic = self._artic().get(normalized)
        partners = self._partners().get(normalized)
        articulation_run = self._hssd_articulation_run().get(normalized)
        voxel_metrics = self._hssd_clearance_voxel_results().get(normalized)
        return {
            "metric": "interaction_clearance",
            "source_session": "clearance-plan-execution-w1-w2",
            "asset_id": normalized,
            "has_keep_clear": nonartic is not None
            or artic is not None
            or articulation_run is not None
            or voxel_metrics is not None,
            "has_nonarticulated_keep_clear": nonartic is not None,
            "has_articulated_swept_volume": artic is not None,
            "has_hssd_articulation_clearance_run": articulation_run is not None,
            "has_hssd_clearance_voxel_metrics": voxel_metrics is not None,
            "has_functional_partners": partners is not None,
            "nonarticulated_keep_clear": nonartic,
            "articulated_swept_volume": artic,
            "hssd_articulation_clearance_run": articulation_run,
            "hssd_clearance_voxel_metrics": voxel_metrics,
            "functional_partners": partners,
        }

    def get(self, hssd_id: str) -> dict[str, Any] | None:
        normalized = normalize_hssd_id(hssd_id)
        record = self._load().get(normalized)
        # The bundled lookup now carries inline interaction_clearance and
        # post_replacement, so the record is self-sufficient for portability.
        # External layers below are OPTIONAL enrichment: when their absolute
        # source paths are missing (e.g. on a fresh clone) each getter returns
        # an ``available: False`` stub rather than failing, and the inline
        # bundled fields remain authoritative.
        interaction_clearance = self.get_clearance_annotations(normalized)
        ud4_affordance = self.get_unified_affordance_annotations(normalized)
        operation_space = self.get_operation_space_annotations(normalized)
        clearance_regions = self.get_clearance_region_annotations(normalized)
        if record is None:
            has_clearance_region = any(
                clearance_regions.get(key) is not None
                for key in (
                    "nonartic_clearance_v2",
                    "official_combined_clearance",
                    "hssd_articulation_clearance_run",
                    "hssd_clearance_voxel_metrics",
                )
            )
            if (
                not interaction_clearance["has_keep_clear"]
                and not interaction_clearance["has_functional_partners"]
                and not has_clearance_region
                and not ud4_affordance["available"]
                and not operation_space["available"]
            ):
                return None
            articulation_run = clearance_regions.get("hssd_articulation_clearance_run") or {}
            category = articulation_run.get("cat") or articulation_run.get("pnm_cat")
            out = {
                "asset_uid": f"hssd:{normalized}",
                "source_id": normalized,
                "source": "auxiliary_annotation_layers",
                "source_scope": "hssd_auxiliary_annotation_only",
                "schema_version": "auxiliary_annotation_lookup_v0",
                "category": category,
                "category_key": category,
                "provenance_meta": {
                    "note": (
                        "This HSSD id is not present in the generated asset-library "
                        "policy lookup, but auxiliary clearance/affordance layers "
                        "contain annotations for it."
                    )
                },
            }
        else:
            out = dict(record)
        # Authoritative interaction_clearance/post_replacement are the bundled
        # inline fields (portable). Only fill from external enrichment when the
        # bundled record lacks them (e.g. the auxiliary-only branch above).
        if not out.get("interaction_clearance"):
            out["interaction_clearance"] = interaction_clearance
        else:
            out["interaction_clearance_external"] = interaction_clearance
        out.setdefault("post_replacement",
                       {"articulated": False, "realization_kind": "static_only"})
        out["ud4_affordance"] = ud4_affordance
        out["operation_space"] = operation_space
        out["clearance_regions"] = clearance_regions
        scenebenchmark = out.get("scenebenchmark_fd_sa")
        if not isinstance(scenebenchmark, dict):
            scenebenchmark = build_scenebenchmark_annotation(out)
            out["scenebenchmark_fd_sa"] = scenebenchmark
        hints = scenebenchmark.get("functional_hints")
        if isinstance(hints, dict):
            out.setdefault("scenebenchmark_functional_hints", hints)
        dependencies = scenebenchmark.get("functional_dependencies")
        if dependencies:
            out.setdefault("functional_dependencies", dependencies)
        support_regions = scenebenchmark.get("support_regions")
        if support_regions:
            out.setdefault("support_regions", support_regions)
        return out

    def require(self, hssd_id: str) -> dict[str, Any]:
        normalized = normalize_hssd_id(hssd_id)
        record = self.get(normalized)
        if record is None:
            raise KeyError(f"HSSD id not found in annotation lookup: {normalized}")
        return record

    def search_category(self, query: str, limit: int = 20) -> list[dict[str, Any]]:
        q = str(query or "").strip().lower().replace("_", " ")
        if not q:
            return []
        matches = []
        for record in self._load().values():
            haystack = " ".join(
                str(record.get(key) or "")
                for key in ("category", "category_key", "asset_uid")
            ).lower().replace("_", " ")
            if q in haystack:
                matches.append(record)
                if len(matches) >= limit:
                    break
        return matches


_DEFAULT_STORE: AssetLibraryAnnotationStore | None = None


def get_hssd_asset_annotations(hssd_id: str) -> dict[str, Any] | None:
    global _DEFAULT_STORE
    if _DEFAULT_STORE is None:
        _DEFAULT_STORE = AssetLibraryAnnotationStore()
    return _DEFAULT_STORE.get(hssd_id)
