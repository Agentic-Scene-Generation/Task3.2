"""Execution budgets and semantic reuse for expensive asset acquisition."""

from __future__ import annotations

import re

from dataclasses import dataclass, field
from typing import Any


_FAMILY_ALIASES: dict[str, tuple[str, ...]] = {
    # Match specific functional roles before broader furniture aliases. For
    # example, "bedside table lamp" must not collapse to "nightstand" merely
    # because it contains the phrase "bedside table".
    "table_lamp": (
        "bedside table lamp",
        "lamp for bedside table",
        "bedside lamp",
        "table lamp",
        "desk lamp",
        "reading lamp",
    ),
    "wall_sconce": ("wall sconce", "wall lamp", "wall light"),
    "blanket": ("throw blanket", "knit blanket", "blanket"),
    "book": ("hardcover book", "paperback book", "book"),
    "shelf": ("floating shelf", "wall shelf", "shelf"),
    "nightstand": ("nightstand", "bedside table", "bedside cabinet", "床头柜"),
    "wardrobe": ("wardrobe", "closet", "armoire", "衣柜"),
    "bookshelf": ("bookshelf", "bookcase", "书架"),
    "ceiling_light": (
        "ceiling light",
        "ceiling lamp",
        "pendant light",
        "chandelier",
        "吊灯",
        "吸顶灯",
    ),
    "wall_art": ("painting", "artwork", "canvas print", "poster", "wall art", "挂画"),
    "mirror": ("mirror", "镜子", "镜面"),
    "wall_clock": ("wall clock", "clock", "挂钟"),
    "blackboard": ("blackboard", "chalkboard", "whiteboard"),
    "bed": ("bed", "床"),
    "sofa": ("sofa", "couch", "沙发"),
    # Classroom roles must not share one cache entry: a cached student desk is
    # not a valid substitute for the teacher desk at the front of the room.
    "student_desk": ("student desk", "pupil desk"),
    "teacher_desk": (
        "teacher's desk",
        "teacher desk",
        "instructor desk",
        "lectern",
        "podium",
    ),
    "desk": ("desk", "writing table", "书桌"),
    "table": ("table", "桌子"),
    "chair": ("chair", "stool", "椅子", "凳子"),
    "cabinet": ("cabinet", "cupboard", "柜子"),
    "dresser": ("dresser", "chest of drawers", "斗柜"),
    "rug": (
        "rug",
        "carpet",
        "runner",
        "floor mat",
        "doormat",
        "yoga mat",
        "地毯",
    ),
    "plant": ("plant", "potted plant", "绿植", "植物"),
}

_STYLE_WORDS = {
    "a",
    "an",
    "the",
    "modern",
    "minimalist",
    "classic",
    "contemporary",
    "wooden",
    "wood",
    "metal",
    "metallic",
    "framed",
    "round",
    "circular",
    "rectangular",
    "square",
    "large",
    "small",
    "medium",
    "decorative",
    "stylish",
    "simple",
    "silver",
    "black",
    "white",
    "brown",
}

# ``ObjectType`` is a placement-stage ownership tag in SceneSmith, not a narrow
# furniture ontology.  Keep the user-facing scope text in one place so request
# analysis and visual admission cannot disagree about floor-supported decor.
_PLACEMENT_ROLE_CONTRACTS: dict[str, str] = {
    "furniture": (
        "floor_placed: standalone floor-supported objects that participate in the "
        "room layout, including conventional furniture, floor plants in pots, "
        "floor lamps, rugs, baskets, and other freestanding decor"
    ),
    "wall_mounted": (
        "wall_mounted: objects whose intended support is a wall, including art, "
        "mirrors, boards, shelves, sconces, and mounted planters"
    ),
    "ceiling_mounted": (
        "ceiling_mounted: objects whose intended support is the ceiling, including "
        "lights, fans, projectors, and hanging decor"
    ),
    "manipuland": (
        "surface_placed: hand-scale or tabletop objects placed on furniture or "
        "other support surfaces; this excludes full-size floor plants and rugs"
    ),
}

_FLOOR_LAYOUT_FAMILIES = frozenset({"plant", "rug", "floor_lamp"})

# Bump whenever semantic admission meaning changes.  Persistent HSSD decisions
# must never outlive the prompt/placement contract that produced them.
ASSET_SEMANTIC_CONTRACT_VERSION = "6.0"


def placement_role_contract(role: str) -> str:
    """Return the support/placement scope represented by a SceneSmith role."""

    normalized = str(role).strip().lower()
    return _PLACEMENT_ROLE_CONTRACTS.get(
        normalized,
        f"{normalized or 'unknown'}: use the requested object's declared support mode",
    )


def is_floor_layout_family(description: str, short_name: str = "") -> bool:
    """Return whether a family belongs to the floor-placement stage."""

    return semantic_asset_family(description, short_name) in _FLOOR_LAYOUT_FAMILIES


def semantic_asset_family(description: str, short_name: str = "") -> str:
    """Map stylistic variants to a stable object family for reuse and budgets."""
    text = " ".join(f"{short_name} {description}".lower().replace("_", " ").split())
    for family, aliases in _FAMILY_ALIASES.items():
        for alias in aliases:
            plural_suffix = (
                r"(?:s|es)?"
                if alias
                and alias[-1].isascii()
                and alias[-1].isalpha()
                and not alias.endswith("s")
                else ""
            )
            if re.search(
                rf"(?<![a-z0-9]){re.escape(alias)}{plural_suffix}(?![a-z0-9])",
                text,
            ):
                return family

    tokens = [
        token
        for token in re.findall(r"[a-z0-9]+|[\u3400-\u9fff]+", text)
        if token not in _STYLE_WORDS
    ]
    return "_".join(tokens[:4]) or "unknown"


def canonical_requirement_family(value: str) -> str:
    """Map a requirement phrase or scene instance ID to one stable family."""

    normalized = re.sub(
        r"(?:[\s_-]+(?:instance[\s_-]*)?\d+)+$",
        "",
        str(value).strip().lower(),
    )
    return semantic_asset_family(normalized)


@dataclass
class AssetGateFailure:
    index: int
    description: str
    reason: str


@dataclass
class AssetGatePlan:
    allowed_indices: list[int] = field(default_factory=list)
    cached_assets: list[Any] = field(default_factory=list)
    failures: list[AssetGateFailure] = field(default_factory=list)
    families_by_index: dict[int, str] = field(default_factory=dict)


class AssetRuntimeGate:
    """Per-stage circuit breaker that never drops required families silently."""

    def __init__(self) -> None:
        self.configure(stage="", budget={}, required_objects=[])

    @staticmethod
    def is_asset_admitted(asset: Any) -> bool:
        """Return whether an asset may participate in runtime reuse."""

        metadata = getattr(asset, "metadata", {}) or {}
        return not bool(
            metadata.get("repair_placeholder", False)
            or metadata.get("asset_admission_failed", False)
        )

    def configure(
        self,
        *,
        stage: str,
        budget: dict[str, Any],
        required_objects: list[str],
    ) -> None:
        # A stage regeneration rebuilds request counters but deliberately keeps
        # already admitted real assets.  Dropping this cache made the replacement
        # designer reacquire identical HSSD assets and consume the time intended
        # for placement and critique.
        previous_stage = getattr(self, "stage", "")
        previous_success_cache = getattr(self, "success_cache", {})
        self.stage = stage
        self.configured = bool(budget)
        self.max_asset_requests = max(0, int(budget.get("max_asset_requests", 0) or 0))
        self.max_optional_families = max(
            0, int(budget.get("max_optional_object_families", 0) or 0)
        )
        self.max_assets_per_request = max(
            0, int(budget.get("max_assets_per_request", 0) or 0)
        )
        self.max_retries_per_family = max(
            1, int(budget.get("max_semantic_retries_per_family", 2) or 2)
        )
        self.required_families = {
            semantic_asset_family(value) for value in required_objects if str(value).strip()
        }
        self.request_count = 0
        self.family_attempts: dict[str, int] = {}
        self.optional_families: set[str] = set()
        self.success_cache = (
            {
                family: list(assets)
                for family, assets in previous_success_cache.items()
                if any(self.is_asset_admitted(asset) for asset in assets)
            }
            if stage and stage == previous_stage
            else {}
        )
        for family, assets in list(self.success_cache.items()):
            self.success_cache[family] = [
                asset
                for asset in assets
                if self.is_asset_admitted(asset)
            ]
            if not self.success_cache[family]:
                del self.success_cache[family]

    @property
    def enabled(self) -> bool:
        return self.configured

    def plan(self, descriptions: list[str], short_names: list[str]) -> AssetGatePlan:
        plan = AssetGatePlan()
        candidates: list[tuple[int, str, bool]] = []
        seen_in_request: set[str] = set()

        for index, description in enumerate(descriptions):
            short_name = short_names[index] if index < len(short_names) else ""
            family = semantic_asset_family(description, short_name)
            plan.families_by_index[index] = family

            cached = self.success_cache.get(family)
            if cached:
                if family not in seen_in_request:
                    plan.cached_assets.extend(cached[:1])
                    seen_in_request.add(family)
                continue

            required = family in self.required_families
            attempts = self.family_attempts.get(family, 0)
            if attempts >= self.max_retries_per_family:
                requirement = "required" if required else "optional"
                plan.failures.append(
                    AssetGateFailure(
                        index=index,
                        description=description,
                        reason=(
                            f"Semantic {requirement} asset family '{family}' exhausted "
                            f"its {self.max_retries_per_family} acquisition attempt(s). "
                            "Reuse a cached/local asset or invoke deterministic repair; "
                            "do not retry stylistic paraphrases."
                        ),
                    )
                )
                continue

            if (
                self.max_asset_requests > 0
                and self.request_count >= self.max_asset_requests
                and not required
            ):
                plan.failures.append(
                    AssetGateFailure(
                        index=index,
                        description=description,
                        reason="Optional asset request budget exhausted for this stage.",
                    )
                )
                continue

            if (
                not required
                and family not in self.optional_families
                and self.max_optional_families > 0
                and len(self.optional_families) >= self.max_optional_families
            ):
                plan.failures.append(
                    AssetGateFailure(
                        index=index,
                        description=description,
                        reason=(
                            "Optional object-family budget exhausted; required objects "
                            "remain eligible."
                        ),
                    )
                )
                continue

            if family in seen_in_request:
                continue
            seen_in_request.add(family)
            candidates.append((index, family, required))

        candidates.sort(key=lambda item: (not item[2], item[0]))
        if self.max_assets_per_request > 0:
            allowed = candidates[: self.max_assets_per_request]
            for index, family, required in candidates[self.max_assets_per_request :]:
                plan.failures.append(
                    AssetGateFailure(
                        index=index,
                        description=descriptions[index],
                        reason=(
                            f"Per-request asset limit ({self.max_assets_per_request}) "
                            f"deferred family '{family}'"
                            + ("; required family must be requested next." if required else ".")
                        ),
                    )
                )
        else:
            allowed = candidates

        plan.allowed_indices = sorted(index for index, _, _ in allowed)
        if plan.allowed_indices:
            self.request_count += 1
        for _, family, required in allowed:
            self.family_attempts[family] = self.family_attempts.get(family, 0) + 1
            if not required:
                self.optional_families.add(family)
        return plan

    def remember_success(self, family: str, asset: Any) -> None:
        if not self.is_asset_admitted(asset):
            return
        cached = self.success_cache.setdefault(family, [])
        asset_id = str(getattr(asset, "object_id", ""))
        if all(str(getattr(existing, "object_id", "")) != asset_id for existing in cached):
            cached.append(asset)

    def invalidate_family(self, family: str) -> int:
        """Remove a semantically named family from the reusable success cache."""

        removed = len(self.success_cache.get(family, []) or [])
        self.success_cache.pop(family, None)
        return removed

    def clear_success_cache(self) -> None:
        self.success_cache.clear()
