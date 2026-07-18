"""Execution budgets and semantic reuse for expensive asset acquisition."""

from __future__ import annotations

import re

from dataclasses import dataclass, field
from typing import Any


_FAMILY_ALIASES: dict[str, tuple[str, ...]] = {
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
    "bed": ("bed", "床"),
    "sofa": ("sofa", "couch", "沙发"),
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


def semantic_asset_family(description: str, short_name: str = "") -> str:
    """Map stylistic variants to a stable object family for reuse and budgets."""
    text = " ".join(f"{short_name} {description}".lower().replace("_", " ").split())
    for family, aliases in _FAMILY_ALIASES.items():
        for alias in aliases:
            if re.search(rf"(?<![a-z0-9]){re.escape(alias)}(?![a-z0-9])", text):
                return family

    tokens = [
        token
        for token in re.findall(r"[a-z0-9]+|[\u3400-\u9fff]+", text)
        if token not in _STYLE_WORDS
    ]
    return "_".join(tokens[:4]) or "unknown"


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

    def configure(
        self,
        *,
        stage: str,
        budget: dict[str, Any],
        required_objects: list[str],
    ) -> None:
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
        self.success_cache: dict[str, list[Any]] = {}

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
        cached = self.success_cache.setdefault(family, [])
        asset_id = str(getattr(asset, "object_id", ""))
        if all(str(getattr(existing, "object_id", "")) != asset_id for existing in cached):
            cached.append(asset)

    def clear_success_cache(self) -> None:
        self.success_cache.clear()
