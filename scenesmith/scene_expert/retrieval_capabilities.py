"""Read-only native category eligibility for optional SceneExpert planning hints."""

from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from scenesmith.scene_expert.schemas import StageBrief
from scenesmith.scenebenchmark_critic.object_taxonomy import (
    canonical_object_category,
    categories_are_equivalent,
)

logger = logging.getLogger(__name__)
_BUCKETS = ("large_objects", "wall_objects", "ceiling_objects", "small_objects")
_STAGES = {
    "furniture": ("furniture_agent", "FURNITURE"),
    "wall_mounted": ("wall_agent", "WALL_MOUNTED"),
    "ceiling_mounted": ("ceiling_agent", "CEILING_MOUNTED"),
    "manipuland": ("manipuland_agent", "MANIPULAND"),
}


@dataclass(frozen=True)
class NativeAssetCapabilities:
    """Known categorical eligibility, without asserting retrieval or mesh success."""

    bucket: str
    categories: dict[str, frozenset[str]]
    metadata_path: str
    metadata_sha256: str

    def classify(self, name: str) -> tuple[str, list[str], list[str]]:
        """Use the most specific known head; preserve unknown or ambiguous names."""
        category = canonical_object_category(name)
        if category not in self.categories and any(
            connector in category for connector in ("_with_", "_and_", "_for_")
        ):
            return "unknown", [], []
        matches = [
            key
            for key in self.categories
            if category == key or category.endswith("_" + key)
        ]
        if not matches:
            return "unknown", [], []
        specificity = max(len(key.split("_")) for key in matches)
        matches = sorted(key for key in matches if len(key.split("_")) == specificity)
        buckets = sorted(set().union(*(self.categories[key] for key in matches)))
        return (
            "eligible" if self.bucket in buckets else "known_unavailable",
            matches,
            buckets,
        )

    def prompt_context(self) -> str:
        """Bound the catalog excerpt while explicitly retaining unknown categories."""
        eligible = sorted(
            k for k, buckets in self.categories.items() if self.bucket in buckets
        )
        return (
            "Native HSSD categorical eligibility for this stage uses bucket "
            f"{self.bucket}. Optional recommendations known only in other native "
            "buckets will be omitted. Keep every required object and its declared "
            "stage; report a required capability gap rather than changing inventory. "
            "Unknown categories remain uncertain. Eligibility does not guarantee "
            "text retrieval, geometry, placement or critic success. "
            f"Eligible category excerpt ({min(80, len(eligible))}/{len(eligible)}): "
            + ", ".join(eligible[:80])
            + ". The excerpt is not exhaustive; an unlisted name is not forbidden."
        )

    def filter_optional(
        self, brief: StageBrief, required: list[str]
    ) -> tuple[StageBrief, dict[str, Any]]:
        """Omit only optional proposals with positive evidence of ineligibility."""
        kept, dropped, observations = [], [], []
        for item in brief.optional_asset_recommendations:
            status, matches, buckets = self.classify(item.name)
            category = canonical_object_category(item.name)
            protected = any(
                categories_are_equivalent(category, canonical_object_category(label))
                or category.endswith("_" + canonical_object_category(label))
                for label in required
                if canonical_object_category(label)
            )
            if status == "known_unavailable" and not protected:
                dropped.append(item.name)
            else:
                kept.append(item)
            observations.append(
                {
                    "name": item.name,
                    "status": status,
                    "matched_categories": matches,
                    "native_buckets": buckets,
                    "required_inventory_protected": protected,
                }
            )
        constraints = list(brief.constraints_for_designer)
        if dropped:
            constraints.append(
                "Native retrieval capability: omit these OPTIONAL recommendations "
                f"from this stage because their known native categories are not "
                f"eligible for {self.bucket}: {', '.join(dropped)}. Preserve every "
                "required object and its declared owner. Other optional design "
                "choices remain available; this is not a placement or quality verdict."
            )
        audit = {
            "enabled": True,
            "bucket": self.bucket,
            "metadata_path": self.metadata_path,
            "metadata_sha256": self.metadata_sha256,
            "optional_observations": observations,
            "omitted_optional_names": dropped,
            "required_inventory_unchanged": True,
            "unknown_categories_preserved": True,
        }
        return (
            brief.model_copy(
                update={
                    "optional_asset_recommendations": kept,
                    "constraints_for_designer": constraints,
                }
            ),
            audit,
        )


def native_capabilities_from_config(
    cfg: dict[str, Any], *, enabled: bool = False
) -> dict[str, NativeAssetCapabilities]:
    """Load actual configured HSSD metadata only for an explicitly enabled wrapper."""
    if not enabled:
        return {}
    result = {}
    for stage, (agent, object_type) in _STAGES.items():
        manager = (cfg.get(agent, {}) or {}).get("asset_manager", {}) or {}
        if manager.get("general_asset_source") != "hssd":
            continue
        hssd = manager.get("hssd", {}) or {}
        bucket = (hssd.get("object_type_mapping", {}) or {}).get(object_type)
        path = Path(str(hssd.get("preprocessed_path", ""))) / "object_categories.json"
        if bucket not in _BUCKETS:
            logger.warning(
                "Native capability unavailable for %s: unknown bucket", stage
            )
            continue
        try:
            raw = path.read_bytes()
            data = json.loads(raw)
            categories: dict[str, set[str]] = {}
            for name in _BUCKETS:
                entries = data[name]
                if not isinstance(entries, list) or any(
                    not isinstance(x, str) for x in entries
                ):
                    raise ValueError(f"Invalid native category bucket {name}")
                for entry in entries:
                    for synset in entry.split(","):
                        lemma = re.sub(r"\.[nvars]\.\d+$", "", synset.strip())
                        key = canonical_object_category(lemma)
                        if key:
                            categories.setdefault(key, set()).add(name)
        except (OSError, ValueError, TypeError, KeyError) as exc:
            logger.warning("Native capability unavailable for %s: %s", stage, exc)
            continue
        result[stage] = NativeAssetCapabilities(
            bucket=bucket,
            categories={k: frozenset(v) for k, v in categories.items()},
            metadata_path=str(path),
            metadata_sha256=hashlib.sha256(raw).hexdigest(),
        )
    return result
