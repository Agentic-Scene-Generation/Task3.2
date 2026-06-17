"""Code-level safety guardrails for furniture refinement loops.

The controller is intentionally small: it limits repeated tool use, tracks the
best hard-valid checkpoint, and decides whether a new critique candidate should
be accepted or rolled back. It does not encode room-specific layout rules.
"""

from __future__ import annotations

import copy
import logging
import re

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from scenesmith.agent_utils.scoring import CritiqueWithScores

console_logger = logging.getLogger(__name__)


DEFAULT_SCORE_WEIGHTS = {
    "prompt_following": 0.25,
    "functionality": 0.20,
    "layout_plausibility": 0.20,
    "layout": 0.15,
    "realism": 0.10,
    "reachability": 0.05,
    "holistic_completeness": 0.05,
}

DEFAULT_ALIASES = {
    "twin_bed": ["twin bed", "single bed"],
    "bed": ["bed", "beds"],
    "nightstand": ["nightstand", "nightstands", "bedside table", "bedside tables"],
    "wardrobe": ["wardrobe", "wardrobes", "closet", "closets"],
    "dresser": ["dresser", "dressers", "chest of drawers"],
    "desk": ["desk", "desks"],
    "chair": ["chair", "chairs"],
    "sofa": ["sofa", "sofas", "couch", "couches"],
    "table": ["table", "tables"],
    "cabinet": ["cabinet", "cabinets"],
    "bookshelf": ["bookshelf", "bookshelves", "bookcase", "bookcases"],
}

NUMBER_WORDS = {
    "a": 1,
    "an": 1,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
}


@dataclass
class SafetyEvaluation:
    """Hard-validity and weighted score for one critique candidate."""

    weighted_score: float
    hard_valid: bool
    hard_reasons: list[str] = field(default_factory=list)
    soft_reasons: list[str] = field(default_factory=list)


@dataclass
class HardStateEvaluation:
    """Deterministic hard-check result independent of critic scoring."""

    hard_valid: bool
    hard_reasons: list[str] = field(default_factory=list)
    soft_reasons: list[str] = field(default_factory=list)


@dataclass
class CandidateDecision:
    """Decision after evaluating a newly critiqued scene state."""

    accepted: bool
    rollback_to_best: bool
    should_finish: bool
    message: str
    evaluation: SafetyEvaluation


def _cfg_get(cfg: Any, key: str, default: Any) -> Any:
    if cfg is None:
        return default
    try:
        if hasattr(cfg, "get"):
            return cfg.get(key, default)
    except Exception:
        pass
    return getattr(cfg, key, default)


def _plain_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    try:
        return {str(k): v for k, v in value.items()}
    except Exception:
        return dict(value)


def _normalize_score_name(name: str) -> str:
    return name.lower().replace(" ", "_").replace("-", "_")


def _contains_alias(text: str, alias: str) -> bool:
    normalized = text.lower().replace("_", " ")
    escaped = re.escape(alias.lower())
    if " " in alias:
        return re.search(rf"(^|[^a-z0-9]){escaped}([^a-z0-9]|$)", normalized) is not None
    return re.search(rf"(^|[^a-z0-9]){escaped}([^a-z0-9]|$)", normalized) is not None


def _has_unnegated_collision(text: str) -> bool:
    terms = ("collision", "collisions", "colliding", "penetration")
    negations = (
        "no collision",
        "no collisions",
        "zero collision",
        "zero collisions",
        "without collision",
        "without collisions",
        "no new collision",
        "no new collisions",
    )
    for term in terms:
        for match in re.finditer(re.escape(term), text):
            window = text[max(0, match.start() - 24) : match.end() + 24]
            if any(negation in window for negation in negations):
                continue
            return True
    return False


class FurnitureSafetyController:
    """Safety controller for furniture placement refinement.

    The controller keeps hard constraints out of prompt-only behavior. A planner
    can still ask for extra changes, but the controller refuses expensive or
    destructive calls and restores the best hard-valid checkpoint at the end.
    """

    def __init__(self, cfg: Any | None = None):
        self.enabled = bool(_cfg_get(cfg, "enabled", False))
        self.max_critique_design_cycles = int(
            _cfg_get(cfg, "max_critique_design_cycles", 3)
        )
        self.max_move_calls = int(_cfg_get(cfg, "max_move_calls", 80))
        self.max_moves_initial_design = int(
            _cfg_get(cfg, "max_moves_initial_design", 30)
        )
        self.max_moves_design_change = int(
            _cfg_get(cfg, "max_moves_design_change", 12)
        )
        self.max_moves_per_object_per_call = int(
            _cfg_get(cfg, "max_moves_per_object_per_call", 4)
        )
        self.max_physics_checks_per_designer_call = int(
            _cfg_get(cfg, "max_physics_checks_per_designer_call", 4)
        )
        self.max_rescales_per_object = int(_cfg_get(cfg, "max_rescales_per_object", 1))
        self.max_generate_assets_calls_after_initial = int(
            _cfg_get(cfg, "max_generate_assets_calls_after_initial", 0)
        )
        self.min_accept_delta = float(_cfg_get(cfg, "min_accept_delta", 0.05))
        self.accept_score_threshold = float(
            _cfg_get(cfg, "accept_score_threshold", 0.78)
        )
        self.rescale_min_factor = float(_cfg_get(cfg, "rescale_min_factor", 0.8))
        self.rescale_max_factor = float(_cfg_get(cfg, "rescale_max_factor", 1.25))
        self.prompt_following_hard_min = int(
            _cfg_get(cfg, "prompt_following_hard_min", 8)
        )
        self.functionality_hard_min = int(_cfg_get(cfg, "functionality_hard_min", 4))
        self.reachability_hard_min = int(_cfg_get(cfg, "reachability_hard_min", 5))
        self.score_thresholds_are_hard = bool(
            _cfg_get(cfg, "score_thresholds_are_hard", False)
        )
        self.room_bounds_tolerance_m = float(
            _cfg_get(cfg, "room_bounds_tolerance_m", 0.02)
        )

        self.score_weights = {
            **DEFAULT_SCORE_WEIGHTS,
            **_plain_dict(_cfg_get(cfg, "score_weights", {})),
        }
        self.size_bounds = _plain_dict(_cfg_get(cfg, "size_bounds", {}))
        self.required_object_names = [
            str(x).lower()
            for x in list(_cfg_get(cfg, "required_object_names", []) or [])
        ]

        self.scene_description = ""
        self.required_terms: set[str] = set()
        self.required_counts: dict[str, int] = {}
        self.design_change_calls = 0
        self.move_calls = 0
        self.active_designer_call: str | None = None
        self.move_calls_this_call = 0
        self.physics_checks_this_call = 0
        self.moves_by_object_this_call: dict[str, int] = {}
        self.generate_asset_calls = 0
        self.rescale_counts: dict[str, int] = {}
        self.should_finish = False

        self.best_scene_state: dict[str, Any] | None = None
        self.best_scores: CritiqueWithScores | None = None
        self.best_render_dir: Path | None = None
        self.best_weighted_score = -1.0
        self.best_reasons: list[str] = []

    def reset_for_scene(self, scene_description: str) -> None:
        """Reset counters and infer required objects for a new scene."""
        self.scene_description = scene_description or ""
        self.required_terms = self._infer_required_terms(self.scene_description)
        self.required_counts = self._infer_required_counts(self.scene_description)
        if self.required_object_names:
            self.required_terms.update(self.required_object_names)
            for name in self.required_object_names:
                self.required_counts.setdefault(name, 1)
        self.design_change_calls = 0
        self.move_calls = 0
        self.active_designer_call = None
        self.move_calls_this_call = 0
        self.physics_checks_this_call = 0
        self.moves_by_object_this_call = {}
        self.generate_asset_calls = 0
        self.rescale_counts = {}
        self.should_finish = False
        self.best_scene_state = None
        self.best_scores = None
        self.best_render_dir = None
        self.best_weighted_score = -1.0
        self.best_reasons = []
        console_logger.info(
            "Furniture safety controller reset: required_terms=%s",
            sorted(self.required_terms),
        )

    def _infer_required_terms(self, prompt: str) -> set[str]:
        terms = set()
        text = prompt.lower()
        for canonical, aliases in DEFAULT_ALIASES.items():
            if any(_contains_alias(text, alias) for alias in aliases):
                terms.add(canonical)
        if "twin_bed" in terms:
            terms.discard("bed")
        return terms

    def _infer_required_counts(self, prompt: str) -> dict[str, int]:
        counts: dict[str, int] = {}
        text = prompt.lower().replace("_", " ")
        number_pattern = "|".join([r"\d+", *NUMBER_WORDS.keys()])
        for canonical, aliases in DEFAULT_ALIASES.items():
            best_count = 0
            for alias in aliases:
                escaped_alias = re.escape(alias.lower())
                pattern = (
                    rf"(^|[^a-z0-9])(?:(?P<count>{number_pattern})\s+)?"
                    rf"(?:\w+\s+){{0,2}}{escaped_alias}"
                    rf"([^a-z0-9]|$)"
                )
                for match in re.finditer(pattern, text):
                    count_text = match.groupdict().get("count")
                    count = 1
                    if count_text:
                        count = (
                            int(count_text)
                            if count_text.isdigit()
                            else NUMBER_WORDS.get(count_text, 1)
                        )
                    best_count = max(best_count, count)
            if best_count > 0:
                counts[canonical] = best_count
        if "twin_bed" in counts:
            counts.pop("bed", None)
        return counts

    def _infer_category(self, text: str) -> str | None:
        for canonical, aliases in DEFAULT_ALIASES.items():
            if any(_contains_alias(text, alias) for alias in [canonical, *aliases]):
                return canonical
        return None

    def is_required_object(self, object_id: str, object_text: str = "") -> bool:
        """Return whether the object appears to satisfy a prompt-required term."""
        if not self.required_terms:
            return False
        text = f"{object_id} {object_text}".lower().replace("_", " ")
        for term in self.required_terms:
            aliases = DEFAULT_ALIASES.get(term, [term])
            if any(_contains_alias(text, alias) for alias in [term, *aliases]):
                return True
        return False

    def record_design_change(self, has_prior_critique: bool) -> tuple[bool, str]:
        """Gate critique-design cycles."""
        if not self.enabled or not has_prior_critique:
            return True, ""
        if self.should_finish:
            return (
                False,
                "Safety controller: a hard-valid checkpoint is already accepted; "
                "do not request further design changes. Finish the furniture stage.",
            )
        if self.design_change_calls >= self.max_critique_design_cycles:
            self.should_finish = True
            return (
                False,
                "Safety controller: critique-design cycle budget exhausted "
                f"({self.max_critique_design_cycles}). Finish with the best "
                "hard-valid checkpoint instead of making more changes.",
            )
        self.design_change_calls += 1
        return True, ""

    def begin_designer_call(self, call_kind: str) -> None:
        """Start per-designer-call budgets."""
        if not self.enabled:
            return
        self.active_designer_call = call_kind
        self.move_calls_this_call = 0
        self.physics_checks_this_call = 0
        self.moves_by_object_this_call = {}

    def end_designer_call(self) -> None:
        """Clear per-designer-call budget state."""
        if not self.enabled:
            return
        self.active_designer_call = None
        self.move_calls_this_call = 0
        self.physics_checks_this_call = 0
        self.moves_by_object_this_call = {}

    def _current_move_limit(self) -> int:
        if self.active_designer_call == "initial":
            return self.max_moves_initial_design
        return self.max_moves_design_change

    def record_move(self, object_id: str) -> tuple[bool, str]:
        if not self.enabled:
            return True, ""
        if self.move_calls >= self.max_move_calls:
            return (
                False,
                "Safety controller blocked move_furniture_tool: move budget "
                f"exhausted ({self.max_move_calls}). Request a critique or finish.",
            )
        call_limit = self._current_move_limit()
        if self.active_designer_call and call_limit > 0:
            if self.move_calls_this_call >= call_limit:
                return (
                    False,
                    "Safety controller blocked move_furniture_tool: this designer "
                    f"call already used {call_limit} move(s). Request a critique "
                    "or finish instead of continuing local search.",
                )
        object_moves = self.moves_by_object_this_call.get(object_id, 0)
        if (
            self.active_designer_call
            and self.max_moves_per_object_per_call > 0
            and object_moves >= self.max_moves_per_object_per_call
        ):
            return (
                False,
                "Safety controller blocked move_furniture_tool: object "
                f"{object_id} has already been moved {object_moves} time(s) in "
                "this designer call. Stop iterating on the same object and "
                "request a critique.",
            )
        self.move_calls += 1
        self.move_calls_this_call += 1
        self.moves_by_object_this_call[object_id] = object_moves + 1
        return True, ""

    def record_physics_check(self) -> tuple[bool, str]:
        if not self.enabled:
            return True, ""
        if (
            self.active_designer_call
            and self.max_physics_checks_per_designer_call > 0
            and self.physics_checks_this_call
            >= self.max_physics_checks_per_designer_call
        ):
            return (
                False,
                "Safety controller blocked check_physics: this designer call "
                f"already used {self.max_physics_checks_per_designer_call} "
                "physics check(s). Use the latest result, request critique, or "
                "finish instead of continuing local search.",
            )
        self.physics_checks_this_call += 1
        return True, ""

    def record_generate_assets(self) -> tuple[bool, str]:
        if not self.enabled:
            return True, ""
        max_calls = 1 + self.max_generate_assets_calls_after_initial
        if self.generate_asset_calls >= max_calls:
            return (
                False,
                "Safety controller blocked generate_assets: asset generation has "
                "already run for this furniture stage. Reuse available assets, "
                "repair placement, or finish with the best checkpoint.",
            )
        self.generate_asset_calls += 1
        return True, ""

    def record_remove(self, object_id: str, object_text: str = "") -> tuple[bool, str]:
        if not self.enabled:
            return True, ""
        if self.is_required_object(object_id, object_text):
            return (
                False,
                "Safety controller blocked remove_furniture_tool: this object appears "
                "to satisfy a prompt-required furniture item. Move it locally instead "
                "of deleting it.",
            )
        return True, ""

    def record_rescale(
        self,
        object_id: str,
        scale_factor: float,
        object_text: str = "",
        current_dimensions: tuple[float, float, float] | None = None,
    ) -> tuple[bool, str]:
        if not self.enabled:
            return True, ""
        if not (self.rescale_min_factor <= scale_factor <= self.rescale_max_factor):
            return (
                False,
                "Safety controller blocked rescale_furniture_tool: scale_factor "
                f"{scale_factor:.3f} is outside "
                f"[{self.rescale_min_factor}, {self.rescale_max_factor}].",
            )
        count = self.rescale_counts.get(object_id, 0)
        if count >= self.max_rescales_per_object:
            return (
                False,
                "Safety controller blocked rescale_furniture_tool: each object may "
                f"be rescaled at most {self.max_rescales_per_object} time(s).",
            )
        bounds_message = self._check_size_bounds(
            object_id=object_id,
            object_text=object_text,
            scale_factor=scale_factor,
            current_dimensions=current_dimensions,
        )
        if bounds_message:
            return False, bounds_message
        self.rescale_counts[object_id] = count + 1
        return True, ""

    def _check_size_bounds(
        self,
        object_id: str,
        object_text: str,
        scale_factor: float,
        current_dimensions: tuple[float, float, float] | None,
    ) -> str:
        if current_dimensions is None:
            return ""
        category = self._infer_category(f"{object_id} {object_text}")
        if category is None:
            return ""
        bounds = _plain_dict(self.size_bounds.get(category))
        if not bounds:
            return ""
        min_dims = list(bounds.get("min", []))
        max_dims = list(bounds.get("max", []))
        if len(min_dims) != 3 or len(max_dims) != 3:
            return ""
        new_dims = [float(dim) * scale_factor for dim in current_dimensions]
        for idx, dim in enumerate(new_dims):
            if dim < float(min_dims[idx]) or dim > float(max_dims[idx]):
                return (
                    "Safety controller blocked rescale_furniture_tool: resulting "
                    f"{category} dimensions {new_dims} would exceed configured "
                    f"category bounds min={min_dims}, max={max_dims}."
                )
        return ""

    def evaluate_scores(self, scores: CritiqueWithScores) -> SafetyEvaluation:
        """Compute weighted score and classify hard vs soft issues."""
        score_by_name = {
            _normalize_score_name(score.name): score for score in scores.get_scores()
        }
        weighted_sum = 0.0
        weight_sum = 0.0
        for name, weight in self.score_weights.items():
            score = score_by_name.get(_normalize_score_name(name))
            if score is None:
                continue
            weighted_sum += float(weight) * (float(score.grade) / 10.0)
            weight_sum += float(weight)
        if weight_sum == 0:
            all_scores = scores.get_scores()
            weighted_score = (
                sum(float(score.grade) for score in all_scores) / (10.0 * len(all_scores))
                if all_scores
                else 0.0
            )
        else:
            weighted_score = weighted_sum / weight_sum

        critique_text = scores.critique.lower()
        comments_text = " ".join(score.comment.lower() for score in scores.get_scores())
        text = f"{critique_text} {comments_text}"

        hard_reasons: list[str] = []
        soft_reasons: list[str] = []

        prompt_following = score_by_name.get("prompt_following")
        if prompt_following and prompt_following.grade < self.prompt_following_hard_min:
            hard_reasons.append(
                f"prompt_following={prompt_following.grade} below "
                f"{self.prompt_following_hard_min}"
            )

        if self.score_thresholds_are_hard:
            functionality = score_by_name.get("functionality")
            if functionality and functionality.grade < self.functionality_hard_min:
                hard_reasons.append(
                    f"functionality={functionality.grade} below "
                    f"{self.functionality_hard_min}"
                )

            reachability = score_by_name.get("reachability")
            if reachability and reachability.grade < self.reachability_hard_min:
                hard_reasons.append(
                    f"reachability={reachability.grade} below "
                    f"{self.reachability_hard_min}"
                )

        if _has_unnegated_collision(text):
            hard_reasons.append("physics collision indicated by critique")

        door_terms = (
            "door blocked",
            "blocked door",
            "doorway blocked",
            "blocks the door",
            "blocks door",
            "door clearance",
            "open connection blocked",
        )
        if any(term in text for term in door_terms):
            hard_reasons.append("door or open-connection blockage indicated by critique")

        if "window" in text and not hard_reasons:
            soft_reasons.append(
                "window-related concern is treated as soft/medium unless paired with "
                "collision, missing required objects, or door blockage"
            )

        return SafetyEvaluation(
            weighted_score=weighted_score,
            hard_valid=not hard_reasons,
            hard_reasons=hard_reasons,
            soft_reasons=soft_reasons,
        )

    def evaluate_scene_state(
        self, scene: Any, physics_context: str | None = None
    ) -> HardStateEvaluation:
        """Run deterministic hard checks that do not depend on critic judgment."""
        hard_reasons: list[str] = []
        soft_reasons: list[str] = []

        required_counts = self.required_counts or {
            term: 1 for term in self.required_terms
        }
        if required_counts:
            observed_counts = {term: 0 for term in required_counts}
            for object_id, obj in getattr(scene, "objects", {}).items():
                if getattr(obj, "immutable", False):
                    continue
                category = self._infer_category(
                    f"{object_id} {getattr(obj, 'name', '')} "
                    f"{getattr(obj, 'description', '')}"
                )
                if category in observed_counts:
                    observed_counts[category] += 1
            for term, required_count in required_counts.items():
                if observed_counts.get(term, 0) < required_count:
                    hard_reasons.append(
                        f"missing required {term}: expected {required_count}, "
                        f"found {observed_counts.get(term, 0)}"
                    )

        room_bounds = self._room_bounds_xy(scene)
        if room_bounds is not None:
            min_x, min_y, max_x, max_y = room_bounds
            tol = self.room_bounds_tolerance_m
            for object_id, obj in getattr(scene, "objects", {}).items():
                if getattr(obj, "immutable", False):
                    continue
                if not self._is_furniture_object(obj):
                    continue
                world_bounds = obj.compute_world_bounds()
                if world_bounds is None:
                    continue
                world_min, world_max = world_bounds
                if (
                    world_min[0] < min_x - tol
                    or world_max[0] > max_x + tol
                    or world_min[1] < min_y - tol
                    or world_max[1] > max_y + tol
                ):
                    hard_reasons.append(
                        f"{object_id} full bounding box exceeds room bounds: "
                        f"x=[{world_min[0]:.3f}, {world_max[0]:.3f}] vs "
                        f"[{min_x:.3f}, {max_x:.3f}], "
                        f"y=[{world_min[1]:.3f}, {world_max[1]:.3f}] vs "
                        f"[{min_y:.3f}, {max_y:.3f}]"
                    )

        if physics_context:
            hard_from_physics, soft_from_physics = self._parse_physics_context(
                physics_context
            )
            hard_reasons.extend(hard_from_physics)
            soft_reasons.extend(soft_from_physics)

        return HardStateEvaluation(
            hard_valid=not hard_reasons,
            hard_reasons=hard_reasons,
            soft_reasons=soft_reasons,
        )

    def _room_bounds_xy(self, scene: Any) -> tuple[float, float, float, float] | None:
        room_geometry = getattr(scene, "room_geometry", None)
        if room_geometry is None:
            return None
        length = float(getattr(room_geometry, "length", 0.0) or 0.0)
        width = float(getattr(room_geometry, "width", 0.0) or 0.0)
        if length <= 0 or width <= 0:
            return None
        return (-length / 2.0, -width / 2.0, length / 2.0, width / 2.0)

    def _is_furniture_object(self, obj: Any) -> bool:
        object_type = getattr(obj, "object_type", "")
        value = getattr(object_type, "value", object_type)
        return str(value).lower() == "furniture"

    def _parse_physics_context(self, physics_context: str) -> tuple[list[str], list[str]]:
        text = physics_context.lower()
        if "no physics violations detected" in text:
            return [], []

        hard_reasons: list[str] = []
        soft_reasons: list[str] = []

        hard_sections = (
            "collisions (",
            "thin covering overlaps",
            "thin covering boundary violations",
            "door clearance violations",
            "open connection blocked",
            "wall height exceeded",
        )
        for section in hard_sections:
            if section in text:
                hard_reasons.append(f"physics hard violation: {section.rstrip(' (')}")

        if "window access warnings" in text:
            soft_reasons.append("window access warning")

        if not hard_reasons and "physics violations detected" in text:
            soft_reasons.append("non-hard physics warning")

        return hard_reasons, soft_reasons

    def remember_hard_valid_scene_state(
        self,
        scene_state: dict[str, Any],
        source: str,
        weighted_score: float | None = None,
        scores: CritiqueWithScores | None = None,
        render_dir: Path | None = None,
    ) -> bool:
        """Save a deterministic hard-valid checkpoint when useful."""
        if not self.enabled:
            return False
        candidate_score = 0.0 if weighted_score is None else weighted_score
        if self.best_scene_state is not None and weighted_score is None:
            return False
        if (
            self.best_scene_state is not None
            and weighted_score is not None
            and candidate_score < self.best_weighted_score + self.min_accept_delta
        ):
            return False

        self.best_scene_state = copy.deepcopy(scene_state)
        self.best_scores = copy.deepcopy(scores) if scores is not None else None
        self.best_render_dir = render_dir
        self.best_weighted_score = max(candidate_score, self.best_weighted_score)
        self.best_reasons = [f"deterministic hard-valid checkpoint from {source}"]
        console_logger.info(
            "Safety controller saved deterministic hard-valid checkpoint from %s",
            source,
        )
        return True

    def consider_candidate(
        self,
        scores: CritiqueWithScores,
        scene_state: dict[str, Any],
        render_dir: Path | None,
        hard_state_evaluation: HardStateEvaluation | None = None,
    ) -> CandidateDecision:
        """Evaluate a critiqued candidate and update/rollback best state metadata."""
        evaluation = self.evaluate_scores(scores)
        if hard_state_evaluation is not None:
            evaluation.hard_reasons.extend(hard_state_evaluation.hard_reasons)
            evaluation.soft_reasons.extend(hard_state_evaluation.soft_reasons)
            evaluation.hard_valid = (
                evaluation.hard_valid and hard_state_evaluation.hard_valid
            )
        accepted = False
        rollback_to_best = False

        if evaluation.hard_valid:
            improvement = evaluation.weighted_score - self.best_weighted_score
            if self.best_scene_state is None or improvement >= self.min_accept_delta:
                self.best_scene_state = copy.deepcopy(scene_state)
                self.best_scores = copy.deepcopy(scores)
                self.best_render_dir = render_dir
                self.best_weighted_score = evaluation.weighted_score
                self.best_reasons = []
                accepted = True
                message = (
                    "Safety controller accepted new best hard-valid checkpoint "
                    f"(weighted_score={evaluation.weighted_score:.3f})."
                )
            else:
                rollback_to_best = self.best_scene_state is not None
                message = (
                    "Safety controller rejected candidate: hard constraints pass, "
                    f"but weighted_score={evaluation.weighted_score:.3f} does not "
                    f"improve best={self.best_weighted_score:.3f} by "
                    f"{self.min_accept_delta:.3f}."
                )
        else:
            rollback_to_best = self.best_scene_state is not None
            message = (
                "Safety controller rejected candidate: hard constraints failed "
                f"({'; '.join(evaluation.hard_reasons)})."
            )

        if rollback_to_best:
            self.should_finish = True
        if (
            evaluation.hard_valid
            and evaluation.weighted_score >= self.accept_score_threshold
        ):
            self.should_finish = True
        if self.design_change_calls >= self.max_critique_design_cycles:
            self.should_finish = True

        if self.should_finish:
            message += " Stage should finish now."

        decision = CandidateDecision(
            accepted=accepted,
            rollback_to_best=rollback_to_best,
            should_finish=self.should_finish,
            message=message,
            evaluation=evaluation,
        )
        console_logger.info(decision.message)
        return decision
