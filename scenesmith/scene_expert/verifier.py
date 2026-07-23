"""Verifier: rule-based + score-based quality verification for SceneExpert.

Two layers:
- StageVerifier: quick post-stage check using SceneSmith's existing scores.yaml
  plus rule checks against the SceneTaskSpec.
- FullVerifier: aggregates all stage reports into a final whole-scene assessment.

MVP: primarily rule-based, no extra VLM calls. Reads scores.yaml produced
by SceneSmith's CritiqueWithScores system.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import yaml

from scenesmith.agent_utils.room_size_policy import normalize_room_dimensions
from scenesmith.scene_expert.schemas import (
    FullVerifyReport,
    SceneTaskSpec,
    StageBrief,
    StageVerifyReport,
    VerifyIssue,
)

console_logger = logging.getLogger(__name__)

# Maps SceneSmith score keys (from scores.yaml) to SceneExpert categories.
# Handles both actual Title Case keys and legacy snake_case variants.
# Matching is substring-based on the lowercased key (e.g. "realism" in "realism").
_SCENESMITH_SCORE_MAPPING = {
    # Actual keys written by SceneSmith critics (Title Case, lowercased for matching)
    "realism": "aesthetic",
    "functionality": "semantic",
    # Keep specific keys before generic "layout" because matching is substring-based.
    "layout plausibility": "plausibility",
    "layout_plausibility": "plausibility",
    "human likeness": "plausibility",
    "human-likeness": "plausibility",
    "professional arrangement": "plausibility",
    "layout": "aesthetic",
    "holistic completeness": "semantic",
    "prompt following": "semantic",
    "reachability": "interaction",
    # Floor plan specific
    "room proportions": "semantic",
    "spatial flow": "semantic",
    "natural lighting": "aesthetic",
    "material consistency": "aesthetic",
    # Legacy snake_case variants (kept for backwards compatibility)
    "object_placement_quality": "semantic",
    "functional_arrangement": "semantic",
    "visual_aesthetics": "aesthetic",
    "style_consistency": "aesthetic",
    "physics_validity": "physics",
    "collision_free": "physics",
    "walkability": "walkability",
    "support_relation": "interaction",
    "room_layout_quality": "semantic",
    "space_utilization": "semantic",
}


def _load_scores_yaml(scores_yaml_path: Path) -> tuple[dict[str, float], str]:
    """Load SceneSmith's scores.yaml.

    Returns:
        Tuple of (flat numeric scores dict, summary text string).
        Summary is the critic's full written evaluation — the richest signal.
    """
    if not scores_yaml_path.exists():
        console_logger.warning(f"scores.yaml not found at {scores_yaml_path}")
        return {}, ""
    with scores_yaml_path.open() as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        return {}, ""

    flat: dict[str, float] = {}
    summary = ""
    for k, v in data.items():
        if k.lower() == "summary":
            summary = str(v) if v else ""
        elif isinstance(v, (int, float)):
            flat[k] = float(v)
        elif isinstance(v, dict):
            # Nested dict: extract "grade" sub-key if present (SceneSmith format)
            grade = v.get("grade") or v.get("score")
            if grade is not None and isinstance(grade, (int, float)):
                flat[k] = float(grade)
            else:
                flat.update(
                    {
                        f"{k}.{sk}": float(sv)
                        for sk, sv in v.items()
                        if isinstance(sv, (int, float))
                    }
                )
    return flat, summary


def _load_score_provenance(
    scores_yaml_path: Path | None,
    critique_summary: str,
    raw_scores: dict[str, float],
) -> dict:
    """Load score provenance, with inference for artifacts from older runs."""
    if scores_yaml_path is not None:
        provenance_path = scores_yaml_path.parent / "score_provenance.yaml"
        if provenance_path.exists():
            try:
                with provenance_path.open() as f:
                    payload = yaml.safe_load(f)
                if isinstance(payload, dict):
                    return payload
            except Exception as exc:
                console_logger.warning(
                    "Failed to read score provenance at %s: %s",
                    provenance_path,
                    exc,
                )

    summary_upper = str(critique_summary or "").upper()
    if "DETERMINISTIC HARD-CHECK FAILED BEFORE VLM SCORING" in summary_upper:
        source = "deterministic_hard_check"
    elif any(
        marker in summary_upper
        for marker in (
            "CRITIC DEGRADED",
            "TRANSIENT LOCAL VLM TIMEOUT",
            "VISUAL CRITIC UNAVAILABLE",
        )
    ):
        source = "critic_fallback"
    elif raw_scores:
        source = "vlm_critic"
    else:
        source = "unavailable"
    return {
        "score_source": source,
        "vlm_scoring_performed": source == "vlm_critic",
        "hard_check_passed": False if source == "deterministic_hard_check" else None,
        "score_scale": (
            "0-10"
            if raw_scores and any(abs(value) > 1.0 for value in raw_scores.values())
            else "0-1"
        ),
        "inferred_for_legacy_artifact": True,
    }


def _map_trustworthy_scores(
    raw_scores: dict[str, float],
    critique_summary: str,
    score_source: str,
    score_scale: str = "",
) -> dict[str, float]:
    """Map only score dimensions supported by the recorded evidence source."""
    if score_source == "vlm_critic":
        return _map_scenesmith_scores(raw_scores, score_scale=score_scale)
    # Synthetic hard-check grades and evidence-derived failures belong only in
    # rule_scores/issues. They must never enter the visual score namespace.
    return {}


# Maps stage name → subdirectory under scene_states/ that holds the stage scores.yaml.
_STAGE_SCORES_SUBDIR = {
    "furniture": "furniture",
    "wall_mounted": "wall",
    "ceiling_mounted": "ceiling",
    "floor_plan": "floor_plan",
}


def _find_scores_yaml(stage_output_dir: str, stage: str = "") -> Path | None:
    """Find the definitive scores.yaml for a given stage.

    Score provenance is stage-local evidence. Never borrow the most recently
    modified score from another stage when the requested stage is unscored.
    """
    root = Path(stage_output_dir)
    if not root.exists():
        return None

    if stage == "floor_plan":
        for candidate in (
            root / "final_floor_plan" / "scores.yaml",
            root / "floor_plans" / "final_floor_plan" / "scores.yaml",
        ):
            if candidate.exists() or (
                candidate.parent / "score_provenance.yaml"
            ).exists():
                return candidate

    # Try stage-specific known path first.
    subdir = _STAGE_SCORES_SUBDIR.get(stage)
    if subdir:
        candidate = root / "scene_states" / subdir / "scores.yaml"
        if candidate.exists() or (candidate.parent / "score_provenance.yaml").exists():
            return candidate

    if stage == "manipuland":
        scene_states_dir = root / "scene_states"
        stage_dirs = (
            [
                path
                for path in scene_states_dir.glob("manipuland_*")
                if path.is_dir()
                and (
                    (path / "scores.yaml").exists()
                    or (path / "score_provenance.yaml").exists()
                )
            ]
            if scene_states_dir.exists()
            else []
        )
        if stage_dirs:
            latest = max(
                stage_dirs,
                key=lambda path: max(
                    (
                        candidate.stat().st_mtime
                        for candidate in (
                            path / "scores.yaml",
                            path / "score_provenance.yaml",
                        )
                        if candidate.exists()
                    ),
                    default=0.0,
                ),
            )
            return latest / "scores.yaml"

    return None


def _map_scenesmith_scores(
    raw_scores: dict[str, float],
    *,
    score_scale: str = "",
) -> dict[str, float]:
    """Map SceneSmith raw scores to SceneExpert categories (0-1 scale)."""
    mapped: dict[str, list[float]] = {}
    ten_point_scale = score_scale == "0-10" or (
        not score_scale and any(abs(value) > 1.0 for value in raw_scores.values())
    )
    for key, value in raw_scores.items():
        # Current SceneSmith CategoryScore values are 0-10. Use provenance for
        # the otherwise ambiguous 0/10 and 1/10 values; legacy normalized
        # artifacts remain supported when provenance says 0-1.
        normalized = value / 10.0 if ten_point_scale else value
        normalized = max(0.0, min(1.0, normalized))
        # Try exact match first, then partial match
        for sm_key, se_cat in _SCENESMITH_SCORE_MAPPING.items():
            if sm_key in key.lower():
                mapped.setdefault(se_cat, []).append(normalized)
                break
    # Average within each category
    return {cat: sum(vals) / len(vals) for cat, vals in mapped.items()}


def _score_value(raw_scores: dict[str, float], *name_parts: str) -> float | None:
    """Return a raw 0-10 score by fuzzy key parts."""
    for key, value in raw_scores.items():
        key_lower = key.lower().replace("_", " ")
        if all(part.lower().replace("_", " ") in key_lower for part in name_parts):
            return float(value)
    return None


def _critique_has_hard_collision(text: str) -> bool:
    """Detect explicit collision/penetration reports while ignoring negations."""
    lowered = text.lower()
    negated = (
        "no collision",
        "no collisions",
        "no overlaps detected",
        "all physics violations have been resolved",
    )
    if any(term in lowered for term in negated):
        return False
    hard_terms = (
        "collision detected",
        "collides with",
        "penetration",
        "physics collision",
        "physics hard violation: collisions",
        "physically impossible",
        "critical issue: physics collision",
    )
    return any(term in lowered for term in hard_terms)


def _critique_mentions_missing_required(
    text: str,
    required_objects: list[str],
) -> list[str]:
    """Extract missing required objects from critic prose."""
    lowered = text.lower()
    missing: list[str] = []
    for obj in required_objects:
        obj_lower = obj.lower()
        patterns = (
            rf"\b{re.escape(obj_lower)}\s+missing\b",
            rf"\bmissing\s+(?:required\s+|primary\s+)?{re.escape(obj_lower)}\b",
            rf"\bwithout\s+(?:the\s+)?{re.escape(obj_lower)}\b",
            rf"\b{re.escape(obj_lower)}\s+is\s+absent\b",
        )
        if any(re.search(pattern, lowered) for pattern in patterns):
            missing.append(obj)
    return missing


def _add_issue_once(issues: list[VerifyIssue], issue: VerifyIssue) -> None:
    signature = (issue.issue_type, issue.object_name, issue.description)
    for existing in issues:
        if (
            existing.issue_type,
            existing.object_name,
            existing.description,
        ) == signature:
            return
    issues.append(issue)


def _check_required_objects(
    task_spec: SceneTaskSpec, stage: str, scene_state_info: dict
) -> list[VerifyIssue]:
    """Check if required objects for this stage are present in the scene state.

    Args:
        task_spec: Compiled task specification.
        stage: Current stage name.
        scene_state_info: Lightweight scene info dict (object names, categories).

    Returns:
        List of issues for missing required objects.
    """
    issues: list[VerifyIssue] = []

    stage_required: list[str] = []
    if stage == "furniture":
        stage_required = task_spec.required_large_objects
    elif stage == "wall_mounted":
        stage_required = task_spec.required_wall_objects
    elif stage == "ceiling_mounted":
        stage_required = task_spec.required_ceiling_objects
    elif stage == "manipuland":
        stage_required = task_spec.required_small_objects

    if not stage_required:
        return issues

    present_objects = scene_state_info.get("object_names", [])
    present_lower = [o.lower() for o in present_objects]

    for required in stage_required:
        req_lower = required.lower()
        # Fuzzy: check if any present object contains the required name as substring
        if not any(req_lower in p or p in req_lower for p in present_lower):
            issues.append(
                VerifyIssue(
                    issue_type="missing_object",
                    object_name=required,
                    description=f"Required object '{required}' for stage '{stage}' not found in scene",
                )
            )

    return issues


def _check_stage_population(stage: str, scene_state_info: dict) -> list[VerifyIssue]:
    """Enforce SceneExpert's bounded, non-empty placement-stage contract."""
    object_type_by_stage = {
        "furniture": "furniture",
        "wall_mounted": "wall_mounted",
        "ceiling_mounted": "ceiling_mounted",
        "manipuland": "manipuland",
    }
    object_type = object_type_by_stage.get(stage)
    if object_type is None:
        return []

    counts = scene_state_info.get("object_counts", {}) or {}
    count = int(counts.get(object_type, 0) or 0)
    minimum = int(scene_state_info.get("stage_min_output_objects", 0) or 0)
    maximum = int(scene_state_info.get("stage_max_output_objects", 0) or 0)
    issues: list[VerifyIssue] = []
    if count < minimum:
        issues.append(
            VerifyIssue(
                issue_type="insufficient_stage_objects",
                description=(
                    f"Stage '{stage}' produced {count} {object_type} objects; "
                    f"the completion contract requires at least {minimum}"
                ),
            )
        )
    if maximum > 0 and count > maximum:
        issues.append(
            VerifyIssue(
                issue_type="excessive_stage_objects",
                description=(
                    f"Stage '{stage}' produced {count} {object_type} objects; "
                    f"the completion contract allows at most {maximum}"
                ),
            )
        )
    return issues


def _check_floor_plan_layout(scene_state_info: dict) -> list[VerifyIssue]:
    """Check structural validity and professional single-room proportions."""
    issues: list[VerifyIssue] = []
    if not scene_state_info.get("layout_exists", True):
        issues.append(
            VerifyIssue(
                issue_type="missing_floor_plan_layout",
                description="house_layout.json was not found or could not be parsed",
            )
        )
        return issues

    room_count = int(scene_state_info.get("room_count", 0) or 0)
    if room_count <= 0:
        issues.append(
            VerifyIssue(
                issue_type="empty_floor_plan",
                description="Floor plan contains no rooms",
            )
        )

    invalid_rooms: list[str] = []
    single_room = room_count == 1
    for room in scene_state_info.get("rooms", []):
        if not isinstance(room, dict):
            continue
        room_id = str(room.get("room_id") or room.get("id") or room.get("name") or "")
        width = _first_not_none(room, "length", "depth", "length_m", "depth_m")
        depth = _first_not_none(room, "width", "width_m")
        try:
            if width is not None and float(width) <= 0:
                invalid_rooms.append(room_id or "<unknown>")
            if depth is not None and float(depth) <= 0:
                invalid_rooms.append(room_id or "<unknown>")

            if single_room and width is not None and depth is not None:
                adjustment = normalize_room_dimensions(
                    room_type=str(room.get("type") or room.get("room_type") or "room"),
                    width=float(width),
                    depth=float(depth),
                    prompt=str(room.get("prompt") or ""),
                    mode="room",
                )
                if adjustment.changed:
                    issues.append(
                        VerifyIssue(
                            issue_type="implausible_room_scale",
                            object_name=room_id,
                            description=(
                                f"Room '{room_id or '<unknown>'}' is {float(width):g}m x "
                                f"{float(depth):g}m, outside its professional scale envelope; "
                                f"use approximately {adjustment.width:g}m x "
                                f"{adjustment.depth:g}m or explicitly request a larger room"
                            ),
                        )
                    )
        except (TypeError, ValueError):
            invalid_rooms.append(room_id or "<unknown>")

    if invalid_rooms:
        issues.append(
            VerifyIssue(
                issue_type="invalid_room_dimensions",
                description=(
                    "Rooms have non-positive or unparsable dimensions: "
                    + ", ".join(sorted(set(invalid_rooms)))
                ),
            )
        )
    return issues


def _first_not_none(mapping: dict, *keys: str) -> object | None:
    """Return the first present non-None value, preserving invalid zeroes."""

    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return None


class StageVerifier:
    """Verifies a single stage output against task spec and stage brief."""

    def __init__(self, pass_threshold: float = 0.6) -> None:
        self._pass_threshold = pass_threshold

    def verify(
        self,
        stage: str,
        stage_output_dir: str,
        task_spec: SceneTaskSpec,
        stage_brief: StageBrief | None = None,
        scene_state_info: dict | None = None,
    ) -> StageVerifyReport:
        """Run stage verification.

        Args:
            stage: Stage name (e.g., "furniture").
            stage_output_dir: Path to SceneSmith stage output directory.
            task_spec: Compiled task specification.
            stage_brief: StageBrief injected for this stage (for constraint checking).
            scene_state_info: Lightweight scene info for rule checks.
                Expected keys: "object_names" (list[str]).

        Returns:
            StageVerifyReport with pass/fail, scores, issues, and repair suggestions.
        """
        console_logger.info(f"StageVerifier: verifying stage '{stage}'")

        issues: list[VerifyIssue] = []
        repair_suggestions: list[str] = []

        # --- 1. Load SceneSmith scores ---
        scores_path = _find_scores_yaml(stage_output_dir, stage=stage)
        raw_scores, critique_summary = (
            _load_scores_yaml(scores_path)
            if scores_path is not None and scores_path.exists()
            else ({}, "")
        )
        provenance = _load_score_provenance(
            scores_path,
            critique_summary,
            raw_scores,
        )
        score_source = str(provenance.get("score_source", "unknown"))
        vlm_scoring_performed = bool(
            provenance.get("vlm_scoring_performed", score_source == "vlm_critic")
        )
        verification_evidence = "\n".join(
            part
            for part in (
                critique_summary,
                str(provenance.get("hard_check_evidence", "")),
            )
            if part
        )
        visual_scores = _map_trustworthy_scores(
            raw_scores,
            verification_evidence,
            score_source,
            score_scale=str(provenance.get("score_scale", "")),
        )
        rule_scores: dict[str, float] = {}
        hard_check_passed = provenance.get("hard_check_passed")
        if hard_check_passed is True:
            rule_scores["physics"] = 1.0
        elif hard_check_passed is False:
            rule_scores["physics"] = 0.0

        if not visual_scores:
            console_logger.warning(
                "No trustworthy numeric quality scores for stage %s "
                "(score_source=%s)",
                stage,
                score_source,
            )

        # --- 2. Rule-based checks ---
        if scene_state_info:
            if stage == "floor_plan":
                layout_issues = _check_floor_plan_layout(scene_state_info)
                issues.extend(layout_issues)
                if layout_issues:
                    repair_suggestions.append(
                        "Regenerate the floor plan with valid dimensions inside the "
                        "room-type professional size envelope"
                    )
            object_issues = _check_required_objects(task_spec, stage, scene_state_info)
            issues.extend(object_issues)
            if object_issues:
                for issue in object_issues:
                    repair_suggestions.append(
                        f"Add missing object '{issue.object_name}' to the scene"
                    )

            population_issues = _check_stage_population(stage, scene_state_info)
            issues.extend(population_issues)
            if population_issues:
                repair_suggestions.append(
                    "Regenerate this stage from its input checkpoint and satisfy the "
                    "bounded stage-native object count before continuing"
                )

            degraded_reasons = [
                str(reason)
                for reason in scene_state_info.get("degraded_stage_reasons", [])
                if str(reason).strip()
            ]
            if degraded_reasons:
                issues.append(
                    VerifyIssue(
                        issue_type="degraded_stage",
                        description=(
                            f"Stage '{stage}' exhausted runtime recovery: "
                            + "; ".join(degraded_reasons)
                        ),
                    )
                )

            if stage == "furniture":
                placeholder_names = sorted(
                    {
                        str(name)
                        for name in scene_state_info.get("placeholder_names", [])
                        if name
                    }
                )
                if placeholder_names:
                    issues.append(
                        VerifyIssue(
                            issue_type="placeholder_asset",
                            description=(
                                "Furniture asset generation degraded to primitive "
                                "placeholders: " + ", ".join(placeholder_names)
                            ),
                        )
                    )
                    repair_suggestions.append(
                        "Regenerate placeholder furniture with canonicalized HSSD "
                        "assets before accepting the stage"
                    )

        # --- 2b. Hard critique/score checks for stages where averages are unsafe ---
        # A low average can hide hard failures such as "missing bed" if other
        # dimensions score well. Treat these as blocking issues before pass/fail.
        if (
            score_source == "deterministic_hard_check"
            or provenance.get("hard_check_passed") is False
        ):
            _add_issue_once(
                issues,
                VerifyIssue(
                    issue_type="deterministic_hard_fail",
                    description=(
                        "Deterministic validation failed before VLM scoring; "
                        "synthetic repair grades were excluded from visual metrics"
                    ),
                ),
            )
        if score_source not in {"vlm_critic", "deterministic_hard_check"}:
            _add_issue_once(
                issues,
                VerifyIssue(
                    issue_type="critic_unavailable",
                    description=(
                        f"Stage '{stage}' has no trustworthy visual critic result "
                        f"(score_source={score_source})"
                    ),
                ),
            )
            repair_suggestions.append(
                "Retry the compact visual critic evaluation; do not use placeholder "
                "numeric grades for acceptance or memory"
            )
        elif score_source == "vlm_critic" and not visual_scores:
            _add_issue_once(
                issues,
                VerifyIssue(
                    issue_type="unusable_critic_scores",
                    description=(
                        f"Stage '{stage}' reports a VLM critic source but contains no "
                        "recognized numeric quality dimensions"
                    ),
                ),
            )
        if _critique_has_hard_collision(verification_evidence):
            _add_issue_once(
                issues,
                VerifyIssue(
                    issue_type="physics_collision",
                    description=(
                        f"{stage} deterministic evidence reports a hard collision "
                        "or wall penetration"
                    ),
                ),
            )
            repair_suggestions.append(
                f"Resolve reported {stage} collisions before accepting the stage"
            )

        if stage == "furniture":

            missing_from_critique = _critique_mentions_missing_required(
                verification_evidence,
                task_spec.required_large_objects,
            )
            for required in missing_from_critique:
                _add_issue_once(
                    issues,
                    VerifyIssue(
                        issue_type="missing_object",
                        object_name=required,
                        description=(
                            f"Critic reports required furniture '{required}' "
                            "is missing"
                        ),
                    ),
                )
                repair_suggestions.append(
                    f"Add missing required furniture '{required}' and rescore"
                )

            prompt_following = (
                _score_value(raw_scores, "prompt", "following")
                if score_source == "vlm_critic"
                else None
            )
            if prompt_following is not None and prompt_following < 8:
                _add_issue_once(
                    issues,
                    VerifyIssue(
                        issue_type="low_prompt_following",
                        description=(
                            f"Prompt Following score {prompt_following:g}/10 "
                            "is below the furniture hard minimum 8/10"
                        ),
                    ),
                )
                repair_suggestions.append(
                    "Do not accept furniture stage until prompt-required objects are present"
                )

            functionality = (
                _score_value(raw_scores, "functionality")
                if score_source == "vlm_critic"
                else None
            )
            if functionality is not None and functionality < 4:
                _add_issue_once(
                    issues,
                    VerifyIssue(
                        issue_type="low_functionality",
                        description=(
                            f"Functionality score {functionality:g}/10 indicates "
                            "a hard functional failure"
                        ),
                    ),
                )

        # --- 3. Stage brief constraint check (heuristic) ---
        # If issues exist and brief has failure patterns, add them as avoidance hints
        if issues and stage_brief and stage_brief.failure_patterns_to_avoid:
            for pattern in stage_brief.failure_patterns_to_avoid[:2]:
                repair_suggestions.append(f"Ensure you avoid: {pattern}")

        # --- 4. Compute pass/fail ---
        plausibility_score = visual_scores.get("plausibility")
        pass_plausibility = (
            plausibility_score is None or plausibility_score >= self._pass_threshold
        )
        visual_avg_score = (
            sum(visual_scores.values()) / len(visual_scores)
            if visual_scores
            else None
        )
        pass_score_gate = (
            visual_avg_score is None or visual_avg_score >= self._pass_threshold
        )
        pass_stage = pass_score_gate and pass_plausibility and len(issues) == 0
        if not pass_plausibility:
            repair_suggestions.append(
                "Improve layout plausibility: revise major furniture anchors and "
                "door/window/opening relationships so the room follows human-use "
                "and professional arrangement conventions"
            )

        console_logger.info(
            "StageVerifier stage=%s: visual_avg=%s "
            f"pass={pass_stage} issues={len(issues)} "
            f"plausibility={plausibility_score if plausibility_score is not None else 'n/a'} "
            f"source={score_source}",
            stage,
            (
                f"{visual_avg_score:.2f}"
                if visual_avg_score is not None
                else "n/a"
            ),
        )

        return StageVerifyReport(
            stage=stage,
            pass_stage=pass_stage,
            scores=visual_scores,
            visual_scores=visual_scores,
            rule_scores=rule_scores,
            issues=issues,
            repair_suggestions=repair_suggestions,
            critique_summary=critique_summary,
            score_source=score_source,
            vlm_scoring_performed=vlm_scoring_performed,
            hard_check_report=provenance,
            runtime_repair_events=(
                [
                    str(event)
                    for event in scene_state_info.get(
                        "runtime_repair_events",
                        [],
                    )
                    if str(event).strip()
                ]
                if scene_state_info
                else []
            ),
        )


class FullVerifier:
    """Aggregates stage verify reports into a final whole-scene assessment."""

    def __init__(self, pass_threshold: float = 0.7) -> None:
        self._pass_threshold = pass_threshold

    def verify(
        self,
        stage_reports: list[StageVerifyReport],
        final_scene_path: str = "",
        expected_stages: list[str] | None = None,
    ) -> FullVerifyReport:
        """Compute final scene quality metrics from stage reports.

        Args:
            stage_reports: All stage verifier outputs.
            final_scene_path: Path to final scene output (for future VLM extension).

        Returns:
            FullVerifyReport with aggregated scores.
        """
        expected = list(expected_stages or [])
        completed = [report.stage for report in stage_reports]
        unmatched_completed = list(completed)
        missing: list[str] = []
        for stage in expected:
            if stage in unmatched_completed:
                unmatched_completed.remove(stage)
            else:
                missing.append(stage)
        if not stage_reports:
            return FullVerifyReport(
                expected_stages=expected,
                completed_stages=completed,
                missing_stages=missing,
            )

        # Aggregate visual quality independently from deterministic rule results.
        # Physics rules are pass/fail evidence, not substitute VLM grades.
        visual_scores_by_category: dict[str, list[float]] = {}
        rule_scores_by_category: dict[str, list[float]] = {}
        for report in stage_reports:
            stage_visual_scores = report.visual_scores or report.scores
            for category, score in stage_visual_scores.items():
                visual_scores_by_category.setdefault(category, []).append(score)
            for category, score in report.rule_scores.items():
                rule_scores_by_category.setdefault(category, []).append(score)

        def visual_avg(key: str) -> float:
            vals = visual_scores_by_category.get(key, [])
            return sum(vals) / len(vals) if vals else 0.0

        def rule_avg(key: str) -> float | None:
            vals = rule_scores_by_category.get(key, [])
            return sum(vals) / len(vals) if vals else None

        semantic = visual_avg("semantic")
        aesthetic = visual_avg("aesthetic")
        plausibility = visual_avg("plausibility")
        visual_physics = visual_avg("physics")
        hard_physics = rule_avg("physics")
        interaction = visual_avg("interaction")
        walkability = visual_avg("walkability")

        # Overall remains a visual-quality metric. Hard physics gates acceptance
        # through each stage's pass flag and is reported separately below.
        visual_dimensions = [
            key
            for key in (
                "semantic",
                "aesthetic",
                "plausibility",
                "physics",
                "interaction",
                "walkability",
            )
            if key in visual_scores_by_category
        ]
        overall = (
            sum(visual_avg(key) for key in visual_dimensions)
            / len(visual_dimensions)
            if visual_dimensions
            else 0.0
        )

        has_plausibility = "plausibility" in visual_scores_by_category
        pass_plausibility = not has_plausibility or plausibility >= self._pass_threshold
        collision_free_rate = (
            hard_physics if hard_physics is not None else visual_physics
        )

        report = FullVerifyReport(
            semantic_score=semantic,
            aesthetic_score=aesthetic,
            plausibility_score=plausibility,
            style_consistency=aesthetic,  # proxy
            collision_free_rate=collision_free_rate,
            stability_score=visual_physics,  # VLM physics-quality proxy
            walkable_area_ratio=walkability if walkability > 0 else 0.0,
            reachability_score=interaction,
            support_relation_accuracy=interaction,  # proxy
            overall_score=overall,
            pass_scene=(
                bool(visual_scores_by_category)
                and not missing
                and all(stage.pass_stage for stage in stage_reports)
                and overall >= self._pass_threshold
                and pass_plausibility
            ),
            expected_stages=expected,
            completed_stages=completed,
            missing_stages=missing,
        )

        console_logger.info(
            "FullVerifier: "
            f"semantic={semantic:.2f} aesthetic={aesthetic:.2f} "
            f"plausibility_score={plausibility:.2f} "
            f"visual_physics={visual_physics:.2f} "
            f"hard_physics={hard_physics if hard_physics is not None else 'n/a'} "
            f"interaction={interaction:.2f} walkability={walkability:.2f} "
            f"overall={overall:.2f} pass={'YES' if report.pass_scene else 'NO'}"
        )
        return report
