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
from pathlib import Path

import yaml

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
                    {f"{k}.{sk}": float(sv) for sk, sv in v.items()
                     if isinstance(sv, (int, float))}
                )
    return flat, summary


# Maps stage name → subdirectory under scene_states/ that holds the stage scores.yaml.
_STAGE_SCORES_SUBDIR = {
    "furniture": "furniture",
    "wall_mounted": "wall",
    "ceiling_mounted": "ceiling",
    "floor_plan": "floor_plan",
}


def _find_scores_yaml(stage_output_dir: str, stage: str = "") -> Path | None:
    """Find the definitive scores.yaml for a given stage.

    Prefers the stage-specific path (scene_states/<subdir>/scores.yaml) to avoid
    picking up per-render or per-iteration scores files.  Falls back to most-recently-
    modified scores.yaml anywhere under the directory only when the expected path
    is absent.

    For the manipuland stage, aggregates scores from all
    scene_states/manipuland_*/scores.yaml files and writes a temporary combined file.
    """
    root = Path(stage_output_dir)
    if not root.exists():
        return None

    # Try stage-specific known path first.
    subdir = _STAGE_SCORES_SUBDIR.get(stage)
    if subdir:
        candidate = root / "scene_states" / subdir / "scores.yaml"
        if candidate.exists():
            return candidate

    if stage == "manipuland":
        # Collect all per-object manipuland scores files.
        candidates = sorted(
            (root / "scene_states").glob("manipuland_*/scores.yaml")
        ) if (root / "scene_states").exists() else []
        if candidates:
            # Return the most recent per-object scores file (last manipuland placed).
            return max(candidates, key=lambda p: p.stat().st_mtime)

    # Generic fallback: most recent scores.yaml under scene_states/ only
    # (exclude scene_renders/ which has per-iteration files).
    scene_states_dir = root / "scene_states"
    if scene_states_dir.exists():
        all_candidates = list(scene_states_dir.rglob("scores.yaml"))
        if all_candidates:
            return max(all_candidates, key=lambda p: p.stat().st_mtime)

    # Last resort: anywhere under root.
    all_root_candidates = list(root.rglob("scores.yaml"))
    if all_root_candidates:
        return max(all_root_candidates, key=lambda p: p.stat().st_mtime)

    return None


def _map_scenesmith_scores(raw_scores: dict[str, float]) -> dict[str, float]:
    """Map SceneSmith raw scores to SceneExpert categories (0-1 scale)."""
    mapped: dict[str, list[float]] = {}
    for key, value in raw_scores.items():
        # SceneSmith uses 0-10 scale; normalize to 0-1
        normalized = value / 10.0 if value > 1.0 else value
        # Try exact match first, then partial match
        for sm_key, se_cat in _SCENESMITH_SCORE_MAPPING.items():
            if sm_key in key.lower():
                mapped.setdefault(se_cat, []).append(normalized)
                break
    # Average within each category
    return {cat: sum(vals) / len(vals) for cat, vals in mapped.items()}


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
            _load_scores_yaml(scores_path) if scores_path else ({}, "")
        )
        mapped_scores = _map_scenesmith_scores(raw_scores)

        # If no scores available, use conservative defaults
        if not mapped_scores:
            console_logger.warning(f"No scores.yaml found for stage {stage}, using defaults")
            mapped_scores = {"semantic": 0.5, "aesthetic": 0.5, "physics": 0.5, "interaction": 0.5}

        # --- 2. Rule-based checks ---
        if scene_state_info:
            object_issues = _check_required_objects(task_spec, stage, scene_state_info)
            issues.extend(object_issues)
            if object_issues:
                for issue in object_issues:
                    repair_suggestions.append(
                        f"Add missing object '{issue.object_name}' to the scene"
                    )

        # --- 3. Stage brief constraint check (heuristic) ---
        # If issues exist and brief has failure patterns, add them as avoidance hints
        if issues and stage_brief and stage_brief.failure_patterns_to_avoid:
            for pattern in stage_brief.failure_patterns_to_avoid[:2]:
                repair_suggestions.append(f"Ensure you avoid: {pattern}")

        # --- 4. Compute pass/fail ---
        avg_score = sum(mapped_scores.values()) / max(len(mapped_scores), 1)
        pass_stage = avg_score >= self._pass_threshold and len(issues) == 0

        console_logger.info(
            f"StageVerifier stage={stage}: avg_score={avg_score:.2f} "
            f"pass={pass_stage} issues={len(issues)}"
        )

        return StageVerifyReport(
            stage=stage,
            pass_stage=pass_stage,
            scores=mapped_scores,
            issues=issues,
            repair_suggestions=repair_suggestions,
            critique_summary=critique_summary,
        )


class FullVerifier:
    """Aggregates stage verify reports into a final whole-scene assessment."""

    def __init__(self, pass_threshold: float = 0.7) -> None:
        self._pass_threshold = pass_threshold

    def verify(
        self,
        stage_reports: list[StageVerifyReport],
        final_scene_path: str = "",
    ) -> FullVerifyReport:
        """Compute final scene quality metrics from stage reports.

        Args:
            stage_reports: All stage verifier outputs.
            final_scene_path: Path to final scene output (for future VLM extension).

        Returns:
            FullVerifyReport with aggregated scores.
        """
        if not stage_reports:
            return FullVerifyReport()

        # Aggregate scores across stages
        all_scores: dict[str, list[float]] = {}
        for report in stage_reports:
            for category, score in report.scores.items():
                all_scores.setdefault(category, []).append(score)

        def avg(key: str) -> float:
            vals = all_scores.get(key, [])
            return sum(vals) / len(vals) if vals else 0.0

        semantic = avg("semantic")
        aesthetic = avg("aesthetic")
        physics = avg("physics")
        interaction = avg("interaction")
        walkability = avg("walkability")

        # Derived overall score
        overall = (semantic + aesthetic + physics + interaction + walkability) / max(
            sum(1 for k in ["semantic", "aesthetic", "physics", "interaction", "walkability"] if k in all_scores),
            1,
        )

        report = FullVerifyReport(
            semantic_score=semantic,
            aesthetic_score=aesthetic,
            style_consistency=aesthetic,  # proxy
            collision_free_rate=physics,
            stability_score=physics,      # proxy
            walkable_area_ratio=walkability if walkability > 0 else 0.0,
            reachability_score=interaction,
            support_relation_accuracy=interaction,  # proxy
            overall_score=overall,
            pass_scene=overall >= self._pass_threshold,
        )

        console_logger.info(
            f"FullVerifier: overall={overall:.2f} pass={'YES' if report.pass_scene else 'NO'}"
        )
        return report
