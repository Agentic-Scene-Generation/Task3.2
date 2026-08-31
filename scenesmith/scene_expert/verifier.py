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

from scenesmith.scenebenchmark_critic.object_taxonomy import (
    canonical_object_category,
    categories_are_equivalent,
    semantic_category_match,
)
from scenesmith.scenebenchmark_critic.relation_registry import (
    STAGE_ORDER as CONTRACT_STAGE_ORDER,
    relation_spec,
)
from scenesmith.scenebenchmark_critic.stage_ownership import (
    normalize_result_stage_ownership,
)
from scenesmith.scenebenchmark_critic.metrics.functional_dependency.extensions.intent_contract import (
    SUPPORT_READINESS_FAILURE_CODE,
)
from scenesmith.scenebenchmark_critic.metrics.functional_dependency.extensions.room_containment import (
    ROOM_CONTAINMENT_FAILURE_CODE,
)
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
_DETERMINISTIC_HARD_CHECK_MARKER = "DETERMINISTIC HARD-CHECK FAILED BEFORE VLM SCORING"
_INCOMPLETE_CRITIQUE_PATTERNS = (
    re.compile(
        r"^(?:i'll|i will|let me)\s+(?:first\s+)?(?:validate|check|inspect|review)\b",
        re.I,
    ),
    re.compile(r"\bbefore drawing conclusions\b", re.I),
)


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


def _critique_is_final(summary: str) -> bool:
    """Reject obvious tool-use preambles accidentally persisted as conclusions."""

    normalized = " ".join(str(summary or "").split())
    if not normalized:
        return False
    if any(pattern.search(normalized) for pattern in _INCOMPLETE_CRITIQUE_PATTERNS):
        return False
    return True


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

    if stage == "floor_plan":
        for candidate in (
            root / "final_floor_plan" / "scores.yaml",
            root / "floor_plans" / "final_floor_plan" / "scores.yaml",
        ):
            if candidate.exists():
                return candidate

    # Try stage-specific known path first.
    subdir = _STAGE_SCORES_SUBDIR.get(stage)
    if subdir:
        candidate = root / "scene_states" / subdir / "scores.yaml"
        if candidate.exists():
            return candidate

    if stage == "manipuland":
        # Collect all per-object manipuland scores files.
        candidates = (
            sorted((root / "scene_states").glob("manipuland_*/scores.yaml"))
            if (root / "scene_states").exists()
            else []
        )
        if candidates:
            # Return the most recent per-object scores file (last manipuland placed).
            return max(candidates, key=lambda p: p.stat().st_mtime)

    # With an explicit stage, never borrow another stage's score. A missing
    # stage-specific score is accurate "no visual evidence", while a recent
    # cross-stage score silently creates a false Slow Memory label.
    if stage:
        return None

    # Generic fallback is retained only for legacy callers without a stage.
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


def _score_value(raw_scores: dict[str, float], *name_parts: str) -> float | None:
    """Return a raw 0-10 score by fuzzy key parts."""
    for key, value in raw_scores.items():
        key_lower = key.lower().replace("_", " ")
        if all(part.lower().replace("_", " ") in key_lower for part in name_parts):
            return float(value)
    return None


def _critique_has_hard_collision(text: str) -> bool:
    """Detect a positive collision report in visual-critic prose.

    This helper is used only by the explicit visual-score ablation gate.  In
    particular, it must not turn ``No physics collisions detected`` into a
    collision merely because that sentence contains the phrase ``physics
    collision``.
    """
    sentences = re.split(r"(?<=[.!?])|\n+", text.lower())
    hard_terms = (
        "collision detected",
        "collides with",
        "penetration",
        "physics collision",
        "physically impossible",
        "critical issue: physics collision",
    )
    for sentence in sentences:
        if (
            re.search(
                r"\b(?:no|without|zero)\s+(?:physics\s+)?collisions?\b",
                sentence,
            )
            or "no overlaps detected" in sentence
            or "all physics violations have been resolved" in sentence
        ):
            continue
        if any(term in sentence for term in hard_terms):
            return True
    return False


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


def _result_intent_constraint(result: dict) -> dict:
    evidence = result.get("evidence")
    if not isinstance(evidence, dict):
        return {}
    constraint = evidence.get("intent_constraint")
    return constraint if isinstance(constraint, dict) else {}


def _result_is_due(result: dict, stage: str) -> bool:
    """Whether a deterministic result belongs to the current repair owner."""
    if stage == "final":
        return True
    if stage not in CONTRACT_STAGE_ORDER:
        return False
    result = normalize_result_stage_ownership(result)
    constraint = _result_intent_constraint(result)
    diagnostics = result.get("diagnostics")
    diagnostics = diagnostics if isinstance(diagnostics, dict) else {}
    earliest = str(diagnostics.get("earliest_stage") or constraint.get("stage") or "")
    if not earliest or earliest not in CONTRACT_STAGE_ORDER:
        return False
    return earliest == stage


def _deterministic_core_failures(payload: dict | None, stage: str) -> list[dict]:
    failures: list[dict] = []
    for result in (payload or {}).get("results") or []:
        if not isinstance(result, dict):
            continue
        tier = str(result.get("scoring_tier") or "core").lower()
        label = str(result.get("label") or "").lower()
        metric = str(result.get("metric") or "")
        constraint = _result_intent_constraint(result)
        if constraint and str(constraint.get("strength") or "hard").lower() != "hard":
            continue
        hard_label = label == "fail" or (
            label == "degraded" and metric == "visual_clearance"
        )
        if tier != "core" or not hard_label or not _result_is_due(result, stage):
            continue
        failures.append(result)
    return failures


def non_degradable_blocker_codes(
    issues: list[VerifyIssue],
) -> tuple[str, ...]:
    """Return typed deterministic failures that degraded mode cannot accept."""
    blockers: set[str] = set()
    for issue in issues:
        if (
            issue.scoring_tier != "core"
            or issue.issue_type != "deterministic_relation_failure"
        ):
            continue
        if issue.relation == ROOM_CONTAINMENT_FAILURE_CODE:
            blockers.add(ROOM_CONTAINMENT_FAILURE_CODE)
        if issue.diagnostics.get("support_readiness") is True:
            blockers.add(SUPPORT_READINESS_FAILURE_CODE)
    return tuple(sorted(blockers))


def _non_degradable_blockers(
    stage_reports: list[StageVerifyReport], payload: dict | None
) -> list[str]:
    """Retain typed blockers across resumed-stage finalization."""
    blockers: set[str] = set()
    final_issues = [
        _issue_from_deterministic_result(result)
        for result in _deterministic_core_failures(payload, "final")
    ]
    blockers.update(non_degradable_blocker_codes(final_issues))
    for report in stage_reports:
        blockers.update(non_degradable_blocker_codes(report.issues))
    return sorted(blockers)


def _deterministic_hard_check_report(
    payload: dict | None,
    stage: str,
) -> dict[str, object]:
    """Summarize only actually evaluated, due hard constraints."""

    evaluated: list[dict] = []
    relation_results: list[dict] = []
    for result in (payload or {}).get("results") or []:
        if not isinstance(result, dict) or not _result_is_due(result, stage):
            continue
        tier = str(result.get("scoring_tier") or "core").lower()
        constraint = _result_intent_constraint(result)
        if tier != "core" or (
            constraint and str(constraint.get("strength") or "hard").lower() != "hard"
        ):
            continue
        label = str(result.get("label") or "").lower()
        if label not in {"pass", "fail", "degraded"}:
            continue
        evaluated.append(result)
        if constraint or result.get("relation_type"):
            relation_results.append(result)

    failed = [
        result
        for result in evaluated
        if str(result.get("label") or "").lower() == "fail"
        or (
            str(result.get("label") or "").lower() == "degraded"
            and str(result.get("metric") or "") == "visual_clearance"
        )
    ]
    decided_relations = [
        result
        for result in relation_results
        if str(result.get("label") or "").lower() in {"pass", "fail"}
    ]
    passed_relations = [
        result
        for result in decided_relations
        if str(result.get("label") or "").lower() == "pass"
    ]
    relation_satisfaction = (
        len(passed_relations) / len(decided_relations) if decided_relations else None
    )
    return {
        "evaluation_available": bool(evaluated),
        "hard_passed": not failed if evaluated else None,
        "evaluated_check_ids": [
            str(result.get("check_id") or "") for result in evaluated
        ],
        "failed_check_ids": [str(result.get("check_id") or "") for result in failed],
        "evaluated_relation_check_ids": [
            str(result.get("check_id") or "") for result in decided_relations
        ],
        "failed_relation_check_ids": [
            str(result.get("check_id") or "")
            for result in decided_relations
            if str(result.get("label") or "").lower() == "fail"
        ],
        "relation_satisfaction": relation_satisfaction,
        "constraint_satisfaction_rate": relation_satisfaction,
    }


def _issue_from_deterministic_result(result: dict) -> VerifyIssue:
    constraint = _result_intent_constraint(result)
    relation = str(constraint.get("relation") or result.get("relation_type") or "")
    diagnostics = result.get("diagnostics")
    diagnostics = diagnostics if isinstance(diagnostics, dict) else {}
    binding_issue = bool(diagnostics.get("binding_issue")) or str(
        result.get("check_id") or ""
    ).endswith("__binding")
    metric = str(result.get("metric") or "")
    if binding_issue:
        issue_type = "contract_binding_failure"
    elif metric == "visual_clearance":
        issue_type = "visual_clearance_failure"
    elif metric == "functional_dependency":
        issue_type = "deterministic_relation_failure"
    elif metric == "interaction_clearance":
        issue_type = "interaction_clearance_failure"
    else:
        issue_type = "deterministic_core_failure"
    repair_strategy = ""
    if relation:
        try:
            repair_strategy = relation_spec(relation).repair_strategy or ""
        except (KeyError, ValueError):
            repair_strategy = ""
    primary = str(result.get("primary_object") or result.get("subject_id") or "")
    related = [
        str(item)
        for item in (result.get("related_objects") or result.get("target_ids") or [])
    ]
    return VerifyIssue(
        issue_type=issue_type,
        object_name=primary,
        description=str(
            result.get("reason")
            or f"Deterministic core check {result.get('check_id')} failed"
        ),
        constraint_id=str(constraint.get("constraint_id") or ""),
        relation=relation,
        subject_ids=[primary] if primary else [],
        target_ids=related,
        metric=metric,
        scoring_tier=str(result.get("scoring_tier") or "core"),
        repair_strategy=repair_strategy,
        diagnostics=dict(diagnostics),
    )


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

    stage_required = _required_objects_for_stage(task_spec, stage)

    if not stage_required:
        return issues

    present_objects = scene_state_info.get("object_names", [])
    present_labels = [str(name) for name in present_objects]
    present_records = scene_state_info.get("object_records", [])
    if isinstance(present_records, list) and present_records:
        present_entries = [
            (
                [
                    (
                        str(record.get("name"))
                        if isinstance(record, dict) and record.get("name")
                        else ""
                    ),
                    *(
                        str(alias)
                        for alias in (record.get("aliases") or [])
                        if isinstance(record, dict)
                    ),
                ],
                (
                    str(record.get("description") or "")
                    if isinstance(record, dict)
                    else ""
                ),
            )
            for record in present_records
        ]
    else:
        descriptions = scene_state_info.get("object_descriptions", [])
        present_entries = [
            ([present], str(descriptions[index]) if index < len(descriptions) else "")
            for index, present in enumerate(present_labels)
        ]
    consumed_by_required_label: dict[str, set[int]] = {}

    for required in stage_required:
        required_label = _normalize_object_label(required)
        # A table/place setting describes a collection of manipulands rather
        # than a mesh that an asset manager can create.  Its component counts
        # and seating relation are checked by the intent contract; requiring a
        # literal object here would reject every valid decomposed setting.
        if stage == "manipuland" and required_label in _VIRTUAL_MANIPULAND_GROUPS:
            continue
        consumed_indices = consumed_by_required_label.setdefault(required_label, set())
        match_index = next(
            (
                index
                for index, (present_aliases, description) in enumerate(present_entries)
                if index not in consumed_indices
                and (
                    any(
                        _object_labels_match(required, present)
                        for present in present_aliases
                    )
                    or (
                        description
                        and _description_contains_object_label(required, description)
                    )
                )
            ),
            None,
        )
        if match_index is not None:
            # Repeated requirements of the same semantic label consume distinct
            # instances. A compound asset may still satisfy distinct component
            # labels (for example one ``vase_flowers`` satisfies both ``vase``
            # and ``flowers``).
            consumed_indices.add(match_index)
        else:
            issues.append(
                VerifyIssue(
                    issue_type="missing_object",
                    object_name=required,
                    description=f"Required object '{required}' for stage '{stage}' not found in scene",
                )
            )

    return issues


def _required_objects_for_stage(task_spec: SceneTaskSpec, stage: str) -> list[str]:
    """Return prompt-explicit assets owned by one generated asset stage."""
    return list(
        {
            "furniture": task_spec.required_large_objects,
            "wall_mounted": task_spec.required_wall_objects,
            "ceiling_mounted": task_spec.required_ceiling_objects,
            "manipuland": task_spec.required_small_objects,
        }.get(stage, [])
    )


def _required_asset_evidence(
    task_spec: SceneTaskSpec,
    stage: str,
    object_issues: list[VerifyIssue],
) -> dict[str, object]:
    """Build requirement coverage without conflating it with scene quality."""
    required = [
        item
        for item in _required_objects_for_stage(task_spec, stage)
        if not (
            stage == "manipuland"
            and _normalize_object_label(item) in _VIRTUAL_MANIPULAND_GROUPS
        )
    ]
    if not required:
        return {
            "required_objects": [],
            "required_satisfied_objects": [],
            "required_missing_objects": [],
            "required_coverage": None,
            "requirement_status": "not_applicable",
        }

    missing = [
        issue.object_name
        for issue in object_issues
        if issue.issue_type == "missing_object" and issue.object_name
    ]
    remaining_missing = list(missing)
    satisfied: list[str] = []
    for required_object in required:
        match_index = next(
            (
                index
                for index, missing_object in enumerate(remaining_missing)
                if _normalize_object_label(required_object)
                == _normalize_object_label(missing_object)
            ),
            None,
        )
        if match_index is None:
            satisfied.append(required_object)
        else:
            remaining_missing.pop(match_index)
    coverage = len(satisfied) / len(required)
    if not missing:
        status = "satisfied"
    elif satisfied:
        status = "partial"
    else:
        status = "unsatisfied"
    return {
        "required_objects": required,
        "required_satisfied_objects": satisfied,
        "required_missing_objects": missing,
        "required_coverage": coverage,
        "requirement_status": status,
    }


def evaluate_required_asset_coverage(
    task_spec: SceneTaskSpec,
    stage: str,
    scene_state_info: dict,
) -> dict[str, object]:
    """Public read-only requirement evidence for traces and evaluation."""
    object_issues = _check_required_objects(task_spec, stage, scene_state_info)
    return _required_asset_evidence(task_spec, stage, object_issues)


# These are prompt-level aggregate concepts, not independently instantiated
# scene assets. Their concrete components remain required and are consumed
# above, while SceneBenchmark validates their cardinality and relationship.
_VIRTUAL_MANIPULAND_GROUPS = {"table setting", "place setting"}


def _normalize_object_label(label: str) -> str:
    """Normalize human labels and scene identifiers to a comparable phrase."""
    label_text = str(label).lower()
    # Keep possessive labels such as "teacher's desk" comparable to generated
    # identifiers such as "teacher_desk_0".
    label_text = re.sub(r"\b([a-z0-9]+)'s\b", r"\1", label_text)
    words = re.sub(r"[^a-z0-9]+", " ", label_text).split()
    while words and words[-1].isdigit():
        words.pop()
    if not words:
        return ""
    return canonical_object_category(" ".join(words)).replace("_", " ")


def _object_labels_match(required: str, present: str) -> bool:
    required_label = _normalize_object_label(required)
    present_label = _normalize_object_label(present)
    if not required_label or not present_label:
        return False
    required_components = _object_label_component_categories(required)
    present_components = _object_label_component_categories(present)
    if any(
        categories_are_equivalent(required_category, present_category)
        or semantic_category_match(required_category, present_category)["matched"]
        for required_category in required_components
        for present_category in present_components
    ):
        return True
    return (
        f" {required_label} " in f" {present_label} "
        or f" {present_label} " in f" {required_label} "
    )


def _object_label_component_categories(label: str) -> set[str]:
    """Return both a label's primary category and compound asset components."""
    words = re.sub(r"[^a-z0-9]+", " ", str(label).lower()).split()
    while words and words[-1].isdigit():
        words.pop()
    categories = {canonical_object_category(" ".join(words))} if words else set()
    categories.update(canonical_object_category(word) for word in words)
    return {category for category in categories if category}


def _description_contains_object_label(required: str, description: str) -> bool:
    """Match a required object category mentioned in an asset description.

    Asset names can encode implementation details (for example
    ``chalkboard_land``) while their descriptions retain the user-facing
    category. Token-wise canonicalization lets one asset satisfy a matching
    stage requirement while still consuming a single scene object instance.
    """
    required_tokens = _normalize_object_label(required).split()
    if not required_tokens:
        return False
    description_tokens = [
        _normalize_object_label(token)
        for token in re.sub(r"[^a-z0-9]+", " ", str(description).lower()).split()
    ]
    description_tokens = [token for token in description_tokens if token]
    width = len(required_tokens)
    return any(
        description_tokens[index : index + width] == required_tokens
        for index in range(len(description_tokens) - width + 1)
    )


def _check_floor_plan_layout(scene_state_info: dict) -> list[VerifyIssue]:
    """Check minimal structural validity of the generated floor plan."""
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
    for room in scene_state_info.get("rooms", []):
        if not isinstance(room, dict):
            continue
        room_id = str(room.get("room_id") or room.get("id") or room.get("name") or "")
        width = room.get("width") or room.get("width_m")
        depth = room.get("depth") or room.get("depth_m")
        try:
            if width is not None and float(width) <= 0:
                invalid_rooms.append(room_id or "<unknown>")
            if depth is not None and float(depth) <= 0:
                invalid_rooms.append(room_id or "<unknown>")
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


class StageVerifier:
    """Verifies a stage using deterministic facts plus optional visual scoring.

    Existing ``scores.yaml`` files are generated by an LLM/VLM critique.  They
    remain useful for ranking and audit, but are not reliable enough to create
    a hard failure from a yaw convention or a visual misunderstanding.  The
    default therefore gates only on deterministic inventory/structural checks.
    ``visual_score_hard_gate`` exists for controlled offline ablations only.
    """

    def __init__(
        self,
        pass_threshold: float = 0.6,
        visual_score_hard_gate: bool = False,
        critic_bridge_enabled: bool = True,
    ) -> None:
        self._pass_threshold = pass_threshold
        self._visual_score_hard_gate = bool(visual_score_hard_gate)
        self._critic_bridge_enabled = bool(critic_bridge_enabled)

    def verify(
        self,
        stage: str,
        stage_output_dir: str,
        task_spec: SceneTaskSpec,
        stage_brief: StageBrief | None = None,
        scene_state_info: dict | None = None,
        deterministic_critic_payload: dict | None = None,
    ) -> StageVerifyReport:
        """Run stage verification.

        Args:
            stage: Stage name (e.g., "furniture").
            stage_output_dir: Path to SceneSmith stage output directory.
            task_spec: Compiled task specification.
            stage_brief: StageBrief injected for this stage (for constraint checking).
            scene_state_info: Lightweight scene info for rule checks.
                Expected keys: "object_names" (list[str]).
            deterministic_critic_payload: Fresh core critic output for hard,
                geometry-derived stage checks.

        Returns:
            StageVerifyReport with pass/fail, scores, issues, and repair suggestions.
        """
        console_logger.info(f"StageVerifier: verifying stage '{stage}'")

        issues: list[VerifyIssue] = []
        repair_suggestions: list[str] = []
        requirement_evidence: dict[str, object] = {
            "required_objects": [],
            "required_satisfied_objects": [],
            "required_missing_objects": [],
            "required_coverage": None,
            "requirement_status": "unknown",
        }

        # --- 1. Load SceneSmith scores ---
        scores_path = (
            _find_scores_yaml(stage_output_dir, stage=stage)
            if self._critic_bridge_enabled
            else None
        )
        raw_scores, critique_summary = (
            _load_scores_yaml(scores_path) if scores_path else ({}, "")
        )
        if raw_scores and not _critique_is_final(critique_summary):
            console_logger.warning(
                "StageVerifier: ignoring non-final critic payload for stage %s at %s",
                stage,
                scores_path,
            )
            raw_scores, critique_summary = {}, ""
        mapped_scores = _map_scenesmith_scores(raw_scores)
        bridged_scores = dict(mapped_scores)

        # If no scores available, use conservative defaults
        if not mapped_scores:
            reason = (
                "no scores.yaml was found"
                if self._critic_bridge_enabled
                else "the critic bridge is disabled"
            )
            console_logger.info(
                "StageVerifier: %s for stage %s; using neutral deterministic "
                "verification scores",
                reason,
                stage,
            )
            mapped_scores = {
                "semantic": 0.5,
                "aesthetic": 0.5,
                "plausibility": 0.5,
                "physics": 0.5,
                "interaction": 0.5,
            }

        if _DETERMINISTIC_HARD_CHECK_MARKER in critique_summary.upper():
            _add_issue_once(
                issues,
                VerifyIssue(
                    issue_type="deterministic_hard_failure",
                    description=(
                        "SceneSmith deterministic hard-check failed before visual "
                        "scoring completed"
                    ),
                ),
            )
            repair_suggestions.append(
                "Resolve deterministic hard-check failures before accepting the stage"
            )

        # --- 2. Rule-based checks ---
        if scene_state_info:
            if stage == "floor_plan":
                layout_issues = _check_floor_plan_layout(scene_state_info)
                issues.extend(layout_issues)
                if layout_issues:
                    repair_suggestions.append(
                        "Regenerate the floor plan with at least one valid room and positive dimensions"
                    )
            object_issues = _check_required_objects(task_spec, stage, scene_state_info)
            requirement_evidence = _required_asset_evidence(
                task_spec,
                stage,
                object_issues,
            )
            issues.extend(object_issues)
            if object_issues:
                for issue in object_issues:
                    repair_suggestions.append(
                        f"Add missing object '{issue.object_name}' to the scene"
                    )
        deterministic_failures = _deterministic_core_failures(
            deterministic_critic_payload, stage
        )
        hard_check_report = _deterministic_hard_check_report(
            deterministic_critic_payload, stage
        )
        for failure in deterministic_failures:
            issue = _issue_from_deterministic_result(failure)
            _add_issue_once(issues, issue)
            repair_suggestions.append(
                "Resolve deterministic core check "
                f"{failure.get('check_id') or issue.issue_type} and rerun the fresh critic"
            )

        # --- 2b. Optional visual-score ablation gate ---
        # Inventory is already checked from scene state above and physical
        # feasibility/geometry critic own collisions.  Do not let VLM prose or
        # scores become a second source of hard facts in normal generation.
        if stage == "furniture" and self._visual_score_hard_gate:
            if _critique_has_hard_collision(critique_summary):
                _add_issue_once(
                    issues,
                    VerifyIssue(
                        issue_type="physics_collision",
                        description=(
                            "Furniture critique reports a hard collision or "
                            "wall penetration"
                        ),
                    ),
                )
                repair_suggestions.append(
                    "Resolve reported furniture collisions before accepting the stage"
                )

            missing_from_critique = _critique_mentions_missing_required(
                critique_summary,
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

            prompt_following = _score_value(raw_scores, "prompt", "following")
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

            functionality = _score_value(raw_scores, "functionality")
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
        avg_score = sum(mapped_scores.values()) / max(len(mapped_scores), 1)
        plausibility_score = mapped_scores.get("plausibility")
        pass_plausibility = (
            plausibility_score is None or plausibility_score >= self._pass_threshold
        )
        visual_scores_pass = avg_score >= self._pass_threshold and pass_plausibility
        pass_stage = len(issues) == 0 and (
            visual_scores_pass if self._visual_score_hard_gate else True
        )
        if self._visual_score_hard_gate and not pass_plausibility:
            repair_suggestions.append(
                "Improve layout plausibility: revise major furniture anchors and "
                "door/window/opening relationships so the room follows human-use "
                "and professional arrangement conventions"
            )

        console_logger.info(
            f"StageVerifier stage={stage}: avg_score={avg_score:.2f} "
            f"pass={pass_stage} issues={len(issues)} "
            f"plausibility={plausibility_score if plausibility_score is not None else 'n/a'} "
            f"visual_gate={self._visual_score_hard_gate}"
        )

        return StageVerifyReport(
            stage=stage,
            pass_stage=pass_stage,
            scores=mapped_scores,
            visual_scores=bridged_scores,
            rule_scores={"deterministic_issue_free": 1.0 if not issues else 0.0},
            issues=issues,
            repair_suggestions=repair_suggestions,
            critique_summary=critique_summary,
            score_source=(
                "scenebenchmark_critic"
                if self._critic_bridge_enabled and bool(bridged_scores)
                else "neutral_default"
            ),
            vlm_scoring_performed=(
                self._critic_bridge_enabled and bool(bridged_scores)
            ),
            hard_check_report=hard_check_report,
            **requirement_evidence,
        )


class FullVerifier:
    """Aggregate deterministic stage validity and optional visual scores."""

    def __init__(
        self,
        pass_threshold: float = 0.7,
        visual_score_hard_gate: bool = False,
    ) -> None:
        self._pass_threshold = pass_threshold
        self._visual_score_hard_gate = bool(visual_score_hard_gate)

    def verify(
        self,
        stage_reports: list[StageVerifyReport],
        final_scene_path: str = "",
        deterministic_critic_payload: dict | None = None,
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

        # Aggregate authoritative visual scores across stages. ``scores`` remains
        # a backward-compatible field for older callers, but the explicit
        # neutral default produced while the critic bridge is disabled must not
        # be treated as critic evidence or flow into memory-quality signals.
        all_scores: dict[str, list[float]] = {}
        for report in stage_reports:
            report_scores = report.visual_scores
            if not report_scores and report.score_source == "unknown":
                report_scores = report.scores
            for category, score in report_scores.items():
                all_scores.setdefault(category, []).append(score)

        def avg(key: str) -> float:
            vals = all_scores.get(key, [])
            return sum(vals) / len(vals) if vals else 0.0

        semantic = avg("semantic")
        aesthetic = avg("aesthetic")
        plausibility = avg("plausibility")
        physics = avg("physics")
        interaction = avg("interaction")
        walkability = avg("walkability")

        # Derived overall score
        overall = (
            semantic + aesthetic + plausibility + physics + interaction + walkability
        ) / max(
            sum(
                1
                for k in [
                    "semantic",
                    "aesthetic",
                    "plausibility",
                    "physics",
                    "interaction",
                    "walkability",
                ]
                if k in all_scores
            ),
            1,
        )

        has_plausibility = "plausibility" in all_scores
        pass_plausibility = not has_plausibility or plausibility >= self._pass_threshold

        final_failures = _deterministic_core_failures(
            deterministic_critic_payload, "final"
        )
        non_degradable_blockers = _non_degradable_blockers(
            stage_reports, deterministic_critic_payload
        )
        deterministic_pass = (
            self._deterministic_stage_passes(stage_reports) and not final_failures
        )
        visual_scores_pass = overall >= self._pass_threshold and pass_plausibility
        requirement_status = self._aggregate_requirement_status(stage_reports)
        quality_status = (
            ("passed" if visual_scores_pass else "degraded")
            if all_scores
            else "unknown"
        )
        report = FullVerifyReport(
            semantic_score=semantic,
            aesthetic_score=aesthetic,
            plausibility_score=plausibility,
            style_consistency=aesthetic,  # proxy
            collision_free_rate=physics,
            stability_score=physics,  # proxy
            walkable_area_ratio=walkability if walkability > 0 else 0.0,
            reachability_score=interaction,
            support_relation_accuracy=interaction,  # proxy
            overall_score=overall,
            deterministic_pass=deterministic_pass,
            non_degradable_blockers=non_degradable_blockers,
            pass_scene=(
                deterministic_pass and visual_scores_pass
                if self._visual_score_hard_gate
                else deterministic_pass
            ),
            requirement_status=requirement_status,
            quality_status=quality_status,
        )

        console_logger.info(
            "FullVerifier: "
            f"semantic={semantic:.2f} aesthetic={aesthetic:.2f} "
            f"plausibility_score={plausibility:.2f} physics={physics:.2f} "
            f"interaction={interaction:.2f} walkability={walkability:.2f} "
            f"overall={overall:.2f} pass={'YES' if report.pass_scene else 'NO'} "
            f"visual_gate={self._visual_score_hard_gate}"
        )
        return report

    @staticmethod
    def _aggregate_requirement_status(
        stage_reports: list[StageVerifyReport],
    ) -> str:
        """Aggregate required-asset fulfillment independently from quality."""
        statuses = [
            report.requirement_status
            for report in stage_reports
            if report.requirement_status not in {"not_applicable", "unknown"}
        ]
        if not statuses:
            return "not_applicable"
        if all(status == "satisfied" for status in statuses):
            return "satisfied"
        if all(status == "unsatisfied" for status in statuses):
            return "unsatisfied"
        return "partial"

    @staticmethod
    def _deterministic_stage_passes(stage_reports: list[StageVerifyReport]) -> bool:
        """Evaluate stage hard gates against the latest validated scene state.

        A stage can record a transient deterministic failure (usually a collision)
        and a later stage can repair it while validating the complete scene again.
        The terminal verdict should retain unresolved inventory/contract failures,
        but must not reject that later clean state solely from the stale marker.
        """
        for index, report in enumerate(stage_reports):
            if report.pass_stage:
                continue
            issue_types = {issue.issue_type for issue in report.issues}
            is_transient_hard_failure = issue_types == {"deterministic_hard_failure"}
            later_clean_validation = any(
                later.pass_stage
                and _DETERMINISTIC_HARD_CHECK_MARKER
                not in later.critique_summary.upper()
                for later in stage_reports[index + 1 :]
            )
            if is_transient_hard_failure and later_clean_validation:
                console_logger.info(
                    "FullVerifier: accepting recovered deterministic failure from "
                    "stage=%s after a later clean validation",
                    report.stage,
                )
                continue
            return False
        return True
