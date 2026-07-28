"""Pure mapping from normalized critic evidence to SceneExpert memory records.

Keeping this conversion outside the hook runner limits the integration surface
with SceneSmith orchestration and makes the Critic -> Memory contract directly
testable.
"""

from __future__ import annotations

import hashlib

from scenesmith.scene_expert.memory.schemas import FailureCase, SuccessCase
from scenesmith.scene_expert.memory.text_builder import build_embedding_text
from scenesmith.scene_expert.schemas import CriticEvidence, SceneTaskSpec


def build_critic_memory_records(
    *,
    evidence: CriticEvidence,
    task_spec: SceneTaskSpec,
    stage: str,
    required_objects: list[str],
    trace_ref: str,
    created_at: str,
) -> tuple[list[FailureCase], SuccessCase | None]:
    """Convert authoritative ``core`` critic evidence into durable memory.

    Auxiliary and ignored observations never create durable records. A success
    is admitted only when every core result is known, none failed/degraded, and
    the aggregate scene score clears the conservative 0.75 admission threshold.
    """
    if not evidence.available:
        return [], None

    failure_cases: list[FailureCase] = []
    for result in evidence.core_failures:
        reason = result.reason or (
            f"SceneBenchmark check {result.check_id or result.metric} "
            f"returned {result.label}"
        )
        signature = "|".join(
            (
                task_spec.room_type,
                stage,
                result.check_id,
                result.primary_object,
                result.metric,
                reason,
            )
        )
        digest = hashlib.sha1(signature.encode("utf-8")).hexdigest()[:12]
        case = FailureCase(
            failure_id=(
                f"failure_scenebenchmark_{task_spec.room_type}_{stage}_{digest}"
            ),
            room_type=task_spec.room_type,
            stage=stage,
            object=result.primary_object,
            failure_type=f"scenebenchmark_{result.metric}",
            bad_pattern=reason[:900],
            failure_reason=reason[:900],
            repair_action=(
                result.repair_advice
                or "Satisfy the failed deterministic critic check, then re-run "
                "the same check before accepting the stage."
            ),
            repair_verified=False,
            required_objects=required_objects,
            functional_zones=task_spec.functional_zones,
            scene_summary=(
                f"SceneBenchmark critic reported a {result.label} core check "
                f"for {stage} in {trace_ref}."
            ),
            quality_score=(
                evidence.scene_score
                if evidence.scene_score is not None
                else (0.0 if result.label == "fail" else 0.5)
            ),
            confidence=result.confidence if result.confidence is not None else 0.9,
            created_at=created_at,
            scope="object" if result.primary_object else "stage",
            is_deterministic=True,
            negative_constraint=(
                f"Avoid repeating SceneBenchmark {result.metric} failure: {reason}"
            )[:900],
            critic_check=result.check_id or result.metric,
            trace_ref=trace_ref,
        )
        failure_cases.append(
            case.model_copy(update={"embedding_text": build_embedding_text(case)})
        )

    if failure_cases:
        return failure_cases, None
    if (
        evidence.core_total_checks <= 0
        or evidence.core_pass_count != evidence.core_total_checks
        or evidence.core_unknown_count > 0
        or evidence.scene_score is None
        or evidence.scene_score < 0.75
    ):
        return [], None

    passed_metrics = sorted(evidence.metric_scores)
    signature = "|".join(
        (
            task_spec.room_type,
            task_spec.style,
            stage,
            ",".join(required_objects),
            ",".join(passed_metrics),
        )
    )
    digest = hashlib.sha1(signature.encode("utf-8")).hexdigest()[:12]
    guidance = "Preserve the deterministic relationships verified by SceneBenchmark" + (
        f": {', '.join(passed_metrics)}." if passed_metrics else "."
    )
    success_case = SuccessCase(
        case_id=f"success_scenebenchmark_{task_spec.room_type}_{stage}_{digest}",
        room_type=task_spec.room_type,
        style=task_spec.style,
        stage=stage,
        task_signature=required_objects,
        required_objects=required_objects,
        functional_zones=task_spec.functional_zones,
        scene_summary=(
            f"All {evidence.core_total_checks} SceneBenchmark core checks "
            f"passed for {stage} in {trace_ref}."
        ),
        successful_pattern=[guidance],
        positive_guidance=[
            guidance,
            "Re-run the same deterministic critic checks after adapting the "
            "layout to a new room.",
        ],
        scores={
            f"critic.{metric}": score
            for metric, score in evidence.metric_scores.items()
        },
        trace_ref=trace_ref,
        quality_score=evidence.scene_score,
        confidence=0.9,
        created_at=created_at,
    )
    return (
        [],
        success_case.model_copy(
            update={"embedding_text": build_embedding_text(success_case)}
        ),
    )
