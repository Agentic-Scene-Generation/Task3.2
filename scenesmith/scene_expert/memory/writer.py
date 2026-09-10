"""Strict, evidence-gated long-term memory writer for SceneExpert.

The LLM only proposes compact lessons. Deterministic code owns identity, task
metadata, critic evidence, quality gates, provenance, and promotion into the
active memory bank. A failed or empty LLM response can never manufacture a
retrievable fallback record. Deterministic code may still persist a
non-retrievable Skill candidate when an independently executed native stage has
an authoritative pass and a grounded task contract.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time

from pathlib import Path
from typing import Any

from scenesmith.scene_expert.memory.evidence import resolve_constraint_evidence
from scenesmith.scene_expert.memory.placement import (
    inventory_only,
    object_role,
    valid_episode,
)
from scenesmith.scene_expert.memory.schemas import (
    FailureCase,
    FailureMemoryCandidate,
    MemorySourceProvenance,
    MemoryUpdateOp,
    MemoryWriterResponse,
    PlacementEpisode,
    PlacementExperience,
    Skill,
    SkillApplicability,
    SkillMemoryCandidate,
    SpatialRelationMemory,
    SuccessCase,
    SuccessMemoryCandidate,
)
from scenesmith.scene_expert.memory.skill_bootstrap import bootstrap_grounded_skills
from scenesmith.scene_expert.memory.skill_identity import build_skill_semantic_signature
from scenesmith.scene_expert.memory.text_builder import build_embedding_text
from scenesmith.scene_expert.memory.writer_prompt import (
    WriterPromptBudgetError,
    build_writer_prompt,
    byte_size,
)
from scenesmith.scene_expert.schemas import FullVerifyReport
from scenesmith.scene_expert.structured_llm import (
    SceneExpertStructuredLLMClient,
    StructuredLLMProfile,
    StructuredLLMResult,
)

console_logger = logging.getLogger(__name__)

SUCCESS_MEMORY_MIN_OVERALL_SCORE = 0.75
_SUPPORTED_STAGES = {
    "floor_plan",
    "furniture",
    "wall_mounted",
    "ceiling_mounted",
    "manipuland",
}
_DETERMINISTIC_FAILURE_KEYWORDS = (
    "deterministic",
    "missing mesh",
    "missing file",
    "file missing",
    "hssd",
    "openclip",
    "clip weight",
    "checkpoint missing",
    "degenerate mesh",
    "invalid mesh",
    "mesh file",
    "asset file",
    "candidate file",
    "geometry failure",
    "hard failure",
    "hard constraint",
    "missing required",
)

_SYSTEM_PROMPT = """\
You are SceneExpert's long-term memory curator.

Extract only reusable lessons that are explicitly supported by the supplied
trace and the authoritative SceneSmith/SceneBenchmark critic evidence.

Rules:
- Return the exact JSON schema supplied by the server.
- Always return all four top-level keys using this shape:
  {"success_cases": [{"stage": "furniture", "successful_pattern": ["..."],
  "positive_guidance": ["..."]}], "failure_cases": [], "skills": [],
  "noop_reason": ""}
- stage must be one of floor_plan, furniture, wall_mounted,
  ceiling_mounted, or manipuland.
- Do not invent IDs, scores, object coordinates, task metadata, or provenance.
- A success lesson must describe what transferred well, not merely that a stage passed.
- A failed or degraded final scene may still contain a reusable success from an
  earlier stage, but propose it only when that exact stage has an authoritative
  passing verify_report. It will be stored as stage-local, never scene-level.
- A failure lesson is allowed only when the trace shows a verified repair or a
  deterministic/repeatable hard failure. Never label visual opinion as deterministic.
- A skill must contain a reusable procedure with at least two concrete steps.
- A failed final scene may still yield a Skill candidate only from an exact
  stage whose authoritative verify_report passed. Never propose a Skill from
  the failed stage itself. Deterministic code decides candidate versus active.
- Prefer empty arrays with a clear noop_reason over weak, duplicate, or speculative memory.
- Keep each lesson concise and useful for a different scene with similar requirements.
- Return at most three candidates total to leave enough output space for complete JSON.
- When placement_experience_catalog is supplied, each success/failure MUST select
  one or two exact episode_ids from that catalog, and supply procedure (2-6 steps)
  and applicability. An empty catalog means no new spatial success/failure, NOT
  permission to restate the task. Required inventories remain metadata only.
- Explain HOW to place/adjust relative to an anchor, not merely WHAT is required.
  Include overlooked geometry, order-of-operations or a supported failure pattern.
  A failure requires an exact failing native check for the selected object pair.
- Treat relative offsets/yaws/AABB separation as observed source measurements,
  never semantic front, walkable clearance, optimal values or universal thresholds.
  Do not claim a repair worked: these final-stage episodes do not prove that.
  Python binds observations; do not invent or rewrite measurements/evidence.
- Optional assets and all stages may yield lessons when the catalog supports them.
"""


class MemoryWriter:
    """Generate typed memory candidates and promote only evidence-backed records."""

    @staticmethod
    def _normalize_update_op(raw_op: Any) -> dict[str, Any]:
        """Normalize historical local-model NOOP payloads before validation."""
        if not isinstance(raw_op, dict):
            raise TypeError(
                f"Memory update must be an object, got {type(raw_op).__name__}"
            )
        operation = dict(raw_op)
        if operation.get("content") is None:
            operation["content"] = {}
        if operation.get("target_id") is None:
            operation["target_id"] = ""
        return operation

    def __init__(
        self,
        model: str,
        api_base_url: str | None = None,
        api_key: str | None = None,
        max_tokens: int = 3072,
        retry_max_tokens: int | None = None,
        thinking_mode: str = "none",
        timeout_seconds: float = 90.0,
        temperature: float = 0.1,
        success_min_overall_score: float = SUCCESS_MEMORY_MIN_OVERALL_SCORE,
        skill_min_independent_support: int = 2,
        skill_bootstrap_enabled: bool = True,
        skill_bootstrap_max_candidates_per_scene: int = 5,
        skill_bootstrap_min_procedure_steps: int = 2,
        debug_dir: str | Path | None = None,
        llm_client: SceneExpertStructuredLLMClient | None = None,
    ) -> None:
        self._model = model
        self._debug_dir = Path(debug_dir) if debug_dir else None
        self._success_min_overall_score = float(success_min_overall_score)
        self._skill_min_independent_support = max(
            2,
            int(
                os.environ.get(
                    "SCENEEXPERT_SKILL_MIN_INDEPENDENT_SUPPORT",
                    skill_min_independent_support,
                )
            ),
        )
        self._skill_bootstrap_enabled = self._env_bool(
            "SCENEEXPERT_SKILL_BOOTSTRAP_ENABLED", skill_bootstrap_enabled
        )
        self._skill_bootstrap_max_candidates_per_scene = max(
            0,
            int(
                os.environ.get(
                    "SCENEEXPERT_SKILL_BOOTSTRAP_MAX_CANDIDATES_PER_SCENE",
                    skill_bootstrap_max_candidates_per_scene,
                )
            ),
        )
        self._skill_bootstrap_min_procedure_steps = max(
            2,
            int(
                os.environ.get(
                    "SCENEEXPERT_SKILL_BOOTSTRAP_MIN_PROCEDURE_STEPS",
                    skill_bootstrap_min_procedure_steps,
                )
            ),
        )
        max_tokens = int(
            os.environ.get("SCENEEXPERT_MEMORY_WRITER_MAX_TOKENS", max_tokens)
        )
        retry_tokens = int(
            os.environ.get(
                "SCENEEXPERT_MEMORY_WRITER_RETRY_MAX_TOKENS",
                (
                    retry_max_tokens
                    if retry_max_tokens is not None
                    else max(max_tokens, 4096)
                ),
            )
        )
        self._profile = StructuredLLMProfile(
            thinking_mode=str(thinking_mode or "none"),
            max_tokens=max_tokens,
            retry_max_tokens=retry_tokens,
            timeout_seconds=float(timeout_seconds),
            temperature=float(temperature),
            max_attempts=2,
            response_format="json_schema",
        )
        self._context_tokens = int(
            os.environ.get("SCENEEXPERT_MEMORY_WRITER_CONTEXT_TOKENS", 65536)
        )
        self._max_input_bytes = int(
            os.environ.get("SCENEEXPERT_MEMORY_WRITER_INPUT_MAX_BYTES", 49152)
        )
        if min(self._context_tokens, self._max_input_bytes) <= 0:
            raise ValueError("MemoryWriter context and input limits must be positive")
        self._llm_client = llm_client or SceneExpertStructuredLLMClient(
            model=model,
            api_base_url=api_base_url,
            api_key=api_key,
        )
        self.last_trace: dict[str, Any] = {
            "success": False,
            "source": "not_run",
            "degraded": False,
            "attempt_count": 0,
            "persisted_count": 0,
            "llm_skill_candidate_count": 0,
            "bootstrap_skill_eligible_stage_count": 0,
            "bootstrap_skill_candidate_count": 0,
            "bootstrap_skill_persisted_candidate_count": 0,
            "bootstrap_skill_rejected_count": 0,
            "bootstrap_skill_decisions": [],
            "skill_persisted_candidate_count": 0,
            "skill_promoted_active_count": 0,
            "skill_rejected_count": 0,
            "skill_rejection_reasons": {},
            "skill_decisions": [],
        }

    def write(
        self,
        trace_summary: str,
        full_report: FullVerifyReport,
        related_old_memory: str = "",
        evidence_payload: dict[str, Any] | None = None,
    ) -> list[MemoryUpdateOp]:
        """Return active-bank mutations derived from one completed scene.

        ``evidence_payload`` is the preferred runtime contract. It contains the
        untruncated main critic reports, repair outcomes, task spec, and trace
        identity. ``trace_summary`` remains for human context and compatibility.
        """
        evidence = dict(evidence_payload or {})
        self._prompt_attempts: list[dict] = []
        self._visible_episode_ids: set[str] | None = None
        # Save the complete source before any model request, including failure.
        if self._debug_dir is not None:
            self._debug_dir.mkdir(parents=True, exist_ok=True)
            self._atomic_write_json(
                self._debug_dir / "memory_writer_input.json",
                {
                    "schema_version": "memory-writer-input.v1",
                    "trace_summary": trace_summary,
                    "evidence": evidence,
                    "full_report": full_report.model_dump(),
                    "related_old_memory": related_old_memory,
                },
            )
        self._active_user_budget = self._user_budget(self._context_tokens)

        def messages() -> list[dict[str, Any]]:
            return [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": self._build_user_message(
                        trace_summary=trace_summary,
                        full_report=full_report,
                        related_old_memory=related_old_memory,
                        evidence_payload=evidence,
                    ),
                },
            ]

        def reduce_context(previous: list[dict], error: str) -> list[dict] | None:
            # Read a server's per-slot limit, not its advertised total capacity.
            match = re.search(
                r"(?:available context size\s*\(|[\"']n_ctx[\"']\s*:\s*|maximum context length is\s*)(\d+)",
                error,
            )
            limit = (
                self._user_budget(int(match[1])) if match else self._active_user_budget
            )
            previous_bytes = len(str(previous[-1].get("content", "")).encode("utf-8"))
            self._active_user_budget = min(limit, previous_bytes // 2)
            try:
                return messages()
            except WriterPromptBudgetError:
                return None

        try:
            result = self._llm_client.complete(
                role="memory_writer",
                stage="full_scene",
                event="write_long_term_memory",
                messages=messages(),
                response_model=MemoryWriterResponse,
                profile=self._profile,
                context_reducer=reduce_context,
            )
        except WriterPromptBudgetError as exc:
            result = StructuredLLMResult(
                final_error_kind="context_budget", final_error=str(exc)
            )
        self.last_trace = result.status_dict()
        self.last_trace["prompt_projection_attempts"] = self._prompt_attempts
        self.last_trace.update(
            {
                "llm_skill_candidate_count": 0,
                "bootstrap_skill_eligible_stage_count": 0,
                "bootstrap_skill_candidate_count": 0,
                "bootstrap_skill_persisted_candidate_count": 0,
                "bootstrap_skill_rejected_count": 0,
                "bootstrap_skill_decisions": [],
                "skill_persisted_candidate_count": 0,
                "skill_promoted_active_count": 0,
                "skill_rejected_count": 0,
                "skill_rejection_reasons": {},
                "skill_decisions": [],
                "persisted_count": 0,
            }
        )

        response = result.value if result.success and result.value is not None else None
        llm_candidate_ops = (
            self._response_to_ops(
                response=response,
                trace_summary=trace_summary,
                full_report=full_report,
                evidence_payload=evidence,
            )
            if response is not None
            else []
        )
        bootstrap_ops, bootstrap_decisions = self._bootstrap_skill_ops(
            trace_summary=trace_summary,
            full_report=full_report,
            evidence_payload=evidence,
        )
        candidate_ops = self._dedupe_candidate_ops([*llm_candidate_ops, *bootstrap_ops])
        promoted_ops = self._gate_and_enrich_ops(
            candidate_ops,
            full_report,
            evidence_payload=evidence,
        )
        mutating_ops = [op for op in promoted_ops if op.op in {"ADD", "UPDATE"}]
        active_promotion_ops = [
            op
            for op in mutating_ops
            if op.memory_type != "skill" or str(op.content.get("status")) == "active"
        ]
        skill_decisions = list(getattr(self, "_last_skill_decisions", []))
        rejection_reasons: dict[str, int] = {}
        for decision in skill_decisions:
            if decision.get("decision") != "rejected":
                continue
            for reason in decision.get("reasons", []) or []:
                reason_text = str(reason)
                rejection_reasons[reason_text] = (
                    int(rejection_reasons.get(reason_text, 0)) + 1
                )
        structured_failure = response is None
        if structured_failure and not mutating_ops:
            status = "model_failure_no_write"
        else:
            status = (
                "promoted"
                if active_promotion_ops
                else "persisted_candidate" if mutating_ops else "no_valid_candidates"
            )
        bootstrap_persisted = sum(
            decision.get("decision") == "persisted_candidate"
            and decision.get("source") == "deterministic"
            for decision in skill_decisions
        )
        bootstrap_rejected = sum(
            decision.get("decision") == "rejected"
            and decision.get("source") == "deterministic"
            for decision in skill_decisions
        )
        structured_call_source = self.last_trace.get("source", "")
        self.last_trace.update(
            {
                "write_status": status,
                "candidate_count": len(candidate_ops),
                "generated_candidate_count": (
                    len(response.success_cases)
                    + len(response.failure_cases)
                    + len(response.skills)
                    if response is not None
                    else 0
                ),
                "proposed_mutation_count": len(mutating_ops),
                "persistence_confirmed": False,
                "count_semantics": "writer counts are proposals; store_apply is authoritative",
                "persisted_count": len(mutating_ops),
                "promoted_count": len(active_promotion_ops),
                "candidate_counts": self._op_counts(candidate_ops),
                "persisted_counts": self._op_counts(mutating_ops),
                "proposed_counts": self._op_counts(mutating_ops),
                "promoted_counts": self._op_counts(active_promotion_ops),
                "noop_reason": (
                    response.noop_reason
                    if response is not None
                    else str(result.final_error or result.final_error_kind or "")
                ),
                "fallback_written": False,
                "llm_skill_candidate_count": (
                    len(response.skills) if response is not None else 0
                ),
                "bootstrap_skill_eligible_stage_count": sum(
                    1
                    for stage in {
                        str(decision.get("stage") or "")
                        for decision in bootstrap_decisions
                        if decision.get("decision") == "generated"
                    }
                    if stage
                ),
                "bootstrap_skill_candidate_count": sum(
                    op.memory_type == "skill"
                    and str(op.content.get("source") or "") == "deterministic"
                    for op in candidate_ops
                ),
                "bootstrap_skill_persisted_candidate_count": bootstrap_persisted,
                "bootstrap_skill_rejected_count": bootstrap_rejected,
                "bootstrap_skill_decisions": bootstrap_decisions,
                "skill_persisted_candidate_count": sum(
                    decision.get("decision") == "persisted_candidate"
                    for decision in skill_decisions
                ),
                "skill_promoted_active_count": sum(
                    decision.get("decision") == "promoted_active"
                    for decision in skill_decisions
                ),
                "skill_rejected_count": sum(
                    decision.get("decision") == "rejected"
                    for decision in skill_decisions
                ),
                "skill_rejection_reasons": rejection_reasons,
                "skill_decisions": skill_decisions,
            }
        )
        if structured_failure:
            self.last_trace.update(
                {
                    "structured_call_source": structured_call_source,
                    "source": (
                        "deterministic_skill_bootstrap" if mutating_ops else "no_write"
                    ),
                    "degraded": True,
                }
            )
        self._save_debug_payload(
            status=status,
            result_status=self.last_trace,
            trace_summary=trace_summary,
            full_report=full_report,
            evidence_payload=evidence,
            response=response,
            result_ops=mutating_ops,
        )
        if structured_failure:
            console_logger.warning(
                "MemoryWriter structured output failed after %d attempts; "
                "proposed %d independently gated deterministic Skill candidate(s): %s",
                len(result.attempts),
                bootstrap_persisted,
                result.final_error or result.final_error_kind,
            )
        console_logger.info(
            "MemoryWriter: proposed %d/%d schema-valid candidates; awaiting store; fallback_written=false",
            len(mutating_ops),
            len(candidate_ops),
        )
        return mutating_ops

    def _response_to_ops(
        self,
        *,
        response: MemoryWriterResponse,
        trace_summary: str,
        full_report: FullVerifyReport,
        evidence_payload: dict[str, Any],
    ) -> list[MemoryUpdateOp]:
        context = self._canonical_context(evidence_payload, trace_summary)
        ops: list[MemoryUpdateOp] = []
        self.last_trace["placement_candidate_decisions"] = []
        for candidate in response.success_cases:
            content = self._success_content(candidate, context, full_report)
            if not self._bind_placement(
                content, candidate, evidence_payload, failure=False
            ):
                continue
            ops.append(
                MemoryUpdateOp(op="ADD", memory_type="success_case", content=content)
            )
        for candidate in response.failure_cases:
            content = self._failure_content(candidate, context)
            if not self._bind_placement(
                content, candidate, evidence_payload, failure=True
            ):
                continue
            ops.append(
                MemoryUpdateOp(op="ADD", memory_type="failure_case", content=content)
            )
        for candidate in response.skills:
            content = self._skill_content(candidate, context, full_report)
            ops.append(MemoryUpdateOp(op="ADD", memory_type="skill", content=content))
        return ops

    def _bind_placement(
        self,
        content: dict[str, Any],
        candidate: Any,
        evidence: dict[str, Any],
        *,
        failure: bool,
    ) -> bool:
        """Atomically bind a proposed method to exact observed episodes.

        Legacy callers retain their old API. New runtime catalogs require explicit
        evidence; missing or edited IDs cannot produce active fallback memories.
        """
        if "placement_experience_catalog" not in evidence:
            return True
        catalog = evidence["placement_experience_catalog"] or {}
        by_id: dict[str, PlacementEpisode] = {}
        for raw in catalog.get("episodes", []):
            try:
                episode = PlacementEpisode.model_validate(raw)
            except (ValueError, TypeError):
                continue
            if valid_episode(episode):
                by_id[episode.episode_id] = episode
        ids = candidate.episode_ids
        episodes = [by_id[key] for key in ids if key in by_id]
        procedure = self._clean_list(candidate.procedure)
        reasons = []
        visible = getattr(self, "_visible_episode_ids", None)
        if visible is not None and any(eid not in visible for eid in ids):
            reasons.append("episode_not_in_model_input")
        if not ids or len(ids) != len(set(ids)) or len(episodes) != len(ids):
            reasons.append("missing_or_invalid_episode_binding")
        if any(e.stage != candidate.stage for e in episodes):
            reasons.append("episode_stage_mismatch")
        if len(procedure) < 2 or not self._clean_list(candidate.applicability):
            reasons.append("missing_procedure_or_applicability")
        if inventory_only(procedure):
            reasons.append("redundant_inventory_restatement")
        if failure and not any(
            check["status"] == "verified_fail"
            for e in episodes
            for check in e.native_checks
        ):
            reasons.append("no_exact_pair_failure_evidence")
        if not failure and any(
            check["status"] == "verified_fail"
            for e in episodes
            for check in e.native_checks
        ):
            reasons.append("selected_pair_has_verified_failure")
        if not failure and any(e.stage_passed is not True for e in episodes):
            reasons.append("selected_episode_stage_not_passed")
        self.last_trace.setdefault("placement_candidate_decisions", []).append(
            {
                "stage": candidate.stage,
                "episode_ids": ids,
                "decision": "rejected" if reasons else "bound",
                "reasons": reasons,
            }
        )
        if reasons:
            return False
        experience = PlacementExperience(
            procedure=procedure,
            applicability=self._clean_list(candidate.applicability),
            episodes=episodes,
        )
        content["placement_experience"] = experience.model_dump(mode="json")
        if not failure:
            # These scores describe the selected attempt, not another entry
            # for the same stage that happened before a retry.
            content["scores"] = episodes[0].stage_scores
        content["spatial_relations"] = []
        for episode in episodes:
            relation = SpatialRelationMemory(
                relation_type="observed_relative_pose",
                subject_role=object_role(episode.subject),
                target_role=object_role(episode.anchor),
                evidence_source="scene_geometry",
                evidence_ref=episode.episode_id,
                verification_status="inconclusive",
                confidence=0.5,
            )
            relation.claim_hash = relation.current_claim_hash()
            content["spatial_relations"].append(relation.model_dump(mode="json"))
        content["evidence_refs"] = sorted(
            {ref for e in episodes for ref in e.evidence_refs if ref}
        )
        content["provenance"]["evidence_refs"] = content["evidence_refs"]
        content["provenance"]["scene_state_path"] = episodes[0].evidence_refs[0]
        if failure:
            content["repair_verified"] = False
            content["is_deterministic"] = True
            content["scope"] = "object"
        return True

    def _bootstrap_skill_ops(
        self,
        *,
        trace_summary: str,
        full_report: FullVerifyReport,
        evidence_payload: dict[str, Any],
    ) -> tuple[list[MemoryUpdateOp], list[dict[str, Any]]]:
        """Build candidate-only Skills for passing stages omitted by the LLM.

        This is not a free-form fallback.  The pure bootstrapper requires proof
        that the native stage agent ran, an exact-stage main-critic pass, and a
        grounded task contract.  Store-level independent support is still
        required before any resulting Skill becomes retrievable.
        """
        if not self._skill_bootstrap_enabled:
            return [], [
                {
                    "stage": "*",
                    "decision": "disabled",
                    "reasons": ["skill_bootstrap_disabled"],
                }
            ]
        result = bootstrap_grounded_skills(
            evidence_payload,
            max_candidates=self._skill_bootstrap_max_candidates_per_scene,
            min_procedure_steps=self._skill_bootstrap_min_procedure_steps,
        )
        context = self._canonical_context(evidence_payload, trace_summary)
        ops = [
            MemoryUpdateOp(
                op="ADD",
                memory_type="skill",
                content=self._skill_content(
                    draft.candidate,
                    context,
                    full_report,
                    source="deterministic",
                    activation_reason=(
                        "verified_stage_bootstrap_awaiting_independent_support"
                    ),
                    required_objects_override=list(draft.required_objects),
                    relation_types_override=(
                        []
                        if list(draft.relation_types) == ["required_coverage"]
                        else list(draft.relation_types)
                    ),
                    constraint_ids=set(draft.constraint_ids),
                ),
            )
            for draft in result.drafts
        ]
        return ops, [dict(item) for item in result.decisions]

    def _success_content(
        self,
        candidate: SuccessMemoryCandidate,
        context: dict[str, Any],
        full_report: FullVerifyReport,
    ) -> dict[str, Any]:
        stage_evidence = self._stage_evidence(context, candidate.stage)
        required_objects = self._required_objects(context["task_spec"], candidate.stage)
        scores = self._stage_scores(stage_evidence)
        scene_passed = bool(full_report.pass_scene)
        promotion_scope = "scene" if scene_passed else "stage"
        stage_quality = self._mean_score(scores)
        now = self._now()
        record = SuccessCase(
            case_id=self._record_id("success", candidate, context),
            promotion_scope=promotion_scope,
            source_scene_passed=scene_passed,
            room_type=context["room_type"],
            style=context["style"],
            stage=candidate.stage,
            task_signature=self._unique(required_objects + context["functional_zones"]),
            successful_pattern=self._clean_list(candidate.successful_pattern),
            positive_guidance=self._clean_list(
                candidate.positive_guidance or candidate.successful_pattern
            ),
            scores=scores,
            trace_ref=context["trace_id"],
            required_objects=required_objects,
            functional_zones=context["functional_zones"],
            scene_summary=(
                f"Evidence-backed {promotion_scope}-level {candidate.stage} lesson "
                f"from {context['trace_id']}."
            ),
            confidence=self._evidence_confidence(stage_evidence),
            quality_score=(
                float(full_report.overall_score) if scene_passed else stage_quality
            ),
            created_at=now,
            updated_at=now,
            status="active",
            source="llm",
            source_task_id=context["source_task_id"],
            source_run_id=context["source_run_id"],
            source_task_ids=[context["source_task_id"]],
            source_run_ids=[context["source_run_id"]],
            prompt_fingerprint=context["prompt_fingerprint"],
            evidence_refs=self._evidence_refs(context, candidate.stage),
            critic_evidence=self._critic_evidence(stage_evidence),
            provenance=self._provenance(context, candidate.stage),
            spatial_relations=self._spatial_relations(stage_evidence),
        )
        return record.model_dump()

    def _failure_content(
        self,
        candidate: FailureMemoryCandidate,
        context: dict[str, Any],
    ) -> dict[str, Any]:
        stage_evidence = self._stage_evidence(context, candidate.stage)
        required_objects = self._required_objects(context["task_spec"], candidate.stage)
        now = self._now()
        record = FailureCase(
            failure_id=self._record_id("failure", candidate, context),
            room_type=context["room_type"],
            stage=candidate.stage,
            object=candidate.object,
            failure_type=candidate.failure_type,
            bad_pattern=candidate.bad_pattern,
            failure_reason=candidate.failure_reason,
            repair_action=candidate.repair_action,
            repair_verified=candidate.repair_verified,
            required_objects=required_objects,
            functional_zones=context["functional_zones"],
            scene_summary=f"Evidence-backed {candidate.stage} failure from {context['trace_id']}.",
            confidence=0.7,
            quality_score=0.7,
            created_at=now,
            updated_at=now,
            scope=candidate.scope,
            is_deterministic=candidate.is_deterministic,
            negative_constraint=candidate.negative_constraint or candidate.bad_pattern,
            critic_check=candidate.critic_check,
            trace_ref=context["trace_id"],
            status="active",
            source="llm",
            source_task_id=context["source_task_id"],
            source_run_id=context["source_run_id"],
            source_task_ids=[context["source_task_id"]],
            source_run_ids=[context["source_run_id"]],
            prompt_fingerprint=context["prompt_fingerprint"],
            evidence_refs=self._evidence_refs(context, candidate.stage),
            critic_evidence=self._critic_evidence(stage_evidence),
            provenance=self._provenance(context, candidate.stage),
            spatial_relations=self._spatial_relations(
                stage_evidence,
                focus_terms=[
                    candidate.object,
                    candidate.failure_type,
                    candidate.bad_pattern,
                ],
            ),
        )
        return record.model_dump()

    def _skill_content(
        self,
        candidate: SkillMemoryCandidate,
        context: dict[str, Any],
        full_report: FullVerifyReport,
        *,
        source: str = "llm",
        activation_reason: str = "awaiting_independent_stage_support",
        required_objects_override: list[str] | None = None,
        relation_types_override: list[str] | None = None,
        constraint_ids: set[str] | None = None,
    ) -> dict[str, Any]:
        stage_evidence = self._stage_evidence(context, candidate.stage)
        required_objects = (
            self._clean_list(required_objects_override)
            if required_objects_override is not None
            else self._required_objects(context["task_spec"], candidate.stage)
        )
        spatial_relations = self._spatial_relations(
            stage_evidence,
            constraint_ids=constraint_ids,
        )
        if source == "deterministic":
            spatial_relations = [
                relation.model_copy(
                    update={
                        # Counts/groups are task instances, not reusable Skill
                        # identity. Keep structural qualifiers and instruct the
                        # Skill to consume the current task's cardinality.
                        "cardinality": {
                            key: value
                            for key, value in relation.cardinality.items()
                            if key in {"orientation", "edge_frame"}
                        },
                        "template_parameters": sorted(
                            key
                            for key in relation.cardinality
                            if key not in {"orientation", "edge_frame"}
                        ),
                    }
                )
                for relation in spatial_relations
            ]
            spatial_relations = [
                relation.model_copy(
                    update={"claim_hash": relation.current_claim_hash()}
                )
                for relation in spatial_relations
            ]
        relation_types = (
            self._clean_list(relation_types_override)
            if relation_types_override is not None
            else self._unique(relation.relation_type for relation in spatial_relations)
        )
        now = self._now()
        record = Skill(
            skill_name=candidate.skill_name,
            stage=candidate.stage,
            room_type=context["room_type"],
            room_types=[context["room_type"]] if context["room_type"] else [],
            style=context["style"],
            required_objects=required_objects,
            functional_zones=context["functional_zones"],
            scene_summary=(
                f"Evidence-backed {source} procedure from {context['trace_id']}."
            ),
            preconditions=self._clean_list(candidate.preconditions),
            procedure=self._clean_list(candidate.procedure),
            failure_avoidance=self._clean_list(candidate.failure_avoidance),
            postconditions=self._clean_list(candidate.postconditions),
            confidence=self._evidence_confidence(stage_evidence),
            quality_score=float(full_report.overall_score),
            success_rate=float(full_report.overall_score),
            trace_ref=context["trace_id"],
            created_at=now,
            updated_at=now,
            status="candidate",
            source=source,
            source_task_id=context["source_task_id"],
            source_run_id=context["source_run_id"],
            source_task_ids=[context["source_task_id"]],
            source_run_ids=[context["source_run_id"]],
            prompt_fingerprint=context["prompt_fingerprint"],
            evidence_refs=self._evidence_refs(context, candidate.stage),
            critic_evidence=self._critic_evidence(stage_evidence),
            provenance=self._provenance(context, candidate.stage),
            spatial_relations=spatial_relations,
            applicability=SkillApplicability(
                room_types=[context["room_type"]] if context["room_type"] else [],
                required_object_roles=required_objects,
                required_relation_types=relation_types,
            ),
            skill_aliases=[candidate.skill_name],
            promotion_scope="stage",
            source_scene_passed=bool(full_report.pass_scene),
            independent_support_count=1,
            activation_min_independent_support=max(
                2, int(getattr(self, "_skill_min_independent_support", 2))
            ),
            activation_reason=activation_reason,
        )
        record = record.model_copy(
            update={"semantic_signature": build_skill_semantic_signature(record)}
        )
        return record.model_dump()

    def _gate_and_enrich_ops(
        self,
        ops: list[MemoryUpdateOp],
        full_report: FullVerifyReport,
        evidence_payload: dict[str, Any] | None = None,
    ) -> list[MemoryUpdateOp]:
        """Validate persisted records and enforce deterministic promotion gates."""
        self._last_skill_decisions: list[dict[str, Any]] = []
        promotion_decisions = self.last_trace.setdefault(
            "placement_promotion_decisions", []
        )

        def record_decision(op: MemoryUpdateOp, decision: str, reason: str) -> None:
            if not op.content.get("placement_experience"):
                return
            promotion_decisions.append(
                {
                    "memory_type": op.memory_type,
                    "stage": op.content.get("stage"),
                    "record_id": op.content.get("case_id")
                    or op.content.get("failure_id"),
                    "decision": decision,
                    "reason": reason,
                }
            )

        evidence = dict(evidence_payload or {})
        has_structured_evidence = bool(evidence.get("stages"))
        success_threshold = float(
            getattr(
                self,
                "_success_min_overall_score",
                SUCCESS_MEMORY_MIN_OVERALL_SCORE,
            )
        )
        filtered: list[MemoryUpdateOp] = []
        for op in ops:
            if op.op == "NOOP":
                continue
            if op.memory_type == "success_case":
                record = self._validate_success(op.content)
                if record is None:
                    record_decision(op, "rejected", "schema_invalid")
                    continue
                stage_evidence = self._stage_evidence(evidence, record.stage)
                stage_passed = self._stage_passed(stage_evidence)
                if record.placement_experience is not None:
                    # A retried stage can have different outcomes in its trace.
                    # Use the actual selected episode, never the first stage row.
                    stage_passed = bool(record.placement_experience.episodes) and all(
                        e.stage_passed is True and valid_episode(e)
                        for e in record.placement_experience.episodes
                    )
                scene_success = bool(
                    full_report.pass_scene
                    and full_report.overall_score >= success_threshold
                )
                stage_local_success = bool(
                    has_structured_evidence and not scene_success and stage_passed
                )
                if has_structured_evidence and not stage_passed:
                    record_decision(op, "rejected", "selected_stage_not_verified")
                    console_logger.info(
                        "MemoryWriter: rejected success %s because its exact stage "
                        "did not pass authoritative verification",
                        record.case_id,
                    )
                    continue
                if not scene_success and not stage_local_success:
                    record_decision(
                        op, "rejected", "neither_scene_nor_stage_gate_passed"
                    )
                    console_logger.info(
                        "MemoryWriter: rejected success %s because neither the "
                        "scene gate nor the stage-local degraded gate passed",
                        record.case_id,
                    )
                    continue
                if stage_local_success:
                    record = record.model_copy(
                        update={
                            "promotion_scope": "stage",
                            "source_scene_passed": bool(full_report.pass_scene),
                            "confidence": min(float(record.confidence), 0.75),
                            "quality_score": self._mean_score(
                                record.scores
                                if record.placement_experience
                                else self._stage_scores(stage_evidence)
                            ),
                        }
                    )
                else:
                    record = record.model_copy(
                        update={
                            "promotion_scope": "scene",
                            "source_scene_passed": True,
                        }
                    )
                record = self._rebuild_embedding(record)
                filtered.append(op.model_copy(update={"content": record.model_dump()}))
                record_decision(
                    op,
                    "proposed_to_store",
                    "stage_success" if stage_local_success else "scene_success",
                )
                continue

            if op.memory_type == "failure_case":
                record = self._validate_failure(op.content)
                if record is None:
                    record_decision(op, "rejected", "schema_invalid")
                    continue
                stage_evidence = self._stage_evidence(evidence, record.stage)
                if record.placement_experience is not None:
                    repair_verified = False
                    deterministic = any(
                        row["status"] == "verified_fail"
                        for episode in record.placement_experience.episodes
                        for row in episode.native_checks
                    )
                elif has_structured_evidence:
                    repair_verified = self._repair_verified(stage_evidence)
                    deterministic = self._deterministic_failure_in_evidence(
                        stage_evidence
                    )
                else:
                    repair_verified = record.repair_verified
                    deterministic = self._detect_deterministic_failure(
                        record.model_dump()
                    )
                if not repair_verified and not deterministic:
                    record_decision(op, "rejected", "no_verified_failure_or_repair")
                    console_logger.info(
                        "MemoryWriter: rejected unverified/non-deterministic failure %s",
                        record.failure_id,
                    )
                    continue
                record = record.model_copy(
                    update={
                        "repair_verified": repair_verified,
                        "is_deterministic": deterministic,
                        "scope": (
                            "stage"
                            if deterministic
                            and record.scope == "object"
                            and record.placement_experience is None
                            else record.scope
                        ),
                        "confidence": 0.85,
                    }
                )
                record = self._rebuild_embedding(record)
                filtered.append(op.model_copy(update={"content": record.model_dump()}))
                record_decision(op, "proposed_to_store", "verified_failure")
                continue

            if op.memory_type == "skill":
                record = self._validate_skill(op.content)
                if record is None:
                    self._last_skill_decisions.append(
                        {
                            "skill_name": str(op.content.get("skill_name") or ""),
                            "stage": str(op.content.get("stage") or ""),
                            "source": str(op.content.get("source") or "unknown"),
                            "decision": "rejected",
                            "reasons": ["schema_invalid"],
                        }
                    )
                    continue
                stage_evidence = self._stage_evidence(evidence, record.stage)
                stage_report = self._stage_report(stage_evidence)
                stage_passed = self._stage_passed(stage_evidence)
                repair_verified = self._repair_verified(stage_evidence)
                deterministic_failure = self._deterministic_failure_in_evidence(
                    stage_evidence
                )
                execution_evidence = stage_evidence.get("execution_evidence") or {}
                if not isinstance(execution_evidence, dict):
                    execution_evidence = {}
                stage_agent_invoked = bool(
                    execution_evidence.get("stage_agent_invoked", False)
                )
                deterministic_bootstrap = bool(
                    record.source == "deterministic"
                    and record.activation_reason.startswith("verified_stage_bootstrap")
                )
                procedure_valid = len(self._clean_list(record.procedure)) >= 2
                active_eligible = bool(
                    not deterministic_bootstrap
                    and full_report.pass_scene
                    and full_report.overall_score >= success_threshold
                    and (not has_structured_evidence or stage_passed)
                    and procedure_valid
                    and not deterministic_failure
                )
                if deterministic_bootstrap:
                    candidate_eligible = bool(
                        has_structured_evidence
                        and stage_report
                        and stage_passed
                        and stage_agent_invoked
                        and procedure_valid
                        and not deterministic_failure
                    )
                else:
                    candidate_eligible = bool(
                        has_structured_evidence
                        and stage_report
                        and (stage_passed or repair_verified)
                        and procedure_valid
                        and not deterministic_failure
                    )
                if not active_eligible and not candidate_eligible:
                    reasons: list[str] = []
                    if not full_report.pass_scene:
                        reasons.append("scene_gate_failed")
                    if full_report.overall_score < success_threshold:
                        reasons.append("score_gate_failed")
                    if has_structured_evidence and not stage_report:
                        reasons.append("missing_stage_evidence")
                    elif has_structured_evidence and not stage_passed:
                        reasons.append("stage_gate_failed")
                    if deterministic_failure:
                        reasons.append("deterministic_stage_failure")
                    if deterministic_bootstrap and not stage_agent_invoked:
                        reasons.append("stage_agent_not_proven_invoked")
                    if not procedure_valid:
                        reasons.append("procedure_too_short")
                    if not reasons:
                        reasons.append("insufficient_verified_support")
                    self._last_skill_decisions.append(
                        {
                            "skill_name": record.skill_name,
                            "stage": record.stage,
                            "semantic_signature": record.semantic_signature,
                            "source": record.source,
                            "decision": "rejected",
                            "reasons": reasons,
                        }
                    )
                    console_logger.info(
                        "MemoryWriter: rejected unsupported skill %s reasons=%s",
                        record.skill_name,
                        ",".join(reasons),
                    )
                    continue
                if active_eligible:
                    record = record.model_copy(
                        update={
                            "status": "active",
                            "promotion_scope": "scene",
                            "source_scene_passed": True,
                            "activation_reason": "scene_and_stage_verified",
                        }
                    )
                    decision = "promoted_active"
                else:
                    stage_quality = self._mean_score(self._stage_scores(stage_evidence))
                    record = record.model_copy(
                        update={
                            "status": "candidate",
                            "promotion_scope": "stage",
                            "source_scene_passed": bool(full_report.pass_scene),
                            "quality_score": stage_quality,
                            "success_rate": stage_quality,
                            "confidence": min(float(record.confidence), 0.75),
                            "activation_reason": (
                                "verified_stage_bootstrap_awaiting_independent_support"
                                if deterministic_bootstrap
                                else (
                                    "verified_repair_awaiting_independent_support"
                                    if repair_verified and not stage_passed
                                    else "stage_pass_awaiting_independent_support"
                                )
                            ),
                        }
                    )
                    decision = "persisted_candidate"
                record = self._rebuild_embedding(record)
                filtered.append(op.model_copy(update={"content": record.model_dump()}))
                self._last_skill_decisions.append(
                    {
                        "skill_name": record.skill_name,
                        "stage": record.stage,
                        "semantic_signature": record.semantic_signature,
                        "source": record.source,
                        "decision": decision,
                        "status": record.status,
                        "independent_support_count": record.independent_support_count,
                        "activation_min_independent_support": (
                            record.activation_min_independent_support
                        ),
                        "reasons": [],
                    }
                )
        return filtered

    @staticmethod
    def _dedupe_candidate_ops(ops: list[MemoryUpdateOp]) -> list[MemoryUpdateOp]:
        """Keep one same-scene Skill per deterministic semantic identity.

        LLM operations are placed before bootstrap operations, so a grounded
        model-authored procedure wins when both encode the same semantics.  A
        deterministic draft still covers every omitted relation/coverage scope.
        """
        output: list[MemoryUpdateOp] = []
        seen_skill_signatures: set[str] = set()
        for op in ops:
            if op.memory_type != "skill":
                output.append(op)
                continue
            signature = str(op.content.get("semantic_signature") or "")
            if not signature:
                signature = hashlib.sha256(
                    json.dumps(
                        op.content,
                        ensure_ascii=False,
                        sort_keys=True,
                        default=str,
                    ).encode("utf-8")
                ).hexdigest()
            if signature in seen_skill_signatures:
                continue
            seen_skill_signatures.add(signature)
            output.append(op)
        return output

    @staticmethod
    def _op_counts(ops: list[MemoryUpdateOp]) -> dict[str, int]:
        """Return stable per-type counts for writer observability."""
        counts = {"success_case": 0, "failure_case": 0, "skill": 0}
        for op in ops:
            if op.op in {"ADD", "UPDATE"} and op.memory_type in counts:
                counts[op.memory_type] += 1
        return counts

    def _validate_success(self, content: dict[str, Any]) -> SuccessCase | None:
        try:
            record = SuccessCase.model_validate(content)
        except Exception as exc:
            console_logger.info("MemoryWriter: invalid success record: %s", exc)
            return None
        if record.stage not in _SUPPORTED_STAGES or not record.successful_pattern:
            return None
        return record

    def _validate_failure(self, content: dict[str, Any]) -> FailureCase | None:
        try:
            record = FailureCase.model_validate(content)
        except Exception as exc:
            console_logger.info("MemoryWriter: invalid failure record: %s", exc)
            return None
        if record.stage not in _SUPPORTED_STAGES or not record.bad_pattern:
            return None
        return record

    def _validate_skill(self, content: dict[str, Any]) -> Skill | None:
        try:
            record = Skill.model_validate(content)
        except Exception as exc:
            console_logger.info("MemoryWriter: invalid skill record: %s", exc)
            return None
        if record.stage not in _SUPPORTED_STAGES:
            return None
        return record

    @staticmethod
    def _rebuild_embedding(
        record: SuccessCase | FailureCase | Skill,
    ) -> SuccessCase | FailureCase | Skill:
        """Build retrieval text only after all authoritative enrichment."""
        return record.model_copy(
            update={"embedding_text": build_embedding_text(record)}
        )

    def _provenance(
        self,
        context: dict[str, Any],
        stage: str,
    ) -> MemorySourceProvenance:
        stage_evidence = self._stage_evidence(context, stage)
        report = self._stage_report(stage_evidence)
        return MemorySourceProvenance(
            task_id=context.get("source_task_id", ""),
            run_id=context.get("source_run_id", ""),
            trace_id=context.get("trace_id", ""),
            scene_state_path=str(stage_evidence.get("scene_state_path") or ""),
            stage=stage,
            prompt_fingerprint=context.get("prompt_fingerprint", ""),
            evidence_refs=self._evidence_refs(context, stage),
            critic_source=str(report.get("score_source") or "unknown"),
        )

    @staticmethod
    def _selector_label(value: Any) -> str:
        if isinstance(value, str):
            return " ".join(value.split())
        if not isinstance(value, dict):
            return ""
        parts = [
            str(value.get(key) or "").strip()
            for key in ("role", "category", "object", "name", "id")
        ]
        return ":".join(part for part in parts if part)

    def _spatial_relations(
        self,
        stage_evidence: dict[str, Any],
        focus_terms: list[str] | None = None,
        constraint_ids: set[str] | None = None,
    ) -> list[SpatialRelationMemory]:
        """Extract only relations already grounded in the intent/critic trace."""
        context = dict(stage_evidence.get("relation_context") or {})
        output: list[SpatialRelationMemory] = []
        for constraint in context.get("hard_constraints", []) or []:
            if not isinstance(constraint, dict):
                continue
            constraint_id = str(constraint.get("constraint_id") or "")
            if constraint_ids is not None and constraint_id not in constraint_ids:
                continue
            if focus_terms:
                haystack = (
                    json.dumps(
                        constraint, ensure_ascii=False, sort_keys=True, default=str
                    )
                    .casefold()
                    .replace("_", " ")
                )
                focus_tokens = {
                    token
                    for value in focus_terms
                    for token in re.findall(
                        r"[a-z0-9\u4e00-\u9fff]+",
                        str(value or "").casefold().replace("_", " "),
                    )
                    if len(token) >= 3
                }
                if focus_tokens and not any(
                    token in haystack for token in focus_tokens
                ):
                    continue
            relation_type = str(
                constraint.get("relation_type")
                or constraint.get("relation")
                or constraint.get("predicate")
                or constraint.get("type")
                or ""
            ).strip()
            if not relation_type:
                continue
            subject = (
                constraint.get("subject")
                or constraint.get("subjects")
                or constraint.get("subject_selector")
                or constraint.get("source")
            )
            target = (
                constraint.get("target")
                or constraint.get("targets")
                or constraint.get("target_selector")
                or constraint.get("reference")
            )
            cardinality = {
                key: value
                for key in ("count", "min_count", "max_count", "quantifier")
                if (value := constraint.get(key)) is not None
            }
            for key in ("orientation", "edge_frame", "groups"):
                value = constraint.get(key)
                if value is not None:
                    cardinality[key] = value
            for prefix, selector in (("subject", subject), ("target", target)):
                if not isinstance(selector, dict):
                    continue
                for key in ("count", "min_count", "max_count", "quantifier"):
                    value = selector.get(key)
                    if value is not None:
                        cardinality[f"{prefix}_{key}"] = value
            status, observations = resolve_constraint_evidence(
                constraint, stage_evidence
            )
            verified = status == "verified_pass"
            output.append(
                SpatialRelationMemory(
                    relation_type=relation_type,
                    subject_role=self._selector_label(subject),
                    target_role=self._selector_label(target),
                    cardinality=cardinality,
                    evidence_source=(
                        "deterministic" if observations else "task_contract"
                    ),
                    evidence_ref=constraint_id,
                    geometry_verified=verified,
                    confidence=0.85 if verified else 0.5,
                    verification_status=status,
                    verification_evidence=observations,
                )
            )
        return [
            relation.model_copy(update={"claim_hash": relation.current_claim_hash()})
            for relation in output
        ]

    def _relation_types(self, stage_evidence: dict[str, Any]) -> list[str]:
        return self._unique(
            [
                relation.relation_type
                for relation in self._spatial_relations(stage_evidence)
            ]
        )

    def _canonical_context(
        self,
        evidence_payload: dict[str, Any],
        trace_summary: str,
    ) -> dict[str, Any]:
        task_spec = dict(evidence_payload.get("task_spec") or {})
        prompt = str(
            evidence_payload.get("prompt") or self._extract_prompt(trace_summary)
        )
        prompt_fingerprint = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        trace_id = str(
            evidence_payload.get("trace_id") or self._extract_trace_id(trace_summary)
        )
        stage_paths = [
            str(item.get("scene_state_path", ""))
            for item in evidence_payload.get("stages", []) or []
            if isinstance(item, dict) and item.get("scene_state_path")
        ]
        run_locator = str(
            evidence_payload.get("run_id")
            or evidence_payload.get("output_dir")
            or "|".join(stage_paths)
        )
        run_fingerprint = hashlib.sha256(
            "|".join(
                [
                    run_locator,
                    trace_id,
                    prompt,
                    str(evidence_payload.get("config_hash", "")),
                ]
            ).encode("utf-8")
        ).hexdigest()
        return {
            **evidence_payload,
            "task_spec": task_spec,
            "prompt": prompt,
            "prompt_fingerprint": prompt_fingerprint,
            "source_task_id": f"task_{prompt_fingerprint[:16]}",
            "source_run_id": f"run_{run_fingerprint[:20]}",
            "trace_id": trace_id,
            "room_type": str(task_spec.get("room_type") or "room"),
            "style": str(task_spec.get("style") or ""),
            "functional_zones": self._clean_list(task_spec.get("functional_zones", [])),
        }

    def _record_id(self, prefix: str, candidate: Any, context: dict[str, Any]) -> str:
        payload = {
            "prefix": prefix,
            "source_run_id": context["source_run_id"],
            "prompt_fingerprint": context["prompt_fingerprint"],
            "candidate": candidate.model_dump(),
        }
        digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest()[:16]
        stage = str(candidate.stage).strip().lower()
        return f"{prefix}_{stage}_{digest}"

    @staticmethod
    def _required_objects(task_spec: dict[str, Any], stage: str) -> list[str]:
        key = {
            "floor_plan": "required_large_objects",
            "furniture": "required_large_objects",
            "wall_mounted": "required_wall_objects",
            "ceiling_mounted": "required_ceiling_objects",
            "manipuland": "required_small_objects",
        }.get(stage, "")
        value = task_spec.get(key, []) if key else []
        return MemoryWriter._clean_list(value)

    @staticmethod
    def _stage_evidence(payload: dict[str, Any], stage: str) -> dict[str, Any]:
        for item in payload.get("stages", []) or []:
            if isinstance(item, dict) and str(item.get("stage")) == stage:
                return item
        return {}

    @staticmethod
    def _stage_report(stage_evidence: dict[str, Any]) -> dict[str, Any]:
        report = stage_evidence.get("verify_report") or {}
        return report if isinstance(report, dict) else {}

    def _stage_passed(self, stage_evidence: dict[str, Any]) -> bool:
        return bool(self._stage_report(stage_evidence).get("pass_stage", False))

    def _stage_scores(self, stage_evidence: dict[str, Any]) -> dict[str, float]:
        report = self._stage_report(stage_evidence)
        raw = report.get("visual_scores") or report.get("scores") or {}
        if not isinstance(raw, dict):
            return {}
        return {
            str(key): float(value)
            for key, value in raw.items()
            if isinstance(value, (int, float))
        }

    @staticmethod
    def _mean_score(scores: dict[str, float]) -> float:
        """Return a conservative stage quality when no scene score is valid."""
        values = [float(value) for value in scores.values()]
        if not values:
            return 0.5
        return max(0.0, min(1.0, sum(values) / len(values)))

    def _repair_verified(self, stage_evidence: dict[str, Any]) -> bool:
        repairs = stage_evidence.get("repair_actions") or []
        return any(
            isinstance(item, dict) and bool(item.get("repair_verified"))
            for item in repairs
        )

    def _deterministic_failure_in_evidence(
        self, stage_evidence: dict[str, Any]
    ) -> bool:
        report = self._stage_report(stage_evidence)
        hard_report = report.get("hard_check_report") or {}
        if isinstance(hard_report, dict) and hard_report:
            if (
                hard_report.get("hard_valid") is False
                or hard_report.get("pass") is False
            ):
                return True
            if hard_report.get("failed_checks") or hard_report.get("hard_failures"):
                return True
        # Free-form critique often contains negated phrases such as "no
        # deterministic hard failure". Only structured issues and failed hard
        # checks may certify a deterministic failure.
        evidence_text = json.dumps(
            report.get("issues", []), ensure_ascii=False, default=str
        ).lower()
        return any(
            keyword in evidence_text for keyword in _DETERMINISTIC_FAILURE_KEYWORDS
        )

    def _detect_deterministic_failure(self, content: dict[str, Any]) -> bool:
        text = " ".join(
            str(content.get(key, ""))
            for key in (
                "failure_type",
                "bad_pattern",
                "failure_reason",
                "repair_action",
                "negative_constraint",
                "critic_check",
            )
        ).lower()
        return bool(content.get("is_deterministic")) or any(
            keyword in text for keyword in _DETERMINISTIC_FAILURE_KEYWORDS
        )

    def _critic_evidence(self, stage_evidence: dict[str, Any]) -> list[str]:
        report = self._stage_report(stage_evidence)
        values = [
            f"score_source={report.get('score_source', 'unknown')}",
            self._compact_text(report.get("critique_summary", ""), 1600),
        ]
        for issue in report.get("issues", []) or []:
            if isinstance(issue, dict):
                values.append(
                    self._compact_text(
                        issue.get("description") or issue.get("issue_type") or "", 500
                    )
                )
        return self._unique(values)

    def _evidence_refs(self, context: dict[str, Any], stage: str) -> list[str]:
        stage_evidence = self._stage_evidence(context, stage)
        return self._unique(
            [
                context.get("source_run_id", ""),
                context.get("trace_id", ""),
                stage_evidence.get("scene_state_path", ""),
            ]
        )

    def _evidence_confidence(self, stage_evidence: dict[str, Any]) -> float:
        report = self._stage_report(stage_evidence)
        if report.get("critique_summary") and report.get("score_source") not in {
            "",
            "unknown",
        }:
            return 0.85
        return 0.65

    def _build_user_message(
        self,
        *,
        trace_summary: str,
        full_report: FullVerifyReport,
        related_old_memory: str,
        evidence_payload: dict[str, Any],
    ) -> str:
        message, metadata = build_writer_prompt(
            evidence=evidence_payload,
            final_report=full_report.model_dump(),
            trace_summary=trace_summary,
            related_old_memory=related_old_memory,
            max_user_bytes=getattr(
                self, "_active_user_budget", self._user_budget(self._context_tokens)
            ),
        )
        self._visible_episode_ids = (
            set(metadata["selected_episode_ids"]) if metadata["has_catalog"] else None
        )
        if not hasattr(self, "_prompt_attempts"):
            self._prompt_attempts = []
        self._prompt_attempts.append(metadata)
        if self._debug_dir is not None:
            self._debug_dir.mkdir(parents=True, exist_ok=True)
            self._atomic_write_json(
                self._debug_dir
                / f"memory_writer_prompt_{len(self._prompt_attempts):02d}.json",
                {
                    "projection": metadata,
                    "messages": [
                        {"role": "system", "content": _SYSTEM_PROMPT},
                        {"role": "user", "content": message},
                    ],
                },
            )
        return message

    def _user_budget(self, context_tokens: int) -> int:
        """Reserve schema, output and retry/framing space within a byte envelope."""
        reserve = (
            max(self._profile.max_tokens, self._profile.retry_max_tokens or 0)
            + byte_size(MemoryWriterResponse.model_json_schema())
            + len(_SYSTEM_PROMPT.encode("utf-8"))
            + 4096
        )
        return min(self._max_input_bytes, context_tokens - reserve)

    def record_store_result(self, summary: dict[str, Any]) -> None:
        """Confirm actual store mutations separately from model proposals."""
        self.last_trace.update(
            {
                "store_apply": dict(summary),
                "persistence_confirmed": True,
                "persisted_count": sum(
                    int(summary.get(k, 0)) for k in ("added", "updated", "merged")
                ),
                "promoted_count": int(summary.get("active_records_changed", 0)),
                "persisted_counts": dict(summary.get("changed_counts", {})),
                "promoted_counts": dict(summary.get("active_changed_counts", {})),
                "count_semantics": "persisted/promoted counts confirmed by store; proposed counts are writer output",
            }
        )
        if self._debug_dir is not None:
            path = self._debug_dir / "memory_writer_debug.json"
            try:
                if path.is_file():
                    payload = json.loads(path.read_text(encoding="utf-8"))
                    payload["result_status"] = self.last_trace
                    self._atomic_write_json(path, payload)
            except (OSError, ValueError) as exc:
                self.last_trace["store_audit_error"] = str(exc)
                console_logger.warning(
                    "Memory store succeeded but debug refresh failed: %s", exc
                )

    def _save_debug_payload(
        self,
        *,
        status: str,
        result_status: dict[str, Any],
        trace_summary: str,
        full_report: FullVerifyReport,
        evidence_payload: dict[str, Any],
        response: MemoryWriterResponse | None,
        result_ops: list[MemoryUpdateOp],
    ) -> None:
        if self._debug_dir is None:
            return
        self._debug_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": "sceneexpert.memory_writer_debug.v2",
            "created_at": self._now(),
            "status": status,
            "model": self._model,
            "success_min_overall_score": self._success_min_overall_score,
            "result_status": result_status,
            "full_report": full_report.model_dump(),
            "trace_summary_excerpt": self._compact_text(trace_summary, 6000),
            "evidence_trace_id": evidence_payload.get("trace_id", ""),
            "response": response.model_dump() if response is not None else None,
            "result_ops": [op.model_dump() for op in result_ops],
            "fallback_written": False,
        }
        self._atomic_write_json(self._debug_dir / "memory_writer_debug.json", payload)
        with (self._debug_dir / "memory_writer_debug.jsonl").open(
            "a", encoding="utf-8", newline="\n"
        ) as file:
            file.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")

    def _fallback_success_ops(
        self, trace_summary: str, full_report: FullVerifyReport
    ) -> list[MemoryUpdateOp]:
        """Compatibility shim: fallback records are intentionally never persisted."""
        del trace_summary, full_report
        return []

    @staticmethod
    def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
        temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
        temporary.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
        temporary.replace(path)

    @staticmethod
    def _extract_trace_id(trace_summary: str) -> str:
        match = re.search(r"^Trace:\s*(\S+)", trace_summary, flags=re.MULTILINE)
        return match.group(1) if match else "trace_unknown"

    @staticmethod
    def _extract_prompt(trace_summary: str) -> str:
        match = re.search(r"^Prompt:\s*(.+)$", trace_summary, flags=re.MULTILINE)
        return match.group(1).strip() if match else ""

    @staticmethod
    def _clean_list(values: Any) -> list[str]:
        if not isinstance(values, (list, tuple, set)):
            return []
        return MemoryWriter._unique(str(value).strip() for value in values)

    @staticmethod
    def _unique(values: Any) -> list[str]:
        output: list[str] = []
        seen: set[str] = set()
        for value in values:
            text = str(value or "").strip()
            key = text.casefold()
            if text and key not in seen:
                output.append(text)
                seen.add(key)
        return output

    @staticmethod
    def _compact_text(value: Any, max_chars: int) -> str:
        text = str(value or "")
        return text if len(text) <= max_chars else text[: max_chars - 3] + "..."

    @staticmethod
    def _env_bool(name: str, default: bool) -> bool:
        value = os.environ.get(name)
        if value is None:
            return bool(default)
        return str(value).strip().casefold() not in {"0", "false", "no", "off", ""}

    @staticmethod
    def _now() -> str:
        return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
