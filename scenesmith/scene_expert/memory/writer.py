"""Strict, evidence-gated long-term memory writer for SceneExpert.

The LLM only proposes compact lessons. Deterministic code owns identity, task
metadata, critic evidence, quality gates, provenance, and promotion into the
active memory bank. A failed or empty LLM response is a no-write outcome; it
can never manufacture a retrievable fallback record.
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

from scenesmith.scene_expert.memory.schemas import (
    FailureCase,
    FailureMemoryCandidate,
    MemoryUpdateOp,
    MemoryWriterResponse,
    Skill,
    SkillMemoryCandidate,
    SuccessCase,
    SuccessMemoryCandidate,
)
from scenesmith.scene_expert.memory.text_builder import build_embedding_text
from scenesmith.scene_expert.schemas import FullVerifyReport
from scenesmith.scene_expert.structured_llm import (
    SceneExpertStructuredLLMClient,
    StructuredLLMProfile,
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
- A failure lesson is allowed only when the trace shows a verified repair or a
  deterministic/repeatable hard failure. Never label visual opinion as deterministic.
- A skill must contain a reusable procedure with at least two concrete steps.
- Prefer empty arrays with a clear noop_reason over weak, duplicate, or speculative memory.
- Keep each lesson concise and useful for a different scene with similar requirements.
"""


class MemoryWriter:
    """Generate typed memory candidates and promote only evidence-backed records."""

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
        debug_dir: str | Path | None = None,
        llm_client: SceneExpertStructuredLLMClient | None = None,
    ) -> None:
        self._model = model
        self._debug_dir = Path(debug_dir) if debug_dir else None
        self._success_min_overall_score = float(success_min_overall_score)
        max_tokens = int(
            os.environ.get("SCENEEXPERT_MEMORY_WRITER_MAX_TOKENS", max_tokens)
        )
        retry_tokens = int(
            os.environ.get(
                "SCENEEXPERT_MEMORY_WRITER_RETRY_MAX_TOKENS",
                retry_max_tokens if retry_max_tokens is not None else max(max_tokens, 4096),
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
        user_message = self._build_user_message(
            trace_summary=trace_summary,
            full_report=full_report,
            related_old_memory=related_old_memory,
            evidence_payload=evidence,
        )
        result = self._llm_client.complete(
            role="memory_writer",
            stage="full_scene",
            event="write_long_term_memory",
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            response_model=MemoryWriterResponse,
            profile=self._profile,
        )
        self.last_trace = result.status_dict()

        if not result.success or result.value is None:
            self.last_trace.update(
                {
                    "structured_call_source": self.last_trace.get("source", ""),
                    "source": "no_write",
                    "write_status": "model_failure_no_write",
                    "promoted_count": 0,
                    "fallback_written": False,
                }
            )
            self._save_debug_payload(
                status="model_failure_no_write",
                result_status=self.last_trace,
                trace_summary=trace_summary,
                full_report=full_report,
                evidence_payload=evidence,
                response=None,
                result_ops=[],
            )
            console_logger.warning(
                "MemoryWriter structured output failed after %d attempts; "
                "the active memory bank was not modified: %s",
                len(result.attempts),
                result.final_error or result.final_error_kind,
            )
            return []

        candidate_ops = self._response_to_ops(
            response=result.value,
            trace_summary=trace_summary,
            full_report=full_report,
            evidence_payload=evidence,
        )
        promoted_ops = self._gate_and_enrich_ops(
            candidate_ops,
            full_report,
            evidence_payload=evidence,
        )
        mutating_ops = [op for op in promoted_ops if op.op in {"ADD", "UPDATE"}]
        status = "promoted" if mutating_ops else "no_valid_candidates"
        self.last_trace.update(
            {
                "write_status": status,
                "candidate_count": len(candidate_ops),
                "promoted_count": len(mutating_ops),
                "noop_reason": result.value.noop_reason,
                "fallback_written": False,
            }
        )
        self._save_debug_payload(
            status=status,
            result_status=self.last_trace,
            trace_summary=trace_summary,
            full_report=full_report,
            evidence_payload=evidence,
            response=result.value,
            result_ops=mutating_ops,
        )
        console_logger.info(
            "MemoryWriter: promoted %d/%d schema-valid candidates; fallback_written=false",
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
        for candidate in response.success_cases:
            content = self._success_content(candidate, context, full_report)
            ops.append(
                MemoryUpdateOp(op="ADD", memory_type="success_case", content=content)
            )
        for candidate in response.failure_cases:
            content = self._failure_content(candidate, context)
            ops.append(
                MemoryUpdateOp(op="ADD", memory_type="failure_case", content=content)
            )
        for candidate in response.skills:
            content = self._skill_content(candidate, context, full_report)
            ops.append(MemoryUpdateOp(op="ADD", memory_type="skill", content=content))
        return ops

    def _success_content(
        self,
        candidate: SuccessMemoryCandidate,
        context: dict[str, Any],
        full_report: FullVerifyReport,
    ) -> dict[str, Any]:
        stage_evidence = self._stage_evidence(context, candidate.stage)
        required_objects = self._required_objects(context["task_spec"], candidate.stage)
        scores = self._stage_scores(stage_evidence)
        now = self._now()
        record = SuccessCase(
            case_id=self._record_id("success", candidate, context),
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
            scene_summary=f"Evidence-backed {candidate.stage} lesson from {context['trace_id']}.",
            confidence=self._evidence_confidence(stage_evidence),
            quality_score=float(full_report.overall_score),
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
        )
        return record.model_dump()

    def _skill_content(
        self,
        candidate: SkillMemoryCandidate,
        context: dict[str, Any],
        full_report: FullVerifyReport,
    ) -> dict[str, Any]:
        stage_evidence = self._stage_evidence(context, candidate.stage)
        required_objects = self._required_objects(context["task_spec"], candidate.stage)
        now = self._now()
        record = Skill(
            skill_name=candidate.skill_name,
            stage=candidate.stage,
            room_type=context["room_type"],
            room_types=[context["room_type"]] if context["room_type"] else [],
            style=context["style"],
            required_objects=required_objects,
            functional_zones=context["functional_zones"],
            scene_summary=f"Evidence-backed procedure from {context['trace_id']}.",
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
            status="active",
            source="llm",
            source_task_id=context["source_task_id"],
            source_run_id=context["source_run_id"],
            source_task_ids=[context["source_task_id"]],
            source_run_ids=[context["source_run_id"]],
            prompt_fingerprint=context["prompt_fingerprint"],
            evidence_refs=self._evidence_refs(context, candidate.stage),
            critic_evidence=self._critic_evidence(stage_evidence),
        )
        return record.model_dump()

    def _gate_and_enrich_ops(
        self,
        ops: list[MemoryUpdateOp],
        full_report: FullVerifyReport,
        evidence_payload: dict[str, Any] | None = None,
    ) -> list[MemoryUpdateOp]:
        """Validate persisted records and enforce deterministic promotion gates."""
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
                    continue
                stage_evidence = self._stage_evidence(evidence, record.stage)
                if (
                    not full_report.pass_scene
                    or full_report.overall_score < success_threshold
                    or (has_structured_evidence and not self._stage_passed(stage_evidence))
                ):
                    console_logger.info(
                        "MemoryWriter: rejected success %s because final/stage "
                        "evidence did not pass",
                        record.case_id,
                    )
                    continue
                filtered.append(op.model_copy(update={"content": record.model_dump()}))
                continue

            if op.memory_type == "failure_case":
                record = self._validate_failure(op.content)
                if record is None:
                    continue
                stage_evidence = self._stage_evidence(evidence, record.stage)
                if has_structured_evidence:
                    repair_verified = self._repair_verified(stage_evidence)
                    deterministic = self._deterministic_failure_in_evidence(stage_evidence)
                else:
                    repair_verified = record.repair_verified
                    deterministic = self._detect_deterministic_failure(record.model_dump())
                if not repair_verified and not deterministic:
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
                            if deterministic and record.scope == "object"
                            else record.scope
                        ),
                        "confidence": 0.85,
                    }
                )
                filtered.append(op.model_copy(update={"content": record.model_dump()}))
                continue

            if op.memory_type == "skill":
                record = self._validate_skill(op.content)
                if record is None:
                    continue
                stage_evidence = self._stage_evidence(evidence, record.stage)
                if (
                    not full_report.pass_scene
                    or full_report.overall_score < success_threshold
                    or (has_structured_evidence and not self._stage_passed(stage_evidence))
                    or len(self._clean_list(record.procedure)) < 2
                ):
                    console_logger.info(
                        "MemoryWriter: rejected unsupported skill %s", record.skill_name
                    )
                    continue
                filtered.append(op.model_copy(update={"content": record.model_dump()}))
        return filtered

    def _validate_success(self, content: dict[str, Any]) -> SuccessCase | None:
        try:
            record = SuccessCase.model_validate(content)
        except Exception as exc:
            console_logger.info("MemoryWriter: invalid success record: %s", exc)
            return None
        if record.stage not in _SUPPORTED_STAGES or not record.successful_pattern:
            return None
        if not record.embedding_text:
            record = record.model_copy(update={"embedding_text": build_embedding_text(record)})
        return record

    def _validate_failure(self, content: dict[str, Any]) -> FailureCase | None:
        try:
            record = FailureCase.model_validate(content)
        except Exception as exc:
            console_logger.info("MemoryWriter: invalid failure record: %s", exc)
            return None
        if record.stage not in _SUPPORTED_STAGES or not record.bad_pattern:
            return None
        if not record.embedding_text:
            record = record.model_copy(update={"embedding_text": build_embedding_text(record)})
        return record

    def _validate_skill(self, content: dict[str, Any]) -> Skill | None:
        try:
            record = Skill.model_validate(content)
        except Exception as exc:
            console_logger.info("MemoryWriter: invalid skill record: %s", exc)
            return None
        if record.stage not in _SUPPORTED_STAGES:
            return None
        if not record.embedding_text:
            record = record.model_copy(update={"embedding_text": build_embedding_text(record)})
        return record

    def _canonical_context(
        self,
        evidence_payload: dict[str, Any],
        trace_summary: str,
    ) -> dict[str, Any]:
        task_spec = dict(evidence_payload.get("task_spec") or {})
        prompt = str(evidence_payload.get("prompt") or self._extract_prompt(trace_summary))
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
            if hard_report.get("hard_valid") is False or hard_report.get("pass") is False:
                return True
            if hard_report.get("failed_checks") or hard_report.get("hard_failures"):
                return True
        # Free-form critique often contains negated phrases such as "no
        # deterministic hard failure". Only structured issues and failed hard
        # checks may certify a deterministic failure.
        evidence_text = json.dumps(
            report.get("issues", []), ensure_ascii=False, default=str
        ).lower()
        return any(keyword in evidence_text for keyword in _DETERMINISTIC_FAILURE_KEYWORDS)

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
        payload = {
            "trace_summary": trace_summary,
            "evidence": evidence_payload,
            "final_report": full_report.model_dump(),
            "related_existing_memory": related_old_memory,
        }
        return (
            "Analyze this completed run. Treat evidence.verify_report and final_report "
            "as authoritative. Return only schema-valid reusable candidates.\n"
            + json.dumps(payload, ensure_ascii=False, default=str)
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
    def _now() -> str:
        return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
