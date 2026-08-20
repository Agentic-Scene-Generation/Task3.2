"""Read-only capture of SceneSmith trajectories for offline preference learning."""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time

from pathlib import Path
from typing import Any, Iterable

from scenesmith.scene_expert.schemas import (
    RepairResult,
    SceneTaskSpec,
    StageVerifyReport,
)
from scenesmith.scene_expert.slow_memory.schemas import (
    PreferenceEvidence,
    TrajectoryRecord,
)

console_logger = logging.getLogger(__name__)

_DESIGNER_EVENTS = frozenset({"request_initial_design", "request_design_change"})
_SECRET_PATTERNS = (
    re.compile(r"\bsk-[A-Za-z0-9_-]{8,}\b"),
    re.compile(r"(?i)(bearer\s+)[A-Za-z0-9._~+/-]{12,}"),
    re.compile(r"(?i)((?:api[_-]?key|hf_token|token)\s*[=:]\s*)[^\s,;]+"),
)


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _stable_hash(value: Any, length: int = 24) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:length]


def _stringify(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def redact_sensitive_text(value: Any) -> str:
    """Remove common credential forms before a payload becomes training data."""
    text = _stringify(value)
    for pattern in _SECRET_PATTERNS:
        if pattern.groups:
            text = pattern.sub(lambda match: match.group(1) + "[REDACTED]", text)
        else:
            text = pattern.sub("[REDACTED]", text)
    return text


def _bounded_text(value: Any, max_chars: int) -> tuple[str, bool]:
    text = redact_sensitive_text(value)
    if len(text) <= max_chars:
        return text, True
    return text[:max_chars], False


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _load_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return records
    for line in lines:
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            records.append(payload)
    return records


def _mean_score(values: dict[str, Any]) -> float | None:
    numbers = [
        float(value) for value in values.values() if isinstance(value, (int, float))
    ]
    return sum(numbers) / len(numbers) if numbers else None


class TrajectoryCollector:
    """Capture evidence after main has already made and persisted its decisions.

    The collector never calls a model, changes a scene, or chooses a checkpoint.
    It reads the existing SceneSmith audit payloads after a stage has finished and
    writes a separate, append-only slow-memory observation stream.
    """

    def __init__(
        self,
        *,
        scene_debug_dir: Path,
        prompt: str,
        scene_id: str,
        run_id: str,
        task_spec: SceneTaskSpec,
        experiment_signature: str = "",
        config_hash: str = "",
        model_id: str = "",
        max_prompt_chars: int = 131072,
        max_response_chars: int = 65536,
    ) -> None:
        self.scene_debug_dir = Path(scene_debug_dir)
        self.output_dir = self.scene_debug_dir / "slow_memory"
        self.trajectory_path = self.output_dir / "trajectories.jsonl"
        self.manifest_path = self.output_dir / "capture_manifest.json"
        self.prompt = str(prompt)
        self.scene_id = str(scene_id)
        self.run_id = str(run_id)
        self.task_id = "task_" + _stable_hash(" ".join(self.prompt.split()), 16)
        self.task_spec = task_spec
        self.experiment_signature = experiment_signature
        self.config_hash = config_hash
        self.model_id = model_id
        self.max_prompt_chars = max(1024, int(max_prompt_chars))
        self.max_response_chars = max(1024, int(max_response_chars))
        self._seen_ids = {
            str(record.get("trajectory_id"))
            for record in _load_jsonl(self.trajectory_path)
            if record.get("trajectory_id")
        }
        self._counts = {"designer": 0, "repair": 0, "unlabeled": 0}

    def _relative_ref(self, path: Path) -> str:
        try:
            return path.resolve().relative_to(self.scene_debug_dir.resolve()).as_posix()
        except ValueError:
            return path.name

    def _append(self, record: TrajectoryRecord) -> bool:
        if record.trajectory_id in self._seen_ids:
            return False
        self.output_dir.mkdir(parents=True, exist_ok=True)
        with self.trajectory_path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(record.model_dump_json() + "\n")
        self._seen_ids.add(record.trajectory_id)
        self._counts[record.agent_role] += 1
        if record.evidence.verdict == "unlabeled":
            self._counts["unlabeled"] += 1
        return True

    def _stage_evidence(
        self,
        stage: str,
        report: StageVerifyReport,
        report_ref: str,
    ) -> PreferenceEvidence:
        has_critic = bool(
            report.vlm_scoring_performed
            and report.score_source == "scenebenchmark_critic"
        )
        kind = "critic_and_deterministic" if has_critic else "deterministic"
        visual_score = _mean_score(report.visual_scores or report.scores)
        deterministic_score = _mean_score(report.rule_scores)
        base_score = visual_score if visual_score is not None else deterministic_score
        # Deterministic pass/fail is the primary rank; critic score is a
        # secondary discriminator and cannot promote a hard-invalid response.
        quality_score = (1.0 if report.pass_stage else 0.0) + float(base_score or 0.0)
        evidence_payload = {
            "stage": stage,
            "pass_stage": report.pass_stage,
            "score_source": report.score_source,
            "visual_scores": report.visual_scores,
            "rule_scores": report.rule_scores,
            "issues": [issue.model_dump() for issue in report.issues],
            "hard_check_report": report.hard_check_report,
        }
        return PreferenceEvidence(
            evidence_id="evidence_" + _stable_hash(evidence_payload),
            kind=kind,
            verdict="accepted" if report.pass_stage else "rejected",
            source=(
                "main_scenebenchmark_critic_plus_stage_rules"
                if has_critic
                else "deterministic_stage_verifier"
            ),
            authoritative=True,
            quality_score=quality_score,
            report_ref=report_ref,
            details=evidence_payload,
        )

    def _designer_payloads(self, stage: str) -> list[tuple[Path, dict[str, Any]]]:
        payload_dir = self.scene_debug_dir / "audit" / "llm_payloads"
        records: list[tuple[Path, dict[str, Any]]] = []
        for path in sorted(payload_dir.glob("*.json")):
            payload = _load_json(path)
            if not payload:
                continue
            if payload.get("stage") != stage or payload.get("agent_role") != "designer":
                continue
            if payload.get("event") not in _DESIGNER_EVENTS or payload.get("error"):
                continue
            if not str(payload.get("output") or "").strip():
                continue
            records.append((path, payload))
        return records

    def capture_stage(
        self,
        *,
        stage: str,
        verify_report: StageVerifyReport | None,
        repair_actions: list[RepairResult] | None = None,
    ) -> dict[str, int]:
        """Capture newly persisted designer and repair observations for a stage."""
        added = {"designer": 0, "repair": 0, "unlabeled": 0}
        payloads = self._designer_payloads(stage)
        final_index = len(payloads) - 1
        report_ref = ""
        if verify_report is not None:
            report_payload = verify_report.model_dump(mode="json")
            evidence_path = (
                self.output_dir
                / "evidence"
                / f"{stage}_{_stable_hash(report_payload, 16)}.json"
            )
            evidence_path.parent.mkdir(parents=True, exist_ok=True)
            evidence_path.write_text(
                json.dumps(report_payload, indent=2, ensure_ascii=False),
                encoding="utf-8",
                newline="\n",
            )
            report_ref = self._relative_ref(evidence_path)
        stage_evidence = (
            self._stage_evidence(stage, verify_report, report_ref)
            if verify_report is not None
            else None
        )
        for index, (path, payload) in enumerate(payloads):
            prompt, prompt_complete = _bounded_text(
                payload.get("prompt"), self.max_prompt_chars
            )
            response, response_complete = _bounded_text(
                payload.get("output"), self.max_response_chars
            )
            payload_ref = self._relative_ref(path)
            trajectory_id = "trajectory_" + _stable_hash(
                [self.run_id, self.scene_id, payload_ref, prompt, response]
            )
            if index == final_index and stage_evidence is not None:
                evidence = stage_evidence.model_copy(deep=True)
            else:
                evidence = PreferenceEvidence(
                    evidence_id="evidence_"
                    + _stable_hash([trajectory_id, "unlabeled"]),
                    source="insufficient_candidate_level_evidence",
                    verdict="unlabeled",
                    kind="none",
                    authoritative=False,
                    report_ref=report_ref if verify_report is not None else "",
                )
            record = TrajectoryRecord(
                trajectory_id=trajectory_id,
                created_at=str(payload.get("created_at") or _utc_now()),
                run_id=self.run_id,
                scene_id=self.scene_id,
                task_id=self.task_id,
                experiment_signature=self.experiment_signature,
                config_hash=self.config_hash,
                model_id=self.model_id,
                stage=stage,
                agent_role="designer",
                event=str(payload.get("event") or "designer"),
                context_hash=_stable_hash(prompt, 64),
                prompt=prompt,
                response=response,
                response_hash=_stable_hash(response, 64),
                prompt_complete=prompt_complete,
                response_complete=response_complete,
                evidence=evidence,
                source_refs=[ref for ref in (payload_ref, report_ref) if ref],
                metadata={
                    "task_spec": self.task_spec.model_dump(mode="json"),
                    "capture_policy": (
                        "only_last_designer_call_receives_stage_outcome"
                    ),
                },
            )
            if self._append(record):
                added["designer"] += 1
                if evidence.verdict == "unlabeled":
                    added["unlabeled"] += 1

        for event in _load_jsonl(
            self.scene_debug_dir / "timing" / "repair_events.jsonl"
        ):
            if event.get("stage") != stage:
                continue
            self._capture_repair_event(event, added)

        for action in repair_actions or []:
            self._capture_scene_expert_repair(stage, action, added)

        self._write_manifest()
        return added

    def _capture_repair_event(
        self, event: dict[str, Any], added: dict[str, int]
    ) -> None:
        detail = event.get("detail") if isinstance(event.get("detail"), dict) else {}
        resolved = detail.get("resolved")
        status = str(event.get("status") or "")
        verdict = (
            "accepted" if status == "accepted" and resolved is True else "rejected"
        )
        prompt_payload = {
            "stage": event.get("stage"),
            "source": event.get("source"),
            "strategy": event.get("strategy"),
            "trigger_reasons": event.get("trigger_reasons") or [],
        }
        response_payload = {
            "actions": event.get("actions") or [],
            "affected_objects": event.get("affected_objects") or [],
        }
        prompt, prompt_complete = _bounded_text(prompt_payload, self.max_prompt_chars)
        response, response_complete = _bounded_text(
            response_payload, self.max_response_chars
        )
        event_signature = _stable_hash(event, 64)
        evidence = PreferenceEvidence(
            evidence_id="evidence_" + _stable_hash([event_signature, verdict]),
            kind="deterministic",
            verdict=verdict,
            source=str(event.get("repair_owner") or "scenesmith_core"),
            authoritative=True,
            quality_score=(1.0 if verdict == "accepted" else 0.0),
            report_ref="timing/repair_events.jsonl",
            details={"status": status, "resolved": resolved, "detail": detail},
        )
        record = TrajectoryRecord(
            trajectory_id="trajectory_"
            + _stable_hash([self.run_id, self.scene_id, event_signature]),
            created_at=str(event.get("created_at") or _utc_now()),
            run_id=self.run_id,
            scene_id=self.scene_id,
            task_id=self.task_id,
            experiment_signature=self.experiment_signature,
            config_hash=self.config_hash,
            model_id=self.model_id,
            stage=str(event.get("stage") or ""),
            agent_role="repair",
            event=str(event.get("strategy") or "deterministic_repair"),
            context_hash=_stable_hash(prompt, 64),
            prompt=prompt,
            response=response,
            response_hash=_stable_hash(response, 64),
            prompt_complete=prompt_complete,
            response_complete=response_complete,
            evidence=evidence,
            source_refs=["timing/repair_events.jsonl"],
            metadata={"repair_owner": event.get("repair_owner")},
        )
        if self._append(record):
            added["repair"] += 1

    def _capture_scene_expert_repair(
        self,
        stage: str,
        action: RepairResult,
        added: dict[str, int],
    ) -> None:
        prompt_payload = {
            "stage": stage,
            "repair_type": action.repair_type,
            "failure_type": action.failure_type,
        }
        response_payload = {
            "repair_action": action.repair_action,
            "execution_status": action.execution_status,
        }
        prompt, prompt_complete = _bounded_text(prompt_payload, self.max_prompt_chars)
        response, response_complete = _bounded_text(
            response_payload, self.max_response_chars
        )
        if not response.strip() or response in {
            "{}",
            '{"repair_action": "", "execution_status": ""}',
        }:
            return
        verdict = "accepted" if action.repair_verified else "rejected"
        payload = action.model_dump(mode="json")
        evidence_path = (
            self.output_dir
            / "evidence"
            / f"{stage}_scene_expert_repair_{_stable_hash(payload, 16)}.json"
        )
        evidence_path.parent.mkdir(parents=True, exist_ok=True)
        evidence_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
            newline="\n",
        )
        evidence_ref = self._relative_ref(evidence_path)
        evidence = PreferenceEvidence(
            evidence_id="evidence_" + _stable_hash([payload, verdict]),
            kind="deterministic",
            verdict=verdict,
            source=action.repair_owner,
            authoritative=bool(action.execution_status != "planned"),
            quality_score=(1.0 if action.repair_verified else 0.0),
            report_ref=evidence_ref,
            details=payload,
        )
        record = TrajectoryRecord(
            trajectory_id="trajectory_"
            + _stable_hash([self.run_id, self.scene_id, stage, payload]),
            created_at=_utc_now(),
            run_id=self.run_id,
            scene_id=self.scene_id,
            task_id=self.task_id,
            experiment_signature=self.experiment_signature,
            config_hash=self.config_hash,
            model_id=self.model_id,
            stage=stage,
            agent_role="repair",
            event=action.repair_type,
            context_hash=_stable_hash(prompt, 64),
            prompt=prompt,
            response=response,
            response_hash=_stable_hash(response, 64),
            prompt_complete=prompt_complete,
            response_complete=response_complete,
            evidence=evidence,
            source_refs=[evidence_ref],
        )
        if self._append(record):
            added["repair"] += 1

    def _write_manifest(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": "sceneexpert.trajectory_manifest.v1",
            "updated_at": _utc_now(),
            "run_id": self.run_id,
            "scene_id": self.scene_id,
            "task_id": self.task_id,
            "experiment_signature": self.experiment_signature,
            "config_hash": self.config_hash,
            "model_id": self.model_id,
            "trajectory_path": self.trajectory_path.name,
            "record_count": len(self._seen_ids),
            "records_added_this_process": self._counts,
            "capture_is_observer_only": True,
            "pairing_policy": (
                "offline exporter requires exact context_hash plus independent "
                "authoritative accepted/rejected evidence"
            ),
        }
        self.manifest_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
            newline="\n",
        )
