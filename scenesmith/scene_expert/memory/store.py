"""Versioned, process-safe JSONL storage for SceneExpert fast memory.

JSONL remains the durable and inspectable MVP format. A small atomic manifest
provides bank identity and monotonic revisioning so long-lived ACP workers can
notice writes made by other processes and invalidate vector indexes safely.
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid

from contextlib import contextmanager
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from scenesmith.scene_expert.memory.schemas import (
    MEMORY_SCHEMA_VERSION,
    FailureCase,
    MemoryUpdateOp,
    MemoryUtilityObservation,
    Skill,
    SuccessCase,
)
from scenesmith.scene_expert.memory.skill_identity import build_skill_semantic_signature
from scenesmith.scene_expert.memory.text_builder import build_embedding_text

console_logger = logging.getLogger(__name__)
MANIFEST_SCHEMA_VERSION = "sceneexpert.memory_manifest.v2"


class FastMemoryStore:
    """Persistent memory banks with atomic batches and cross-process refresh."""

    def __init__(self, memory_dir: str) -> None:
        self._dir = Path(memory_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._success_path = self._dir / "success_cases.jsonl"
        self._failure_path = self._dir / "failure_cases.jsonl"
        self._skills_path = self._dir / "skills.jsonl"
        self._events_path = self._dir / "events.jsonl"
        self._manifest_path = self._dir / "manifest.json"
        for path in (
            self._success_path,
            self._failure_path,
            self._skills_path,
            self._events_path,
        ):
            path.touch(exist_ok=True)

        self.success_cases: list[SuccessCase] = []
        self.failure_cases: list[FailureCase] = []
        self.skills: list[Skill] = []
        self._manifest: dict[str, Any] = {}
        self._loaded_revision = -1
        self._loaded_disk_signature: tuple[tuple[int, int], ...] = ()
        self.last_apply_summary: dict[str, Any] = {}

        with self._file_lock():
            self._manifest = self._read_or_create_manifest_unlocked()
            self._reload_from_disk_unlocked()
            counts = self._record_counts()
            if self._manifest.get("counts") != counts:
                self._manifest["counts"] = counts
                self._atomic_write_json(self._manifest_path, self._manifest)
            self._loaded_revision = int(self._manifest.get("revision", 0))
            self._loaded_disk_signature = self._disk_signature()

        console_logger.info(
            "FastMemoryStore loaded bank=%s revision=%d: %d success, %d failure, %d skills",
            self.bank_id,
            self.revision,
            len(self.success_cases),
            len(self.failure_cases),
            len(self.skills),
        )

    @property
    def memory_dir(self) -> Path:
        return self._dir

    @property
    def bank_id(self) -> str:
        return str(self._manifest.get("bank_id", ""))

    @property
    def revision(self) -> int:
        return int(self._manifest.get("revision", 0))

    @property
    def manifest(self) -> dict[str, Any]:
        return dict(self._manifest)

    @property
    def active_success_cases(self) -> list[SuccessCase]:
        return [record for record in self.success_cases if record.status == "active"]

    @property
    def active_failure_cases(self) -> list[FailureCase]:
        return [record for record in self.failure_cases if record.status == "active"]

    @property
    def active_skills(self) -> list[Skill]:
        return [record for record in self.skills if record.status == "active"]

    def refresh_if_changed(self, *, force: bool = False) -> bool:
        """Reload records when another process changes the bank or manifest."""
        disk_manifest = self._read_manifest()
        disk_revision = int(disk_manifest.get("revision", -1)) if disk_manifest else -1
        disk_signature = self._disk_signature()
        if (
            not force
            and disk_revision == self._loaded_revision
            and disk_signature == self._loaded_disk_signature
        ):
            return False

        with self._file_lock():
            self._manifest = self._read_or_create_manifest_unlocked()
            self._reload_from_disk_unlocked()
            self._loaded_revision = int(self._manifest.get("revision", 0))
            self._loaded_disk_signature = self._disk_signature()
        console_logger.info(
            "FastMemoryStore refreshed bank=%s revision=%d", self.bank_id, self.revision
        )
        return True

    def _load(self, path: Path, model_cls: type[BaseModel]) -> list[Any]:
        records: list[Any] = []
        if not path.exists():
            return records
        with path.open(encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                text = line.strip()
                if not text:
                    continue
                try:
                    records.append(model_cls.model_validate(json.loads(text)))
                except Exception as exc:
                    console_logger.warning(
                        "Skipping malformed memory record in %s:%d: %s",
                        path,
                        line_number,
                        exc,
                    )
        return records

    def _reload_from_disk_unlocked(self) -> None:
        self.success_cases = self._load(self._success_path, SuccessCase)
        self.failure_cases = self._load(self._failure_path, FailureCase)
        self.skills = self._load(self._skills_path, Skill)

    @contextmanager
    def _file_lock(self):
        """Advisory directory lock (process-safe on the Linux ACP runtime)."""
        lock_path = self._dir / ".memory.lock"
        with lock_path.open("a+") as lock_file:
            try:
                import fcntl

                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            except Exception:
                # Local Windows development is sequential in tests. ACP runs on
                # Linux where fcntl gives a real inter-process lock.
                time.sleep(0.01)
            try:
                yield
            finally:
                try:
                    import fcntl

                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
                except Exception:
                    pass

    def _new_manifest(self) -> dict[str, Any]:
        now = self._now()
        return {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "record_schema_version": MEMORY_SCHEMA_VERSION,
            "bank_id": str(uuid.uuid4()),
            "revision": 0,
            "bank_revisions": {"success": 0, "failure": 0, "skill": 0},
            "created_at": now,
            "updated_at": now,
            "last_mutation": "initialized",
            "counts": self._record_counts(),
        }

    def _read_manifest(self) -> dict[str, Any]:
        if not self._manifest_path.exists():
            return {}
        try:
            value = json.loads(self._manifest_path.read_text(encoding="utf-8"))
            return value if isinstance(value, dict) else {}
        except Exception as exc:
            console_logger.warning("Memory manifest is unreadable: %s", exc)
            return {}

    def _read_or_create_manifest_unlocked(self) -> dict[str, Any]:
        manifest = self._read_manifest()
        if manifest.get("bank_id") and isinstance(manifest.get("revision"), int):
            changed = False
            if not isinstance(manifest.get("bank_revisions"), dict):
                manifest["bank_revisions"] = {
                    "success": int(manifest.get("revision", 0)),
                    "failure": int(manifest.get("revision", 0)),
                    "skill": int(manifest.get("revision", 0)),
                }
                changed = True
            if not manifest.get("record_schema_version"):
                manifest["record_schema_version"] = MEMORY_SCHEMA_VERSION
                changed = True
            if changed:
                self._atomic_write_json(self._manifest_path, manifest)
            return manifest
        manifest = self._new_manifest()
        self._atomic_write_json(self._manifest_path, manifest)
        return manifest

    def _record_counts(self) -> dict[str, Any]:
        return {
            "success": len(self.success_cases),
            "failure": len(self.failure_cases),
            "skill": len(self.skills),
            "active_success": sum(x.status == "active" for x in self.success_cases),
            "active_failure": sum(x.status == "active" for x in self.failure_cases),
            "active_skill": sum(x.status == "active" for x in self.skills),
            "candidate_skill": sum(x.status == "candidate" for x in self.skills),
            "quarantined_skill": sum(x.status == "quarantined" for x in self.skills),
        }

    def _disk_signature(self) -> tuple[tuple[int, int], ...]:
        signature: list[tuple[int, int]] = []
        for path in (
            self._success_path,
            self._failure_path,
            self._skills_path,
            self._manifest_path,
        ):
            try:
                stat = path.stat()
                signature.append((stat.st_mtime_ns, stat.st_size))
            except FileNotFoundError:
                signature.append((0, 0))
        return tuple(signature)

    @staticmethod
    def _success_signature(case: SuccessCase) -> str:
        return "|".join(
            [
                case.room_type.casefold(),
                case.stage.casefold(),
                case.style.casefold(),
                " ".join(sorted(x.casefold() for x in case.task_signature)),
                " ".join(x.casefold() for x in case.successful_pattern),
            ]
        )

    @staticmethod
    def _failure_signature(case: FailureCase) -> str:
        return "|".join(
            [
                case.room_type.casefold(),
                case.stage.casefold(),
                case.object.casefold(),
                case.failure_type.casefold(),
                case.bad_pattern.casefold(),
                case.failure_reason.casefold(),
            ]
        )

    @staticmethod
    def _skill_signature(skill: Skill) -> str:
        return skill.semantic_signature or build_skill_semantic_signature(skill)

    def _rewrite(self, path: Path, records: list[BaseModel]) -> None:
        temporary = path.with_suffix(path.suffix + f".{uuid.uuid4().hex}.tmp")
        with temporary.open("w", encoding="utf-8", newline="\n") as file:
            for record in records:
                file.write(record.model_dump_json() + "\n")
            file.flush()
            os.fsync(file.fileno())
        temporary.replace(path)

    @staticmethod
    def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
        temporary = path.with_suffix(path.suffix + f".{uuid.uuid4().hex}.tmp")
        with temporary.open("w", encoding="utf-8", newline="\n") as file:
            json.dump(payload, file, indent=2, ensure_ascii=False, default=str)
            file.flush()
            os.fsync(file.fileno())
        temporary.replace(path)

    def append_event(self, event: dict[str, Any]) -> None:
        """Append auditable evidence without promoting it into active memory."""
        with self._file_lock():
            with self._events_path.open("a", encoding="utf-8", newline="\n") as file:
                file.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")
                file.flush()

    def add_success_case(self, case: SuccessCase) -> bool:
        summary = self.apply_updates(
            [
                MemoryUpdateOp(
                    op="ADD", memory_type="success_case", content=case.model_dump()
                )
            ]
        )
        return bool(summary["changed"])

    def add_failure_case(self, case: FailureCase) -> bool:
        summary = self.apply_updates(
            [
                MemoryUpdateOp(
                    op="ADD", memory_type="failure_case", content=case.model_dump()
                )
            ]
        )
        return bool(summary["changed"])

    def add_skill(self, skill: Skill) -> bool:
        summary = self.apply_updates(
            [MemoryUpdateOp(op="ADD", memory_type="skill", content=skill.model_dump())]
        )
        return bool(summary["changed"])

    def update_success_case(self, case_id: str, updates: dict[str, Any]) -> bool:
        summary = self.apply_updates(
            [
                MemoryUpdateOp(
                    op="UPDATE",
                    memory_type="success_case",
                    target_id=case_id,
                    content=updates,
                )
            ]
        )
        return bool(summary["changed"])

    def update_failure_case(self, failure_id: str, updates: dict[str, Any]) -> bool:
        summary = self.apply_updates(
            [
                MemoryUpdateOp(
                    op="UPDATE",
                    memory_type="failure_case",
                    target_id=failure_id,
                    content=updates,
                )
            ]
        )
        return bool(summary["changed"])

    def update_skill(self, skill_name: str, updates: dict[str, Any]) -> bool:
        summary = self.apply_updates(
            [
                MemoryUpdateOp(
                    op="UPDATE",
                    memory_type="skill",
                    target_id=skill_name,
                    content=updates,
                )
            ]
        )
        return bool(summary["changed"])

    def record_skill_outcomes(
        self,
        observations: list[MemoryUtilityObservation],
        *,
        harmful_quarantine_threshold: int = 2,
    ) -> dict[str, Any]:
        """Learn from independently verified downstream skill observations.

        A repeated run of the same task is useful for experiment metrics but is
        not independent evidence for mutating a reusable skill.  Therefore the
        durable bank accepts at most one utility observation per task, retains
        the run ID for provenance, and never treats the skill's source task as
        transfer evidence. Different tasks within one ACP run remain independent.
        """
        updated = 0
        quarantined: list[str] = []
        skipped_non_skill = 0
        skipped_unverified = 0
        skipped_non_independent = 0
        with self._file_lock():
            self._manifest = self._read_or_create_manifest_unlocked()
            self._reload_from_disk_unlocked()
            next_revision = int(self._manifest.get("revision", 0)) + 1

            for observation in observations:
                if observation.memory_type != "skill":
                    skipped_non_skill += 1
                    continue
                if (
                    not observation.prompt_delivered
                    or observation.outcome not in {"positive", "negative"}
                    or not observation.task_id
                    or not observation.run_id
                ):
                    skipped_unverified += 1
                    continue
                skill_index = next(
                    (
                        index
                        for index, skill in enumerate(self.skills)
                        if skill.skill_name.casefold()
                        == observation.memory_id.casefold()
                    ),
                    None,
                )
                if skill_index is None:
                    skipped_unverified += 1
                    continue
                skill = self.skills[skill_index]
                source_tasks = {
                    skill.source_task_id,
                    *skill.source_task_ids,
                } - {""}
                observed_tasks = {
                    item.task_id for item in skill.utility_observations if item.task_id
                }
                if (
                    observation.task_id in source_tasks
                    or observation.task_id in observed_tasks
                ):
                    skipped_non_independent += 1
                    continue

                positive = skill.positive_utility_count + int(
                    observation.outcome == "positive"
                )
                negative = skill.negative_utility_count + int(
                    observation.outcome == "negative"
                )
                status = skill.status
                if (
                    negative >= max(2, int(harmful_quarantine_threshold))
                    and negative > positive
                ):
                    status = "quarantined"
                    quarantined.append(skill.skill_name)
                # Beta(1,1) posterior keeps one successful transfer from
                # spuriously producing a perfect utility estimate.
                success_rate = (positive + 1.0) / (positive + negative + 2.0)
                retained_observations = [
                    *skill.utility_observations[-63:],
                    observation,
                ]
                changed = skill.model_copy(
                    update={
                        "status": status,
                        "positive_utility_count": positive,
                        "negative_utility_count": negative,
                        "success_rate": success_rate,
                        "utility_observations": retained_observations,
                        "usage_count": skill.usage_count + 1,
                        "last_used_at": self._now(),
                        "updated_at": self._now(),
                        "bank_version": next_revision,
                    }
                )
                self.skills[skill_index] = changed.model_copy(
                    update={"embedding_text": build_embedding_text(changed)}
                )
                updated += 1

            if updated:
                self._rewrite(self._skills_path, self.skills)
                now = self._now()
                bank_revisions = dict(self._manifest.get("bank_revisions") or {})
                bank_revisions["skill"] = int(bank_revisions.get("skill", 0)) + 1
                self._manifest.update(
                    {
                        "schema_version": MANIFEST_SCHEMA_VERSION,
                        "record_schema_version": MEMORY_SCHEMA_VERSION,
                        "revision": next_revision,
                        "bank_revisions": bank_revisions,
                        "updated_at": now,
                        "last_mutation": "record_skill_outcomes",
                        "counts": self._record_counts(),
                    }
                )
                self._atomic_write_json(self._manifest_path, self._manifest)

            self._loaded_revision = int(self._manifest.get("revision", 0))
            self._loaded_disk_signature = self._disk_signature()

        summary = {
            "changed": updated > 0,
            "revision": self.revision,
            "updated": updated,
            "quarantined": sorted(set(quarantined)),
            "skipped_non_skill": skipped_non_skill,
            "skipped_unverified": skipped_unverified,
            "skipped_non_independent": skipped_non_independent,
        }
        self.last_apply_summary = summary
        return summary

    def apply_updates(self, ops: list[MemoryUpdateOp]) -> dict[str, Any]:
        """Apply one atomic, deduplicated mutation batch and increment revision once."""
        changed_banks: set[str] = set()
        added = 0
        updated = 0
        merged = 0
        skill_candidate_added = 0
        skill_candidate_merged = 0
        skill_promoted_active = 0
        with self._file_lock():
            self._manifest = self._read_or_create_manifest_unlocked()
            self._reload_from_disk_unlocked()
            next_revision = int(self._manifest.get("revision", 0)) + 1

            for op in ops:
                if op.op == "NOOP":
                    continue
                if op.op == "ADD":
                    skill_before: Skill | None = None
                    incoming_skill: Skill | None = None
                    if op.memory_type == "skill":
                        incoming_skill = self._normalized_skill(op.content)
                        skill_before = self._find_skill_unlocked(incoming_skill)
                    changed, was_merged = self._apply_add_unlocked(op, next_revision)
                    if changed:
                        changed_banks.add(op.memory_type)
                        if was_merged:
                            merged += 1
                        else:
                            added += 1
                        if incoming_skill is not None:
                            skill_after = self._find_skill_unlocked(incoming_skill)
                            if skill_after is not None:
                                if (
                                    skill_before is None
                                    and skill_after.status == "candidate"
                                ):
                                    skill_candidate_added += 1
                                elif (
                                    skill_before is not None
                                    and skill_before.status == "candidate"
                                    and skill_after.status == "candidate"
                                ):
                                    skill_candidate_merged += 1
                                if skill_after.status == "active" and (
                                    skill_before is None
                                    or skill_before.status != "active"
                                ):
                                    skill_promoted_active += 1
                elif op.op == "UPDATE":
                    if self._apply_update_unlocked(op, next_revision):
                        changed_banks.add(op.memory_type)
                        updated += 1

            if "success_case" in changed_banks:
                self._rewrite(self._success_path, self.success_cases)
            if "failure_case" in changed_banks:
                self._rewrite(self._failure_path, self.failure_cases)
            if "skill" in changed_banks:
                self._rewrite(self._skills_path, self.skills)

            if changed_banks:
                now = self._now()
                bank_revisions = dict(self._manifest.get("bank_revisions") or {})
                for memory_type in changed_banks:
                    bank_name = {
                        "success_case": "success",
                        "failure_case": "failure",
                        "skill": "skill",
                    }[memory_type]
                    bank_revisions[bank_name] = (
                        int(bank_revisions.get(bank_name, 0)) + 1
                    )
                self._manifest.update(
                    {
                        "schema_version": MANIFEST_SCHEMA_VERSION,
                        "record_schema_version": MEMORY_SCHEMA_VERSION,
                        "revision": next_revision,
                        "bank_revisions": bank_revisions,
                        "updated_at": now,
                        "last_mutation": "apply_updates",
                        "counts": self._record_counts(),
                    }
                )
                self._atomic_write_json(self._manifest_path, self._manifest)

            self._loaded_revision = int(self._manifest.get("revision", 0))
            self._loaded_disk_signature = self._disk_signature()

        summary = {
            "changed": bool(changed_banks),
            "revision": self.revision,
            "added": added,
            "updated": updated,
            "merged": merged,
            "changed_banks": sorted(changed_banks),
            "skill_candidate_added": skill_candidate_added,
            "skill_candidate_merged": skill_candidate_merged,
            "skill_promoted_active": skill_promoted_active,
        }
        self.last_apply_summary = summary
        return summary

    @staticmethod
    def _normalized_skill(content: dict[str, Any]) -> Skill:
        record = Skill.model_validate(content)
        signature = record.semantic_signature or build_skill_semantic_signature(record)
        source_tasks = FastMemoryStore._unique(
            [
                *record.source_task_ids,
                record.source_task_id,
            ]
        )
        aliases = FastMemoryStore._unique([*record.skill_aliases, record.skill_name])
        return record.model_copy(
            update={
                "semantic_signature": signature,
                "skill_aliases": aliases,
                "source_task_ids": source_tasks,
                "independent_support_count": max(1, len(source_tasks)),
                "activation_min_independent_support": max(
                    2, int(record.activation_min_independent_support)
                ),
            }
        )

    def _find_skill_unlocked(self, incoming: Skill) -> Skill | None:
        incoming_signature = self._skill_signature(incoming)
        return next(
            (
                skill
                for skill in self.skills
                if self._skill_signature(skill) == incoming_signature
            ),
            None,
        )

    def _apply_add_unlocked(
        self, op: MemoryUpdateOp, revision: int
    ) -> tuple[bool, bool]:
        if op.memory_type == "success_case":
            record = SuccessCase.model_validate(op.content).model_copy(
                update={"bank_version": revision}
            )
            return self._add_or_merge_unlocked(
                self.success_cases,
                record,
                identity=lambda item: item.case_id,
                signature=self._success_signature,
            )
        if op.memory_type == "failure_case":
            record = FailureCase.model_validate(op.content).model_copy(
                update={"bank_version": revision}
            )
            return self._add_or_merge_unlocked(
                self.failure_cases,
                record,
                identity=lambda item: item.failure_id,
                signature=self._failure_signature,
            )
        record = self._normalized_skill(op.content).model_copy(
            update={"bank_version": revision}
        )
        record_tasks = {record.source_task_id, *record.source_task_ids} - {""}
        if any(
            skill.skill_name.casefold() == record.skill_name.casefold()
            and bool(
                record_tasks & ({skill.source_task_id, *skill.source_task_ids} - {""})
            )
            and self._skill_signature(skill) != self._skill_signature(record)
            for skill in self.skills
        ):
            # A retry of the same task is not independent evidence and must not
            # fork an existing named Skill merely because the LLM paraphrased
            # its procedure. A genuinely different task may still contribute a
            # distinct, identically named Skill with different semantics.
            return False, True
        return self._add_or_merge_unlocked(
            self.skills,
            record,
            # LLM-generated names are descriptive aliases, not stable identity.
            # Two identically named Skills may encode incompatible rooms,
            # relations, or procedures; only the deterministic semantic
            # signature is safe for cross-task evidence accumulation.
            identity=self._skill_signature,
            signature=self._skill_signature,
        )

    def _add_or_merge_unlocked(
        self,
        records: list[Any],
        incoming: Any,
        *,
        identity,
        signature,
    ) -> tuple[bool, bool]:
        incoming_id = identity(incoming)
        incoming_signature = signature(incoming)
        for index, current in enumerate(records):
            id_matches = identity(current) == incoming_id
            if not id_matches and signature(current) != incoming_signature:
                continue
            if (
                id_matches
                and incoming.source_run_id == current.source_run_id
                and incoming.source_task_id == current.source_task_id
            ):
                return False, True
            if isinstance(current, Skill) and isinstance(incoming, Skill):
                current_tasks = {
                    current.source_task_id,
                    *current.source_task_ids,
                } - {""}
                incoming_tasks = {
                    incoming.source_task_id,
                    *incoming.source_task_ids,
                } - {""}
                incoming_runs = {
                    incoming.source_run_id,
                    *incoming.source_run_ids,
                } - {""}
                if (
                    not incoming_tasks
                    or not incoming_runs
                    or (
                        not (incoming_tasks - current_tasks)
                        and not (
                            current.status == "candidate"
                            and incoming.status == "active"
                        )
                    )
                ):
                    return False, True
            merged = self._merge_observation(current, incoming)
            if merged == current:
                return False, True
            records[index] = merged
            return True, True
        records.append(incoming)
        return True, False

    def _merge_observation(self, current: Any, incoming: Any) -> Any:
        same_run = bool(incoming.source_run_id) and (
            incoming.source_run_id == current.source_run_id
            and incoming.source_task_id == current.source_task_id
        )
        evidence_refs = self._unique(current.evidence_refs + incoming.evidence_refs)
        critic_evidence = self._unique(
            current.critic_evidence + incoming.critic_evidence
        )
        source_task_ids = self._unique(
            current.source_task_ids
            + ([current.source_task_id] if current.source_task_id else [])
            + incoming.source_task_ids
            + ([incoming.source_task_id] if incoming.source_task_id else [])
        )
        source_run_ids = self._unique(
            current.source_run_ids
            + ([current.source_run_id] if current.source_run_id else [])
            + incoming.source_run_ids
            + ([incoming.source_run_id] if incoming.source_run_id else [])
        )
        spatial_by_key = {
            json.dumps(
                relation.model_dump(mode="json"),
                ensure_ascii=False,
                sort_keys=True,
            ): relation
            for relation in [*current.spatial_relations, *incoming.spatial_relations]
        }
        updates: dict[str, Any] = {
            "schema_version": MEMORY_SCHEMA_VERSION,
            "evidence_refs": evidence_refs,
            "critic_evidence": critic_evidence,
            "source_task_ids": source_task_ids,
            "source_run_ids": source_run_ids,
            "bank_version": incoming.bank_version,
            "updated_at": incoming.updated_at or self._now(),
            "quality_score": max(current.quality_score, incoming.quality_score),
            "confidence": max(current.confidence, incoming.confidence),
            "spatial_relations": list(spatial_by_key.values()),
            "provenance": (
                current.provenance
                if current.provenance.trace_id
                else incoming.provenance
            ),
        }
        if not same_run:
            updates["observation_count"] = current.observation_count + 1
        if isinstance(current, Skill) and isinstance(incoming, Skill):
            incoming_is_stronger = (
                incoming.quality_score,
                incoming.confidence,
            ) > (current.quality_score, current.confidence)
            support_count = max(1, len(source_task_ids))
            activation_threshold = max(
                2,
                int(current.activation_min_independent_support),
                int(incoming.activation_min_independent_support),
            )
            if current.status == "quarantined":
                lifecycle_status = "quarantined"
                activation_reason = current.activation_reason or "harm_quarantined"
            elif current.status == "active" or incoming.status == "active":
                lifecycle_status = "active"
                activation_reason = (
                    incoming.activation_reason
                    if incoming.status == "active"
                    else current.activation_reason
                ) or "scene_and_stage_verified"
            elif support_count >= activation_threshold:
                lifecycle_status = "active"
                activation_reason = "independent_stage_support_threshold_met"
            else:
                lifecycle_status = "candidate"
                activation_reason = "awaiting_independent_stage_support"
            updates["applicability"] = current.applicability.model_copy(
                update={
                    "room_types": self._unique(
                        current.applicability.room_types
                        + incoming.applicability.room_types
                    ),
                    "excluded_room_types": self._unique(
                        current.applicability.excluded_room_types
                        + incoming.applicability.excluded_room_types
                    ),
                    "required_object_roles": self._unique(
                        current.applicability.required_object_roles
                        + incoming.applicability.required_object_roles
                    ),
                    "required_relation_types": self._unique(
                        current.applicability.required_relation_types
                        + incoming.applicability.required_relation_types
                    ),
                    "forbidden_conditions": self._unique(
                        current.applicability.forbidden_conditions
                        + incoming.applicability.forbidden_conditions
                    ),
                }
            )
            updates.update(
                {
                    "room_types": self._unique(
                        current.room_types + incoming.room_types
                    ),
                    "required_objects": self._unique(
                        current.required_objects + incoming.required_objects
                    ),
                    "functional_zones": self._unique(
                        current.functional_zones + incoming.functional_zones
                    ),
                    "preconditions": self._unique(
                        current.preconditions + incoming.preconditions
                    ),
                    "procedure": (
                        list(incoming.procedure)
                        if incoming_is_stronger and incoming.procedure
                        else list(current.procedure)
                    ),
                    "failure_avoidance": self._unique(
                        current.failure_avoidance + incoming.failure_avoidance
                    ),
                    "postconditions": self._unique(
                        current.postconditions + incoming.postconditions
                    ),
                    "semantic_signature": (
                        current.semantic_signature
                        or incoming.semantic_signature
                        or self._skill_signature(current)
                    ),
                    "skill_aliases": self._unique(
                        [
                            *current.skill_aliases,
                            current.skill_name,
                            *incoming.skill_aliases,
                            incoming.skill_name,
                        ]
                    ),
                    "source_scene_passed": (
                        current.source_scene_passed or incoming.source_scene_passed
                    ),
                    "promotion_scope": (
                        "scene"
                        if current.promotion_scope == "scene"
                        or incoming.promotion_scope == "scene"
                        else "stage"
                    ),
                    "independent_support_count": support_count,
                    "activation_min_independent_support": activation_threshold,
                    "status": lifecycle_status,
                    "activation_reason": activation_reason,
                }
            )
        if same_run and all(
            getattr(current, key) == value for key, value in updates.items()
        ):
            return current
        merged = current.model_copy(update=updates)
        return merged.model_copy(
            update={"embedding_text": build_embedding_text(merged)}
        )

    def _apply_update_unlocked(self, op: MemoryUpdateOp, revision: int) -> bool:
        if op.memory_type == "success_case":
            return self._update_record_unlocked(
                self.success_cases,
                op.target_id or str(op.content.get("case_id", "")),
                op.content,
                identity_field="case_id",
                model_cls=SuccessCase,
                revision=revision,
            )
        if op.memory_type == "failure_case":
            return self._update_record_unlocked(
                self.failure_cases,
                op.target_id or str(op.content.get("failure_id", "")),
                op.content,
                identity_field="failure_id",
                model_cls=FailureCase,
                revision=revision,
            )
        return self._update_record_unlocked(
            self.skills,
            op.target_id or str(op.content.get("skill_name", "")),
            op.content,
            identity_field="skill_name",
            model_cls=Skill,
            revision=revision,
        )

    def _update_record_unlocked(
        self,
        records: list[Any],
        target_id: str,
        updates: dict[str, Any],
        *,
        identity_field: str,
        model_cls: type[BaseModel],
        revision: int,
    ) -> bool:
        for index, record in enumerate(records):
            if str(getattr(record, identity_field)) != target_id:
                continue
            payload = {**record.model_dump(), **dict(updates)}
            payload[identity_field] = getattr(record, identity_field)
            payload["bank_version"] = revision
            payload["updated_at"] = self._now()
            updated = model_cls.model_validate(payload)
            if updated == record:
                return False
            records[index] = updated
            return True
        console_logger.warning(
            "Memory record not found for update: %s=%s", identity_field, target_id
        )
        return False

    @staticmethod
    def _unique(values: list[str]) -> list[str]:
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
    def _now() -> str:
        return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
