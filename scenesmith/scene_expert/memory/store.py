"""JSON-file-based persistent storage for the SceneExpert fast memory system.

Three banks stored as JSON Lines files:
  {memory_dir}/success_cases.jsonl
  {memory_dir}/failure_cases.jsonl
  {memory_dir}/skills.jsonl
"""

from __future__ import annotations

import json
import logging
import time
from contextlib import contextmanager
from pathlib import Path

from pydantic import BaseModel

from scenesmith.scene_expert.memory.schemas import (
    FailureCase,
    MemoryUpdateOp,
    Skill,
    SuccessCase,
)

console_logger = logging.getLogger(__name__)


class FastMemoryStore:
    """Append-only JSON Lines store for all three memory banks.

    Loads everything into memory on init (files are small for MVP).
    Writes are append-only for success/failure cases; skills are rewritten on update.
    """

    def __init__(self, memory_dir: str) -> None:
        self._dir = Path(memory_dir)
        self._dir.mkdir(parents=True, exist_ok=True)

        self._success_path = self._dir / "success_cases.jsonl"
        self._failure_path = self._dir / "failure_cases.jsonl"
        self._skills_path = self._dir / "skills.jsonl"
        self._events_path = self._dir / "events.jsonl"
        for path in (
            self._success_path,
            self._failure_path,
            self._skills_path,
            self._events_path,
        ):
            path.touch(exist_ok=True)

        self.success_cases: list[SuccessCase] = self._load(
            self._success_path, SuccessCase
        )
        self.failure_cases: list[FailureCase] = self._load(
            self._failure_path, FailureCase
        )
        self.skills: list[Skill] = self._load(self._skills_path, Skill)

        console_logger.info(
            f"FastMemoryStore loaded: {len(self.success_cases)} success cases, "
            f"{len(self.failure_cases)} failure cases, {len(self.skills)} skills"
        )

    # ------------------------------------------------------------------
    # Read helpers
    # ------------------------------------------------------------------

    def _load(self, path: Path, model_cls) -> list:
        if not path.exists():
            return []
        records = []
        with path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(model_cls.model_validate(json.loads(line)))
                except Exception as e:
                    console_logger.warning(
                        f"Skipping malformed memory record in {path}: {e}"
                    )
        return records

    def _reload_from_disk(self) -> None:
        """Refresh in-memory records before a locked write batch."""
        self.success_cases = self._load(self._success_path, SuccessCase)
        self.failure_cases = self._load(self._failure_path, FailureCase)
        self.skills = self._load(self._skills_path, Skill)

    @contextmanager
    def _file_lock(self):
        """Advisory memory-dir lock.

        ACP runs on Linux, where fcntl gives us a process-safe lock. On platforms
        without fcntl this degrades to a best-effort no-op for local editing.
        """
        lock_path = self._dir / ".memory.lock"
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with lock_path.open("a+") as lock_file:
            try:
                import fcntl

                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            except Exception:
                time.sleep(0.01)
            try:
                yield
            finally:
                try:
                    import fcntl

                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
                except Exception:
                    pass

    def _success_signature(self, case: SuccessCase) -> str:
        return "|".join(
            [
                case.room_type.lower(),
                case.stage.lower(),
                case.style.lower(),
                " ".join(sorted(x.lower() for x in case.task_signature)),
                " ".join(x.lower() for x in case.successful_pattern),
            ]
        )

    def _failure_signature(self, case: FailureCase) -> str:
        return "|".join(
            [
                case.room_type.lower(),
                case.stage.lower(),
                case.object.lower(),
                case.failure_type.lower(),
                case.bad_pattern.lower(),
                case.failure_reason.lower(),
            ]
        )

    def _skill_signature(self, skill: Skill) -> str:
        return skill.skill_name.strip().lower()

    # ------------------------------------------------------------------
    # Write helpers
    # ------------------------------------------------------------------

    def _append(self, path: Path, record: BaseModel) -> None:
        with path.open("a") as f:
            f.write(record.model_dump_json() + "\n")

    def _rewrite(self, path: Path, records: list) -> None:
        with path.open("w") as f:
            for r in records:
                f.write(r.model_dump_json() + "\n")

    def append_event(self, event: dict) -> None:
        """Append a durable debug/event record to the shared memory bank."""
        with self._file_lock():
            with self._events_path.open("a", encoding="utf-8", newline="\n") as f:
                f.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")

    # ------------------------------------------------------------------
    # Public write methods
    # ------------------------------------------------------------------

    def add_success_case(self, case: SuccessCase) -> None:
        existing = {self._success_signature(c) for c in self.success_cases}
        if self._success_signature(case) in existing or any(
            c.case_id == case.case_id for c in self.success_cases
        ):
            console_logger.info(
                f"Memory: skipped duplicate success case {case.case_id}"
            )
            return
        self.success_cases.append(case)
        self._append(self._success_path, case)
        console_logger.debug(f"Memory: added success case {case.case_id}")

    def add_failure_case(self, case: FailureCase) -> None:
        existing = {self._failure_signature(c) for c in self.failure_cases}
        if self._failure_signature(case) in existing or any(
            c.failure_id == case.failure_id for c in self.failure_cases
        ):
            console_logger.info(
                f"Memory: skipped duplicate failure case {case.failure_id}"
            )
            return
        self.failure_cases.append(case)
        self._append(self._failure_path, case)
        console_logger.debug(f"Memory: added failure case {case.failure_id}")

    def add_skill(self, skill: Skill) -> None:
        existing = {self._skill_signature(s) for s in self.skills}
        if self._skill_signature(skill) in existing:
            console_logger.info(f"Memory: skipped duplicate skill {skill.skill_name}")
            return
        self.skills.append(skill)
        self._append(self._skills_path, skill)
        console_logger.debug(f"Memory: added skill {skill.skill_name}")

    def update_skill(self, skill_name: str, updates: dict) -> None:
        for skill in self.skills:
            if skill.skill_name == skill_name:
                updated = skill.model_copy(update=updates)
                self.skills[self.skills.index(skill)] = updated
                self._rewrite(self._skills_path, self.skills)
                console_logger.debug(f"Memory: updated skill {skill_name}")
                return
        console_logger.warning(f"Memory: skill not found for update: {skill_name}")

    def apply_updates(self, ops: list[MemoryUpdateOp]) -> None:
        """Apply a batch of memory update operations from the MemoryWriter."""
        with self._file_lock():
            self._reload_from_disk()
            for op in ops:
                if op.op == "NOOP":
                    continue
                if op.op == "ADD":
                    if op.memory_type == "success_case":
                        self.add_success_case(SuccessCase.model_validate(op.content))
                    elif op.memory_type == "failure_case":
                        self.add_failure_case(FailureCase.model_validate(op.content))
                    elif op.memory_type == "skill":
                        self.add_skill(Skill.model_validate(op.content))
                    else:
                        console_logger.warning(
                            f"Unknown memory_type for ADD: {op.memory_type}"
                        )
                elif op.op == "UPDATE":
                    if op.memory_type == "skill":
                        self.update_skill(
                            op.target_id or op.content.get("skill_name", ""), op.content
                        )
                    else:
                        console_logger.warning(
                            f"UPDATE not supported for memory_type: {op.memory_type}"
                        )
                else:
                    console_logger.warning(f"Unknown memory op: {op.op}")
