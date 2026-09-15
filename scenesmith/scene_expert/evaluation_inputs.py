"""Frozen TaskCompiler and intent inputs for controlled paired evaluation.

The first arm reaching a scene materializes the complete compiler result.  All
other arms reuse those exact bytes, so Fast Memory retrieval is the only
semantic treatment dimension.  The store is deliberately independent of Git
metadata and uses an atomic per-scene claim to support separately scheduled or
parallel pair arms.
"""

from __future__ import annotations

import hashlib
import json
import os
import time

from collections.abc import Callable
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "sceneexpert.frozen_compiled_inputs.v1"


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")


def _fingerprint(kind: str, value: Any) -> str:
    digest = hashlib.sha256(_canonical_bytes(value)).hexdigest()[:24]
    return f"sceneexpert.{kind}.v1:{digest}"


def _atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, default=str)
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(path)


class FrozenCompiledInputStore:
    """Materialize or load one immutable compiler bundle per scene identity."""

    def __init__(
        self,
        root: str | Path,
        *,
        wait_timeout_sec: float = 900.0,
        poll_interval_sec: float = 0.2,
    ) -> None:
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.wait_timeout_sec = max(1.0, float(wait_timeout_sec))
        self.poll_interval_sec = max(0.01, float(poll_interval_sec))

    @staticmethod
    def prompt_sha256(prompt: str) -> str:
        normalized = " ".join(str(prompt or "").split())
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    def case_dir(self, *, scene_id: int, prompt: str) -> Path:
        # SceneEval preserves its global scene index in the per-batch CSV.  The
        # prompt suffix prevents accidental aliasing in custom case sets while
        # retaining a human-readable scene locator.
        prompt_hash = self.prompt_sha256(prompt)
        return self.root / f"scene_{int(scene_id):06d}_{prompt_hash[:12]}"

    def load_or_create(
        self,
        *,
        scene_id: int,
        prompt: str,
        producer: Callable[[], dict[str, Any]],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Return compiler payload and identity, invoking ``producer`` once."""
        case_dir = self.case_dir(scene_id=scene_id, prompt=prompt)
        existing = self._load(case_dir, prompt=prompt)
        if existing is not None:
            payload, identity = existing
            return payload, {**identity, "source": "frozen_replay"}

        lock_path = case_dir.with_suffix(".lock")
        deadline = time.monotonic() + self.wait_timeout_sec
        claimed = False
        stale_lock_recovered = False
        while not claimed:
            try:
                fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                with os.fdopen(fd, "w", encoding="utf-8") as stream:
                    stream.write(
                        json.dumps(
                            {"pid": os.getpid(), "created_at": time.time()},
                            sort_keys=True,
                        )
                    )
                claimed = True
            except FileExistsError:
                existing = self._load(case_dir, prompt=prompt)
                if existing is not None:
                    payload, identity = existing
                    return payload, {**identity, "source": "frozen_replay"}
                if time.monotonic() >= deadline:
                    lock_age = self._lock_age_sec(lock_path)
                    if not stale_lock_recovered and lock_age >= self.wait_timeout_sec:
                        # A terminated worker can leave only this tiny claim
                        # file. Compiler calls are bounded well below the stale
                        # threshold, so one guarded recovery is safe.
                        lock_path.unlink(missing_ok=True)
                        stale_lock_recovered = True
                        deadline = time.monotonic() + self.wait_timeout_sec
                        continue
                    raise TimeoutError(
                        f"Timed out waiting for frozen compiler input: {case_dir}"
                    )
                time.sleep(self.poll_interval_sec)

        try:
            # An arm may have completed the bundle between the optimistic read
            # and our lock acquisition.
            existing = self._load(case_dir, prompt=prompt)
            if existing is not None:
                payload, identity = existing
                return payload, {**identity, "source": "frozen_replay"}
            payload = producer()
            identity = self._write(case_dir, prompt=prompt, payload=payload)
            return payload, {**identity, "source": "materialized"}
        finally:
            lock_path.unlink(missing_ok=True)

    @staticmethod
    def _lock_age_sec(path: Path) -> float:
        try:
            return max(0.0, time.time() - path.stat().st_mtime)
        except OSError:
            return 0.0

    def _write(
        self,
        case_dir: Path,
        *,
        prompt: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        task_spec = dict(payload.get("task_spec") or {})
        intent_contract = dict(payload.get("intent_contract") or {})
        compiler_metadata = dict(payload.get("compiler_metadata") or {})
        if not task_spec:
            raise ValueError("Frozen compiler payload requires a non-empty task_spec")
        prompt_hash = self.prompt_sha256(prompt)
        task_fingerprint = _fingerprint("task_spec", task_spec)
        intent_fingerprint = _fingerprint("intent_contract", intent_contract)
        bundle_fingerprint = _fingerprint(
            "compiled_input",
            {
                "prompt_sha256": prompt_hash,
                "task_spec": task_spec,
                "intent_contract": intent_contract,
            },
        )
        identity = {
            "schema_version": SCHEMA_VERSION,
            "prompt_sha256": prompt_hash,
            "task_spec_fingerprint": task_fingerprint,
            "intent_contract_fingerprint": intent_fingerprint,
            "compiled_input_fingerprint": bundle_fingerprint,
            "case_dir": str(case_dir),
        }
        case_dir.mkdir(parents=True, exist_ok=True)
        _atomic_write_json(case_dir / "scene_task_spec.json", task_spec)
        _atomic_write_json(case_dir / "intent_contract.json", intent_contract)
        _atomic_write_json(case_dir / "compiler_metadata.json", compiler_metadata)
        # The identity file is the commit marker and is always written last.
        _atomic_write_json(case_dir / "fingerprint.json", identity)
        return identity

    def _load(
        self,
        case_dir: Path,
        *,
        prompt: str,
    ) -> tuple[dict[str, Any], dict[str, Any]] | None:
        marker = case_dir / "fingerprint.json"
        if not marker.is_file():
            return None
        try:
            identity = json.loads(marker.read_text(encoding="utf-8"))
            task_spec = json.loads(
                (case_dir / "scene_task_spec.json").read_text(encoding="utf-8")
            )
            intent_contract = json.loads(
                (case_dir / "intent_contract.json").read_text(encoding="utf-8")
            )
            compiler_metadata = json.loads(
                (case_dir / "compiler_metadata.json").read_text(encoding="utf-8")
            )
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"Frozen compiler bundle is unreadable: {case_dir}"
            ) from exc
        expected_prompt_hash = self.prompt_sha256(prompt)
        expected = {
            "prompt_sha256": expected_prompt_hash,
            "task_spec_fingerprint": _fingerprint("task_spec", task_spec),
            "intent_contract_fingerprint": _fingerprint(
                "intent_contract", intent_contract
            ),
            "compiled_input_fingerprint": _fingerprint(
                "compiled_input",
                {
                    "prompt_sha256": expected_prompt_hash,
                    "task_spec": task_spec,
                    "intent_contract": intent_contract,
                },
            ),
        }
        for key, expected_value in expected.items():
            if identity.get(key) != expected_value:
                raise ValueError(
                    f"Frozen compiler bundle identity mismatch for {key}: {case_dir}"
                )
        return (
            {
                "task_spec": task_spec,
                "intent_contract": intent_contract,
                "compiler_metadata": compiler_metadata,
            },
            identity,
        )
