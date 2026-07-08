"""Per-stage online working memory for render/design/critic loops.

This is intentionally separate from SceneExpert long-term fast memory.  It is a
local scratchpad for the current scene/stage: every render can leave a compact
record, the critic can enrich that record with scores, and the next designer
call can retrieve recent lessons without waiting for end-of-scene MemoryWriter.
"""

from __future__ import annotations

import json
import logging
import os
import time
import hashlib
from pathlib import Path
from typing import Any

from scenesmith.agent_utils.scoring import compute_total_score, scores_to_dict
from scenesmith.scene_expert.context_bundle import (
    StageContextBundle,
    build_llm_call_debug_record,
)

console_logger = logging.getLogger(__name__)

_OBJECT_ALIASES: dict[str, tuple[str, ...]] = {
    "bed": ("bed", "beds"),
    "nightstand": ("nightstand", "nightstands", "bedside table", "bedside_table"),
    "wardrobe": ("wardrobe", "wardrobes", "closet", "closets", "corner_wardrobe"),
}


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _compact(text: str, max_chars: int = 700) -> str:
    value = " ".join(str(text or "").split())
    if len(value) <= max_chars:
        return value
    return value[: max_chars - 3] + "..."


def _object_names(scene: Any) -> list[str]:
    try:
        return [
            str(obj.name)
            for obj in scene.objects.values()
            if getattr(obj, "name", None)
        ]
    except Exception:
        return []


def _infer_category(text: str) -> str | None:
    normalized = str(text or "").lower().replace("_", " ")
    for category, aliases in _OBJECT_ALIASES.items():
        if any(alias.replace("_", " ") in normalized for alias in aliases):
            return category
    return None


def _count_required_categories(object_names: list[str]) -> dict[str, int]:
    counts = {category: 0 for category in _OBJECT_ALIASES}
    for name in object_names:
        category = _infer_category(name)
        if category in counts:
            counts[category] += 1
    return counts


def _extract_grade(scores: dict[str, Any], *name_parts: str) -> float | None:
    for key, value in scores.items():
        key_lower = str(key).lower().replace("_", " ")
        if not all(part.lower().replace("_", " ") in key_lower for part in name_parts):
            continue
        if isinstance(value, dict):
            grade = value.get("grade") or value.get("score")
            if isinstance(grade, (int, float)):
                return float(grade)
        if isinstance(value, (int, float)):
            return float(value)
    return None


def _deterministic_quality(
    *,
    object_names: list[str],
    required_counts: dict[str, int],
    scores: dict[str, Any],
    critique: str,
) -> dict[str, Any]:
    required_counts = {
        str(key).lower(): int(value)
        for key, value in (required_counts or {}).items()
        if int(value) > 0
    }
    observed_counts = _count_required_categories(object_names)
    missing: list[str] = []
    for category, required in required_counts.items():
        observed = observed_counts.get(category, 0)
        if observed < required:
            missing.extend([category] * (required - observed))

    prompt_following = _extract_grade(scores, "prompt", "following")
    critique_lower = str(critique or "").lower()
    claims_complete = any(
        term in critique_lower
        for term in (
            "all required",
            "all furniture quantities match",
            "bed - present",
            "bed, two nightstands",
            "all required furniture",
        )
    )
    inconsistent = bool(missing) and (
        claims_complete or (prompt_following is not None and prompt_following >= 8)
    )
    hard_valid = not missing
    note = ""
    if missing:
        note = (
            "Deterministic state check: missing required furniture "
            + ", ".join(missing)
            + f"; observed_counts={observed_counts}."
        )
        if inconsistent:
            note += " Ignore contradictory critic/designer text that claims completion."

    return {
        "required_counts": required_counts,
        "observed_counts": observed_counts,
        "missing_required_objects": missing,
        "hard_valid": hard_valid,
        "critic_inconsistent_with_state": inconsistent,
        "deterministic_note": note,
    }


def _scene_hash(scene: Any) -> str:
    try:
        return str(scene.content_hash())
    except Exception:
        return ""


def _score_dict(scores: Any | None) -> dict[str, Any]:
    if scores is None:
        return {}
    try:
        return scores_to_dict(scores)
    except Exception:
        if isinstance(scores, dict):
            return dict(scores)
    return {}


def _score_total(scores: Any | None) -> float | None:
    if scores is None:
        return None
    try:
        return float(compute_total_score(scores))
    except Exception:
        return None


def _canonical_stage(stage: str) -> str:
    if stage == "wall":
        return "wall_mounted"
    if stage == "ceiling":
        return "ceiling_mounted"
    if stage.startswith("manipulands_"):
        return "manipuland"
    return stage


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=str)


class StageWorkingMemory:
    """Scene-local working memory bank keyed by placement stage."""

    def __init__(self, root_dir: Path, stage: str, enabled: bool = True) -> None:
        self.root_dir = Path(root_dir)
        self.stage = stage
        self.enabled = enabled
        self.memory_dir = self.root_dir / "stage_working_memory" / stage
        self.memory_path = self.memory_dir / "memory.jsonl"
        self.timing_path = self.root_dir / "timing_stats.jsonl"
        self.scene_root_dir = (
            self.root_dir.parent if self.root_dir.name.startswith("room_") else self.root_dir
        )
        self.debug_memory_dir = (
            self.scene_root_dir / "scene_expert" / "working_memory" / stage
        )
        self.debug_memory_path = self.debug_memory_dir / "memory.jsonl"
        self.debug_timing_path = (
            self.scene_root_dir / "scene_expert" / "timing" / "stage_working_timing.jsonl"
        )
        self.debug_llm_path = (
            self.scene_root_dir / "scene_expert" / "timing" / "llm_calls.jsonl"
        )
        self.debug_context_dir = (
            self.scene_root_dir / "scene_expert" / "context_bundles" / stage
        )
        public_dir = os.environ.get("SCENEEXPERT_ACTIVE_MEMORY_BANK_DIR", "")
        self.public_memory_dir = Path(public_dir) if public_dir else None
        self.public_events_path = (
            self.public_memory_dir / "events.jsonl" if self.public_memory_dir else None
        )
        self.required_counts: dict[str, int] = {}
        if enabled:
            self.memory_dir.mkdir(parents=True, exist_ok=True)
            self.memory_path.touch(exist_ok=True)
            self.timing_path.touch(exist_ok=True)
            self.debug_memory_dir.mkdir(parents=True, exist_ok=True)
            self.debug_memory_path.touch(exist_ok=True)
            self.debug_timing_path.parent.mkdir(parents=True, exist_ok=True)
            self.debug_timing_path.touch(exist_ok=True)
            self.debug_llm_path.parent.mkdir(parents=True, exist_ok=True)
            self.debug_llm_path.touch(exist_ok=True)
            self.debug_context_dir.mkdir(parents=True, exist_ok=True)
            if self.public_events_path is not None:
                self.public_events_path.parent.mkdir(parents=True, exist_ok=True)
                self.public_events_path.touch(exist_ok=True)

    def set_required_counts(self, required_counts: dict[str, int] | None) -> None:
        """Set deterministic required-object counts for this stage."""
        self.required_counts = {
            str(key).lower(): int(value)
            for key, value in (required_counts or {}).items()
            if int(value) > 0
        }

    def save_render_record(
        self,
        *,
        render_dir: Path,
        role: str,
        event: str,
        scene: Any,
        text: str = "",
        scores: Any | None = None,
        critique: str = "",
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Save a compact record for one render or scored render."""
        if not self.enabled:
            return {}

        render_dir = Path(render_dir)
        images = sorted(str(path) for path in render_dir.glob("*.png"))
        score_data = _score_dict(scores)
        object_names = _object_names(scene)
        deterministic_quality = _deterministic_quality(
            object_names=object_names,
            required_counts=self.required_counts,
            scores=score_data,
            critique=critique or text,
        )
        record = {
            "schema_version": "1.0",
            "created_at": _now(),
            "stage": self.stage,
            "role": role,
            "event": event,
            "render_dir": str(render_dir),
            "images": images,
            "scores_path": str(render_dir / "scores.yaml")
            if (render_dir / "scores.yaml").exists()
            else "",
            "scores": score_data,
            "score_total": _score_total(scores),
            "critique": _compact(critique, max_chars=900),
            "text": _compact(text, max_chars=900),
            "scene_hash": _scene_hash(scene),
            "object_names": object_names,
            "object_count": len(object_names),
            "deterministic_quality": deterministic_quality,
            "extra": extra or {},
        }
        _write_json(render_dir / "render_memory.json", record)
        _append_jsonl(self.memory_path, record)
        _append_jsonl(self.debug_memory_path, record)
        self._commit_public_stage_event(record)
        console_logger.info(
            "[StageWorkingMemory] saved stage=%s role=%s event=%s render=%s "
            "scores=%s objects=%d",
            self.stage,
            role,
            event,
            render_dir,
            bool(score_data),
            record["object_count"],
        )
        return record

    def save_context_bundle(self, bundle: StageContextBundle) -> None:
        """Persist the structured context used before an LLM call."""
        if not self.enabled:
            return
        safe_event = "".join(
            c if c.isalnum() or c in ("-", "_") else "_" for c in bundle.event
        ) or "context"
        path = self.debug_context_dir / f"{int(time.time() * 1000)}_{safe_event}.json"
        try:
            bundle.save(path)
        except Exception as e:
            console_logger.warning("Failed to save StageContextBundle: %s", e)

    def record_llm_call(
        self,
        *,
        agent_role: str,
        event: str,
        prompt: Any,
        output: Any = "",
        result: Any = None,
        raw_response: Any = None,
        error: str = "",
    ) -> None:
        """Persist prompt/response metadata for one LLM call."""
        if not self.enabled:
            return
        record = build_llm_call_debug_record(
            stage=self.stage,
            agent_role=agent_role,
            event=event,
            prompt=prompt,
            output=output,
            result=result,
            raw_response=raw_response,
            error=error,
        )
        payload = record.model_dump()
        _append_jsonl(self.debug_llm_path, payload)
        if self.public_events_path is not None:
            event_payload = {
                "schema_version": "1.0",
                "created_at": _now(),
                "event_type": "llm_call",
                "stage": self.stage,
                "payload": payload,
            }
            _append_jsonl(self.public_events_path, event_payload)

    def retrieve_for_designer(
        self,
        *,
        query: str = "",
        max_items: int = 3,
    ) -> str:
        """Retrieve compact recent/scored lessons for the next designer turn."""
        if not self.enabled or not self.memory_path.exists():
            return ""

        records: list[dict[str, Any]] = []
        with self.memory_path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except Exception:
                    continue

        if not records:
            console_logger.info(
                "[StageWorkingMemory] retrieve stage=%s query=%r -> 0 records",
                self.stage,
                _compact(query, 80),
            )
            return ""

        query_tokens = {token.lower() for token in query.replace(",", " ").split()}

        def rank(record: dict[str, Any]) -> tuple[float, float]:
            quality = record.get("deterministic_quality") or {}
            text = " ".join(
                [
                    str(record.get("text", "")),
                    str(record.get("critique", "")),
                    str(quality.get("deterministic_note", "")),
                    " ".join(record.get("object_names", [])),
                ]
            ).lower()
            overlap = sum(1 for token in query_tokens if token and token in text)
            has_scores = 1.0 if record.get("scores") else 0.0
            is_critic = 1.0 if record.get("role") == "critic" else 0.0
            invalid_penalty = 4.0 if quality.get("critic_inconsistent_with_state") else 0.0
            hard_valid_bonus = 0.5 if quality.get("hard_valid", True) else 0.0
            # Invalid records with high hallucinated scores must not outrank
            # deterministic failure notes.
            score_total = 0.0 if quality.get("critic_inconsistent_with_state") else (
                record.get("score_total") or 0.0
            )
            return (
                overlap + has_scores + is_critic + hard_valid_bonus - invalid_penalty,
                score_total,
            )

        selected = sorted(records, key=rank, reverse=True)[:max_items]
        console_logger.info(
            "[StageWorkingMemory] retrieve stage=%s query=%r -> %d/%d records",
            self.stage,
            _compact(query, 80),
            len(selected),
            len(records),
        )
        lines = [
            f"=== Stage Working Memory: {self.stage} ===",
            "Use these recent render/critic notes to preserve what worked and avoid repeating failed changes.",
        ]
        for index, record in enumerate(selected, start=1):
            score_total = record.get("score_total")
            score_text = f", total_score={score_total:.1f}" if isinstance(score_total, (int, float)) else ""
            lines.append(
                f"{index}. [{record.get('role')}/{record.get('event')}{score_text}] "
                f"objects={record.get('object_names', [])}"
            )
            quality = record.get("deterministic_quality") or {}
            if quality.get("deterministic_note"):
                lines.append(
                    f"   deterministic: {_compact(quality['deterministic_note'], 320)}"
                )
            if record.get("critique") and not quality.get("critic_inconsistent_with_state"):
                lines.append(f"   critic: {_compact(record['critique'], 260)}")
            elif record.get("text"):
                lines.append(f"   note: {_compact(record['text'], 260)}")
            if record.get("render_dir"):
                lines.append(f"   render_dir: {record['render_dir']}")
        lines.append("=== End Stage Working Memory ===")
        return "\n".join(lines)

    def record_timing(
        self,
        *,
        module: str,
        event: str,
        elapsed_sec: float,
        extra: dict[str, Any] | None = None,
    ) -> None:
        """Append a timing event for later optimization analysis."""
        if not self.enabled:
            return
        record = {
            "schema_version": "1.0",
            "created_at": _now(),
            "stage": self.stage,
            "module": module,
            "event": event,
            "elapsed_sec": round(float(elapsed_sec), 3),
            "extra": extra or {},
        }
        _append_jsonl(self.timing_path, record)
        _append_jsonl(self.debug_timing_path, record)
        console_logger.info(
            "[Timing] stage=%s module=%s event=%s elapsed=%.3fs",
            self.stage,
            module,
            event,
            elapsed_sec,
        )

    def _commit_public_stage_event(self, record: dict[str, Any]) -> None:
        """Append a durable stage event and optional memory case to the shared bank."""
        if self.public_events_path is None or self.public_memory_dir is None:
            return
        event_payload = {
            "schema_version": "1.0",
            "created_at": _now(),
            "event_type": "stage_working_memory",
            "stage": self.stage,
            "role": record.get("role", ""),
            "event": record.get("event", ""),
            "render_dir": record.get("render_dir", ""),
            "scene_hash": record.get("scene_hash", ""),
            "payload": record,
        }
        _append_jsonl(self.public_events_path, event_payload)

        if record.get("role") != "critic":
            return
        try:
            quality = record.get("deterministic_quality") or {}
            scores = record.get("scores") or {}
            hard_valid = bool(quality.get("hard_valid", True))
            if hard_valid and scores:
                self._commit_public_success_case(record)
            elif not hard_valid or record.get("event") == "deterministic_hard_fail":
                self._commit_public_failure_case(record)
        except Exception as e:
            console_logger.warning("Failed to commit public stage memory event: %s", e)

    def _memory_id(self, prefix: str, record: dict[str, Any]) -> str:
        payload = "|".join(
            [
                self.stage,
                str(record.get("role", "")),
                str(record.get("event", "")),
                str(record.get("scene_hash", "")),
                str(record.get("render_dir", "")),
                str(record.get("critique", ""))[:300],
            ]
        )
        digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]
        return f"{prefix}_{self.stage}_{digest}"

    def _room_type_from_record(self, record: dict[str, Any]) -> str:
        text = " ".join(
            [
                self.stage,
                str(record.get("text", "")),
                str(record.get("critique", "")),
                " ".join(str(x) for x in record.get("object_names", [])),
            ]
        ).lower()
        if "bedroom" in text or "bed" in text or "nightstand" in text:
            return "bedroom"
        return "room"

    def _normalized_quality(self, record: dict[str, Any]) -> float:
        total = record.get("score_total")
        scores = record.get("scores") or {}
        if not isinstance(total, (int, float)) or not scores:
            return 0.0
        max_total = 10.0 * max(1, len(scores))
        return max(0.0, min(1.0, float(total) / max_total))

    def _commit_public_success_case(self, record: dict[str, Any]) -> None:
        quality_score = self._normalized_quality(record)
        if quality_score < 0.75:
            return
        from scenesmith.scene_expert.memory.schemas import SuccessCase
        from scenesmith.scene_expert.memory.store import FastMemoryStore
        from scenesmith.scene_expert.memory.text_builder import build_embedding_text

        case = SuccessCase(
            case_id=self._memory_id("success", record),
            room_type=self._room_type_from_record(record),
            style="",
            stage=self.stage,
            task_signature=list(record.get("object_names", []))[:12],
            required_objects=list((record.get("deterministic_quality") or {}).get("required_counts", {}).keys()),
            scene_summary=f"Stage {self.stage} critic accepted render {record.get('render_dir', '')}.",
            successful_pattern=[
                _compact(record.get("critique", ""), 500)
                or f"{self.stage} produced a hard-valid scored candidate."
            ],
            scores={k: float(v.get("grade", v)) for k, v in (record.get("scores") or {}).items() if isinstance(v, (int, float, dict)) and (not isinstance(v, dict) or isinstance(v.get("grade"), (int, float)))},
            trace_ref=str(record.get("render_dir", "")),
            quality_score=quality_score,
            confidence=0.45,
            created_at=_now(),
        )
        if not case.embedding_text:
            case = case.model_copy(update={"embedding_text": build_embedding_text(case)})
        FastMemoryStore(str(self.public_memory_dir)).add_success_case(case)

    def _commit_public_failure_case(self, record: dict[str, Any]) -> None:
        from scenesmith.scene_expert.memory.schemas import FailureCase
        from scenesmith.scene_expert.memory.store import FastMemoryStore
        from scenesmith.scene_expert.memory.text_builder import build_embedding_text

        quality = record.get("deterministic_quality") or {}
        note = quality.get("deterministic_note") or record.get("critique") or record.get("text") or ""
        failure_type = "deterministic_hard_fail"
        lowered = str(note).lower()
        if "missing required" in lowered:
            failure_type = "missing_required_object"
        elif "door" in lowered or "open-connection" in lowered or "opening" in lowered:
            failure_type = "door_or_opening_clearance"
        elif "collision" in lowered or "overlap" in lowered:
            failure_type = "collision_or_overlap"
        case = FailureCase(
            failure_id=self._memory_id("failure", record),
            room_type=self._room_type_from_record(record),
            stage=self.stage,
            object="",
            failure_type=failure_type,
            bad_pattern=_compact(note, 900),
            failure_reason=_compact(note, 900),
            repair_action="Run stage repair loop and re-score before accepting this candidate.",
            repair_verified=False,
            required_objects=list((quality.get("required_counts") or {}).keys()),
            scene_summary=f"Stage {self.stage} hard-failed at render {record.get('render_dir', '')}.",
            quality_score=max(0.0, self._normalized_quality(record)),
            confidence=0.6,
            created_at=_now(),
            scope="stage",
            is_deterministic=True,
            negative_constraint=_compact(note, 700),
            critic_check="Verify deterministic hard constraints before invoking VLM scoring.",
            trace_ref=str(record.get("render_dir", "")),
        )
        if not case.embedding_text:
            case = case.model_copy(update={"embedding_text": build_embedding_text(case)})
        FastMemoryStore(str(self.public_memory_dir)).add_failure_case(case)


def save_generic_render_memory(
    *,
    root_dir: Path,
    stage: str,
    render_dir: Path,
    scene: Any,
    rendering_mode: str,
    render_name: str | None,
    elapsed_sec: float,
) -> None:
    """Save a render-only record from RenderingManager."""
    stage = _canonical_stage(stage)
    memory = StageWorkingMemory(root_dir=root_dir, stage=stage, enabled=True)
    memory.save_render_record(
        render_dir=render_dir,
        role="render",
        event=render_name or rendering_mode,
        scene=scene,
        text=f"Rendered stage={stage}, mode={rendering_mode}, render_name={render_name or ''}",
        extra={"rendering_mode": rendering_mode, "render_elapsed_sec": elapsed_sec},
    )
    memory.record_timing(
        module="rendering_manager",
        event=render_name or rendering_mode,
        elapsed_sec=elapsed_sec,
        extra={"render_dir": str(render_dir)},
    )
