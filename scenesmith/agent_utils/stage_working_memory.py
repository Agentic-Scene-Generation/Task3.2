"""Per-stage online working memory for render/design/critic loops.

This is intentionally separate from SceneExpert long-term fast memory.  It is a
local scratchpad for the current scene/stage: every render can leave a compact
record, the critic can enrich that record with scores, and the next designer
call can retrieve recent lessons without waiting for end-of-scene MemoryWriter.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

from scenesmith.agent_utils.scoring import compute_total_score, scores_to_dict

console_logger = logging.getLogger(__name__)


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
        if enabled:
            self.memory_dir.mkdir(parents=True, exist_ok=True)
            self.memory_path.touch(exist_ok=True)
            self.timing_path.touch(exist_ok=True)
            self.debug_memory_dir.mkdir(parents=True, exist_ok=True)
            self.debug_memory_path.touch(exist_ok=True)
            self.debug_timing_path.parent.mkdir(parents=True, exist_ok=True)
            self.debug_timing_path.touch(exist_ok=True)

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
            "object_names": _object_names(scene),
            "object_count": len(_object_names(scene)),
            "extra": extra or {},
        }
        _write_json(render_dir / "render_memory.json", record)
        _append_jsonl(self.memory_path, record)
        _append_jsonl(self.debug_memory_path, record)
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
            text = " ".join(
                [
                    str(record.get("text", "")),
                    str(record.get("critique", "")),
                    " ".join(record.get("object_names", [])),
                ]
            ).lower()
            overlap = sum(1 for token in query_tokens if token and token in text)
            has_scores = 1.0 if record.get("scores") else 0.0
            is_critic = 1.0 if record.get("role") == "critic" else 0.0
            return (overlap + has_scores + is_critic, record.get("score_total") or 0.0)

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
            if record.get("critique"):
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
