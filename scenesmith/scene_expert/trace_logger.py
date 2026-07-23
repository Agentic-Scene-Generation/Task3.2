"""TraceLogger: structured JSON trace writer for SceneExpert.

Writes a complete per-run trace file capturing all inputs, outputs,
verifier reports, and repair actions for every stage. Traces feed both
the fast memory system and offline SFT/DPO sample construction.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

from scenesmith.scene_expert.schemas import (
    FullVerifyReport,
    HarnessContext,
    MemoryPack,
    RepairResult,
    StageBrief,
    StageCost,
    StageExecutionEvidence,
    StageTraceEntry,
    StageVerifyReport,
)

console_logger = logging.getLogger(__name__)


class TraceLogger:
    """Accumulates stage-level trace entries and serializes to JSON.

    One TraceLogger instance per scene generation run.
    """

    SCHEMA_VERSION = "1.2"

    def __init__(
        self,
        output_dir: str,
        scene_index: int,
        prompt: str,
        experiment_name: str = "",
        config_hash: str = "",
        task_spec_status: dict | None = None,
        task_spec: dict | None = None,
    ) -> None:
        self._output_dir = Path(output_dir)
        self._traces_dir = self._output_dir / "traces"
        self._traces_dir.mkdir(parents=True, exist_ok=True)

        self._trace_id = f"trace_{scene_index:06d}"
        self._scene_id = f"scene_{scene_index:03d}"
        self._scene_debug_dir = self._output_dir / self._scene_id / "scene_expert"
        self._stage_debug_dir = self._scene_debug_dir / "stages"
        self._trace_debug_dir = self._scene_debug_dir / "trace"
        self._memory_debug_dir = self._scene_debug_dir / "memory"
        self._visual_debug_dir = self._scene_debug_dir / "visuals"
        for path in (
            self._stage_debug_dir,
            self._trace_debug_dir,
            self._memory_debug_dir,
            self._visual_debug_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)

        self._prompt = prompt
        self._experiment_name = experiment_name
        self._config_hash = config_hash
        self._task_spec = dict(task_spec or {})
        self._stage_entries: list[StageTraceEntry] = []
        self._start_time = time.time()
        self._full_report: FullVerifyReport | None = None
        self._exports: dict = {}
        self._component_status: dict[str, dict] = {
            "task_compiler": dict(task_spec_status or {})
        }

    def record_component_status(self, component: str, status: dict) -> None:
        """Record whether a SceneExpert component used model output or fallback."""
        self._component_status[component] = dict(status)

    def _degraded_components(self) -> list[str]:
        return [
            name
            for name, status in self._component_status.items()
            if bool(status.get("degraded", False))
        ]

    def log_stage(
        self,
        stage: str,
        memory_pack: MemoryPack,
        stage_brief: StageBrief | None,
        scene_state_path: str,
        verify_report: StageVerifyReport | None,
        repair_actions: list[RepairResult],
        qwen_calls: int = 0,
        stage_time_sec: float | None = None,
        execution_evidence: StageExecutionEvidence | None = None,
    ) -> None:
        """Record a completed stage's data."""
        elapsed = (
            time.time() - self._start_time if stage_time_sec is None else stage_time_sec
        )
        entry = StageTraceEntry(
            stage=stage,
            memory_pack=memory_pack,
            stage_brief=stage_brief,
            scene_state_path=scene_state_path,
            verify_report=verify_report,
            repair_actions=repair_actions,
            cost=StageCost(qwen_calls=qwen_calls, stage_time_sec=round(elapsed, 1)),
            execution_evidence=execution_evidence or StageExecutionEvidence(),
        )
        self._stage_entries.append(entry)
        self._save_stage_entry(entry)
        self.save_partial(status="running")
        console_logger.debug(f"TraceLogger: logged stage {stage}")

    def save_stage_context(
        self,
        stage: str,
        memory_pack: MemoryPack,
        stage_brief: StageBrief | None,
        phase: str = "pre",
        execution_evidence: StageExecutionEvidence | None = None,
    ) -> Path:
        """Save pre/post-stage planning context for interrupted runs."""
        payload = {
            "schema_version": self.SCHEMA_VERSION,
            "trace_id": self._trace_id,
            "scene_id": self._scene_id,
            "stage": stage,
            "phase": phase,
            "time_sec": round(time.time() - self._start_time, 1),
            "memory_pack": memory_pack.model_dump(),
            "stage_brief": stage_brief.model_dump() if stage_brief else None,
            "execution_evidence": (
                execution_evidence.model_dump() if execution_evidence else None
            ),
        }
        path = (
            self._stage_debug_dir
            / f"{len(self._stage_entries):03d}_{stage}_{phase}.json"
        )
        self._write_json(path, payload)
        return path

    def save_stage_visual_manifest(self, stage: str, output_dir: str) -> Path:
        """Index existing render/debug artifacts for a stage."""
        root = Path(output_dir)
        render_dirs = []
        if root.exists():
            render_dirs = sorted(
                path for path in root.rglob("renders_*") if path.is_dir()
            )
        renders = []
        for render_dir in render_dirs:
            pngs = sorted(str(path) for path in render_dir.glob("*.png"))
            if not pngs:
                continue
            renders.append(
                {
                    "dir": str(render_dir),
                    "images": pngs,
                    "scores": (
                        str(render_dir / "scores.yaml")
                        if (render_dir / "scores.yaml").exists()
                        else ""
                    ),
                    "scene_state": (
                        str(render_dir / "scene_state.json")
                        if (render_dir / "scene_state.json").exists()
                        else ""
                    ),
                    "dmd": (
                        str(render_dir / "scene.dmd.yaml")
                        if (render_dir / "scene.dmd.yaml").exists()
                        else (
                            str(render_dir / "floor_plan.dmd.yaml")
                            if (render_dir / "floor_plan.dmd.yaml").exists()
                            else ""
                        )
                    ),
                }
            )

        payload = {
            "schema_version": self.SCHEMA_VERSION,
            "trace_id": self._trace_id,
            "scene_id": self._scene_id,
            "stage": stage,
            "output_dir": str(root),
            "render_count": len(renders),
            "renders": renders,
        }
        path = self._visual_debug_dir / f"{stage}_visuals.json"
        self._write_json(path, payload)
        return path

    def finalize(
        self,
        full_report: FullVerifyReport,
        exports: dict,
        model: str = "",
    ) -> dict:
        """Set the final report and return the full trace dict (before saving)."""
        self._full_report = full_report
        self._exports = exports

        trace = {
            "schema_version": self.SCHEMA_VERSION,
            "trace_id": self._trace_id,
            "scene_id": self._scene_id,
            "status": "completed",
            "degraded": bool(self._degraded_components()),
            "degraded_components": self._degraded_components(),
            "component_status": self._component_status,
            "experiment_name": self._experiment_name,
            "config_hash": self._config_hash,
            "prompt": self._prompt,
            "task_spec": self._task_spec,
            "model": model,
            "total_time_sec": round(time.time() - self._start_time, 1),
            "stages": [entry.model_dump() for entry in self._stage_entries],
            "final_report": full_report.model_dump(),
            "exports": exports,
        }
        return trace

    def save_partial(self, status: str = "partial", error: str = "") -> Path:
        """Save an inspectable partial trace without requiring finalize()."""
        trace = {
            "schema_version": self.SCHEMA_VERSION,
            "trace_id": self._trace_id,
            "scene_id": self._scene_id,
            "status": status,
            "degraded": bool(self._degraded_components()),
            "degraded_components": self._degraded_components(),
            "component_status": self._component_status,
            "error": error,
            "experiment_name": self._experiment_name,
            "config_hash": self._config_hash,
            "prompt": self._prompt,
            "task_spec": self._task_spec,
            "total_time_sec": round(time.time() - self._start_time, 1),
            "stages": [entry.model_dump() for entry in self._stage_entries],
        }
        path = self._trace_debug_dir / f"{self._trace_id}_partial.json"
        self._write_json(path, trace)
        return path

    def save(self, trace: dict | None = None) -> Path:
        """Save the trace to a JSON file. Returns the file path."""
        if trace is None:
            # Build minimal trace if finalize() was not called
            trace = {
                "schema_version": self.SCHEMA_VERSION,
                "trace_id": self._trace_id,
                "scene_id": self._scene_id,
                "status": "partial",
                "degraded": bool(self._degraded_components()),
                "degraded_components": self._degraded_components(),
                "component_status": self._component_status,
                "experiment_name": self._experiment_name,
                "config_hash": self._config_hash,
                "prompt": self._prompt,
                "task_spec": self._task_spec,
                "stages": [entry.model_dump() for entry in self._stage_entries],
            }

        trace_path = self._traces_dir / f"{self._trace_id}.json"
        self._write_json(trace_path, trace)
        self._write_json(self._trace_debug_dir / f"{self._trace_id}.json", trace)
        console_logger.info(f"TraceLogger: saved trace to {trace_path}")
        return trace_path

    def save_memory_update_ops(self, ops: list, full_report: FullVerifyReport) -> Path:
        """Mirror final memory-writer ops into the per-scene debug directory."""
        payload = {
            "schema_version": self.SCHEMA_VERSION,
            "trace_id": self._trace_id,
            "scene_id": self._scene_id,
            "op_count": len(ops),
            "full_report": full_report.model_dump(),
            "updates": [
                op.model_dump() if hasattr(op, "model_dump") else op for op in ops
            ],
        }
        path = self._memory_debug_dir / "memory_update_ops.json"
        self._write_json(path, payload)

        jsonl_path = self._memory_debug_dir / "memory_update_ops.jsonl"
        with jsonl_path.open("w", encoding="utf-8", newline="\n") as f:
            for op in ops:
                record = op.model_dump() if hasattr(op, "model_dump") else op
                f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        return path

    def _save_stage_entry(self, entry: StageTraceEntry) -> None:
        stage_index = len(self._stage_entries)
        payload = {
            "schema_version": self.SCHEMA_VERSION,
            "trace_id": self._trace_id,
            "scene_id": self._scene_id,
            "stage_index": stage_index,
            "entry": entry.model_dump(),
        }
        stage_path = self._stage_debug_dir / f"{stage_index:03d}_{entry.stage}.json"
        self._write_json(stage_path, payload)
        jsonl_path = self._stage_debug_dir / "stage_trace.jsonl"
        with jsonl_path.open("a", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")

    def _write_json(self, path: Path, payload: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="\n") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False, default=str)

    def build_trace_summary(self) -> str:
        """Build a human-readable summary of the trace for the MemoryWriter.

        Includes the full SceneSmith critic summary text per stage — this is the
        richest signal available for memory extraction.
        """
        lines = [f"Trace: {self._trace_id}", f"Prompt: {self._prompt}", "Stages:"]
        for entry in self._stage_entries:
            stage_line = f"  [{entry.stage}]"
            if entry.stage_brief:
                stage_line += f" objective={entry.stage_brief.stage_objective!r}"
            if entry.verify_report:
                passed = "PASS" if entry.verify_report.pass_stage else "FAIL"
                visual_scores = ", ".join(
                    f"{k}={v:.2f}"
                    for k, v in entry.verify_report.visual_scores.items()
                )
                rule_scores = ", ".join(
                    f"{k}={v:.2f}"
                    for k, v in entry.verify_report.rule_scores.items()
                )
                stage_line += (
                    f" verify={passed} visual_scores=({visual_scores}) "
                    f"rule_scores=({rule_scores})"
                )
                if entry.verify_report.runtime_repair_events:
                    stage_line += (
                        " runtime_recovery="
                        f"{entry.verify_report.runtime_repair_events}"
                    )
                if entry.verify_report.issues:
                    issue_types = [i.issue_type for i in entry.verify_report.issues]
                    stage_line += f" issues={issue_types}"
            if entry.repair_actions:
                repairs = [r.repair_type for r in entry.repair_actions]
                stage_line += f" repairs={repairs}"
            lines.append(stage_line)

            # Include critic summary — the most informative per-stage content.
            if entry.verify_report and entry.verify_report.critique_summary:
                # Truncate very long summaries to keep the trace summary manageable.
                summary_text = entry.verify_report.critique_summary
                if len(summary_text) > 800:
                    summary_text = summary_text[:800] + "... [truncated]"
                lines.append(f"    Critic: {summary_text}")

        if self._full_report:
            lines.append(
                f"Final: overall={self._full_report.overall_score:.2f} "
                f"plausibility={self._full_report.plausibility_score:.2f} "
                f"pass={'YES' if self._full_report.pass_scene else 'NO'}"
            )
        return "\n".join(lines)
