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
    StageTraceEntry,
    StageVerifyReport,
)

console_logger = logging.getLogger(__name__)


class TraceLogger:
    """Accumulates stage-level trace entries and serializes to JSON.

    One TraceLogger instance per scene generation run.
    """

    SCHEMA_VERSION = "1.1"

    def __init__(
        self,
        output_dir: str,
        scene_index: int,
        prompt: str,
        experiment_name: str = "",
        config_hash: str = "",
    ) -> None:
        self._output_dir = Path(output_dir)
        self._traces_dir = self._output_dir / "traces"
        self._traces_dir.mkdir(parents=True, exist_ok=True)

        self._trace_id = f"trace_{scene_index:06d}"
        self._scene_id = f"scene_{scene_index:03d}"
        self._prompt = prompt
        self._experiment_name = experiment_name
        self._config_hash = config_hash
        self._stage_entries: list[StageTraceEntry] = []
        self._start_time = time.time()
        self._full_report: FullVerifyReport | None = None
        self._exports: dict = {}

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
    ) -> None:
        """Record a completed stage's data."""
        elapsed = time.time() - self._start_time if stage_time_sec is None else stage_time_sec
        entry = StageTraceEntry(
            stage=stage,
            memory_pack=memory_pack,
            stage_brief=stage_brief,
            scene_state_path=scene_state_path,
            verify_report=verify_report,
            repair_actions=repair_actions,
            cost=StageCost(qwen_calls=qwen_calls, stage_time_sec=round(elapsed, 1)),
        )
        self._stage_entries.append(entry)
        console_logger.debug(f"TraceLogger: logged stage {stage}")

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
            "experiment_name": self._experiment_name,
            "config_hash": self._config_hash,
            "prompt": self._prompt,
            "model": model,
            "total_time_sec": round(time.time() - self._start_time, 1),
            "stages": [entry.model_dump() for entry in self._stage_entries],
            "final_report": full_report.model_dump(),
            "exports": exports,
        }
        return trace

    def save(self, trace: dict | None = None) -> Path:
        """Save the trace to a JSON file. Returns the file path."""
        if trace is None:
            # Build minimal trace if finalize() was not called
            trace = {
                "schema_version": self.SCHEMA_VERSION,
                "trace_id": self._trace_id,
                "scene_id": self._scene_id,
                "experiment_name": self._experiment_name,
                "config_hash": self._config_hash,
                "prompt": self._prompt,
                "stages": [entry.model_dump() for entry in self._stage_entries],
            }

        trace_path = self._traces_dir / f"{self._trace_id}.json"
        with trace_path.open("w") as f:
            json.dump(trace, f, indent=2, default=str)
        console_logger.info(f"TraceLogger: saved trace to {trace_path}")
        return trace_path

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
                scores = ", ".join(f"{k}={v:.2f}" for k, v in entry.verify_report.scores.items())
                stage_line += f" verify={passed} scores=({scores})"
                if entry.verify_report.issues:
                    issue_types = [i.issue_type for i in entry.verify_report.issues]
                    stage_line += f" issues={issue_types}"
            if entry.repair_actions:
                repairs = [r.repair_type for r in entry.repair_actions]
                stage_line += f" repairs={repairs}"
            lines.append(stage_line)

            # Include critic summary — the most informative per-stage content.
            if (
                entry.verify_report
                and entry.verify_report.critique_summary
            ):
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
