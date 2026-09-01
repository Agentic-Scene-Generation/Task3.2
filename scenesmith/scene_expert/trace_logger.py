"""TraceLogger: structured JSON trace writer for SceneExpert.

Writes a complete per-run trace file capturing all inputs, outputs,
verifier reports, and repair actions for every stage. Traces feed both
the fast memory system and offline SFT/DPO sample construction.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import subprocess
import time

from pathlib import Path
from typing import Iterable

from scenesmith.scene_expert.experiment_identity import stable_source_bundle_hash
from scenesmith.scene_expert.schemas import (
    FullVerifyReport,
    MemoryPack,
    RepairResult,
    SceneTaskSpec,
    StageBrief,
    StageCost,
    StageExecutionEvidence,
    StageRelationContext,
    StageTraceEntry,
    StageVerifyReport,
)

console_logger = logging.getLogger(__name__)


_DEFAULT_CODE_PROVENANCE_PATHS = (
    "configurations/config.yaml",
    "configurations/experiment/ablation_4c_qwen3_hybrid_memory.yaml",
    "configurations/experiment/ablation_5_qwen3_full.yaml",
    "configurations/scene_expert/base_scene_expert.yaml",
    "scripts/run_parallel_critic_on.sh",
    "scripts/run_sceneexpert_full_memory_pair.sh",
    "scenesmith/scene_expert/paired_metrics.py",
    "scenesmith/experiments/indoor_scene_generation.py",
    "scenesmith/scene_expert/config_utils.py",
    "scenesmith/scene_expert/global_planner.py",
    "scenesmith/scene_expert/harness.py",
    "scenesmith/scene_expert/hooks.py",
    "scenesmith/scene_expert/experiment_identity.py",
    "scenesmith/scene_expert/memory/activity.py",
    "scenesmith/scene_expert/memory/hybrid_retriever.py",
    "scenesmith/scene_expert/memory/index.py",
    "scenesmith/scene_expert/memory/injection.py",
    "scenesmith/scene_expert/memory/retriever.py",
    "scenesmith/scene_expert/memory/schemas.py",
    "scenesmith/scene_expert/memory/skill_identity.py",
    "scenesmith/scene_expert/memory/skill_policy.py",
    "scenesmith/scene_expert/memory/store.py",
    "scenesmith/scene_expert/memory/writer.py",
    "scenesmith/scene_expert/run_metrics.py",
    "scenesmith/scene_expert/schemas.py",
    "scenesmith/scene_expert/task_compiler.py",
    "scenesmith/scene_expert/trace_logger.py",
    "scenesmith/scene_expert/verifier.py",
    "scenesmith/scene_expert/repair_controller.py",
    "scenesmith/scenebenchmark_critic/intent_contract.py",
    "scenesmith/furniture_agents/stateful_furniture_agent.py",
    "scenesmith/manipuland_agents/stateful_manipuland_agent.py",
    "scenesmith/manipuland_agents/cross_stage_inventory.py",
    "scenesmith/manipuland_agents/tools/manipuland_tools.py",
    "scenesmith/agent_utils/clearance_zones.py",
    "scenesmith/scenebenchmark_critic/asset_library_annotations.py",
    "scenesmith/scenebenchmark_critic/metrics/functional_dependency/builder.py",
    "scenesmith/scenebenchmark_critic/metrics/functional_dependency/relations.py",
    # Local ACP launchers intentionally live under ignored tmp/. Their actual
    # bytes still belong to reproducibility when present on the runtime host.
    "tmp/acp/acp_qwen38_4c_generate.sh",
    "tmp/acp/acp_qwen38_4c_reuse.sh",
    "tmp/acp/acp_qwen38_full_reuse.sh",
)


def collect_code_provenance(
    repo_root: Path | None = None,
    source_paths: Iterable[str] = _DEFAULT_CODE_PROVENANCE_PATHS,
) -> dict[str, object]:
    """Capture the code identity loaded at scene-run startup.

    A replay can outlive a commit or start from a dirty worktree.  Resolved
    Hydra configuration alone therefore cannot identify the code that produced
    a trace.  This helper intentionally records both Git state and hashes of
    the modules that own the SceneExpert/repair behavior under investigation.
    """
    root = repo_root or Path(__file__).resolve().parents[2]
    root = root.resolve()
    provenance: dict[str, object] = {
        "repo_root": str(root),
        "git_revision": "",
        "git_status": "",
        "git_status_hash": "",
        "dirty": None,
        "source_hashes": {},
        "source_bundle_hash": "",
    }

    git_executable = _git_executable()

    def git_output(*args: str) -> str:
        if git_executable is None:
            return ""
        try:
            result = subprocess.run(
                [git_executable, *args],
                cwd=root,
                check=False,
                capture_output=True,
                text=True,
                timeout=5,
            )
        except (OSError, subprocess.TimeoutExpired):
            return ""
        return result.stdout.strip() if result.returncode == 0 else ""

    revision = git_output("rev-parse", "HEAD")
    status = git_output("status", "--porcelain=v1", "--untracked-files=normal")
    provenance["git_revision"] = revision
    provenance["git_status"] = status
    provenance["git_status_hash"] = hashlib.sha256(status.encode("utf-8")).hexdigest()
    provenance["dirty"] = bool(status) if revision else None

    resolved_source_paths = list(source_paths)
    entrypoint = str(os.environ.get("ACP_ENTRYPOINT") or "").strip()
    if entrypoint:
        try:
            entrypoint_path = Path(entrypoint).resolve()
            entrypoint_relative = entrypoint_path.relative_to(root).as_posix()
            if entrypoint_relative not in resolved_source_paths:
                resolved_source_paths.append(entrypoint_relative)
        except (OSError, ValueError):
            pass

    source_hashes: dict[str, str] = {}
    for relative_path in resolved_source_paths:
        path = (root / relative_path).resolve()
        try:
            path.relative_to(root)
        except ValueError:
            continue
        if path.is_file():
            # Keep provenance stable across Windows and Linux checkouts. Git
            # tracks these Python sources as text, while a Windows worktree may
            # materialize CRLF bytes for the same committed content.
            content = path.read_bytes().replace(b"\r\n", b"\n")
            source_hashes[str(relative_path)] = hashlib.sha256(content).hexdigest()
    provenance["source_hashes"] = source_hashes
    provenance["source_bundle_hash"] = stable_source_bundle_hash(source_hashes)
    provenance["source_file_count"] = len(source_hashes)
    return provenance


def _git_executable() -> str | None:
    """Find Git even when isolated workers receive a minimal ``PATH``."""
    configured = os.environ.get("GIT")
    windows_candidates = (
        str(Path(os.environ.get("ProgramFiles", "")) / "Git" / "cmd" / "git.exe"),
        str(Path(os.environ.get("ProgramFiles", "")) / "Git" / "bin" / "git.exe"),
        str(Path(os.environ.get("ProgramFiles(x86)", "")) / "Git" / "cmd" / "git.exe"),
    )
    candidates = (
        shutil.which(configured) if configured else None,
        shutil.which("git"),
        "/usr/bin/git",
        "/bin/git",
        *windows_candidates,
    )
    for candidate in candidates:
        if candidate and Path(candidate).is_file() and os.access(candidate, os.X_OK):
            return candidate
    return None


class TraceLogger:
    """Accumulates stage-level trace entries and serializes to JSON.

    One TraceLogger instance per scene generation run.
    """

    SCHEMA_VERSION = "1.7"

    def __init__(
        self,
        output_dir: str,
        scene_index: int,
        prompt: str,
        experiment_name: str = "",
        config_hash: str = "",
        experiment_signature: str = "",
        control_signature: str = "",
        task_spec_status: dict | None = None,
        task_spec: dict | None = None,
        code_provenance: dict[str, object] | None = None,
        component_flags: dict[str, bool] | None = None,
        memory_identity: dict[str, object] | None = None,
        evaluation_contract: dict[str, object] | None = None,
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
        self._experiment_signature = experiment_signature
        self._control_signature = control_signature
        self._task_spec = dict(task_spec or {})
        self._code_provenance = dict(code_provenance or {})
        self._memory_identity = dict(memory_identity or {})
        self._evaluation_contract = dict(evaluation_contract or {})
        self._component_flags = {
            str(name): bool(enabled)
            for name, enabled in dict(component_flags or {}).items()
        }
        self._stage_entries: list[StageTraceEntry] = []
        self._start_time = time.time()
        self._full_report: FullVerifyReport | None = None
        self._exports: dict = {}
        self._task_compiler: dict = {}
        self._intent_compiler: dict = {}
        self._component_status: dict[str, dict] = {
            "task_compiler": dict(task_spec_status or {})
        }

    def record_component_status(self, component: str, status: dict) -> None:
        """Record whether an optional component used model output or fallback."""
        self._component_status[component] = dict(status)

    def _degraded_components(self) -> list[str]:
        return [
            name
            for name, status in self._component_status.items()
            if bool(status.get("degraded", False))
        ]

    def record_task_compiler(
        self, task_spec: SceneTaskSpec, compiler_trace: dict | None = None
    ) -> Path:
        """Persist the inventory-only TaskCompiler result."""
        self._task_compiler = {
            "compiler_status": task_spec.compiler_status,
            "failure_reason": task_spec.compiler_failure_reason,
            "compiler_spec_version": task_spec.compiler_spec_version,
            "task_spec": task_spec.model_dump(mode="json", exclude_none=True),
            "structured_output": dict(compiler_trace or {}),
        }
        self._task_spec = task_spec.model_dump(mode="json", exclude_none=True)
        self.record_component_status(
            "task_compiler",
            {
                "source": (
                    "fallback" if task_spec.compiler_status == "degraded" else "llm"
                ),
                "degraded": task_spec.compiler_status == "degraded",
                "failure_reason": task_spec.compiler_failure_reason,
            },
        )
        path = self._trace_debug_dir / "task_compiler.json"
        self._write_json(path, self._task_compiler)
        self.save_partial(status="running")
        return path

    def record_intent_compiler(self, trace: dict) -> Path:
        """Persist independent intent compilation status and contract details."""
        self._intent_compiler = dict(trace or {})
        path = self._trace_debug_dir / "intent_compiler.json"
        self._write_json(path, self._intent_compiler)
        self.save_partial(status="running")
        return path

    def log_stage(
        self,
        stage: str,
        memory_pack: MemoryPack,
        relation_context: StageRelationContext | None = None,
        planner_trace: dict | None = None,
        stage_brief: StageBrief | None = None,
        scene_state_path: str = "",
        verify_report: StageVerifyReport | None = None,
        repair_actions: list[RepairResult] | None = None,
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
            relation_context=relation_context,
            planner_trace=dict(planner_trace or {}),
            stage_brief=stage_brief,
            scene_state_path=scene_state_path,
            verify_report=verify_report,
            repair_actions=list(repair_actions or []),
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
        relation_context: StageRelationContext | None,
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
            "relation_context": (
                relation_context.model_dump(mode="json")
                if relation_context is not None
                else None
            ),
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
        outcome_status = str(full_report.outcome_status or "COMPLETE").casefold()
        generation_status = str(full_report.generation_status or "unknown").casefold()
        trace_status = (
            "degraded_incomplete"
            if outcome_status == "degraded_incomplete" or generation_status == "partial"
            else "completed"
        )
        result_degraded = bool(
            generation_status in {"partial", "failed"}
            or full_report.requirement_status in {"partial", "unsatisfied"}
            or full_report.quality_status in {"degraded", "failed"}
        )

        trace = {
            "schema_version": self.SCHEMA_VERSION,
            "trace_id": self._trace_id,
            "scene_id": self._scene_id,
            "status": trace_status,
            "degraded": bool(
                self._degraded_components()
                or trace_status == "degraded_incomplete"
                or result_degraded
            ),
            "degraded_components": self._degraded_components(),
            "component_flags": self._component_flags,
            "component_status": self._component_status,
            "experiment_name": self._experiment_name,
            "config_hash": self._config_hash,
            "experiment_signature": self._experiment_signature,
            "control_signature": self._control_signature,
            "code_provenance": self._code_provenance,
            "memory_identity": self._memory_identity,
            "evaluation_contract": self._evaluation_contract,
            "prompt": self._prompt,
            "task_compiler": self._task_compiler,
            "intent_compiler": self._intent_compiler,
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
            "component_flags": self._component_flags,
            "component_status": self._component_status,
            "error": error,
            "experiment_name": self._experiment_name,
            "config_hash": self._config_hash,
            "experiment_signature": self._experiment_signature,
            "control_signature": self._control_signature,
            "code_provenance": self._code_provenance,
            "memory_identity": self._memory_identity,
            "evaluation_contract": self._evaluation_contract,
            "prompt": self._prompt,
            "task_compiler": self._task_compiler,
            "intent_compiler": self._intent_compiler,
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
                "component_flags": self._component_flags,
                "component_status": self._component_status,
                "experiment_name": self._experiment_name,
                "config_hash": self._config_hash,
                "experiment_signature": self._experiment_signature,
                "control_signature": self._control_signature,
                "code_provenance": self._code_provenance,
                "memory_identity": self._memory_identity,
                "evaluation_contract": self._evaluation_contract,
                "prompt": self._prompt,
                "task_compiler": self._task_compiler,
                "intent_compiler": self._intent_compiler,
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
                scores = ", ".join(
                    f"{k}={v:.2f}" for k, v in entry.verify_report.scores.items()
                )
                stage_line += f" verify={passed} scores=({scores})"
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

    def build_memory_writer_evidence(self) -> dict[str, object]:
        """Return the structured, untruncated evidence contract for memory writing.

        The long-term writer consumes the existing main critic output through
        ``verify_report`` and never has to infer scores, task metadata, or repair
        outcomes from the human-readable summary.
        """
        stages: list[dict[str, object]] = []
        for entry in self._stage_entries:
            stages.append(
                {
                    "stage": entry.stage,
                    "scene_state_path": entry.scene_state_path,
                    "stage_brief": (
                        entry.stage_brief.model_dump() if entry.stage_brief else None
                    ),
                    "relation_context": (
                        entry.relation_context.model_dump()
                        if entry.relation_context
                        else None
                    ),
                    "verify_report": (
                        entry.verify_report.model_dump()
                        if entry.verify_report
                        else None
                    ),
                    "repair_actions": [
                        action.model_dump() for action in entry.repair_actions
                    ],
                    "execution_evidence": entry.execution_evidence.model_dump(),
                    "retrieved_memory_ids": (
                        list(entry.memory_pack.success_case_ids)
                        + list(entry.memory_pack.failure_case_ids)
                        + list(entry.memory_pack.skill_names)
                    ),
                    "memory_pack": entry.memory_pack.model_dump(),
                }
            )
        return {
            "schema_version": "sceneexpert.memory_writer_evidence.v1",
            "trace_id": self._trace_id,
            "scene_id": self._scene_id,
            "run_id": str(self._output_dir.resolve()),
            "experiment_name": self._experiment_name,
            "config_hash": self._config_hash,
            "experiment_signature": self._experiment_signature,
            "control_signature": self._control_signature,
            "prompt": self._prompt,
            "task_spec": dict(self._task_spec),
            "stages": stages,
            "full_report": (
                self._full_report.model_dump() if self._full_report else None
            ),
            "component_flags": dict(self._component_flags),
            "component_status": dict(self._component_status),
            "code_provenance": dict(self._code_provenance),
            "memory_identity": dict(self._memory_identity),
            "evaluation_contract": dict(self._evaluation_contract),
        }
