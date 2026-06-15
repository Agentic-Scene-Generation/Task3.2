"""Bootstrap the SceneExpert fast memory store from existing trace JSON files.

Usage (from repo root):
    python scripts/bootstrap_memory_from_traces.py \
        --traces-dir outputs/ \
        --memory-dir outputs/scene_expert_memory/ablation_4 \
        [--dry-run]

What it does:
    1. Recursively finds all trace_*.json files under --traces-dir.
    2. For each trace, calls MemoryWriter (Qwen3) to extract memory update ops.
    3. Applies ops to the FastMemoryStore at --memory-dir.

Run this ONCE before starting ablation_4 to seed memory from ablation_3 runs.
Idempotency: safe to re-run — Qwen3 is prompted to avoid duplicates, but
duplicate entries may appear if traces are imported multiple times.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

# Make sure the repo root is on the path regardless of cwd.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scenesmith.scene_expert.memory.store import FastMemoryStore
from scenesmith.scene_expert.memory.writer import MemoryWriter
from scenesmith.scene_expert.schemas import FullVerifyReport

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("bootstrap_memory")


def _load_trace(trace_path: Path) -> dict | None:
    try:
        with trace_path.open() as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"Failed to load {trace_path}: {e}")
        return None


def _trace_to_full_report(trace: dict) -> FullVerifyReport:
    """Reconstruct FullVerifyReport from a trace's final_report block."""
    final = trace.get("final_report", {})
    return FullVerifyReport(
        semantic_score=final.get("semantic_score", 0.5),
        aesthetic_score=final.get("aesthetic_score", 0.5),
        plausibility_score=final.get("plausibility_score", 0.5),
        style_consistency=final.get("style_consistency", 0.5),
        collision_free_rate=final.get("collision_free_rate", 0.5),
        stability_score=final.get("stability_score", 0.5),
        walkable_area_ratio=final.get("walkable_area_ratio", 0.0),
        reachability_score=final.get("reachability_score", 0.5),
        support_relation_accuracy=final.get("support_relation_accuracy", 0.5),
        overall_score=final.get("overall_score", 0.5),
        pass_scene=final.get("pass_scene", False),
    )


def _trace_to_summary(trace: dict) -> str:
    """Build a human-readable trace summary identical to TraceLogger.build_trace_summary()."""
    lines = [
        f"Trace: {trace.get('trace_id', 'unknown')}",
        f"Prompt: {trace.get('prompt', '')}",
        "Stages:",
    ]
    for stage_entry in trace.get("stages", []):
        stage = stage_entry.get("stage", "?")
        line = f"  [{stage}]"
        brief = stage_entry.get("stage_brief")
        if brief:
            line += f" objective={brief.get('stage_objective', '')!r}"
        report = stage_entry.get("verify_report")
        if report:
            passed = "PASS" if report.get("pass_stage") else "FAIL"
            scores = ", ".join(
                f"{k}={v:.2f}" for k, v in report.get("scores", {}).items()
            )
            line += f" verify={passed} scores=({scores})"
        repairs = stage_entry.get("repair_actions", [])
        if repairs:
            types = [r.get("repair_type", "?") for r in repairs]
            line += f" repairs={types}"
        lines.append(line)

    final = trace.get("final_report", {})
    overall = final.get("overall_score", 0.0)
    plausibility = final.get("plausibility_score", 0.0)
    passed = "YES" if final.get("pass_scene") else "NO"
    lines.append(
        f"Final: overall={overall:.2f} plausibility={plausibility:.2f} pass={passed}"
    )
    return "\n".join(lines)


def _build_related_memory_text(store: FastMemoryStore) -> str:
    """Give the writer a snapshot of current memory to avoid near-duplicates."""
    hints: list[str] = []
    for sc in store.success_cases[-5:]:
        hints.append(sc.to_hint_text())
    for fc in store.failure_cases[-5:]:
        hints.append(fc.to_hint_text())
    for sk in store.skills[-3:]:
        hints.append(f"[Skill] {sk.skill_name}")
    return "\n".join(hints) if hints else ""


def bootstrap(
    traces_dir: Path,
    memory_dir: str,
    dry_run: bool,
    model: str,
    api_base: str,
    api_key: str,
) -> None:
    trace_files = sorted(traces_dir.rglob("trace_*.json"))
    if not trace_files:
        logger.warning(f"No trace_*.json files found under {traces_dir}")
        return

    logger.info(f"Found {len(trace_files)} trace file(s) under {traces_dir}")

    store = FastMemoryStore(memory_dir)
    writer = MemoryWriter(model=model, api_base_url=api_base, api_key=api_key)

    total_ops = 0
    for trace_path in trace_files:
        logger.info(f"Processing {trace_path}")
        trace = _load_trace(trace_path)
        if trace is None:
            continue

        trace_summary = _trace_to_summary(trace)
        full_report = _trace_to_full_report(trace)
        related = _build_related_memory_text(store)

        ops = writer.write(
            trace_summary=trace_summary,
            full_report=full_report,
            related_old_memory=related,
        )
        logger.info(f"  → {len(ops)} ops from {trace_path.name}")

        if dry_run:
            for op in ops:
                logger.info(f"  [DRY-RUN] {op.op} {op.memory_type}: {op.content}")
        else:
            store.apply_updates(ops)
            total_ops += len(ops)

    if dry_run:
        logger.info("Dry-run complete — no files written.")
    else:
        logger.info(
            f"Bootstrap complete: {total_ops} ops applied to {memory_dir}\n"
            f"  success_cases: {len(store.success_cases)}\n"
            f"  failure_cases: {len(store.failure_cases)}\n"
            f"  skills:        {len(store.skills)}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--traces-dir",
        default="outputs",
        help="Root directory to search for trace_*.json files (default: outputs/)",
    )
    parser.add_argument(
        "--memory-dir",
        default=str(
            Path(os.environ.get("SCENEEXPERT_MEMORY_DIR", "outputs/scene_expert_memory"))
            / "ablation_4"
        ),
        help="Target memory store directory (default: outputs/scene_expert_memory/ablation_4)",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get(
            "SCENEEXPERT_MODEL_ID",
            os.environ.get("SCENEEXPERT_MODEL", "Qwen/Qwen3.5-35B-A3B"),
        ),
        help="Qwen3 model name (default: Qwen/Qwen3.5-35B-A3B)",
    )
    parser.add_argument(
        "--api-base",
        default=os.environ.get("OPENAI_BASE_URL", "http://localhost:8000/v1"),
        help="vLLM API base URL",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("OPENAI_API_KEY", "dummy"),
        help="API key (dummy for local vLLM)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print ops without writing to disk",
    )
    args = parser.parse_args()

    bootstrap(
        traces_dir=Path(args.traces_dir),
        memory_dir=args.memory_dir,
        dry_run=args.dry_run,
        model=args.model,
        api_base=args.api_base,
        api_key=args.api_key,
    )


if __name__ == "__main__":
    main()
