#!/usr/bin/env python3
"""Replay only MemoryWriter against archived evidence into a NEW isolated bank.

No generation, critic, shared-base mutation, training, Git checks or old-bank writes.
Prefer exact memory_writer_input.json; old runs can use trace + memory_activity.
The --dry-run path projects requests without contacting any model service.
"""
from __future__ import annotations

import argparse
import json
import sys

from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scenesmith.scene_expert.memory.store import FastMemoryStore
from scenesmith.scene_expert.memory.writer import MemoryWriter
from scenesmith.scene_expert.schemas import FullVerifyReport


def read_json(path: Path) -> dict:
    """Read a source artifact, accepting a UTF-8 BOM from transferred files."""
    return json.loads(path.read_text(encoding="utf-8-sig"))


def load_input(scene_expert_dir: Path) -> dict:
    """Load exact input or transparently reconstruct the old evidence contract."""
    exact = scene_expert_dir / "memory/memory_writer_input.json"
    if exact.is_file():
        payload = read_json(exact)
        if payload.get("schema_version") != "memory-writer-input.v1":
            raise ValueError("Unsupported writer input schema")
        return {**payload, "replay_source": "exact_writer_input"}
    debug = read_json(scene_expert_dir / "memory/memory_writer_debug.json")
    paths = sorted(
        p
        for p in (scene_expert_dir / "trace").glob("trace_*.json")
        if not p.name.endswith("_partial.json")
    )
    if len(paths) != 1:
        raise ValueError(
            "Old replay requires exactly one final trace and original writer debug"
        )
    trace = read_json(paths[0])
    activity = read_json(scene_expert_dir / "memory_activity.json")
    entries = list((activity.get("stages") or {}).values())
    entries += [r["entry"] for r in activity.get("stage_attempt_history", [])]
    episodes = [
        e
        for entry in entries
        for e in (entry.get("placement_episodes") or {}).get("episodes", [])
    ]
    if not any("placement_episodes" in entry for entry in entries):
        raise ValueError(
            "No archived placement catalog; refusing to reconstruct speculative evidence"
        )
    evidence = {
        k: trace.get(k)
        for k in (
            "trace_id",
            "scene_id",
            "experiment_name",
            "config_hash",
            "experiment_signature",
            "control_signature",
            "prompt",
            "task_spec",
            "component_flags",
            "code_provenance",
            "memory_identity",
            "evaluation_contract",
        )
    }
    evidence["stages"] = [
        {
            k: entry.get(k)
            for k in (
                "stage",
                "scene_state_path",
                "stage_brief",
                "relation_context",
                "verify_report",
                "repair_actions",
                "execution_evidence",
                "retrieved_memory_ids",
                "memory_pack",
            )
        }
        for entry in trace.get("stages", [])
    ]
    # Keep original run provenance even when the package has moved machines.
    states = [str(s.get("scene_state_path") or "") for s in evidence["stages"]]
    evidence["run_id"] = trace.get("run_id") or next((s for s in states if s), "")
    if not evidence["run_id"] or not evidence.get("trace_id"):
        raise ValueError("Missing original run provenance")
    evidence["full_report"] = debug["full_report"]
    evidence["placement_experience_catalog"] = {"episodes": episodes}
    return {
        "schema_version": "memory-writer-input.v1",
        "replay_source": "reconstructed_final_trace_and_activity_not_exact_original_request",
        "trace_summary": debug.get("trace_summary_excerpt", ""),
        "related_old_memory": "",
        "evidence": evidence,
        "full_report": debug["full_report"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-expert-dir", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Must not exist; a separate audit/ and bank/ will be created here",
    )
    parser.add_argument("--model", default="unsloth/Qwen3.8-27B-GGUF")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    source, destination = args.scene_expert_dir.resolve(), args.output_dir.resolve()
    if destination == source or source in destination.parents:
        parser.error("Output must be outside the source scene_expert directory")
    payload = load_input(source)
    report = FullVerifyReport.model_validate(payload["full_report"])
    destination.mkdir(parents=True, exist_ok=False)
    writer = MemoryWriter(model=args.model, debug_dir=destination / "audit")
    if args.dry_run:
        writer._build_user_message(
            trace_summary=payload["trace_summary"],
            full_report=report,
            related_old_memory=payload["related_old_memory"],
            evidence_payload=payload["evidence"],
        )
        summary = {"dry_run": True, "projection": writer._prompt_attempts}
    else:
        ops = writer.write(
            payload["trace_summary"],
            report,
            related_old_memory=payload["related_old_memory"],
            evidence_payload=payload["evidence"],
        )
        bank = FastMemoryStore(destination / "bank")
        applied = bank.apply_updates(ops)
        writer.record_store_result(applied)
        summary = {
            "writer": writer.last_trace,
            "store_apply": applied,
            "spatial_records": sum(
                bool(r.placement_experience)
                for r in [*bank.success_cases, *bank.failure_cases]
            ),
        }
    summary.update(
        {
            "source": str(source),
            "replay_source": payload["replay_source"],
            "model": args.model,
            "generation_rerun": False,
        }
    )
    (destination / "replay_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if args.dry_run or writer.last_trace.get("success") else 1


if __name__ == "__main__":
    raise SystemExit(main())
