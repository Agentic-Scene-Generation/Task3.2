"""Resume a frozen SceneEval campaign without repeating completed candidate groups."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from scenesmith.scene_expert.slow_memory.campaign import freeze_memory
from scenesmith.scene_expert.slow_memory.dpo import (
    export_dpo_dataset,
    load_trajectories,
)
from scenesmith.scene_expert.slow_memory.paired import (
    read_json,
    validate_pair_inputs,
    write_json,
)
from scenesmith.scene_expert.slow_memory.paired_provenance import (
    collect_pair_code_provenance,
)


def validated_sources(root: Path) -> tuple[list[Path], set[str], list[dict]]:
    sources, completed, audits = [], set(), []
    for attempt in sorted((root / "attempts").glob("attempt_*")):
        files, groups, errors = validate_pair_inputs(attempt / "runs/paired_initial")
        records, diagnostics = load_trajectories(files)
        sources.extend(files)
        completed.update(row.task_id for row in records)
        audits.append(
            {
                "attempt": attempt.name,
                "groups": groups,
                "errors": errors,
                "load_diagnostics": diagnostics,
            }
        )
    return sources, completed, audits


def run_collection(attempt: Path, env: dict[str, str]) -> int:
    """Keep ACP visibly active while the canonical launcher owns its services."""
    started = time.monotonic()
    with (
        (attempt / "launcher.log").open("w") as log,
        subprocess.Popen(
            ["bash", str(REPO / "tmp/acp/acp_qwen38_full_generate.sh")],
            cwd=REPO,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
        ) as process,
    ):
        while True:
            try:
                return process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                print(
                    f"[campaign] {attempt.name} running; elapsed={(time.monotonic()-started)/60:.1f} min; "
                    f"pid={process.pid}; log={attempt / 'launcher.log'}",
                    flush=True,
                )


def publish_audit(
    root: Path,
    plan: dict,
    sources: list[Path],
    completed: set[str],
    audits: list[dict],
    selected: list[dict],
    generation_exit: int,
) -> dict:
    """Refresh both data policies after each chunk, including a zero-yield chunk."""
    exports = {}
    for policy in ("strict", "verified_relative_v1"):
        exports[policy] = export_dpo_dataset(
            trajectory_sources=sources,
            output_dir=root / "datasets" / policy,
            preference_policy=policy,
            min_quality_margin=0.03 if policy != "strict" else 0.05,
            split_assignments=plan["split_assignments"],
            completion_view="first_turn",
        )["stats"]
    summary = {
        "schema_version": "sceneexpert.campaign_audit.v1",
        "updated_at": time.time(),
        "completed_task_count": len(completed),
        "planned_collection_tasks": sum(
            row["split"] != "test" for row in plan["tasks"]
        ),
        "generation_exit": generation_exit,
        "selected_task_ids": [row["task_id"] for row in selected],
        "selected_missing_task_ids": [
            row["task_id"] for row in selected if row["task_id"] not in completed
        ],
        "attempts": audits,
        "exports": exports,
        "missing_task_ids": [
            row["task_id"]
            for row in plan["tasks"]
            if row["split"] != "test" and row["task_id"] not in completed
        ],
        "protocol": {
            "scope": "furniture_initial",
            "memory": "frozen",
            "writer": "disabled_in_collection_only",
        },
    }
    write_json(root / "campaign_audit.json", summary)
    print(
        json.dumps(
            {key: value for key, value in summary.items() if key != "attempts"},
            indent=2,
        ),
        flush=True,
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--action", choices=("collect", "export"), default="collect")
    parser.add_argument("--parallelism", type=int, default=2)
    parser.add_argument("--max-tasks", type=int, default=0)
    parser.add_argument("--chunk-size", type=int, default=12)
    args = parser.parse_args()
    root = args.root.resolve()
    plan = read_json(root / "campaign.json")
    annotations = REPO / "scripts/assets/annotations.csv"
    if (
        hashlib.sha256(annotations.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
        != plan["annotations_sha256"]
    ):
        raise ValueError(
            "SceneEval annotations changed after the task partition was frozen"
        )
    if (
        not 1 <= args.parallelism <= 7
        or args.max_tasks < 0
        or not 1 <= args.chunk_size <= 512
    ):
        parser.error("parallelism must be 1..7; max-tasks >= 0; chunk-size 1..512")
    import fcntl

    with (root / "campaign.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        exit_marker = root / "campaign_exit_status.env"
        exit_marker.unlink(missing_ok=True)
        sources, completed, audits = validated_sources(root)
        provenance = collect_pair_code_provenance()
        identity = root / "collection_source.json"
        if identity.exists() and read_json(identity) != provenance:
            raise ValueError(
                "campaign source changed; use a new campaign to preserve reproducibility"
            )
        write_json(identity, provenance)
        buckets = {
            split: [
                row
                for row in plan["tasks"]
                if row["split"] == split and row["task_id"] not in completed
            ]
            for split in ("train", "validation")
        }
        pending = []
        while any(buckets.values()):
            for split in ["train"] * 5 + ["validation"]:
                if buckets[split]:
                    pending.append(buckets[split].pop(0))
        if args.max_tasks:
            pending = pending[: args.max_tasks]
        selected = pending if args.action == "collect" else []
        generation_exit = 0
        summary = publish_audit(
            root, plan, sources, completed, audits, selected, generation_exit
        )
        for offset in range(0, len(selected), args.chunk_size):
            pending = selected[offset : offset + args.chunk_size]
            previous_completed = set(completed)
            from scenesmith.scene_expert.slow_memory.paired_runtime import (
                snapshot_codec_preflight,
            )

            write_json(root / "pair_preflight.json", snapshot_codec_preflight())
            attempts = root / "attempts"
            attempts.mkdir(exist_ok=True)
            indices = [
                int(path.name.removeprefix("attempt_"))
                for path in attempts.glob("attempt_*")
                if path.name.removeprefix("attempt_").isdigit()
            ]
            attempt = attempts / f"attempt_{max(indices, default=-1) + 1:03d}"
            attempt.mkdir()
            write_json(attempt / "selected_tasks.json", pending)
            seed = os.environ.get("MEMORY_SEED_DIR", "").strip()
            memory = freeze_memory(root, plan, Path(seed) if seed else None)
            env = dict(os.environ)
            env.update(
                {
                    "PROJECT_ROOT": str(REPO),
                    "RUN_ID": f"{root.name}_{attempt.name}",
                    "OUTPUT_ROOT": str(attempt / "runs"),
                    "PYTHON_BIN": sys.executable,
                    "CASE_SET": "sceneeval500",
                    "SCENEEVAL_SIZE": "500",
                    "SCENEEVAL_ANNOTATIONS": str(annotations),
                    "DIFFICULTY_SELECTION": "all",
                    "SCENE_SELECTION": ",".join(
                        str(row["sceneeval_id"]) for row in pending
                    ),
                    "MAX_CASES": str(len(pending)),
                    "ACP_PARALLELISM": str(args.parallelism),
                    "PIPELINE_STOP_STAGE": "furniture",
                    "SCENEEXPERT_COMPONENT_MEMORY_WRITER_ENABLED": "false",
                    "SCENEEXPERT_MEMORY_DIR": str(memory),
                    "SCENEEXPERT_MEMORY_READ_ONLY": "true",
                    "SCENEEXPERT_INITIAL_PAIRS_DIR": str(
                        attempt / "runs/paired_initial"
                    ),
                    "SCENEEXPERT_PAIR_MAX_GROUPS": str(len(pending)),
                    "SCENEEXPERT_PAIR_SOURCE_HASH": provenance["source_bundle_hash"],
                    "SCENEEXPERT_CODE_PROVENANCE_GIT_ENABLED": "false",
                    "CRITIC_PROBE_CONTINUE_ON_BATCH_FAILURE": "true",
                }
            )
            chunk_exit = run_collection(attempt, env)
            write_json(
                attempt / "exit.json",
                {"generation_exit": chunk_exit, "finished_at": time.time()},
            )
            generation_exit = generation_exit or chunk_exit
            sources, completed, audits = validated_sources(root)
            freeze_memory(root, plan)
            summary = publish_audit(
                root, plan, sources, completed, audits, selected, generation_exit
            )
            if completed == previous_completed:
                # A nominal zero exit is insufficient when no independently
                # executed group has complete, valid evidence.
                break
        # An intentionally bounded batch may finish while the larger frozen
        # campaign still has pending tasks. That is a successful resumable run.
        exit_code = 2 if generation_exit or summary["selected_missing_task_ids"] else 0
        exit_marker.write_text(f"exit_code={exit_code}\n", encoding="utf-8")
        return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
