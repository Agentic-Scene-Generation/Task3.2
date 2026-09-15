#!/usr/bin/env python3
"""Bounded Writer-only collection with preregistered inputs and isolated banks.

Never selects winners, merges banks, generates scenes or starts training.
Validation and dry-run require no model service. Each input gets one replay.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.replay_sceneexpert_memory_writer import (
    execute_replay,
    load_input,
    read_json,
    save_summary,
)
from scenesmith.scene_expert.schemas import FullVerifyReport


def fingerprint(value: object) -> str:
    """Stable JSON fingerprint, independent of filesystem formatting."""
    return hashlib.sha256(
        json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode()
    ).hexdigest()


def prepare(manifest: dict, project_root: Path) -> list[tuple[dict, Path, dict]]:
    """Validate every source and holdout exclusion before any model call."""
    if manifest.get("schema_version") != "bounded-writer-replay.v1":
        raise ValueError("Unsupported manifest schema")
    cases = manifest.get("cases", [])
    if not 1 <= len(cases) <= 8:
        raise ValueError("Select 1 to 8 sources; no automatic expansion")
    holdouts = manifest.get("holdout_cases", [])
    if not holdouts:
        raise ValueError("Register held-out task identities before collection")
    entries = cases + holdouts
    ids = [e["case_id"] for e in entries]
    hashes = [e["prompt_sha256"] for e in entries]
    if any(not re.fullmatch(r"[A-Za-z0-9_-]+", i) for i in ids):
        raise ValueError("Unsafe case ID")
    if any(not re.fullmatch(r"[a-f0-9]{64}", h) for h in hashes):
        raise ValueError("Invalid prompt fingerprint")
    if len(set(ids)) != len(ids) or len(set(hashes)) != len(hashes):
        raise ValueError("Duplicate source or source/holdout task overlap")
    prepared = []
    for case in cases:
        relative = Path(case["scene_expert_dir"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Source paths must be project-relative without traversal")
        source = (project_root / relative).resolve()
        if not source.is_relative_to(project_root.resolve()):
            raise ValueError("Source resolves outside project root")
        payload = load_input(source)
        FullVerifyReport.model_validate(payload["full_report"])
        if fingerprint(payload["evidence"]["prompt"]) != case["prompt_sha256"]:
            raise ValueError(f"Prompt changed: {case['case_id']}")
        if fingerprint(payload) != case["input_sha256"]:
            raise ValueError(
                f"Writer evidence changed: {case['case_id']}; review before rerun"
            )
        prepared.append((case, source, payload))
    return prepared


def review_records(destination: Path, case_id: str) -> list[dict]:
    """Expose complete persisted records for human triage, without promotion."""
    rows = []
    for filename in ("success_cases.jsonl", "failure_cases.jsonl", "skills.jsonl"):
        path = destination / "bank" / filename
        if path.is_file():
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    rows.append(
                        {
                            "source_case": case_id,
                            "bank_file": filename,
                            "decision": "pending_human_review",
                            "reason": "",
                            "record": json.loads(line),
                        }
                    )
    return rows


def run_collection(
    manifest: dict,
    prepared: list,
    destination: Path,
    *,
    model: str,
    base_url: str,
    dry_run: bool = False,
) -> int:
    """Checkpoint every result; service errors stop remaining paid work."""
    destination = destination.resolve()
    for _, source, _ in prepared:
        if destination == source or destination.is_relative_to(source):
            raise ValueError("Output must be outside source directories")
    destination.mkdir(parents=True, exist_ok=False)
    (destination / "plan.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    summary = {
        "schema_version": "bounded-writer-results.v1",
        "dry_run": dry_run,
        "generation_rerun": False,
        "automatic_seed_selection": False,
        "plan_sha256": fingerprint(manifest),
        "planned": len(prepared),
        "phase": "running",
        "cases": [],
        "holdout_cases": manifest["holdout_cases"],
    }
    save_summary(destination, summary)
    review = []
    failed = False
    for case, source, payload in prepared:
        target = destination / case["case_id"]
        target.mkdir()
        result, code = execute_replay(payload, target, model, base_url, dry_run=dry_run)
        result.update({"source": str(source), "input_sha256": fingerprint(payload)})
        save_summary(target, result)
        review.extend(review_records(target, case["case_id"]))
        summary["cases"].append(
            {
                "case_id": case["case_id"],
                "exit_code": code,
                "phase": result["phase"],
                "spatial_records": result.get("spatial_records", 0),
            }
        )
        (destination / "review_candidates.json").write_text(
            json.dumps(review, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        save_summary(destination, summary)
        print(json.dumps(summary["cases"][-1]), flush=True)
        failed = failed or code != 0
        if code:
            break
    summary["phase"] = (
        "completed_with_failures"
        if failed
        else "dry_run_completed" if dry_run else "awaiting_human_review"
    )
    summary["not_attempted"] = [
        c["case_id"] for c, _, _ in prepared[len(summary["cases"]) :]
    ]
    summary["persisted_records_for_review"] = len(review)
    save_summary(destination, summary)
    return 1 if failed else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--project-root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default="unsloth/Qwen3.8-27B-GGUF")
    parser.add_argument("--api-base-url", default="http://127.0.0.1:8002/v1")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    manifest = read_json(args.manifest)
    prepared = prepare(manifest, args.project_root)
    if args.output_dir.exists():
        parser.error("Output already exists; never overwrite or resume implicitly")
    if args.validate_only:
        print(
            f"Validated {len(prepared)} sources; no output, model calls or bank writes"
        )
        return 0
    return run_collection(
        manifest,
        prepared,
        args.output_dir,
        model=args.model,
        base_url=args.api_base_url,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    raise SystemExit(main())
