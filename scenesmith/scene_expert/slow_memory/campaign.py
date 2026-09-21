"""Freeze SceneEval task splits before a resumable collection campaign starts."""

from __future__ import annotations

import csv
import hashlib
import json
import random
import shutil
from collections import Counter
from pathlib import Path
from typing import Any


def task_id(prompt: str) -> str:
    """Match TrajectoryCollector's normalized original-task identity."""
    encoded = json.dumps(" ".join(prompt.split()), ensure_ascii=False, sort_keys=True)
    return "task_" + hashlib.sha256(encoded.encode()).hexdigest()[:16]


def prepare_campaign(
    annotations: Path,
    *,
    seed: int = 42,
    train: int = 128,
    validation: int = 24,
    test: int = 32,
) -> dict[str, Any]:
    """Select unique single-room tasks outside the previously used first 100 rows."""
    with annotations.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    by_difficulty: dict[str, list[dict[str, Any]]] = {}
    seen: set[str] = set()
    for row in rows:
        identity = task_id(row["Description"])
        if (
            int(row["ID"]) < 100
            or row["SceneScope"] != "single_room"
            or identity in seen
        ):
            continue
        seen.add(identity)
        by_difficulty.setdefault(row["Difficulty"], []).append(
            {
                "sceneeval_id": int(row["ID"]),
                "task_id": identity,
                "prompt": row["Description"],
                "difficulty": row["Difficulty"],
            }
        )
    rng = random.Random(seed)
    for values in by_difficulty.values():
        rng.shuffle(values)
    ordered = []
    while any(by_difficulty.values()):
        for difficulty in sorted(by_difficulty):
            if by_difficulty[difficulty]:
                ordered.append(by_difficulty[difficulty].pop())
    if min(train, validation, test) <= 0 or train + validation + test > len(ordered):
        raise ValueError(
            "campaign requires positive splits within the unique task pool"
        )
    tasks, offset = [], 0
    for split, count in (("test", test), ("validation", validation), ("train", train)):
        tasks.extend(
            {**row, "split": split} for row in ordered[offset : offset + count]
        )
        offset += count
    return {
        "schema_version": "sceneexpert.sceneeval_campaign.v1",
        "seed": seed,
        "annotations_sha256": hashlib.sha256(
            annotations.read_bytes().replace(b"\r\n", b"\n")
        ).hexdigest(),
        "annotations_hash_normalization": "crlf_to_lf",
        "excluded_prior_id_range": [0, 99],
        "tasks": tasks,
        "split_assignments": {row["task_id"]: row["split"] for row in tasks},
        "counts": dict(Counter(row["split"] for row in tasks)),
        "selection": "stratified_without_observed_labels",
    }


def save_campaign(path: Path, campaign: dict[str, Any]) -> None:
    """Reuse an identical plan; never silently change an existing partition."""
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != campaign:
            raise ValueError("existing campaign differs; use a new campaign path")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(campaign, indent=2), encoding="utf-8")


def freeze_memory(root: Path, plan: dict[str, Any], seed: Path | None = None) -> Path:
    """Create one validated bank and reject changes or known task leakage on resume."""
    from scenesmith.scene_expert.memory.store import FastMemoryStore

    bank = root / "frozen_memory" / "ablation_4c"
    marker = root / "memory_snapshot.json"
    if not marker.exists():
        if bank.exists():
            raise ValueError(
                "incomplete memory snapshot; inspect it before starting a new campaign"
            )
        if seed:
            FastMemoryStore(str(seed), read_only=True)
            bank.mkdir(parents=True)
            for name in (
                "success_cases.jsonl",
                "failure_cases.jsonl",
                "skills.jsonl",
                "events.jsonl",
                "manifest.json",
            ):
                shutil.copyfile(seed / name, bank / name)
        else:
            FastMemoryStore(str(bank))
    store = FastMemoryStore(str(bank), read_only=True)
    snapshot = store.snapshot_identity()
    excluded = set(plan["split_assignments"])
    excluded.update(
        "task_" + hashlib.sha256(row["prompt"].encode()).hexdigest()[:16]
        for row in plan["tasks"]
    )
    for record in [*store.success_cases, *store.failure_cases, *store.skills]:
        source_ids = {record.source_task_id, *record.source_task_ids}
        if source_ids & excluded:
            raise ValueError(
                "seed memory contains a campaign task; choose a disjoint memory bank"
            )
    if marker.exists():
        if json.loads(marker.read_text(encoding="utf-8"))["snapshot"] != snapshot:
            raise ValueError("frozen memory changed during the campaign")
    else:
        marker.write_text(
            json.dumps(
                {
                    "source": str(seed) if seed else "empty_cold_start",
                    "snapshot": snapshot,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    return bank.parent
