"""Byte-bound, independently qualified admission to the named v2 pilot."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

CERTIFIED_605_IDS = frozenset({368, 371, 238, 479, 302, 266})
SOURCE_605 = "6ec532643f657e101b173f60dbe55fc920c3fcd281eec448cf85f52fa50f5d18"
PROFILE = "furniture_initial_pairlocal_pilot_v2"
ORIGINAL_CAMPAIGN_SHA256 = (
    "cff0490851f3ad5356e547a4a8289c16a20cc4a8e555f4afdbde9a3957826a1d"
)
APPROVED_SELECTION_SHA256 = (
    "a24533a87457956024a535822953a310711fe427718961b5bba09e7d6f1d38be"
)


def file_binding(root: Path, binding: dict[str, Any]) -> Path:
    """Resolve a declared proof inside the package and require its actual bytes."""
    path = (root / binding["path"]).resolve()
    if (
        not path.is_relative_to(root.resolve())
        or not path.is_file()
        or path.is_symlink()
    ):
        raise ValueError("qualification proof path is missing or outside the dataset")
    if hashlib.sha256(path.read_bytes()).hexdigest() != binding["sha256"]:
        raise ValueError("qualification proof bytes changed")
    return path


def validate_qualified_dataset(dataset_dir: Path) -> list[str]:
    """Refuse raw exports, unproved source bridging, repeats and sub-floor pilots."""
    errors: list[str] = []
    try:
        manifest = json.loads((dataset_dir / "qualification.json").read_bytes())
        if (
            manifest.get("schema_version")
            != "sceneexpert.pairlocal_qualified_dataset.v1"
            or manifest.get("profile") != PROFILE
        ):
            raise ValueError("missing named pair-local qualification protocol")
        if (
            manifest["original_campaign"]["sha256"] != ORIGINAL_CAMPAIGN_SHA256
            or manifest["approved_selection"]["sha256"] != APPROVED_SELECTION_SHA256
        ):
            raise ValueError("pilot task partition or approved prefix differs")
        original = json.loads(
            file_binding(dataset_dir, manifest["original_campaign"]).read_bytes()
        )
        selection = json.loads(
            file_binding(dataset_dir, manifest["approved_selection"]).read_bytes()
        )
        assignments = {row["sceneeval_id"]: row for row in original["tasks"]}
        supplement = {
            row["sceneeval_id"]: row for row in selection["supplemental_validation"]
        }
        if assignments.keys() & supplement.keys():
            raise ValueError("supplemental validation overlaps the original184")
        bridge = json.loads(
            file_binding(dataset_dir, manifest["source_bridge"]).read_bytes()
        )
        if (
            bridge.get("passed") is not True
            or bridge.get("old_source_fingerprint") != SOURCE_605
            or set(bridge.get("certified_original_ids", [])) != CERTIFIED_605_IDS
            or not bridge.get("new_source_fingerprint")
            or bridge["new_source_fingerprint"] == SOURCE_605
        ):
            raise ValueError(
                "six-certified605 source compatibility bridge is not proven"
            )
        for binding in bridge["evidence_bindings"]:
            file_binding(dataset_dir, binding)
        if not bridge["evidence_bindings"]:
            raise ValueError("source bridge has no actual proof bindings")
        entries = manifest["pairs"]
        rows_by_id = {}
        for split in ("train", "validation", "test"):
            path = dataset_dir / f"{split}.jsonl"
            if not path.exists():
                continue
            for line in path.read_bytes().splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                pair_id = row["pair_id"]
                if pair_id in rows_by_id:
                    raise ValueError("duplicate pair across dataset splits")
                rows_by_id[pair_id] = (
                    split,
                    hashlib.sha256(line).hexdigest(),
                    row["task_id"],
                )
        seen_ids, seen_pairs = set(), set()
        original_train, validation = 0, 0
        inherited = set()
        for entry in entries:
            original_id, pair_id, split = (
                entry["sceneeval_id"],
                entry["pair_id"],
                entry["split"],
            )
            if (
                type(original_id) is not int
                or original_id in seen_ids
                or pair_id in seen_pairs
            ):
                raise ValueError("qualification must have one pair per original ID")
            seen_ids.add(original_id)
            seen_pairs.add(pair_id)
            assignment = (
                assignments.get(original_id)
                if entry.get("cohort") == "original184"
                else supplement.get(original_id)
            )
            if not assignment or assignment["split"] != split:
                raise ValueError(
                    "qualified original ID is outside its frozen partition"
                )
            if rows_by_id.get(pair_id) != (
                split,
                entry["dataset_row_sha256"],
                assignment["task_id"],
            ):
                raise ValueError("qualified pair row or split changed")
            if split == "test" or entry.get("semantic_qualified") is not True:
                raise ValueError("test or semantically unqualified pair in pilot")
            if entry["source_fingerprint"] == SOURCE_605:
                if (
                    original_id not in CERTIFIED_605_IDS
                    or entry.get("cohort") != "original184"
                ):
                    raise ValueError(
                        "source bridge cannot inherit other old candidates"
                    )
                inherited.add(original_id)
            elif entry["source_fingerprint"] != bridge["new_source_fingerprint"]:
                raise ValueError("pair is outside the two explicitly bridged sources")
            for key in ("semantic_decision", "native_pair", "full_review_delivery"):
                file_binding(dataset_dir, entry[key])
            if split == "train":
                if entry.get("cohort") != "original184":
                    raise ValueError("training requires original frozen train tasks")
                original_train += 1
            elif split == "validation":
                validation += 1
            else:
                raise ValueError("unexpected qualification split")
        if seen_pairs != set(rows_by_id):
            raise ValueError("raw or unqualified rows remain in the dataset")
        if inherited != CERTIFIED_605_IDS:
            raise ValueError(
                "gap-filling profile requires exactly the six certified inherited originals"
            )
        if original_train < 8 or validation < 4:
            raise ValueError(
                f"pair-local pilot requires actual8originalT4independentV; observed{original_train}T{validation}V"
            )
    except (OSError, ValueError, KeyError, TypeError) as exc:
        errors.append(f"pair-local qualification gate: {exc}")
    return errors
