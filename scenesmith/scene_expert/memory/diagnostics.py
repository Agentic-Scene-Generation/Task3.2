"""Read-only replay of saved memory acceptance; no model or bank writes.

Run ``python -m scenesmith.scene_expert.memory.diagnostics --scene-dir PATH``.
Pass a scene directory, not the run root. JSON goes to stdout. This is an audit
of recorded evidence, not a scene-generation launcher or a causal benefit test.
"""

from __future__ import annotations

import argparse
import json

from pathlib import Path

from scenesmith.scene_expert.evaluation_io import read_object
from scenesmith.scene_expert.memory.adaptation import scoped_relations
from scenesmith.scene_expert.memory.injection import build_memory_injection_bundle
from scenesmith.scene_expert.memory.skill_policy import _category_compatible
from scenesmith.scene_expert.memory.usage import collect_memory_usage
from scenesmith.scene_expert.schemas import (
    MemoryPack,
    SceneTaskSpec,
    StageBrief,
    StageRelationContext,
)


def diagnose_scene(scene_dir: Path) -> dict:
    """Explain stored/replayed decisions without inventing missing evidence."""
    scene_dir = Path(scene_dir).resolve()
    warnings: list[str] = []
    activity = read_object(scene_dir / "scene_expert/memory_activity.json", warnings)
    try:
        entries = [
            (row["stage"], row["entry"])
            for row in activity.get("stage_attempt_history", [])
        ] + list((activity.get("stages") or {}).items())
    except (TypeError, KeyError, AttributeError):
        entries = []
        warnings.append("invalid_stage_activity")
    stages = []
    for index, (stage, entry) in enumerate(entries, 1):
        row = {
            "stage": stage,
            "stage_observation_index": index,
            "replay_matches_recorded": None,
        }
        try:
            injection = entry.get("injection") or {}
            row["recorded_decisions"] = injection.get("adaptation_decisions")
            row["recorded_accepted_ids"] = [
                [item["source"]["memory_type"], item["source"]["memory_id"]]
                for item in injection.get("accepted_items", [])
            ]
            if not all(
                key in injection
                for key in ("planner_stage_brief", "task_spec", "accepted_items")
            ):
                raise ValueError("Missing recorded planner/task/acceptance inputs")
            pack = MemoryPack.model_validate(entry["retrieval"])
            raw_brief = injection["planner_stage_brief"]
            raw_context = injection.get("relation_context") or entry.get(
                "relation_context"
            )
            brief = StageBrief.model_validate(raw_brief) if raw_brief else None
            bundle = build_memory_injection_bundle(
                stage=stage,
                stage_brief=brief,
                memory_pack=pack,
                task_spec=SceneTaskSpec.model_validate(injection["task_spec"]),
                relation_context=(
                    StageRelationContext.model_validate(raw_context)
                    if raw_context
                    else None
                ),
            )
            row["retrieved_count"] = len(pack.selections)
            row["planner_choice_count"] = len(brief.memory_adaptations) if brief else 0
            row["binding_audit"] = []
            for source in pack.selections:
                for choice in (brief.memory_adaptations if brief else []):
                    if (choice.memory_type, choice.memory_id) != (
                        source.memory_type,
                        source.memory_id,
                    ):
                        continue
                    roles = sorted(
                        {
                            str(relation.get(key) or "")
                            for relation in scoped_relations(source, choice)
                            for key in ("subject_role", "target_role")
                        }
                        - {""}
                    )
                    row["binding_audit"].append(
                        {
                            "memory_type": source.memory_type,
                            "memory_id": source.memory_id,
                            "source_relation_indices": choice.source_relation_indices,
                            "required_source_roles": roles,
                            "bindings": [
                                binding.model_dump() for binding in choice.bindings
                            ],
                            "unbound_source_roles": [
                                role
                                for role in roles
                                if not any(
                                    _category_compatible(role, binding.source_role)
                                    for binding in choice.bindings
                                )
                            ],
                        }
                    )
            row["replayed_decisions"] = bundle.adaptation_decisions
            row["replayed_accepted_ids"] = [
                [item.source.memory_type, item.source.memory_id]
                for item in bundle.accepted_items
            ]
            if row["recorded_decisions"] is not None:
                row["replay_matches_recorded"] = (
                    row["recorded_decisions"] == row["replayed_decisions"]
                    and row["recorded_accepted_ids"] == row["replayed_accepted_ids"]
                )
        except (ValueError, TypeError, KeyError, AttributeError) as exc:
            row["replay_error"] = str(exc)
            warnings.append(f"adaptation_replay_unavailable:{stage}:{index}")
        stages.append(row)
    usage = collect_memory_usage(scene_dir, activity)
    warnings.extend(usage["warnings"])
    if not stages:
        warnings.append("no_stage_activity")
    if any(row["replay_matches_recorded"] is False for row in stages):
        warnings.append("replay_differs_from_recorded_acceptance")
    return {
        "schema_version": "memory-diagnostics.v1",
        "scene_dir": str(scene_dir),
        "stages": stages,
        "usage": usage,
        "summary": {
            "retrieved": len(usage["items"]),
            "accepted": sum(item["accepted"] for item in usage["items"]),
            "delivered": sum(item["delivered"] is True for item in usage["items"]),
            "action_observed": sum(
                item["action_observed"] is True for item in usage["items"]
            ),
        },
        "warnings": sorted(set(warnings)),
        "interpretation": "Replay uses current local code, not a fresh Planner call. Matching acceptance and observed delivery are distinct from causal gain. Missing raw evidence is unknown, never proof of rejection or success.",
    }


def main() -> int:
    """Emit a bounded audit; optionally require actual delivery for a smoke test."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-dir", type=Path, required=True)
    parser.add_argument("--require-delivery", action="store_true")
    parser.add_argument(
        "--require-action",
        action="store_true",
        help="Also require a delivered item with related successful tool evidence (not causal gain)",
    )
    args = parser.parse_args()
    result = diagnose_scene(args.scene_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result["warnings"]:
        return 2
    if (args.require_delivery or args.require_action) and not result["summary"][
        "delivered"
    ]:
        return 1
    if args.require_action and not result["summary"]["action_observed"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
