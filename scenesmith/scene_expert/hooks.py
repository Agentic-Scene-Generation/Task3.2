"""SceneExpertHookRunner: pre/post-stage hooks injected into SceneSmith's _generate_room.

This is the main integration point between SceneExpert and SceneSmith.
It is created once per scene (in _generate_single_scene) and passed down to
_generate_room, where it is called before and after each stage agent runs.

Pre-stage hook:  Memory retrieval → StageBrief → injects into scene.text_description
Post-stage hook: Stage verification → Repair decision → Trace logging

Ablation mode controls which components are active:
  "disabled"         → hooks are never created; SceneSmith runs as-is
  "harness_only"     → Harness FSM + GlobalPlanner, NO memory retrieval
  "harness_memory"   → Harness FSM + GlobalPlanner + FastMemory (MVP default)
  "full"             → harness_memory + observer-only Slow Memory capture
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from scenesmith.agent_utils.room import RoomScene
from scenesmith.scene_expert.behavior import apply_behavior_template
from scenesmith.scene_expert.config_utils import (
    resolve_component_flags,
    resolve_scene_expert_config,
    resolve_stage_policies,
)
from scenesmith.scene_expert.context_bundle import build_stage_context_bundle
from scenesmith.scene_expert.experiment_identity import (
    stable_config_hash as _stable_config_hash,
    stable_experiment_signature as _stable_experiment_signature,
)
from scenesmith.scene_expert.failure_evidence import main_hard_failure_report
from scenesmith.scene_expert.global_planner import GlobalPlanner
from scenesmith.scene_expert.harness import Harness, RepairDecision
from scenesmith.scene_expert.memory.activity import MemoryActivityLogger
from scenesmith.scene_expert.memory.injection import build_memory_injection_bundle
from scenesmith.scene_expert.memory.retriever import MemoryRetriever
from scenesmith.scene_expert.memory.schemas import MemoryUtilityObservation
from scenesmith.scene_expert.memory.store import FastMemoryStore
from scenesmith.scene_expert.memory.writer import MemoryWriter
from scenesmith.scene_expert.relation_context import StageRelationProjector
from scenesmith.scene_expert.repair_controller import RepairController
from scenesmith.scene_expert.schemas import (
    FullVerifyReport,
    MemoryInjectionBundle,
    MemoryPack,
    RepairResult,
    SceneTaskSpec,
    StageBrief,
    StageExecutionEvidence,
    StageRelationContext,
    StageVerifyReport,
)
from scenesmith.scene_expert.slow_memory.trajectory import TrajectoryCollector
from scenesmith.scene_expert.task_compiler import TaskCompiler
from scenesmith.scene_expert.trace_logger import TraceLogger, collect_code_provenance
from scenesmith.scene_expert.verifier import FullVerifier, StageVerifier
from scenesmith.scenebenchmark_critic.config import critic_config_from_any
from scenesmith.scenebenchmark_critic.intent_compiler import IntentCompiler
from scenesmith.scenebenchmark_critic.object_taxonomy import (
    canonical_object_category,
    categories_are_equivalent,
    constraint_evaluation_stage,
    generation_owner,
    is_structural_anchor,
)
from scenesmith.scenebenchmark_critic.relation_registry import (
    STAGE_ORDER as CONTRACT_STAGE_ORDER,
)

console_logger = logging.getLogger(__name__)

# Valid ablation modes
ABLATION_MODES = frozenset(["disabled", "harness_only", "harness_memory", "full"])


@dataclass(frozen=True)
class StageCommitResult:
    """Whether a verified stage can advance the production pipeline."""

    stage: str
    passed: bool
    retryable: bool = False
    reason: str = ""
    quality_failure: bool = False


def _empty_memory_pack() -> MemoryPack:
    return MemoryPack(success_hints=[], failure_hints=[], skill_texts=[])


def _deep_merge_dicts(base: dict, override: dict) -> dict:
    """Merge nested dicts without mutating either input."""
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged


def _cfg_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "y", "on")
    return bool(value)


def _cfg_int(value: Any, default: int) -> int:
    if value is None or value == "":
        return default
    return int(value)


def _cfg_float(value: Any, default: float) -> float:
    if value is None or value == "":
        return default
    return float(value)


def _format_stage_relation_context(context: StageRelationContext) -> str:
    """Render the exact hard contract for the active stage."""
    text = (
        f"=== SceneExpert Hard Intent Contract: {context.stage} (authoritative) ===\n"
        + json.dumps(context.hard_constraints, ensure_ascii=False, sort_keys=True)
        + "\n=== End SceneExpert Hard Intent Contract ==="
    )
    if context.resolved_opening_reservations:
        text += (
            "\n\n=== Resolved Floor Plan Opening Reservations "
            "(authoritative geometry) ===\n"
            + json.dumps(
                context.resolved_opening_reservations,
                ensure_ascii=False,
                sort_keys=True,
            )
            + "\n=== End Resolved Floor Plan Opening Reservations ==="
        )
    return text


def _attach_stage_relation_context(
    scene: Any,
    *,
    relation_context: StageRelationContext | None,
    intent_contract: dict[str, Any] | None,
    task_spec: SceneTaskSpec,
) -> None:
    """Inject only stage hard rows while retaining the full critic contract."""
    if relation_context is not None:
        relation_text = _format_stage_relation_context(relation_context)
        scene.text_description = scene.text_description + "\n\n" + relation_text
        setattr(
            scene,
            "scene_expert_relation_context",
            relation_context.model_dump(mode="json"),
        )
    if intent_contract:
        setattr(scene, "scene_expert_intent_contract", intent_contract)
        setattr(scene, "scenebenchmark_intent_contract", intent_contract)
        metadata = getattr(scene, "metadata", None)
        if not isinstance(metadata, dict):
            metadata = {}
            setattr(scene, "metadata", metadata)
        metadata["scenebenchmark_intent_contract"] = intent_contract
    task_spec_payload = task_spec.model_dump()
    setattr(scene, "scene_expert_task_spec", task_spec_payload)
    metadata = getattr(scene, "metadata", None)
    if not isinstance(metadata, dict):
        metadata = {}
        setattr(scene, "metadata", metadata)
    metadata["scene_expert_task_spec"] = task_spec_payload


def _build_hybrid_retriever(
    memory_store: FastMemoryStore,
    memory_dir: str,
    memory_cfg: dict,
    ret_cfg: dict,
    timing_path: Path | None = None,
    exclude_source_task_id: str = "",
):
    """Construct the optional hybrid retriever from memory config."""
    from scenesmith.scene_expert.memory.embedding import SceneMemoryEmbedder
    from scenesmith.scene_expert.memory.hybrid_retriever import HybridMemoryRetriever
    from scenesmith.scene_expert.memory.scoring import HybridScoreWeights

    emb_cfg = memory_cfg.get("embedding", {})
    idx_cfg = memory_cfg.get("index", {})
    backend = idx_cfg.get("backend", "numpy")
    if backend != "numpy":
        raise NotImplementedError(
            f"SceneExpert hybrid memory currently supports numpy index only, got {backend!r}."
        )

    weights_cfg = memory_cfg.get("hybrid_weights", {})
    weights = HybridScoreWeights(
        embedding_similarity=_cfg_float(weights_cfg.get("embedding_similarity"), 0.45),
        object_overlap=_cfg_float(weights_cfg.get("object_overlap"), 0.20),
        room_stage_match=_cfg_float(weights_cfg.get("room_stage_match"), 0.15),
        memory_quality_score=_cfg_float(weights_cfg.get("memory_quality_score"), 0.10),
        recency_or_verified=_cfg_float(weights_cfg.get("recency_or_verified"), 0.10),
    )

    embedder = SceneMemoryEmbedder(
        model_dir=emb_cfg.get("model_dir"),
        model_id=emb_cfg.get("model_id", "BAAI/bge-m3"),
        device=emb_cfg.get("device", "cpu"),
        batch_size=_cfg_int(emb_cfg.get("batch_size"), 8),
        max_length=_cfg_int(emb_cfg.get("max_length"), 512),
        normalize=_cfg_bool(emb_cfg.get("normalize"), True),
    )
    return HybridMemoryRetriever(
        store=memory_store,
        memory_dir=memory_dir,
        embedder=embedder,
        index_dir=idx_cfg.get("dir"),
        max_success=_cfg_int(ret_cfg.get("max_success_cases"), 3),
        max_failure=_cfg_int(ret_cfg.get("max_failure_cases"), 3),
        max_skills=_cfg_int(ret_cfg.get("max_skills"), 2),
        recall_top_k=_cfg_int(ret_cfg.get("recall_top_k"), 30),
        sim_threshold=_cfg_float(ret_cfg.get("sim_threshold"), 0.0),
        object_overlap_threshold=_cfg_float(
            ret_cfg.get("object_overlap_threshold"),
            0.15,
        ),
        weights=weights,
        require_indexes=_cfg_bool(idx_cfg.get("require_ready"), True),
        auto_build_indexes=_cfg_bool(idx_cfg.get("auto_build_missing"), False),
        timing_path=timing_path,
        exclude_source_task_id=exclude_source_task_id,
    )


def _intent_compiler_model(cfg_dict: dict) -> str:
    return (
        cfg_dict.get("furniture_agent", {})
        .get("openai", {})
        .get(
            "model",
            cfg_dict.get("llm", {}).get("model_id", "Qwen/Qwen3.5-35B-A3B"),
        )
    )


def _compile_intent_contract_if_enabled(
    *,
    prompt: str,
    scene_id: int,
    output_dir: Path,
    cfg_dict: dict,
    task_spec: SceneTaskSpec | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Compile v5 exactly once when the embedded critic is enabled.

    The private config entries are consumed by ``_generate_room`` so the
    contract survives the floor-plan boundary even when SceneExpert itself is
    disabled.  The compiler itself falls back to the deterministic prompt
    parser when the model cannot return a valid contract.
    """
    critic_config = critic_config_from_any(cfg_dict)
    if not critic_config.enabled:
        return {}, {}
    normalized_prompt = " ".join(str(prompt or "").split())
    prompt_hash = hashlib.sha256(normalized_prompt.encode("utf-8")).hexdigest()
    task_spec_payload = (
        task_spec.model_dump(mode="json", exclude_none=True) if task_spec else {}
    )
    task_spec_hash = hashlib.sha256(
        json.dumps(
            task_spec_payload,
            ensure_ascii=False,
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    cache_key = {
        "prompt_sha256": prompt_hash,
        "task_spec_sha256": task_spec_hash,
        "spec_version": IntentCompiler.SPEC_VERSION,
        "schema_version": IntentCompiler.SCHEMA_VERSION,
    }
    cached_contract = cfg_dict.get("_scenebenchmark_intent_contract")
    cached_trace = cfg_dict.get("_scenebenchmark_intent_trace")
    cached_key = cfg_dict.get("_scenebenchmark_intent_cache_key")
    if (
        isinstance(cached_contract, dict)
        and isinstance(cached_trace, dict)
        and cached_key == cache_key
    ):
        return cached_contract, cached_trace
    compiler_cfg = critic_config.intent_compiler
    compiler = IntentCompiler(
        model=_intent_compiler_model(cfg_dict),
        api_base_url=os.environ.get("OPENAI_BASE_URL", "http://localhost:8000/v1"),
        api_key=os.environ.get("OPENAI_API_KEY", "dummy"),
        max_tokens=_cfg_int(compiler_cfg.get("max_tokens"), 8192),
        temperature=0.0,
    )
    try:
        contract = compiler.compile(prompt, task_spec=task_spec)
    except Exception:
        trace = getattr(compiler, "last_trace", {})
        trace_path = (
            output_dir
            / f"scene_{scene_id:03d}"
            / "scene_expert"
            / "trace"
            / "intent_compiler.json"
        )
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        trace_path.write_text(
            json.dumps(trace, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
            newline="\n",
        )
        raise
    trace = dict(getattr(compiler, "last_trace", {}))
    trace_path = (
        output_dir
        / f"scene_{scene_id:03d}"
        / "scene_expert"
        / "trace"
        / "intent_compiler.json"
    )
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    trace_path.write_text(
        json.dumps(trace, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
        newline="\n",
    )
    cfg_dict["_scenebenchmark_intent_contract"] = contract
    cfg_dict["_scenebenchmark_intent_trace"] = trace
    cfg_dict["_scenebenchmark_intent_cache_key"] = cache_key
    return contract, trace


_TASK_SPEC_STAGE_FIELDS = {
    "furniture": "required_large_objects",
    "wall_mounted": "required_wall_objects",
    "ceiling_mounted": "required_ceiling_objects",
    "manipuland": "required_small_objects",
}
_NON_OBJECT_INVENTORY_CATEGORIES = frozenset(
    {
        "",
        "room",
        "floor",
        "ceiling",
        "wall",
        "back_wall",
        "front_wall",
        "side_wall",
        "main_wall",
        "opposite_wall",
        "adjacent_wall",
        "entrance",
        "entry",
        "door",
        "opening",
        "window",
    }
)
_GENERIC_INVENTORY_CATEGORIES = frozenset({"chair", "desk", "table"})


def _inventory_category(value: Any) -> str:
    return canonical_object_category(value)


def _categories_match_inventory(first: str, second: str) -> bool:
    if categories_are_equivalent(first, second):
        return True
    if first in _GENERIC_INVENTORY_CATEGORIES:
        return second.endswith(f"_{first}")
    if second in _GENERIC_INVENTORY_CATEGORIES:
        return first.endswith(f"_{second}")
    return False


def _contract_inventory_ownership(
    contract: dict[str, Any], existing_owners: dict[str, str]
) -> tuple[dict[str, tuple[str, int]], list[str]]:
    """Return contract-owned inventory categories and their generation stages."""
    counts: dict[str, int] = {}
    supported_counts: dict[str, dict[tuple[str, str, str, str], int]] = {}
    owner_evidence: dict[str, dict[int, set[str]]] = {}

    owner_bearing_relations = {
        "on_top_of",
        "on_floor",
        "object_on_floor",
        "mounted_on_wall",
        "hung_on_wall",
        "mounted_on_ceiling",
        "hung_from_ceiling",
    }

    def record(selector: Any, stage: str, *, priority: int) -> None:
        if not isinstance(selector, dict):
            return
        category = _inventory_category(selector.get("category"))
        if category in _NON_OBJECT_INVENTORY_CATEGORIES or is_structural_anchor(
            category
        ):
            return
        try:
            count = max(1, int(selector.get("count") or 1))
        except (TypeError, ValueError):
            count = 1
        counts[category] = max(counts.get(category, 0), count)
        owner_evidence.setdefault(category, {}).setdefault(priority, set()).add(stage)

    constraints = contract.get("constraints") if isinstance(contract, dict) else []
    for constraint in constraints or []:
        if not isinstance(constraint, dict):
            continue
        if str(constraint.get("strength") or "hard").lower() != "hard":
            continue
        relation = str(constraint.get("relation") or "")
        explicit_relation = relation != "required_count"
        subject = constraint.get("subjects")
        subject_category = _inventory_category(
            subject.get("category") if isinstance(subject, dict) else ""
        )
        if subject_category not in _NON_OBJECT_INVENTORY_CATEGORIES:
            subject_stage = generation_owner(
                subject_category,
                relation=relation,
                endpoint="subject",
                declared_owner=existing_owners.get(subject_category, ""),
            )
            subject_priority = (
                3
                if explicit_relation and relation in owner_bearing_relations
                else 2 if explicit_relation else 0
            )
            record(subject, subject_stage, priority=subject_priority)

            if relation == "on_top_of":
                target = constraint.get("targets") or {}
                target_category = _inventory_category(target.get("category"))
                if target_category not in _NON_OBJECT_INVENTORY_CATEGORIES:
                    support_key = (
                        target_category,
                        str(target.get("role") or ""),
                        _inventory_category(target.get("secondary_category")),
                        str(target.get("secondary_role") or ""),
                    )
                    try:
                        supported_count = max(1, int(subject.get("count") or 1))
                    except (TypeError, ValueError):
                        supported_count = 1
                    per_support = supported_counts.setdefault(subject_category, {})
                    per_support[support_key] = max(
                        per_support.get(support_key, 0), supported_count
                    )

        # A target can be the only explicit mention of an object. It keeps its
        # intrinsic owner instead of inheriting a relation's later stage.
        target = constraint.get("targets")
        target_category = _inventory_category(
            target.get("category") if isinstance(target, dict) else ""
        )
        record(
            target,
            generation_owner(
                target_category,
                relation=relation,
                endpoint="target",
                declared_owner=existing_owners.get(target_category, ""),
            ),
            priority=1 if explicit_relation else 0,
        )

    # Distinct support cohorts cannot consume the same physical instance. A
    # prompt requiring objects on both a bed and desks therefore needs the sum
    # of those minima, while duplicate descriptions of the same cohort retain
    # only their maximum count.
    for category, per_support in supported_counts.items():
        if len(per_support) > 1:
            counts[category] = max(counts.get(category, 0), sum(per_support.values()))

    ownership: dict[str, tuple[str, int]] = {}
    conflicts: list[str] = []
    for category, count in counts.items():
        evidence = owner_evidence.get(category) or {}
        if not evidence:
            continue
        candidate_owners = evidence[max(evidence)]
        if len(candidate_owners) > 1:
            conflicts.append(
                f"category {category!r} has conflicting strongest ownership "
                f"evidence: {sorted(candidate_owners)}"
            )
        ownership[category] = (
            max(candidate_owners, key=CONTRACT_STAGE_ORDER.index),
            count,
        )
    return ownership, conflicts


def _reconcile_task_spec_stage_ownership(
    task_spec: SceneTaskSpec, contract: dict[str, Any]
) -> SceneTaskSpec:
    """Move contract-owned inventory to its validated pipeline stage.

    The legacy TaskCompiler remains a recall fallback for categories absent from
    the independent intent contract. Contract-covered categories, however, must
    not be generated or verified before their dependencies exist.
    """
    existing_owners = {
        _inventory_category(label): stage
        for stage, field in _TASK_SPEC_STAGE_FIELDS.items()
        for label in getattr(task_spec, field)
    }
    ownership, _conflicts = _contract_inventory_ownership(contract, existing_owners)
    if not ownership:
        return task_spec

    # A larger sum of mutually exclusive support cohorts supersedes a smaller
    # global exact count. Preserve it as a minimum so valid extra instances do
    # not make the reconciled contract self-contradictory.
    for constraint in contract.get("constraints") or []:
        if not isinstance(constraint, dict):
            continue
        if str(constraint.get("relation") or "") != "required_count":
            continue
        subjects = constraint.get("subjects") or {}
        category = _inventory_category(subjects.get("category"))
        owned_category = next(
            (
                owned
                for owned in ownership
                if _categories_match_inventory(category, owned)
            ),
            None,
        )
        if owned_category is None:
            continue
        desired_count = ownership[owned_category][1]
        try:
            current_count = max(1, int(subjects.get("count") or 1))
        except (TypeError, ValueError):
            current_count = 1
        if desired_count <= current_count:
            continue
        subjects["count"] = desired_count
        subjects["quantifier"] = "minimum"
        constraint["subjects"] = subjects
        constraint["reconciliation_reason"] = "disjoint_support_cohort_minimum"

    # StageRelationProjector consumes the contract's ``stage`` field directly.
    # Keep it aligned with the inventory reconciliation so an object cannot be
    # generated by furniture and then projected only to manipuland.
    for constraint in contract.get("constraints") or []:
        if not isinstance(constraint, dict):
            continue
        relation = str(constraint.get("relation") or "")
        subjects = constraint.get("subjects") or {}
        targets = constraint.get("targets") or {}

        def reconciled_endpoint_stage(selector: dict[str, Any], endpoint: str) -> str:
            category = _inventory_category(selector.get("category"))
            owned_category = next(
                (
                    owned
                    for owned in ownership
                    if _categories_match_inventory(category, owned)
                ),
                None,
            )
            if owned_category is not None:
                return ownership[owned_category][0]
            return generation_owner(
                category,
                relation=relation,
                endpoint=endpoint,
                declared_owner=existing_owners.get(category, ""),
            )

        endpoint_stages = [
            reconciled_endpoint_stage(subjects, "subject"),
            reconciled_endpoint_stage(targets, "target"),
        ]
        constraint["stage"] = constraint_evaluation_stage(*endpoint_stages)

    reconciled = {stage: [] for stage in _TASK_SPEC_STAGE_FIELDS}
    matched_counts = {category: 0 for category in ownership}
    for source_stage, field in _TASK_SPEC_STAGE_FIELDS.items():
        for label in getattr(task_spec, field):
            category = _inventory_category(label)
            if is_structural_anchor(category):
                continue
            matched_category = next(
                (
                    owned
                    for owned in ownership
                    if _categories_match_inventory(category, owned)
                ),
                None,
            )
            if matched_category is None:
                reconciled[source_stage].append(label)
                continue
            target_stage, _count = ownership[matched_category]
            reconciled.get(target_stage, reconciled[source_stage]).append(label)
            matched_counts[matched_category] += 1

    for category, (stage, count) in ownership.items():
        if stage in reconciled:
            reconciled[stage].extend(
                [category] * max(0, count - matched_counts[category])
            )

    updates = {
        field: reconciled[stage] for stage, field in _TASK_SPEC_STAGE_FIELDS.items()
    }
    return task_spec.model_copy(update=updates)


def _audit_stage_ownership(
    before: SceneTaskSpec,
    after: SceneTaskSpec,
    contract: dict[str, Any],
) -> dict[str, Any]:
    """Validate inventory preservation and relation-stage ordering."""
    owners: dict[str, set[str]] = {}
    before_counts: dict[str, int] = {}
    after_counts: dict[str, int] = {}
    for stage, field in _TASK_SPEC_STAGE_FIELDS.items():
        for label in getattr(before, field):
            category = _inventory_category(label)
            if category and not is_structural_anchor(category):
                before_counts[category] = before_counts.get(category, 0) + 1
        for label in getattr(after, field):
            category = _inventory_category(label)
            if category and not is_structural_anchor(category):
                owners.setdefault(category, set()).add(stage)
                after_counts[category] = after_counts.get(category, 0) + 1

    errors: list[str] = []
    before_owners = {
        _inventory_category(label): stage
        for stage, field in _TASK_SPEC_STAGE_FIELDS.items()
        for label in getattr(before, field)
    }
    _ownership, ownership_conflicts = _contract_inventory_ownership(
        contract, before_owners
    )
    errors.extend(ownership_conflicts)
    for category, stages in sorted(owners.items()):
        if len(stages) != 1:
            errors.append(
                f"category {category!r} has multiple generation owners: {sorted(stages)}"
            )
    for category, count in sorted(before_counts.items()):
        if after_counts.get(category, 0) < count:
            errors.append(
                f"category {category!r} lost inventory instances: "
                f"before={count} after={after_counts.get(category, 0)}"
            )

    def endpoint_owner(selector: Any, relation: str, endpoint: str) -> str | None:
        if not isinstance(selector, dict):
            return None
        category = _inventory_category(selector.get("category"))
        if not category or category in _NON_OBJECT_INVENTORY_CATEGORIES:
            return None
        if is_structural_anchor(category):
            return "floor_plan"
        owned_category = next(
            (known for known in owners if _categories_match_inventory(category, known)),
            None,
        )
        declared = (
            next(iter(owners[owned_category])) if owned_category is not None else ""
        )
        return generation_owner(
            category,
            relation=relation,
            endpoint=endpoint,
            declared_owner=declared,
        )

    for constraint in contract.get("constraints") or []:
        if not isinstance(constraint, dict):
            continue
        relation = str(constraint.get("relation") or "")
        endpoint_owners = [
            owner
            for owner in (
                endpoint_owner(constraint.get("subjects"), relation, "subject"),
                endpoint_owner(constraint.get("targets"), relation, "target"),
            )
            if owner is not None
        ]
        if not endpoint_owners:
            continue
        expected = constraint_evaluation_stage(*endpoint_owners)
        actual = str(constraint.get("stage") or "")
        if actual not in CONTRACT_STAGE_ORDER or CONTRACT_STAGE_ORDER.index(
            actual
        ) < CONTRACT_STAGE_ORDER.index(expected):
            errors.append(
                f"constraint {constraint.get('constraint_id') or '<unknown>'} "
                f"stage {actual!r} precedes endpoint owner {expected!r}"
            )

    return {
        "status": "ok" if not errors else "invalid",
        "generation_owners": {
            category: next(iter(stages))
            for category, stages in sorted(owners.items())
            if len(stages) == 1
        },
        "before_counts": before_counts,
        "after_counts": after_counts,
        "errors": errors,
    }


class SceneExpertHookRunner:
    """Per-scene hook runner that wraps SceneSmith stage execution.

    One instance is created per scene (prompt). It holds the task spec,
    all SceneExpert module references, and accumulated per-stage trace data.

    Thread safety: NOT thread-safe. Use one instance per scene.
    """

    def __init__(
        self,
        prompt: str,
        scene_id: int,
        output_dir: Path,
        mode: str,
        component_flags: dict[str, bool],
        task_spec: SceneTaskSpec,
        harness: Harness,
        global_planner: GlobalPlanner,
        relation_projector: StageRelationProjector,
        retriever: Any | None,
        stage_verifier: StageVerifier,
        full_verifier: FullVerifier,
        repair_controller: RepairController,
        trace_logger: TraceLogger | None,
        memory_writer: MemoryWriter | None,
        memory_store: FastMemoryStore | None,
        qwen_model: str,
        trajectory_collector: TrajectoryCollector | None = None,
        experiment_name: str = "",
        config_hash: str = "",
        experiment_signature: str = "",
        start_stage: str = "floor_plan",
        allow_long_term_memory_updates: bool = True,
        intent_contract: dict[str, Any] | None = None,
        intent_trace: dict[str, Any] | None = None,
        task_compiler_trace: dict[str, Any] | None = None,
        critic_config: Any | None = None,
        stage_policies: dict[str, str] | None = None,
    ) -> None:
        self._prompt = prompt
        self._scene_id = scene_id
        self._output_dir = output_dir
        self._mode = mode
        self._component_flags = dict(component_flags)
        self._scene_debug_dir = output_dir / f"scene_{scene_id:03d}" / "scene_expert"
        self._retrieval_timing_path = (
            self._scene_debug_dir / "timing" / "memory_retrieval.jsonl"
        )
        self._context_debug_dir = self._scene_debug_dir / "context_bundles"

        self._task_spec = task_spec
        self._harness = harness
        self._global_planner = global_planner
        self._relation_projector = relation_projector
        self._retriever = retriever
        self._stage_verifier = stage_verifier
        self._full_verifier = full_verifier
        self._repair_controller = repair_controller
        self._trace_logger = trace_logger
        self._memory_writer = memory_writer
        self._memory_store = memory_store
        self._trajectory_collector = trajectory_collector
        self._qwen_model = qwen_model
        self._experiment_name = experiment_name
        self._config_hash = config_hash
        self._experiment_signature = experiment_signature
        self._start_stage = start_stage
        # A normal, intentionally truncated pipeline (for example the
        # floor-plan-only shared base used by critic probes) is not a complete
        # scene outcome.  It may read memory and emit local audit artifacts,
        # but it must not promote long-term memories or update skill utility.
        self._allow_long_term_memory_updates = allow_long_term_memory_updates
        self._intent_contract = dict(intent_contract or {})
        self._intent_trace = dict(intent_trace or {})
        self._critic_config = critic_config
        self._stage_policies = dict(stage_policies or {})
        self._stage_order_baseline = self._initial_completed_stages(start_stage)
        self._room_start_stage = (
            "furniture" if start_stage == "floor_plan" else start_stage
        )
        self._room_stage_order_baseline = self._initial_completed_stages(
            self._room_start_stage
        )

        if self._trace_enabled():
            self._trace_logger.record_task_compiler(task_spec, task_compiler_trace)
        if self._intent_trace and self._trace_enabled():
            self._trace_logger.record_intent_compiler(self._intent_trace)
        self._stage_reports: list[StageVerifyReport] = []
        self._completed_stages: list[str] = list(self._stage_order_baseline)
        self._qwen_calls = len(self._intent_trace.get("attempts") or [])
        self._memory_activity = MemoryActivityLogger(
            self._scene_debug_dir,
            scene_id=f"scene_{self._scene_id:03d}",
            task_spec=self._task_spec,
            experiment_signature=self._experiment_signature,
            task_id=(
                "task_" + hashlib.sha256(self._prompt.encode("utf-8")).hexdigest()[:16]
            ),
            run_id=str(self._output_dir.resolve()),
        )
        self._pending_skill_observations: list[MemoryUtilityObservation] = []

        # Current stage state (populated in pre_stage, consumed in post_stage)
        self._current_stage: str = ""
        self._current_memory_pack: MemoryPack = _empty_memory_pack()
        self._current_stage_brief: StageBrief | None = None
        self._current_injection_bundle = MemoryInjectionBundle(stage="")
        self._current_execution_evidence = StageExecutionEvidence()
        self._current_relation_context: StageRelationContext | None = None
        self._current_planner_trace: dict[str, Any] = {}
        self._current_stage_policy: str = "auto"
        self._stage_start_time: float = 0.0
        # Original text_description per stage (so we can restore if needed)
        self._original_text_descriptions: dict[str, str] = {}
        self._last_injected_floor_plan_prompt: str = prompt
        # Failed stage attempts are retried by the room pipeline. Keep the
        # instruction and original report until a subsequent verification passes.
        self._pending_stage_repairs: dict[
            str, tuple[RepairResult, StageVerifyReport]
        ] = {}
        self._latest_deterministic_payload: dict[str, Any] | None = None
        self._latest_scene: RoomScene | None = None

    @property
    def floor_plan_reservation_manifest(self) -> dict[str, Any] | None:
        context = self._current_relation_context
        if context is None or context.floor_plan_manifest is None:
            return None
        return context.floor_plan_manifest.model_dump(mode="json")

    def stage_policy(self, stage: str) -> str:
        """Return the resolved non-skipping policy for one stage."""
        return str(self._stage_policies.get(stage, "auto"))

    def should_skip_stage_agent(self, stage: str) -> bool:
        """Backward-compatible guard that never skips a native SceneSmith stage."""
        if self._current_planner_trace.get("status") == "no_op":
            console_logger.warning(
                "[SceneExpert] Ignoring legacy no_op for %s; native stage execution "
                "is mandatory under stage_policy=%s",
                stage,
                self.stage_policy(stage),
            )
        return False

    def mark_stage_agent_invoked(self, stage: str) -> None:
        """Record the native-agent boundary without changing main's execution."""
        if stage != self._current_stage:
            console_logger.error(
                "[SceneExpert] Stage invocation mismatch: current=%s invoked=%s",
                self._current_stage,
                stage,
            )
        self._current_execution_evidence.stage_agent_invoked = True
        self._write_stage_policy_audit()

    def _write_stage_policy_audit(self) -> None:
        """Persist required/optional policy resolution as a non-fatal artifact."""
        if not self._current_stage:
            return
        try:
            audit_dir = self._scene_debug_dir / "stage_policy"
            audit_dir.mkdir(parents=True, exist_ok=True)
            brief = self._current_stage_brief
            payload = {
                "schema_version": "scenesmith.stage_policy.v1",
                "stage": self._current_stage,
                "configured_policy": self.stage_policy(self._current_stage),
                "effective_policy": self._current_stage_policy,
                "optional_assets_allowed": (
                    brief.optional_assets_allowed
                    if brief is not None
                    else self._current_stage_policy == "auto"
                ),
                "required_objects": self._stage_required_objects(self._current_stage),
                "optional_asset_recommendations": (
                    [
                        item.model_dump(mode="json")
                        for item in brief.optional_asset_recommendations
                    ]
                    if brief is not None
                    else []
                ),
                "planner_status": self._current_planner_trace.get(
                    "status", "disabled_or_unavailable"
                ),
                "stage_agent_invoked": (
                    self._current_execution_evidence.stage_agent_invoked
                ),
                "stage_skip_allowed": False,
            }
            path = audit_dir / f"{self._current_stage}.json"
            path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception as error:
            console_logger.warning(
                "[SceneExpert] Failed to write stage-policy audit for %s: %s",
                self._current_stage,
                error,
            )

    def accept_degraded_stage(self, stage: str) -> None:
        """Record a terminal quality-failed stage as advanced by the pipeline.

        The pipeline may deliberately retain a deterministic, non-retryable
        quality failure for probe output.  It must still advance the harness
        FSM; otherwise the next stage is rejected as an apparent stage skip.
        Verification errors and retryable failures never reach this method.
        """
        if stage != self._current_stage:
            raise ValueError(
                "Cannot accept degraded stage "
                f"{stage!r}; current stage is {self._current_stage!r}"
            )
        if self._component_enabled("harness"):
            self._harness.validate_stage_order(self._completed_stages, stage)
        self._completed_stages.append(stage)
        self._pending_stage_repairs.pop(stage, None)
        console_logger.warning(
            "[SceneExpert] Accepted stage %s with degraded quality", stage
        )

    def _component_enabled(self, name: str) -> bool:
        """Return one resolved feature gate for this run."""
        return bool(self._component_flags.get(name, False))

    def _trace_enabled(self) -> bool:
        """Return whether trace output is enabled and initialized."""
        return self._component_enabled("trace") and self._trace_logger is not None

    def _build_execution_evidence(self, prompt: str) -> StageExecutionEvidence:
        """Capture proof that optional context crossed the designer boundary."""
        brief_text = self._current_injection_bundle.brief_text
        memory_text = self._current_injection_bundle.memory_text
        placement_reference = self._current_injection_bundle.placement_text
        final_text = self._current_injection_bundle.final_text
        return StageExecutionEvidence(
            task_spec_source=(
                "fallback"
                if self._task_spec.compiler_status == "degraded"
                else "compiled"
            ),
            stage_brief_source=(
                "global_planner"
                if self._current_stage_brief is not None
                else "disabled_or_unavailable"
            ),
            stage_policy=self._current_stage_policy,
            optional_assets_allowed=(
                self._current_stage_brief.optional_assets_allowed
                if self._current_stage_brief is not None
                else self._current_stage_policy == "auto"
            ),
            required_objects=self._stage_required_objects(self._current_stage),
            optional_asset_recommendations=(
                [
                    item.model_dump(mode="json")
                    for item in self._current_stage_brief.optional_asset_recommendations
                ]
                if self._current_stage_brief is not None
                else []
            ),
            retrieved_memory_ids=self._current_injection_bundle.selected_memory_ids,
            retrieved_skill_names=(
                self._current_injection_bundle.retrieved_skill_names
            ),
            planner_selected_skill_names=(
                self._current_injection_bundle.planner_selected_skill_names
            ),
            prompt_delivered_skill_names=(
                self._current_injection_bundle.prompt_delivered_skill_names
            ),
            injected_brief_hash=(
                hashlib.sha256(brief_text.encode("utf-8")).hexdigest()
                if brief_text
                else ""
            ),
            injected_memory_hash=(
                hashlib.sha256(memory_text.encode("utf-8")).hexdigest()
                if memory_text
                else ""
            ),
            designer_prompt_hash=hashlib.sha256(
                str(prompt).encode("utf-8")
            ).hexdigest(),
            designer_prompt_contains_brief=bool(
                brief_text and brief_text in str(prompt)
            ),
            designer_prompt_contains_memory=bool(
                self._current_injection_bundle.selected_memory_ids
                and final_text
                and final_text in str(prompt)
            ),
            placement_reference_injected=bool(
                placement_reference and placement_reference in str(prompt)
            ),
            final_injection_hash=(
                hashlib.sha256(final_text.encode("utf-8")).hexdigest()
                if final_text
                else ""
            ),
            experiment_signature=self._experiment_signature,
            degraded=self._task_spec.compiler_status == "degraded",
        )

    def _record_memory_pre_stage_activity(self) -> None:
        """Persist exact retrieval, planner, and injection state for this stage."""
        try:
            self._memory_activity.record_pre_stage(
                stage=self._current_stage,
                memory_pack=self._current_memory_pack,
                relation_context=self._current_relation_context,
                planner_trace=self._current_planner_trace,
                injection_bundle=self._current_injection_bundle,
                execution_evidence=self._current_execution_evidence,
            )
        except Exception as error:
            console_logger.warning(
                "[SceneExpert] Failed to record memory activity for %s: %s",
                self._current_stage,
                error,
            )

    def _record_memory_post_stage_activity(
        self,
        *,
        stage: str,
        verify_report: StageVerifyReport | None,
        repair_actions: list[RepairResult],
        scene_state_path: str,
    ) -> None:
        """Persist the critic outcome linked to the exact retrieved records."""
        try:
            observations = self._memory_activity.record_post_stage(
                stage=stage,
                verify_report=verify_report,
                repair_actions=repair_actions,
                scene_state_path=scene_state_path,
            )
            if isinstance(observations, list):
                pending = list(getattr(self, "_pending_skill_observations", []))
                pending.extend(observations)
                self._pending_skill_observations = pending
        except Exception as error:
            console_logger.warning(
                "[SceneExpert] Failed to record post-stage memory activity for %s: %s",
                stage,
                error,
            )

    def _flush_skill_outcomes(self) -> dict[str, Any] | None:
        """Commit verified skill utility once, after all stage retrieval is done."""
        observations = list(getattr(self, "_pending_skill_observations", []))
        if self._memory_store is None or not observations:
            return None
        try:
            summary = self._memory_store.record_skill_outcomes(observations)
            self._memory_activity.record_skill_learning(summary=summary)
            self._pending_skill_observations = []
            return summary
        except Exception as error:
            console_logger.warning(
                "[SceneExpert] Failed to persist skill utility observations: %s",
                error,
            )
            return None

    def _capture_main_repair_activity(self) -> None:
        """Mirror main repair evidence into the SceneExpert audit, read-only."""
        try:
            self._memory_activity.capture_main_repair_events(
                self._scene_debug_dir / "timing" / "repair_events.jsonl",
                self._scene_debug_dir / "timing" / "stage_working_timing.jsonl",
            )
        except Exception as error:
            console_logger.warning(
                "[SceneExpert] Failed to capture deterministic repair activity: %s",
                error,
            )
    def record_runtime_failure_continuation(self, provenance: dict[str, Any]) -> None:
        """Attach checkpoint-gated runtime salvage to the stage trace evidence."""
        if not isinstance(provenance, dict):
            return
        self._current_execution_evidence.degraded = True
        self._current_execution_evidence.continuation_policy = str(
            provenance.get("continuation_policy") or ""
        )
        self._current_execution_evidence.runtime_failure = dict(
            provenance.get("failure") or {}
        )

    def _save_context_bundle(
        self,
        *,
        stage: str,
        agent_role: str,
        event: str,
        scene: RoomScene | None = None,
        prompt: Any = "",
        last_hard_issues: list[str] | None = None,
    ) -> None:
        """Save a structured pre-LLM context snapshot for audit/debug."""
        if not self._trace_enabled():
            return
        try:
            bundle = build_stage_context_bundle(
                stage=stage,
                agent_role=agent_role,
                event=event,
                task_spec=self._task_spec,
                relation_context=self._current_relation_context,
                stage_brief=self._current_stage_brief,
                scene=scene,
                memory_pack=self._current_memory_pack,
                history_summary=(
                    self._build_scene_state_summary() if scene is not None else ""
                ),
                last_hard_issues=last_hard_issues or [],
                prompt=prompt,
                trace_id=f"trace_{self._scene_id:06d}",
                scene_id=f"scene_{self._scene_id:03d}",
                metadata={
                    "mode": self._mode,
                    "experiment_name": self._experiment_name,
                    "config_hash": self._config_hash,
                    "experiment_signature": self._experiment_signature,
                },
            )
            safe_stage = "".join(
                c if c.isalnum() or c in ("-", "_") else "_" for c in stage
            )
            safe_event = "".join(
                c if c.isalnum() or c in ("-", "_") else "_" for c in event
            )
            path = (
                self._context_debug_dir
                / safe_stage
                / f"{int(time.time() * 1000)}_{agent_role}_{safe_event}.json"
            )
            bundle.save(path)
        except Exception as e:
            console_logger.warning(
                "[SceneExpert] Failed to save StageContextBundle for %s/%s: %s",
                stage,
                event,
                e,
            )

    def _record_memory_retrieval_timing(
        self,
        *,
        stage: str,
        elapsed_sec: float,
        pack: MemoryPack | None = None,
        error: str = "",
    ) -> None:
        """Record pre-stage memory retrieval timing even for empty/fallback stores."""
        if not self._trace_enabled():
            return
        if not error and bool(
            getattr(self._retriever, "writes_detailed_timing", False)
        ):
            return
        try:
            record = {
                "schema_version": "1.0",
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "stage": stage,
                "module": "scene_expert_memory_retrieval",
                "retriever": (
                    type(self._retriever).__name__
                    if self._retriever is not None
                    else "none"
                ),
                "elapsed_sec": round(float(elapsed_sec), 6),
                "success_hints": len(pack.success_hints) if pack else 0,
                "failure_hints": len(pack.failure_hints) if pack else 0,
                "skills": len(pack.skill_texts) if pack else 0,
                "has_placement_reference": bool(pack and pack.placement_reference),
                "error": error,
            }
            self._retrieval_timing_path.parent.mkdir(parents=True, exist_ok=True)
            with self._retrieval_timing_path.open(
                "a",
                encoding="utf-8",
                newline="\n",
            ) as f:
                f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        except Exception as timing_error:
            console_logger.warning(
                "Failed to record SceneExpert memory retrieval timing: %s",
                timing_error,
            )

    def _stage_score_quality(self, report: StageVerifyReport) -> float:
        if not report.visual_scores:
            return 0.0
        return max(
            0.0,
            min(
                1.0,
                sum(report.visual_scores.values()) / len(report.visual_scores),
            ),
        )

    def _run_stage_verifier(self, **kwargs: Any) -> StageVerifyReport:
        """Run post-stage verification only when its independent gate is active."""
        stage = str(kwargs.get("stage", self._current_stage))
        if not self._component_enabled("verifier"):
            return StageVerifyReport(
                stage=stage,
                pass_stage=True,
                critique_summary="Verifier disabled by component gate.",
            )
        return self._stage_verifier.verify(**kwargs)

    def _commit_stage_memory(
        self,
        *,
        stage: str,
        verify_report: StageVerifyReport | None,
        scene_state_path: str,
        repair_actions: list[RepairResult],
    ) -> None:
        """Persist stage evidence without bypassing final memory promotion.

        Stage and critic events are valuable even when a scene later fails, but
        they are not independently sufficient for reusable long-term memory.
        The final strict MemoryWriter is the only active-bank promotion path.
        """
        if (
            self._memory_store is None
            or verify_report is None
            or not self._component_enabled("stage_working_memory")
            or not self._component_enabled("verifier")
        ):
            return
        try:
            quality = self._stage_score_quality(verify_report)
            event = {
                "schema_version": "1.0",
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "event_type": "stage_verify",
                "trace_id": f"trace_{self._scene_id:06d}",
                "scene_id": f"scene_{self._scene_id:03d}",
                "stage": stage,
                "scene_state_path": scene_state_path,
                "pass_stage": verify_report.pass_stage,
                "quality_score": quality,
                "scores": verify_report.visual_scores,
                "issues": [issue.model_dump() for issue in verify_report.issues],
                "repair_actions": [
                    (
                        action.model_dump()
                        if hasattr(action, "model_dump")
                        else getattr(action, "__dict__", str(action))
                    )
                    for action in repair_actions
                ],
                "critique_summary": verify_report.critique_summary[:2000],
                "retrieved_memory_ids": list(
                    dict.fromkeys(
                        [
                            *self._current_memory_pack.success_case_ids,
                            *self._current_memory_pack.failure_case_ids,
                            *self._current_memory_pack.skill_names,
                        ]
                    )
                ),
                "retrieved_source_task_ids": dict(
                    self._current_memory_pack.retrieved_source_task_ids
                ),
                "retrieved_source_run_ids": dict(
                    self._current_memory_pack.retrieved_source_run_ids
                ),
                "memory_bank_id": self._current_memory_pack.memory_bank_id,
                "memory_bank_revision": self._current_memory_pack.memory_bank_revision,
                "execution_evidence": self._current_execution_evidence.model_dump(),
                "promotion_status": "evidence_only",
            }
            self._memory_store.append_event(event)
        except Exception as e:
            console_logger.warning(
                "[SceneExpert] Stage-level public memory commit failed for %s: %s",
                stage,
                e,
            )

    def _stage_required_objects(self, stage: str) -> list[str]:
        if stage == "floor_plan":
            return list(self._task_spec.required_architectural_features)
        if stage == "furniture":
            return list(self._task_spec.required_large_objects)
        if stage == "wall_mounted":
            return list(self._task_spec.required_wall_objects)
        if stage == "ceiling_mounted":
            return list(self._task_spec.required_ceiling_objects)
        if stage == "manipuland":
            return list(self._task_spec.required_small_objects)
        return []

    def _inject_pending_stage_repair(self, stage: str, scene: RoomScene) -> bool:
        """Append the failed stage's deterministic repair instruction once retried."""
        pending_repair = self._pending_stage_repairs.get(stage)
        if pending_repair is None:
            return False
        instruction = pending_repair[0].repair_action.strip()
        if not instruction:
            return False
        scene.text_description += "\n\n[REPAIR INSTRUCTION]\n" + instruction
        pending_repair[0].execution_status = "executed"
        return True

    # ------------------------------------------------------------------
    # Pre-stage hook: called BEFORE the SceneSmith stage agent runs
    # ------------------------------------------------------------------

    def pre_floor_plan(self) -> str:
        """Prepare SceneExpert context for the house-level floor_plan stage.

        Floor plan generation runs in an isolated subprocess and receives only a
        prompt string, so this returns an enhanced prompt instead of mutating a
        RoomScene.
        """
        stage = "floor_plan"
        console_logger.info(f"[SceneExpert/{self._mode}] pre_stage: {stage}")
        if self._component_enabled("harness"):
            self._validate_stage_transition(stage)
        self._current_stage = stage
        self._current_stage_policy = self.stage_policy(stage)
        self._stage_start_time = time.time()
        self._qwen_calls = 0

        self._current_relation_context = self._relation_projector.project(
            stage=stage,
            task_spec=self._task_spec,
            intent_contract=self._intent_contract,
        )

        if self._retriever is not None and self._component_enabled(
            "fast_memory_retrieval"
        ):
            try:
                retrieval_start = time.time()
                self._current_memory_pack = self._retriever.retrieve(
                    self._task_spec,
                    stage,
                    relation_context=self._current_relation_context,
                )
                retrieval_elapsed = time.time() - retrieval_start
                n_hints = len(self._current_memory_pack.success_hints) + len(
                    self._current_memory_pack.failure_hints
                )
                self._record_memory_retrieval_timing(
                    stage=stage,
                    elapsed_sec=retrieval_elapsed,
                    pack=self._current_memory_pack,
                )
                console_logger.info(
                    f"[SceneExpert] Memory retrieved for {stage}: "
                    f"{n_hints} hints, {len(self._current_memory_pack.skill_texts)} skills "
                    f"in {retrieval_elapsed:.2f}s"
                )
            except Exception as e:
                self._record_memory_retrieval_timing(
                    stage=stage,
                    elapsed_sec=(
                        time.time() - retrieval_start
                        if "retrieval_start" in locals()
                        else 0.0
                    ),
                    pack=None,
                    error=str(e),
                )
                console_logger.warning(f"Memory retrieval failed for {stage}: {e}")
                self._current_memory_pack = _empty_memory_pack()
        else:
            self._current_memory_pack = _empty_memory_pack()

        self._current_stage_brief = None
        self._current_planner_trace = {}
        if self._component_enabled("global_planner"):
            try:
                planner_start = time.time()
                context = self._harness.build_context(
                    stage=stage,
                    task_spec=self._task_spec,
                    memory_pack=self._current_memory_pack,
                    relation_context=self._current_relation_context,
                    stage_policy=self._current_stage_policy,
                )
                self._current_stage_brief = self._global_planner.generate_stage_brief(
                    context=context,
                    scene_state_summary="No floor plan has been generated yet.",
                    original_task=self._prompt,
                )
                self._current_planner_trace = dict(
                    getattr(self._global_planner, "last_trace", {}) or {}
                )
                self._qwen_calls += len(
                    [
                        item
                        for item in self._current_planner_trace.get("attempts", [])
                        if int(item.get("attempt", 99)) < 2
                    ]
                )
                console_logger.info(
                    f"[SceneExpert] StageBrief generated for {stage}: "
                    f"{len(self._current_stage_brief.constraints_for_designer)} constraints "
                    f"in {time.time() - planner_start:.2f}s"
                )
            except Exception as e:
                console_logger.warning(
                    f"GlobalPlanner failed for {stage}, running without StageBrief: {e}"
                )

        self._current_injection_bundle = build_memory_injection_bundle(
            stage=stage,
            stage_brief=self._current_stage_brief,
            memory_pack=self._current_memory_pack,
        )
        self._current_stage_brief = self._current_injection_bundle.enriched_stage_brief
        enhanced = self._prompt
        if (
            self._component_enabled("prompt_injection")
            and self._current_injection_bundle.final_text
        ):
            enhanced += "\n\n" + self._current_injection_bundle.final_text
        if self._current_relation_context is not None:
            enhanced += "\n\n" + _format_stage_relation_context(
                self._current_relation_context
            )
        self._last_injected_floor_plan_prompt = enhanced
        self._current_execution_evidence = self._build_execution_evidence(enhanced)
        self._write_stage_policy_audit()
        self._record_memory_pre_stage_activity()
        self._save_context_bundle(
            stage=stage,
            agent_role="global_planner",
            event="pre_floor_plan",
            prompt=enhanced,
        )
        if self._trace_enabled():
            self._trace_logger.save_stage_context(
                stage=stage,
                memory_pack=self._current_memory_pack,
                relation_context=self._current_relation_context,
                stage_brief=self._current_stage_brief,
                phase="pre",
                execution_evidence=self._current_execution_evidence,
            )
        return enhanced

    def post_floor_plan(self, scene_dir: Path) -> None:
        """Verify and log the house-level floor_plan stage."""
        stage = "floor_plan"
        console_logger.info(f"[SceneExpert/{self._mode}] post_stage: {stage}")

        manifest = self.floor_plan_reservation_manifest
        if manifest and manifest.get("enabled"):
            from scenesmith.agent_utils.house import HouseLayout
            from scenesmith.floor_plan_agents.reservation_validator import (
                validate_floor_plan_reservations,
            )

            layout_path = scene_dir / "house_layout.json"
            with layout_path.open(encoding="utf-8") as stream:
                layout = HouseLayout.from_dict(json.load(stream), house_dir=scene_dir)
            deterministic = validate_floor_plan_reservations(layout, manifest)
            if not deterministic.passed:
                issue_types = [
                    str(issue.get("issue_type") or "reservation_failure")
                    for issue in deterministic.issues
                ]
                raise RuntimeError(
                    "Floor plan failed post-stage reservation validation: "
                    + ", ".join(issue_types)
                )

        scene_state_info = self._extract_floor_plan_state_info(scene_dir)
        verify_report: StageVerifyReport | None = None
        repair_actions: list[RepairResult] = []
        try:
            verify_start = time.time()
            verify_report = self._run_stage_verifier(
                stage=stage,
                stage_output_dir=str(scene_dir),
                task_spec=self._task_spec,
                stage_brief=self._current_stage_brief,
                scene_state_info=scene_state_info,
            )
            console_logger.info(
                "[SceneExpertTiming] stage=%s module=stage_verifier elapsed=%.2fs",
                stage,
                time.time() - verify_start,
            )
            self._stage_reports.append(verify_report)

            if not verify_report.pass_stage:
                console_logger.warning(
                    f"[SceneExpert] Stage {stage} FAILED verification: "
                    f"issues={[i.issue_type for i in verify_report.issues]}"
                )
                if self._component_enabled("repair"):
                    decision = self._harness.decide_repair(stage, verify_report)
                else:
                    decision = RepairDecision(
                        should_repair=False,
                        strategy="skip",
                        reason="Repair disabled by component gate",
                    )
                if decision.should_repair:
                    repair_result = self._repair_controller.repair(
                        repair_type=decision.strategy,
                        stage=stage,
                        verify_report=verify_report,
                        scene_path=str(scene_dir),
                        stage_brief=self._current_stage_brief,
                        task_spec=self._task_spec,
                    )
                    repair_actions.append(repair_result)
                    self._repair_controller.record_failure_to_memory(
                        stage=stage,
                        room_type=self._task_spec.room_type,
                        repair_result=repair_result,
                        verify_report=verify_report,
                        repair_verified=False,
                    )
            else:
                console_logger.info(f"[SceneExpert] Stage {stage} PASSED verification")
        except Exception as e:
            console_logger.warning(
                f"[SceneExpert] Verification failed for {stage}: {e}"
            )

        elapsed = time.time() - self._stage_start_time
        self._commit_stage_memory(
            stage=stage,
            verify_report=verify_report,
            scene_state_path=str(scene_dir),
            repair_actions=repair_actions,
        )
        self._record_memory_post_stage_activity(
            stage=stage,
            verify_report=verify_report,
            repair_actions=repair_actions,
            scene_state_path=str(scene_dir),
        )
        if self._trace_enabled():
            self._trace_logger.log_stage(
                stage=stage,
                memory_pack=self._current_memory_pack,
                relation_context=self._current_relation_context,
                planner_trace=self._current_planner_trace,
                stage_brief=self._current_stage_brief,
                scene_state_path=str(scene_dir),
                verify_report=verify_report,
                repair_actions=repair_actions,
                qwen_calls=self._qwen_calls,
                stage_time_sec=round(elapsed, 1),
                execution_evidence=self._current_execution_evidence,
            )
            self._trace_logger.save_stage_context(
                stage=stage,
                memory_pack=self._current_memory_pack,
                relation_context=self._current_relation_context,
                stage_brief=self._current_stage_brief,
                phase="post",
                execution_evidence=self._current_execution_evidence,
            )
            self._trace_logger.save_stage_visual_manifest(stage, str(scene_dir))
        self._completed_stages.append(stage)
        console_logger.info(
            "[SceneExpertTiming] stage=%s module=stage_total elapsed=%.2fs",
            stage,
            elapsed,
        )

    def pre_stage(self, stage: str, scene: RoomScene) -> None:
        """Retrieve memory, generate StageBrief, inject into scene.text_description.

        Called from _generate_room immediately before each stage's agent is built.

        Args:
            stage: Current stage name (e.g., "furniture").
            scene: The RoomScene that will be passed to the stage agent.
        """
        console_logger.info(f"[SceneExpert/{self._mode}] pre_stage: {stage}")
        if self._component_enabled("harness"):
            self._validate_stage_transition(stage)
        self._current_stage = stage
        self._current_stage_policy = self.stage_policy(stage)
        self._stage_start_time = time.time()
        self._qwen_calls = 0

        # Save original text_description for restoration after stage
        self._original_text_descriptions[stage] = scene.text_description
        # Keep an explicit immutable task-facing description on the scene.
        # Stage agents use this for deterministic requirement inference and
        # room-type dispatch; SceneExpert's injected brief/memory remains
        # available separately for LLM prompting.
        setattr(scene, "scene_expert_original_description", scene.text_description)

        self._current_relation_context = self._relation_projector.project(
            stage=stage,
            task_spec=self._task_spec,
            intent_contract=self._intent_contract,
            scene=scene,
        )

        # --- Step 1: Memory retrieval (skip in harness_only mode) ---
        if self._retriever is not None and self._component_enabled(
            "fast_memory_retrieval"
        ):
            try:
                retrieval_start = time.time()
                self._current_memory_pack = self._retriever.retrieve(
                    self._task_spec,
                    stage,
                    relation_context=self._current_relation_context,
                )
                retrieval_elapsed = time.time() - retrieval_start
                n_hints = len(self._current_memory_pack.success_hints) + len(
                    self._current_memory_pack.failure_hints
                )
                self._record_memory_retrieval_timing(
                    stage=stage,
                    elapsed_sec=retrieval_elapsed,
                    pack=self._current_memory_pack,
                )
                console_logger.info(
                    f"[SceneExpert] Memory retrieved for {stage}: "
                    f"{n_hints} hints, {len(self._current_memory_pack.skill_texts)} skills "
                    f"in {retrieval_elapsed:.2f}s"
                )
            except Exception as e:
                self._record_memory_retrieval_timing(
                    stage=stage,
                    elapsed_sec=(
                        time.time() - retrieval_start
                        if "retrieval_start" in locals()
                        else 0.0
                    ),
                    pack=None,
                    error=str(e),
                )
                console_logger.warning(f"Memory retrieval failed for {stage}: {e}")
                self._current_memory_pack = _empty_memory_pack()
        else:
            self._current_memory_pack = _empty_memory_pack()

        # --- Step 2: Global Planner -> StageBrief ---
        self._current_stage_brief = None
        self._current_planner_trace = {}
        if self._component_enabled("global_planner"):
            try:
                planner_start = time.time()
                scene_state_summary = self._build_scene_state_summary()
                context = self._harness.build_context(
                    stage=stage,
                    task_spec=self._task_spec,
                    memory_pack=self._current_memory_pack,
                    relation_context=self._current_relation_context,
                    stage_policy=self._current_stage_policy,
                )
                self._current_stage_brief = self._global_planner.generate_stage_brief(
                    context=context,
                    scene_state_summary=scene_state_summary,
                    original_task=str(
                        getattr(scene, "scene_expert_original_description", "")
                        or self._prompt
                    ),
                )
                self._current_planner_trace = dict(
                    getattr(self._global_planner, "last_trace", {}) or {}
                )
                self._qwen_calls += len(
                    [
                        item
                        for item in self._current_planner_trace.get("attempts", [])
                        if int(item.get("attempt", 99)) < 2
                    ]
                )
                console_logger.info(
                    f"[SceneExpert] StageBrief generated for {stage}: "
                    f"{len(self._current_stage_brief.constraints_for_designer)} constraints "
                    f"in {time.time() - planner_start:.2f}s"
                )
            except Exception as e:
                console_logger.warning(
                    f"GlobalPlanner failed for {stage}, running without StageBrief: {e}"
                )

        # --- Step 3: Build and inject one canonical memory-aware bundle ---
        self._current_injection_bundle = build_memory_injection_bundle(
            stage=stage,
            stage_brief=self._current_stage_brief,
            memory_pack=self._current_memory_pack,
        )
        self._current_stage_brief = self._current_injection_bundle.enriched_stage_brief
        injection_text = self._current_injection_bundle.final_text
        if self._component_enabled("prompt_injection") and injection_text:
            scene.text_description += "\n\n" + injection_text
            setattr(scene, "scene_expert_brief", injection_text)
            if self._current_injection_bundle.memory_text:
                setattr(
                    scene,
                    "scene_expert_memory_directives",
                    self._current_injection_bundle.memory_text,
                )
            briefs = getattr(scene, "scene_expert_briefs", {})
            if not isinstance(briefs, dict):
                briefs = {}
            briefs[stage] = injection_text
            setattr(scene, "scene_expert_briefs", briefs)
            console_logger.debug(
                f"[SceneExpert] Injected StageBrief into scene.text_description for {stage}"
            )
        if (
            self._component_enabled("prompt_injection")
            and self._current_injection_bundle.placement_text
        ):
            console_logger.info(
                f"[SceneExpert] Injected placement reference for {stage} "
                f"({self._current_injection_bundle.placement_text.count(chr(10))+1} lines)"
            )
        if self._inject_pending_stage_repair(stage, scene):
            console_logger.info(
                "[SceneExpert] Injected retry repair instruction for %s", stage
            )
        _attach_stage_relation_context(
            scene,
            relation_context=self._current_relation_context,
            intent_contract=self._intent_contract,
            task_spec=self._task_spec,
        )
        setattr(scene, "scene_expert_stage", stage)
        setattr(
            scene,
            "scene_expert_slow_memory_capture_enabled",
            self._component_enabled("slow_memory_capture"),
        )
        self._save_context_bundle(
            stage=stage,
            agent_role="designer",
            event="pre_stage",
            scene=scene,
            prompt=scene.text_description,
        )
        self._current_execution_evidence = self._build_execution_evidence(
            scene.text_description
        )
        self._write_stage_policy_audit()
        self._record_memory_pre_stage_activity()
        if self._trace_enabled():
            self._trace_logger.save_stage_context(
                stage=stage,
                memory_pack=self._current_memory_pack,
                relation_context=self._current_relation_context,
                stage_brief=self._current_stage_brief,
                phase="pre",
                execution_evidence=self._current_execution_evidence,
            )

    # ------------------------------------------------------------------
    # Post-stage hook: called AFTER the SceneSmith stage agent completes
    # ------------------------------------------------------------------

    def post_stage(
        self, stage: str, scene: RoomScene, room_dir: Path
    ) -> StageCommitResult:
        """Verify a stage and return whether it may advance the pipeline.

        Called from _generate_room immediately after the stage's checkpoint is saved.
        A failed result remains uncommitted.  The room pipeline uses the returned
        retry request to reload the prior checkpoint, execute the same stage, and
        invoke this hook again for deterministic re-verification.

        Args:
            stage: Completed stage name.
            scene: The RoomScene after stage completion.
            room_dir: Room output directory (for finding scores.yaml).
        """
        console_logger.info(f"[SceneExpert/{self._mode}] post_stage: {stage}")

        # Restore original text_description (keep scene clean for next stage)
        if stage in self._original_text_descriptions:
            scene.text_description = self._original_text_descriptions[stage]
        self._latest_scene = scene

        # Extract lightweight scene state info for rule checks
        scene_state_info = self._extract_scene_state_info_from_scene(scene)

        # Verify stage
        verify_report: StageVerifyReport | None = None
        repair_actions: list[RepairResult] = []
        passed = False
        retryable = False
        verification_error = False
        result_reason = ""
        try:
            verify_start = time.time()
            deterministic_critic_payload = None
            critic_config = getattr(self, "_critic_config", None)
            if critic_config is not None and critic_config.enabled:
                from scenesmith.scenebenchmark_critic.api import evaluate_room_scene

                deterministic_critic_payload = evaluate_room_scene(
                    scene,
                    config=critic_config,
                    stage=f"{stage}_post_stage",
                    annotate_assets=False,
                )
                self._latest_deterministic_payload = deterministic_critic_payload
            verify_report = self._run_stage_verifier(
                stage=stage,
                stage_output_dir=str(room_dir),
                task_spec=self._task_spec,
                stage_brief=self._current_stage_brief,
                scene_state_info=scene_state_info,
                deterministic_critic_payload=deterministic_critic_payload,
            )
            console_logger.info(
                "[SceneExpertTiming] stage=%s module=stage_verifier elapsed=%.2fs",
                stage,
                time.time() - verify_start,
            )
            if not verify_report.pass_stage:
                console_logger.warning(
                    f"[SceneExpert] Stage {stage} FAILED verification: "
                    f"issues={[i.issue_type for i in verify_report.issues]}"
                )
                if self._component_enabled("repair"):
                    decision = self._harness.decide_repair(stage, verify_report)
                else:
                    decision = RepairDecision(
                        should_repair=False,
                        strategy="skip",
                        reason="Repair disabled by component gate",
                    )
                result_reason = decision.reason
                if decision.should_repair:
                    repair_result = self._repair_controller.repair(
                        repair_type=decision.strategy,
                        stage=stage,
                        verify_report=verify_report,
                        scene_path=str(room_dir),
                        stage_brief=self._current_stage_brief,
                        task_spec=self._task_spec,
                    )
                    repair_actions.append(repair_result)
                    self._pending_stage_repairs[stage] = (
                        repair_result,
                        verify_report,
                    )
                    retryable = True
                    self._repair_controller.record_failure_to_memory(
                        stage=stage,
                        room_type=self._task_spec.room_type,
                        repair_result=repair_result,
                        verify_report=verify_report,
                        repair_verified=False,
                    )
                if not retryable:
                    self._stage_reports.append(verify_report)
            else:
                passed = True
                self._stage_reports.append(verify_report)
                prior_repair = self._pending_stage_repairs.pop(stage, None)
                if prior_repair is not None:
                    repair_result, failed_report = prior_repair
                    repair_result.repair_verified = True
                    repair_result.new_scene_state = str(room_dir)
                    repair_actions.append(repair_result)
                    self._repair_controller.record_failure_to_memory(
                        stage=stage,
                        room_type=self._task_spec.room_type,
                        repair_result=repair_result,
                        verify_report=failed_report,
                        repair_verified=True,
                    )
                console_logger.info(f"[SceneExpert] Stage {stage} PASSED verification")

        except Exception as e:
            verification_error = True
            result_reason = f"verification error: {e}"
            console_logger.warning(
                f"[SceneExpert] Verification failed for {stage}: {e}"
            )

        # Log stage trace entry
        elapsed = time.time() - self._stage_start_time
        self._commit_stage_memory(
            stage=stage,
            verify_report=verify_report,
            scene_state_path=str(room_dir),
            repair_actions=repair_actions,
        )
        self._record_memory_post_stage_activity(
            stage=stage,
            verify_report=verify_report,
            repair_actions=repair_actions,
            scene_state_path=str(room_dir),
        )
        trajectory_collector = getattr(self, "_trajectory_collector", None)
        if trajectory_collector is not None:
            try:
                try:
                    final_scene_context = build_stage_context_bundle(
                        stage=stage,
                        agent_role="designer",
                        event="post_stage_observation",
                        task_spec=self._task_spec,
                        relation_context=self._current_relation_context,
                        stage_brief=self._current_stage_brief,
                        scene=scene,
                        memory_pack=self._current_memory_pack,
                        history_summary=self._build_scene_state_summary(),
                        last_hard_issues=[
                            str(issue.description or issue.issue_type)
                            for issue in (verify_report.issues if verify_report else [])
                        ],
                        trace_id=f"trace_{self._scene_id:06d}",
                        scene_id=f"scene_{self._scene_id:03d}",
                        metadata={
                            "observer_only": True,
                            "mode": self._mode,
                            "config_hash": self._config_hash,
                            "experiment_signature": self._experiment_signature,
                        },
                    ).model_dump(mode="json")
                except Exception as context_exc:
                    final_scene_context = {}
                    console_logger.warning(
                        "[SceneExpert] Slow-memory final context capture failed "
                        "(trajectory capture will continue): %s",
                        context_exc,
                    )
                capture_summary = trajectory_collector.capture_stage(
                    stage=stage,
                    verify_report=verify_report,
                    repair_actions=repair_actions,
                    final_scene_context=final_scene_context,
                    scene_state_path=str(room_dir),
                )
                if self._trace_enabled():
                    self._trace_logger.record_component_status(
                        "slow_memory_capture",
                        {
                            "success": True,
                            "observer_only": True,
                            "stage": stage,
                            **capture_summary,
                        },
                    )
            except Exception as exc:
                console_logger.warning(
                    "[SceneExpert] Slow-memory capture failed (non-fatal): %s",
                    exc,
                    exc_info=True,
                )
        if self._trace_enabled():
            self._trace_logger.log_stage(
                stage=stage,
                memory_pack=self._current_memory_pack,
                relation_context=self._current_relation_context,
                planner_trace=self._current_planner_trace,
                stage_brief=self._current_stage_brief,
                scene_state_path=str(room_dir),
                verify_report=verify_report,
                repair_actions=repair_actions,
                qwen_calls=self._qwen_calls,
                stage_time_sec=round(elapsed, 1),
                execution_evidence=self._current_execution_evidence,
            )
            self._trace_logger.save_stage_context(
                stage=stage,
                memory_pack=self._current_memory_pack,
                relation_context=self._current_relation_context,
                stage_brief=self._current_stage_brief,
                phase="post",
                execution_evidence=self._current_execution_evidence,
            )
            self._trace_logger.save_stage_visual_manifest(stage, str(room_dir))
        if passed:
            self._completed_stages.append(stage)
        console_logger.info(
            "[SceneExpertTiming] stage=%s module=stage_total elapsed=%.2fs",
            stage,
            elapsed,
        )
        return StageCommitResult(
            stage=stage,
            passed=passed,
            retryable=retryable,
            reason=result_reason,
            quality_failure=(
                verify_report is not None and not passed and not verification_error
            ),
        )

    # ------------------------------------------------------------------
    # Finalize: called after all stages complete
    # ------------------------------------------------------------------

    def _write_long_term_memory(
        self, full_report: FullVerifyReport
    ) -> dict[str, Any] | None:
        """Run the strict writer and atomically apply its evidence-gated ops."""
        if (
            self._memory_writer is None
            or self._memory_store is None
            or not self._component_enabled("memory_writer")
            or not self._component_enabled("verifier")
        ):
            return None
        try:
            memory_start = time.time()
            trace_summary = (
                self._trace_logger.build_trace_summary()
                if self._trace_enabled()
                else json.dumps(
                    [report.model_dump() for report in self._stage_reports],
                    ensure_ascii=False,
                    default=str,
                )
            )
            related_old_memory = self._format_related_memory_for_writer()
            ops = self._memory_writer.write(
                trace_summary=trace_summary,
                full_report=full_report,
                related_old_memory=related_old_memory,
                evidence_payload=(
                    self._trace_logger.build_memory_writer_evidence()
                    if self._trace_enabled()
                    else {
                        "trace_id": f"trace_{self._scene_id:06d}",
                        "run_id": str(self._output_dir.resolve()),
                        "prompt": self._prompt,
                        "experiment_name": self._experiment_name,
                        "config_hash": self._config_hash,
                        "experiment_signature": self._experiment_signature,
                        "task_spec": self._task_spec.model_dump(),
                        "stages": [
                            {
                                "stage": report.stage,
                                "verify_report": report.model_dump(),
                                "repair_actions": [],
                            }
                            for report in self._stage_reports
                        ],
                    }
                ),
            )
            if self._trace_enabled():
                self._trace_logger.save_memory_update_ops(ops, full_report)
            apply_summary = self._memory_store.apply_updates(ops)
            self._memory_activity.record_writer(
                proposed_ops=ops,
                writer_trace=dict(self._memory_writer.last_trace),
                apply_summary=apply_summary,
            )
            if self._trace_enabled():
                self._trace_logger.record_component_status(
                    "memory_writer",
                    {
                        **dict(self._memory_writer.last_trace),
                        "store_apply": apply_summary,
                    },
                )
            console_logger.info(
                "[SceneExpert] Memory update: %d proposed, %d added, "
                "%d merged, revision=%d in %.2fs",
                len(ops),
                apply_summary["added"],
                apply_summary["merged"],
                apply_summary["revision"],
                time.time() - memory_start,
            )
            return apply_summary
        except Exception as e:
            console_logger.warning(f"Memory update failed (non-fatal): {e}")
            try:
                self._memory_activity.record_writer(
                    proposed_ops=[],
                    writer_trace=(
                        dict(self._memory_writer.last_trace)
                        if self._memory_writer is not None
                        else {}
                    ),
                    apply_summary=None,
                    error=f"{type(e).__name__}: {e}",
                )
            except Exception as activity_error:
                console_logger.warning(
                    "[SceneExpert] Failed to record MemoryWriter failure: %s",
                    activity_error,
                )
            if self._trace_enabled():
                self._trace_logger.record_component_status(
                    "memory_writer",
                    {
                        "success": False,
                        "degraded": True,
                        "source": "exception",
                        "write_status": "exception_no_write",
                        "fallback_written": False,
                        "error": f"{type(e).__name__}: {e}",
                    },
                )
                self._trace_logger.save_memory_update_ops([], full_report)
            return None

    def finalize(self, final_scene_path: str) -> FullVerifyReport:
        """Run full verifier, save trace, update memory.

        Called from _generate_single_scene after _run_sequential_room_generation
        returns and before the function exits.

        Args:
            final_scene_path: Path to the final scene output directory.
        """
        console_logger.info(
            f"[SceneExpert/{self._mode}] finalizing scene {self._scene_id:03d}"
        )
        finalize_start = time.time()

        # Full verifier
        full_report = FullVerifyReport()
        try:
            full_verify_start = time.time()
            if self._component_enabled("verifier"):
                critic_config = getattr(self, "_critic_config", None)
                latest_scene = getattr(self, "_latest_scene", None)
                if (
                    critic_config is not None
                    and critic_config.enabled
                    and latest_scene is not None
                ):
                    from scenesmith.scenebenchmark_critic.api import evaluate_room_scene

                    self._latest_deterministic_payload = evaluate_room_scene(
                        latest_scene,
                        config=critic_config,
                        stage="final_scene_verification",
                        annotate_assets=False,
                    )
                full_report = self._full_verifier.verify(
                    stage_reports=self._stage_reports,
                    final_scene_path=final_scene_path,
                    deterministic_critic_payload=self._latest_deterministic_payload,
                )
            console_logger.info(
                "[SceneExpertTiming] stage=full_scene module=full_verifier elapsed=%.2fs",
                time.time() - full_verify_start,
            )
        except Exception as e:
            console_logger.warning(f"FullVerifier failed: {e}")

        # Save trace
        final_path = Path(final_scene_path)
        combined_path = (
            final_path
            if final_path.name == "combined_house"
            else final_path / "combined_house"
        )
        exports = {
            "scene_dir": final_scene_path,
            "drake": str(combined_path / "house.dmd.yaml"),
            "blend": str(combined_path / "house.blend"),
        }
        if self._trace_enabled():
            self._trace_logger.finalize(
                full_report=full_report,
                exports=exports,
                model=self._qwen_model,
            )

        # Only terminal pipeline runs own a complete scene outcome.  Shared
        # bases deliberately stop at floor_plan and are later resumed by a
        # critic-on run; promoting their partial reports would contaminate the
        # active bank and count one task twice.
        if getattr(self, "_allow_long_term_memory_updates", True):
            self._write_long_term_memory(full_report)
            self._flush_skill_outcomes()
        else:
            self._pending_skill_observations = []
            if self._trace_enabled():
                self._trace_logger.record_component_status(
                    "memory_writer",
                    {
                        "success": True,
                        "skipped": True,
                        "write_status": "skipped_non_terminal_pipeline",
                        "reason": (
                            "Long-term memory promotion requires a run whose "
                            "configured stop_stage is manipuland."
                        ),
                    },
                )
            console_logger.info(
                "[SceneExpert] Skipped long-term memory and skill updates for "
                "non-terminal pipeline run"
            )

        if self._trace_enabled():
            trace_dict = self._trace_logger.finalize(
                full_report=full_report,
                exports=exports,
                model=self._qwen_model,
            )
            trace_path = self._trace_logger.save(trace_dict)
            console_logger.info(f"[SceneExpert] Trace saved to {trace_path}")
        self._capture_main_repair_activity()

        console_logger.info(
            f"[SceneExpert] Scene {self._scene_id:03d} complete: "
            f"overall={full_report.overall_score:.2f} "
            f"pass={'YES' if full_report.pass_scene else 'NO'} "
            f"mode={self._mode}"
        )
        console_logger.info(
            "[SceneExpertTiming] stage=full_scene module=finalize_total elapsed=%.2fs",
            time.time() - finalize_start,
        )
        return full_report

    def finalize_failure(self, error: str = "") -> None:
        """Persist a failed trace and curate only recognized main hard-gate evidence.

        This does not catch, suppress, retry, or otherwise change main's failure.
        It gives the additive memory layer a chance to learn a negative lesson
        from an authoritative deterministic gate that fires before ``post_stage``.
        Arbitrary runtime/infrastructure exceptions remain trace-only.
        """
        if not self._trace_enabled():
            trajectory_collector = getattr(self, "_trajectory_collector", None)
            failure_report = main_hard_failure_report(error, self._current_stage)
            if trajectory_collector is not None and failure_report is not None:
                try:
                    trajectory_collector.capture_stage(
                        stage=self._current_stage,
                        verify_report=failure_report,
                        repair_actions=[],
                    )
                except Exception as exc:
                    console_logger.warning(
                        "[SceneExpert] Failure trajectory capture failed: %s", exc
                    )
            return
        failure_report = main_hard_failure_report(error, self._current_stage)
        if failure_report is None:
            self.save_partial_trace(error=error)
            return

        if self._current_stage not in self._completed_stages:
            self._stage_reports.append(failure_report)
            self._trace_logger.log_stage(
                stage=self._current_stage,
                memory_pack=self._current_memory_pack,
                relation_context=self._current_relation_context,
                planner_trace=self._current_planner_trace,
                stage_brief=self._current_stage_brief,
                scene_state_path=str(self._scene_debug_dir.parent),
                verify_report=failure_report,
                repair_actions=[],
                qwen_calls=self._qwen_calls,
                stage_time_sec=max(0.0, time.time() - self._stage_start_time),
                execution_evidence=self._current_execution_evidence,
            )
            self._commit_stage_memory(
                stage=self._current_stage,
                verify_report=failure_report,
                scene_state_path=str(self._scene_debug_dir.parent),
                repair_actions=[],
            )
            self._record_memory_post_stage_activity(
                stage=self._current_stage,
                verify_report=failure_report,
                repair_actions=[],
                scene_state_path=str(self._scene_debug_dir.parent),
            )
            trajectory_collector = getattr(self, "_trajectory_collector", None)
            if trajectory_collector is not None:
                try:
                    trajectory_collector.capture_stage(
                        stage=self._current_stage,
                        verify_report=failure_report,
                        repair_actions=[],
                    )
                except Exception as exc:
                    console_logger.warning(
                        "[SceneExpert] Failure trajectory capture failed: %s", exc
                    )

        missing_stages = [
            stage
            for stage in CONTRACT_STAGE_ORDER
            if stage not in self._completed_stages
        ]
        full_report = FullVerifyReport(
            deterministic_pass=False,
            pass_scene=False,
            expected_stages=list(CONTRACT_STAGE_ORDER),
            completed_stages=list(self._completed_stages),
            missing_stages=missing_stages,
            outcome_status="FAILED",
            degraded_reasons=[str(error)],
            metric_sources={"failure": "scenesmith_main_hard_gate"},
        )
        exports = {"scene_dir": str(self._scene_debug_dir.parent)}
        self._trace_logger.finalize(
            full_report=full_report,
            exports=exports,
            model=self._qwen_model,
        )
        self._write_long_term_memory(full_report)
        self._flush_skill_outcomes()
        trace_dict = self._trace_logger.finalize(
            full_report=full_report,
            exports=exports,
            model=self._qwen_model,
        )
        trace_dict.update({"status": "failed", "degraded": True, "error": str(error)})
        self._trace_logger.save(trace_dict)
        self._capture_main_repair_activity()

    def save_partial_trace(self, error: str = "") -> None:
        """Persist a partial trace from an exception path."""
        if not self._trace_enabled():
            return
        try:
            path = self._trace_logger.save_partial(status="failed", error=error)
            self._capture_main_repair_activity()
            console_logger.info(f"[SceneExpert] Partial trace saved to {path}")
        except Exception as save_error:
            console_logger.warning(
                f"[SceneExpert] Failed to save partial trace: {save_error}"
            )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _initial_completed_stages(self, start_stage: str) -> list[str]:
        """Return the stage-order prefix already satisfied by a resumed run."""
        if start_stage not in CONTRACT_STAGE_ORDER:
            return []
        return CONTRACT_STAGE_ORDER[: CONTRACT_STAGE_ORDER.index(start_stage)]

    def _validate_stage_transition(self, stage: str) -> None:
        """Enforce Harness FSM order while tolerating sequential multi-room runs."""
        try:
            self._harness.validate_stage_order(self._completed_stages, stage)
            return
        except ValueError:
            # _generate_room runs a full room pipeline per room. When a new room
            # starts, the same per-scene hook sees the start stage again. Reset the
            # FSM baseline for that room instead of treating it as an LLM skip.
            if stage == self._room_start_stage and self._completed_stages:
                console_logger.info(
                    "[SceneExpert] Resetting Harness stage-order baseline for "
                    f"new room at stage '{stage}'"
                )
                self._completed_stages = list(self._room_stage_order_baseline)
                self._harness.validate_stage_order(self._completed_stages, stage)
                return
            raise

    def _build_scene_state_summary(self) -> str:
        """Build a text summary of completed stages for the GlobalPlanner."""
        if not self._completed_stages:
            return "Empty scene — no objects placed yet."
        return "Completed stages: " + ", ".join(self._completed_stages)

    def _extract_floor_plan_state_info(self, scene_dir: Path) -> dict:
        """Extract lightweight floor-plan facts for rule-based verification."""
        layout_path = scene_dir / "house_layout.json"
        if not layout_path.exists():
            return {"layout_exists": False, "room_count": 0, "rooms": []}
        try:
            with layout_path.open() as f:
                data = json.load(f)
        except Exception as e:
            return {
                "layout_exists": False,
                "room_count": 0,
                "rooms": [],
                "layout_error": str(e),
            }

        rooms = data.get("room_specs") or data.get("rooms") or []
        if isinstance(rooms, dict):
            rooms = list(rooms.values())
        if not isinstance(rooms, list):
            rooms = []
        return {
            "layout_exists": True,
            "room_count": len(rooms),
            "rooms": rooms,
        }

    def _format_related_memory_for_writer(self) -> str:
        """Build compact related-memory context for MemoryWriter deduplication."""
        if self._retriever is None:
            return ""

        lines: list[str] = []
        seen: set[str] = set()
        for stage in CONTRACT_STAGE_ORDER:
            try:
                pack = self._retriever.retrieve(self._task_spec, stage)
            except Exception:
                continue
            for item in (
                pack.success_hints
                + pack.failure_hints
                + pack.skill_texts
                + ([pack.placement_reference] if pack.placement_reference else [])
            ):
                text = item.strip()
                if not text or text in seen:
                    continue
                seen.add(text)
                lines.append(f"- [{stage}] {text}")
        return "\n".join(lines[:24])

    def _extract_scene_state_info_from_scene(self, scene: RoomScene) -> dict:
        """Extract object names from the live RoomScene for rule-based checks."""
        try:
            names: list[str] = []
            records: list[dict[str, Any]] = []

            def append_name(value: Any) -> None:
                if isinstance(value, str) and value.strip():
                    names.append(value.strip())

            def append_component_names(value: Any, aliases: list[str]) -> None:
                """Collect semantic names from supported composite metadata."""
                if not isinstance(value, dict):
                    return
                component_name = value.get("name")
                append_name(component_name)
                if isinstance(component_name, str) and component_name.strip():
                    aliases.append(component_name.strip())
                for key in (
                    "container_asset",
                    "fill_assets",
                    "member_assets",
                    "components",
                    "members",
                ):
                    nested = value.get(key)
                    if isinstance(nested, dict):
                        append_component_names(nested, aliases)
                    elif isinstance(nested, list):
                        for item in nested:
                            append_component_names(item, aliases)

            for obj in scene.objects.values():
                name = getattr(obj, "name", None)
                append_name(name)
                description = getattr(obj, "description", None)
                aliases: list[str] = []
                metadata = getattr(obj, "metadata", None)
                if not isinstance(metadata, dict):
                    metadata = {}
                if metadata.get("composite_type"):
                    append_component_names(metadata, aliases)
                records.append(
                    {
                        "name": name if isinstance(name, str) else "",
                        "description": (
                            description if isinstance(description, str) else ""
                        ),
                        "aliases": aliases,
                    }
                )
            return {
                "object_names": names,
                "object_records": records,
            }
        except Exception:
            return {"object_names": []}


# ------------------------------------------------------------------
# Factory function
# ------------------------------------------------------------------


def build_hook_runner(
    prompt: str,
    scene_id: int,
    output_dir: Path,
    cfg_dict: dict,
) -> SceneExpertHookRunner | None:
    """Build a SceneExpertHookRunner from config.

    Returns None if scene_expert is disabled (ablation mode "disabled" or
    scene_expert config block missing).

    Args:
        prompt: Raw scene prompt.
        scene_id: Scene index.
        output_dir: Base experiment output directory.
        cfg_dict: Full Hydra config as plain dict.

    Returns:
        Configured SceneExpertHookRunner, or None if disabled.
    """
    # Deep-merge root defaults with experiment overrides. A shallow ``or`` here
    # would discard nested memory and component defaults.
    root_se_cfg = dict(cfg_dict.get("scene_expert", {}) or {})
    se_cfg = resolve_scene_expert_config(cfg_dict)
    if not se_cfg:
        _compile_intent_contract_if_enabled(
            prompt=prompt,
            scene_id=scene_id,
            output_dir=output_dir,
            cfg_dict=cfg_dict,
        )
        return None
    memory_cfg = se_cfg.get("memory", {}) or {}
    behavior_cfg = se_cfg.get("behavior", {}) or {}
    component_flags = resolve_component_flags(cfg_dict)

    mode = se_cfg.get("mode", "disabled")
    if not se_cfg.get("enabled", False) or not any(component_flags.values()):
        _compile_intent_contract_if_enabled(
            prompt=prompt,
            scene_id=scene_id,
            output_dir=output_dir,
            cfg_dict=cfg_dict,
        )
        return None

    if mode not in ABLATION_MODES:
        _compile_intent_contract_if_enabled(
            prompt=prompt,
            scene_id=scene_id,
            output_dir=output_dir,
            cfg_dict=cfg_dict,
        )
        console_logger.warning(
            f"Unknown scene_expert.mode={mode!r}. "
            f"Valid: {sorted(ABLATION_MODES)}. Disabling SceneExpert."
        )
        return None

    stage_policies = resolve_stage_policies(cfg_dict)
    console_logger.info(f"[SceneExpert] Building hook runner (mode={mode})")

    # Model / API settings (shared with SceneSmith agents)
    critic_config = critic_config_from_any(cfg_dict)
    model = _intent_compiler_model(cfg_dict)
    api_base = os.environ.get("OPENAI_BASE_URL", "http://localhost:8000/v1")
    api_key = os.environ.get("OPENAI_API_KEY", "dummy")

    # Persistent memory storage, retrieval, and writing are independently gated.
    memory_dir = memory_cfg.get(
        "dir",
        cfg_dict.get("paths", {}).get("memory_dir", "outputs/scene_expert_memory"),
    )
    exclude_source_task_id = ""
    ret_cfg = memory_cfg.get("retrieval", {}) or {}
    if _cfg_bool(ret_cfg.get("exclude_same_task"), True):
        exclude_source_task_id = (
            "task_" + hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]
        )
    use_memory_store = any(
        component_flags[name]
        for name in (
            "fast_memory_retrieval",
            "memory_writer",
            "stage_working_memory",
        )
    )
    scene_debug_dir = output_dir / f"scene_{scene_id:03d}" / "scene_expert"
    os.environ["SCENEEXPERT_LLM_DEBUG_PATH"] = str(
        scene_debug_dir / "timing" / "scene_expert_llm_calls.jsonl"
    )
    if not use_memory_store:
        os.environ.pop("SCENEEXPERT_ACTIVE_MEMORY_BANK_DIR", None)

    memory_store: FastMemoryStore | None = None
    retriever: Any | None = None
    memory_writer: MemoryWriter | None = None
    structured_llm_client: Any | None = None
    structured_llm_cfg = se_cfg.get("structured_llm", {}) or {}

    if component_flags["structured_llm"]:
        from scenesmith.scene_expert.structured_llm import (
            SceneExpertStructuredLLMClient,
        )

        structured_llm_client = SceneExpertStructuredLLMClient(
            model=model,
            api_base_url=api_base,
            api_key=api_key,
            profiles=structured_llm_cfg.get("roles", {}) or {},
            debug_path=scene_debug_dir / "timing" / "structured_llm.jsonl",
        )

    if use_memory_store:
        ret_cfg = memory_cfg.get("retrieval", {})
        memory_store = FastMemoryStore(memory_dir)
        os.environ["SCENEEXPERT_ACTIVE_MEMORY_BANK_DIR"] = str(memory_dir)

    if component_flags["fast_memory_retrieval"]:
        if memory_store is None:
            raise RuntimeError("Memory retrieval requires an initialized memory store")
        retriever_type = memory_cfg.get("retriever_type", "lexical")
        if retriever_type == "hybrid":
            retriever = _build_hybrid_retriever(
                memory_store=memory_store,
                memory_dir=memory_dir,
                memory_cfg=memory_cfg,
                ret_cfg=ret_cfg,
                timing_path=scene_debug_dir / "timing" / "memory_retrieval.jsonl",
                exclude_source_task_id=exclude_source_task_id,
            )
        elif retriever_type == "lexical":
            retriever = MemoryRetriever(
                store=memory_store,
                max_success=_cfg_int(ret_cfg.get("max_success_cases"), 3),
                max_failure=_cfg_int(ret_cfg.get("max_failure_cases"), 3),
                max_skills=_cfg_int(ret_cfg.get("max_skills"), 2),
                exclude_source_task_id=exclude_source_task_id,
            )
        else:
            raise ValueError(
                f"Unsupported SceneExpert memory retriever_type={retriever_type!r}. "
                "Use 'lexical' or 'hybrid'."
            )
    if component_flags["memory_writer"]:
        writer_kwargs: dict[str, Any] = {
            "model": model,
            "api_base_url": api_base,
            "api_key": api_key,
            "debug_dir": scene_debug_dir / "memory",
            "llm_client": structured_llm_client,
            "success_min_overall_score": _cfg_float(
                (memory_cfg.get("writer", {}) or {}).get(
                    "success_min_overall_score", 0.75
                ),
                0.75,
            ),
        }
        if component_flags["structured_llm"]:
            writer_role_cfg = (structured_llm_cfg.get("roles", {}) or {}).get(
                "memory_writer", {}
            ) or {}
            writer_kwargs.update(
                max_tokens=_cfg_int(writer_role_cfg.get("max_tokens"), 2048),
                retry_max_tokens=_cfg_int(
                    writer_role_cfg.get("retry_max_tokens"), 4096
                ),
                thinking_mode=str(writer_role_cfg.get("thinking_mode", "none")),
                timeout_seconds=_cfg_float(
                    writer_role_cfg.get("timeout_seconds"), 90.0
                ),
                temperature=_cfg_float(writer_role_cfg.get("temperature"), 0.1),
            )
        memory_writer = MemoryWriter(**writer_kwargs)

    # Verifier thresholds
    ver_cfg = se_cfg.get("verifier", {})
    stage_verifier = StageVerifier(
        pass_threshold=ver_cfg.get("stage_pass_threshold", 0.6),
        visual_score_hard_gate=ver_cfg.get("visual_score_hard_gate", False),
        critic_bridge_enabled=component_flags["critic_bridge"],
    )
    full_verifier = FullVerifier(
        pass_threshold=ver_cfg.get("full_pass_threshold", 0.7),
        visual_score_hard_gate=ver_cfg.get("visual_score_hard_gate", False),
    )

    # Preserve main's ownership order: TaskCompiler first, optional behavior
    # expansion second, then the authoritative critic intent compiler consumes
    # the resulting task spec. Disabling the wrapper compiler uses only its
    # deterministic fallback; critic never suppresses or replaces this step.
    from omegaconf import OmegaConf

    from scenesmith.scene_expert.task_compiler import _fallback_spec_from_prompt

    task_compiler: TaskCompiler | None = None
    task_compiler_trace: dict[str, Any] = {}
    if component_flags["task_compiler"]:
        task_compiler = TaskCompiler(
            model=model,
            api_base_url=api_base,
            api_key=api_key,
            llm_client=structured_llm_client,
        )
        try:
            task_spec = task_compiler.compile(prompt)
        except Exception as e:
            console_logger.warning(
                f"TaskCompiler failed, using fallback task spec from prompt text: {e}"
            )
            task_spec = _fallback_spec_from_prompt(prompt)
        task_compiler_trace = dict(getattr(task_compiler, "last_trace", {}) or {})
    else:
        task_spec = _fallback_spec_from_prompt(prompt)
        task_compiler_trace = {
            "status": "disabled",
            "attempts": [],
            "failure_reason": "TaskCompiler disabled by component gate",
        }

    task_spec, behavior_spec = apply_behavior_template(
        prompt,
        task_spec,
        config=behavior_cfg,
        output_path=scene_debug_dir / "behavior_spec.json",
        model=model,
        api_base_url=api_base,
        api_key=api_key,
    )
    if behavior_spec is not None:
        console_logger.info(
            "[SceneExpert] Applied deterministic behavior template; spec=%s",
            scene_debug_dir / "behavior_spec.json",
        )

    intent_contract, intent_trace = _compile_intent_contract_if_enabled(
        prompt=prompt,
        scene_id=scene_id,
        output_dir=output_dir,
        cfg_dict=cfg_dict,
        task_spec=task_spec,
    )
    pre_reconciliation_task_spec = task_spec
    task_spec = _reconcile_task_spec_stage_ownership(task_spec, intent_contract)
    ownership_audit = _audit_stage_ownership(
        pre_reconciliation_task_spec,
        task_spec,
        intent_contract,
    )
    task_compiler_trace["ownership_reconciliation"] = ownership_audit
    if ownership_audit["errors"]:
        raise ValueError(
            "Stage ownership reconciliation failed: "
            + "; ".join(ownership_audit["errors"])
        )

    # Harness always assembles planner context, while its FSM and budget
    # controls are independently gated at each control boundary.
    se_omega = OmegaConf.create(se_cfg)
    harness = Harness(
        se_omega,
        budget_enabled=component_flags["harness_budget"],
    )
    harness.reset()

    global_planner = GlobalPlanner(
        model=model,
        api_base_url=api_base,
        api_key=api_key,
        llm_client=structured_llm_client,
    )
    relation_projector = StageRelationProjector(
        floor_plan_reservation_gate_enabled=bool(
            _deep_merge_dicts(
                root_se_cfg.get("floor_plan_reservations", {}),
                se_cfg.get("floor_plan_reservations", {}),
            ).get("enabled", False)
        ),
    )
    repair_controller = RepairController(memory_store=memory_store)
    start_stage = (
        cfg_dict.get("experiment", {})
        .get("pipeline", {})
        .get("start_stage", "floor_plan")
    )
    stop_stage = (
        cfg_dict.get("experiment", {})
        .get("pipeline", {})
        .get("stop_stage", CONTRACT_STAGE_ORDER[-1])
    )
    config_hash = _stable_config_hash(cfg_dict)
    experiment_signature = _stable_experiment_signature(cfg_dict)
    trace_logger: TraceLogger | None = None
    if component_flags["trace"]:
        trace_logger = TraceLogger(
            output_dir=str(output_dir),
            scene_index=scene_id,
            prompt=prompt,
            experiment_name=cfg_dict.get("name", ""),
            config_hash=config_hash,
            experiment_signature=experiment_signature,
            task_spec=task_spec.model_dump(mode="json", exclude_none=True),
            task_spec_status={
                "source": (
                    "fallback" if task_spec.compiler_status == "degraded" else "llm"
                ),
                "degraded": task_spec.compiler_status == "degraded",
            },
            code_provenance=collect_code_provenance(),
            component_flags=component_flags,
        )

    trajectory_collector: TrajectoryCollector | None = None
    if component_flags["slow_memory_capture"]:
        capture_cfg = (se_cfg.get("slow_memory", {}) or {}).get("capture", {}) or {}
        code_provenance = collect_code_provenance()
        trajectory_collector = TrajectoryCollector(
            scene_debug_dir=scene_debug_dir,
            prompt=prompt,
            scene_id=f"scene_{scene_id:03d}",
            run_id=str(output_dir.resolve()),
            task_spec=task_spec,
            experiment_signature=experiment_signature,
            config_hash=config_hash,
            model_id=model,
            capture_mode=mode,
            component_flags=component_flags,
            code_provenance=code_provenance,
            max_prompt_chars=_cfg_int(capture_cfg.get("max_prompt_chars"), 131072),
            max_response_chars=_cfg_int(capture_cfg.get("max_response_chars"), 1048576),
        )

    return SceneExpertHookRunner(
        prompt=prompt,
        scene_id=scene_id,
        output_dir=output_dir,
        mode=mode,
        component_flags=component_flags,
        task_spec=task_spec,
        harness=harness,
        global_planner=global_planner,
        relation_projector=relation_projector,
        retriever=retriever,
        stage_verifier=stage_verifier,
        full_verifier=full_verifier,
        repair_controller=repair_controller,
        trace_logger=trace_logger,
        memory_writer=memory_writer,
        memory_store=memory_store,
        trajectory_collector=trajectory_collector,
        qwen_model=model,
        experiment_name=cfg_dict.get("name", ""),
        config_hash=config_hash,
        experiment_signature=experiment_signature,
        start_stage=start_stage,
        allow_long_term_memory_updates=(stop_stage == CONTRACT_STAGE_ORDER[-1]),
        intent_contract=intent_contract,
        intent_trace=intent_trace,
        task_compiler_trace=task_compiler_trace,
        critic_config=critic_config,
        stage_policies=stage_policies,
    )
