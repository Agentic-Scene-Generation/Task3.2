import unittest

import json
import os
import sys
import types

from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from scripts.build_memory_index import build_memory_indexes
from scenesmith.scene_expert.memory.embedding import (
    SceneMemoryEmbedder,
    resolve_memory_embedding_model_dir,
)
from scenesmith.scene_expert.memory.hybrid_retriever import HybridMemoryRetriever
from scenesmith.scene_expert.memory.index import NumpyMemoryIndex
from scenesmith.scene_expert.memory.retriever import _tokenize
from scenesmith.scene_expert.memory.schemas import (
    FailureCase,
    MemoryUpdateOp,
    Skill,
    SuccessCase,
)
from scenesmith.scene_expert.memory.store import FastMemoryStore
from scenesmith.scene_expert.memory.text_builder import build_embedding_text
from scenesmith.scene_expert.memory.writer import MemoryWriter
from scenesmith.scene_expert.context_bundle import (
    build_llm_call_debug_record,
    build_stage_context_bundle,
)
from scenesmith.scene_expert.hooks import (
    SceneExpertHookRunner,
    _reconcile_task_spec_stage_ownership,
)
from scenesmith.scene_expert.global_planner import (
    GlobalPlanner,
    _SYSTEM_PROMPT,
    _reconcile_floor_plan_zone_guidance,
    _reconcile_stage_brief,
)
from scenesmith.scene_expert.repair_taxonomy import (
    FailureCategory,
    classify_hard_reasons,
)
from scenesmith.scene_expert.schemas import (
    FullVerifyReport,
    HarnessContext,
    MemoryPack,
    SceneTaskSpec,
    StageBrief,
    StageVerifyReport,
    VerifyIssue,
)
from scenesmith.agent_utils.scoring import CategoryScore, FurnitureCritiqueWithScores
from scenesmith.agent_utils.stage_working_memory import StageWorkingMemory
from scenesmith.scene_expert.task_compiler import (
    _fallback_spec_from_prompt,
    _normalize_stage_ownership,
)
from scenesmith.scene_expert.verifier import (
    FullVerifier,
    StageVerifier,
    _map_scenesmith_scores,
)


class SceneExpertMemoryTest(unittest.TestCase):
    def test_stage_brief_and_planner_preserve_immutable_task_priority(self) -> None:
        task = "A study with guest chairs against the wall facing into the room."
        context = HarnessContext(
            stage="furniture",
            task_spec=SceneTaskSpec(room_type="study", style="standard"),
            memory_pack=MemoryPack(),
        )
        planner = object.__new__(GlobalPlanner)

        message = planner._build_user_message(
            context,
            "",
            original_task=task,
        )
        brief = StageBrief(stage="furniture", stage_objective="Place furniture")

        self.assertIn("## Immutable User Task", message)
        self.assertIn(task, message)
        self.assertIn("takes priority over", message)
        self.assertIn("explicit object, position, or facing", brief.to_injection_text())
        self.assertIn("Immutable User Task", _SYSTEM_PROMPT)
        self.assertIn("minimum, not an exhaustive list", _SYSTEM_PROMPT)

    def test_stage_brief_removes_planner_invented_inventory_exclusivity(self) -> None:
        brief = StageBrief(
            stage="furniture",
            stage_objective="Place only the bed required by the task",
            constraints_for_designer=[
                "Place the bed in the sleeping zone.",
                "Do not place any other furniture; only the bed is required.",
                "Only place furniture objects during the furniture stage.",
            ],
            checks_for_critic=["Verify only the bed is required."],
            failure_patterns_to_avoid=["Avoid adding any additional furniture."],
        )

        reconciled = _reconcile_stage_brief(
            brief,
            original_task="A bedroom with rustic farmhouse decor.",
            room_type="bedroom",
            required_objects=["bed"],
        )
        text = reconciled.to_injection_text()

        self.assertNotIn("only the bed", text.lower())
        self.assertNotIn("additional furniture", text.lower())
        self.assertIn("Only place furniture objects", text)
        self.assertIn("additional objects owned by this stage are allowed", text)

    def test_stage_brief_preserves_user_requested_sparse_inventory(self) -> None:
        constraint = "Do not place any other furniture; only the bed is required."
        brief = StageBrief(
            stage="furniture",
            stage_objective="Place only the bed required by the task",
            constraints_for_designer=[constraint],
        )

        reconciled = _reconcile_stage_brief(
            brief,
            original_task="A bedroom with only a bed and no other furniture.",
            room_type="bedroom",
            required_objects=["bed"],
        )

        self.assertEqual([constraint], reconciled.constraints_for_designer)
        self.assertEqual(brief.stage_objective, reconciled.stage_objective)

    def test_stage_brief_does_not_treat_only_aesthetic_as_closed_inventory(
        self,
    ) -> None:
        brief = StageBrief(
            stage="furniture",
            stage_objective="Place only the bed required by the task",
            constraints_for_designer=[
                "Do not place any other furniture; only the bed is required."
            ],
        )

        reconciled = _reconcile_stage_brief(
            brief,
            original_task="A bedroom with only warm colors and rustic decor.",
            room_type="bedroom",
            required_objects=["bed"],
        )

        self.assertNotIn("only the bed", reconciled.to_injection_text().lower())

    def test_structural_only_ceiling_brief_does_not_invent_lighting_requirement(
        self,
    ) -> None:
        brief = StageBrief(
            stage="ceiling_mounted",
            stage_objective="Install exposed beams and pendant lighting.",
            constraints_for_designer=[
                "Place exposed beams across the ceiling.",
                "Add a pendant light at the room center.",
            ],
            checks_for_critic=[
                "Verify the beams are clear of furniture.",
                "Fail the stage when lighting is absent.",
            ],
        )

        reconciled = _reconcile_stage_brief(
            brief,
            original_task="A bedroom with rustic decor and exposed wooden beams.",
            room_type="bedroom",
            required_objects=["exposed wooden beams"],
        )
        text = reconciled.to_injection_text().lower()

        self.assertNotIn("pendant", text)
        self.assertNotIn("fail the stage when lighting is absent", text)
        self.assertIn("structural ceiling spans", text)
        self.assertIn("absent lighting as a ceiling-stage failure", text)

    def test_floor_plan_brief_keeps_furniture_zones_nonstructural(self) -> None:
        brief = StageBrief(
            stage="floor_plan",
            stage_objective="Create a long rectangular room.",
            constraints_for_designer=[
                "Physically separate the living and dining zones."
            ],
            checks_for_critic=[
                "Verify that the living and dining zones are spatially distinct."
            ],
        )

        reconciled = _reconcile_floor_plan_zone_guidance(
            brief,
            original_task=(
                "A long living room with separate living and dining areas and "
                "furniture in each area."
            ),
            functional_zones=["living_zone", "dining_zone"],
        )
        text = reconciled.to_injection_text().lower()

        self.assertNotIn("physically separate", text)
        self.assertNotIn("spatially distinct", text)
        self.assertIn("later furniture zones", text)
        self.assertIn("opening-free wall length", text)

    def test_floor_plan_brief_preserves_explicit_structural_zoning(self) -> None:
        brief = StageBrief(
            stage="floor_plan",
            stage_objective="Create a divided room.",
            constraints_for_designer=[
                "Physically separate the living and dining zones."
            ],
        )

        reconciled = _reconcile_floor_plan_zone_guidance(
            brief,
            original_task=(
                "Divide the room with an interior wall to create separate living "
                "and dining rooms."
            ),
            functional_zones=["living_zone", "dining_zone"],
        )

        self.assertEqual(brief, reconciled)

    def test_empty_stage_inventory_uses_noop_brief(self) -> None:
        context = HarnessContext(
            stage="wall_mounted",
            task_spec=SceneTaskSpec(
                room_type="living room",
                style="standard",
                required_large_objects=["tv_stand"],
                required_small_objects=["television"],
            ),
            memory_pack=MemoryPack(),
        )
        planner = object.__new__(GlobalPlanner)

        brief = planner.generate_stage_brief(
            context,
            original_task=(
                "A sofa faces a TV stand and television on the opposite wall."
            ),
        )

        self.assertIn("empty wall_mounted stage", brief.stage_objective)
        self.assertIn("No objects are allocated", brief.constraints_for_designer[0])
        self.assertNotIn("television", "\n".join(brief.constraints_for_designer))

    def test_llm_debug_record_marks_scenebenchmark_prompt_context(self) -> None:
        record = build_llm_call_debug_record(
            stage="furniture",
            agent_role="critic",
            event="observe_scene",
            prompt=(
                "long critic prompt\n\n"
                "Additional SceneBenchmark geometry critic context:\n"
                "functional_dependency"
            ),
        )

        self.assertTrue(record.prompt_contains_scenebenchmark_context)

    def test_memory_writer_normalizes_null_noop_fields(self) -> None:
        normalized = MemoryWriter._normalize_update_op(
            {
                "op": "NOOP",
                "memory_type": "success_case",
                "target_id": None,
                "content": None,
            }
        )

        op = MemoryUpdateOp.model_validate(normalized)

        self.assertEqual(op.target_id, "")
        self.assertEqual(op.content, {})

    def test_task_compiler_fallback_preserves_required_bedroom_objects(self) -> None:
        spec = _fallback_spec_from_prompt(
            "A bedroom with a bed, two nightstands, and a wardrobe in the corner."
        )

        self.assertEqual("bedroom", spec.room_type)
        self.assertEqual(
            ["bed", "nightstand", "nightstand", "wardrobe"],
            spec.required_large_objects,
        )
        self.assertIn("sleeping_zone", spec.functional_zones)

    def test_task_compiler_fallback_owns_new_scene_categories_by_stage(self) -> None:
        spec = _fallback_spec_from_prompt(
            "A bedroom with one dressing table, one stool, one wall-mounted mirror, "
            "and four floor plants. An office area has four desks, four computer "
            "monitors, one water dispenser, and one wastebasket."
        )

        self.assertIn("dressing_table", spec.required_large_objects)
        self.assertIn("stool", spec.required_large_objects)
        self.assertEqual(4, spec.required_large_objects.count("plant"))
        self.assertIn("water_dispenser", spec.required_large_objects)
        self.assertNotIn("table", spec.required_large_objects)
        self.assertEqual(["mirror"], spec.required_wall_objects)
        self.assertEqual(4, spec.required_small_objects.count("monitor"))
        self.assertIn("wastebasket", spec.required_small_objects)

    def test_task_compiler_normalizes_alias_inventory_without_double_counting(
        self,
    ) -> None:
        spec = _normalize_stage_ownership(
            SceneTaskSpec(
                room_type="living room",
                style="functional",
                required_large_objects=["large floor plant"] * 4 + ["plant"] * 4,
                required_small_objects=(
                    ["plate"] * 5
                    + ["cutlery"] * 5
                    + ["glass"] * 5
                    + ["table setting"] * 5
                    + ["wastebasket"]
                ),
            ),
            prompt="Put the wastebasket on the floor.",
        )

        self.assertEqual(4, spec.required_large_objects.count("plant"))
        self.assertIn("wastebasket", spec.required_large_objects)
        self.assertNotIn("wastebasket", spec.required_small_objects)
        self.assertFalse(
            any("setting" in value for value in spec.required_small_objects)
        )

    def test_repair_taxonomy_classifies_core_hard_failures(self) -> None:
        failures = classify_hard_reasons(
            [
                "missing required bed",
                "physics hard violation: door clearance violations",
                "geometry construction failed for likely wardrobe asset",
            ]
        )

        categories = {failure.category for failure in failures}

        self.assertIn(FailureCategory.MISSING_REQUIRED_OBJECT, categories)
        self.assertIn(FailureCategory.DOOR_OR_OPENING_CLEARANCE, categories)
        self.assertIn(FailureCategory.ASSET_INVALID, categories)

    def test_stage_context_bundle_formats_compact_llm_text(self) -> None:
        spec = SceneTaskSpec(
            room_type="bedroom",
            style="standard",
            required_large_objects=["bed", "nightstand", "wardrobe"],
        )

        bundle = build_stage_context_bundle(
            stage="furniture",
            agent_role="designer",
            event="request_initial_design",
            task_spec=spec,
            last_hard_issues=["door clearance violation"],
            prompt="A bedroom with a bed and wardrobe.",
        )
        text = bundle.to_llm_text()

        self.assertIn("StageContextBundle: furniture / designer", text)
        self.assertIn("door clearance violation", text)
        self.assertIn("bedroom", text)

    def test_stage_context_bundle_prefers_immutable_original_task(self) -> None:
        scene = types.SimpleNamespace(
            room_geometry=None,
            objects={},
            text_description=("Mutable StageBrief: guest chairs must face the desk."),
            scene_expert_original_description=(
                "A study with guest chairs facing into the room."
            ),
        )

        bundle = build_stage_context_bundle(
            stage="furniture",
            agent_role="critic",
            event="request_critique",
            scene=scene,
        )

        self.assertIn("original_task=A study", bundle.scene_summary)
        self.assertNotIn("Mutable StageBrief", bundle.scene_summary)

    def test_stage_working_memory_commits_public_failure_event(self) -> None:
        class FakeTransform:
            def translation(self):
                return np.array([0.0, 0.0, 0.0])

        class FakeObject:
            name = "wardrobe"
            object_type = "furniture"
            immutable = False
            metadata = {}
            transform = FakeTransform()

            def compute_world_bounds(self):
                return np.array([0.0, 0.0, 0.0]), np.array([0.8, 0.6, 2.0])

        class FakeScene:
            objects = {"wardrobe_0": FakeObject()}

            def content_hash(self):
                return "fake-scene"

        scores = FurnitureCritiqueWithScores(
            critique="DETERMINISTIC HARD-CHECK FAILED: door clearance violation",
            realism=CategoryScore("realism", 3, "bad"),
            functionality=CategoryScore("functionality", 2, "blocked"),
            layout=CategoryScore("layout", 3, "bad"),
            layout_plausibility=CategoryScore("layout_plausibility", 2, "bad"),
            holistic_completeness=CategoryScore("holistic_completeness", 2, "bad"),
            prompt_following=CategoryScore("prompt_following", 2, "missing bed"),
            reachability=CategoryScore("reachability", 2, "blocked"),
        )

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            public_dir = root / "scene_expert_memory" / "ablation_test"
            old_env = os.environ.get("SCENEEXPERT_ACTIVE_MEMORY_BANK_DIR")
            os.environ["SCENEEXPERT_ACTIVE_MEMORY_BANK_DIR"] = str(public_dir)
            try:
                memory = StageWorkingMemory(
                    root_dir=root / "scene_000" / "room_bedroom",
                    stage="furniture",
                    enabled=True,
                )
                memory.set_required_counts({"bed": 1})
                render_dir = root / "renders_001"
                render_dir.mkdir()
                memory.save_render_record(
                    render_dir=render_dir,
                    role="critic",
                    event="deterministic_hard_fail",
                    scene=FakeScene(),
                    scores=scores,
                    critique=scores.critique,
                )
            finally:
                if old_env is None:
                    os.environ.pop("SCENEEXPERT_ACTIVE_MEMORY_BANK_DIR", None)
                else:
                    os.environ["SCENEEXPERT_ACTIVE_MEMORY_BANK_DIR"] = old_env

            self.assertTrue((public_dir / "events.jsonl").exists())
            self.assertTrue((public_dir / "failure_cases.jsonl").exists())
            self.assertGreater((public_dir / "failure_cases.jsonl").stat().st_size, 0)

    def test_furniture_stage_verifier_fails_hard_missing_and_collision(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            scores_dir = root / "scene_states" / "furniture"
            scores_dir.mkdir(parents=True)
            (scores_dir / "scores.yaml").write_text(
                "\n".join(
                    [
                        "Realism:",
                        "  grade: 4",
                        "  comment: collision detected with wall",
                        "Functionality:",
                        "  grade: 3",
                        "  comment: room incomplete",
                        "Layout Plausibility:",
                        "  grade: 7",
                        "Prompt Following:",
                        "  grade: 4",
                        "  comment: missing primary bed",
                        "Summary: bed missing and collision detected",
                    ]
                ),
                encoding="utf-8",
            )

            report = StageVerifier(
                pass_threshold=0.6,
                visual_score_hard_gate=True,
            ).verify(
                stage="furniture",
                stage_output_dir=str(root),
                task_spec=SceneTaskSpec(
                    room_type="bedroom",
                    style="standard",
                    required_large_objects=[
                        "bed",
                        "nightstand",
                        "nightstand",
                        "wardrobe",
                    ],
                ),
                scene_state_info={
                    "object_names": ["nightstand_0", "nightstand_1", "wardrobe_0"]
                },
            )

            self.assertFalse(report.pass_stage)
            issue_types = {issue.issue_type for issue in report.issues}
            self.assertIn("missing_object", issue_types)
            self.assertIn("physics_collision", issue_types)

    def test_furniture_visual_scores_are_advisory_by_default(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            scores_dir = root / "scene_states" / "furniture"
            scores_dir.mkdir(parents=True)
            (scores_dir / "scores.yaml").write_text(
                "\n".join(
                    [
                        "Functionality:",
                        "  grade: 2",
                        "Prompt Following:",
                        "  grade: 6",
                        "Layout Plausibility:",
                        "  grade: 4",
                        "Summary: No physics collisions detected. All required furniture is present.",
                    ]
                ),
                encoding="utf-8",
            )

            report = StageVerifier(pass_threshold=0.6).verify(
                stage="furniture",
                stage_output_dir=str(root),
                task_spec=SceneTaskSpec(
                    room_type="study",
                    style="standard",
                    required_large_objects=["desk"],
                ),
                scene_state_info={"object_names": ["desk_0"]},
            )

            self.assertTrue(report.pass_stage)
            self.assertEqual([], report.issues)

    def test_deterministic_hard_fail_is_never_advisory(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            scores_dir = root / "scene_states" / "ceiling"
            scores_dir.mkdir(parents=True)
            (scores_dir / "scores.yaml").write_text(
                "\n".join(
                    [
                        "Realism:",
                        "  grade: 3",
                        "Functionality:",
                        "  grade: 2",
                        'Summary: "DETERMINISTIC HARD-CHECK FAILED BEFORE VLM SCORING. Hard issues: collisions."',
                    ]
                ),
                encoding="utf-8",
            )

            stage_report = StageVerifier(pass_threshold=0.6).verify(
                stage="ceiling_mounted",
                stage_output_dir=str(root),
                task_spec=SceneTaskSpec(room_type="bedroom", style="standard"),
                scene_state_info={"object_names": []},
            )
            full_report = FullVerifier().verify([stage_report])

        self.assertFalse(stage_report.pass_stage)
        self.assertIn(
            "deterministic_hard_failure",
            {issue.issue_type for issue in stage_report.issues},
        )
        self.assertFalse(full_report.deterministic_pass)
        self.assertFalse(full_report.pass_scene)

    def test_later_clean_stage_recovers_transient_hard_failure(self) -> None:
        failed_wall = StageVerifyReport(
            stage="wall_mounted",
            pass_stage=False,
            issues=[VerifyIssue(issue_type="deterministic_hard_failure")],
            critique_summary=(
                "DETERMINISTIC HARD-CHECK FAILED BEFORE VLM SCORING. "
                "Hard issues: collisions."
            ),
        )
        clean_ceiling = StageVerifyReport(stage="ceiling_mounted", pass_stage=True)

        full_report = FullVerifier().verify([failed_wall, clean_ceiling])

        self.assertTrue(full_report.deterministic_pass)
        self.assertTrue(full_report.pass_scene)

    def test_later_clean_stage_does_not_recover_missing_objects(self) -> None:
        failed_wall = StageVerifyReport(
            stage="wall_mounted",
            pass_stage=False,
            issues=[VerifyIssue(issue_type="missing_object", object_name="television")],
        )
        clean_ceiling = StageVerifyReport(stage="ceiling_mounted", pass_stage=True)

        full_report = FullVerifier().verify([failed_wall, clean_ceiling])

        self.assertFalse(full_report.deterministic_pass)

    def test_contract_stage_ownership_reassigns_monitor_before_verification(
        self,
    ) -> None:
        task_spec = SceneTaskSpec(
            room_type="study",
            style="standard",
            required_large_objects=["desk", "computer monitor", "trash can"],
            required_small_objects=["desk lamp", "notebook", "pen holder"],
        )
        contract = {
            "constraints": [
                {
                    "stage": "furniture",
                    "strength": "hard",
                    "subjects": {"category": "desk", "count": 1},
                    "targets": {"category": "back_wall", "count": 1},
                },
                {
                    "stage": "manipuland",
                    "strength": "hard",
                    "subjects": {"category": "computer_monitor", "count": 1},
                    "targets": {"category": "desk", "count": 1},
                },
                {
                    "stage": "manipuland",
                    "strength": "hard",
                    "subjects": {"category": "trash_can", "count": 1},
                    "targets": {"category": "desk", "count": 1},
                },
            ]
        }

        reconciled = _reconcile_task_spec_stage_ownership(task_spec, contract)

        self.assertEqual(["desk"], reconciled.required_large_objects)
        self.assertCountEqual(
            ["computer monitor", "trash can", "desk lamp", "notebook", "pen holder"],
            reconciled.required_small_objects,
        )
        report = StageVerifier(pass_threshold=0.6).verify(
            stage="furniture",
            stage_output_dir="/path/that/does/not/exist",
            task_spec=reconciled,
            scene_state_info={"object_names": ["desk_0"]},
        )
        self.assertEqual([], report.issues)
        manipuland_report = StageVerifier(pass_threshold=0.6).verify(
            stage="manipuland",
            stage_output_dir="/path/that/does/not/exist",
            task_spec=reconciled,
            scene_state_info={
                "object_names": [
                    "computer_monitor_0",
                    "trash_can_0",
                    "desk_lamp_0",
                    "notebook_0",
                    "pen_holder_0",
                ]
            },
        )
        self.assertEqual([], manipuland_report.issues)

    def test_contract_stage_ownership_reassigns_wall_alias_before_verification(
        self,
    ) -> None:
        task_spec = SceneTaskSpec(
            room_type="classroom",
            style="standard",
            required_large_objects=["chalkboard", "teacher's desk"],
        )
        reconciled = _reconcile_task_spec_stage_ownership(
            task_spec,
            {
                "constraints": [
                    {
                        "stage": "wall_mounted",
                        "strength": "hard",
                        "subjects": {"category": "chalkboard", "count": 1},
                        "targets": {"category": "wall", "count": 1},
                    }
                ]
            },
        )

        self.assertEqual(["teacher's desk"], reconciled.required_large_objects)
        self.assertEqual(["chalkboard"], reconciled.required_wall_objects)
        report = StageVerifier(pass_threshold=0.6).verify(
            stage="furniture",
            stage_output_dir="/path/that/does/not/exist",
            task_spec=reconciled,
            scene_state_info={"object_names": ["teacher_desk_0"]},
        )
        self.assertTrue(report.pass_stage)
        self.assertEqual([], report.issues)

    def test_wall_inventory_verifier_uses_canonical_instructional_aliases(
        self,
    ) -> None:
        task_spec = SceneTaskSpec(
            room_type="classroom",
            style="standard",
            required_wall_objects=["instructional_surface"],
        )

        for present_name in (
            "chalkboard_0",
            "blackboard_0",
            "whiteboard_0",
            "projection_screen_0",
        ):
            report = StageVerifier(pass_threshold=0.6).verify(
                stage="wall_mounted",
                stage_output_dir="/path/that/does/not/exist",
                task_spec=task_spec,
                scene_state_info={"object_names": [present_name]},
            )
            self.assertTrue(report.pass_stage, present_name)
            self.assertEqual([], report.issues)

        unrelated = StageVerifier(pass_threshold=0.6).verify(
            stage="wall_mounted",
            stage_output_dir="/path/that/does/not/exist",
            task_spec=task_spec,
            scene_state_info={"object_names": ["painting_0"]},
        )
        self.assertFalse(unrelated.pass_stage)
        self.assertEqual(
            ["instructional_surface"], [issue.object_name for issue in unrelated.issues]
        )

    def test_manipuland_verifier_normalizes_names_and_enforces_counts(self) -> None:
        task_spec = SceneTaskSpec(
            room_type="bedroom",
            style="standard",
            required_small_objects=[
                "table lamp",
                "table lamp",
                "alarm clock",
                "book",
                "magazines",
            ],
        )

        issues = (
            StageVerifier(pass_threshold=0.6)
            .verify(
                stage="manipuland",
                stage_output_dir="/path/that/does/not/exist",
                task_spec=task_spec,
                scene_state_info={
                    "object_names": [
                        "table_lamp",
                        "table_lamp",
                        "alarm_clock",
                        "book",
                        "magazine",
                    ]
                },
            )
            .issues
        )

        self.assertEqual([], issues)

        missing_one_lamp = (
            StageVerifier(pass_threshold=0.6)
            .verify(
                stage="manipuland",
                stage_output_dir="/path/that/does/not/exist",
                task_spec=task_spec,
                scene_state_info={
                    "object_names": ["table_lamp", "alarm_clock", "book", "magazine"]
                },
            )
            .issues
        )

        self.assertEqual(
            ["table lamp"], [issue.object_name for issue in missing_one_lamp]
        )

    def test_manipuland_verifier_allows_compound_asset_components(self) -> None:
        report = StageVerifier(pass_threshold=0.6).verify(
            stage="manipuland",
            stage_output_dir="/path/that/does/not/exist",
            task_spec=SceneTaskSpec(
                room_type="dining room",
                style="standard",
                required_small_objects=["vase", "flowers"],
            ),
            scene_state_info={"object_names": ["vase_flowers_0"]},
        )

        self.assertTrue(report.pass_stage)
        self.assertEqual([], report.issues)

    def test_manipuland_verifier_accepts_decomposed_table_settings(self) -> None:
        required = [
            *("plate" for _ in range(5)),
            *("cutlery" for _ in range(5)),
            *("drinking glass" for _ in range(5)),
            *("table setting" for _ in range(5)),
        ]
        present = [
            *(f"dinner_plate_{index}" for index in range(5)),
            *(f"cutlery_{index}" for index in range(5)),
            *(f"glass_{index}" for index in range(5)),
        ]

        report = StageVerifier(pass_threshold=0.6).verify(
            stage="manipuland",
            stage_output_dir="/path/that/does/not/exist",
            task_spec=SceneTaskSpec(
                room_type="dining room",
                style="standard",
                required_small_objects=required,
            ),
            scene_state_info={"object_names": present},
        )

        self.assertTrue(report.pass_stage)
        self.assertEqual([], report.issues)

    def test_live_composite_metadata_exposes_component_names_to_verifier(self) -> None:
        scene = SimpleNamespace(
            objects={
                "filled_container_0": SimpleNamespace(
                    name="filled_vase",
                    metadata={
                        "composite_type": "filled_container",
                        "container_asset": {"name": "vase"},
                        "fill_assets": [{"name": "flowers"}],
                    },
                )
            }
        )
        runner = object.__new__(SceneExpertHookRunner)

        scene_state_info = runner._extract_scene_state_info_from_scene(scene)
        report = StageVerifier(pass_threshold=0.6).verify(
            stage="manipuland",
            stage_output_dir="/path/that/does/not/exist",
            task_spec=SceneTaskSpec(
                room_type="dining room",
                style="standard",
                required_small_objects=["vase", "flowers"],
            ),
            scene_state_info=scene_state_info,
        )

        self.assertIn("vase", scene_state_info["object_names"])
        self.assertIn("flowers", scene_state_info["object_names"])
        self.assertTrue(report.pass_stage)
        self.assertEqual([], report.issues)

    def test_live_manipuland_description_exposes_compound_component(self) -> None:
        scene = SimpleNamespace(
            objects={
                "vase_0": SimpleNamespace(
                    name="vase",
                    description=(
                        "Elegant ceramic vase with a bouquet of fresh flowers"
                    ),
                    object_type="manipuland",
                    metadata={},
                )
            }
        )
        runner = object.__new__(SceneExpertHookRunner)

        scene_state_info = runner._extract_scene_state_info_from_scene(scene)
        report = StageVerifier(pass_threshold=0.6).verify(
            stage="manipuland",
            stage_output_dir="/path/that/does/not/exist",
            task_spec=SceneTaskSpec(
                room_type="dining room",
                style="standard",
                required_small_objects=["vase", "flowers"],
            ),
            scene_state_info=scene_state_info,
        )

        self.assertTrue(report.pass_stage)
        self.assertEqual([], report.issues)

    def test_wall_verifier_matches_tv_display_to_television(self) -> None:
        report = StageVerifier(pass_threshold=0.6).verify(
            stage="wall_mounted",
            stage_output_dir="/path/that/does/not/exist",
            task_spec=SceneTaskSpec(
                room_type="living room",
                style="standard",
                required_wall_objects=["television"],
            ),
            scene_state_info={"object_names": ["tv_display"]},
        )

        self.assertEqual([], report.issues)

    def test_verifier_normalizes_english_possessive_labels(self) -> None:
        report = StageVerifier(pass_threshold=0.6).verify(
            stage="furniture",
            stage_output_dir="/path/that/does/not/exist",
            task_spec=SceneTaskSpec(
                room_type="classroom",
                style="standard",
                required_large_objects=["teacher's desk"],
            ),
            scene_state_info={"object_names": ["teacher_desk_0"]},
        )

        self.assertTrue(report.pass_stage)
        self.assertEqual([], report.issues)

    def test_layout_plausibility_maps_to_scene_expert_category(self) -> None:
        mapped = _map_scenesmith_scores(
            {
                "Layout Plausibility": 4,
                "Layout": 9,
                "Realism": 8,
            }
        )

        self.assertAlmostEqual(0.4, mapped["plausibility"])
        self.assertAlmostEqual(0.85, mapped["aesthetic"])

    def test_full_verifier_keeps_low_plausibility_as_an_observability_signal(
        self,
    ) -> None:
        report = StageVerifyReport(
            stage="furniture",
            pass_stage=True,
            scores={
                "semantic": 1.0,
                "aesthetic": 1.0,
                "plausibility": 0.4,
                "physics": 1.0,
                "interaction": 1.0,
            },
        )

        full_report = FullVerifier(pass_threshold=0.7).verify([report])

        self.assertAlmostEqual(0.4, full_report.plausibility_score)
        self.assertAlmostEqual(0.88, full_report.overall_score)
        self.assertTrue(full_report.pass_scene)

    def test_full_verifier_can_gate_visual_scores_for_offline_ablation(self) -> None:
        report = StageVerifyReport(
            stage="furniture",
            pass_stage=True,
            scores={
                "semantic": 1.0,
                "aesthetic": 1.0,
                "plausibility": 0.4,
                "physics": 1.0,
                "interaction": 1.0,
            },
        )

        full_report = FullVerifier(
            pass_threshold=0.7,
            visual_score_hard_gate=True,
        ).verify([report])

        self.assertFalse(full_report.pass_scene)

    def test_chinese_aliases_expand_to_english_tokens(self) -> None:
        tokens = set(_tokenize("卧室里需要一张床、两个床头柜和一个衣柜"))

        self.assertIn("bedroom", tokens)
        self.assertIn("bed", tokens)
        self.assertIn("nightstand", tokens)
        self.assertIn("bedside_table", tokens)
        self.assertIn("wardrobe", tokens)

    def test_success_case_fallback_embedding_text_is_structured(self) -> None:
        record = SuccessCase(
            case_id="success_bedroom_001",
            room_type="bedroom",
            style="modern",
            stage="furniture",
            task_signature=["bed", "nightstand", "wardrobe"],
            successful_pattern=["bed centered on main wall"],
            scores={"semantic": 0.9, "physics": 0.8},
        )

        text = build_embedding_text(record)

        self.assertIn("memory_type=success", text)
        self.assertIn("stage=furniture", text)
        self.assertIn("room_type=bedroom", text)
        self.assertIn("required_objects=bed, nightstand, wardrobe", text)
        self.assertIn("success_pattern=bed centered on main wall", text)

    def test_failure_case_uses_negative_constraint_in_hint_and_embedding_text(
        self,
    ) -> None:
        record = FailureCase(
            failure_id="fail_mesh_001",
            room_type="bedroom",
            stage="furniture",
            failure_type="deterministic_asset_error",
            bad_pattern="candidate mesh cannot be loaded",
            negative_constraint="do not retry the same missing HSSD mesh",
            critic_check="verify the replacement asset file exists",
            repair_action="mark candidate invalid and retrieve a different asset",
            repair_verified=True,
            is_deterministic=True,
            scope="stage",
        )

        self.assertIn(
            "do not retry the same missing HSSD mesh",
            record.to_hint_text(),
        )

        text = build_embedding_text(record)

        self.assertIn("memory_type=failure", text)
        self.assertIn("scope=stage", text)
        self.assertIn("is_deterministic=true", text)
        self.assertIn(
            "negative_constraint=do not retry the same missing HSSD mesh",
            text,
        )

    def test_memory_writer_gates_low_quality_success_and_keeps_failure(self) -> None:
        writer = MemoryWriter.__new__(MemoryWriter)
        full_report = FullVerifyReport(overall_score=0.4, pass_scene=False)
        ops = [
            MemoryUpdateOp(
                op="ADD",
                memory_type="success_case",
                content={
                    "case_id": "success_low_score",
                    "room_type": "bedroom",
                    "stage": "furniture",
                    "task_signature": ["bed"],
                    "successful_pattern": ["bed exists"],
                    "scores": {"semantic": 0.4},
                },
            ),
            MemoryUpdateOp(
                op="ADD",
                memory_type="failure_case",
                content={
                    "failure_id": "fail_missing_mesh",
                    "room_type": "bedroom",
                    "stage": "furniture",
                    "failure_type": "missing_mesh",
                    "bad_pattern": "HSSD candidate file missing",
                    "failure_reason": "missing mesh file",
                    "repair_action": "retrieve another asset",
                    "repair_verified": False,
                },
            ),
        ]

        filtered = writer._gate_and_enrich_ops(ops, full_report)

        self.assertEqual(1, len(filtered))
        failure = filtered[0]
        self.assertEqual("failure_case", failure.memory_type)
        self.assertIs(True, failure.content["is_deterministic"])
        self.assertEqual("stage", failure.content["scope"])
        self.assertTrue(failure.content["embedding_text"])

    def test_memory_writer_extracts_reasoning_content_and_markdown_json(self) -> None:
        writer = MemoryWriter.__new__(MemoryWriter)
        message = types.SimpleNamespace(
            content=None,
            model_dump=lambda: {
                "content": None,
                "model_extra": {
                    "reasoning_content": (
                        "```json\n"
                        '{"updates":[{"op":"NOOP","memory_type":"success_case","content":{}}]}'
                        "\n```"
                    )
                },
            },
        )
        response = types.SimpleNamespace(
            choices=[types.SimpleNamespace(message=message, finish_reason="stop")]
        )

        raw = writer._extract_response_text(response)
        data = writer._parse_json_payload(raw)

        self.assertEqual(1, len(data["updates"]))
        self.assertEqual("NOOP", data["updates"][0]["op"])

    def test_memory_writer_builds_conservative_fallback_success_ops(self) -> None:
        writer = MemoryWriter.__new__(MemoryWriter)
        trace_summary = "\n".join(
            [
                "Trace: trace_000001",
                "Prompt: A bedroom with a bed, two nightstands, and a wardrobe.",
                "Stages:",
                "  [furniture] objective='Complete furniture' verify=PASS "
                "scores=(semantic=0.90, aesthetic=0.80, physics=0.90)",
            ]
        )
        full_report = FullVerifyReport(overall_score=0.8, pass_scene=True)

        ops = writer._fallback_success_ops(trace_summary, full_report)
        filtered = writer._gate_and_enrich_ops(ops, full_report)

        self.assertEqual(1, len(filtered))
        op = filtered[0]
        self.assertEqual("ADD", op.op)
        self.assertEqual("success_case", op.memory_type)
        self.assertEqual("furniture", op.content["stage"])
        self.assertEqual("bedroom", op.content["room_type"])
        self.assertIn("bed", op.content["required_objects"])
        self.assertTrue(op.content["embedding_text"])

    def test_embedding_model_dir_resolves_to_bge_m3_under_models_dir(self) -> None:
        with patch.dict(
            "os.environ",
            {
                "SCENEEXPERT_MODELS_DIR": "/models",
            },
            clear=True,
        ):
            self.assertEqual(
                Path("/models/bge-m3"),
                resolve_memory_embedding_model_dir(),
            )

        with patch.dict(
            "os.environ",
            {
                "SCENEEXPERT_MODELS_DIR": "/models",
                "SCENEEXPERT_MEMORY_EMBEDDING_MODEL_DIR": "/custom/bge-m3",
            },
            clear=True,
        ):
            self.assertEqual(
                Path("/custom/bge-m3"),
                resolve_memory_embedding_model_dir(),
            )

    def test_embedder_pins_flagembedding_to_single_device(self) -> None:
        calls: dict[str, object] = {}

        class DummyBGEM3FlagModel:
            def __init__(self, model_dir: str, **kwargs: object) -> None:
                calls["model_dir"] = model_dir
                calls["kwargs"] = kwargs

            def encode(self, texts: list[str], **kwargs: object) -> dict[str, object]:
                return {"dense_vecs": [[1.0, 0.0] for _ in texts]}

        fake_module = types.SimpleNamespace(BGEM3FlagModel=DummyBGEM3FlagModel)
        with TemporaryDirectory() as tmp:
            with patch.dict(sys.modules, {"FlagEmbedding": fake_module}):
                embedder = SceneMemoryEmbedder(model_dir=tmp, device="cpu")
                matrix = embedder.encode(["bedroom furniture"])

        self.assertEqual(str(Path(tmp)), calls["model_dir"])
        kwargs = calls["kwargs"]
        self.assertIsInstance(kwargs, dict)
        self.assertEqual(["cpu"], kwargs["devices"])
        self.assertEqual((1, 2), matrix.shape)

    def test_numpy_memory_index_searches_normalized_vectors(self) -> None:
        with TemporaryDirectory() as tmp:
            index = NumpyMemoryIndex.for_bank(Path(tmp), "success", "furniture")
            index.build(
                vectors=np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
                metadata=[
                    {"memory_id": "bed_case"},
                    {"memory_id": "sofa_case"},
                ],
                manifest={"embedding_model_dir": "/models/bge-m3"},
            )

            loaded = NumpyMemoryIndex.for_bank(Path(tmp), "success", "furniture")
            results = loaded.search(np.asarray([0.9, 0.1], dtype=np.float32), top_k=1)

            self.assertEqual(1, len(results))
            self.assertEqual("bed_case", results[0][1]["memory_id"])

    def test_build_memory_indexes_writes_numpy_files_with_fallback_text(self) -> None:
        class DummyEmbedder:
            def __init__(self) -> None:
                self.texts: list[str] = []

            def encode(self, texts: list[str]) -> np.ndarray:
                self.texts.extend(texts)
                return np.asarray([[1.0, 0.0] for _ in texts], dtype=np.float32)

        with TemporaryDirectory() as tmp:
            memory_dir = Path(tmp) / "memory"
            memory_dir.mkdir()
            record = SuccessCase(
                case_id="success_bedroom_001",
                room_type="bedroom",
                stage="furniture",
                task_signature=["bed", "nightstand"],
                successful_pattern=["bed centered on main wall"],
            )
            (memory_dir / "success_cases.jsonl").write_text(
                record.model_dump_json() + "\n",
                encoding="utf-8",
            )

            embedder = DummyEmbedder()
            summaries = build_memory_indexes(
                memory_dir=memory_dir,
                embedding_model_dir=Path("/models/bge-m3"),
                stages=("furniture",),
                memory_types=("success",),
                embedder=embedder,
            )

            self.assertEqual(1, len(summaries))
            self.assertEqual(1, summaries[0]["count"])
            self.assertIn("memory_type=success", embedder.texts[0])

            index = NumpyMemoryIndex.for_bank(
                memory_dir / "indexes",
                "success",
                "furniture",
            )
            index.load()
            self.assertEqual((1, 2), index.vectors.shape)
            self.assertEqual(
                "success_bedroom_001",
                index.metadata[0]["memory_id"],
            )
            self.assertEqual(
                str(Path("/models/bge-m3")),
                index.manifest["embedding_model_dir"],
            )

    def test_hybrid_retriever_strict_mode_fails_on_missing_index(self) -> None:
        class DummyEmbedder:
            def encode(self, texts: list[str]) -> np.ndarray:
                return np.asarray([[1.0, 0.0] for _ in texts], dtype=np.float32)

        with TemporaryDirectory() as tmp:
            memory_dir = Path(tmp) / "memory"
            memory_dir.mkdir()
            success = SuccessCase(
                case_id="success_bedroom_001",
                room_type="bedroom",
                stage="furniture",
                required_objects=["bed"],
                positive_guidance=["place bed first"],
            )
            (memory_dir / "success_cases.jsonl").write_text(
                success.model_dump_json() + "\n",
                encoding="utf-8",
            )
            store = FastMemoryStore(str(memory_dir))

            with self.assertRaisesRegex(
                FileNotFoundError,
                "Hybrid memory index is missing",
            ):
                HybridMemoryRetriever(
                    store=store,
                    memory_dir=str(memory_dir),
                    embedder=DummyEmbedder(),
                    require_indexes=True,
                    auto_build_indexes=False,
                )

    def test_hybrid_retriever_auto_builds_missing_index(self) -> None:
        class DummyEmbedder:
            model_dir = Path("/models/bge-m3")
            model_id = "BAAI/bge-m3"
            device = "cpu"
            batch_size = 8
            max_length = 512

            def encode(self, texts: list[str]) -> np.ndarray:
                return np.asarray([[1.0, 0.0] for _ in texts], dtype=np.float32)

        with TemporaryDirectory() as tmp:
            memory_dir = Path(tmp) / "memory"
            memory_dir.mkdir()
            success = SuccessCase(
                case_id="success_bedroom_001",
                room_type="bedroom",
                stage="furniture",
                required_objects=["bed"],
                positive_guidance=["place bed first"],
            )
            (memory_dir / "success_cases.jsonl").write_text(
                success.model_dump_json() + "\n",
                encoding="utf-8",
            )
            store = FastMemoryStore(str(memory_dir))
            retriever = HybridMemoryRetriever(
                store=store,
                memory_dir=str(memory_dir),
                embedder=DummyEmbedder(),
                max_success=1,
                max_failure=0,
                max_skills=0,
                require_indexes=True,
                auto_build_indexes=True,
            )

            index = NumpyMemoryIndex.for_bank(
                memory_dir / "indexes",
                "success",
                "furniture",
            )
            self.assertTrue(index.vectors_path.exists())
            self.assertTrue(index.metadata_path.exists())

            pack = retriever.retrieve(
                SceneTaskSpec(
                    room_type="bedroom",
                    style="standard",
                    required_large_objects=["bed"],
                ),
                "furniture",
            )
            self.assertEqual(1, len(pack.success_hints))
            self.assertIn("place bed first", pack.success_hints[0])

    def test_hybrid_retriever_reads_numpy_indexes_and_reranks_memory(self) -> None:
        class DummyEmbedder:
            def encode(self, texts: list[str]) -> np.ndarray:
                del texts
                return np.asarray([[1.0, 0.0]], dtype=np.float32)

        with TemporaryDirectory() as tmp:
            memory_dir = Path(tmp) / "memory"
            memory_dir.mkdir()
            success = SuccessCase(
                case_id="success_bedroom_001",
                room_type="bedroom",
                style="modern",
                stage="furniture",
                required_objects=["bed", "nightstand", "wardrobe"],
                successful_pattern=["bed centered on main wall"],
                positive_guidance=["use bed as the anchor"],
                placement_reference=["bed_1 (bed): x=0.0, y=0.0, yaw=0"],
                scores={"semantic": 0.9, "aesthetic": 0.8, "physics": 0.9},
            )
            failure = FailureCase(
                failure_id="fail_asset_001",
                room_type="kitchen",
                stage="furniture",
                failure_type="missing_mesh",
                bad_pattern="HSSD candidate file missing",
                negative_constraint="do not retry the same missing HSSD file",
                repair_action="retrieve another asset",
                is_deterministic=True,
                scope="stage",
            )
            skill = Skill(
                skill_name="arrange_bedroom_anchor",
                stage="furniture",
                room_types=["bedroom"],
                required_objects=["bed", "nightstand"],
                preconditions=["bedroom furniture stage"],
                procedure=["place bed first", "place nightstands beside bed"],
                failure_avoidance=["do not block the wardrobe"],
            )
            (memory_dir / "success_cases.jsonl").write_text(
                success.model_dump_json() + "\n",
                encoding="utf-8",
            )
            (memory_dir / "failure_cases.jsonl").write_text(
                failure.model_dump_json() + "\n",
                encoding="utf-8",
            )
            (memory_dir / "skills.jsonl").write_text(
                skill.model_dump_json() + "\n",
                encoding="utf-8",
            )

            build_memory_indexes(
                memory_dir=memory_dir,
                embedding_model_dir=Path("/models/bge-m3"),
                stages=("furniture",),
                memory_types=("success", "failure", "skill"),
                embedder=DummyEmbedder(),
            )
            store = FastMemoryStore(str(memory_dir))
            retriever = HybridMemoryRetriever(
                store=store,
                memory_dir=str(memory_dir),
                embedder=DummyEmbedder(),
                max_success=1,
                max_failure=1,
                max_skills=1,
                require_indexes=True,
            )
            task_spec = SceneTaskSpec(
                room_type="bedroom",
                style="modern",
                required_large_objects=["bed", "nightstand", "wardrobe"],
                functional_zones=["sleeping_zone", "storage_zone"],
            )

            pack = retriever.retrieve(task_spec, "furniture")

            self.assertEqual(1, len(pack.success_hints))
            self.assertIn("use bed as the anchor", pack.success_hints[0])
            self.assertIn("Reference Layout", pack.placement_reference)
            self.assertEqual(1, len(pack.failure_hints))
            self.assertIn("do not retry", pack.failure_hints[0])
            self.assertEqual(1, len(pack.skill_texts))
            self.assertIn("arrange_bedroom_anchor", pack.skill_texts[0])

    def test_hybrid_retriever_writes_timing_jsonl(self) -> None:
        class DummyEmbedder:
            def encode(self, texts: list[str]) -> np.ndarray:
                del texts
                return np.asarray([[1.0, 0.0]], dtype=np.float32)

        with TemporaryDirectory() as tmp:
            memory_dir = Path(tmp) / "memory"
            memory_dir.mkdir()
            timing_path = (
                Path(tmp) / "scene_expert" / "timing" / "memory_retrieval.jsonl"
            )
            success = SuccessCase(
                case_id="success_bedroom_001",
                room_type="bedroom",
                stage="furniture",
                required_objects=["bed"],
                positive_guidance=["place bed first"],
            )
            (memory_dir / "success_cases.jsonl").write_text(
                success.model_dump_json() + "\n",
                encoding="utf-8",
            )
            build_memory_indexes(
                memory_dir=memory_dir,
                embedding_model_dir=Path("/models/bge-m3"),
                stages=("furniture",),
                memory_types=("success",),
                embedder=DummyEmbedder(),
            )
            store = FastMemoryStore(str(memory_dir))
            retriever = HybridMemoryRetriever(
                store=store,
                memory_dir=str(memory_dir),
                embedder=DummyEmbedder(),
                max_success=1,
                max_failure=0,
                max_skills=0,
                require_indexes=True,
                timing_path=timing_path,
            )

            retriever.retrieve(
                SceneTaskSpec(
                    room_type="bedroom",
                    style="standard",
                    required_large_objects=["bed"],
                ),
                "furniture",
            )

            self.assertTrue(timing_path.exists())
            timing = json.loads(timing_path.read_text(encoding="utf-8").splitlines()[0])
            self.assertEqual("hybrid", timing["retriever_type"])
            self.assertIn("embedding_encode_sec", timing)
            self.assertIn("index_load_sec", timing)
            self.assertIn("vector_search_sec", timing)
            self.assertIn("rerank_sec", timing)
            self.assertIn("total_sec", timing)


if __name__ == "__main__":
    unittest.main()
