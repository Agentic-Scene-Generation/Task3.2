"""Tests for deterministic behavior-template planning and task merge."""

import json
import subprocess
import sys
import unittest

from collections import Counter
from types import SimpleNamespace

from scenesmith.scene_expert.behavior import (
    BehaviorSpec,
    TemplateBehaviorPlanner,
    apply_behavior_template,
    build_behavior_spec,
    merge_behavior_assets,
)
from scenesmith.scene_expert.schemas import SceneTaskSpec


class TemplateBehaviorPlannerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.planner = TemplateBehaviorPlanner()

    def test_bedroom_matches_latest_template_contract(self) -> None:
        spec = self.planner.plan("A minimalist bedroom with a bed and wardrobe.")

        self.assertEqual(spec.generation_mode, "deterministic_template")
        self.assertEqual(spec.persona.name, "Maya")
        self.assertEqual(spec.target_rooms, ["bedroom"])
        self.assertEqual([room.room_type for room in spec.rooms], ["bedroom"])
        room = spec.rooms[0]
        self.assertEqual(
            set(room.weekly_schedule),
            {
                "Monday",
                "Tuesday",
                "Wednesday",
                "Thursday",
                "Friday",
                "Saturday",
                "Sunday",
            },
        )
        self.assertEqual(room.weekly_schedule["Monday"][0].start, "07:00")
        self.assertEqual(room.weekly_schedule["Tuesday"][0].start, "07:07")
        self.assertEqual(room.weekly_schedule["Saturday"][0].start, "08:14")
        furniture = {item.name for item in room.assets_by_stage["furniture"]}
        small = {item.name for item in room.assets_by_stage["manipuland"]}
        self.assertTrue({"bed", "desk", "chair", "wardrobe"} <= furniture)
        self.assertTrue({"laptop", "book", "phone"} <= small)
        self.assertIn(
            ("chair", "facing", "desk"),
            {(rel.subject, rel.relation, rel.object) for rel in room.relations},
        )
        self.assertIn("bedroom", spec.object_needs)
        self.assertIn("bedroom", spec.detailed_routines)
        self.assertIn(
            "AGENTSENSE_ROOM_BEHAVIOR_REQUIREMENTS",
            spec.room_behavior_blocks["bedroom"],
        )
        self.assertIn("REQUIRED small objects/manipulands", spec.enriched_prompt)

    def test_explicit_quantity_wins_and_is_grouped(self) -> None:
        spec = self.planner.plan("A bedroom with three books on a desk.")
        books = [
            item
            for item in spec.rooms[0].assets_by_stage["manipuland"]
            if item.name == "book" and item.source == "explicit_prompt"
        ]

        self.assertEqual(len(books), 1)
        self.assertEqual(books[0].quantity, 3)

    def test_wall_asset_uses_wall_stage(self) -> None:
        spec = self.planner.plan("A bathroom with a mirror and toilet.")
        wall_names = {
            item.name for item in spec.rooms[0].assets_by_stage["wall_mounted"]
        }

        self.assertEqual(wall_names, {"mirror"})

    def test_task_compiler_room_alias_is_normalized(self) -> None:
        spec = self.planner.plan("A quiet room.", room_type="living_room")

        self.assertEqual(spec.rooms[0].room_type, "livingroom")

    def test_wheelchair_does_not_count_as_explicit_chair(self) -> None:
        spec = self.planner.plan("A kitchen for a wheelchair user.")
        explicit = {
            item.name
            for items in spec.rooms[0].assets_by_stage.values()
            for item in items
            if item.source == "explicit_prompt"
        }

        self.assertNotIn("chair", explicit)

    def test_only_week_horizon_is_supported(self) -> None:
        with self.assertRaisesRegex(ValueError, "Only horizon='week'"):
            self.planner.plan("A bedroom", horizon="day")

    def test_model_persona_is_used_when_valid(self) -> None:
        response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=(
                            '{"name":"Lin","age":72,"job":"retired teacher",'
                            '"health":"limited mobility","traits":["patient"],'
                            '"description":"Lin reads every evening at home."}'
                        )
                    )
                )
            ]
        )
        client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=lambda **_: response)
            )
        )

        spec = TemplateBehaviorPlanner(model="test", client=client).plan(
            "A living room for an elderly reader."
        )

        self.assertEqual(spec.persona.name, "Lin")
        self.assertEqual(spec.persona_generation, "model")

    def test_model_failure_falls_back_to_maya(self) -> None:
        def fail(**_) -> None:
            raise RuntimeError("offline")

        client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=fail))
        )

        spec = TemplateBehaviorPlanner(model="test", client=client).plan("A bedroom")

        self.assertEqual(spec.persona.name, "Maya")
        self.assertEqual(spec.persona_generation, "fallback")

    def test_model_infers_room_when_prompt_has_no_room_label(self) -> None:
        room_response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content='{"rooms":["kitchen"]}')
                )
            ]
        )
        persona_response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=(
                            '{"name":"Kai","age":40,"job":"chef",'
                            '"health":"good","traits":["organized"],'
                            '"description":"Kai prepares meals every day."}'
                        )
                    )
                )
            ]
        )
        responses = iter([room_response, persona_response])
        client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=lambda **_: next(responses))
            )
        )

        spec = TemplateBehaviorPlanner(model="test", client=client).plan(
            "A compact home space for someone who cooks every day."
        )

        self.assertEqual(spec.target_rooms, ["kitchen"])
        self.assertIn(
            "refrigerator",
            {
                item.name
                for item in spec.assets_by_room_and_stage["kitchen"]["furniture"]
            },
        )

    def test_latest_source_prompt_object_keywords_are_supported(self) -> None:
        bedroom = self.planner.plan(
            "A bedroom with a lamp, computer, mug, and water glass."
        )
        living = self.planner.plan("A living room with a magazine and plant.")
        kitchen = self.planner.plan("A kitchen with a dining table and coffee maker.")

        explicit = lambda spec: {
            item.name
            for items in spec.rooms[0].assets_by_stage.values()
            for item in items
            if item.source == "explicit_prompt"
        }
        self.assertTrue({"lamp", "computer", "mug", "water glass"} <= explicit(bedroom))
        self.assertTrue({"magazine", "plant"} <= explicit(living))
        self.assertTrue({"kitchen table", "coffee maker"} <= explicit(kitchen))

    def test_compatible_spec_round_trips_as_json(self) -> None:
        spec = build_behavior_spec(
            "A small apartment with a bedroom, kitchen, and living room."
        )
        restored = BehaviorSpec.model_validate_json(spec.model_dump_json())

        self.assertEqual(restored.target_rooms, ["bedroom", "kitchen", "livingroom"])
        self.assertEqual(
            set(restored.weekly_schedule),
            {
                "Monday",
                "Tuesday",
                "Wednesday",
                "Thursday",
                "Friday",
                "Saturday",
                "Sunday",
            },
        )
        self.assertEqual(set(restored.object_needs), set(restored.target_rooms))
        self.assertEqual(
            set(restored.assets_by_room_and_stage), set(restored.target_rooms)
        )
        self.assertEqual(
            set(restored.assets_by_room_and_stage["bedroom"]),
            {"furniture", "wall_mounted", "ceiling_mounted", "manipuland"},
        )
        self.assertEqual(set(restored.placement_relations), set(restored.target_rooms))

    def test_source_compatible_and_grouped_views_are_separate(self) -> None:
        spec = build_behavior_spec("A bedroom.")

        legacy_bed = next(
            item for item in spec.object_needs["bedroom"] if item.name == "bed"
        )
        grouped_bed = next(
            item
            for item in spec.assets_by_room_and_stage["bedroom"]["furniture"]
            if item.name == "bed"
        )

        # Preserve the latest source view while preventing its substring match
        # ("bed" in "bedroom") from becoming a Task3.2 explicit requirement.
        self.assertEqual(
            legacy_bed.reason, "explicitly mentioned in the SceneSmith prompt"
        )
        self.assertEqual(grouped_bed.source, "behavior_template")


class BehaviorAssetMergeTest(unittest.TestCase):
    def test_merge_uses_max_quantity_and_preserves_explicit_inventory(self) -> None:
        behavior = TemplateBehaviorPlanner().plan(
            "A bedroom with three books and a wardrobe."
        )
        task = SceneTaskSpec(
            room_type="bedroom",
            style="minimalist",
            required_large_objects=["bed", "bed"],
            required_small_objects=["book", "book"],
        )

        merged = merge_behavior_assets(
            task,
            behavior,
            inferred_assets_are_required=True,
        )

        self.assertEqual(Counter(merged.required_large_objects)["bed"], 2)
        self.assertEqual(Counter(merged.required_small_objects)["book"], 3)
        self.assertIn("desk", merged.required_large_objects)
        self.assertEqual(merged.interaction_constraints, [])

    def test_soft_mode_only_merges_prompt_explicit_assets(self) -> None:
        behavior = TemplateBehaviorPlanner().plan(
            "A minimalist bedroom with a wardrobe."
        )
        task = SceneTaskSpec(room_type="bedroom", style="minimalist")

        merged = merge_behavior_assets(
            task,
            behavior,
            inferred_assets_are_required=False,
        )

        self.assertEqual(merged.required_large_objects, ["wardrobe"])
        self.assertEqual(merged.required_small_objects, [])
        self.assertEqual(merged.interaction_constraints, [])

    def test_merge_only_uses_primary_room(self) -> None:
        behavior = TemplateBehaviorPlanner().plan(
            "A bedroom connected to a kitchen with a refrigerator."
        )
        task = SceneTaskSpec(room_type="bedroom", style="standard")

        merged = merge_behavior_assets(
            task,
            behavior,
            inferred_assets_are_required=True,
        )

        self.assertIn("bed", merged.required_large_objects)
        self.assertNotIn("refrigerator", merged.required_large_objects)


class BehaviorIntegrationTest(unittest.TestCase):
    def test_disabled_integration_is_strict_noop(self) -> None:
        task = SceneTaskSpec(room_type="bedroom", style="standard")

        merged, behavior = apply_behavior_template(
            "A bedroom.",
            task,
            config={"enabled": False},
            output_path=self._temporary_output("disabled.json"),
        )

        self.assertIs(merged, task)
        self.assertIsNone(behavior)

    def test_enabled_integration_writes_loadable_spec(self) -> None:
        output = self._temporary_output("behavior_spec.json")
        task = SceneTaskSpec(room_type="bedroom", style="standard")

        merged, behavior = apply_behavior_template(
            "A bedroom with two nightstands.",
            task,
            config={
                "enabled": True,
                "planner": "template",
                "horizon": "week",
                "inferred_assets_are_required": True,
            },
            output_path=output,
        )

        self.assertIsNotNone(behavior)
        self.assertTrue(output.is_file())
        restored = BehaviorSpec.model_validate_json(output.read_text(encoding="utf-8"))
        self.assertEqual(restored.target_rooms, ["bedroom"])
        self.assertEqual(Counter(merged.required_large_objects)["nightstand"], 2)

    def test_module_cli_runs_and_writes_grouped_assets(self) -> None:
        output = self._temporary_output("cli_behavior_spec.json")
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "scenesmith.scene_expert.behavior",
                "--prompt",
                "A bathroom with a mirror and toilet.",
                "--output",
                str(output),
            ],
            check=False,
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        payload = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(payload["target_rooms"], ["bathroom"])
        wall_assets = payload["assets_by_room_and_stage"]["bathroom"]["wall_mounted"]
        self.assertEqual([item["name"] for item in wall_assets], ["mirror"])

    def _temporary_output(self, name: str):
        from pathlib import Path
        from tempfile import mkdtemp

        return Path(mkdtemp(prefix="behavior-template-test-")) / name


if __name__ == "__main__":
    unittest.main()
