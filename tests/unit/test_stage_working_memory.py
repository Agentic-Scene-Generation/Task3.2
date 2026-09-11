import json
import os
import unittest

from pathlib import Path
from tempfile import TemporaryDirectory

from scenesmith.agent_utils.scoring import CategoryScore, FurnitureCritiqueWithScores
from scenesmith.agent_utils.stage_working_memory import (
    StageWorkingMemory,
    _canonical_stage,
    _score_total,
)


class _DummyObject:
    def __init__(self, name: str) -> None:
        self.name = name


class _DummyScene:
    def __init__(self) -> None:
        self.objects = {
            "bed_0": _DummyObject("bed_0"),
            "nightstand_0": _DummyObject("nightstand_0"),
        }

    def content_hash(self) -> str:
        return "scene-hash"


class _MissingBedScene:
    def __init__(self) -> None:
        self.objects = {
            "nightstand_0": _DummyObject("nightstand"),
            "nightstand_1": _DummyObject("nightstand"),
            "corner_wardrobe_0": _DummyObject("corner_wardrobe"),
        }

    def content_hash(self) -> str:
        return "missing-bed-scene"


class _StudyScene:
    def __init__(self) -> None:
        self.objects = {
            "desk_0": _DummyObject("desk"),
            "office_chair_0": _DummyObject("office_chair"),
            "guest_armchair_0": _DummyObject("guest_armchair"),
            "guest_armchair_1": _DummyObject("guest_armchair"),
            "bookshelf_0": _DummyObject("bookshelf"),
        }

    def content_hash(self) -> str:
        return "study-scene"


def _score(name: str, grade: int) -> CategoryScore:
    return CategoryScore(name=name, grade=grade, comment=f"{name} score")


class StageWorkingMemoryTest(unittest.TestCase):
    def test_frozen_public_bank_stays_unchanged(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            public_dir = root / "frozen_memory" / "ablation_4c"
            public_dir.mkdir(parents=True)
            events_path = public_dir / "events.jsonl"
            events_path.write_text('{"event_type":"seed"}\n', encoding="utf-8")
            before = events_path.read_bytes()
            old_env = os.environ.get("SCENEEXPERT_ACTIVE_MEMORY_BANK_DIR")
            old_read_only_env = os.environ.get(
                "SCENEEXPERT_ACTIVE_MEMORY_BANK_READ_ONLY"
            )
            os.environ["SCENEEXPERT_ACTIVE_MEMORY_BANK_DIR"] = str(public_dir)
            os.environ["SCENEEXPERT_ACTIVE_MEMORY_BANK_READ_ONLY"] = "true"
            try:
                memory = StageWorkingMemory(
                    root_dir=root / "scene_000" / "room_bedroom",
                    stage="furniture",
                    enabled=True,
                )
                render_dir = root / "renders_001"
                render_dir.mkdir()
                memory.save_render_record(
                    render_dir=render_dir,
                    role="designer",
                    event="request_initial_design",
                    scene=_DummyScene(),
                )
                memory.record_llm_call(
                    agent_role="designer",
                    event="request_initial_design",
                    prompt="design a bedroom",
                    output="done",
                )
                memory.record_planner_orchestration(
                    call_id="call_1",
                    phase="dispatch",
                    operation="request_initial_design",
                    child_agent="designer",
                    status="completed",
                )
            finally:
                if old_env is None:
                    os.environ.pop("SCENEEXPERT_ACTIVE_MEMORY_BANK_DIR", None)
                else:
                    os.environ["SCENEEXPERT_ACTIVE_MEMORY_BANK_DIR"] = old_env
                if old_read_only_env is None:
                    os.environ.pop("SCENEEXPERT_ACTIVE_MEMORY_BANK_READ_ONLY", None)
                else:
                    os.environ["SCENEEXPERT_ACTIVE_MEMORY_BANK_READ_ONLY"] = (
                        old_read_only_env
                    )

            self.assertEqual(before, events_path.read_bytes())
            self.assertTrue(memory.debug_memory_path.exists())
            self.assertTrue(memory.debug_llm_path.exists())

    def test_repair_event_is_saved_with_pose_changes(self) -> None:
        with TemporaryDirectory() as tmp:
            memory = StageWorkingMemory(Path(tmp) / "room_bedroom", "furniture")
            record = memory.record_repair_event(
                source="initial_design",
                strategy="prompt_contract_furniture_relations",
                status="accepted",
                trigger_reasons=["window_clearance__wardrobe_0"],
                actions=["wardrobe_0:window_clearance"],
                affected_objects=[
                    {
                        "object_id": "wardrobe_0",
                        "before": {"xy": [-1.8, -1.7]},
                        "after": {"xy": [0.58, -1.7]},
                    }
                ],
            )

            rows = memory.debug_repair_path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(1, len(rows))
            saved = json.loads(rows[0])
            self.assertEqual(record, saved)
            self.assertEqual("accepted", saved["status"])
            self.assertEqual("scenesmith_core", saved["repair_owner"])
            self.assertEqual([0.58, -1.7], saved["affected_objects"][0]["after"]["xy"])

    def test_score_total_and_stage_canonicalization(self) -> None:
        scores = FurnitureCritiqueWithScores(
            critique="layout is usable",
            realism=_score("realism", 8),
            functionality=_score("functionality", 7),
            layout=_score("layout", 6),
            layout_plausibility=_score("layout_plausibility", 5),
            holistic_completeness=_score("holistic_completeness", 8),
            prompt_following=_score("prompt_following", 9),
            reachability=_score("reachability", 7),
        )

        self.assertEqual(50.0, _score_total(scores))
        self.assertIsNone(_score_total(None))
        self.assertEqual("wall_mounted", _canonical_stage("wall"))
        self.assertEqual("ceiling_mounted", _canonical_stage("ceiling"))
        self.assertEqual("manipuland", _canonical_stage("manipulands_table"))
        self.assertEqual("furniture", _canonical_stage("furniture"))

    def test_render_record_is_saved_and_retrieved(self) -> None:
        with TemporaryDirectory() as tmp:
            root_dir = Path(tmp)
            render_dir = root_dir / "scene_renders" / "furniture" / "renders_001"
            render_dir.mkdir(parents=True)
            (render_dir / "0_top.png").write_bytes(b"image")

            memory = StageWorkingMemory(
                root_dir=root_dir,
                stage="furniture",
                enabled=True,
            )
            record = memory.save_render_record(
                render_dir=render_dir,
                role="critic",
                event="critique",
                scene=_DummyScene(),
                text="place the bed first",
                critique="keep both nightstands beside the bed",
            )

            self.assertEqual("scene-hash", record["scene_hash"])
            self.assertEqual(2, record["object_count"])
            self.assertTrue((render_dir / "render_memory.json").is_file())
            memory_lines = memory.memory_path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(1, len(memory_lines))
            self.assertEqual("critic", json.loads(memory_lines[0])["role"])

            retrieved = memory.retrieve_for_designer(
                query="bed nightstands",
                max_items=1,
            )
            self.assertIn("keep both nightstands beside the bed", retrieved)
            self.assertIn(str(render_dir), retrieved)

    def test_missing_required_object_overrides_hallucinated_success_critique(
        self,
    ) -> None:
        with TemporaryDirectory() as tmp:
            root_dir = Path(tmp)
            render_dir = root_dir / "scene_renders" / "furniture" / "renders_003"
            render_dir.mkdir(parents=True)
            (render_dir / "0_top.png").write_bytes(b"image")

            scores = FurnitureCritiqueWithScores(
                critique="All required furniture is present: bed, two nightstands, wardrobe.",
                realism=_score("realism", 8),
                functionality=_score("functionality", 10),
                layout=_score("layout", 9),
                layout_plausibility=_score("layout_plausibility", 8),
                holistic_completeness=_score("holistic_completeness", 7),
                prompt_following=_score("prompt_following", 10),
                reachability=_score("reachability", 10),
            )
            memory = StageWorkingMemory(
                root_dir=root_dir,
                stage="furniture",
                enabled=True,
            )
            memory.set_required_counts({"bed": 1, "nightstand": 2, "wardrobe": 1})
            record = memory.save_render_record(
                render_dir=render_dir,
                role="critic",
                event="critique",
                scene=_MissingBedScene(),
                scores=scores,
                critique=scores.critique,
            )

            quality = record["deterministic_quality"]
            self.assertFalse(quality["hard_valid"])
            self.assertTrue(quality["critic_inconsistent_with_state"])
            self.assertEqual(["bed"], quality["missing_required_objects"])

            retrieved = memory.retrieve_for_designer(
                query="bed missing required furniture",
                max_items=1,
            )
            self.assertIn("missing required furniture bed", retrieved)
            self.assertIn("Ignore contradictory critic", retrieved)
            self.assertNotIn("critic: All required furniture is present", retrieved)

    def test_required_counts_use_shared_furniture_category_hierarchy(self) -> None:
        with TemporaryDirectory() as tmp:
            root_dir = Path(tmp)
            render_dir = root_dir / "scene_renders" / "furniture" / "renders_004"
            render_dir.mkdir(parents=True)

            memory = StageWorkingMemory(root_dir, "furniture", enabled=True)
            memory.set_required_counts({"desk": 1, "chair": 3, "bookshelf": 1})
            record = memory.save_render_record(
                render_dir=render_dir,
                role="designer",
                event="design",
                scene=_StudyScene(),
            )

            quality = record["deterministic_quality"]
            self.assertTrue(quality["hard_valid"])
            self.assertEqual(
                {"desk": 1, "chair": 3, "bookshelf": 1},
                quality["observed_counts"],
            )
            self.assertEqual([], quality["missing_required_objects"])

    def test_planner_orchestration_records_dispatch_and_resume(self) -> None:
        with TemporaryDirectory() as tmp:
            memory = StageWorkingMemory(Path(tmp), "furniture", enabled=True)

            memory.record_planner_orchestration(
                call_id="furniture:request_initial_design:001",
                phase="dispatch",
                operation="request_initial_design",
                child_agent="designer",
                status="started",
            )
            memory.record_planner_orchestration(
                call_id="furniture:request_initial_design:001",
                phase="resume",
                operation="request_initial_design",
                child_agent="designer",
                status="completed",
            )

            records = [
                json.loads(line)
                for line in memory.debug_orchestration_path.read_text(
                    encoding="utf-8"
                ).splitlines()
            ]
            self.assertEqual(["dispatch", "resume"], [row["phase"] for row in records])
            self.assertEqual("designer", records[0]["child_agent"])
            self.assertTrue(records[0]["created_at"].endswith("Z"))


if __name__ == "__main__":
    unittest.main()
