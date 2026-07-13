import json
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


def _score(name: str, grade: int) -> CategoryScore:
    return CategoryScore(name=name, grade=grade, comment=f"{name} score")


class StageWorkingMemoryTest(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
