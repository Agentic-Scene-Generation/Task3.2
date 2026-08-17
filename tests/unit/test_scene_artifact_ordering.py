import unittest

from pathlib import Path


class SceneArtifactOrderingTest(unittest.TestCase):
    def test_canonical_snapshots_precede_scene_expert_finalize(self) -> None:
        source_path = (
            Path(__file__).resolve().parents[2]
            / "scenesmith"
            / "experiments"
            / "indoor_scene_generation.py"
        )
        source = source_path.read_text(encoding="utf-8")
        snapshot_loop = source.index(
            "for name, types in snapshots[:snapshot_count]:"
        )
        final_verification = source.index(
            'final_scene_path=str(scene_dir / "combined_house")',
            snapshot_loop,
        )

        self.assertLess(snapshot_loop, final_verification)
        self.assertIn(
            "house_scene.assemble(",
            source[snapshot_loop:final_verification],
        )


if __name__ == "__main__":
    unittest.main()
