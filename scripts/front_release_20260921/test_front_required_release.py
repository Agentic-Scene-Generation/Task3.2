import importlib.util
import unittest
from pathlib import Path


class FrontContract(unittest.TestCase):
    def resolve(self, row, candidate=None):
        path = Path(__file__).with_name("front_required_release.py")
        self.assertTrue(
            path.exists(), "required-front resolver has not been implemented"
        )
        spec = importlib.util.spec_from_file_location("required_front", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.resolve(row, candidate)

    def test_missing_and_invalid_axis_get_explicit_nonsemantic_fallback(self):
        for axis in (None, [0, 0, 0], [0, 1, 0], [float("nan"), 0, 1]):
            got = self.resolve(
                {"canonical_front": {"canonical_orientation_axis": axis}}
            )
            self.assertEqual(got["canonical_orientation_axis"], [0.0, 0.0, 1.0])
            self.assertFalse(got["canonical_orientation_is_semantic_front"])

    def test_uncertain_keeps_direction_but_revokes_semantic_claim(self):
        got = self.resolve(
            {
                "canonical_front": {
                    "canonical_orientation_axis": [-1, 0, 0],
                    "canonical_orientation_is_semantic_front": True,
                }
            },
            {"result": {"front_status": "uncertain"}},
        )
        self.assertEqual(got["canonical_orientation_axis"], [-1.0, 0.0, 0.0])
        self.assertFalse(got["canonical_orientation_is_semantic_front"])
        self.assertEqual(got["luna_front_status"], "uncertain")

    def test_unique_uses_evidence_normalized_frame_not_native_z_up(self):
        c = {
            "result": {
                "front_status": "unique",
                "confidence": 0.9,
                "up_confidence": 0.9,
                "front_axis": [0, 0, 1],
            },
            "front_axis_evidence_local": [0, 0, 1],
            "front_axis_source_local": [0, -1, 0],
            "source_to_evidence_matrix": [[1, 0, 0], [0, 0, 1], [0, -1, 0]],
            "consistency_flags": [],
        }
        got = self.resolve({}, c)
        self.assertEqual(got["canonical_orientation_axis"], [0.0, 0.0, 1.0])
        self.assertTrue(got["canonical_orientation_is_semantic_front"])
        self.assertFalse(got["is_strict_front"])

    def test_nonhorizontal_luna_does_not_change_mesh_upright(self):
        c = {
            "result": {
                "front_status": "unique",
                "confidence": 0.99,
                "up_confidence": 0.99,
            },
            "front_axis_evidence_local": [0, 1, 0],
            "consistency_flags": [],
        }
        self.assertFalse(self.resolve({}, c)["canonical_orientation_is_semantic_front"])

    def test_multiple_chooses_one_without_claiming_uniqueness(self):
        c = {
            "result": {
                "front_status": "multiple_valid",
                "alternative_front_axes": [[1, 0, 0], [-1, 0, 0]],
                "confidence": 0.9,
                "up_confidence": 0.9,
            },
            "consistency_flags": [],
        }
        got = self.resolve({}, c)
        self.assertEqual(got["canonical_orientation_axis"], [1.0, 0.0, 0.0])
        self.assertEqual(got["luna_front_status"], "multiple_valid")


if __name__ == "__main__":
    unittest.main()
