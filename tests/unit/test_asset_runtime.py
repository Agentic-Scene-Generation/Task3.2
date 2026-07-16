import unittest

from types import SimpleNamespace

from scenesmith.agent_utils.asset_runtime import (
    AssetRuntimeGate,
    semantic_asset_family,
)


class AssetRuntimeGateTest(unittest.TestCase):
    def test_semantic_variants_share_one_family(self) -> None:
        self.assertEqual(
            semantic_asset_family("metal-framed circular mirror"),
            semantic_asset_family("rectangular silver wall mirror"),
        )
        self.assertEqual(semantic_asset_family("卧室床头柜"), "nightstand")

    def test_required_family_survives_optional_budget(self) -> None:
        gate = AssetRuntimeGate()
        gate.configure(
            stage="furniture",
            budget={
                "max_asset_requests": 1,
                "max_optional_object_families": 1,
                "max_assets_per_request": 4,
                "max_semantic_retries_per_family": 2,
            },
            required_objects=["bed"],
        )
        first = gate.plan(["decorative plant"], ["plant"])
        second = gate.plan(["double bed", "area rug"], ["bed", "rug"])

        self.assertEqual(first.allowed_indices, [0])
        self.assertEqual(second.allowed_indices, [0])
        self.assertTrue(any(failure.index == 1 for failure in second.failures))

    def test_success_is_reused_without_new_request(self) -> None:
        gate = AssetRuntimeGate()
        gate.configure(
            stage="wall_mounted",
            budget={
                "max_asset_requests": 2,
                "max_optional_object_families": 2,
                "max_assets_per_request": 2,
                "max_semantic_retries_per_family": 1,
            },
            required_objects=[],
        )
        first = gate.plan(["round wall mirror"], ["mirror"])
        gate.remember_success("mirror", SimpleNamespace(object_id="mirror_asset"))
        second = gate.plan(["silver framed rectangular mirror"], ["mirror_v2"])

        self.assertEqual(first.allowed_indices, [0])
        self.assertFalse(second.allowed_indices)
        self.assertEqual(second.cached_assets[0].object_id, "mirror_asset")
        self.assertEqual(gate.request_count, 1)

    def test_failed_family_stops_stylistic_retries(self) -> None:
        gate = AssetRuntimeGate()
        gate.configure(
            stage="wall_mounted",
            budget={"max_semantic_retries_per_family": 1},
            required_objects=[],
        )

        first = gate.plan(["round wall mirror"], ["mirror"])
        second = gate.plan(["rectangular silver mirror"], ["mirror_v2"])

        self.assertEqual(first.allowed_indices, [0])
        self.assertFalse(second.allowed_indices)
        self.assertIn("exhausted", second.failures[0].reason)


if __name__ == "__main__":
    unittest.main()
