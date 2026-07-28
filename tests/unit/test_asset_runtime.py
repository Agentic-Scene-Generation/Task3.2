import unittest

from types import SimpleNamespace

from scenesmith.agent_utils.asset_runtime import (
    AssetRuntimeGate,
    semantic_asset_family,
)


class AssetRuntimeGateTest(unittest.TestCase):
    def test_floor_covering_variants_share_the_rug_family(self) -> None:
        for description in ("square rug", "hallway runner", "woven floor mat"):
            with self.subTest(description=description):
                self.assertEqual(semantic_asset_family(description), "rug")

    def test_semantic_variants_share_one_family(self) -> None:
        self.assertEqual(
            semantic_asset_family("metal-framed circular mirror"),
            semantic_asset_family("rectangular silver wall mirror"),
        )
        self.assertEqual(semantic_asset_family("卧室床头柜"), "nightstand")

    def test_classroom_desk_roles_do_not_share_cache_family(self) -> None:
        self.assertEqual(
            semantic_asset_family("individual student desk"),
            "student_desk",
        )
        self.assertEqual(
            semantic_asset_family("teacher's desk with drawers"),
            "teacher_desk",
        )
        self.assertEqual(semantic_asset_family("writing desk"), "desk")

    def test_required_family_matching_accepts_plural_object_names(self) -> None:
        self.assertEqual(semantic_asset_family("two nightstands"), "nightstand")
        self.assertEqual(semantic_asset_family("built-in wardrobes"), "wardrobe")

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

    def test_invalidated_required_family_can_acquire_a_new_asset(self) -> None:
        gate = AssetRuntimeGate()
        gate.configure(
            stage="furniture",
            budget={"max_semantic_retries_per_family": 2},
            required_objects=["bed"],
        )
        first = gate.plan(["double bed"], ["bed"])
        rejected = SimpleNamespace(object_id="bad_bed", metadata={})
        gate.remember_success("bed", rejected)

        removed = gate.invalidate_family("bed")
        second = gate.plan(["double bed"], ["bed"])

        self.assertEqual([0], first.allowed_indices)
        self.assertEqual(1, removed)
        self.assertEqual([0], second.allowed_indices)
        self.assertFalse(second.cached_assets)

    def test_admission_failed_asset_is_never_reused(self) -> None:
        gate = AssetRuntimeGate()
        gate.configure(
            stage="furniture",
            budget={"max_semantic_retries_per_family": 2},
            required_objects=["bed"],
        )
        rejected = SimpleNamespace(
            object_id="bad_bed",
            metadata={"asset_admission_failed": True},
        )

        gate.remember_success("bed", rejected)
        plan = gate.plan(["double bed"], ["bed"])

        self.assertEqual([0], plan.allowed_indices)
        self.assertFalse(plan.cached_assets)

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

    def test_stage_regeneration_keeps_only_admitted_real_assets(self) -> None:
        gate = AssetRuntimeGate()
        budget = {"max_semantic_retries_per_family": 1}
        gate.configure(stage="furniture", budget=budget, required_objects=["bed"])
        real_asset = SimpleNamespace(object_id="bed_asset", metadata={})
        placeholder = SimpleNamespace(
            object_id="fake_bed",
            metadata={"repair_placeholder": True},
        )
        gate.remember_success("bed", real_asset)
        gate.remember_success("bed", placeholder)

        gate.configure(stage="furniture", budget=budget, required_objects=["bed"])
        plan = gate.plan(["double bed"], ["bed"])

        self.assertFalse(plan.allowed_indices)
        self.assertEqual(
            [asset.object_id for asset in plan.cached_assets],
            ["bed_asset"],
        )


if __name__ == "__main__":
    unittest.main()
