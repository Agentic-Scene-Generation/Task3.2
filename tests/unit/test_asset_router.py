"""Unit tests for the asset router module."""

import base64
import tempfile
import unittest

from pathlib import Path
from unittest.mock import MagicMock, patch

from scenesmith.agent_utils.asset_router import AssetRouter
from scenesmith.agent_utils.asset_router.dataclasses import AnalysisResult, AssetItem
from scenesmith.agent_utils.asset_router.rendered_asset_choice import (
    choose_hssd_candidate_from_iso_renders,
    infer_floor_covering_footprint_shape,
)
from scenesmith.agent_utils.hssd_retrieval_server.dataclasses import HssdRetrievalResult
from scenesmith.agent_utils.room import AgentType, ObjectType


class TestAnalysisResultWasModified(unittest.TestCase):
    """Test the was_modified computed property logic."""

    def test_single_item_not_modified(self) -> None:
        """Single item with no original_description is not modified."""
        item = AssetItem(
            description="wooden ladder",
            short_name="ladder",
            dimensions=[0.5, 0.3, 2.0],
            object_type=ObjectType.FURNITURE,
            strategies=["generated"],
        )
        result = AnalysisResult(
            items=[item],
            original_description=None,
            discarded_manipulands=None,
        )
        assert not result.was_modified

    def test_with_original_description_is_modified(self) -> None:
        """Items with original_description set is modified (was split/filtered)."""
        items = [
            AssetItem(
                description="dining table",
                short_name="dining_table",
                dimensions=[1.5, 0.9, 0.75],
                object_type=ObjectType.FURNITURE,
                strategies=["generated"],
            ),
        ]
        result = AnalysisResult(
            items=items,
            original_description="dining table and four chairs",
            discarded_manipulands=None,
        )
        assert result.was_modified

    def test_with_discarded_manipulands_is_modified(self) -> None:
        """Request with discarded manipulands is modified."""
        item = AssetItem(
            description="ladder",
            short_name="ladder",
            dimensions=[0.5, 0.3, 2.0],
            object_type=ObjectType.FURNITURE,
            strategies=["generated"],
        )
        result = AnalysisResult(
            items=[item],
            original_description="ladder with flower pots",
            discarded_manipulands=["flower pots"],
        )
        assert result.was_modified


class TestAssetRouterItemTypeValidation(unittest.TestCase):
    """Test validate_item_types method behavior."""

    def test_furniture_items_valid_for_furniture_agent(self) -> None:
        """Furniture items are valid for furniture agent."""
        router = AssetRouter(
            agent_type=AgentType.FURNITURE, vlm_service=MagicMock(), cfg=MagicMock()
        )

        items = [
            AssetItem(
                description="desk",
                short_name="desk",
                dimensions=[1.2, 0.6, 0.75],
                object_type=ObjectType.FURNITURE,
                strategies=["generated"],
            ),
        ]

        error = router.validate_item_types(items)
        assert error is None

    def test_manipuland_items_valid_for_manipuland_agent(self) -> None:
        """Manipuland items are valid for manipuland agent."""
        router = AssetRouter(
            agent_type=AgentType.MANIPULAND, vlm_service=MagicMock(), cfg=MagicMock()
        )

        items = [
            AssetItem(
                description="coffee mug",
                short_name="mug",
                dimensions=[0.08, 0.08, 0.1],
                object_type=ObjectType.MANIPULAND,
                strategies=["generated"],
            ),
        ]

        error = router.validate_item_types(items)
        assert error is None

    def test_either_type_valid_for_both_agents(self) -> None:
        """EITHER type items are valid for both furniture and manipuland agents."""
        item = AssetItem(
            description="potted plant",
            short_name="potted_plant",
            dimensions=[0.3, 0.3, 0.6],
            object_type=ObjectType.EITHER,
            strategies=["generated"],
        )

        furniture_router = AssetRouter(
            agent_type=AgentType.FURNITURE, vlm_service=MagicMock(), cfg=MagicMock()
        )
        assert furniture_router.validate_item_types([item]) is None

        manipuland_router = AssetRouter(
            agent_type=AgentType.MANIPULAND, vlm_service=MagicMock(), cfg=MagicMock()
        )
        assert manipuland_router.validate_item_types([item]) is None

    def test_wrong_type_returns_error(self) -> None:
        """Wrong item type for agent returns error message."""
        router = AssetRouter(
            agent_type=AgentType.FURNITURE, vlm_service=MagicMock(), cfg=MagicMock()
        )

        items = [
            AssetItem(
                description="coffee mug",
                short_name="mug",
                dimensions=[0.08, 0.08, 0.1],
                object_type=ObjectType.MANIPULAND,
                strategies=["generated"],
            ),
        ]

        error = router.validate_item_types(items)
        assert error is not None
        assert "manipuland" in error.lower()


class TestRenderedHssdAssetChoice(unittest.TestCase):
    """Test VLM-assisted selection among rendered HSSD candidates."""

    _PNG_1X1 = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/"
        "x8AAwMCAO+/p9sAAAAASUVORK5CYII="
    )

    def _candidate(
        self,
        hssd_id: str,
        name: str,
        score: float,
        size: tuple[float, float, float] = (1.0, 0.5, 0.6),
    ) -> HssdRetrievalResult:
        return HssdRetrievalResult(
            mesh_path=f"/tmp/{hssd_id}.glb",
            hssd_id=hssd_id,
            object_name=name,
            similarity_score=score,
            size=size,
            category="bedroom",
        )

    def _write_iso(self, root: Path, hssd_id: str) -> None:
        asset_dir = root / hssd_id
        asset_dir.mkdir(parents=True, exist_ok=True)
        (asset_dir / "iso.png").write_bytes(self._PNG_1X1)

    def _write_view(self, root: Path, hssd_id: str, view_name: str) -> None:
        asset_dir = root / hssd_id
        asset_dir.mkdir(parents=True, exist_ok=True)
        (asset_dir / f"{view_name}.png").write_bytes(self._PNG_1X1)

    def test_reorders_candidates_when_vlm_selects_rendered_iso(self) -> None:
        candidates = [
            self._candidate("asset_a", "generic bed", 0.91),
            self._candidate("asset_b", "wood nightstand", 0.89),
            self._candidate("asset_c", "small table", 0.87),
        ]
        vlm_service = MagicMock()
        vlm_service.create_completion.return_value = (
            '{"selected_index": 2, "selected_hssd_id": "asset_b", '
            '"reason": "closest bedside table"}'
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            for candidate in candidates:
                self._write_iso(root, candidate.hssd_id)

            choice = choose_hssd_candidate_from_iso_renders(
                candidates=candidates,
                object_description="wooden nightstand beside a bed",
                scene_context=(
                    "A bedroom with a bed centered on the main wall and a "
                    "nightstand with a table lamp on each side of the bed."
                ),
                vlm_service=vlm_service,
                model="test-model",
                reasoning_effort="low",
                verbosity="low",
                vision_detail="low",
                rendered_assets_dir=root,
                top_n=3,
            )

        self.assertEqual(
            [candidate.hssd_id for candidate in choice.candidates],
            ["asset_b", "asset_a", "asset_c"],
        )
        self.assertEqual(choice.selected_hssd_id, "asset_b")
        self.assertEqual(choice.selected_index, 2)
        self.assertEqual(choice.used_image_count, 3)
        vlm_service.create_completion.assert_called_once()
        prompt = vlm_service.create_completion.call_args.kwargs["messages"][0][
            "content"
        ][0]["text"]
        self.assertIn("Original scene prompt", prompt)
        self.assertIn("nightstand with a table lamp", prompt)

    @patch(
        "scenesmith.scenebenchmark_critic.asset_library_annotations."
        "get_hssd_asset_annotations"
    )
    def test_adds_verified_scenebenchmark_front_view(self, mock_annotations) -> None:
        candidates = [
            self._candidate("a" * 40, "wardrobe", 0.91),
            self._candidate("b" * 40, "wardrobe", 0.89),
        ]
        mock_annotations.return_value = {
            "canonical_front": {
                "asset_local_front_axis": "+X",
                "canonical_orientation_is_semantic_front": True,
                "is_strict_front": True,
            }
        }
        vlm_service = MagicMock()
        vlm_service.create_completion.return_value = (
            '{"selected_index": 1, "selected_hssd_id": "'
            + "a" * 40
            + '", "reason": "visible doors"}'
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            for candidate in candidates:
                self._write_iso(root, candidate.hssd_id)
                self._write_view(root, candidate.hssd_id, "right")
            choice = choose_hssd_candidate_from_iso_renders(
                candidates=candidates,
                object_description="modern double-door wardrobe",
                scene_context="A bedroom with storage furniture.",
                vlm_service=vlm_service,
                model="test-model",
                reasoning_effort="none",
                verbosity="low",
                vision_detail="low",
                rendered_assets_dir=root,
                top_n=2,
            )

        content = vlm_service.create_completion.call_args.kwargs["messages"][0][
            "content"
        ]
        labels = [item["text"] for item in content if item["type"] == "text"]
        evidence_labels = [
            label for label in labels if label.startswith("CANDIDATE_INDEX=")
        ]
        self.assertEqual(choice.used_image_count, 4)
        self.assertTrue(
            any(
                "SceneBenchmark semantic-front +X (right)" in label
                for label in evidence_labels
            )
        )
        self.assertFalse(any("fallback-axis" in label for label in evidence_labels))

    @patch(
        "scenesmith.scenebenchmark_critic.asset_library_annotations."
        "get_hssd_asset_annotations"
    )
    def test_adds_both_sides_of_strict_fallback_axis(self, mock_annotations) -> None:
        candidates = [
            self._candidate("c" * 40, "wardrobe", 0.91),
            self._candidate("d" * 40, "wardrobe", 0.89),
        ]
        mock_annotations.return_value = {
            "canonical_front": {
                "asset_local_front_axis": [0.0, 0.0, 1.0],
                "canonical_orientation_is_semantic_front": False,
                "is_strict_front": True,
            }
        }
        vlm_service = MagicMock()
        vlm_service.create_completion.return_value = (
            '{"selected_index": 1, "selected_hssd_id": "'
            + "c" * 40
            + '", "reason": "doors visible on fallback-axis view"}'
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            for candidate in candidates:
                self._write_iso(root, candidate.hssd_id)
                self._write_view(root, candidate.hssd_id, "back")
                self._write_view(root, candidate.hssd_id, "front")
            choice = choose_hssd_candidate_from_iso_renders(
                candidates=candidates,
                object_description="modern double-door wardrobe",
                scene_context="A bedroom with storage furniture.",
                vlm_service=vlm_service,
                model="test-model",
                reasoning_effort="none",
                verbosity="low",
                vision_detail="low",
                rendered_assets_dir=root,
                top_n=2,
            )

        content = vlm_service.create_completion.call_args.kwargs["messages"][0][
            "content"
        ]
        labels = [item["text"] for item in content if item["type"] == "text"]
        self.assertEqual(choice.used_image_count, 6)
        self.assertTrue(any("fallback-axis -Y (back)" in label for label in labels))
        self.assertTrue(any("fallback-axis +Y (front)" in label for label in labels))

    def test_keeps_retrieval_order_when_too_few_iso_images_exist(self) -> None:
        candidates = [
            self._candidate("asset_a", "generic bed", 0.91),
            self._candidate("asset_b", "wood nightstand", 0.89),
        ]
        vlm_service = MagicMock()

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self._write_iso(root, "asset_a")

            choice = choose_hssd_candidate_from_iso_renders(
                candidates=candidates,
                object_description="wooden nightstand beside a bed",
                scene_context="A bedroom with matching bedside furniture.",
                vlm_service=vlm_service,
                model="test-model",
                reasoning_effort="low",
                verbosity="low",
                vision_detail="low",
                rendered_assets_dir=root,
                top_n=2,
            )

        self.assertEqual(choice.candidates, candidates)
        self.assertIsNone(choice.selected_hssd_id)
        self.assertEqual(choice.used_image_count, 1)
        vlm_service.create_completion.assert_not_called()

    def test_floor_covering_prompt_keeps_planar_shape_and_thickness_semantics(
        self,
    ) -> None:
        candidates = [
            self._candidate("asset_a", "rectangular rug", 0.91),
            self._candidate("asset_b", "round carpet", 0.89),
        ]
        vlm_service = MagicMock()
        vlm_service.create_completion.return_value = (
            '{"selected_index": 1, "selected_hssd_id": "asset_a", '
            '"reason": "rectangular aspect ratio"}'
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            for candidate in candidates:
                self._write_iso(root, candidate.hssd_id)
            choose_hssd_candidate_from_iso_renders(
                candidates=candidates,
                object_description="low-pile patterned area rug",
                object_short_name="area_rug",
                requested_dimensions=[2.4, 1.6, 0.02],
                requested_shape="rectangular",
                scene_context="A living room with a rug under the coffee table.",
                vlm_service=vlm_service,
                model="test-model",
                reasoning_effort="low",
                verbosity="low",
                vision_detail="low",
                rendered_assets_dir=root,
                top_n=2,
            )

        prompt = vlm_service.create_completion.call_args.kwargs["messages"][0][
            "content"
        ][0]["text"]
        self.assertIn("(2.4, 1.6, 0.02)", prompt)
        self.assertIn("Requested footprint shape: rectangular", prompt)
        self.assertIn("near-zero visual thickness is expected", prompt)

    def test_square_floor_covering_rejects_incompatible_vlm_choice(self) -> None:
        candidates = [
            self._candidate(
                "rectangular_rug", "patterned rug", 0.91, size=(1.6, 2.5, 0.01)
            ),
            self._candidate("square_rug", "woven rug", 0.89, size=(2.0, 2.2, 0.01)),
            self._candidate("runner", "rug runner", 0.87, size=(0.8, 2.4, 0.01)),
        ]
        vlm_service = MagicMock()
        vlm_service.create_completion.return_value = (
            '{"selected_index": 1, "selected_hssd_id": "rectangular_rug", '
            '"reason": "best visible pattern"}'
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            for candidate in candidates:
                self._write_iso(root, candidate.hssd_id)
            choice = choose_hssd_candidate_from_iso_renders(
                candidates=candidates,
                object_description="geometric square rug",
                object_short_name="area_rug",
                requested_dimensions=[2.0, 2.0, 0.02],
                requested_shape="square",
                scene_context="A living room with a square rug.",
                vlm_service=vlm_service,
                model="test-model",
                reasoning_effort="low",
                verbosity="low",
                vision_detail="low",
                rendered_assets_dir=root,
                top_n=3,
            )

        self.assertEqual(choice.selected_hssd_id, "square_rug")
        self.assertEqual(choice.selected_index, 2)
        self.assertEqual(choice.candidates[0].hssd_id, "square_rug")
        self.assertIn("square footprint guard", choice.reason)
        vlm_service.create_completion.assert_called_once()

    def test_infers_explicit_square_floor_covering_shape(self) -> None:
        self.assertEqual(
            infer_floor_covering_footprint_shape(
                "Geometric patterned square rug", "area_rug"
            ),
            "square",
        )
        self.assertEqual(
            infer_floor_covering_footprint_shape("Round bath mat"), "circular"
        )


class TestAnalysisResponseParsing(unittest.TestCase):
    """Test parsing of VLM analysis responses."""

    def test_parse_single_furniture_item(self) -> None:
        """Parse single furniture item response."""
        router = AssetRouter(
            agent_type=AgentType.FURNITURE, vlm_service=MagicMock(), cfg=MagicMock()
        )

        response = {
            "items": [
                {
                    "description": "wooden ladder",
                    "short_name": "ladder",
                    "dimensions": [0.5, 0.3, 2.0],
                    "object_type": "FURNITURE",
                    "strategies": ["generated"],
                }
            ],
            "original_description": None,
            "discarded_manipulands": None,
        }

        result = router._parse_analysis_response(response)
        assert len(result.items) == 1
        assert result.items[0].description == "wooden ladder"
        assert result.items[0].object_type == ObjectType.FURNITURE
        assert not result.was_modified

    def test_parse_composite_split(self) -> None:
        """Parse response with composite split into multiple items."""
        router = AssetRouter(
            agent_type=AgentType.MANIPULAND, vlm_service=MagicMock(), cfg=MagicMock()
        )

        response = {
            "items": [
                {
                    "description": "fruit bowl",
                    "short_name": "fruit_bowl",
                    "dimensions": [0.3, 0.3, 0.10],
                    "object_type": "MANIPULAND",
                    "strategies": ["generated"],
                },
                {
                    "description": "apple",
                    "short_name": "apple",
                    "dimensions": [0.08, 0.08, 0.08],
                    "object_type": "MANIPULAND",
                    "strategies": ["generated"],
                },
            ],
            "original_description": "fruit bowl with apples",
        }

        result = router._parse_analysis_response(response)
        assert len(result.items) == 2
        assert result.was_modified
        assert result.original_description == "fruit bowl with apples"

    def test_parse_error_response(self) -> None:
        """Parse error response from VLM."""
        router = AssetRouter(
            agent_type=AgentType.FURNITURE, vlm_service=MagicMock(), cfg=MagicMock()
        )

        response = {
            "items": [],
            "original_description": None,
            "discarded_manipulands": None,
            "error": "Request is for a manipuland (coffee mug), not furniture.",
        }

        result = router._parse_analysis_response(response)
        assert len(result.items) == 0
        assert result.error is not None
        assert "manipuland" in result.error.lower()

    def test_parse_error_response_preserves_original_description(self) -> None:
        """Error responses preserve original_description for debugging."""
        router = AssetRouter(
            agent_type=AgentType.FURNITURE, vlm_service=MagicMock(), cfg=MagicMock()
        )

        response = {
            "items": [],
            "original_description": "stack of 4 car tires",
            "discarded_manipulands": None,
            "error": "Stackable items should be handled by manipuland agent.",
        }

        result = router._parse_analysis_response(response)
        assert len(result.items) == 0
        assert result.error is not None
        assert result.original_description == "stack of 4 car tires"

    def test_parse_with_discarded_manipulands(self) -> None:
        """Parse response with discarded manipulands (furniture agent filtering)."""
        router = AssetRouter(
            agent_type=AgentType.FURNITURE, vlm_service=MagicMock(), cfg=MagicMock()
        )

        response = {
            "items": [
                {
                    "description": "bookshelf",
                    "short_name": "bookshelf",
                    "dimensions": [1.0, 0.3, 2.0],
                    "object_type": "FURNITURE",
                    "strategies": ["generated"],
                }
            ],
            "original_description": "bookshelf with books and decorations",
            "discarded_manipulands": ["books", "decorations"],
        }

        result = router._parse_analysis_response(response)
        assert len(result.items) == 1
        assert result.was_modified
        assert result.discarded_manipulands == ["books", "decorations"]

    def test_parse_lowercase_object_type(self) -> None:
        """Object type parsing is case-insensitive."""
        router = AssetRouter(
            agent_type=AgentType.FURNITURE, vlm_service=MagicMock(), cfg=MagicMock()
        )

        response = {
            "items": [
                {
                    "description": "desk",
                    "short_name": "desk",
                    "dimensions": [1.2, 0.6, 0.75],
                    "object_type": "furniture",  # lowercase
                    "strategies": ["generated"],
                }
            ],
            "original_description": None,
        }

        result = router._parse_analysis_response(response)
        assert len(result.items) == 1
        assert result.items[0].object_type == ObjectType.FURNITURE


if __name__ == "__main__":
    unittest.main()
