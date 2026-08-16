import json

from pathlib import Path

from PIL import Image

from scenesmith.agent_utils.furniture_image_layout import (
    ANNOTATED_ARTIFACT_NAME,
    LAYOUT_ARTIFACT_NAME,
    RAW_ARTIFACT_NAME,
    build_grounded_furniture_layout_reference,
    format_layout_contract,
    load_grounding_vocabulary,
    merge_grounding_regions,
    normalize_layout_response,
    write_grounding_annotation,
)

VOCABULARY_PATH = (
    Path(__file__).parents[2]
    / "scenesmith/agent_utils/data/hssd_furniture_grounding_vocabulary.yaml"
)


def _regions():
    return [
        {
            "detection_id": "F001",
            "box_xyxy": [10.0, 10.0, 50.0, 50.0],
            "candidate_labels": [{"label": "bed", "score": 0.9}],
            "top_score": 0.9,
        },
        {
            "detection_id": "F002",
            "box_xyxy": [60.0, 10.0, 90.0, 40.0],
            "candidate_labels": [{"label": "nightstand", "score": 0.8}],
            "top_score": 0.8,
        },
    ]


def test_versioned_vocabulary_has_expected_inventory_and_exclusions():
    vocabulary = load_grounding_vocabulary(VOCABULARY_PATH)
    phrases = vocabulary["flattened_phrases"]

    assert len(phrases) == 115
    assert len(set(phrases)) == 115
    assert {"bed", "chair", "nightstand", "rug"}.issubset(phrases)
    assert not {"door", "window", "wall", "floor", "ceiling"}.intersection(phrases)
    excluded = json.dumps(vocabulary["excluded_groups"]).lower()
    assert all(name in excluded for name in ("door", "window", "wall art"))


def test_region_merge_clips_filters_and_assigns_deterministic_ids():
    detections = [
        {"phrase": "chair", "score": 0.8, "box_xyxy": [60, 10, 100, 50]},
        {"phrase": "armchair", "score": 0.9, "box_xyxy": [61, 11, 99, 49]},
        {"phrase": "bed", "score": 0.7, "box_xyxy": [-5, 60, 50, 110]},
        {"phrase": "bad", "score": 1.0, "box_xyxy": [10, 10, 10, 20]},
        {"phrase": "nan", "score": 0.5, "box_xyxy": [float("nan"), 0, 1, 1]},
    ]

    first = merge_grounding_regions(
        detections,
        image_width=100,
        image_height=100,
        same_region_iou_threshold=0.8,
    )
    second = merge_grounding_regions(
        reversed(detections),
        image_width=100,
        image_height=100,
        same_region_iou_threshold=0.8,
    )

    assert first == second
    assert [region["detection_id"] for region in first] == ["F001", "F002"]
    assert [item["label"] for item in first[0]["candidate_labels"]] == [
        "armchair",
        "chair",
    ]
    assert first[1]["box_xyxy"] == [0.0, 60.0, 50.0, 100.0]


def test_annotation_preserves_dimensions_and_source(tmp_path):
    source_path = tmp_path / "source.png"
    output_path = tmp_path / "annotated.png"
    Image.new("RGB", (120, 80), "white").save(source_path)
    before = source_path.read_bytes()

    write_grounding_annotation(source_path, _regions(), output_path)

    assert Image.open(output_path).size == (120, 80)
    assert source_path.read_bytes() == before
    assert output_path.read_bytes() != before


def test_normalization_binds_ids_reorders_and_rejects_coordinates():
    payload = {
        "items": [
            {
                "detection_id": "F002",
                "furniture_name": "nightstand",
                "placement_order": 8,
                "role": "dependent",
                "semantic_location": "x=0.4 beside the bed",
                "facing": "toward image top",
                "relative_relations": [
                    {"target_id": "F001", "relation": "0.5 meters from the bed"},
                    {"target_id": "F001", "relation": "flanks the bed"},
                ],
                "notes": "pixel 42 confirms it",
            },
            {
                "detection_id": "F001",
                "furniture_name": "bed",
                "placement_order": 8,
                "role": "primary_anchor",
                "semantic_location": "centered against a clear wall",
                "facing": "faces into the room",
                "relative_relations": [],
                "notes": "main anchor in foreground",
            },
            {"detection_id": "F001", "furniture_name": "duplicate bed"},
            {"detection_id": "F999", "furniture_name": "invented sofa"},
            {"detection_id": "F002", "furniture_name": "wall art"},
        ],
        "ignored_detection_ids": ["F999", "F002", "F002"],
        "unboxed_visible_furniture": ["chair", "door", "table at 25%"],
        "coverage_notes": ["No other furniture is clearly visible"],
    }

    normalized = normalize_layout_response(payload, _regions(), max_items=20)

    assert [item["detection_id"] for item in normalized["items"]] == [
        "F001",
        "F002",
    ]
    assert [item["placement_order"] for item in normalized["items"]] == [1, 2]
    nightstand = normalized["items"][1]
    assert nightstand["semantic_location"] == "no reliable relation"
    assert nightstand["notes"] == ""
    assert nightstand["relative_relations"] == [
        {"target_id": "F001", "relation": "flanks the bed"}
    ]
    assert normalized["ignored_detection_ids"] == []
    assert normalized["unboxed_visible_furniture"] == ["chair"]

    contract = format_layout_contract(normalized)
    assert "box_xyxy" not in contract
    assert "0.5" not in contract
    assert nightstand["facing"] == "no reliable relation"
    assert "pixel" not in contract.lower()
    assert normalized["items"][0]["notes"] == ""


def test_floor_standing_names_are_not_filtered_as_architecture():
    payload = {
        "items": [
            {
                "detection_id": "F001",
                "furniture_name": "floor lamp",
                "placement_order": 1,
                "role": "secondary",
                "semantic_location": "beside the seating group",
                "facing": "no reliable relation",
                "relative_relations": [],
                "notes": "",
            }
        ]
    }

    normalized = normalize_layout_response(payload, _regions(), max_items=20)

    assert normalized["items"][0]["furniture_name"] == "floor lamp"


def test_legacy_max_items_no_longer_truncates_valid_detections():
    regions = []
    items = []
    for index in range(25):
        detection_id = f"F{index + 1:03d}"
        regions.append(
            {
                "detection_id": detection_id,
                "box_xyxy": [index, 0, index + 1, 1],
                "candidate_labels": [{"label": "chair", "score": 0.8}],
                "top_score": 0.8,
            }
        )
        items.append(
            {
                "detection_id": detection_id,
                "furniture_name": "chair",
                "placement_order": index + 1,
            }
        )

    normalized = normalize_layout_response({"items": items}, regions, max_items=20)

    assert len(normalized["items"]) == 25


def test_quality_mode_redacts_unreliable_directional_fields():
    payload = {
        "items": [
            {
                "detection_id": "F001",
                "furniture_name": "bed",
                "approximate_position": "centered against the rear wall",
                "wall_relation": "headboard on rear wall",
                "facing_relation": "toward room center",
                "nearby_landmarks": ["window on its right"],
                "relative_relations": [
                    {"target_id": "F002", "relation": "beside nightstand"}
                ],
            }
        ]
    }

    relations = normalize_layout_response(
        payload, _regions(), quality_mode="relations_only"
    )["items"][0]
    assert relations["approximate_position"] == "no reliable relation"
    assert relations["wall_relation"] == ""
    assert relations["nearby_landmarks"] == []
    assert relations["relative_relations"]

    inventory = normalize_layout_response(
        payload, _regions(), quality_mode="inventory_only"
    )["items"][0]
    assert inventory["facing_relation"] == "no reliable relation"
    assert inventory["relative_relations"] == []


class _FakeGroundingClient:
    def __init__(self):
        self.ground_calls = 0
        self.health_calls = 0

    def health(self):
        self.health_calls += 1
        return {"ready": True, "model": "fake-grounding-dino", "device": "cuda:0"}

    def ground_image(self, image_path, categories):
        self.ground_calls += 1
        assert Path(image_path).is_file()
        return {
            "model": "fake-grounding-dino",
            "image_width": 100,
            "image_height": 80,
            "detections": [
                {"phrase": "bed", "score": 0.9, "box_xyxy": [10, 10, 70, 70]},
                {
                    "phrase": "nightstand",
                    "score": 0.8,
                    "box_xyxy": [72, 15, 95, 50],
                },
            ],
            "batches": [{"categories": list(categories), "token_count": 100}],
        }


class _FakeVLM:
    def __init__(self):
        self.calls = []

    def create_completion(self, **kwargs):
        self.calls.append(kwargs)
        return json.dumps(
            {
                "items": [
                    {
                        "detection_id": "F001",
                        "furniture_name": "bed",
                        "placement_order": 1,
                        "role": "primary_anchor",
                        "semantic_location": "centered against a clear wall",
                        "facing": "faces into the room",
                        "relative_relations": [],
                        "notes": "main anchor",
                    },
                    {
                        "detection_id": "F002",
                        "furniture_name": "nightstand",
                        "placement_order": 2,
                        "role": "dependent",
                        "semantic_location": "flanks the bed at its head side",
                        "facing": "no reliable relation",
                        "relative_relations": [
                            {"target_id": "F001", "relation": "flanks the bed"}
                        ],
                        "notes": "paired support",
                    },
                ],
                "ignored_detection_ids": [],
                "unboxed_visible_furniture": [],
                "coverage_notes": [],
            }
        )


def test_end_to_end_orchestration_writes_audit_and_uses_cache(tmp_path):
    image_path = tmp_path / "context_edited.png"
    Image.new("RGB", (100, 80), "white").save(image_path)
    client = _FakeGroundingClient()
    vlm = _FakeVLM()
    cfg = {
        "enabled": True,
        "cache": True,
        "max_items": 20,
        "vocabulary_path": str(VOCABULARY_PATH),
        "max_coverage_regrounds": 0,
    }

    first = build_grounded_furniture_layout_reference(
        image_path=image_path,
        scene_prompt="a calm bedroom",
        cfg=cfg,
        vlm_service=vlm,
        model="test-vlm",
        client=client,
    )
    second = build_grounded_furniture_layout_reference(
        image_path=image_path,
        scene_prompt="a calm bedroom",
        cfg=cfg,
        vlm_service=vlm,
        model="test-vlm",
        client=client,
    )
    changed_prompt = build_grounded_furniture_layout_reference(
        image_path=image_path,
        scene_prompt="a formal bedroom",
        cfg=cfg,
        vlm_service=vlm,
        model="test-vlm",
        client=client,
    )

    assert first == second
    assert first == changed_prompt
    assert "Image-Grounded Furniture Layout Contract" in first
    assert client.ground_calls == 2
    assert len(vlm.calls) == 2
    request = vlm.calls[0]
    assert request["vision_detail"] == "high"
    assert request["response_format"] == {"type": "json_object"}
    assert len(request["messages"][0]["content"]) == 5
    for artifact_name in (
        RAW_ARTIFACT_NAME,
        ANNOTATED_ARTIFACT_NAME,
        LAYOUT_ARTIFACT_NAME,
    ):
        assert (tmp_path / artifact_name).is_file()


def test_disabled_mode_does_not_touch_client_or_vlm(tmp_path):
    image_path = tmp_path / "context_edited.png"
    Image.new("RGB", (10, 10), "white").save(image_path)

    class ExplodingClient:
        def health(self):
            raise AssertionError("disabled mode must not contact sidecar")

    assert (
        build_grounded_furniture_layout_reference(
            image_path=image_path,
            scene_prompt="room",
            cfg={"enabled": False},
            vlm_service=_FakeVLM(),
            model="test-vlm",
            client=ExplodingClient(),
        )
        == ""
    )
