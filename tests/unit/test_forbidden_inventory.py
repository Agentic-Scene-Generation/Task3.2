from scenesmith.scenebenchmark_critic.evaluator import evaluate_case_pack
from scenesmith.scenebenchmark_critic.intent_contract import build_intent_contract
from scenesmith.scenebenchmark_critic.intent_contract import (
    semantic_selector_matches,
    selected_ids,
)


def _evaluate(prompt, objects):
    contract = build_intent_contract(prompt)
    return evaluate_case_pack(
        {
            "stage": "final",
            "scene_geometry": {"objects": objects},
            "intent_contract": contract,
            "checks": [],
        },
        config={
            "scenebenchmark_critic": {
                "enabled": True,
                "metrics": ["functional_dependency"],
            }
        },
    )


def test_explicit_forbidden_wardrobe_fails_when_present() -> None:
    result = _evaluate(
        "A bedroom with no wardrobe.",
        [{"id": "wardrobe_0", "category_norm": "wardrobe", "object_type": "furniture"}],
    )

    failure = next(
        row
        for row in result["results"]
        if row["relation_type"] == "forbidden_inventory"
    )
    assert failure["label"] == "fail"
    assert failure["related_objects"] == ["wardrobe_0"]


def test_explicit_forbidden_coffee_table_passes_when_absent() -> None:
    result = _evaluate(
        "A living room without a coffee table.",
        [{"id": "sofa_0", "category_norm": "sofa", "object_type": "furniture"}],
    )

    verdict = next(
        row
        for row in result["results"]
        if row["relation_type"] == "forbidden_inventory"
    )
    assert verdict["label"] == "pass"


def test_declared_category_hierarchy_is_shared_by_inventory_and_forbidden_checks() -> (
    None
):
    objects = [
        {
            "id": "portrait_0",
            "category_norm": "frame_portrait",
            "object_type": "wall_mounted",
        },
        {
            "id": "record_0",
            "category_norm": "vinyl_record",
            "object_type": "manipuland",
        },
        {
            "id": "pillow_0",
            "category_norm": "throw_pillow",
            "object_type": "manipuland",
        },
        {
            "id": "frame_control",
            "category_norm": "frame_table",
            "object_type": "furniture",
        },
    ]

    assert selected_ids({"category": "picture_frame"}, objects) == ["portrait_0"]
    assert selected_ids({"category": "record"}, objects) == ["record_0"]
    assert selected_ids({"category": "pillow"}, objects) == ["pillow_0"]
    details = semantic_selector_matches({"category": "picture_frame"}, objects)
    assert details == [
        {
            "requested_category": "picture_frame",
            "observed_category": "frame_portrait",
            "matched": True,
            "match_kind": "taxonomy_parent",
            "match_path": ["frame_portrait", "picture_frame"],
            "object_id": "portrait_0",
            "context_matched": True,
        }
    ]
