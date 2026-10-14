from copy import deepcopy

import pytest

from scenesmith.scenebenchmark_critic.metrics.functional_dependency import relations


def test_non_support_relation_never_reverses_into_support(monkeypatch) -> None:
    shelf = {"id": "bookshelf_0", "category_norm": "bookshelf"}
    book = {"id": "book_0", "category_norm": "book"}

    monkeypatch.setattr(
        relations,
        "_relation_target_is_valid",
        lambda subject, target, relation: (
            (subject["id"], target["id"], relation)
            == ("book_0", "bookshelf_0", "object_on_support")
        ),
    )
    monkeypatch.setattr(relations, "_infer_relation_type", lambda *_args: None)

    label, _confidence, reason, _diagnostics = relations._eval_relation_over_targets(
        None, shelf, [book], "furniture_faces_furniture"
    )

    assert label == "fail"
    assert "target category is not compatible" in reason


def _facing_fixture() -> tuple[dict, dict, dict]:
    def obj(name: str, category: str, y: float) -> dict:
        return {
            "id": name,
            "category_norm": category,
            "object_type": "furniture",
            "yaw_deg": 0.0,
            "bbox_world": {
                "center": [0.0, y, 0.5],
                "min": [-0.5, y - 0.5, 0.0],
                "max": [0.5, y + 0.5, 1.0],
                "size": [1.0, 1.0, 1.0],
            },
        }

    sofa = obj("sofa_0", "sofa", 0.0)
    machine = obj("machine_0", "interactive_exhibit", 4.0)
    check = {
        "subject_id": sofa["id"],
        "target_ids": [machine["id"]],
        "evidence": {
            "intent_constraint": {
                "relation": "faces",
                "source": "explicit_prompt",
                "strength": "hard",
            },
            "dependency": {"distance_required": False, "max_angle_deg": 60.0},
        },
    }
    return sofa, machine, check


@pytest.mark.parametrize("yaw, expected", [(0.0, "pass"), (180.0, "fail")])
def test_explicit_facing_to_open_vocabulary_furniture_uses_measured_geometry(
    yaw: float, expected: str
) -> None:
    sofa, machine, check = _facing_fixture()
    sofa["yaw_deg"] = yaw
    before = deepcopy((sofa, machine, check))
    label, _, reason, _ = relations._eval_relation_over_targets(
        None, sofa, [machine], "furniture_faces_furniture", check=check
    )
    assert label == expected
    assert "subject front angle" in reason
    assert (sofa, machine, check) == before


@pytest.mark.parametrize("missing", ["yaw", "subject_bbox", "target_bbox"])
def test_explicit_open_vocabulary_facing_missing_geometry_stays_unknown(
    missing: str,
) -> None:
    sofa, machine, check = _facing_fixture()
    if missing == "yaw":
        sofa.pop("yaw_deg")
    elif missing == "subject_bbox":
        sofa.pop("bbox_world")
    else:
        machine.pop("bbox_world")
    label, _, _, _ = relations._eval_relation_over_targets(
        None, sofa, [machine], "furniture_faces_furniture", check=check
    )
    assert label == "unknown"


@pytest.mark.parametrize("source", ["model_inferred", "hssd_annotations:legacy"])
def test_facing_target_extension_does_not_change_non_prompt_priors(source: str) -> None:
    sofa, machine, check = _facing_fixture()
    check["evidence"]["intent_constraint"]["source"] = source
    label, _, reason, _ = relations._eval_relation_over_targets(
        None, sofa, [machine], "furniture_faces_furniture", check=check
    )
    assert label == "fail"
    assert "target category is not compatible" in reason


@pytest.mark.parametrize("invalid", ["small_target", "architecture", "unbound", "soft"])
def test_explicit_facing_target_extension_preserves_endpoint_and_authority_gates(
    invalid: str,
) -> None:
    sofa, machine, check = _facing_fixture()
    if invalid == "small_target":
        machine.update(category_norm="book", object_type="manipuland")
    elif invalid == "architecture":
        machine.update(category_norm="wall", object_type="wall_mounted")
    elif invalid == "unbound":
        check["target_ids"] = ["different_machine"]
    else:
        check["evidence"]["intent_constraint"]["strength"] = "soft"
    assert not relations._explicit_facing_has_furniture_endpoints(check, sofa, machine)
