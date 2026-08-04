from scenesmith.scenebenchmark_critic.metrics.functional_dependency.extensions.workstation_alignment import (
    evaluate_workstation_focal_alignment,
)


def _bbox(center: tuple[float, float], size: tuple[float, float, float]) -> dict:
    return {
        "center": [*center, size[2] / 2.0],
        "size": list(size),
        "min": [center[0] - size[0] / 2.0, center[1] - size[1] / 2.0, 0.0],
        "max": [center[0] + size[0] / 2.0, center[1] + size[1] / 2.0, size[2]],
    }


def _case_pack(*, office_x: float = 0.2) -> dict:
    return {
        "scene_geometry": {
            "objects": [
                {
                    "id": "study_desk_0",
                    "object_type": "furniture",
                    "category": "desk",
                    "yaw_deg": 180.0,
                    "bbox_world": _bbox((0.0, 1.79), (1.6, 0.8, 0.8)),
                    "support_regions": [{"region_id": "desk_top"}],
                },
                {
                    "id": "office_chair_0",
                    "object_type": "furniture",
                    "category": "office_chair",
                    "yaw_deg": 0.0,
                    "bbox_world": _bbox((office_x, 0.92), (0.6, 0.6, 0.9)),
                },
                {
                    "id": "guest_chair_0",
                    "object_type": "furniture",
                    "category": "guest_chair",
                    "yaw_deg": 90.0,
                    "bbox_world": _bbox((2.1, 0.45), (0.55, 0.66, 0.9)),
                },
                {
                    "id": "monitor_0",
                    "object_type": "manipuland",
                    "category": "monitor",
                    "yaw_deg": 180.0,
                    "bbox_world": _bbox((0.2, 1.75), (0.6, 0.2, 0.4)),
                    "placement_info": {"parent_surface_id": "desk_top"},
                },
            ]
        },
        "checks": [
            {
                "metric": "functional_dependency",
                "relation_type": "seating_to_work_surface",
                "check_source": "intent_contract",
                "subject_id": "office_chair_0",
                "target_ids": ["study_desk_0"],
            }
        ],
    }


def test_office_workseat_aligns_with_its_desk_monitor_and_not_guest_chair() -> None:
    results = evaluate_workstation_focal_alignment(_case_pack())

    assert {
        (result["relation_type"], result["primary_object"]) for result in results
    } == {
        ("workstation_focal_alignment", "office_chair_0"),
        ("display_faces_user", "monitor_0"),
    }
    assert {result["label"] for result in results} == {"pass"}


def test_workstation_alignment_rejects_off_center_office_chair() -> None:
    results = evaluate_workstation_focal_alignment(_case_pack(office_x=0.9))
    workseat_result = next(
        result
        for result in results
        if result["relation_type"] == "workstation_focal_alignment"
    )

    assert workseat_result["primary_object"] == "office_chair_0"
    assert workseat_result["label"] == "fail"
    assert workseat_result["diagnostics"]["priority"] == "lateral_alignment"
