from scenesmith.scenebenchmark_critic.metrics.functional_dependency.extensions.wall_backed_storage import (
    evaluate_wall_backed_storage_alignment,
)


def _bbox(center, size):
    return {
        "center": [*center, size[2] / 2.0],
        "size": list(size),
        "min": [center[0] - size[0] / 2.0, center[1] - size[1] / 2.0, 0.0],
        "max": [center[0] + size[0] / 2.0, center[1] + size[1] / 2.0, size[2]],
    }


def _case_pack(storage_x: float, *, yaw_deg: float = 270.0) -> dict:
    return {
        "scene_geometry": {
            "rooms": [{"bbox": {"min": [-2.5, -2.0, 0.0], "max": [2.5, 2.0, 2.7]}}],
            "objects": [
                {
                    "id": "west_boundary",
                    "category": "wall",
                    "bbox_world": _bbox((-2.5, 0.0), (0.1, 4.0, 2.7)),
                },
                {
                    "id": "storage_piece_alpha",
                    "object_type": "furniture",
                    "category": "shelving_unit",
                    "yaw_deg": yaw_deg,
                    "bbox_world": _bbox((storage_x, 0.0), (0.8, 0.3, 1.8)),
                    "footprint_world": [
                        [storage_x - 0.4, -0.15],
                        [storage_x + 0.4, -0.15],
                        [storage_x + 0.4, 0.15],
                        [storage_x - 0.4, 0.15],
                    ],
                    "functional_hints": {"category_group": "storage"},
                },
            ],
        }
    }


def test_reports_geometry_candidates_for_unbacked_storage() -> None:
    result = evaluate_wall_backed_storage_alignment(_case_pack(-1.0))[0]

    assert result["label"] == "fail"
    assert result["primary_object"] == "storage_piece_alpha"
    candidates = result["diagnostics"]["candidate_poses"]
    assert candidates
    assert all(item["wall_id"] == "west_boundary" for item in candidates)
    assert min(item["translation_m"] for item in candidates) > 0.0


def test_storage_already_at_wall_passes() -> None:
    result = evaluate_wall_backed_storage_alignment(_case_pack(-2.05))[0]

    assert result["label"] == "pass"


def test_storage_at_wall_with_front_parallel_to_wall_fails() -> None:
    result = evaluate_wall_backed_storage_alignment(_case_pack(-2.05, yaw_deg=0.0))[0]

    assert result["label"] == "fail"
    assert result["diagnostics"]["front_error_deg"] == 90.0
