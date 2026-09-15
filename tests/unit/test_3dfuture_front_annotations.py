from types import SimpleNamespace

from scenesmith.agent_utils.asset_manager import (
    _get_scenebenchmark_front_axis_annotation_record,
    _normalize_hssd_annotation_front_axis,
)
from scenesmith.scenebenchmark_critic.adapter import _hssd_annotation_functional_hints


ASSET_UID = "3dfuture:00033fdb-e9f8-4abe-aa9b-878761356d35"


def test_all_assets_front_lookup_loads_3dfuture_semantic_front() -> None:
    record = _get_scenebenchmark_front_axis_annotation_record(ASSET_UID)

    assert record is not None
    front = record["canonical_front"]
    assert front["canonical_orientation_is_semantic_front"] is True
    assert (
        _normalize_hssd_annotation_front_axis(front["asset_local_front_axis"]) == "-Y"
    )


def test_critic_uses_3dfuture_front_hints_from_legacy_mesh_id() -> None:
    # all-assets retrieval transports a namespaced ID in this legacy field.
    obj = SimpleNamespace(metadata={"hssd_mesh_id": ASSET_UID})

    hints = _hssd_annotation_functional_hints(obj)

    assert hints["asset_annotation_source"] == "3dfuture_annotations"
    assert hints["asset_annotation_asset_uid"] == ASSET_UID
    assert hints["canonical_orientation_is_semantic_front"] is True
    assert hints["asset_local_front_axis"] == [0.0, 0.0, 1.0]
