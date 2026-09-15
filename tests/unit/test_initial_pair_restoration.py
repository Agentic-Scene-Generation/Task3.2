"""Restoration permits quaternion conversion noise, never scene edits."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest

from scenesmith.scene_expert.slow_memory.paired import digest, read_json
from scenesmith.scene_expert.slow_memory.paired_restoration import (
    save_restoration_proof,
    validate_restoration_proof,
)


def _state() -> dict:
    return {
        "objects": {
            "nightstand_0": {
                "transform": {
                    "translation": [0.0, 0.0, 0.0],
                    "rotation_wxyz": [
                        0.9996803707709186,
                        0.0,
                        0.0,
                        -0.025281540604135816,
                    ],
                },
                "support_surfaces": [],
                "bbox_max": [1.0, 1.0, 1.0],
                "metadata": {"quality": 0.1},
            }
        },
        "metadata": {"memory": "frozen"},
    }


@pytest.mark.parametrize("sign", [1, -1])
def test_008_observed_rotation_roundoff_has_explicit_proof(
    tmp_path: Path, sign: int
) -> None:
    source = _state()
    restored = copy.deepcopy(source)
    rotation = restored["objects"]["nightstand_0"]["transform"]["rotation_wxyz"]
    rotation[3] = -0.02528154060413582
    rotation[:] = [sign * v for v in rotation]
    assert source != restored
    save_restoration_proof(tmp_path, source, source, restored, [])
    proof = read_json(tmp_path / "restoration_proof.json")
    assert proof["status"] == "verified"
    assert len(proof["rotation_roundoff"]) == 1
    assert proof["source_raw_state_sha256"] != proof["restored_state_sha256"]
    assert proof["rotation_roundoff"][0]["max_component_error"] < 1e-17


@pytest.mark.parametrize(
    "change",
    ["translation", "rotation", "bbox", "metadata", "missing_object", "nonfinite"],
)
def test_real_state_changes_still_fail_with_field_diagnostics(
    tmp_path: Path, change: str
) -> None:
    source, restored = _state(), _state()
    obj = restored["objects"]["nightstand_0"]
    if change == "translation":
        obj["transform"]["translation"][0] = 1e-18
    elif change == "rotation":
        obj["transform"]["rotation_wxyz"][3] += 1e-7
    elif change == "bbox":
        obj["bbox_max"][0] += 1e-10
    elif change == "metadata":
        obj["metadata"]["quality"] += 1e-16
    elif change == "nonfinite":
        obj["transform"]["rotation_wxyz"][0] = float("nan")
    else:
        restored["objects"].clear()
    with pytest.raises((ValueError, KeyError)):
        save_restoration_proof(tmp_path, source, source, restored, [])
    proof = read_json(tmp_path / "restoration_proof.json")
    assert proof["status"] == "failed"
    assert proof["differences"]


def test_asset_bindings_cannot_disguise_metadata_changes(tmp_path: Path) -> None:
    from scenesmith.scene_expert.slow_memory.paired import write_json

    source, restored = _state(), _state()
    restored["metadata"]["memory"] = "changed"
    write_json(tmp_path / "raw_state.json", source)
    write_json(tmp_path / "restored_state.json", restored)
    forged = {
        "asset_bindings": [
            {
                "path": ["metadata", "memory"],
                "source": "frozen",
                "mapped": "changed",
                "raw_asset": "asset.sdf",
                "sha256": "fake",
            }
        ]
    }
    write_json(tmp_path / "restoration_proof.json", forged)
    with pytest.raises(ValueError, match="asset binding"):
        validate_restoration_proof(
            tmp_path,
            {"raw_files": {"asset.sdf": "fake"}},
            {"restoration_proof_sha256": digest(forged)},
        )
