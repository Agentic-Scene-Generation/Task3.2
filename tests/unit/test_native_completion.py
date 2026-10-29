"""Synthetic execution fixtures for native/final-camera completion checks."""

from __future__ import annotations

import copy
import json
import os
import struct
import subprocess
import sys
import zlib

from pathlib import Path

import pytest

from scenesmith.scene_expert.native_completion import (
    check_native_pair_completion,
    required_final_view_sources,
)


def _png(size: int = 2) -> bytes:
    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(
            ">I", zlib.crc32(kind + data)
        )

    return b"\x89PNG\r\n\x1a\n" + b"".join([
        chunk(b"IHDR", struct.pack(">IIBBBBB", size, size, 8, 6, 0, 0, 0)),
        chunk(b"IDAT", zlib.compress((b"\0" + b"\0\0\0\xff" * size) * size)),
        chunk(b"IEND", b""),
    ])


@pytest.fixture
def pair(tmp_path: Path) -> dict:
    rows = []
    for arm in ("base", "adapter"):
        output = tmp_path / arm
        scene = output / "scene_synthetic"
        blend = scene / "combined_house/house.blend"
        blend.parent.mkdir(parents=True)
        blend.write_bytes(b"synthetic checkpoint; not a real scene")
        views = scene / "critic_final_views"
        views.mkdir()
        paths = [views / "00_top.png", views / "01_side.png"]
        for path in paths:
            path.write_bytes(_png())
        (views / "render_manifest.json").write_text(json.dumps({
            "input": str(blend), "renderer": "BlenderRenderer.render_agent_observation_views",
            "engine": "eevee", "resolution": 2, "samples": 4,
            "renders": [str(path) for path in paths],
        }), encoding="utf-8")
        rows.append({"arm": arm, "output": str(output), "exit_code": 0,
                     "native_generation_exit_code": 0, "final_view_render_exit_codes": [0]})
    return {"case_id": "synthetic", "results": rows}


def _check(pair: dict, root: Path) -> dict:
    return check_native_pair_completion(pair, project_root=root, resolution=2)


def _views(pair: dict) -> Path:
    return Path(pair["results"][0]["output"]) / "scene_synthetic/critic_final_views"


def test_complete_execution_preserves_codes_and_makes_no_quality_claim(pair: dict, tmp_path: Path) -> None:
    original = copy.deepcopy(pair)
    result = _check(pair, tmp_path)
    assert result["execution_complete"] is True
    assert result["aggregate_exit_code"] == 0
    assert len(result["arms"][0]["final_views"]) == 2
    assert result["arms"][0]["native_generation_exit_code"] == 0
    assert result["scientific_quality_assessed"] is False
    assert result["DPO_effect"] == "unknown"
    assert pair == original


@pytest.mark.parametrize("code", [124, 1, None, False, "0"])
def test_native_failure_or_malformed_code_cannot_complete(pair: dict, tmp_path: Path, code: object) -> None:
    pair["results"][0]["native_generation_exit_code"] = code
    result = _check(pair, tmp_path)
    assert result["aggregate_exit_code"] == 2
    assert result["arms"][0]["native_generation_exit_code"] == code


@pytest.mark.parametrize("codes", [[], [124], [1], [None], [False], ["0"], [0, 0]])
def test_native_success_cannot_hide_failed_or_missing_render(pair: dict, tmp_path: Path, codes: list) -> None:
    pair["results"][0]["final_view_render_exit_codes"] = codes
    result = _check(pair, tmp_path)
    assert result["execution_complete"] is False
    assert result["arms"][0]["native_generation_exit_code"] == 0
    assert result["arms"][0]["final_view_render_exit_codes"] == codes


@pytest.mark.parametrize("name", ["blend", "render_manifest.json", "00_top.png", "01_side.png"])
def test_successful_codes_still_require_actual_artifacts(pair: dict, tmp_path: Path, name: str) -> None:
    path = _views(pair) / name
    if name == "blend":
        path = _views(pair).parent / "combined_house/house.blend"
    path.unlink()
    assert _check(pair, tmp_path)["aggregate_exit_code"] == 2


@pytest.mark.parametrize("corruption", ["signature", "checksum", "truncated", "dimensions"])
def test_final_png_requires_complete_structure_and_dimensions(pair: dict, tmp_path: Path, corruption: str) -> None:
    path = _views(pair) / "00_top.png"
    data = bytearray(path.read_bytes())
    if corruption == "signature":
        data[0] ^= 1
    elif corruption == "checksum":
        data[-1] ^= 1
    elif corruption == "truncated":
        data = data[:-5]
    else:
        data = _png(3)
    path.write_bytes(data)
    assert _check(pair, tmp_path)["aggregate_exit_code"] == 2


@pytest.mark.parametrize("field", ["input", "renders", "samples"])
def test_manifest_must_bind_the_expected_checkpoint_and_views(pair: dict, tmp_path: Path, field: str) -> None:
    path = _views(pair) / "render_manifest.json"
    manifest = json.loads(path.read_text())
    if field == "input":
        manifest[field] = pair["results"][1]["output"] + "/scene_synthetic/combined_house/house.blend"
    elif field == "renders":
        manifest[field] = [manifest[field][0], manifest[field][0]]
    else:
        manifest[field] = 1
    path.write_text(json.dumps(manifest), encoding="utf-8")
    assert _check(pair, tmp_path)["aggregate_exit_code"] == 2


def test_duplicate_arms_do_not_create_a_pair(pair: dict, tmp_path: Path) -> None:
    pair["results"][1]["arm"] = "base"
    assert _check(pair, tmp_path)["aggregate_exit_code"] == 2


def test_two_arms_cannot_reuse_one_output(pair: dict, tmp_path: Path) -> None:
    pair["results"][1]["output"] = pair["results"][0]["output"]
    assert _check(pair, tmp_path)["aggregate_exit_code"] == 2


def test_stale_destination_view_does_not_replace_missing_current_render(tmp_path: Path) -> None:
    top = tmp_path / "0_top.png"
    top.write_bytes(_png())
    (tmp_path / "01_side.png").write_bytes(_png())
    with pytest.raises(ValueError, match="0_side"):
        required_final_view_sources([top])


def test_duplicate_current_view_is_rejected(tmp_path: Path) -> None:
    top = tmp_path / "0_top.png"
    top.write_bytes(_png())
    with pytest.raises(ValueError, match="Duplicate"):
        required_final_view_sources([top, top])


def test_empty_current_render_is_rejected(tmp_path: Path) -> None:
    top = tmp_path / "0_top.png"
    top.touch()
    with pytest.raises(ValueError, match="empty"):
        required_final_view_sources([top])


def test_producer_accepts_both_current_views(tmp_path: Path) -> None:
    paths = [tmp_path / "0_top.png", tmp_path / "0_side.png"]
    for path in paths:
        path.write_bytes(_png())
    assert set(required_final_view_sources(paths)) == {"0_top", "0_side"}


@pytest.mark.parametrize("render_code,expected_exit", [(0, 0), (124, 2)])
def test_cli_propagates_render_failure_without_running_a_renderer(pair: dict, tmp_path: Path, render_code: int, expected_exit: int) -> None:
    pair["results"][0]["final_view_render_exit_codes"] = [render_code]
    path = tmp_path / "synthetic-pair.json"
    path.write_text(json.dumps(pair), encoding="utf-8")
    script = Path(__file__).resolve().parents[2] / "scripts/check_sceneexpert_native_completion.py"
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1")
    result = subprocess.run([sys.executable, str(script), "--pair-report", str(path),
                             "--project-root", str(tmp_path), "--resolution", "2"],
                            capture_output=True, text=True, env=env, check=False)
    assert result.returncode == expected_exit, result.stderr
    assert json.loads(result.stdout)["aggregate_exit_code"] == expected_exit
