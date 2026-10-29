"""Check native execution artifacts separately from scientific scene quality."""

from __future__ import annotations

import hashlib
import json
import struct
import zlib

from pathlib import Path
from typing import Any, Iterable


FINAL_VIEW_NAMES = {"0_top": "00_top.png", "0_side": "01_side.png"}


def required_final_view_sources(rendered: Iterable[Path]) -> dict[str, Path]:
    """Require both views from this render call before publishing its manifest."""
    sources: dict[str, Path] = {}
    for value in rendered:
        path = Path(value)
        if path.stem not in FINAL_VIEW_NAMES:
            continue
        if path.stem in sources:
            raise ValueError(f"Duplicate final render view: {path.stem}")
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"Missing or empty final render view: {path}")
        sources[path.stem] = path
    missing = set(FINAL_VIEW_NAMES) - set(sources)
    if missing:
        raise ValueError(f"Missing final render views: {', '.join(sorted(missing))}")
    return sources


def _inside(value: Any, root: Path, project_root: Path) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Artifact path must be a nonempty string")
    path = Path(value)
    path = (path if path.is_absolute() else project_root / path).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"Artifact escapes its expected root: {value}")
    return path


def _png_evidence(path: Path, resolution: int) -> dict[str, Any]:
    """Read PNG structure and checksums; this does not judge image content."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        signature = stream.read(8)
        digest.update(signature)
        if signature != b"\x89PNG\r\n\x1a\n":
            raise ValueError(f"Invalid PNG signature: {path}")
        dimensions = None
        has_image_data = False
        while True:
            header = stream.read(8)
            if len(header) != 8:
                raise ValueError(f"Truncated PNG: {path}")
            digest.update(header)
            length, kind = struct.unpack(">I4s", header)
            if dimensions is None and (kind != b"IHDR" or length != 13):
                raise ValueError(f"Missing PNG header: {path}")
            crc = zlib.crc32(kind)
            remaining = length
            initial = b""
            while remaining:
                data = stream.read(min(remaining, 1024 * 1024))
                if not data:
                    raise ValueError(f"Truncated PNG chunk: {path}")
                digest.update(data)
                crc = zlib.crc32(data, crc)
                if kind == b"IHDR":
                    initial += data
                remaining -= len(data)
            checksum = stream.read(4)
            digest.update(checksum)
            if len(checksum) != 4 or struct.unpack(">I", checksum)[0] != crc:
                raise ValueError(f"Invalid PNG checksum: {path}")
            if kind == b"IHDR":
                if dimensions is not None:
                    raise ValueError(f"Duplicate PNG header: {path}")
                dimensions = struct.unpack(">II", initial[:8])
                if dimensions != (resolution, resolution):
                    raise ValueError(f"Unexpected final view dimensions: {path}")
            elif kind == b"IDAT":
                has_image_data = has_image_data or length > 0
            elif kind == b"IEND":
                if length or not has_image_data or stream.read(1):
                    raise ValueError(f"Incomplete PNG image: {path}")
                break
    return {"path": str(path), "sha256": digest.hexdigest(), "dimensions": dimensions}


def check_native_pair_completion(
    report: dict[str, Any],
    *,
    project_root: Path,
    expected_arms: tuple[str, ...] = ("base", "adapter"),
    resolution: int = 1536,
    samples: int = 4,
    engine: str = "eevee",
) -> dict[str, Any]:
    """Require successful native/render exits and the fixed final artifacts."""
    if (
        not expected_arms
        or len(set(expected_arms)) != len(expected_arms)
        or any(not isinstance(arm, str) or not arm.strip() for arm in expected_arms)
        or type(resolution) is not int
        or resolution <= 0
        or type(samples) is not int
        or samples <= 0
    ):
        raise ValueError("Invalid expected arms or final render settings")
    project_root = Path(project_root).resolve()
    failures: list[str] = []
    evidence: list[dict[str, Any]] = []
    rows = report.get("results")
    if not isinstance(rows, list):
        rows = []
        failures.append("results must be a list")
    if [row.get("arm") if isinstance(row, dict) else None for row in rows] != list(expected_arms):
        failures.append("assigned arm count, identity or order differs")
    outputs: set[Path] = set()
    for row in rows:
        if not isinstance(row, dict):
            failures.append("each arm result must be an object")
            continue
        arm = row.get("arm")
        native_exit = row.get("native_generation_exit_code")
        render_exits = row.get("final_view_render_exit_codes")
        item = {"arm": arm, "native_generation_exit_code": native_exit,
                "final_view_render_exit_codes": render_exits, "final_views": []}
        evidence.append(item)
        if type(native_exit) is not int or native_exit != 0:
            failures.append(f"{arm}: native generation did not succeed")
        if type(row.get("exit_code")) is not int or row["exit_code"] != 0:
            failures.append(f"{arm}: supervisor reported a failed or unknown exit")
        if not isinstance(render_exits, list) or len(render_exits) != 1 or any(
            type(code) is not int or code != 0 for code in render_exits
        ):
            failures.append(f"{arm}: final camera rendering did not succeed")
        try:
            output = _inside(row.get("output"), project_root, project_root)
            if output == project_root or output in outputs:
                raise ValueError("Arm output roots must be distinct scene runs")
            outputs.add(output)
            blends = sorted(output.glob("scene_*/combined_house/house.blend"))
            if len(blends) != 1 or not blends[0].is_file() or blends[0].stat().st_size == 0:
                raise ValueError("Expected one nonempty combined house checkpoint")
            blend = blends[0].resolve()
            if not blend.is_relative_to(output):
                raise ValueError("Combined house checkpoint escapes arm output")
            view_root = blend.parent.parent / "critic_final_views"
            manifest_path = view_root / "render_manifest.json"
            if not manifest_path.resolve().is_relative_to(output):
                raise ValueError("Final render manifest escapes arm output")
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if not isinstance(manifest, dict):
                raise ValueError("Final render manifest must be an object")
            if _inside(manifest.get("input"), output, project_root) != blend:
                raise ValueError("Final render manifest names another checkpoint")
            if (
                type(manifest.get("resolution")) is not int
                or manifest["resolution"] != resolution
                or type(manifest.get("samples")) is not int
                or manifest["samples"] != samples
                or manifest.get("engine") != engine
                or manifest.get("renderer") != "BlenderRenderer.render_agent_observation_views"
            ):
                raise ValueError("Final render settings differ from the expected protocol")
            renders = manifest.get("renders")
            if not isinstance(renders, list) or len(renders) != 2:
                raise ValueError("Expected exactly two final camera paths")
            paths = [_inside(value, view_root, project_root) for value in renders]
            expected = {view_root / name for name in FINAL_VIEW_NAMES.values()}
            if len(set(paths)) != 2 or set(paths) != expected:
                raise ValueError("Final camera identities are missing or duplicated")
            item["final_views"] = [_png_evidence(path, resolution) for path in paths]
            item["manifest_sha256"] = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        except (OSError, ValueError, TypeError, struct.error) as error:
            failures.append(f"{arm}: {error}")
    complete = not failures
    return {
        "schema": "sceneexpert.native_execution_completion.v1",
        "execution_complete": complete,
        "aggregate_exit_code": 0 if complete else 2,
        "failures": failures,
        "arms": evidence,
        "scientific_quality_assessed": False,
        "DPO_effect": "unknown",
    }
