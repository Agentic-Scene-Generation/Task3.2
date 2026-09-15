"""Auditable raw-state restoration: private assets and rotation roundoff only."""

from __future__ import annotations

import copy
import math
from collections.abc import Iterator
from pathlib import Path, PurePosixPath
from typing import Any

from scenesmith.scene_expert.slow_memory.paired import digest, read_json, write_json

RESTORATION_PROTOCOL = "sceneexpert.raw_restoration.v1"
# Quaternion components are bounded by one. Allow only double-precision
# conversion noise, never translation/bounds/metadata drift or a repair.
QUATERNION_ATOL = 64 * math.ulp(1.0)


def _objects(state: dict[str, Any]) -> Iterator[tuple[list[Any], dict[str, Any]]]:
    for key, obj in state.get("objects", {}).items():
        yield ["objects", key], obj
    geometry = state.get("room_geometry") or {}
    for index, wall in enumerate(geometry.get("walls") or []):
        yield ["room_geometry", "walls", index], wall
    if geometry.get("floor"):
        yield ["room_geometry", "floor"], geometry["floor"]


def _get(state: Any, path: list[Any]) -> Any:
    for key in path:
        state = state[key]
    return state


def state_differences(expected: Any, actual: Any, path: str = "$") -> list[str]:
    """List bounded, field-level mismatches without dumping scene/prompt content."""
    if isinstance(expected, dict) and isinstance(actual, dict):
        result = []
        for key in sorted(expected.keys() | actual.keys()):
            if key not in expected or key not in actual:
                result.append(f"{path}/{key}")
            else:
                result.extend(
                    state_differences(expected[key], actual[key], f"{path}/{key}")
                )
            if len(result) >= 64:
                return result[:64]
        return result
    if (
        isinstance(expected, list)
        and isinstance(actual, list)
        and len(expected) == len(actual)
    ):
        result = []
        for index, (a, b) in enumerate(zip(expected, actual, strict=True)):
            result.extend(state_differences(a, b, f"{path}/{index}"))
            if len(result) >= 64:
                break
        return result[:64]
    # Distinguish JSON booleans/numbers too; hashes keep their exact encoding.
    return [] if digest(expected) == digest(actual) else [path]


def equivalent_restored_state(
    expected: dict[str, Any], actual: dict[str, Any]
) -> list[dict[str, Any]]:
    """Accept only equivalent unit quaternions at declared native transform fields."""
    normalized = copy.deepcopy(actual)
    adjustments = []
    for base, obj in _objects(expected):
        transforms = [(base + ["transform"], obj["transform"])]
        transforms.extend(
            (base + ["support_surfaces", i, "transform"], surf["transform"])
            for i, surf in enumerate(obj.get("support_surfaces") or [])
        )
        for path, transform in transforms:
            field = path + ["rotation_wxyz"]
            a, b = transform["rotation_wxyz"], _get(actual, field)
            if digest(a) == digest(b):
                continue
            if (
                len(a) != 4
                or len(b) != 4
                or not all(
                    type(v) in (int, float) and math.isfinite(v) for v in [*a, *b]
                )
                or abs(math.hypot(*a) - 1.0) > QUATERNION_ATOL
                or abs(math.hypot(*b) - 1.0) > QUATERNION_ATOL
            ):
                raise ValueError(f"invalid unit quaternion at {field}")
            sign = 1 if sum(x * y for x, y in zip(a, b, strict=True)) >= 0 else -1
            error = max(abs(x - sign * y) for x, y in zip(a, b, strict=True))
            if error > QUATERNION_ATOL:
                raise ValueError(f"restored rotation changed at {field}: {error}")
            _get(normalized, path)["rotation_wxyz"] = copy.deepcopy(a)
            adjustments.append(
                {"path": field, "max_component_error": error, "quaternion_sign": sign}
            )
    differences = state_differences(expected, normalized)
    if differences:
        raise ValueError("restored raw state differs at " + ", ".join(differences[:8]))
    return adjustments


def relocate_raw_assets(
    state: dict[str, Any],
    directory: Path,
    snapshot: dict[str, Any],
    source_candidate: Path,
    raw_files: dict[str, str],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    """Resolve typed asset fields to verified files in the copied raw scene."""
    mapped = copy.deepcopy(state)
    root = (directory / "raw_scene").resolve()
    room = root / snapshot["room_relative"]
    if not room.resolve().is_relative_to(root):
        raise ValueError("raw room path escapes its private asset copy")
    roots = [
        PurePosixPath(snapshot["source_scene_root"]),
        PurePosixPath(source_candidate.as_posix()) / "scene",
    ]
    fields = [
        (base + [key], obj[key])
        for base, obj in _objects(state)
        for key in ("geometry_path", "sdf_path", "image_path")
        if obj.get(key)
    ]
    if state.get("room_geometry", {}).get("sdf_path"):
        fields.append(
            (["room_geometry", "sdf_path"], state["room_geometry"]["sdf_path"])
        )
    bindings = []
    unavailable_images = []
    for field, value in fields:
        old = PurePosixPath(value)
        if old.is_absolute():
            relative = next(
                (old.relative_to(base) for base in roots if old.is_relative_to(base)),
                None,
            )
            if relative is None:
                raise ValueError(
                    f"asset path is outside captured scene roots at {field}"
                )
            target = root / str(relative)
        else:
            target = room / value
        target = target.resolve()
        if not target.is_relative_to(root):
            raise ValueError(f"asset path escapes private raw scene at {field}")
        relative_name = target.relative_to(root).as_posix()
        if relative_name not in raw_files or not target.is_file():
            # Legacy asset selection leaves unused relative preview-image names
            # in SceneObject. Main deterministic checks never read these; actual
            # model-input media remains mandatory in validate_pair_inputs().
            if field[-1] == "image_path" and not old.is_absolute():
                unavailable_images.append(
                    {"path": field, "source": value, "raw_asset": relative_name}
                )
                continue
            raise ValueError(f"retained asset missing at {field}: {relative_name}")
        # Match native safe_relative_path: outside-room files serialize absolute.
        serialized = (
            str(target.relative_to(room))
            if target.is_relative_to(room)
            else str(target)
        )
        if serialized != value:
            _get(mapped, field[:-1])[field[-1]] = serialized
            bindings.append(
                {
                    "path": field,
                    "source": value,
                    "mapped": serialized,
                    "raw_asset": relative_name,
                    "sha256": raw_files[relative_name],
                }
            )
    return mapped, bindings, unavailable_images


def save_restoration_proof(
    directory: Path,
    source: dict[str, Any],
    mapped: dict[str, Any],
    restored: dict[str, Any],
    bindings: list[dict[str, Any]],
    unavailable_images: list[dict[str, Any]] | None = None,
) -> None:
    """Persist both actual restored state and its equivalence proof, even on failure."""
    write_json(directory / "restored_state.json", restored)
    proof = {
        "schema_version": RESTORATION_PROTOCOL,
        "source_raw_state_sha256": digest(source),
        "mapped_state_sha256": digest(mapped),
        "restored_state_sha256": digest(restored),
        "asset_bindings": bindings,
        "unavailable_reference_images": unavailable_images or [],
        "quaternion_atol": QUATERNION_ATOL,
    }
    try:
        proof["rotation_roundoff"] = equivalent_restored_state(mapped, restored)
        proof["status"] = "verified"
    except (ValueError, KeyError, TypeError, IndexError) as exc:
        proof.update(
            status="failed",
            error=str(exc),
            differences=state_differences(mapped, restored),
        )
        raise
    finally:
        write_json(directory / "restoration_proof.json", proof)


def validate_restoration_proof(
    directory: Path, candidate: dict[str, Any], scoring: dict[str, Any]
) -> None:
    """Recheck serialized equivalence and asset bindings before exporting a label."""
    proof = read_json(directory / "restoration_proof.json")
    source = read_json(directory / "raw_state.json")
    restored = read_json(directory / "restored_state.json")
    mapped = copy.deepcopy(source)
    allowed_fields = {
        tuple(base + [key])
        for base, obj in _objects(source)
        for key in ("geometry_path", "sdf_path", "image_path")
        if obj.get(key)
    }
    allowed_fields.add(("room_geometry", "sdf_path"))
    for item in proof.get("unavailable_reference_images", []):
        field = item["path"]
        if (
            tuple(field) not in allowed_fields
            or field[-1] != "image_path"
            or _get(source, field) != item["source"]
            or _get(restored, field) != item["source"]
            or item["raw_asset"] in candidate["raw_files"]
        ):
            raise ValueError("unavailable reference-image provenance mismatch")
    seen = set()
    for binding in proof["asset_bindings"]:
        field = binding["path"]
        relative = PurePosixPath(binding["raw_asset"])
        original = PurePosixPath(binding["source"])
        target = PurePosixPath(binding["mapped"].replace("\\", "/"))
        if (
            tuple(field) not in allowed_fields
            or tuple(field) in seen
            or relative.is_absolute()
            or ".." in relative.parts
            or not original.is_absolute()
            or not target.is_absolute()
            or len(target.parts) <= len(relative.parts)
            or target.parts[-len(relative.parts) - 1] != "raw_scene"
            or original.parts[-len(relative.parts) :] != relative.parts
            or target.parts[-len(relative.parts) :] != relative.parts
            or _get(source, field) != binding["source"]
            or candidate["raw_files"].get(binding["raw_asset"]) != binding["sha256"]
        ):
            raise ValueError("restoration asset binding mismatch")
        seen.add(tuple(field))
        _get(mapped, field[:-1])[field[-1]] = binding["mapped"]
    adjustments = equivalent_restored_state(mapped, restored)
    if (
        proof.get("schema_version") != RESTORATION_PROTOCOL
        or proof.get("status") != "verified"
        or scoring.get("restoration_proof_sha256") != digest(proof)
        or proof.get("source_raw_state_sha256") != digest(source)
        or proof.get("source_raw_state_sha256") != candidate["raw_state_hash"]
        or proof.get("mapped_state_sha256") != digest(mapped)
        or proof.get("restored_state_sha256") != digest(restored)
        or proof.get("restored_state_sha256") != candidate["evaluation_state_hash"]
        or proof.get("rotation_roundoff") != adjustments
        or proof.get("quaternion_atol") != QUATERNION_ATOL
    ):
        raise ValueError("raw restoration proof mismatch")
