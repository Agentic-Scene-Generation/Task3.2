"""Bounded, read-only review bundles with original project-relative paths."""

from __future__ import annotations

import hashlib
import io
import json
import os
import tarfile
from pathlib import Path, PurePosixPath
from typing import Any

MIB = 1024**2
TEXT_SUFFIXES = {
    ".json",
    ".jsonl",
    ".yaml",
    ".yml",
    ".log",
    ".out",
    ".err",
    ".txt",
    ".md",
    ".csv",
    ".tsv",
    ".env",
    ".sh",
}
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}
SKIP_DIRS = {".git", ".venv", "__pycache__", ".cache"}


def _policy(path: Path, relative: Path) -> tuple[int | None, str]:
    parts = relative.parts
    if "paired_initial" in parts:
        pair_parts = parts[parts.index("paired_initial") + 1 :]
        if any(part in {"input_scene", "raw_scene", "scene"} for part in pair_parts):
            return None, "replay_assets_retained_on_server"
    if path.name == "candidate_payload.json":
        return None, "redundant_raw_payload_use_trajectory_and_first_request"
    if any(part in SKIP_DIRS for part in parts):
        return None, "cache_or_environment"
    suffix = path.suffix.lower()
    if suffix in TEXT_SUFFIXES:
        if "llm_payloads" in parts:
            return 80, "raw_debug_payload"
        if suffix in {".log", ".out", ".err", ".txt"}:
            return 30, "log"
        return 10, "structured_evidence"
    if suffix in IMAGE_SUFFIXES:
        # Include original model-input/tool media and exported DPO images.
        # Intermediate renders/textures remain on the server by default.
        if ("slow_memory" in parts and "media" in parts) or (
            "images" in parts and any(p in {"dpo", "dpo_probe"} for p in parts)
        ):
            return 40, "evidence_image"
        return None, "intermediate_render_or_texture"
    return None, "large_asset_database_or_nonreview_file"


def _sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _stat_identity(path: Path) -> tuple[int, int]:
    stat = path.stat()
    return stat.st_size, stat.st_mtime_ns


def _log_excerpt(path: Path) -> bytes:
    with path.open("rb") as stream:
        size = path.stat().st_size
        head = stream.read(64 * 1024)
        tail_start = max(len(head), size - 256 * 1024)
        stream.seek(tail_start)
        tail = stream.read(256 * 1024)
    note = (
        f"[REVIEW PACKAGE: log excerpt; original retained on server; "
        f"byte ranges [0,{len(head)}) and [{tail_start},{tail_start + len(tail)}); "
        f"original bytes={size}]\n\n"
    )
    return (
        note
        + head.decode("utf-8", errors="replace")
        + "\n\n[REVIEW PACKAGE: omitted middle, if any]\n\n"
        + tail.decode("utf-8", errors="replace")
    ).encode("utf-8")


def _add_bytes(archive: tarfile.TarFile, name: str, data: bytes) -> None:
    item = tarfile.TarInfo(name)
    item.size, item.mode = len(data), 0o644
    archive.addfile(item, io.BytesIO(data))


def package_results(
    *,
    project_root: Path,
    run_id: str,
    output: Path,
    collection_root: Path | None = None,
    log_root: Path | None = None,
    max_file_bytes: int = 32 * MIB,
    max_total_bytes: int = 512 * MIB,
) -> dict[str, Any]:
    """Copy selected result bytes without changing files or experiment verdicts."""
    if not run_id or any(
        c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
        for c in run_id
    ):
        raise ValueError(
            "run_id must contain only letters, digits, underscores or hyphens"
        )
    if max_file_bytes <= 0 or max_total_bytes <= 0:
        raise ValueError("package size limits must be positive")
    project_root, output = project_root.resolve(), output.resolve()
    checksum_path = output.with_name(output.name + ".sha256")
    if output.exists() or checksum_path.exists():
        raise FileExistsError(
            "package/checksum already exists; choose a new output name"
        )
    roots = [
        (collection_root or project_root / "outputs/slow_memory" / run_id).absolute(),
        (log_root or project_root / "tmp/acp_logs" / run_id).absolute(),
    ]
    for root in roots:
        if root.is_symlink() or not root.resolve().is_relative_to(project_root):
            raise ValueError("source roots must be real directories within the project")
        if output.is_relative_to(root.resolve()):
            raise ValueError("write the package outside the source directories")
    manifest: dict[str, Any] = {
        "schema_version": "sceneexpert.review_package.v1",
        "run_id": run_id,
        "purpose": "local_diagnosis_only",
        "full_replay_available": False,
        "full_pair_reaudit_available": False,
        "experiment_verdict_recomputed": False,
        "layout": "original_project_relative_paths",
        "source_project_root": str(project_root),
        "source_roots": [root.relative_to(project_root).as_posix() for root in roots],
        "limits": {
            "max_file_bytes": max_file_bytes,
            "max_total_bytes": max_total_bytes,
        },
        "included": [],
        "omitted": [],
        "warnings": [],
    }
    candidates: list[tuple[int, str, Path, str]] = []
    seen: set[str] = set()
    for root in roots:
        if not root.is_dir():
            manifest["warnings"].append(
                f"source_missing: {root.relative_to(project_root).as_posix()}"
            )
            continue
        for parent, dirs, files in os.walk(root, followlinks=False):
            directory = Path(parent)
            for name in sorted(dirs[:]):
                child = directory / name
                relative = child.relative_to(project_root)
                _, reason = _policy(child, relative)
                if child.is_symlink() or reason in {
                    "replay_assets_retained_on_server",
                    "cache_or_environment",
                }:
                    dirs.remove(name)
                    manifest["omitted"].append(
                        {
                            "path": relative.as_posix(),
                            "kind": "subtree",
                            "reason": (
                                "symlink_not_followed" if child.is_symlink() else reason
                            ),
                        }
                    )
            for name in sorted(files):
                path = directory / name
                relative = path.relative_to(project_root)
                key = relative.as_posix()
                if key in seen:
                    continue
                seen.add(key)
                if path.is_symlink() or not path.is_file():
                    manifest["omitted"].append(
                        {"path": key, "reason": "symlink_or_special_file"}
                    )
                    continue
                priority, reason = _policy(path, relative)
                if priority is None:
                    manifest["omitted"].append(
                        {
                            "path": key,
                            "size_bytes": path.stat().st_size,
                            "reason": reason,
                        }
                    )
                    continue
                candidates.append((priority, key, path, reason))
    if not candidates:
        raise ValueError("no review files found; check RUN_ID and source paths")
    prefix = f"{run_id}_review"
    output.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    created = False
    included_originals: dict[str, dict[str, Any]] = {}
    try:
        # Exclusive creation protects previously downloaded packages.
        destination = output.open("xb")
        created = True
        with (
            destination,
            tarfile.open(fileobj=destination, mode="w:gz", compresslevel=6) as archive,
        ):
            for priority, key, path, kind in sorted(candidates):
                before = _stat_identity(path)
                # Archive copies of latest-run sometimes are real directories.
                # Omit only a byte-identical copy with a selected hydra counterpart.
                parts = Path(key).parts
                if "latest-run" in parts:
                    alias = list(parts)
                    alias[alias.index("latest-run")] = "hydra"
                    canonical_key = Path(*alias).as_posix()
                    counterpart = included_originals.get(canonical_key)
                    if counterpart and counterpart["size_bytes"] == before[0]:
                        digest = _sha(path)
                        if _stat_identity(path) != before:
                            raise ValueError(f"source changed while packaging: {key}")
                        if digest == counterpart["sha256"]:
                            manifest["omitted"].append(
                                {
                                    "path": key,
                                    "size_bytes": before[0],
                                    "reason": "identical_latest_run_copy",
                                    "canonical_path": canonical_key,
                                }
                            )
                            continue
                if before[0] > max_file_bytes:
                    if kind != "log":
                        manifest["omitted"].append(
                            {
                                "path": key,
                                "size_bytes": before[0],
                                "reason": "per_file_limit",
                            }
                        )
                        if priority in {10, 40}:
                            manifest["warnings"].append(f"{kind}_omitted: {key}")
                        continue
                    data = _log_excerpt(path)
                    archive_key = "_package/log_excerpts/" + key + ".excerpt.txt"
                    kind = "log_excerpt"
                else:
                    if total + before[0] > max_total_bytes:
                        manifest["omitted"].append(
                            {
                                "path": key,
                                "size_bytes": before[0],
                                "reason": "total_size_limit",
                            }
                        )
                        manifest["warnings"].append("total_size_limit_reached")
                        continue
                    data = path.read_bytes()
                    archive_key = key
                if _stat_identity(path) != before:
                    raise ValueError(
                        f"source changed while packaging; retry after the run stops: {key}"
                    )
                if total + len(data) > max_total_bytes:
                    manifest["omitted"].append(
                        {
                            "path": key,
                            "size_bytes": before[0],
                            "reason": "total_size_limit",
                        }
                    )
                    manifest["warnings"].append("total_size_limit_reached")
                    continue
                _add_bytes(archive, f"{prefix}/{archive_key}", data)
                total += len(data)
                record = {
                    "path": archive_key,
                    "source_path": key,
                    "kind": kind,
                    "size_bytes": len(data),
                    "sha256": hashlib.sha256(data).hexdigest(),
                }
                manifest["included"].append(record)
                if kind == "log_excerpt":
                    manifest["omitted"].append(
                        {
                            "path": key,
                            "size_bytes": before[0],
                            "reason": "oversized_log_excerpt_included",
                            "excerpt_path": archive_key,
                        }
                    )
                else:
                    included_originals[key] = record
            if not manifest["included"]:
                raise ValueError("size limits excluded all review files")
            manifest["selected_bytes"] = total
            manifest["warnings"] = sorted(set(manifest["warnings"]))
            readme = (
                b"SceneExpert lightweight review package\n\n"
                b"Original files keep project-relative paths and unchanged bytes.\n"
                b"Oversized log excerpts are derivatives under _package/log_excerpts/.\n"
                b"manifest.json lists included hashes and omitted files/subtrees.\n"
                b"This is not a full scene replay or training-ready archive. Retain the\n"
                b"complete source run on the server; full asset-based pair audits must\n"
                b"run there. Copied experiment audit reports are not recomputed here.\n"
            )
            _add_bytes(archive, f"{prefix}/_package/README.txt", readme)
            manifest["included"].append(
                {
                    "path": "_package/README.txt",
                    "kind": "package_note",
                    "size_bytes": len(readme),
                    "sha256": hashlib.sha256(readme).hexdigest(),
                }
            )
            _add_bytes(
                archive,
                f"{prefix}/_package/manifest.json",
                json.dumps(manifest, ensure_ascii=False, indent=2).encode(),
            )
        verify_package(output)
        with checksum_path.open("x", encoding="utf-8") as handle:
            handle.write(f"{_sha(output)}  {output.name}\n")
    except BaseException:
        # This invocation owns the exclusively created output, never source data.
        if created:
            output.unlink(missing_ok=True)
        raise
    return {
        "archive": str(output),
        "checksum": str(checksum_path),
        "archive_bytes": output.stat().st_size,
        "selected_bytes": total,
        "included_count": len(manifest["included"]),
        "omitted_count": len(manifest["omitted"]),
        "warnings": manifest["warnings"],
    }


def verify_package(path: Path) -> dict[str, Any]:
    """Check member hashes without extracting files or trusting experiment verdicts."""
    with tarfile.open(path, "r:gz") as archive:
        members = archive.getmembers()
        names = [member.name for member in members]
        manifests = [name for name in names if name.endswith("/_package/manifest.json")]
        if len(manifests) != 1 or len(set(names)) != len(names):
            raise ValueError("invalid review package manifest/member list")
        prefix = manifests[0].removesuffix("/_package/manifest.json")
        for member in members:
            relative = PurePosixPath(member.name)
            if (
                not member.isfile()
                or relative.is_absolute()
                or ".." in relative.parts
                or "\\" in member.name
                or ":" in member.name
                or not member.name.startswith(prefix + "/")
            ):
                raise ValueError("unsafe review package member")
        with archive.extractfile(manifests[0]) as stream:
            manifest = json.load(stream)
        if manifest.get("schema_version") != "sceneexpert.review_package.v1":
            raise ValueError("unsupported review package schema")
        expected = {prefix + "/" + row["path"]: row for row in manifest["included"]}
        if len(expected) != len(manifest["included"]):
            raise ValueError("duplicate package inventory entry")
        if set(names) != {*expected, manifests[0]}:
            raise ValueError("package member inventory differs from manifest")
        for name, row in expected.items():
            if archive.getmember(name).size != row["size_bytes"]:
                raise ValueError(f"package size mismatch: {name}")
            with archive.extractfile(name) as stream:
                if hashlib.file_digest(stream, "sha256").hexdigest() != row["sha256"]:
                    raise ValueError(f"package checksum mismatch: {name}")
    return {
        "integrity_verified": True,
        "file_count": len(expected),
        "purpose": manifest["purpose"],
    }
