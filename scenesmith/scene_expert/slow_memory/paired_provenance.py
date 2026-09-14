"""Content-based pair identity for manually synchronized, Git-free runtimes."""

from __future__ import annotations

import os
import hashlib
import json
from pathlib import Path
from typing import Any

from scenesmith.scene_expert.trace_logger import collect_code_provenance

REPO = Path(__file__).resolve().parents[3]
_REQUIRED_SOURCES = (
    "main.py",
    "pyproject.toml",
    "scenesmith/agent_utils/base_stateful_agent.py",
    "scenesmith/scene_expert/slow_memory/paired_runtime.py",
    "scenesmith/scene_expert/slow_memory/paired_provenance.py",
    "scenesmith/scene_expert/slow_memory/paired_wire.py",
    "scripts/collect_sceneexpert_initial_pairs.py",
    "configurations/scene_expert/base_scene_expert.yaml",
    "tmp/acp/acp_qwen38_initial_pairs.sh",
    "tmp/acp/acp_qwen38_slow_memory_recollect.sh",
    "tmp/acp/acp_qwen38_full_generate.sh",
    "tmp/acp/acp_qwen38_4c_generate.sh",
)


def collect_pair_code_provenance(repo_root: Path | None = None) -> dict[str, Any]:
    """Hash source/config/prompt/launcher contents without invoking Git.

    Paths are relative and CRLF is normalized by the shared trace collector.
    Runtime outputs, caches, repository ownership and Git metadata are excluded.
    """
    provenance = collect_code_provenance(repo_root=repo_root or REPO, include_git=False)
    sources = provenance["source_hashes"]
    missing = [name for name in _REQUIRED_SOURCES if name not in sources]
    if missing:
        raise ValueError("paired runtime source files missing: " + ", ".join(missing))
    return {
        "identity_kind": "source_bundle_sha256",
        "source_bundle_hash": hashlib.sha256(
            json.dumps(sources, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "source_hashes": sources,
        "source_file_count": len(sources),
    }


def verify_pair_code_provenance(
    expected: dict[str, Any] | None = None, *, repo_root: Path | None = None
) -> dict[str, Any]:
    """Reject code drift from launcher startup or the candidate group's snapshot."""
    current = collect_pair_code_provenance(repo_root)
    if expected is not None:
        if expected.get("identity_kind") != "source_bundle_sha256" or not expected.get(
            "source_hashes"
        ):
            raise ValueError(
                "snapshot lacks content-based code identity; start a fresh pair run"
            )
        if (
            current["source_bundle_hash"] != expected.get("source_bundle_hash")
            or current["source_hashes"] != expected["source_hashes"]
        ):
            before, after = expected["source_hashes"], current["source_hashes"]
            changed = sorted(
                name
                for name in before.keys() | after.keys()
                if before.get(name) != after.get(name)
            )
            raise ValueError(
                "paired runtime source changed; refuse replay/export: "
                + ", ".join(changed[:8])
            )
    startup_hash = os.environ.get("SCENEEXPERT_PAIR_SOURCE_HASH", "")
    if startup_hash and current["source_bundle_hash"] != startup_hash:
        raise ValueError(
            "paired runtime source changed since ACP preflight; restart after synchronizing code"
        )
    return current
