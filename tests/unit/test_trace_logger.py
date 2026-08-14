"""Regression coverage for trace reproducibility metadata."""

from __future__ import annotations

import hashlib
import json

from scenesmith.scene_expert.schemas import FullVerifyReport
from scenesmith.scene_expert.trace_logger import TraceLogger, collect_code_provenance


def test_collect_code_provenance_hashes_requested_source_files(tmp_path) -> None:
    source = tmp_path / "tracked.py"
    source.write_text("answer = 42\n", encoding="utf-8")

    provenance = collect_code_provenance(
        repo_root=tmp_path,
        source_paths=["tracked.py", "missing.py", "../outside.py"],
    )

    assert provenance["repo_root"] == str(tmp_path)
    assert provenance["source_hashes"] == {
        "tracked.py": hashlib.sha256(b"answer = 42\n").hexdigest()
    }
    assert provenance["git_revision"] == ""
    assert provenance["dirty"] is None


def test_trace_logger_persists_code_provenance_in_partial_and_final_traces(
    tmp_path,
) -> None:
    provenance = {
        "git_revision": "deadbeef",
        "dirty": True,
        "source_hashes": {"scenesmith/example.py": "abc123"},
    }
    logger = TraceLogger(
        output_dir=str(tmp_path),
        scene_index=7,
        prompt="A reproducible test scene.",
        code_provenance=provenance,
    )

    partial_path = logger.save_partial(status="running")
    partial = json.loads(partial_path.read_text(encoding="utf-8"))
    assert partial["code_provenance"] == provenance

    final_trace = logger.finalize(FullVerifyReport(), exports={})
    assert final_trace["code_provenance"] == provenance
    final_path = logger.save(final_trace)
    assert (
        json.loads(final_path.read_text(encoding="utf-8"))["code_provenance"]
        == provenance
    )
