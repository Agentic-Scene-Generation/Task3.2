"""Review archives preserve evidence, bound transfers and never alter source runs."""

from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

from scenesmith.scene_expert.review_package import MIB, package_results, verify_package

ROOT = Path(__file__).resolve().parents[2]
RUN = "qwen38_initial_pairs_007"
RESULTS = f"outputs/slow_memory/{RUN}"
LOGS = f"tmp/acp_logs/{RUN}"


def _write(root: Path, relative: str, data: bytes = b"{}\n") -> Path:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def _bundle(root: Path, **kwargs: object) -> tuple[dict, dict, dict[str, bytes]]:
    summary = package_results(
        project_root=root, run_id=RUN, output=root / "review.tar.gz", **kwargs
    )
    with tarfile.open(summary["archive"], "r:gz") as archive:
        content = {
            item.name.removeprefix(f"{RUN}_review/"): archive.extractfile(item).read()
            for item in archive.getmembers()
        }
    return summary, json.loads(content["_package/manifest.json"]), content


def test_original_layout_evidence_and_source_bytes_are_preserved(
    tmp_path: Path,
) -> None:
    pair = f"{RESULTS}/runs/paired_initial"
    keep = [
        f"{pair}/pair_audit.json",
        f"{pair}/group_1/snapshot.json",
        f"{pair}/group_1/A/raw_state.json",
        f"{pair}/group_1/A/report.json",
        f"{pair}/group_1/A/first_request.json",
        f"{pair}/group_1/B/failure.json",
        f"{pair}/group_1/A/slow_memory/trajectories.jsonl",
        f"{pair}/group_1/A/slow_memory/media/tool.png",
        f"{pair}/dpo/images/input.jpg",
        f"{pair}/dpo/all.jsonl",
        f"{RESULTS}/memory/skills.jsonl",
        f"{RESULTS}/runs/metrics/summary.csv",
        f"{LOGS}/llama_qwen38_27b.log",
    ]
    for name in keep:
        _write(tmp_path, name, name.encode())
    omit = [
        f"{pair}/group_1/input_scene/asset.obj",
        f"{pair}/group_1/A/raw_scene/asset.glb",
        f"{pair}/group_1/B/scene/asset.png",
        f"{pair}/group_1/A/candidate_payload.json",
        f"{RESULTS}/runs/session.db",
        f"{RESULTS}/model.gguf",
        f"{RESULTS}/cache/__pycache__/code.pyc",
        f"{RESULTS}/runs/render/texture.png",
    ]
    for name in omit:
        _write(tmp_path, name, b"unnecessary payload")
    before = {name: (tmp_path / name).read_bytes() for name in keep + omit}
    summary, manifest, content = _bundle(tmp_path)
    assert set(content) == {*keep, "_package/manifest.json", "_package/README.txt"}
    for name, data in before.items():
        assert (tmp_path / name).read_bytes() == data
        if name in keep:
            assert content[name] == data
    assert manifest["full_pair_reaudit_available"] is False
    assert manifest["experiment_verdict_recomputed"] is False
    assert verify_package(Path(summary["archive"]))["integrity_verified"] is True
    digest = hashlib.sha256(Path(summary["archive"]).read_bytes()).hexdigest()
    assert Path(summary["checksum"]).read_text().split()[0] == digest


def test_identical_latest_run_is_deduplicated_only_against_archived_bytes(
    tmp_path: Path,
) -> None:
    hydra = f"{RESULTS}/runs/hydra"
    latest = f"{RESULTS}/runs/latest-run"
    for directory in (hydra, latest):
        _write(tmp_path, f"{directory}/same.json", b"same")
        _write(tmp_path, f"{directory}/huge.json", b"x" * 100)
    _write(tmp_path, f"{hydra}/changed.json", b"old")
    _write(tmp_path, f"{latest}/changed.json", b"new")
    _, manifest, content = _bundle(tmp_path, max_file_bytes=32)
    assert f"{latest}/same.json" not in content
    assert content[f"{hydra}/same.json"] == b"same"
    assert content[f"{latest}/changed.json"] == b"new"
    reasons = {row["path"]: row["reason"] for row in manifest["omitted"]}
    assert reasons[f"{latest}/same.json"] == "identical_latest_run_copy"
    assert reasons[f"{latest}/huge.json"] == "per_file_limit"


def test_counterpart_outside_selected_roots_does_not_discard_latest_copy(
    tmp_path: Path,
) -> None:
    for name in ("hydra", "latest-run"):
        _write(tmp_path, f"{RESULTS}/runs/{name}/audit.json", b"audit")
    _, _, content = _bundle(
        tmp_path, collection_root=tmp_path / RESULTS / "runs/latest-run"
    )
    assert content[f"{RESULTS}/runs/latest-run/audit.json"] == b"audit"


def test_log_excerpts_are_explicit_and_limits_have_omission_records(
    tmp_path: Path,
) -> None:
    _write(tmp_path, f"{RESULTS}/pair_audit.json")
    _write(tmp_path, f"{RESULTS}/oversized.json", b"x" * 1024)
    _write(tmp_path, f"{LOGS}/terminal.log", b"START\n" + b"m" * MIB + b"\nCRASH END")
    summary, manifest, content = _bundle(tmp_path, max_file_bytes=100)
    assert f"{LOGS}/terminal.log" not in content
    excerpt = content[f"_package/log_excerpts/{LOGS}/terminal.log.excerpt.txt"]
    assert b"byte ranges" in excerpt and b"START" in excerpt and b"CRASH END" in excerpt
    assert len(excerpt) < 321 * 1024
    assert summary["warnings"]
    assert any(row["reason"] == "per_file_limit" for row in manifest["omitted"])


def test_total_budget_prioritizes_structured_evidence_and_records_missing_logs(
    tmp_path: Path,
) -> None:
    _write(tmp_path, f"{RESULTS}/report.json", b"1" * 70)
    _write(tmp_path, f"{RESULTS}/terminal.log", b"2" * 60)
    summary, manifest, content = _bundle(tmp_path, max_total_bytes=100)
    assert summary["selected_bytes"] <= 100
    assert f"{RESULTS}/report.json" in content
    assert f"{RESULTS}/terminal.log" not in content
    assert "total_size_limit_reached" in summary["warnings"]
    assert any(
        warning.startswith("source_missing:") for warning in manifest["warnings"]
    )


def test_logs_only_startup_failure_can_be_packaged(tmp_path: Path) -> None:
    _write(tmp_path, f"{LOGS}/terminal.log", b"preflight failed")
    _, manifest, content = _bundle(tmp_path)
    assert content[f"{LOGS}/terminal.log"] == b"preflight failed"
    assert f"source_missing: {RESULTS}" in manifest["warnings"]


def test_symlinks_are_never_followed(tmp_path: Path) -> None:
    _write(tmp_path, f"{RESULTS}/audit.json")
    external = _write(tmp_path, "outside/secrets.json", b"outside")
    try:
        (tmp_path / RESULTS / "file.json").symlink_to(external)
        (tmp_path / RESULTS / "cycle").symlink_to(
            tmp_path / RESULTS, target_is_directory=True
        )
    except OSError:
        pytest.skip("Symlink creation is unavailable")
    _, manifest, content = _bundle(tmp_path)
    assert all(b"outside" != value for value in content.values())
    assert {row["reason"] for row in manifest["omitted"]} >= {
        "symlink_or_special_file",
        "symlink_not_followed",
    }


@pytest.mark.parametrize("existing", ["review.tar.gz", "review.tar.gz.sha256"])
def test_preexisting_packages_are_not_overwritten(
    tmp_path: Path, existing: str
) -> None:
    _write(tmp_path, f"{RESULTS}/audit.json")
    original = _write(tmp_path, existing, b"keep")
    with pytest.raises(FileExistsError):
        _bundle(tmp_path)
    assert original.read_bytes() == b"keep"


def test_exclusive_creation_race_never_deletes_another_archive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write(tmp_path, f"{RESULTS}/audit.json")
    open_file = Path.open
    output = tmp_path / "review.tar.gz"

    def racing_open(path: Path, mode: str = "r", *args: object, **kwargs: object):
        if path == output and mode == "xb":
            with open_file(path, "wb") as stream:
                stream.write(b"another invocation")
            raise FileExistsError("concurrent output")
        return open_file(path, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", racing_open)
    with pytest.raises(FileExistsError):
        _bundle(tmp_path)
    assert output.read_bytes() == b"another invocation"


def test_changed_sources_abort_and_remove_only_owned_archive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _write(tmp_path, f"{RESULTS}/audit.json")
    read_bytes = Path.read_bytes

    def changing_read(path: Path) -> bytes:
        data = read_bytes(path)
        if path == source:
            path.write_bytes(b"new live result")
        return data

    monkeypatch.setattr(Path, "read_bytes", changing_read)
    with pytest.raises(ValueError, match="source changed"):
        _bundle(tmp_path)
    assert not (tmp_path / "review.tar.gz").exists()
    assert source.read_text() == "new live result"


@pytest.mark.parametrize("tamper", ["bytes", "extra", "unsafe", "duplicate"])
def test_integrity_verifier_rejects_tampered_archives(
    tmp_path: Path, tamper: str
) -> None:
    source_key = f"{RESULTS}/audit.json"
    _write(tmp_path, source_key, b"true")
    _, _, content = _bundle(tmp_path)
    if tamper == "bytes":
        content[source_key] = b"fake"
    elif tamper == "extra":
        content["extra.json"] = b"{}"
    elif tamper == "unsafe":
        content["../escape.json"] = b"{}"
    elif tamper == "duplicate":
        manifest = json.loads(content["_package/manifest.json"])
        manifest["included"].append(manifest["included"][0])
        content["_package/manifest.json"] = json.dumps(manifest).encode()
    output = tmp_path / "tampered.tar.gz"
    with tarfile.open(output, "w:gz") as archive:
        for name, data in content.items():
            item = tarfile.TarInfo(f"{RUN}_review/{name}")
            item.size = len(data)
            archive.addfile(item, io.BytesIO(data))
    with pytest.raises(ValueError):
        verify_package(output)


def test_cli_runs_without_any_installed_packages(tmp_path: Path) -> None:
    _write(tmp_path, f"{RESULTS}/audit.json")
    output = tmp_path / "review.tar.gz"
    command = [
        sys.executable,
        "-S",
        str(ROOT / "scripts/package_sceneexpert_results.py"),
    ]
    result = subprocess.run(
        command
        + ["--project-root", str(tmp_path), "--run-id", RUN, "--output", str(output)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    verified = subprocess.run(
        command + ["--verify", str(output)], capture_output=True, text=True, check=False
    )
    assert verified.returncode == 0, verified.stderr
    assert json.loads(verified.stdout)["integrity_verified"] is True


def test_acp_packaging_wrapper_uses_explicit_roots_and_requires_no_gpu(
    tmp_path: Path,
) -> None:
    git_bash = Path("C:/Program Files/Git/bin/bash.exe")
    bash = (
        str(git_bash)
        if os.name == "nt" and git_bash.is_file()
        else shutil.which("bash")
    )
    if not bash:
        pytest.skip("Bash is required")

    def shell_path(path: Path) -> str:
        value = path.resolve().as_posix()
        return f"/{value[0].lower()}{value[2:]}" if os.name == "nt" else value

    _write(tmp_path, "result/audit.json")
    # Copy only the stdlib helper and package implementation into a bare project.
    for name in (
        "scripts/package_sceneexpert_results.py",
        "scenesmith/scene_expert/review_package.py",
    ):
        _write(tmp_path, name, (ROOT / name).read_bytes())
    output = tmp_path / "download/review.tar.gz"
    result = subprocess.run(
        [bash, shell_path(ROOT / "tmp/acp/pack_qwen38_initial_pairs.sh")],
        env={
            **os.environ,
            "PROJECT_ROOT": shell_path(tmp_path),
            "RUN_ID": RUN,
            "PYTHON_BIN": shell_path(Path(sys.executable)),
            "COLLECTION_ROOT": shell_path(tmp_path / "result"),
            "ACP_LOG_DIR": shell_path(tmp_path / "missing-logs"),
            "PACKAGE_PATH": shell_path(output),
        },
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert verify_package(output)["integrity_verified"] is True
