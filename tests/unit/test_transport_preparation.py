"""Preparation must reject a stale CLI pin even when every plan hash matches."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from scenesmith.scene_expert.transport_preflight import check_transport_preparation


def _fixture(root, *, pin=None, options="", before="", after=""):
    files = {
        "control/packer.py": b'raise RuntimeError("packer must never be imported")\n',
        "scripts/package_sceneexpert_exact_transport.py": b"# fixture entry point\n",
        "scenesmith/scene_expert/transport_media.py": b"# fixture module\n",
    }
    actual = hashlib.sha256(files["control/packer.py"]).hexdigest()
    text = (
        "#!/bin/bash\nset -euo pipefail\n" + before
        + "exec python3 scripts/package_sceneexpert_exact_transport.py \\\n"
        + f" --packer control/packer.py --packer-sha256 {pin or actual} "
        + "--plan control/plan.json --prefix 'tmp/results/review set' "
        + options + "\n" + after
    )
    files["control/pack.sh"] = text.encode()
    for name, data in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    plan = {
        "control_input_sha256": {
            n: hashlib.sha256(v).hexdigest() for n, v in files.items()
        }
    }
    (root / "control/plan.json").write_text(json.dumps(plan), encoding="utf-8")
    return actual


def _check(root):
    return check_transport_preparation(
        root, Path("control/plan.json"), Path("control/pack.sh")
    )


def test_correct_bindings_do_not_import_packer_execute_shell_or_create_output(tmp_path):
    marker = tmp_path / "shell-must-not-run"
    pin = _fixture(tmp_path, before=f"touch {marker.as_posix()}\n")
    before = {
        p.relative_to(tmp_path): p.read_bytes()
        for p in tmp_path.rglob("*") if p.is_file()
    }
    proof = _check(tmp_path)
    assert proof["passed"] and proof["packer_sha256"] == pin
    assert len(proof["checked_source_sha256"]) == 4
    assert proof["packer_imported"] is proof["shell_executed"] is False
    assert not marker.exists() and not (tmp_path / "tmp").exists()
    assert before == {
        p.relative_to(tmp_path): p.read_bytes()
        for p in tmp_path.rglob("*") if p.is_file()
    }


def test_stale_wrapper_pin_fails_despite_matching_frozen_file_hashes(tmp_path):
    _fixture(tmp_path, pin="0" * 64)
    with pytest.raises(ValueError, match="pin differs"):
        _check(tmp_path)


@pytest.mark.parametrize("name", ["control/packer.py", "control/pack.sh"])
def test_unsealed_source_changes_fail(tmp_path, name):
    _fixture(tmp_path)
    path = tmp_path / name
    path.write_bytes(path.read_bytes() + b"\n# unsealed change\n")
    with pytest.raises(ValueError, match="Plan/source SHA256 mismatch"):
        _check(tmp_path)


@pytest.mark.parametrize(
    "options,after,expected",
    [
        ("--packer control/other.py", "", "duplicate"),
        ("--verify elsewhere.json", "", "Unsupported"),
        ("", "exec python3 another.py\n", "Exactly one"),
    ],
)
def test_ambiguous_or_non_packaging_commands_fail(tmp_path, options, after, expected):
    _fixture(tmp_path, options=options, after=after)
    with pytest.raises(ValueError, match=expected):
        _check(tmp_path)


def test_shell_expansion_is_never_evaluated(tmp_path):
    _fixture(tmp_path)
    path = tmp_path / "control/pack.sh"
    path.write_text(path.read_text().replace("control/packer.py", "'$PACKER'"))
    plan_path = tmp_path / "control/plan.json"
    plan = json.loads(plan_path.read_text())
    plan["control_input_sha256"]["control/pack.sh"] = hashlib.sha256(
        path.read_bytes()
    ).hexdigest()
    plan_path.write_text(json.dumps(plan))
    with pytest.raises(ValueError, match="literal"):
        _check(tmp_path)


def test_plan_must_be_an_object(tmp_path):
    _fixture(tmp_path)
    (tmp_path / "control/plan.json").write_text("[]")
    with pytest.raises(ValueError, match="JSON object"):
        _check(tmp_path)


def test_prefix_cannot_write_sibling_files_outside_project(tmp_path):
    _fixture(tmp_path)
    path = tmp_path / "control/pack.sh"
    path.write_text(path.read_text().replace("'tmp/results/review set'", "."))
    plan_path = tmp_path / "control/plan.json"
    plan = json.loads(plan_path.read_text())
    plan["control_input_sha256"]["control/pack.sh"] = hashlib.sha256(
        path.read_bytes()
    ).hexdigest()
    plan_path.write_text(json.dumps(plan))
    with pytest.raises(ValueError, match="below the project root"):
        _check(tmp_path)


def test_readonly_cli_checks_the_prepared_fixture(tmp_path):
    _fixture(tmp_path)
    script = (
        Path(__file__).parents[2] / "scripts/check_sceneexpert_transport_preparation.py"
    )
    result = subprocess.run(
        [
            sys.executable, str(script), "--project-root", str(tmp_path),
            "--plan", "control/plan.json", "--entrypoint", "control/pack.sh",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    proof = json.loads(result.stdout)
    assert proof["passed"] and proof["packer_imported"] is False
    assert not (tmp_path / "tmp").exists()
