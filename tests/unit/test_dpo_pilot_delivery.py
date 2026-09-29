"""Pilot capacity selection and real Bash finalization on success and failure."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from scenesmith.scene_expert.review_package import verify_package
from scripts.train_sceneexpert_dpo import _select_capacity_pair

ROOT = Path(__file__).resolve().parents[2]


def test_capacity_probe_selects_longest_pair_without_truncating_or_mutating():
    class Dataset(list):
        def select(self, indices):
            return Dataset(self[i] for i in indices)

    class Processor:
        def apply_chat_template(self, messages, **kwargs):
            assert kwargs["enable_thinking"] is False
            return "".join(messages)

        def encode(self, text, **kwargs):
            return list(text)

    rows = Dataset([
        {"prompt": ["abc"], "chosen": ["a"], "rejected": ["bb"],
         "chat_template_kwargs": {"enable_thinking": False}},
        {"prompt": ["abcdef"], "chosen": ["a"], "rejected": ["bbbb"],
         "chat_template_kwargs": {"enable_thinking": False}},
    ])
    before = json.dumps(rows)
    selected, report = _select_capacity_pair(Processor(), rows)
    assert selected == [rows[1]]
    assert report["source_train_row_index"] == 1
    assert report["padded_text_sequence_tokens"] == 10
    assert json.dumps(rows) == before
    rows[0]["images"] = ["image"]
    with pytest.raises(ValueError, match="image token"):
        _select_capacity_pair(Processor(), rows)


@pytest.mark.parametrize(
    "preflight_exit,training_exit,package_failure",
    [(0, 0, False), (2, 0, False), (0, 1, False), (0, 3, False), (0, 0, True)],
)
def test_training_finalization_preserves_failures_and_verifies_package(
    tmp_path, preflight_exit, training_exit, package_failure,
):
    git_bash = Path("C:/Program Files/Git/bin/bash.exe")
    bash = str(git_bash) if os.name == "nt" and git_bash.exists() else shutil.which("bash")
    if not bash:
        pytest.skip("Bash is required")

    def shell_path(path):
        value = path.resolve().as_posix()
        return f"/{value[0].lower()}{value[2:]}" if os.name == "nt" else value

    for name in ("scripts/package_sceneexpert_results.py", "scenesmith/scene_expert/review_package.py"):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((ROOT / name).read_bytes())
    (tmp_path / "scripts/train_sceneexpert_dpo.py").write_text(
        "import os,sys\nfrom pathlib import Path\n"
        "dry='--dry-run' in sys.argv\n"
        "if not dry:\n"
        " Path('outputs/slow_memory/pilot/training_manifest.json').write_text('{}')\n"
        "print('preflight' if dry else 'training closed')\n"
        "raise SystemExit(int(os.environ['PREFLIGHT_EXIT' if dry else 'TRAIN_EXIT']))\n"
    )
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}")
    archive = tmp_path / "tmp/results/slow_memory/review.tar.gz"
    if package_failure:
        archive.parent.mkdir(parents=True)
        archive.write_bytes(b"preserve existing package")
    result = subprocess.run(
        [bash, shell_path(ROOT / "tmp/acp/acp_qwen38_dpo_train.sh")],
        env={**os.environ,
             "PROJECT_ROOT": shell_path(tmp_path),
             "TRAIN_PYTHON": shell_path(Path(sys.executable)),
             "BASE_MODEL": shell_path(model), "RUN_ID": "pilot",
             "PACKAGE_PATH": shell_path(archive),
             "PREFLIGHT_EXIT": str(preflight_exit), "TRAIN_EXIT": str(training_exit)},
        capture_output=True, text=True, check=False,
    )
    expected = preflight_exit or training_exit or (2 if package_failure else 0)
    assert result.returncode == expected, result.stderr
    output = tmp_path / "outputs/slow_memory/pilot"
    assert (output / "exit_status.env").read_text().strip() == f"exit_code={preflight_exit or training_exit}"
    assert f"package_exit_code={2 if package_failure else 0}" in (output / "workflow_exit_status.env").read_text()
    if package_failure:
        assert archive.read_bytes() == b"preserve existing package"
    else:
        assert verify_package(archive)["integrity_verified"]
    if preflight_exit:
        assert not (output / "training_manifest.json").exists()
