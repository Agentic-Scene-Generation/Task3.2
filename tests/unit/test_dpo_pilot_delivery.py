"""Pilot capacity selection and real Bash finalization on success and failure."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import signal
import time
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
    "preflight_exit,training_exit,package_failure,incomplete",
    [(0, 0, False, False), (2, 0, False, False), (0, 1, False, False),
     (0, 3, False, False), (0, 0, True, False), (0, 0, False, True)],
)
def test_training_finalization_preserves_failures_and_verifies_package(
    tmp_path, preflight_exit, training_exit, package_failure, incomplete,
):
    git_bash = Path("C:/Program Files/Git/bin/bash.exe")
    bash = str(git_bash) if os.name == "nt" and git_bash.exists() else shutil.which("bash")
    if not bash:
        pytest.skip("Bash is required")

    def shell_path(path):
        value = path.resolve().as_posix()
        return f"/{value[0].lower()}{value[2:]}" if os.name == "nt" else value

    for name in ("scripts/package_sceneexpert_results.py", "scenesmith/scene_expert/review_package.py",
                 "scripts/supervise_sceneexpert_dpo.py", "scenesmith/scene_expert/slow_memory/training_lifecycle.py"):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((ROOT / name).read_bytes())
    (tmp_path / "scripts/train_sceneexpert_dpo.py").write_text(
        "import os,sys,json\nfrom pathlib import Path\n"
        "dry='--dry-run' in sys.argv\n"
        "if not dry and os.environ['INCOMPLETE'] != '1':\n"
        " p=Path('outputs/slow_memory/pilot')\n"
        " m={'execution_id':os.environ['SCENEEXPERT_TRAIN_EXECUTION_ID'],"
        "'completion':{'completed':True,'optimizer_steps':8,'expected_optimizer_steps':8},"
        "'train_metrics':{'train_loss':0.6,'nonzero_lora_b_tensors':256},"
        "'evaluation_metrics':{'eval_loss':0.65}}\n"
        " (p/'training_manifest.json').write_text(json.dumps(m))\n"
        " (p/'preflight.json').write_text(json.dumps({'dataset_validation':{'split_counts':{'validation':7}}}))\n"
        " (p/'adapter').mkdir()\n"
        " for n in ['train_results.json','eval_results.json','adapter/adapter_config.json','adapter/adapter_model.safetensors']:\n"
        "  (p/n).write_text('fixture')\n"
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
             "PREFLIGHT_EXIT": str(preflight_exit), "TRAIN_EXIT": str(training_exit),
             "INCOMPLETE": '1' if incomplete else '0'},
        capture_output=True, text=True, check=False,
    )
    effective_exit = preflight_exit or training_exit or (4 if incomplete else 0)
    expected = effective_exit or (2 if package_failure else 0)
    assert result.returncode == expected, result.stderr
    output = tmp_path / "outputs/slow_memory/pilot"
    assert (output / "exit_status.env").read_text().strip() == f"exit_code={effective_exit}"
    assert f"package_exit_code={2 if package_failure else 0}" in (output / "workflow_exit_status.env").read_text()
    if package_failure:
        assert archive.read_bytes() == b"preserve existing package"
    else:
        assert verify_package(archive)["integrity_verified"]
    if preflight_exit:
        assert not (output / "training_manifest.json").exists()


@pytest.mark.skipif(os.name == "nt", reason="POSIX process-group termination contract")
def test_supervisor_records_sigterm_and_stops_training_child(tmp_path):
    command = [sys.executable, str(ROOT / 'scripts/supervise_sceneexpert_dpo.py'),
               '--output-dir', str(tmp_path), '--', sys.executable, '-u', '-c',
               'import time; print("training started", flush=True); time.sleep(120)']
    process = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    try:
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            if (tmp_path / 'train.log').exists() and 'training started' in (tmp_path / 'train.log').read_text():
                break
            time.sleep(0.05)
        else:
            pytest.fail('child did not start')
        process.send_signal(signal.SIGTERM)
        assert process.wait(timeout=10) == 143
        status = json.loads((tmp_path / 'training_supervisor.json').read_text())
        assert status['received_signal'] == signal.SIGTERM
        assert status['process_returncode'] == -signal.SIGTERM
        assert status['completion_check']['completed'] is False
        with pytest.raises(ProcessLookupError):
            os.kill(status['pid'], 0)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
