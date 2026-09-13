"""Exercise ACP exit propagation and packaging using stubs, without GPU work."""

from __future__ import annotations

import os
import shutil
import subprocess

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _bash() -> str:
    git_bash = Path("C:/Program Files/Git/bin/bash.exe")
    if os.name == "nt" and git_bash.is_file():
        return str(git_bash)
    candidate = shutil.which("bash")
    if candidate:
        return candidate
    pytest.skip("Bash is required for launcher orchestration tests")


def _path(path: Path) -> str:
    value = path.resolve().as_posix()
    if os.name == "nt":
        return f"/{value[0].lower()}{value[2:]}"
    return value


def _script(path: Path, text: str) -> None:
    path.write_text(
        "#!/usr/bin/env bash\nset -eu\n" + text, encoding="utf-8", newline="\n"
    )
    path.chmod(0o755)


@pytest.mark.parametrize("generation_exit,audit_exit", [(0, 0), (0, 2), (7, 2)])
def test_wrapper_packages_audit_even_after_failure_and_preserves_exit_status(
    tmp_path: Path, generation_exit: int, audit_exit: int
) -> None:
    python_stub = tmp_path / "python-stub"
    _script(
        python_stub,
        """if [[ "$1" == "-c" ]]; then exit 0; fi
while [[ $# -gt 0 ]]; do
  case "$1" in
    --output-dir) output_dir="$2"; shift 2 ;;
    *) shift ;;
  esac
done
printf '%s\\n' '{"stub_audit_ran":true}' > "$output_dir/collection_audit.json"
exit "$STUB_AUDIT_EXIT"
""",
    )
    full_stub = tmp_path / "full-stub"
    _script(
        full_stub,
        """printf '%s\\n' "$ACP_ENTRYPOINT" > "$OUTPUT_ROOT/entrypoint.txt"
exit "$STUB_GENERATION_EXIT"
""",
    )
    batch = tmp_path / "new-batch"
    command = [_bash(), _path(ROOT / "tmp/acp/acp_qwen38_slow_memory_recollect.sh")]
    env = {
        **os.environ,
        "PROJECT_ROOT": _path(ROOT),
        "FULL_LAUNCHER": _path(full_stub),
        "PYTHON_BIN": _path(python_stub),
        "COLLECTION_ROOT": _path(batch),
        "OUTPUT_ROOT": _path(batch / "runs"),
        "ACP_PARALLELISM": "1",
        "MODEL_NAME": "unsloth/Qwen3.8-27B-GGUF",
        "MIN_DPO_PAIRS": "0",
        "STUB_AUDIT_EXIT": str(audit_exit),
        "STUB_GENERATION_EXIT": str(generation_exit),
    }
    result = subprocess.run(command, env=env, capture_output=True, text=True)
    assert result.returncode == (generation_exit or audit_exit), result.stderr
    review = batch / "runs/collection"
    assert (review / "collection_audit.json").is_file()
    assert (review / "collection_manifest.env").is_file()
    assert (review / "collection_entrypoint.sh").is_file()
    status = (review / "collection_exit_status.env").read_text(encoding="utf-8")
    assert f"exit_code={generation_exit or audit_exit}\n" in status
    assert (
        "acp_qwen38_slow_memory_recollect.sh"
        in (batch / "runs/entrypoint.txt").read_text()
    )
    # A repeated launch may not mix or overwrite this batch.
    retry = subprocess.run(command, env=env, capture_output=True, text=True)
    assert retry.returncode == 2
    assert "already exists" in retry.stderr
    assert (review / "collection_exit_status.env").read_text(encoding="utf-8") == status


def test_full_launcher_preserves_outer_entrypoint(tmp_path: Path) -> None:
    base_stub = tmp_path / "base-stub"
    _script(base_stub, 'printf "%s\\n" "$ACP_ENTRYPOINT" "$SCENEEXPERT_EXPERIMENT"\n')
    result = subprocess.run(
        [_bash(), _path(ROOT / "tmp/acp/acp_qwen38_full_generate.sh")],
        env={
            **os.environ,
            "BASE_ACP_SCRIPT": _path(base_stub),
            "ACP_ENTRYPOINT": "outer-wrapper",
        },
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == ["outer-wrapper", "ablation_5_qwen3_full"]
