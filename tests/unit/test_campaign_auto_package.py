"""Exercise the actual Bash workflow with a CPU-only simulated collector."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from scenesmith.scene_expert.review_package import verify_package

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    "collection_exit,package_failure", [(0, False), (2, False), (0, True)]
)
def test_campaign_packages_success_and_failure_without_masking_exit(
    tmp_path: Path, collection_exit: int, package_failure: bool
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

    for name in (
        "scripts/package_sceneexpert_results.py",
        "scenesmith/scene_expert/review_package.py",
    ):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((ROOT / name).read_bytes())
    (tmp_path / "scripts/prepare_sceneexpert_campaign.py").write_text(
        "print('prepared')\n"
    )
    (tmp_path / "scripts/run_sceneexpert_campaign.py").write_text(
        "from pathlib import Path\n"
        "import os\n"
        "root=Path('outputs/slow_memory/test_campaign')\n"
        "root.mkdir(parents=True)\n"
        "(root/'campaign.json').write_text('{}')\n"
        "(root/'campaign_audit.json').write_text('{}')\n"
        "code=int(os.environ['TEST_COLLECTION_EXIT'])\n"
        "(root/'campaign_exit_status.env').write_text(f'exit_code={code}\\n')\n"
        "print('collector closed')\n"
        "raise SystemExit(code)\n"
    )
    archive = tmp_path / "tmp/results/slow_memory/review.tar.gz"
    if package_failure:
        archive.parent.mkdir(parents=True)
        archive.write_bytes(b"previous package must be preserved")
    result = subprocess.run(
        [bash, shell_path(ROOT / "tmp/acp/acp_qwen38_dpo_campaign.sh")],
        env={
            **os.environ,
            "PROJECT_ROOT": shell_path(tmp_path),
            "PYTHON_BIN": shell_path(Path(sys.executable)),
            "RUN_ID": "test_campaign",
            "PACKAGE_PATH": shell_path(archive),
            "TEST_COLLECTION_EXIT": str(collection_exit),
        },
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == (
        2 if package_failure else collection_exit
    ), result.stderr
    status = next(
        (tmp_path / "tmp/acp_logs/test_campaign").glob("*/workflow_exit_status.env")
    ).read_text()
    assert f"collection_exit_code={collection_exit}" in status
    assert f"package_exit_code={2 if package_failure else 0}" in status
    if package_failure:
        assert archive.read_bytes() == b"previous package must be preserved"
    else:
        assert verify_package(archive)["integrity_verified"] is True
        assert archive.with_name(archive.name + ".sha256").is_file()
