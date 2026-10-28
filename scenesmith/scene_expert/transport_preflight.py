"""Read-only binding checks before allocating resources for exact transport."""

from __future__ import annotations

import hashlib
import json
import re
import shlex
from pathlib import Path
from typing import Any

_COMMAND = "scripts/package_sceneexpert_exact_transport.py"
_MODULE = "scenesmith/scene_expert/transport_media.py"
_VALUE_OPTIONS = {
    "--packer", "--packer-sha256", "--plan", "--prefix", "--project-root"
}


def _literal_path(root: Path, value: str) -> Path:
    if not value or any(c in value for c in ("$", "`", "~", "\n", "\r")):
        raise ValueError("Transport paths must be literal, without shell expansion")
    path = (root / value).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Transport paths must remain inside the project root")
    return path


def _pack_options(text: str) -> dict[str, str]:
    lines = text.replace("\r\n", "\n").replace("\\\n", "").splitlines()
    commands = []
    for line in lines:
        tokens = shlex.split(line, comments=True, posix=True)
        if tokens and tokens[0] == "exec":
            commands.append(tokens)
    if len(commands) != 1:
        raise ValueError("Exactly one literal exec packaging command is required")
    command = commands[0]
    if len(command) < 3 or command[1] not in {"python", "python3", "python3.11"}:
        raise ValueError("The packaging command must use an explicit Python launcher")
    if command[2] != _COMMAND:
        raise ValueError("The packaging command must use the exact-transport entry point")
    options: dict[str, str] = {}
    index = 3
    while index < len(command):
        option = command[index]
        if option == "--include-postpack":
            if option in options:
                raise ValueError("Duplicate transport option")
            options[option] = "true"
            index += 1
            continue
        if option not in _VALUE_OPTIONS or option in options:
            raise ValueError("Unsupported or duplicate transport option")
        if index + 1 == len(command) or command[index + 1].startswith("--"):
            raise ValueError("Missing transport option value")
        options[option] = command[index + 1]
        index += 2
    if not {"--packer", "--packer-sha256", "--plan", "--prefix"} <= options.keys():
        raise ValueError("The packaging command lacks a required binding")
    return options


def check_transport_preparation(
    project_root: Path, plan_path: Path, entrypoint_path: Path
) -> dict[str, Any]:
    """Cross-check the frozen plan, wrapper, packer and CLI pin without imports.

    Only the literal packaging command and its four source bindings are checked.
    Other shell statements are never evaluated; this is not a shell-safety audit
    or a substitute for the full runtime/model/resource preflight.
    """
    root = project_root.resolve()
    plan_file = _literal_path(root, str(plan_path))
    entrypoint = _literal_path(root, str(entrypoint_path))
    plan_raw = plan_file.read_bytes()
    plan = json.loads(plan_raw)
    if not isinstance(plan, dict):
        raise ValueError("The transport plan must be a JSON object")
    controls = plan.get("control_input_sha256")
    if not isinstance(controls, dict):
        raise ValueError("The plan must bind control_input_sha256")
    options = _pack_options(entrypoint.read_text(encoding="utf-8"))
    if _literal_path(root, options["--plan"]) != plan_file:
        raise ValueError("The wrapper selects a different plan")
    if (
        "--project-root" in options
        and _literal_path(root, options["--project-root"]) != root
    ):
        raise ValueError("The wrapper selects a different project root")
    packer = _literal_path(root, options["--packer"])
    prefix = _literal_path(root, options["--prefix"])
    if prefix == root:
        raise ValueError("The transport output prefix must be below the project root")
    pin = options["--packer-sha256"]
    if not re.fullmatch(r"[0-9a-f]{64}", pin):
        raise ValueError("A full lowercase packer SHA256 is required")
    checked = {}
    for path in (entrypoint, packer, root / _COMMAND, root / _MODULE):
        name = path.relative_to(root).as_posix()
        expected = controls.get(name)
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if expected != actual:
            raise ValueError(f"Plan/source SHA256 mismatch for {name}")
        checked[name] = actual
    if checked[packer.relative_to(root).as_posix()] != pin:
        raise ValueError("Wrapper packer SHA256 pin differs from the frozen plan/source")
    return {
        "passed": True,
        "plan_path": plan_file.relative_to(root).as_posix(),
        "plan_sha256": hashlib.sha256(plan_raw).hexdigest(),
        "entrypoint_path": entrypoint.relative_to(root).as_posix(),
        "packer_sha256": pin,
        "output_prefix": prefix.relative_to(root).as_posix(),
        "checked_source_sha256": checked,
        "packer_imported": False,
        "shell_executed": False,
        "output_created": False,
    }
