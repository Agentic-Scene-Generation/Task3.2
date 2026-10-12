#!/usr/bin/env python3
"""Resolve the collection runtime's CUDA 13 libraries without starting a model."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import sysconfig
from pathlib import Path

REQUIRED_LIBRARIES = ("libcudart.so.13", "libcublas.so.13", "libcublasLt.so.13")


def resolve_runtime(project_root: Path, explicit: str = "") -> tuple[Path, list[dict]]:
    """Respect explicit paths; otherwise inspect only project/interpreter runtimes."""
    candidates = (
        [Path(explicit)]
        if explicit
        else [
            *sorted((project_root / ".venv").glob("lib/python*/site-packages/nvidia/cu13/lib")),
            *sorted((project_root / ".venv_dpo").glob("lib/python*/site-packages/nvidia/cu13/lib")),
            Path(sysconfig.get_path("purelib")) / "nvidia/cu13/lib",
        ]
    )
    inspected: list[dict] = []
    for candidate in dict.fromkeys(candidates):
        missing = [name for name in REQUIRED_LIBRARIES if not (candidate / name).is_file()]
        inspected.append({"path": str(candidate), "missing_libraries": missing})
        if not missing:
            return candidate.resolve(), inspected
    raise ValueError(
        "No complete CUDA 13 runtime: " + json.dumps(inspected)
        + ". Set CUDA13_LIB_DIR to a directory containing the required SONAMEs."
    )


def check_binary(binary: Path, runtime: Path) -> dict:
    """Check native linkage on CPU, retaining diagnostics and missing dependencies."""
    if not binary.is_file() or not os.access(binary, os.X_OK):
        raise ValueError(f"llama-server is not executable: {binary}")
    search = [str(runtime), str(binary.parent), str(binary.parent / "build/bin")]
    inherited = os.environ.get("LD_LIBRARY_PATH", "")
    if inherited:
        search.append(inherited)
    result = subprocess.run(
        ["ldd", str(binary)],
        env={**os.environ, "LD_LIBRARY_PATH": ":".join(search)},
        capture_output=True, text=True, timeout=30, check=False,
    )
    diagnostics = result.stdout + result.stderr
    if result.returncode or "not found" in diagnostics:
        raise ValueError(f"Unresolved native dependencies for {binary}: {diagnostics}")
    return {"binary": str(binary), "return_code": result.returncode, "ldd": diagnostics}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--cuda13-lib-dir", default="")
    parser.add_argument("--binary", type=Path, action="append", default=[])
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    report = {
        "schema_version": "sceneexpert.cuda_runtime_preflight.v1",
        "scope": "cpu_native_dependencies_only",
        "required_libraries": list(REQUIRED_LIBRARIES),
        "explicit_override": bool(args.cuda13_lib_dir),
    }
    try:
        runtime, inspected = resolve_runtime(args.project_root, args.cuda13_lib_dir)
        report.update(cuda13_lib_dir=str(runtime), inspected_candidates=inspected)
        report["binary_checks"] = [check_binary(binary, runtime) for binary in args.binary]
        report["status"] = "passed"
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        report.update(status="failed", error=str(exc))
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if report["status"] != "passed":
        sys.stderr.write(report["error"] + "\n")
        return 2
    sys.stdout.write(report["cuda13_lib_dir"] + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
