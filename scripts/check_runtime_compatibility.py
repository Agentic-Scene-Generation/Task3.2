"""Fail-fast checks for the vLLM/OpenAI/Agents SDK compatibility contract.

This script intentionally performs no network or model calls.  It exercises the
two import/data-model boundaries that have broken ACP runs in the past, before
vLLM allocates GPU memory or SceneSmith starts per-scene worker processes.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import sys

from dataclasses import dataclass
from typing import Any, Callable


RECOMMENDED_OPENAI_VERSION = "2.44.0"
RECOMMENDED_AGENTS_VERSION = "0.6.4"
REPAIR_COMMAND = (
    "python -m pip install --upgrade "
    f"'openai=={RECOMMENDED_OPENAI_VERSION}' "
    f"'openai-agents=={RECOMMENDED_AGENTS_VERSION}'"
)


@dataclass(frozen=True)
class CompatibilityReport:
    """Result of checking the runtime packages used by ACP."""

    versions: dict[str, str]
    errors: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.errors


def _package_version(
    package: str,
    version_reader: Callable[[str], str],
) -> str:
    try:
        return version_reader(package)
    except importlib.metadata.PackageNotFoundError:
        return "missing"
    except Exception as exc:  # pragma: no cover - defensive metadata fallback
        return f"unknown ({type(exc).__name__}: {exc})"


def check_runtime_compatibility(
    *,
    importer: Callable[[str], Any] = importlib.import_module,
    version_reader: Callable[[str], str] = importlib.metadata.version,
) -> CompatibilityReport:
    """Check both vLLM's and Agents SDK's OpenAI SDK expectations."""

    versions = {
        "openai": _package_version("openai", version_reader),
        "openai-agents": _package_version("openai-agents", version_reader),
        "vllm": _package_version("vllm", version_reader),
    }
    errors: list[str] = []

    try:
        responses_types = importer("openai.types.responses")
        if not hasattr(responses_types, "NamespaceTool"):
            errors.append(
                "openai.types.responses.NamespaceTool is missing; "
                "vLLM 0.22.x cannot import its tool parser."
            )
    except Exception as exc:
        errors.append(
            "cannot import openai.types.responses: "
            f"{type(exc).__name__}: {exc}"
        )

    try:
        usage_module = importer("agents.usage")
        usage_module.Usage()
    except Exception as exc:
        errors.append(
            "OpenAI Agents SDK cannot construct Usage(): "
            f"{type(exc).__name__}: {exc}"
        )

    return CompatibilityReport(versions=versions, errors=tuple(errors))


def main() -> int:
    report = check_runtime_compatibility()
    version_text = ", ".join(
        f"{package}={version}" for package, version in report.versions.items()
    )
    print(f"  Runtime dependency versions: {version_text}")

    if report.ok:
        print("  Runtime compatibility preflight passed")
        return 0

    print("ERROR: incompatible Python runtime dependencies detected.", file=sys.stderr)
    for error in report.errors:
        print(f"  - {error}", file=sys.stderr)
    print("Repair the active virtual environment with:", file=sys.stderr)
    print(f"  {REPAIR_COMMAND}", file=sys.stderr)
    print("Then rerun this preflight before submitting ACP:", file=sys.stderr)
    print("  python scripts/check_runtime_compatibility.py", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
