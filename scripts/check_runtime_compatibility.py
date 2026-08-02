"""Fail-fast checks for SceneExpert's client and vLLM server runtimes.

The SceneSmith process and the vLLM server intentionally have independent
Python dependency contracts.  This script exercises the real import boundaries
for either side without making a network call or loading model weights.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import os
import re
import sys

from dataclasses import dataclass
from typing import Any, Callable


RECOMMENDED_OPENAI_VERSION = "2.44.0"
RECOMMENDED_AGENTS_VERSION = "0.6.4"
RECOMMENDED_VLLM_VERSION = "0.22.1"
CLIENT_REPAIR_COMMAND = (
    "python -m pip install --upgrade "
    f"'openai=={RECOMMENDED_OPENAI_VERSION}' "
    f"'openai-agents=={RECOMMENDED_AGENTS_VERSION}'"
)
SERVER_REPAIR_COMMAND = (
    "SCENEEXPERT_VLLM_FORCE_REBUILD=1 bash scripts/bootstrap_vllm_runtime.sh"
)


def _expected_vllm_distribution_version(
    version: str,
    torch_backend: str | None,
) -> str:
    """Return the binary distribution version required by an ABI contract."""

    if "+" in version or not torch_backend:
        return version
    # vLLM 0.22.1 publishes its non-default CUDA 12.9 binary as a PEP 440
    # local-version wheel. The unqualified 0.22.1 distribution is CUDA 13.
    if version == "0.22.1" and torch_backend == "cu129":
        return f"{version}+{torch_backend}"
    return version


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
    check_client: bool = True,
    check_vllm_native: bool = False,
    expected_vllm_version: str | None = None,
    expected_torch_backend: str | None = None,
) -> CompatibilityReport:
    """Check requested client/server boundaries in the active Python runtime."""

    versions: dict[str, str] = {}
    if check_client:
        versions.update(
            {
                "openai": _package_version("openai", version_reader),
                "openai-agents": _package_version("openai-agents", version_reader),
            }
        )
    if check_vllm_native:
        versions.update(
            {
                "vllm": _package_version("vllm", version_reader),
                "torch": _package_version("torch", version_reader),
            }
        )
    errors: list[str] = []

    if check_client:
        try:
            responses_types = importer("openai.types.responses")
            if not hasattr(responses_types, "NamespaceTool"):
                errors.append(
                    "openai.types.responses.NamespaceTool is missing; "
                    "the configured vLLM tool parser cannot use this OpenAI SDK."
                )
        except Exception as exc:
            errors.append(
                f"cannot import openai.types.responses: {type(exc).__name__}: {exc}"
            )

        try:
            usage_module = importer("agents.usage")
            usage_module.Usage()
        except Exception as exc:
            errors.append(
                "OpenAI Agents SDK cannot construct Usage(): "
                f"{type(exc).__name__}: {exc}"
            )

    if check_vllm_native:
        installed_vllm = versions["vllm"]
        required_vllm = _expected_vllm_distribution_version(
            expected_vllm_version or "",
            expected_torch_backend,
        )
        if installed_vllm == "missing":
            errors.append("vLLM is not installed in the selected server runtime.")
        elif required_vllm and installed_vllm != required_vllm:
            errors.append(
                "vLLM version contract mismatch: "
                f"installed={installed_vllm}, expected={required_vllm}. "
                "Do not share the SceneSmith application environment with the "
                "vLLM server environment or mix CUDA wheel variants."
            )

        if installed_vllm != "missing":
            try:
                importer("vllm")
                # Importing the CUDA platform forces the compiled extension and
                # its CUDA libraries to resolve. Metadata-only checks missed the
                # libcudart.so.13 failure seen in ACP.
                importer("vllm.platforms.cuda")
            except Exception as exc:
                detail = f"{type(exc).__name__}: {exc}"
                missing_cuda = re.search(r"libcudart\.so\.\d+", str(exc))
                if missing_cuda:
                    detail += (
                        f"; missing {missing_cuda.group(0)} means the installed "
                        "vLLM wheel and CUDA runtime use different ABIs"
                    )
                errors.append(f"vLLM native CUDA import failed: {detail}")

        try:
            torch_module = importer("torch")
            cuda_version = getattr(getattr(torch_module, "version", None), "cuda", None)
            versions["torch-cuda"] = str(cuda_version or "missing")
            backend_match = re.fullmatch(r"cu(\d{2,3})", expected_torch_backend or "")
            if backend_match and cuda_version:
                digits = backend_match.group(1)
                expected_cuda = f"{int(digits[:-1])}.{digits[-1]}"
                if str(cuda_version) != expected_cuda:
                    errors.append(
                        "Torch CUDA backend contract mismatch: "
                        f"installed={cuda_version}, expected={expected_cuda} "
                        f"({expected_torch_backend})."
                    )
        except Exception as exc:
            errors.append(
                "cannot inspect the vLLM runtime's Torch CUDA backend: "
                f"{type(exc).__name__}: {exc}"
            )

    return CompatibilityReport(versions=versions, errors=tuple(errors))


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scope",
        choices=("client", "server", "all"),
        default="all",
        help="Runtime boundary to validate (default: all).",
    )
    parser.add_argument(
        "--expected-vllm-version",
        default=os.environ.get("SCENEEXPERT_VLLM_VERSION", RECOMMENDED_VLLM_VERSION),
        help="Exact vLLM server version required by the deployment contract.",
    )
    parser.add_argument(
        "--expected-torch-backend",
        default=os.environ.get("SCENEEXPERT_VLLM_TORCH_BACKEND", "cu129"),
        help="Expected uv Torch backend for the vLLM server (default: cu129).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    report = check_runtime_compatibility(
        check_client=args.scope in {"client", "all"},
        check_vllm_native=args.scope in {"server", "all"},
        expected_vllm_version=(
            args.expected_vllm_version if args.scope in {"server", "all"} else None
        ),
        expected_torch_backend=(
            args.expected_torch_backend if args.scope in {"server", "all"} else None
        ),
    )
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
    print("Repair the incompatible runtime with:", file=sys.stderr)
    if args.scope in {"client", "all"}:
        print(f"  client: {CLIENT_REPAIR_COMMAND}", file=sys.stderr)
    if args.scope in {"server", "all"}:
        print(f"  server: {SERVER_REPAIR_COMMAND}", file=sys.stderr)
    print("Then rerun this preflight before submitting ACP:", file=sys.stderr)
    print(
        f"  python scripts/check_runtime_compatibility.py --scope {args.scope}",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
