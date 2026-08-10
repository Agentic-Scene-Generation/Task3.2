import importlib.metadata
import unittest

from pathlib import Path
from types import SimpleNamespace

from scripts.check_runtime_compatibility import check_runtime_compatibility


class _Usage:
    def __init__(self):
        self.requests = 0


def _version_reader(versions):
    def read(package):
        if package not in versions:
            raise importlib.metadata.PackageNotFoundError(package)
        return versions[package]

    return read


class RuntimeCompatibilityTest(unittest.TestCase):
    def test_bootstrap_uses_resumable_official_vllm_wheel_cache(self):
        project_root = Path(__file__).resolve().parents[2]
        bootstrap = (project_root / "scripts" / "bootstrap_vllm_runtime.sh").read_text(
            encoding="utf-8"
        )

        self.assertIn("https://wheels.vllm.ai/", bootstrap)
        self.assertIn("--continue-at -", bootstrap)
        self.assertIn("verify_wheel_sha256", bootstrap)
        self.assertIn("SCENEEXPERT_VLLM_WHEEL_CACHE", bootstrap)
        self.assertNotIn("github.com/vllm-project/vllm/releases/download", bootstrap)

    def test_compatible_contract_passes(self):
        modules = {
            "openai.types.responses": SimpleNamespace(NamespaceTool=object),
            "agents.usage": SimpleNamespace(Usage=_Usage),
        }

        report = check_runtime_compatibility(
            importer=modules.__getitem__,
            version_reader=_version_reader(
                {"openai": "2.44.0", "openai-agents": "0.6.4", "vllm": "0.22.0"}
            ),
        )

        self.assertTrue(report.ok)
        self.assertEqual(report.errors, ())

    def test_missing_namespace_tool_reports_vllm_boundary(self):
        modules = {
            "openai.types.responses": SimpleNamespace(),
            "agents.usage": SimpleNamespace(Usage=_Usage),
        }

        report = check_runtime_compatibility(
            importer=modules.__getitem__,
            version_reader=_version_reader(
                {"openai": "2.11.0", "openai-agents": "0.6.4"}
            ),
        )

        self.assertFalse(report.ok)
        self.assertIn("NamespaceTool is missing", report.errors[0])
        self.assertNotIn("vllm", report.versions)

    def test_agents_usage_schema_error_is_reported(self):
        class BrokenUsage:
            def __init__(self):
                raise ValueError("cache_write_tokens field required")

        modules = {
            "openai.types.responses": SimpleNamespace(NamespaceTool=object),
            "agents.usage": SimpleNamespace(Usage=BrokenUsage),
        }

        report = check_runtime_compatibility(
            importer=modules.__getitem__,
            version_reader=_version_reader(
                {"openai": "2.45.0", "openai-agents": "0.6.4", "vllm": "0.22.0"}
            ),
        )

        self.assertFalse(report.ok)
        self.assertIn("cache_write_tokens field required", report.errors[0])

    def test_server_contract_imports_native_cuda_boundary(self):
        modules = {
            "vllm": SimpleNamespace(),
            "vllm.platforms.cuda": SimpleNamespace(),
            "torch": SimpleNamespace(version=SimpleNamespace(cuda="12.9")),
        }

        report = check_runtime_compatibility(
            importer=modules.__getitem__,
            version_reader=_version_reader(
                {"vllm": "0.22.1+cu129", "torch": "2.10.0"}
            ),
            check_client=False,
            check_vllm_native=True,
            expected_vllm_version="0.22.1",
            expected_torch_backend="cu129",
        )

        self.assertTrue(report.ok)

    def test_server_contract_rejects_default_cuda_wheel_for_cu129(self):
        modules = {
            "vllm": SimpleNamespace(),
            "vllm.platforms.cuda": SimpleNamespace(),
            "torch": SimpleNamespace(version=SimpleNamespace(cuda="12.9")),
        }

        report = check_runtime_compatibility(
            importer=modules.__getitem__,
            version_reader=_version_reader(
                {"vllm": "0.22.1", "torch": "2.11.0+cu129"}
            ),
            check_client=False,
            check_vllm_native=True,
            expected_vllm_version="0.22.1",
            expected_torch_backend="cu129",
        )

        self.assertFalse(report.ok)
        self.assertTrue(
            any("expected=0.22.1+cu129" in error for error in report.errors)
        )

    def test_server_contract_rejects_wrong_vllm_and_missing_cuda_abi(self):
        def importer(module):
            if module == "vllm":
                return SimpleNamespace()
            if module == "vllm.platforms.cuda":
                raise ImportError("libcudart.so.13: cannot open shared object file")
            if module == "torch":
                return SimpleNamespace(version=SimpleNamespace(cuda="12.4"))
            raise KeyError(module)

        report = check_runtime_compatibility(
            importer=importer,
            version_reader=_version_reader({"vllm": "0.25.1", "torch": "2.5.1"}),
            check_client=False,
            check_vllm_native=True,
            expected_vllm_version="0.22.1",
            expected_torch_backend="cu129",
        )

        self.assertFalse(report.ok)
        self.assertTrue(
            any("version contract mismatch" in error for error in report.errors)
        )
        self.assertTrue(any("libcudart.so.13" in error for error in report.errors))
        self.assertTrue(
            any(
                "Torch CUDA backend contract mismatch" in error
                for error in report.errors
            )
        )


if __name__ == "__main__":
    unittest.main()
