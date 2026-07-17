import importlib.metadata
import unittest

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
        self.assertEqual(report.versions["vllm"], "missing")

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


if __name__ == "__main__":
    unittest.main()
