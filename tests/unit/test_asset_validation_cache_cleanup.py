import json
import tempfile
import unittest

from pathlib import Path

from scenesmith.agent_utils.asset_runtime import ASSET_SEMANTIC_CONTRACT_VERSION
from scenesmith.agent_utils.asset_structure import ASSET_STRUCTURE_CONTRACT_VERSION
from scripts.clean_asset_validation_cache import clean_asset_validation_cache


class AssetValidationCacheCleanupTest(unittest.TestCase):
    def test_archives_stale_and_quarantined_entries_but_keeps_valid_v8(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            valid_path = root / "valid.json"
            stale_path = root / "stale.json"
            quarantined_path = root / "quarantined.json"
            structural = {
                "status": "pass",
                "contract_version": ASSET_STRUCTURE_CONTRACT_VERSION,
                "geometry_fingerprint": "f" * 64,
            }
            valid_path.write_text(
                json.dumps(
                    {
                        "schema_version": ASSET_SEMANTIC_CONTRACT_VERSION,
                        "candidate_id": "standalone_bed",
                        "structural_check": structural,
                    }
                ),
                encoding="utf-8",
            )
            stale_path.write_text(
                json.dumps(
                    {
                        "schema_version": "7.0",
                        "candidate_id": "legacy_bed",
                    }
                ),
                encoding="utf-8",
            )
            quarantined_path.write_text(
                json.dumps(
                    {
                        "schema_version": ASSET_SEMANTIC_CONTRACT_VERSION,
                        "candidate_id": ("02e22ed56270338bef5f8436e82023c36cd29104"),
                        "structural_check": structural,
                    }
                ),
                encoding="utf-8",
            )

            dry_run = clean_asset_validation_cache(cache_dir=root)
            applied = clean_asset_validation_cache(cache_dir=root, apply=True)

            self.assertEqual(2, dry_run["stale_count"])
            self.assertEqual("APPLIED", applied["status"])
            self.assertTrue(valid_path.exists())
            self.assertFalse(stale_path.exists())
            self.assertFalse(quarantined_path.exists())
            self.assertTrue(Path(applied["report_path"]).exists())


if __name__ == "__main__":
    unittest.main()
