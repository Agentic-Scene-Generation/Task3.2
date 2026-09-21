"""CPU-only smoke check of real front readers and isolated runtime functions.

This does NOT validate importing/running SceneSmith with Blender or Drake.
Functions are compiled unchanged from the actual source AST to avoid importing
unavailable graphics dependencies. Data readers and hint builder run in full.
"""
from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]


def load(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    reader = load(
        "scenesmith.scenebenchmark_critic.current_asset_annotations",
        "scenesmith/scenebenchmark_critic/current_asset_annotations.py",
    )
    annotations = load(
        "merge_annotation_reader",
        "scenesmith/scenebenchmark_critic/asset_library_annotations.py",
    )
    namespace = {
        "Path": Path,
        "_VALID_HORIZONTAL_FRONT_AXES": {"+X", "-X", "+Y", "-Y"},
        "_HSSD_METADATA_ID_KEYS": ("hssd_mesh_id", "asset_id", "object_id"),
        "get_current_asset_annotation": reader.get_current_asset_annotation,
        "get_hssd_asset_annotations": annotations.get_hssd_asset_annotations,
        "build_scenebenchmark_annotation": annotations.build_scenebenchmark_annotation,
    }
    for relative, names in [
        ("scenesmith/agent_utils/asset_manager.py", {
            "_get_scenebenchmark_front_axis_annotation_record",
            "_normalize_axis_string", "_hsm_vector_to_blender_front_axis",
            "_normalize_hssd_annotation_front_axis",
        }),
        ("scenesmith/scenebenchmark_critic/adapter.py", {
            "_hssd_annotation_functional_hints", "_hssd_asset_id_from_metadata",
            "normalize_hssd_id_for_adapter",
        }),
    ]:
        tree = ast.parse((ROOT / relative).read_text())
        nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
        assert {n.name for n in nodes} == names
        isolated = "from __future__ import annotations\n" + "\n".join(ast.unparse(n) for n in nodes)
        exec(compile(isolated, relative, "exec"), namespace)
    uid = "3dfuture:00033fdb-e9f8-4abe-aa9b-878761356d35"
    record = namespace["_get_scenebenchmark_front_axis_annotation_record"](uid)
    front = record["canonical_front"]
    assert front["canonical_orientation_is_semantic_front"] is True
    assert namespace["_normalize_hssd_annotation_front_axis"](front["asset_local_front_axis"]) == "-Y"
    hints = namespace["_hssd_annotation_functional_hints"](SimpleNamespace(metadata={"hssd_mesh_id": uid}))
    assert hints["asset_annotation_source"] == "3dfuture_annotations", hints["asset_annotation_source"]
    assert hints["classification_source"] == "3dfuture_annotations"
    assert hints["asset_annotation_asset_uid"] == uid
    assert hints["canonical_orientation_is_semantic_front"] is True
    assert hints["asset_local_front_axis"] == [0., 0., 1.]
    data = ROOT / "scenesmith/scenebenchmark_critic/asset_annotation_data"
    manifest = json.loads((data / "FRONT_REQUIRED_20260921.json").read_text())
    for filename, digest in manifest["sha256"].items():
        assert hashlib.sha256((data / filename).read_bytes()).hexdigest() == digest, filename
    assert hashlib.sha256((data / "canonical_front_required.json.gz").read_bytes()).hexdigest() == manifest["front_overlay_sha256"]
    print(json.dumps({"passed": True, "front_assets_preserved": manifest["total"],
        "sample": uid, "asset_local_axis": [0, 0, 1], "runtime_axis": "-Y",
        "provenance": hints["asset_annotation_source"],
        "scope": "real data readers and isolated actual functions; not full runtime import"}))


if __name__ == "__main__":
    main()
