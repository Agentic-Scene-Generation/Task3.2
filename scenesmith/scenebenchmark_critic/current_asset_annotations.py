"""Read the synchronized existing annotation release; no invented missing fields."""
from __future__ import annotations

from functools import lru_cache
import gzip
import json
from pathlib import Path
from typing import Any

DATA_ROOT = Path(__file__).parent / "asset_annotation_data"

@lru_cache(maxsize=4)
def _source(source: str) -> dict[str, Any]:
    files = {
        "hssd": "hssd_annotation_lookup.json.gz",
        "3dfuture": "3dfuture/3dfuture_annotation_lookup.json.gz",
        "others": "external/others_annotation_lookup.json.gz",
        "generated": "generated_annotations.json.gz",
    }
    with gzip.open(DATA_ROOT / files[source], "rt", encoding="utf-8") as stream:
        return json.load(stream)

def get_current_asset_annotation(asset_uid: str) -> dict[str, Any]:
    """Return an existing record by UID; generated records explicitly remain incomplete."""
    source = asset_uid.split(":", 1)[0] if ":" in asset_uid else "hssd"
    rows = _source(source)
    key = asset_uid.split(":", 1)[-1] if source in {"hssd", "generated"} else asset_uid
    record = dict(rows[key])
    # Non-generated style history is published as an authoritative overlay;
    # do not expose stale style fields embedded in the older source lookup.
    if source != "generated":
        overlay = _style_overlay()
        normalized_uid = asset_uid if ":" in asset_uid else "hssd:" + asset_uid
        style = overlay.get(normalized_uid, overlay.get(key))
        if style is not None:
            record["asset_style"] = style
    return record

@lru_cache(maxsize=1)
def _style_overlay() -> dict[str, Any]:
    with gzip.open(DATA_ROOT / "asset_style_annotations_v2.shared.json.gz", "rt", encoding="utf-8") as stream:
        overlay = json.load(stream)
    return overlay.get("records", overlay)
