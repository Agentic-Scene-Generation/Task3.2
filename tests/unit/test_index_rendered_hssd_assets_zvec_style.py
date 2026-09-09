from __future__ import annotations

import gzip
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from scripts.index_rendered_hssd_assets_zvec import (
    HssdMeshMetadata,
    LlamaEmbeddingClient,
    RenderedAsset,
    StylePath,
    build_asset_content,
    discover_rendered_assets,
    embed_asset_views,
    load_asset_style_annotations,
)


ASSET_ID = "0123456789abcdef0123456789abcdef01234567"


def _write_style_overlay(path: Path) -> None:
    records = {
        ASSET_ID: {
            "schema_version": "asset_style@2.0",
            "style_status": "source_metadata",
            "styles": [
                {
                    "level_1_id": "traditional",
                    "level_2_id": "european_classic",
                    "evidence_type": "source_metadata",
                },
                {
                    "level_1_id": "transitional",
                    "level_2_id": None,
                    "evidence_type": "source_metadata",
                },
            ],
        },
        "1111111111111111111111111111111111111111": {
            "schema_version": "asset_style@2.0",
            "style_status": "needs_review",
            "styles": [
                {
                    "level_1_id": "modern",
                    "level_2_id": None,
                    "evidence_type": "unreviewed_guess",
                }
            ],
        },
        "2222222222222222222222222222222222222222": {
            "schema_version": "asset_style@2.0",
            "style_status": "visual_reviewed",
            "styles": [
                {
                    "level_1_id": "industrial",
                    "level_2_id": None,
                    "evidence_type": "multiview_visual_review",
                }
            ],
        },
    }
    for record in records.values():
        record['provenance'] = {'ontology_id':'bonn_furniture_styles_hierarchical@1.0'}
        for item in record['styles']:
            item['confidence'] = 0.9
            if item.get('level_2_id'):
                item.update(source_dataset='3dfuture',source_label='European Classic')
            if record['style_status']=='visual_reviewed':
                item['visual_evidence_id']='evidence'
                record['visual_review']={'evidence_id':'evidence','model_id':'test',
                    'prompt_sha256':'a'*64,'adjudication_status':'agreed'}
    with gzip.open(path, "wt", encoding="utf-8") as stream:
        json.dump({"records": records}, stream)


def test_load_style_annotations_accepts_only_confirmed_v2_paths(tmp_path: Path):
    overlay = tmp_path / "styles.json.gz"
    _write_style_overlay(overlay)

    styles = load_asset_style_annotations(overlay)

    assert styles[ASSET_ID] == [
        StylePath("traditional", "european_classic", "source_metadata"),
        StylePath("transitional", None, "source_metadata"),
    ]
    assert styles["2222222222222222222222222222222222222222"] == [
        StylePath("industrial", None, "multiview_visual_review")
    ]
    assert "1111111111111111111111111111111111111111" not in styles


def test_asset_content_adds_verified_multilabel_styles():
    asset = RenderedAsset(
        asset_id=ASSET_ID,
        image_paths={"front": Path("front.png")},
        metadata=None,
        object_groups=[],
        asset_path=None,
        style_paths=[
            StylePath("traditional", "european_classic", "source_metadata"),
            StylePath("transitional", None, "source_metadata"),
        ],
    )

    assert build_asset_content(asset) == (
        f"HSSD asset id: {ASSET_ID}. "
        "Verified styles: traditional > european classic; transitional. "
        "Rendered views: front."
    )


def test_discovery_joins_style_overlay_by_hssd_id(tmp_path: Path):
    render_root = tmp_path / "renders"
    asset_dir = render_root / ASSET_ID
    asset_dir.mkdir(parents=True)
    (asset_dir / "front.png").write_bytes(b"not-empty")
    style = StylePath("modern", None, "source_metadata")
    metadata = HssdMeshMetadata(
        mesh_id=ASSET_ID,
        name="Test chair",
        up="0,1,0",
        front="0,0,1",
        wordnet_key="chair.n.01",
    )

    assets = discover_rendered_assets(
        render_root=render_root,
        metadata_by_id={ASSET_ID: metadata},
        groups_by_wordnet={"chair.n.01": ["seating"]},
        hssd_root=None,
        include_views={"front"},
        styles_by_id={ASSET_ID: [style]},
    )

    assert len(assets) == 1
    assert assets[0].asset_id == ASSET_ID
    assert assets[0].style_paths == [style]


def test_embedding_http_request_contains_verified_styles(tmp_path: Path):
    received_payloads: list[dict] = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802 - stdlib callback name
            length = int(self.headers["Content-Length"])
            received_payloads.append(json.loads(self.rfile.read(length)))
            body = json.dumps({"embedding": [1.0, 0.0, 0.0, 0.0]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format, *args):  # noqa: A002
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        image_path = tmp_path / "front.png"
        image_path.write_bytes(b"rendered-image")
        asset = RenderedAsset(
            asset_id=ASSET_ID,
            image_paths={"front": image_path},
            metadata=None,
            object_groups=[],
            asset_path=None,
            style_paths=[
                StylePath("traditional", "european_classic", "source_metadata"),
                StylePath("transitional", None, "source_metadata"),
            ],
        )
        client = LlamaEmbeddingClient(
            f"http://127.0.0.1:{server.server_port}", request_retries=0
        )

        embedding = embed_asset_views(asset, client, 4, "multi_image")
    finally:
        server.shutdown()
        server.server_close()
        thread.join()

    assert embedding == [1.0, 0.0, 0.0, 0.0]
    prompt = received_payloads[0]["content"]["prompt_string"]
    assert "Verified styles: traditional > european classic; transitional." in prompt
    assert "front view: <__media__>" in prompt
