"""Shared weights are verified, never silently replaced by the downloader."""

from __future__ import annotations

import hashlib
import io

import pytest

from scripts import download_sceneexpert_base as download


def manifest(contents: dict[str, bytes]) -> list[dict]:
    return [
        {
            "Path": name,
            "Size": len(value),
            "Sha256": hashlib.sha256(value).hexdigest(),
            "Revision": "revision",
        }
        for name, value in contents.items()
    ]


def test_verified_existing_model_only_writes_external_report(tmp_path, monkeypatch):
    contents = {
        "model.safetensors": b"weights",
        "config.json": b'{"model_type":"qwen3_5"}',
    }
    model = tmp_path / "shared"
    model.mkdir()
    for name, value in contents.items():
        (model / name).write_bytes(value)
    monkeypatch.setattr(download, "fetch_manifest", lambda _: manifest(contents))
    result = download.verify_or_download(
        "Qwen/example", model, tmp_path / "report.json"
    )
    assert result["status"] == "verified"
    assert sorted(p.name for p in model.iterdir()) == sorted(contents)
    assert all((model / name).read_bytes() == value for name, value in contents.items())


def test_missing_file_download_is_checked_before_publication(tmp_path, monkeypatch):
    contents = {"model.safetensors": b"weights", "config.json": b"{}"}
    monkeypatch.setattr(download, "fetch_manifest", lambda _: manifest(contents))
    monkeypatch.setattr(
        download.urllib.request,
        "urlopen",
        lambda url, **_: io.BytesIO(contents[url.rsplit("/", 1)[1]]),
    )
    result = download.verify_or_download(
        "Qwen/example", tmp_path / "model", tmp_path / "report.json"
    )
    assert len(result["files"]) == 2
    assert not list((tmp_path / "model").glob("*.sceneexpert-download"))


def test_hash_mismatch_preserves_existing_shared_model(tmp_path, monkeypatch):
    path = tmp_path / "model.safetensors"
    path.write_bytes(b"existing model")
    monkeypatch.setattr(
        download, "fetch_manifest", lambda _: manifest({path.name: b"different model"})
    )
    with pytest.raises(ValueError, match="preserved"):
        download.verify_or_download("Qwen/example", tmp_path, tmp_path / "report.json")
    assert path.read_bytes() == b"existing model"
    assert not (tmp_path / "report.json").exists()
