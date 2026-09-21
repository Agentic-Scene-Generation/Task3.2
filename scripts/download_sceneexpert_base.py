"""Download missing ModelScope files and verify existing Qwen weights without overwrites."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
import urllib.parse
import urllib.request
from pathlib import Path


def fetch_manifest(model_id: str) -> list[dict]:
    url = f"https://modelscope.cn/api/v1/models/{model_id}/repo/files?Revision=master&Recursive=true"
    with urllib.request.urlopen(url, timeout=60) as response:
        payload = json.load(response)
    return [
        row
        for row in payload["Data"]["Files"]
        if row.get("Type") == "blob" and not row["Path"].startswith(".")
    ]


def verify_or_download(
    model_id: str, destination: Path, report_path: Path | None = None
) -> dict:
    """Verify SHA-256, resume by completed files, and preserve mismatching existing files."""
    rows = fetch_manifest(model_id)
    if not rows or not any(row["Path"].endswith(".safetensors") for row in rows):
        raise ValueError("ModelScope repository has no safetensors weights")
    destination.mkdir(parents=True, exist_ok=True)
    verified = []
    for row in rows:
        relative = Path(row["Path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("unsafe repository file path")
        path = destination / relative
        expected = row["Sha256"]
        if path.is_symlink():
            raise ValueError(f"refusing shared-model symlink: {path}")
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            partial = path.with_name(path.name + ".sceneexpert-download")
            url = f"https://modelscope.cn/models/{model_id}/resolve/master/{urllib.parse.quote(row['Path'])}"
            with (
                urllib.request.urlopen(url, timeout=120) as response,
                partial.open("wb") as output,
            ):
                while chunk := response.read(8 * 1024**2):
                    output.write(chunk)
            with partial.open("rb") as stream:
                if hashlib.file_digest(stream, "sha256").hexdigest() != expected:
                    raise ValueError(f"download hash mismatch: {relative}")
            # Never replace an existing shared file created by another downloader.
            os.link(partial, path)
            partial.unlink()
        with path.open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        if path.stat().st_size != row["Size"] or actual != expected:
            raise ValueError(
                f"existing model file differs; preserved for inspection: {path}"
            )
        verified.append(
            {
                "path": row["Path"],
                "sha256": actual,
                "size": row["Size"],
                "revision": row["Revision"],
            }
        )
        print(f"verified {len(verified)}/{len(rows)} {relative}", flush=True)
    config = json.loads((destination / "config.json").read_text())
    result = {
        "model_id": model_id,
        "source": "ModelScope",
        "verified_at": time.time(),
        "destination": str(destination.resolve()),
        "files": verified,
        "model_type": config.get("model_type"),
        "architectures": config.get("architectures"),
        "status": "verified",
    }
    report_path = (
        report_path or destination / "_sceneexpert_modelscope_verification.json"
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(result, indent=2))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default="Qwen/Qwen3.8-27B")
    parser.add_argument(
        "--destination",
        type=Path,
        default=Path("/mnt/afs/task3_2/share_model/Qwen/Qwen3.8-27B"),
    )
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    verify_or_download(args.model_id, args.destination, args.report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
