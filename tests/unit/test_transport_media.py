"""Malformed image lookalikes must remain exact structured evidence bytes."""

from __future__ import annotations

import base64
import hashlib
import re

from pathlib import Path

import pytest

from scenesmith.scene_expert.transport_media import (
    CanonicalInlineImages,
    load_pinned_transport,
)

INLINE = re.compile(rb"data:image/(png|jpeg|jpg|webp);base64,([A-Za-z0-9+/=]{1024,})")


@pytest.mark.parametrize(
    "invalid",
    [b"A" * 1733, b"A" * 1024 + b"=A", b"A" * 1024 + b"AB=="],
)
def test_mixed_json_evidence_retains_invalid_spans_and_factors_valid_bytes(
    invalid: bytes,
) -> None:
    image = bytes(range(256)) * 5
    valid = base64.b64encode(image)
    data = (
        b'{"invalid":"data:image/png;base64,'
        + invalid
        + b'","valid":"data:image/webp;base64,'
        + valid
        + b'","duplicate":"data:image/png;base64,'
        + valid
        + b'","suffix":"unaltered\\ntext"}\n'
    )
    matcher = CanonicalInlineImages(INLINE)
    matches = list(matcher.finditer(data))
    assert len(matches) == 2
    assert len(matcher.skipped) == 1
    assert matcher.skipped[0]["disposition"] == "retained_literal_original_bytes"
    parts: list[bytes] = []
    offset = 0
    for match in matches:
        left, right = match.span(2)
        parts.extend([data[offset:left], base64.b64encode(image)])
        offset = right
    parts.append(data[offset:])
    assert b"".join(parts) == data
    assert invalid in parts[0]
    assert matcher.skipped[0]["source_sha256"] == hashlib.sha256(data).hexdigest()


def test_bad_hash_refuses_import_before_any_packer_side_effect(tmp_path: Path) -> None:
    sentinel = tmp_path / "should_not_exist"
    packer = tmp_path / "packer.py"
    packer.write_text(
        f"from pathlib import Path\nPath({str(sentinel)!r}).touch()\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        load_pinned_transport(tmp_path, packer, "0" * 64)
    assert not sentinel.exists()


def test_loading_changes_only_in_memory_matcher(tmp_path: Path) -> None:
    packer = tmp_path / "packer.py"
    original = (
        b"import re\nINLINE = re.compile(rb'data:image/(png);base64,([A-Za-z0-9+/=]{1024,})')\n"
        b"def package(*args): return 2\ndef verify(*args): return {'verified': True}\n"
    )
    packer.write_bytes(original)
    engine, matcher = load_pinned_transport(
        tmp_path, packer, hashlib.sha256(original).hexdigest()
    )
    assert engine.INLINE is matcher
    assert engine.package() == 2
    assert packer.read_bytes() == original
