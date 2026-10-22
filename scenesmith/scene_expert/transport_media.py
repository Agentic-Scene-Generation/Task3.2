"""Exact-byte inline-image matching for future pinned transport packers."""

from __future__ import annotations

import base64
import binascii
import hashlib
import importlib.util
import re

from collections.abc import Iterator
from pathlib import Path
from types import ModuleType


class CanonicalInlineImages:
    """Leave malformed image-like text inside the packer's literal byte spans."""

    def __init__(self, pattern: re.Pattern[bytes]) -> None:
        self.pattern = pattern
        self.skipped: list[dict[str, str | int]] = []

    def finditer(self, data: bytes) -> Iterator[re.Match[bytes]]:
        """Yield only matches that survive canonical base64 reconstruction."""
        source_sha: str | None = None
        for match in self.pattern.finditer(data):
            encoded = match.group(2)
            reason = ""
            try:
                decoded = base64.b64decode(encoded, validate=True)
            except (binascii.Error, ValueError):
                reason = "invalid_base64"
            else:
                if base64.b64encode(decoded) != encoded:
                    reason = "noncanonical_base64"
            if reason:
                if source_sha is None:
                    source_sha = hashlib.sha256(data).hexdigest()
                left, right = match.span(2)
                self.skipped.append(
                    {
                        "source_sha256": source_sha,
                        "start": left,
                        "end": right,
                        "bytes": right - left,
                        "reason": reason,
                        "disposition": "retained_literal_original_bytes",
                    }
                )
                continue
            yield match


def load_pinned_transport(
    project_root: Path, packer_path: Path, expected_sha256: str
) -> tuple[ModuleType, CanonicalInlineImages]:
    """Load a caller-pinned future packer and replace only its image matcher.

    The external packer's on-disk code, selection rules, byte recipes, size limits,
    expected-output accounting and exit codes are left unchanged. A future plan
    must bind this module and its entry point as well as the external packer.
    """
    root = project_root.resolve()
    path = (root / packer_path).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError("The pinned packer must be a file inside the project root")
    if not re.fullmatch(r"[0-9a-f]{64}", expected_sha256):
        raise ValueError("A full lowercase packer SHA256 is required")
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected_sha256:
        raise ValueError("Pinned packer SHA256 mismatch; packer was not imported")
    spec = importlib.util.spec_from_file_location("sceneexpert_pinned_transport", path)
    if spec is None or spec.loader is None:
        raise ValueError("Cannot load the pinned transport packer")
    engine = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(engine)
    if not isinstance(getattr(engine, "INLINE", None), re.Pattern):
        raise TypeError("Pinned packer must expose an INLINE byte regex")
    if not isinstance(engine.INLINE.pattern, bytes):
        raise TypeError("Pinned packer's INLINE regex must operate on bytes")
    if not callable(getattr(engine, "package", None)) or not callable(
        getattr(engine, "verify", None)
    ):
        raise TypeError("Pinned packer must expose package and verify entry points")
    matcher = CanonicalInlineImages(engine.INLINE)
    engine.INLINE = matcher
    return engine, matcher
