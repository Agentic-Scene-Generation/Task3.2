"""Prompt-boundary helpers for SceneExpert stage injection."""

from __future__ import annotations

import re


_INJECTED_BLOCK_PATTERNS = (
    re.compile(
        r"\n*=== SceneExpert Stage Brief:[^\n]*===.*?"
        r"=== End Stage Brief ===\n*",
        flags=re.DOTALL,
    ),
    re.compile(
        r"\n*=== SceneExpert Retrieved Memory Directives ===.*?"
        r"=== End Retrieved Memory Directives ===\n*",
        flags=re.DOTALL,
    ),
    re.compile(
        r"\n*=== SceneExpert Stage Completion Contract:[^\n]*===.*?"
        r"=== End Stage Completion Contract ===\n*",
        flags=re.DOTALL,
    ),
    re.compile(
        r"\n*=== Reference Layout \([^\n]*\) ===.*?"
        r"=== End Reference Layout ===\n*",
        flags=re.DOTALL,
    ),
)


def strip_sceneexpert_injected_blocks(prompt: str) -> str:
    """Remove transient SceneExpert blocks persisted by an upstream stage.

    Floor-plan workers serialize their effective prompt into each ``RoomSpec``.
    Without this boundary cleanup, the floor-plan brief and retrieved memories
    become part of every downstream room prompt, where negative examples can be
    misread as positive asset requirements.
    """
    cleaned = str(prompt or "")
    for pattern in _INJECTED_BLOCK_PATTERNS:
        cleaned = pattern.sub("\n", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()
