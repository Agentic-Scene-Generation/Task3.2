"""Compact, lossless critic feedback for SceneExpert runtime consumers.

SceneSmith's public ``CritiqueWithScores`` types intentionally remain unchanged.
When SceneExpert is enabled, the natural-language ``critique`` field follows the
contract below and is normalized here for designer repair, verification, tracing,
and long-term memory.  Legacy prose remains supported as an opaque fallback.
"""

from __future__ import annotations

import re

from pydantic import BaseModel, Field


_FIELD_NAMES = (
    "SEVERITY",
    "CATEGORY",
    "OBJECTS",
    "OBSERVATION",
    "REASON",
    "REQUIRED_CHANGE",
    "PRESERVE",
    "ACCEPTANCE_CHECK",
)
_FIELD_PATTERN = re.compile(
    rf"(?im)^(?P<name>{'|'.join(_FIELD_NAMES)}):\s*(?P<value>.*)$"
)
_FINDING_PATTERN = re.compile(
    r"(?is)^FINDING\s+\d+\s*:?\s*$" r"(?P<body>.*?)" r"^END_FINDING\s*$",
    flags=re.MULTILINE,
)


class CriticFinding(BaseModel):
    """One actionable, independently verifiable critic finding."""

    severity: str = "major"
    category: str = "quality"
    object_ids: list[str] = Field(default_factory=list)
    observation: str = ""
    reason: str = ""
    required_change: str = ""
    preserve: list[str] = Field(default_factory=list)
    acceptance_check: str = ""

    @property
    def is_blocking(self) -> bool:
        return self.severity.casefold() in {
            "blocking",
            "critical",
            "hard",
        }

    def to_designer_text(self, index: int) -> str:
        objects = ", ".join(self.object_ids) or "scene/stage"
        lines = [
            f"{index}. [{self.severity.upper()}/{self.category}] objects={objects}",
        ]
        if self.observation:
            lines.append(f"   Observed: {self.observation}")
        if self.reason:
            lines.append(f"   Why: {self.reason}")
        if self.required_change:
            lines.append(f"   Required change: {self.required_change}")
        if self.preserve:
            lines.append(f"   Preserve: {'; '.join(self.preserve)}")
        if self.acceptance_check:
            lines.append(f"   Accept when: {self.acceptance_check}")
        return "\n".join(lines)


class CriticFeedback(BaseModel):
    """Normalized critic decision shared by downstream SceneExpert modules."""

    status: str = "UNKNOWN"
    summary: str = ""
    findings: list[CriticFinding] = Field(default_factory=list)
    raw_text: str = ""
    structured: bool = False

    @property
    def blocking_findings(self) -> list[CriticFinding]:
        return [finding for finding in self.findings if finding.is_blocking]

    def to_designer_text(self, max_chars: int = 5000) -> str:
        if not self.structured:
            return _truncate(self.raw_text, max_chars)
        lines = [
            "=== Authoritative Critic Repair Brief ===",
            f"Status: {self.status}",
        ]
        if self.summary:
            lines.append(f"Summary: {self.summary}")
        if self.findings:
            lines.append("Findings:")
            lines.extend(
                finding.to_designer_text(index)
                for index, finding in enumerate(self.findings, start=1)
            )
        else:
            lines.append("Findings: none")
        lines.append("=== End Critic Repair Brief ===")
        return _truncate("\n".join(lines), max_chars)


def critic_feedback_contract() -> str:
    """Return the SceneExpert-only contract appended to existing critic prompts."""

    return """\
# SceneExpert Compact Repair-Brief Contract

Keep the existing structured score fields and their one-sentence comments.
In the `critique` field, do NOT write essay-style sections. Return exactly:

STATUS: PASS | REPAIR_REQUIRED
SUMMARY: one concise overall sentence
FINDING 1
SEVERITY: BLOCKING | MAJOR | REFINEMENT
CATEGORY: short issue type
OBJECTS: exact object IDs, comma-separated, or scene/stage
OBSERVATION: concrete evidence from the render/state
REASON: why this is functionally or visually wrong
REQUIRED_CHANGE: one actionable designer instruction
PRESERVE: correct relationships that must not be damaged, separated by semicolons
ACCEPTANCE_CHECK: observable condition proving the repair succeeded
END_FINDING

Repeat FINDING blocks as needed. Include EVERY blocking issue; blocking issues
have no count limit. While any blocking issue exists, omit optional refinements.
If there are no blocking issues, include at most three major/refinement findings.
Never invent object IDs or coordinates. A PASS may contain zero findings.
"""


def direct_critic_scoring_instructions(instructions: str) -> str:
    """Make the framework-driven scoring mode explicit and non-contradictory.

    SceneSmith's native critic instructions require the model to collect evidence
    with tools.  SceneExpert's direct path has already collected and attached that
    evidence, then deliberately removes all tools from the scoring agent.  This
    authoritative mode override keeps the stage-specific rubric while preventing
    the model from attempting an impossible tool workflow or emitting a narrated
    checklist instead of the structured score object.
    """

    return (
        str(instructions or "").rstrip()
        + """

# SceneExpert Direct Evidence Scoring Mode - Authoritative Override

For this scoring request only, the framework has already completed every required
observation, scene-state, validation, physics, and orientation-evidence step and
has attached the resulting evidence to the user message. This mode OVERRIDES any
earlier instruction to call or narrate tools. No tools are available or required.

Evaluate only the supplied candidate and return the final structured output in one
response. Do not emit a checklist, tool call, Markdown, code fence, or prose outside
the output schema. Keep every category comment to one evidence-based sentence and
follow the SceneExpert Compact Repair-Brief Contract inside the `critique` field.
"""
    ).strip()


def parse_critic_feedback(text: str) -> CriticFeedback:
    """Parse the compact contract, preserving legacy prose as a safe fallback."""

    raw_text = str(text or "").strip()
    status_match = re.search(r"(?im)^STATUS:\s*([A-Z_]+)\s*$", raw_text)
    summary_match = re.search(r"(?im)^SUMMARY:\s*(.+?)\s*$", raw_text)
    findings: list[CriticFinding] = []

    for match in _FINDING_PATTERN.finditer(raw_text):
        values = {
            field_match.group("name").upper(): field_match.group("value").strip()
            for field_match in _FIELD_PATTERN.finditer(match.group("body"))
        }
        if not values:
            continue
        findings.append(
            CriticFinding(
                severity=values.get("SEVERITY", "major").casefold(),
                category=values.get("CATEGORY", "quality").casefold(),
                object_ids=_split_values(values.get("OBJECTS", "")),
                observation=values.get("OBSERVATION", ""),
                reason=values.get("REASON", ""),
                required_change=values.get("REQUIRED_CHANGE", ""),
                preserve=_split_values(values.get("PRESERVE", ""), delimiter=";"),
                acceptance_check=values.get("ACCEPTANCE_CHECK", ""),
            )
        )

    structured = status_match is not None and (
        bool(findings) or "FINDING" not in raw_text.upper()
    )
    return CriticFeedback(
        status=(status_match.group(1).upper() if status_match else "UNKNOWN"),
        summary=(summary_match.group(1).strip() if summary_match else ""),
        findings=findings,
        raw_text=raw_text,
        structured=structured,
    )


def feedback_issue_text(finding: CriticFinding) -> str:
    """Return a compact memory/verifier description without dropping evidence."""

    parts = [finding.observation, finding.reason]
    return " ".join(part.strip() for part in parts if part.strip())


def feedback_repair_text(finding: CriticFinding) -> str:
    """Return the repair action plus its acceptance condition."""

    action = finding.required_change.strip()
    check = finding.acceptance_check.strip()
    if action and check:
        return f"{action} Verify: {check}"
    return action or check


def _split_values(value: str, delimiter: str = ",") -> list[str]:
    ignored = {"", "none", "n/a", "scene/stage", "scene", "stage"}
    return [
        item
        for raw_item in str(value or "").split(delimiter)
        if (item := raw_item.strip()) and item.casefold() not in ignored
    ]


def _truncate(value: str, max_chars: int) -> str:
    if max_chars <= 0 or len(value) <= max_chars:
        return value
    return value[: max(0, max_chars - 3)].rstrip() + "..."
