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


def critic_feedback_contract(stage: str = "") -> str:
    """Return the SceneExpert-only contract appended to existing critic prompts."""

    scope_rule = _STAGE_SCOPE_RULES.get(
        stage,
        "Evaluate only objects and relationships owned by the current stage.",
    )
    return f"""\
# SceneExpert Compact Repair-Brief Contract

Keep the existing structured score fields and their one-sentence comments.
Current stage: {stage or "unspecified"}
Stage ownership: {scope_rule}
Never create findings or score penalties for objects owned by another stage.
In the `critique` field, do NOT write essay-style sections. Return exactly:

STATUS: PASS | REPAIR_REQUIRED
SUMMARY: one concise overall sentence
FINDING 1
SEVERITY: BLOCKING | MAJOR | REFINEMENT
CATEGORY: short issue type
OBJECTS: exact object IDs, comma-separated, or scene/stage
OBSERVATION: one short concrete fact from the render/state
REQUIRED_CHANGE: one short actionable designer instruction
ACCEPTANCE_CHECK: one observable condition proving the repair succeeded
END_FINDING

Return at most three FINDING blocks. Merge related evidence and objects so
all blocking issue classes are still represented. While any blocking issue exists,
omit optional refinements. Never invent object IDs or coordinates. A PASS may
contain zero findings. The parser still accepts legacy REASON and PRESERVE
fields, but do not emit them in this compact scoring transaction.
"""


_DIRECT_STAGE_RUBRICS = {
    "floor_plan": (
        "Judge architectural room proportions, circulation and connectivity, "
        "window/daylight provision, material suitability, and adherence to the "
        "requested architectural brief. Ignore furniture that belongs to later stages."
    ),
    "furniture": (
        "Judge real-world plausibility, functionality, anchor and secondary-object "
        "relationships, orientation, completeness, prompt adherence, and reachability."
    ),
    "wall_mounted": (
        "Judge realistic mounting height, functional accessibility, spacing from "
        "openings/furniture, visual balance, completeness, and prompt adherence."
    ),
    "ceiling_mounted": (
        "Judge realistic fixture placement, functional coverage, spacing and "
        "clearance, visual balance, and prompt adherence."
    ),
    "manipuland": (
        "Judge realistic support relationships, usability, local arrangement, "
        "appropriate completeness, and prompt adherence."
    ),
}

_STAGE_SCOPE_RULES = {
    "floor_plan": (
        "Own only room footprints and dimensions, walls, doors, windows, openings, "
        "floor surfaces/materials, architectural circulation, connectivity, and "
        "daylight. Furniture and all decorative/ceiling/manipuland objects are "
        "downstream and are absent by design. Never report missing furniture or "
        "penalize furniture placement, orientation, furniture clearance, or "
        "furniture completeness. A window finding is in scope only as a change to "
        "the window/opening itself."
    ),
    "furniture": (
        "Own only large furniture assets, their poses, functional relationships, "
        "clearance, reachability, and interaction with existing architecture. Do "
        "not penalize absent wall decorations, ceiling fixtures, or manipulands."
    ),
    "wall_mounted": (
        "Own only wall-mounted assets and their mounting, spacing, accessibility, "
        "and interaction with existing openings/furniture. Do not redesign upstream "
        "architecture/furniture or require ceiling/manipuland objects."
    ),
    "ceiling_mounted": (
        "Own only ceiling-mounted assets and their coverage, spacing, clearance, "
        "and mounting. Do not redesign upstream stages or require manipulands."
    ),
    "manipuland": (
        "Own only small supported/manipulable objects and their support, usability, "
        "spacing, and completeness. Treat all upstream geometry as fixed context."
    ),
}

_STAGE_CATEGORY_EXCLUSIONS = {
    "floor_plan": (
        "missing_furniture",
        "furniture_missing",
        "furniture_completeness",
        "furniture_layout",
        "furniture_orientation",
        "missing_wall_object",
        "missing_ceiling",
        "missing_manipuland",
        "decor_completeness",
    ),
    "furniture": (
        "missing_wall_object",
        "missing_ceiling",
        "missing_manipuland",
        "wall_decor_completeness",
        "ceiling_completeness",
        "manipuland_completeness",
    ),
    "wall_mounted": (
        "missing_ceiling",
        "missing_manipuland",
        "ceiling_completeness",
        "manipuland_completeness",
    ),
    "ceiling_mounted": (
        "missing_manipuland",
        "manipuland_completeness",
    ),
}

_STAGE_CATEGORY_REQUIREMENTS = {
    "floor_plan": (
        "room",
        "footprint",
        "dimension",
        "proportion",
        "geometry",
        "layout",
        "shape",
        "boundary",
        "wall",
        "door",
        "window",
        "opening",
        "floor",
        "material",
        "circulation",
        "flow",
        "connectivity",
        "access",
        "daylight",
        "lighting",
        "architecture",
    ),
}


def critic_finding_is_in_stage_scope(
    finding: CriticFinding,
    stage: str,
) -> bool:
    """Return whether a critic finding belongs to the current stage contract."""
    category = re.sub(
        r"[^a-z0-9_]+",
        "_",
        str(finding.category or "").casefold(),
    ).strip("_")
    if any(marker in category for marker in _STAGE_CATEGORY_EXCLUSIONS.get(stage, ())):
        return False
    required_markers = _STAGE_CATEGORY_REQUIREMENTS.get(stage, ())
    if not required_markers:
        return True
    evidence = " ".join(
        [
            category,
            *finding.object_ids,
        ]
    ).casefold()
    return any(marker in evidence for marker in required_markers)


def scope_critic_feedback(
    feedback: CriticFeedback,
    stage: str,
) -> CriticFeedback:
    """Drop structured cross-stage findings before planner/verifier consumption."""
    if not feedback.structured:
        return feedback
    scoped_findings = [
        finding
        for finding in feedback.findings
        if critic_finding_is_in_stage_scope(finding, stage)
    ]
    if len(scoped_findings) == len(feedback.findings):
        return feedback
    status = feedback.status
    summary = feedback.summary
    if not scoped_findings:
        status = "PASS"
        summary = (
            f"No actionable `{stage}` issue remains after enforcing the stage "
            "ownership contract."
        )
    return feedback.model_copy(
        update={
            "status": status,
            "summary": summary,
            "findings": scoped_findings,
        }
    )


def direct_critic_scoring_instructions(
    *,
    stage: str,
    scene_context: str,
    category_names: list[str],
) -> str:
    """Build the clean system contract for one-shot structured visual scoring.

    The native SceneSmith critic prompt is intentionally not embedded here: it
    requires an agentic tool workflow, while this scorer receives evidence
    directly and has no tools. Keeping these two protocols separate removes the
    contradictory instructions that caused Agents SDK recovery loops.
    """

    categories = ", ".join(category_names)
    rubric = _DIRECT_STAGE_RUBRICS.get(
        stage,
        "Judge realism, functionality, layout quality, completeness, and prompt adherence.",
    )
    scope_rule = _STAGE_SCOPE_RULES.get(
        stage,
        "Evaluate only objects and relationships owned by the current stage.",
    )
    context = str(scene_context or "").strip()
    if len(context) > 4000:
        context = context[:4000].rstrip() + "\n...[scene context truncated]..."
    return f"""\
/no_think
You are the final visual quality critic for the `{stage}` stage.
The framework has already collected and attached all render, exact-state,
validation, physics, reachability, and orientation evidence available for this
candidate. No tools are available or required. Never request, call, or narrate
tools, and never emit a checklist.

Evaluation rubric:
{rubric}

Authoritative stage ownership:
{scope_rule}
The scene/task context may describe downstream requirements. Those requirements
are context only until their owning stage and must not lower a current-stage
score or create a finding.

Required score categories: {categories}
Score every category from 0 through 10 using a stable scale: 0-2 unusable,
3-5 major repair required, 6-7 acceptable with issues, 8-9 strong, 10 exceptional.
Deterministic validation is authoritative for actual collision/connectivity facts;
use visual evidence for semantic, functional, relational, and aesthetic quality.
Each category comment must be one concise evidence-based sentence.

Scene/task context:
{context or "(no additional scene text)"}

Return exactly the structured object required by the response schema in one
response. Do not output Markdown, code fences, analysis, or prose outside it.

{critic_feedback_contract(stage)}
""".strip()


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
