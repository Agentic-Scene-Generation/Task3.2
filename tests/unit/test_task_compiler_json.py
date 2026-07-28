"""Regression tests for TaskCompiler's resilient JSON parsing."""

from scenesmith.scene_expert.schemas import SceneTaskSpec
from scenesmith.scene_expert.task_compiler import (
    _SYSTEM_PROMPT,
    _extract_json_from_text,
)


def test_task_compiler_repairs_a_truncated_optional_constraint() -> None:
    """A complete task spec should survive a model cutoff in optional metadata."""
    raw = """{
      "room_type": "study",
      "style": "standard",
      "required_large_objects": ["desk"],
      "required_wall_objects": [],
      "required_ceiling_objects": [],
      "required_small_objects": [],
      "functional_zones": ["working_zone"],
      "interaction_constraints": [],
      "aesthetic_constraints": [],
      "intent_constraints": [
        {
          "relation": "against_wall",
          "subjects": {"category": "chair"},
          "targets": {"category": "wall"},
          "confidence"""

    spec = SceneTaskSpec.model_validate(_extract_json_from_text(raw))

    assert spec.room_type == "study"
    assert spec.required_large_objects == ["desk"]
    assert spec.intent_constraints[0]["relation"] == "against_wall"


def test_task_compiler_requests_only_auxiliary_intent_constraints() -> None:
    """Model relations are useful evidence but cannot self-authorize hard rules."""
    assert '"intent_constraints"' in _SYSTEM_PROMPT
    assert '"source": "model_inferred"' in _SYSTEM_PROMPT


def test_task_compiler_requests_two_anchor_between_relations() -> None:
    assert "centered_between" in _SYSTEM_PROMPT
    assert "secondary_category" in _SYSTEM_PROMPT
    assert 'ordinary\n  "X between A and B", emit "between"' in _SYSTEM_PROMPT
    assert "never a hard requirement" in _SYSTEM_PROMPT
    assert "Do not invent coordinates" in _SYSTEM_PROMPT
    assert "Emit at most eight intent constraints" in _SYSTEM_PROMPT
