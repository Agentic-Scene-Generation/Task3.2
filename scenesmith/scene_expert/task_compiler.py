"""TaskCompiler: converts a raw text prompt into a structured SceneTaskSpec.

Single Qwen3 call with role=task_compiler. Uses JSON output mode for reliability
with smaller open models.
"""

from __future__ import annotations

import json
import logging
import os
import re

from openai import OpenAI

from scenesmith.scene_expert.schemas import SceneTaskSpec

console_logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
/no_think
You are the task_compiler for SceneExpert, a 3D indoor scene generation system.
Your job is to extract structured scene requirements from a natural-language prompt.

You MUST output valid JSON matching this exact schema:
{
  "room_type": "string — primary room type (e.g. bedroom, kitchen, living room, office)",
  "style": "string — aesthetic style (e.g. cozy modern, industrial, minimalist, farmhouse)",
  "required_large_objects": ["list of furniture-scale objects that must be in the room"],
  "required_wall_objects": ["list of wall-mounted objects (paintings, mirrors, shelves, lights)"],
  "required_ceiling_objects": ["list of ceiling-mounted objects (lights, fans, sprinklers)"],
  "required_small_objects": ["list of small manipulable objects (books, cups, plants, tools)"],
  "functional_zones": ["list of spatial zones within the room (e.g. sleeping_zone, working_zone)"],
  "interaction_constraints": [
    "constraints about robot reachability, clearance, support surfaces",
    "e.g. 'nightstand should be reachable from the accessible side of the bed'"
  ],
  "aesthetic_constraints": [
    "visual and style constraints",
    "e.g. 'modern material palette', 'balanced visual density', 'avoid overcrowding'"
  ]
}

Rules:
- Be comprehensive — extract ALL objects mentioned in the prompt.
- Infer reasonable functional zones based on the room type and objects.
- Infer reachability constraints for any small objects placed on furniture surfaces.
- Keep object names concise (e.g. "bed" not "a large king-sized bed").
- Output ONLY the JSON object, no other text.

Example input: "A bedroom with a bed, two nightstands, and a wardrobe."
Example output:
{
  "room_type": "bedroom",
  "style": "standard",
  "required_large_objects": ["bed", "nightstand", "nightstand", "wardrobe"],
  "required_wall_objects": [],
  "required_ceiling_objects": [],
  "required_small_objects": [],
  "functional_zones": ["sleeping_zone", "storage_zone"],
  "interaction_constraints": ["nightstands should be accessible from both sides of the bed"],
  "aesthetic_constraints": ["balanced furniture placement", "clear walking paths"]
}
"""

_ROOM_TYPE_KEYWORDS: dict[str, list[str]] = {
    "bedroom": ["bedroom", "bed", "nightstand", "wardrobe", "sleeping"],
    "living room": ["living room", "living", "sofa", "couch", "tv", "coffee table"],
    "kitchen": ["kitchen", "stove", "oven", "fridge", "sink", "counter"],
    "bathroom": ["bathroom", "toilet", "bathtub", "shower", "sink"],
    "office": ["office", "desk", "chair", "computer", "monitor", "study"],
    "dining room": ["dining room", "dining", "dining table", "chairs"],
    "garage": ["garage", "car", "workbench", "tools"],
    "basement": ["basement", "laundry", "storage"],
}

_STYLE_KEYWORDS: dict[str, list[str]] = {
    "modern": ["modern", "contemporary", "sleek", "minimalist"],
    "cozy": ["cozy", "warm", "comfortable", "homey"],
    "industrial": ["industrial", "metal", "raw", "exposed"],
    "farmhouse": ["farmhouse", "rustic", "country", "wooden"],
    "scandinavian": ["scandinavian", "nordic", "simple", "functional"],
    "luxury": ["luxury", "elegant", "upscale", "premium"],
}


def _extract_json_from_text(text: str) -> dict:
    """Extract JSON from model output, handling markdown code fences."""
    if not text:
        raise ValueError("Empty response text")
    # Strip markdown code fences if present
    text = text.strip()
    fence_match = re.search(r"```(?:json)?\s*([\s\S]+?)\s*```", text)
    if fence_match:
        text = fence_match.group(1)
    # Find first { ... } block
    brace_match = re.search(r"\{[\s\S]+\}", text)
    if brace_match:
        text = brace_match.group(0)
    return json.loads(text)


def _fallback_spec_from_prompt(prompt: str) -> SceneTaskSpec:
    """Parse room_type and style from prompt text when model call fails."""
    prompt_lower = prompt.lower()

    room_type = "room"
    for rtype, keywords in _ROOM_TYPE_KEYWORDS.items():
        if any(kw in prompt_lower for kw in keywords):
            room_type = rtype
            break

    style = "standard"
    for stype, keywords in _STYLE_KEYWORDS.items():
        if any(kw in prompt_lower for kw in keywords):
            style = stype
            break

    console_logger.info(
        f"TaskCompiler: fallback spec from prompt text: room_type={room_type}, style={style}"
    )
    return SceneTaskSpec(room_type=room_type, style=style)


class TaskCompiler:
    """Converts a raw text prompt to a structured SceneTaskSpec via Qwen3."""

    def __init__(
        self,
        model: str,
        api_base_url: str | None = None,
        api_key: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.1,
    ) -> None:
        self._model = model
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._client = OpenAI(
            base_url=api_base_url or os.environ.get("OPENAI_BASE_URL", "http://localhost:8000/v1"),
            api_key=api_key or os.environ.get("OPENAI_API_KEY", "dummy"),
        )

    def compile(self, prompt: str) -> SceneTaskSpec:
        """Parse a raw text prompt into a SceneTaskSpec.

        Args:
            prompt: Natural-language scene description.

        Returns:
            Structured SceneTaskSpec.

        Raises:
            ValueError: If the model response cannot be parsed.
        """
        console_logger.info(f"TaskCompiler: compiling prompt: {prompt[:100]}...")

        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": f"Extract scene requirements from: {prompt}"},
            ],
            temperature=self._temperature,
            max_tokens=self._max_tokens,
        )

        raw = response.choices[0].message.content
        # Qwen3 with --reasoning-parser may put output in reasoning_content
        if raw is None:
            raw = getattr(response.choices[0].message, "reasoning_content", None)
        console_logger.debug(f"TaskCompiler raw response: {raw}")

        try:
            data = _extract_json_from_text(raw)
            task_spec = SceneTaskSpec.model_validate(data)
            console_logger.info(
                f"TaskCompiler: room_type={task_spec.room_type}, style={task_spec.style}, "
                f"large_objects={task_spec.required_large_objects}"
            )
            return task_spec
        except Exception as e:
            raise ValueError(f"TaskCompiler failed to parse model response: {e}\nRaw: {raw}") from e

