"""TaskCompiler: converts a raw text prompt into a structured SceneTaskSpec.

Single Qwen3 call with role=task_compiler. Uses JSON output mode for reliability
with smaller open models.
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path

from scenesmith.scene_expert.context_bundle import build_llm_call_debug_record
from scenesmith.scene_expert.schemas import SceneTaskSpec
from scenesmith.agent_utils.thinking import chat_template_kwargs_from_effort

console_logger = logging.getLogger(__name__)


def _append_llm_debug(record: dict) -> None:
    path = os.environ.get("SCENEEXPERT_LLM_DEBUG_PATH", "")
    if not path:
        return
    try:
        debug_path = Path(path)
        debug_path.parent.mkdir(parents=True, exist_ok=True)
        with debug_path.open("a", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    except Exception as e:
        console_logger.warning("TaskCompiler failed to write LLM debug record: %s", e)


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

_NUMBER_WORDS: dict[str, int] = {
    "one": 1,
    "a": 1,
    "an": 1,
    "single": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
}

_OBJECT_ALIASES: dict[str, tuple[str, list[str], str]] = {
    "bed": ("large", ["bed", "beds"], "bed"),
    "nightstand": (
        "large",
        ["nightstand", "nightstands", "bedside table", "bedside tables"],
        "nightstand",
    ),
    "wardrobe": ("large", ["wardrobe", "wardrobes", "closet", "closets"], "wardrobe"),
    "sofa": ("large", ["sofa", "sofas", "couch", "couches"], "sofa"),
    "table": ("large", ["table", "tables", "desk", "desks"], "table"),
    "chair": ("large", ["chair", "chairs"], "chair"),
    "painting": ("wall", ["painting", "paintings", "artwork", "artworks"], "painting"),
    "mirror": ("wall", ["mirror", "mirrors"], "mirror"),
    "shelf": (
        "wall",
        ["shelf", "shelves", "floating shelf", "floating shelves"],
        "shelf",
    ),
    "ceiling light": (
        "ceiling",
        ["ceiling light", "ceiling lights", "pendant light", "pendant lights", "lamp"],
        "ceiling light",
    ),
    "book": ("small", ["book", "books"], "book"),
    "plant": ("small", ["plant", "plants"], "plant"),
}


def _extract_count_before_alias(text: str, alias: str) -> int:
    """Return a conservative count for an object mention in fallback parsing."""
    alias_pattern = re.escape(alias.lower()).replace(r"\ ", r"\s+")
    number_pattern = "|".join([r"\d+", *map(re.escape, _NUMBER_WORDS)])
    pattern = (
        rf"(?:(?P<count>{number_pattern})\s+)?" rf"(?:\w+\s+){{0,2}}{alias_pattern}\b"
    )
    best = 0
    for match in re.finditer(pattern, text):
        count_text = match.groupdict().get("count")
        if not count_text:
            count = 1
        elif count_text.isdigit():
            count = int(count_text)
        else:
            count = _NUMBER_WORDS.get(count_text, 1)
        best = max(best, count)
    return best


def _extract_required_objects_from_prompt(prompt_lower: str) -> dict[str, list[str]]:
    """Small deterministic parser used only when the model compiler fails."""
    required = {
        "large": [],
        "wall": [],
        "ceiling": [],
        "small": [],
    }
    for _, (bucket, aliases, canonical) in _OBJECT_ALIASES.items():
        count = 0
        for alias in aliases:
            count = max(count, _extract_count_before_alias(prompt_lower, alias))
        if count > 0:
            required[bucket].extend([canonical] * count)
    return required


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

    required = _extract_required_objects_from_prompt(prompt_lower)
    functional_zones: list[str] = []
    if room_type == "bedroom" or any(
        obj in required["large"] for obj in ("bed", "nightstand", "wardrobe")
    ):
        functional_zones.extend(["sleeping_zone", "storage_zone"])
    if any(obj in required["large"] for obj in ("table", "chair")):
        functional_zones.append("working_or_dining_zone")

    interaction_constraints: list[str] = []
    if "bed" in required["large"] and "nightstand" in required["large"]:
        interaction_constraints.append(
            "nightstands should flank the bed and remain reachable from the bed"
        )
    if "wardrobe" in required["large"]:
        interaction_constraints.append(
            "wardrobe doors should have clear access and should not block the room door"
        )

    console_logger.info(
        "TaskCompiler: fallback spec from prompt text: room_type=%s, style=%s, "
        "large_objects=%s",
        room_type,
        style,
        required["large"],
    )
    return SceneTaskSpec(
        room_type=room_type,
        style=style,
        required_large_objects=required["large"],
        required_wall_objects=required["wall"],
        required_ceiling_objects=required["ceiling"],
        required_small_objects=required["small"],
        functional_zones=functional_zones,
        interaction_constraints=interaction_constraints,
        aesthetic_constraints=["balanced placement", "clear walking paths"],
    )


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
        from openai import OpenAI

        self._model = model
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._client = OpenAI(
            base_url=api_base_url
            or os.environ.get("OPENAI_BASE_URL", "http://localhost:8000/v1"),
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
        user_message = f"Extract scene requirements from: {prompt}"

        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            temperature=self._temperature,
            max_tokens=self._max_tokens,
            extra_body=chat_template_kwargs_from_effort("none"),
        )

        message = response.choices[0].message
        raw = message.content
        # Qwen3 with --reasoning-parser may put output in reasoning_content.
        if not raw:
            raw = getattr(message, "reasoning_content", None)
        if not raw:
            extra = getattr(message, "model_extra", None)
            if isinstance(extra, dict):
                raw = extra.get("reasoning_content")
        console_logger.debug(f"TaskCompiler raw response: {raw}")
        _append_llm_debug(
            build_llm_call_debug_record(
                stage="task_compiler",
                agent_role="task_compiler",
                event="compile",
                prompt=user_message,
                output=raw or "",
                raw_response=response,
            ).model_dump()
        )

        try:
            data = _extract_json_from_text(raw)
            task_spec = SceneTaskSpec.model_validate(data)
            console_logger.info(
                f"TaskCompiler: room_type={task_spec.room_type}, style={task_spec.style}, "
                f"large_objects={task_spec.required_large_objects}"
            )
            return task_spec
        except Exception as e:
            raise ValueError(
                f"TaskCompiler failed to parse model response: {e}\nRaw: {raw}"
            ) from e
