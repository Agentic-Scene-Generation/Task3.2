"""Deterministic resident behavior and grouped-asset planner.

Derived from AgentSense behavior-planning work.
Copyright (c) 2025 Zikang Leng
Licensed under the MIT License.
"""

from __future__ import annotations

import json
import re

from typing import Any

from scenesmith.agent_utils.thinking import chat_template_kwargs_from_effort
from scenesmith.scene_expert.behavior.schemas import (
    ActionStep,
    AssetNeed,
    BehaviorRelation,
    BehaviorSpec,
    DetailedActivity,
    ObjectNeed,
    PersonaSpec,
    RoomBehaviorSpec,
    ScheduleActivity,
)

DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

ROOM_ALIASES = {
    "bedroom": ["bedroom", "bed room", "sleeping room", "master bedroom"],
    "kitchen": ["kitchen", "kitchenette", "cooking area"],
    "livingroom": [
        "living room",
        "livingroom",
        "lounge",
        "sitting room",
        "family room",
    ],
    "bathroom": ["bathroom", "bath room", "washroom", "restroom", "toilet room"],
}

ROOM_LABELS = {
    "bedroom": "bedroom",
    "kitchen": "kitchen",
    "livingroom": "living room",
    "bathroom": "bathroom",
}

STAGE_OVERRIDES = {
    "mirror": "wall_mounted",
}

KNOWN_OBJECTS: dict[str, dict[str, str]] = {
    "bed": {"room": "bedroom", "stage": "furniture"},
    "nightstand": {"room": "bedroom", "stage": "furniture"},
    "nightstands": {"room": "bedroom", "stage": "furniture", "canonical": "nightstand"},
    "wardrobe": {"room": "bedroom", "stage": "furniture"},
    "closet": {"room": "bedroom", "stage": "furniture"},
    "desk": {"room": "bedroom", "stage": "furniture"},
    "chair": {"room": "bedroom", "stage": "furniture"},
    "lamp": {"room": "bedroom", "stage": "furniture"},
    "book": {"room": "bedroom", "stage": "manipuland", "support": "nightstand"},
    "books": {
        "room": "bedroom",
        "stage": "manipuland",
        "canonical": "book",
        "support": "desk",
    },
    "phone": {"room": "bedroom", "stage": "manipuland", "support": "nightstand"},
    "laptop": {"room": "bedroom", "stage": "manipuland", "support": "desk"},
    "computer": {"room": "bedroom", "stage": "furniture", "support": "desk"},
    "mug": {
        "room": "bedroom",
        "stage": "manipuland",
        "support": "nightstand",
    },
    "water glass": {
        "room": "bedroom",
        "stage": "manipuland",
        "support": "nightstand",
    },
    "sofa": {"room": "livingroom", "stage": "furniture"},
    "couch": {"room": "livingroom", "stage": "furniture", "canonical": "sofa"},
    "coffee table": {"room": "livingroom", "stage": "furniture"},
    "tv": {"room": "livingroom", "stage": "furniture", "canonical": "television"},
    "television": {"room": "livingroom", "stage": "furniture"},
    "remote": {
        "room": "livingroom",
        "stage": "manipuland",
        "canonical": "remote control",
        "support": "coffee table",
    },
    "magazine": {
        "room": "livingroom",
        "stage": "manipuland",
        "support": "coffee table",
    },
    "plant": {"room": "livingroom", "stage": "furniture"},
    "bookshelf": {"room": "livingroom", "stage": "furniture"},
    "dining table": {
        "room": "kitchen",
        "stage": "furniture",
        "canonical": "kitchen table",
    },
    "kitchen table": {"room": "kitchen", "stage": "furniture"},
    "counter": {
        "room": "kitchen",
        "stage": "furniture",
        "canonical": "kitchen counter",
    },
    "kitchen counter": {"room": "kitchen", "stage": "furniture"},
    "fridge": {"room": "kitchen", "stage": "furniture", "canonical": "refrigerator"},
    "refrigerator": {"room": "kitchen", "stage": "furniture"},
    "sink": {"room": "kitchen", "stage": "furniture"},
    "stove": {"room": "kitchen", "stage": "furniture"},
    "coffee maker": {"room": "kitchen", "stage": "furniture"},
    "plate": {"room": "kitchen", "stage": "manipuland", "support": "kitchen counter"},
    "bowl": {"room": "kitchen", "stage": "manipuland", "support": "kitchen counter"},
    "pan": {"room": "kitchen", "stage": "manipuland", "support": "stove"},
    "cutting board": {
        "room": "kitchen",
        "stage": "manipuland",
        "support": "kitchen counter",
    },
    "toothbrush": {
        "room": "bathroom",
        "stage": "manipuland",
        "support": "bathroom counter",
    },
    "toothpaste": {
        "room": "bathroom",
        "stage": "manipuland",
        "support": "bathroom counter",
    },
    "towel": {"room": "bathroom", "stage": "manipuland", "support": "towel rack"},
    "mirror": {"room": "bathroom", "stage": "wall_mounted"},
    "shower": {"room": "bathroom", "stage": "furniture"},
    "toilet": {"room": "bathroom", "stage": "furniture"},
    "bathroom counter": {"room": "bathroom", "stage": "furniture"},
}

ROOM_DEFAULT_OBJECTS = {
    "bedroom": [
        ("bed", "furniture", True, 1, None),
        ("nightstand", "furniture", True, 1, None),
        ("wardrobe", "furniture", False, 1, None),
        ("desk", "furniture", True, 1, None),
        ("chair", "furniture", True, 1, None),
        ("laptop", "manipuland", True, 1, "desk"),
        ("book", "manipuland", True, 2, "nightstand"),
        ("phone", "manipuland", True, 1, "nightstand"),
        ("mug", "manipuland", False, 1, "desk"),
    ],
    "kitchen": [
        ("kitchen counter", "furniture", True, 1, None),
        ("sink", "furniture", True, 1, None),
        ("refrigerator", "furniture", True, 1, None),
        ("stove", "furniture", False, 1, None),
        ("coffee maker", "furniture", False, 1, None),
        ("cutting board", "manipuland", True, 1, "kitchen counter"),
        ("plate", "manipuland", True, 2, "kitchen counter"),
        ("mug", "manipuland", False, 1, "kitchen counter"),
    ],
    "livingroom": [
        ("sofa", "furniture", True, 1, None),
        ("coffee table", "furniture", True, 1, None),
        ("television", "furniture", False, 1, None),
        ("bookshelf", "furniture", False, 1, None),
        ("remote control", "manipuland", True, 1, "coffee table"),
        ("magazine", "manipuland", False, 2, "coffee table"),
        ("book", "manipuland", False, 2, "bookshelf"),
    ],
    "bathroom": [
        ("bathroom counter", "furniture", True, 1, None),
        ("sink", "furniture", True, 1, None),
        ("mirror", "wall_mounted", True, 1, None),
        ("toilet", "furniture", True, 1, None),
        ("shower", "furniture", False, 1, None),
        ("toothbrush", "manipuland", True, 1, "bathroom counter"),
        ("toothpaste", "manipuland", True, 1, "bathroom counter"),
        ("towel", "manipuland", True, 1, "towel rack"),
    ],
}

ROOM_ACTIONS = {
    "bedroom": [
        ("walk", "bedroom", "enter the bedroom"),
        ("touch", "lamp", "turn attention to lighting"),
        ("sit", "chair", "sit for the task"),
        ("lookat", "laptop", "focus on work or media"),
        ("touch", "book", "handle reading material"),
        ("touch", "phone", "check personal device"),
    ],
    "kitchen": [
        ("walk", "kitchen", "enter the kitchen"),
        ("open", "refrigerator", "check ingredients"),
        ("touch", "cutting board", "prepare food"),
        ("touch", "plate", "set up serving surface"),
        ("drink", "mug", "drink coffee or water"),
        ("close", "refrigerator", "finish kitchen access"),
    ],
    "livingroom": [
        ("walk", "livingroom", "enter the living room"),
        ("sit", "sofa", "sit down to relax"),
        ("touch", "remote control", "control media"),
        ("lookat", "television", "watch or check display"),
        ("touch", "magazine", "browse reading material"),
    ],
    "bathroom": [
        ("walk", "bathroom", "enter the bathroom"),
        ("touch", "toothbrush", "start hygiene routine"),
        ("touch", "toothpaste", "use toothpaste"),
        ("touch", "towel", "dry hands or face"),
        ("lookat", "mirror", "check appearance"),
    ],
}


def _format_time(minutes: int) -> str:
    minutes %= 24 * 60
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


def _minutes(value: str) -> int:
    hour, minute = map(int, value.split(":"))
    return hour * 60 + minute


def _explicit_rooms_from_prompt(prompt: str, fallback_room: str | None) -> list[str]:
    prompt_lower = prompt.lower()
    rooms = [
        room
        for room, aliases in ROOM_ALIASES.items()
        if any(
            re.search(rf"\b{re.escape(alias)}s?\b", prompt_lower) for alias in aliases
        )
    ]
    normalized_fallback = re.sub(r"[\s_-]+", "", str(fallback_room or "").lower())
    if not rooms and normalized_fallback in ROOM_ALIASES:
        rooms.append(normalized_fallback)
    return rooms


def _quantity(prompt: str, object_name: str) -> int | None:
    match = re.search(
        rf"\b(one|two|three|four|five|six|seven|eight|nine|\d+)\s+{re.escape(object_name)}s?\b",
        prompt.lower(),
    )
    if not match:
        return None
    numbers = {
        "one": 1,
        "two": 2,
        "three": 3,
        "four": 4,
        "five": 5,
        "six": 6,
        "seven": 7,
        "eight": 8,
        "nine": 9,
    }
    value = match.group(1)
    return int(value) if value.isdigit() else numbers[value]


def _persona() -> PersonaSpec:
    return PersonaSpec(
        name="Maya",
        age=31,
        job="remote creative technologist",
        health="mild eye strain from screen-heavy work",
        traits=["introverted", "organized", "imaginative", "comfort-seeking"],
        description=(
            "Maya, she is 31 years old, a remote creative technologist, and has "
            "the following health situation: mild eye strain from screen-heavy work. "
            "She spends long stretches at home, keeps practical routines, and "
            "prefers rooms that support focused work, quiet recovery, reading, "
            "simple meals, and small rituals that make the space feel personal."
        ),
    )


def _extract_json_object(text: str) -> dict[str, Any]:
    match = re.search(r"\{.*\}", str(text), flags=re.DOTALL)
    return json.loads(match.group(0) if match else str(text))


def _day_schedule(day: str, day_index: int, rooms: list[str]) -> list[ScheduleActivity]:
    weekend = day in {"Saturday", "Sunday"}
    cursor = 7 * 60 + (60 if weekend else 0) + (day_index % 3) * 7
    result: list[ScheduleActivity] = []

    def add(
        name: str, room: str | None, duration: int, location: str = "at_home"
    ) -> None:
        nonlocal cursor
        start, end = _format_time(cursor), _format_time(cursor + duration)
        result.append(
            ScheduleActivity(
                activity=name,
                start=start,
                end=end,
                room=room,
                location=location,
                detail_level="fine" if room in rooms else "coarse",
            )
        )
        cursor += duration

    if "bedroom" in rooms:
        add("wake_up_and_check_phone", "bedroom", 18)
    if "bathroom" in rooms:
        add("brush_teeth_and_wash_face", "bathroom", 20)
    if "kitchen" in rooms:
        add("prepare_breakfast_and_coffee", "kitchen", 35)
    else:
        add("simple_breakfast_elsewhere", None, 20, "home_other")
    if weekend:
        if "livingroom" in rooms:
            add("read_or_watch_show_on_sofa", "livingroom", 70)
        if "bedroom" in rooms:
            add("organize_clothes_and_room", "bedroom", 45)
        add("errands_or_walk_outside", None, 120, "outside")
    elif "bedroom" in rooms:
        add("remote_work_at_desk", "bedroom", 180)
        add("outside_break_or_commute", None, 60, "outside")
    else:
        add("work_or_study_at_home", None, 150, "home_other")
        add("outside_break_or_commute", None, 60, "outside")
    if "kitchen" in rooms:
        add("cook_simple_dinner", "kitchen", 55)
    if "livingroom" in rooms:
        add("relax_with_media_or_reading", "livingroom", 80)
    if "bathroom" in rooms:
        add("evening_hygiene", "bathroom", 18)
    if "bedroom" in rooms:
        add("read_before_sleep", "bedroom", 45)
        add("sleep", "bedroom", 480)
    else:
        add("sleep", None, 480, "home_other")
    return result


def _expand_steps(activity: ScheduleActivity) -> list[ActionStep]:
    if not activity.room:
        return []
    templates = ROOM_ACTIONS.get(activity.room, [])
    start, end = _minutes(activity.start), _minutes(activity.end)
    if end <= start:
        end += 24 * 60
    span = max(1, end - start)
    return [
        ActionStep(
            step=index + 1,
            action=action,
            object=object_name,
            start=_format_time(start + round(span * index / len(templates))),
            end=_format_time(start + round(span * (index + 1) / len(templates))),
            note=f"{note} during {activity.activity}",
        )
        for index, (action, object_name, note) in enumerate(templates)
    ]


def _assets(
    prompt: str,
    room: str,
    schedule: dict[str, list[DetailedActivity]],
    *,
    source_compatible_matching: bool = False,
) -> list[AssetNeed]:
    activities = sorted(
        {item.activity for values in schedule.values() for item in values}
    )
    merged: dict[tuple[str, str | None], AssetNeed] = {}
    for name, stage, required, quantity, support in ROOM_DEFAULT_OBJECTS[room]:
        merged[(name, support)] = AssetNeed(
            name=name,
            stage=STAGE_OVERRIDES.get(name, stage),
            role="furniture" if stage != "manipuland" else "manipuland",
            required=required,
            quantity=quantity,
            support_target=support,
            reason="needed by generated weekly behavior",
            source="behavior_template",
            source_activities=activities,
        )
    prompt_lower = prompt.lower()
    for label, meta in KNOWN_OBJECTS.items():
        if source_compatible_matching:
            matched = label in prompt_lower
        else:
            plural_alias = f"{label}s"
            if plural_alias in KNOWN_OBJECTS and re.search(
                rf"\b{re.escape(plural_alias)}\b", prompt_lower
            ):
                continue
            matched = bool(re.search(rf"\b{re.escape(label)}s?\b", prompt_lower))
        if not matched or meta["room"] != room:
            continue
        name = meta.get("canonical", label)
        support = meta.get("support")
        key = (name, support)
        explicit_quantity = _quantity(prompt, label) or 1
        if key in merged:
            current = merged[key]
            merged[key] = current.model_copy(
                update={
                    "required": True,
                    "quantity": max(current.quantity, explicit_quantity),
                    "reason": "explicitly mentioned in the SceneSmith prompt",
                    "source": "explicit_prompt",
                }
            )
        else:
            merged[key] = AssetNeed(
                name=name,
                stage=meta["stage"],
                role=("furniture" if meta["stage"] != "manipuland" else "manipuland"),
                required=True,
                quantity=explicit_quantity,
                support_target=support,
                reason="explicitly mentioned in the SceneSmith prompt",
                source="explicit_prompt",
            )
    return sorted(
        merged.values(), key=lambda item: (item.stage, not item.required, item.name)
    )


def _relations(room: str, assets: list[AssetNeed]) -> list[BehaviorRelation]:
    relations = [
        BehaviorRelation(
            subject=item.name,
            relation="on",
            object=item.support_target,
            required=item.required,
            reason=item.reason,
        )
        for item in assets
        if item.support_target
    ]
    if room == "bedroom":
        relations.append(
            BehaviorRelation(
                subject="chair",
                relation="facing",
                object="desk",
                required=True,
                reason="remote work routine requires usable workstation",
            )
        )
    elif room == "livingroom":
        relations.append(
            BehaviorRelation(
                subject="sofa",
                relation="facing",
                object="coffee table or television",
                required=True,
                reason="relaxation routine requires reachable media area",
            )
        )
    return relations


def _format_item(item: AssetNeed | ObjectNeed) -> str:
    label = f"{item.quantity}x {item.name}" if item.quantity != 1 else item.name
    return f"{label} on {item.support_target}" if item.support_target else label


def _format_room_behavior_block(
    persona: PersonaSpec,
    room: str,
    routines: dict[str, list[DetailedActivity]],
    assets: list[AssetNeed | ObjectNeed],
    relations: list[BehaviorRelation],
) -> str:
    weekly_activities = [
        f"{day}: {activity.activity} ({activity.start}-{activity.end})"
        for day in DAYS
        for activity in routines.get(day, [])[:2]
    ]
    required_furniture = [
        item for item in assets if item.role == "furniture" and item.required
    ]
    optional_furniture = [
        item for item in assets if item.role == "furniture" and not item.required
    ]
    required_small = [
        item for item in assets if item.role == "manipuland" and item.required
    ]
    optional_small = [
        item for item in assets if item.role == "manipuland" and not item.required
    ]

    def joined(items: list[AssetNeed | ObjectNeed]) -> str:
        return ", ".join(_format_item(item) for item in items) or "none"

    required_relations = "; ".join(
        f"{relation.subject} {relation.relation} {relation.object}"
        for relation in relations
        if relation.required
    )
    return "\n".join(
        [
            "",
            "AGENTSENSE_ROOM_BEHAVIOR_REQUIREMENTS:",
            f"Room: {'living room' if room == 'livingroom' else room}",
            f"Resident behavior context: {persona.description.strip()}",
            "Fine-grained weekly activities in this room: "
            + ("; ".join(weekly_activities[:12]) or "none"),
            f"REQUIRED furniture/assets: {joined(required_furniture)}",
            "Furniture inventory rule: generate/place each REQUIRED furniture "
            "category to the listed quantity exactly once; during later revisions "
            "add only missing categories or missing quantities and do not duplicate "
            "existing required furniture.",
            f"Optional furniture/assets: {joined(optional_furniture)}",
            f"REQUIRED small objects/manipulands: {joined(required_small)}",
            f"Optional small objects/manipulands: {joined(optional_small)}",
            "Required spatial/functional relations: " + (required_relations or "none"),
            "Treat REQUIRED items and relations as hard scene requirements unless "
            "physically impossible.",
        ]
    )


class TemplateBehaviorPlanner:
    """Build the latest deterministic AgentSense-style weekly behavior plan."""

    def __init__(
        self,
        *,
        model: str | None = None,
        api_base_url: str | None = None,
        api_key: str | None = None,
        client: Any | None = None,
    ) -> None:
        self._model = model
        self._client = client
        if self._client is None and model:
            from openai import OpenAI

            self._client = OpenAI(
                base_url=api_base_url or "http://localhost:8000/v1",
                api_key=api_key or "dummy",
            )

    def _detect_rooms(self, prompt: str, fallback_room: str | None) -> list[str]:
        detected = _explicit_rooms_from_prompt(prompt, fallback_room)
        if detected:
            return detected
        if self._client is None or not self._model:
            return ["bedroom"]
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Return only JSON. Identify indoor room types in the "
                            "prompt."
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            "Supported room types are bedroom, kitchen, livingroom, "
                            "bathroom. If no room is explicit, infer the single best "
                            f"room. Prompt: {prompt}\n"
                            'Return format: {"rooms": ["bedroom"]}'
                        ),
                    },
                ],
                temperature=0.1,
                max_tokens=128,
                extra_body=chat_template_kwargs_from_effort("none"),
            )
            message = response.choices[0].message
            raw = message.content or getattr(message, "reasoning_content", None)
            if not raw:
                extra = getattr(message, "model_extra", None)
                raw = (
                    extra.get("reasoning_content") if isinstance(extra, dict) else None
                )
            data = _extract_json_object(raw or "")
            rooms = [
                _explicit_rooms_from_prompt("", str(value))[0]
                for value in data.get("rooms", [])
                if _explicit_rooms_from_prompt("", str(value))
            ]
            return list(dict.fromkeys(rooms)) or ["bedroom"]
        except Exception:  # Room detection must never block offline generation.
            return ["bedroom"]

    def _generate_persona(
        self, prompt: str, rooms: list[str]
    ) -> tuple[PersonaSpec, str]:
        fallback = _persona()
        if self._client is None or not self._model:
            return fallback, "fallback"
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Create one realistic resident persona for behavior-driven "
                            "indoor scene generation. Return only JSON."
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            f"Scene prompt: {prompt}\n"
                            f"Rooms that will be generated: {', '.join(rooms)}\n"
                            "Return JSON with keys: name, age, job, health, traits, "
                            "description. The description should explain lifestyle "
                            "habits that create object needs."
                        ),
                    },
                ],
                temperature=0.1,
                max_tokens=512,
                extra_body=chat_template_kwargs_from_effort("none"),
            )
            message = response.choices[0].message
            raw = message.content or getattr(message, "reasoning_content", None)
            if not raw:
                extra = getattr(message, "model_extra", None)
                raw = (
                    extra.get("reasoning_content") if isinstance(extra, dict) else None
                )
            data = _extract_json_object(raw or "")
            return (
                PersonaSpec.model_validate({**fallback.model_dump(), **data}),
                "model",
            )
        except Exception:  # Provider and malformed-output failures degrade safely.
            return fallback, "fallback"

    def plan(
        self,
        prompt: str,
        *,
        room_type: str | None = None,
        horizon: str = "week",
    ) -> BehaviorSpec:
        if horizon != "week":
            raise ValueError(
                "Only horizon='week' is supported by the template planner."
            )
        rooms = self._detect_rooms(prompt, room_type)
        persona, persona_generation = self._generate_persona(prompt, rooms)
        weekly_schedule = {
            day: _day_schedule(day, day_index, rooms)
            for day_index, day in enumerate(DAYS)
        }
        room_specs = []
        legacy_assets_by_room: dict[str, list[AssetNeed]] = {}
        for room in rooms:
            room_schedule: dict[str, list[DetailedActivity]] = {}
            for day, activities in weekly_schedule.items():
                room_schedule[day] = [
                    DetailedActivity(
                        activity=item.activity,
                        start=item.start,
                        end=item.end,
                        steps=_expand_steps(item),
                    )
                    for item in activities
                    if item.room == room
                ]
            assets = _assets(prompt, room, room_schedule)
            legacy_assets = _assets(
                prompt,
                room,
                room_schedule,
                source_compatible_matching=True,
            )
            legacy_assets_by_room[room] = legacy_assets
            grouped = {
                stage: [item for item in assets if item.stage == stage]
                for stage in (
                    "furniture",
                    "wall_mounted",
                    "ceiling_mounted",
                    "manipuland",
                )
            }
            room_specs.append(
                RoomBehaviorSpec(
                    room_type=room,
                    weekly_schedule=room_schedule,
                    assets_by_stage=grouped,
                    relations=_relations(room, assets),
                )
            )
        object_needs = {
            room.room_type: sorted(
                [
                    ObjectNeed(
                        name=item.name,
                        role=item.role,
                        required=item.required,
                        quantity=item.quantity,
                        support_target=item.support_target,
                        reason=item.reason,
                        source_activities=item.source_activities,
                    )
                    for item in legacy_assets_by_room[room.room_type]
                ],
                key=lambda item: (
                    item.role != "furniture",
                    not item.required,
                    item.name,
                ),
            )
            for room in room_specs
        }
        assets_by_room_and_stage = {
            room.room_type: room.assets_by_stage for room in room_specs
        }
        placement_relations = {
            room: _relations(room, legacy_assets_by_room[room]) for room in rooms
        }
        detailed_routines = {
            room.room_type: room.weekly_schedule for room in room_specs
        }
        room_behavior_blocks = {
            room.room_type: _format_room_behavior_block(
                persona,
                room.room_type,
                room.weekly_schedule,
                object_needs[room.room_type],
                placement_relations[room.room_type],
            )
            for room in room_specs
        }
        enriched_prompt = (
            f"{prompt}\n\n"
            "This scene should be generated for a single long-term resident inferred "
            "by AgentSense. Use the behavior requirements below to make the space "
            "functional for a full week of life, not just a sparse static prompt.\n"
            f"Resident: {persona.description}\n"
            f"Target rooms: {', '.join(ROOM_LABELS[room] for room in rooms)}."
            + "\n".join(room_behavior_blocks[room] for room in rooms)
        )
        return BehaviorSpec(
            scene_prompt=prompt,
            target_rooms=rooms,
            persona=persona,
            persona_generation=persona_generation,
            weekly_schedule=weekly_schedule,
            detailed_routines=detailed_routines,
            object_needs=object_needs,
            assets_by_room_and_stage=assets_by_room_and_stage,
            placement_relations=placement_relations,
            room_behavior_blocks=room_behavior_blocks,
            enriched_prompt=enriched_prompt,
            rooms=room_specs,
        )


def build_behavior_spec(
    scene_prompt: str,
    *,
    model: str | None = None,
    api_base_url: str | None = None,
    api_key: str | None = None,
    client: Any | None = None,
    room_type: str | None = None,
    horizon: str = "week",
) -> BehaviorSpec:
    """Build a directly serializable prompt-to-grouped-assets behavior spec."""
    return TemplateBehaviorPlanner(
        model=model,
        api_base_url=api_base_url,
        api_key=api_key,
        client=client,
    ).plan(scene_prompt, room_type=room_type, horizon=horizon)
