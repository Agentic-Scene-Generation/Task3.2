#!/usr/bin/env python3
"""Build a paired HSSD orientation ablation for SceneBenchmark critic.

The script reuses a rendered furniture-stage state rather than regenerating
assets. Each variant therefore changes only yaw, then sends identical fragment
renders to Qwen twice. The critic-on prompt includes the scoped deterministic
SceneBenchmark context; the critic-off prompt omits it. Qwen feedback remains
non-authoritative evidence and never changes the SceneBenchmark gate.

Example:
    .venv/bin/python scripts/run_orientation_critic_ablation.py

    # Re-run only critic and manifests while iterating on the report layout:
    .venv/bin/python scripts/run_orientation_critic_ablation.py --no-render --no-vlm
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf
from openai import OpenAI
from pydrake.math import RigidTransform, RollPitchYaw

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scenesmith.agent_utils.blender.server_manager import BlenderServer
from scenesmith.agent_utils.rendering import render_scene_for_agent_observation
from scenesmith.agent_utils.room import RoomScene
from scenesmith.scenebenchmark_critic.api import (
    evaluate_room_scene,
    format_prompt_context,
)
from scenesmith.scenebenchmark_critic.config import CriticConfig
from scenesmith.scenebenchmark_critic.intent_contract import SCHEMA_VERSION
from scenesmith.scenebenchmark_critic.reports import write_report


DEFAULT_SOURCE_STATE = (
    PROJECT_ROOT
    / "outputs/tmp/scene044_timeout_repro_20260829_195016/critic_on/batch_045"
    / "hydra/scene_044/room_living_room/scene_renders/furniture/renders_003"
    / "scene_state.json"
)
DEFAULT_MODEL = "unsloth/Qwen3.8-27B-GGUF"
DEFAULT_VLM_BASE_URL = "http://127.0.0.1:8002/v1"


@dataclass(frozen=True)
class Variant:
    name: str
    title: str
    target_ids: tuple[str, ...]
    yaw_delta_degrees: float
    task_requirement: str
    expected_orientation: str
    fragment_prompt: str


VARIANTS = (
    Variant(
        name="baseline",
        title="Baseline layout",
        target_ids=(),
        yaw_delta_degrees=0.0,
        task_requirement=(
            "Each dining chair must face the dining table, and the sofa must face "
            "the television."
        ),
        expected_orientation="All dining chairs face the dining table and the sofa faces the TV.",
        fragment_prompt=(
            "Inspect the dining table, its four chairs, the sofa, and the TV. "
            "Do the chairs face the dining table, and does the sofa face the TV?"
        ),
    ),
    Variant(
        name="chairs_away_from_table",
        title="Four dining chairs rotated away from the table",
        target_ids=(
            "dining_chair_0",
            "dining_chair_1",
            "dining_chair_2",
            "dining_chair_3",
        ),
        yaw_delta_degrees=180.0,
        task_requirement="Each dining chair must face the dining table.",
        expected_orientation="Dining chairs face away from the dining table.",
        fragment_prompt=(
            "Focus only on the dining table and its four chairs. "
            "Does each chair face the table?"
        ),
    ),
    Variant(
        name="sofa_away_from_tv",
        title="Sofa rotated away from the TV",
        target_ids=("sofa_0",),
        yaw_delta_degrees=180.0,
        task_requirement="The sofa must face the television.",
        expected_orientation="The sofa faces away from the TV.",
        fragment_prompt=(
            "Focus only on the sofa, coffee table, TV stand, and television. "
            "Does the sofa face the television?"
        ),
    ),
    Variant(
        name="combined_bad_orientation",
        title="Dining chairs and sofa rotated away from their functional targets",
        target_ids=(
            "dining_chair_0",
            "dining_chair_1",
            "dining_chair_2",
            "dining_chair_3",
            "sofa_0",
        ),
        yaw_delta_degrees=180.0,
        task_requirement=(
            "Each dining chair must face the dining table, and the sofa must face "
            "the television."
        ),
        expected_orientation="Dining chairs face away from the table and sofa faces away from the TV.",
        fragment_prompt=(
            "Inspect the dining table and chairs, then the sofa and television. "
            "Are the chairs oriented toward the dining table and is the sofa oriented toward the TV?"
        ),
    ),
    Variant(
        name="single_chair_away_from_table",
        title="One dining chair rotated away from the table",
        target_ids=("dining_chair_0",),
        yaw_delta_degrees=180.0,
        task_requirement="Each dining chair must face the dining table.",
        expected_orientation="One dining chair faces away while the other three remain correct.",
        fragment_prompt=(
            "Focus only on the dining table and its four chairs. "
            "Does every chair face the table?"
        ),
    ),
    Variant(
        name="chairs_sideways_to_table",
        title="Four dining chairs rotated sideways to the table",
        target_ids=(
            "dining_chair_0",
            "dining_chair_1",
            "dining_chair_2",
            "dining_chair_3",
        ),
        yaw_delta_degrees=90.0,
        task_requirement="Each dining chair must face the dining table.",
        expected_orientation="Dining chairs are sideways rather than facing the table.",
        fragment_prompt=(
            "Focus only on the dining table and its four chairs. "
            "Does each chair face the table?"
        ),
    ),
    Variant(
        name="sofa_sideways_to_tv",
        title="Sofa rotated sideways to the TV",
        target_ids=("sofa_0",),
        yaw_delta_degrees=90.0,
        task_requirement="The sofa must face the television.",
        expected_orientation="The sofa is sideways rather than facing the TV.",
        fragment_prompt=(
            "Focus only on the sofa, coffee table, TV stand, and television. "
            "Does the sofa face the television?"
        ),
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-state", type=Path, default=DEFAULT_SOURCE_STATE)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Default: outputs/tmp/critic_orientation_ablation_<timestamp>.",
    )
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--no-vlm", action="store_true")
    parser.add_argument(
        "--vlm-only-output-root",
        type=Path,
        default=None,
        help="Reuse existing fragment_renders and update VLM evidence without rerendering.",
    )
    parser.add_argument("--vlm-base-url", default=DEFAULT_VLM_BASE_URL)
    parser.add_argument("--vlm-model", default=DEFAULT_MODEL)
    parser.add_argument("--vlm-timeout", type=float, default=120.0)
    parser.add_argument("--resolution", type=int, default=768)
    parser.add_argument(
        "--variants",
        default="",
        help="Comma-separated variant names. Default: run every built-in variant.",
    )
    return parser.parse_args()


def source_room_dir(source_state: Path) -> Path:
    """Return the room root for a furniture render state path."""
    expected = ("scene_renders", "furniture")
    parts = source_state.parts
    for index in range(len(parts) - 1):
        if parts[index : index + 2] == expected:
            return Path(*parts[:index])
    raise ValueError(f"Expected a furniture render state path, got {source_state}")


def make_output_root(value: Path | None) -> Path:
    if value is not None:
        return value.resolve()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return PROJECT_ROOT / "outputs/tmp" / f"critic_orientation_ablation_{timestamp}"


def load_scene(source_state: Path, case_dir: Path) -> RoomScene:
    """Restore the persisted HSSD room state with absolute asset paths."""
    state = json.loads(source_state.read_text(encoding="utf-8"))
    source_dir = source_room_dir(source_state)
    scene = RoomScene(room_geometry=None, scene_dir=source_dir, room_id="living_room")
    scene.restore_from_state_dict(state)
    # Resolve every relative HSSD asset against the source before changing the
    # artifact destination to this case. This makes copied states self-describing.
    for obj in scene.objects.values():
        for attr in ("geometry_path", "sdf_path", "image_path"):
            path = getattr(obj, attr, None)
            if path is not None:
                setattr(obj, attr, Path(path).resolve())
    if scene.room_geometry and scene.room_geometry.sdf_path:
        scene.room_geometry.sdf_path = Path(scene.room_geometry.sdf_path).resolve()
    scene.scene_dir = case_dir
    scene.room_id = "orientation_ablation_living_room"
    scene.room_type = "living_room"
    _attach_orientation_contract(scene)
    return scene


def _attach_orientation_contract(scene: RoomScene) -> None:
    text = (
        "A living room with a dining table and four dining chairs facing the table, "
        "plus a sofa facing a television on a TV stand."
    )
    scene.text_description = text
    contract = {
        "schema_version": SCHEMA_VERSION,
        "prompt_sha256": hashlib.sha256(
            " ".join(text.split()).encode("utf-8")
        ).hexdigest(),
        "constraints": [
            {
                "constraint_id": "chairs_face_dining_table",
                "relation": "across_from",
                "stage": "furniture",
                "strength": "hard",
                "subjects": {"category": "dining_chair", "count": 4},
                "targets": {"category": "dining_table", "count": 1},
                "source": "explicit_prompt",
                "evidence_span": "four dining chairs facing the table",
            },
            {
                "constraint_id": "sofa_faces_television",
                "relation": "across_from",
                "stage": "furniture",
                "strength": "hard",
                "subjects": {"category": "sofa", "count": 1},
                "targets": {"category": "tv_stand", "count": 1},
                "source": "explicit_prompt",
                "evidence_span": "sofa facing a television on a TV stand",
            },
        ],
    }
    scene.metadata["scenebenchmark_intent_contract"] = contract
    scene.scenebenchmark_intent_contract = contract


def rotate_yaw(
    scene: RoomScene, object_id: str, yaw_delta_degrees: float
) -> dict[str, Any]:
    obj = scene.objects.get(object_id)
    if obj is None:
        raise KeyError(f"Source scene does not contain target {object_id}")
    before = obj.transform.rotation().ToQuaternion()
    yaw = RollPitchYaw(obj.transform.rotation()).yaw_angle()
    obj.transform = RigidTransform(
        rpy=RollPitchYaw(0.0, 0.0, yaw + math.radians(yaw_delta_degrees)),
        p=obj.transform.translation(),
    )
    after = obj.transform.rotation().ToQuaternion()
    return {
        "object_id": object_id,
        "yaw_delta_degrees": yaw_delta_degrees,
        "translation": [float(v) for v in obj.transform.translation()],
        "rotation_before_wxyz": [before.w(), before.x(), before.y(), before.z()],
        "rotation_after_wxyz": [after.w(), after.x(), after.y(), after.z()],
    }


def apply_variant(scene: RoomScene, variant: Variant) -> list[dict[str, Any]]:
    return [
        rotate_yaw(scene, object_id, variant.yaw_delta_degrees)
        for object_id in variant.target_ids
    ]


def selected_variants(raw_names: str) -> tuple[Variant, ...]:
    if not raw_names.strip():
        return VARIANTS
    variants_by_name = {variant.name: variant for variant in VARIANTS}
    names = [name.strip() for name in raw_names.split(",") if name.strip()]
    unknown = [name for name in names if name not in variants_by_name]
    if unknown:
        raise ValueError(f"Unknown orientation ablation variants: {', '.join(unknown)}")
    return tuple(variants_by_name[name] for name in names)


def render_config(resolution: int):
    return OmegaConf.create(
        {
            "layout": "top_plus_sides",
            "top_view_width": resolution,
            "top_view_height": resolution,
            "side_view_count": 2,
            "side_view_width": resolution,
            "side_view_height": resolution,
            "background_color": [1.0, 1.0, 1.0],
            "annotations": {
                "enable_set_of_mark_labels": False,
                "enable_bounding_boxes": False,
                "enable_direction_arrows": False,
                "enable_partial_walls": True,
                "enable_support_surface_debug": False,
                "enable_convex_hull_debug": False,
            },
        }
    )


def render_case(
    scene: RoomScene,
    case_dir: Path,
    resolution: int,
    *,
    render_name: str,
    include_object_ids: list[str] | None = None,
) -> list[str]:
    render_dir = case_dir / render_name
    server = BlenderServer(
        port_range=(8400, 8450),
        server_startup_delay=0.2,
        port_cleanup_delay=0.2,
        log_file=case_dir / "blender_server.log",
    )
    server.start()
    try:
        server.wait_until_ready(timeout=60)
        rendered = render_scene_for_agent_observation(
            scene=scene,
            cfg=render_config(resolution),
            blender_server=server,
            rendering_mode="furniture_selection",
            taa_samples=4,
            include_objects=include_object_ids,
        )
        render_dir.mkdir(parents=True, exist_ok=True)
        paths: list[str] = []
        for source in rendered:
            destination = render_dir / source.name
            shutil.copy2(source, destination)
            paths.append(str(destination.relative_to(case_dir)))
        return paths
    finally:
        server.stop()


def fragment_object_ids(variant: Variant) -> list[str]:
    """Return only the furniture necessary to judge a variant's orientation."""
    dining_ids = [
        "dining_table_0",
        "dining_chair_0",
        "dining_chair_1",
        "dining_chair_2",
        "dining_chair_3",
    ]
    media_ids = ["sofa_0", "coffee_table_0", "tv_stand_0", "television_0"]
    if "chair" in variant.name and "table" in variant.name:
        return dining_ids
    if "sofa" in variant.name and "tv" in variant.name:
        return media_ids
    return dining_ids + media_ids


def image_data_url(image_path: Path) -> str:
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def query_vlm(
    *,
    image_paths: list[Path],
    prompt: str,
    base_url: str,
    model: str,
    timeout: float,
) -> dict[str, Any]:
    """Ask Qwen for a narrow observation; failure remains explicit evidence."""
    content: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": (
                "You are observing a rendered room fragment. "
                + prompt
                + " Return strict JSON only: "
                '{"chairs_face_table":"yes|no|uncertain|not_visible",'
                '"sofa_faces_tv":"yes|no|uncertain|not_visible",'
                '"evidence":"short visual description"}. '
                "Do not infer from object labels or assumed intent."
            ),
        }
    ]
    content.extend(
        {"type": "image_url", "image_url": {"url": image_data_url(path)}}
        for path in image_paths
    )
    client = OpenAI(base_url=base_url, api_key="sk-123", timeout=timeout, max_retries=0)
    raw = ""
    reasoning = ""
    finish_reason = ""
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": content}],
            temperature=0.0,
            # Qwen3.8 reserves this budget for hidden reasoning plus final JSON.
            max_tokens=1024,
            # The local Qwen3.8 chat template accepts low/medium/xhigh only.
            extra_body={"chat_template_kwargs": {"reasoning_effort": "low"}},
        )
        choice = response.choices[0]
        raw = choice.message.content or ""
        reasoning = str(getattr(choice.message, "reasoning_content", "") or "")
        finish_reason = str(choice.finish_reason or "")
        return {
            "schema_version": "orientation_ablation.vlm_evidence.v1",
            "authority": "evidence_only",
            "status": "success",
            "model": model,
            "base_url": base_url,
            "prompt": prompt,
            "image_paths": [str(path) for path in image_paths],
            "raw_response": raw,
            "reasoning_content": reasoning,
            "finish_reason": finish_reason,
            "parsed_response": json.loads(raw),
        }
    except Exception as exc:
        return {
            "schema_version": "orientation_ablation.vlm_evidence.v1",
            "authority": "evidence_only",
            "model": model,
            "base_url": base_url,
            "prompt": prompt,
            "image_paths": [str(path) for path in image_paths],
            "status": "unavailable",
            "raw_response": raw,
            "reasoning_content": reasoning,
            "finish_reason": finish_reason,
            "error": f"{type(exc).__name__}: {exc}",
        }


def scoped_critic_prompt_context(payload: dict[str, Any], variant: Variant) -> str:
    """Format only the deterministic failures owned by an ablation variant."""
    target_ids = set(variant.target_ids)
    scoped_results = [
        result
        for result in payload.get("results") or []
        if str(result.get("primary_object") or "") in target_ids
    ]
    return format_prompt_context(
        {
            "results": scoped_results,
            "case_pack": {
                "intent_contract": (
                    (payload.get("case_pack") or {}).get("intent_contract") or {}
                )
            },
        },
        max_issues=4,
    )


def critic_llm_prompt(
    *,
    variant: Variant,
    benchmark_context: str | None,
) -> str:
    """Build matched Qwen critic prompts with optional deterministic evidence."""
    prompt = (
        "You are reviewing one rendered room fragment as a furniture critic. "
        f"Task requirement: {variant.task_requirement} "
        + variant.fragment_prompt
        + " Return strict JSON only: "
        '{"verdict":"pass|fail|uncertain","feedback":"short critique"}. '
        "Judge only the requested orientation relationship."
    )
    if benchmark_context:
        prompt += (
            "\n\nAdditional SceneBenchmark geometry critic context:\n"
            + benchmark_context
            + "\n\nThe deterministic feedback is authoritative for this "
            "functional orientation. Do not overturn a listed hard failure "
            "based on visual appearance alone."
        )
    return prompt


def query_critic_llm(
    *,
    image_paths: list[Path],
    variant: Variant,
    condition: str,
    benchmark_context: str | None,
    base_url: str,
    model: str,
    timeout: float,
) -> dict[str, Any]:
    """Run Qwen against identical renders with or without critic context."""
    prompt = critic_llm_prompt(
        variant=variant,
        benchmark_context=benchmark_context,
    )
    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    content.extend(
        {"type": "image_url", "image_url": {"url": image_data_url(path)}}
        for path in image_paths
    )
    client = OpenAI(base_url=base_url, api_key="sk-123", timeout=timeout, max_retries=0)
    raw = ""
    reasoning = ""
    finish_reason = ""
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": content}],
            temperature=0.0,
            max_tokens=1024,
            extra_body={"chat_template_kwargs": {"reasoning_effort": "low"}},
        )
        choice = response.choices[0]
        raw = choice.message.content or ""
        reasoning = str(getattr(choice.message, "reasoning_content", "") or "")
        finish_reason = str(choice.finish_reason or "")
        return {
            "schema_version": "orientation_ablation.critic_llm_feedback.v1",
            "condition": condition,
            "status": "success",
            "model": model,
            "base_url": base_url,
            "prompt": prompt,
            "benchmark_context": benchmark_context,
            "image_paths": [str(path) for path in image_paths],
            "raw_response": raw,
            "reasoning_content": reasoning,
            "finish_reason": finish_reason,
            "parsed_response": json.loads(raw),
        }
    except Exception as exc:
        return {
            "schema_version": "orientation_ablation.critic_llm_feedback.v1",
            "condition": condition,
            "status": "unavailable",
            "model": model,
            "base_url": base_url,
            "prompt": prompt,
            "benchmark_context": benchmark_context,
            "image_paths": [str(path) for path in image_paths],
            "raw_response": raw,
            "reasoning_content": reasoning,
            "finish_reason": finish_reason,
            "error": f"{type(exc).__name__}: {exc}",
        }


def critic_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    interesting = {
        "dining_set",
        "edge_distribution",
        "seating_to_media",
        "seating_to_work_surface",
    }
    return [
        {
            "check_id": row.get("check_id"),
            "relation_type": row.get("relation_type"),
            "label": row.get("label"),
            "primary_object": row.get("primary_object"),
            "related_objects": row.get("related_objects"),
            "reason": row.get("reason"),
        }
        for row in payload.get("results") or []
        if row.get("relation_type") in interesting
    ]


def write_summary(output_root: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Orientation Critic Ablation",
        "",
        "All variants reuse the same HSSD source scene. Only listed object yaw changes; ",
        "the deterministic critic and paired Qwen critic feedback are reported independently.",
        "",
        "| Variant | Intended perturbation | Critic on | Critic off | Qwen on | Qwen off | Top view | Fragment |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        case = row["name"]
        summary = row["critic_on_summary"]
        llm_on = row.get("llm_on") or {}
        llm_off = row.get("llm_off") or {}
        llm_on_text = json.dumps(
            llm_on.get("parsed_response") or llm_on.get("status") or "not run"
        )
        llm_off_text = json.dumps(
            llm_off.get("parsed_response") or llm_off.get("status") or "not run"
        )
        images = row.get("render_paths") or []
        fragments = row.get("fragment_render_paths") or []
        top = next((path for path in images if path.endswith("0_top.png")), "")
        fragment = next((path for path in fragments if path.endswith("0_top.png")), "")
        lines.append(
            "| "
            + f"[{case}]({case}/manifest.json) | {row['expected_orientation']} | "
            + f"fail={summary.get('fail', 0)}, degraded={summary.get('degraded', 0)} | "
            + "disabled (same poses) | "
            + llm_on_text.replace("|", "\\|")
            + " | "
            + llm_off_text.replace("|", "\\|")
            + f" | [{top}]({case}/{top}) | [{fragment}]({case}/{fragment}) |"
        )
    lines.extend(
        [
            "",
            "`critic_off` deliberately skips `evaluate_room_scene`; both Qwen calls "
            "receive identical renders and differ only by the injected deterministic context.",
        ]
    )
    (output_root / "ppt_summary.md").write_text("\n".join(lines), encoding="utf-8")


def scene_summary(payload: dict[str, Any]) -> dict[str, Any]:
    return dict((payload.get("summary") or {}).get("scene_summary") or {})


def run_vlm_only(args: argparse.Namespace) -> None:
    output_root = args.vlm_only_output_root.resolve()
    if args.no_vlm:
        raise ValueError("--vlm-only-output-root cannot be combined with --no-vlm")
    if not output_root.is_dir():
        raise FileNotFoundError(f"Ablation output root not found: {output_root}")
    variants = {variant.name: variant for variant in VARIANTS}
    summary_rows: list[dict[str, Any]] = []
    for case_dir in sorted(path for path in output_root.iterdir() if path.is_dir()):
        manifest_path = case_dir / "manifest.json"
        if not manifest_path.exists():
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        variant = variants.get(str(manifest.get("variant")))
        if variant is None:
            raise ValueError(f"Unknown ablation variant in {manifest_path}")
        fragments = [
            path for path in sorted((case_dir / "fragment_renders").glob("*.png"))
        ]
        if not fragments:
            raise FileNotFoundError(f"No fragment renders found for {case_dir.name}")
        vlm = query_vlm(
            image_paths=fragments,
            prompt=variant.fragment_prompt,
            base_url=args.vlm_base_url,
            model=args.vlm_model,
            timeout=args.vlm_timeout,
        )
        (case_dir / "vlm_evidence.json").write_text(
            json.dumps(vlm, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        manifest["vlm_evidence"] = "vlm_evidence.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        summary_rows.append(
            {
                "name": variant.name,
                "expected_orientation": manifest["expected_orientation"],
                "render_paths": manifest.get("render_paths") or [],
                "fragment_render_paths": manifest.get("fragment_render_paths") or [],
                "critic_on_summary": (manifest.get("critic_on") or {}).get("summary")
                or {},
                "llm_on": vlm,
                "llm_off": {},
            }
        )
        print(f"completed VLM evidence for {variant.name}: {case_dir}")
    write_summary(output_root, summary_rows)
    print(f"PPT summary: {output_root / 'ppt_summary.md'}")


def main() -> None:
    args = parse_args()
    if args.vlm_only_output_root is not None:
        run_vlm_only(args)
        return
    source_state = args.source_state.resolve()
    if not source_state.exists():
        raise FileNotFoundError(f"Source state not found: {source_state}")
    output_root = make_output_root(args.output_root)
    if output_root.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing output root: {output_root}"
        )
    output_root.mkdir(parents=True)
    (output_root / "README.md").write_text(
        "This is a temporary HSSD orientation ablation. Critic-on and critic-off "
        "Qwen calls share identical fragment renders; only critic-on receives "
        "scoped deterministic context. Qwen feedback is evidence-only.\n",
        encoding="utf-8",
    )

    critic_config = CriticConfig(
        enabled=True,
        metrics=("functional_dependency",),
        hard_gate=True,
    )
    summary_rows: list[dict[str, Any]] = []
    for variant in selected_variants(args.variants):
        case_dir = output_root / variant.name
        case_dir.mkdir()
        scene = load_scene(source_state, case_dir)
        changes = apply_variant(scene, variant)
        state_path = case_dir / "scene_state.json"
        state_path.write_text(
            json.dumps(scene.to_state_dict(), indent=2), encoding="utf-8"
        )

        payload = evaluate_room_scene(
            scene, config=critic_config, stage="orientation_ablation"
        )
        write_report(case_dir / "critic_on", payload)
        disabled = {
            "schema_version": "orientation_ablation.critic_off.v1",
            "enabled": False,
            "evaluation_skipped": True,
            "same_scene_state": "../scene_state.json",
            "reason": "Ablation control: no deterministic critic execution.",
        }

        render_paths: list[str] = []
        fragment_render_paths: list[str] = []
        if not args.no_render:
            render_paths = render_case(
                scene, case_dir, args.resolution, render_name="renders"
            )
            fragment_render_paths = render_case(
                scene,
                case_dir,
                args.resolution,
                render_name="fragment_renders",
                include_object_ids=fragment_object_ids(variant),
            )
        llm_on: dict[str, Any] = {"status": "not_run"}
        llm_off: dict[str, Any] = {"status": "not_run"}
        prompt_context = scoped_critic_prompt_context(payload, variant)
        (case_dir / "critic_on" / "prompt_context.txt").write_text(
            prompt_context + "\n", encoding="utf-8"
        )
        if not args.no_vlm and fragment_render_paths:
            image_paths = [
                case_dir / path
                for path in fragment_render_paths
                if path.endswith(".png")
            ]
            llm_on = query_critic_llm(
                image_paths=image_paths,
                variant=variant,
                condition="critic_on",
                benchmark_context=prompt_context,
                base_url=args.vlm_base_url,
                model=args.vlm_model,
                timeout=args.vlm_timeout,
            )
            llm_off = query_critic_llm(
                image_paths=image_paths,
                variant=variant,
                condition="critic_off",
                benchmark_context=None,
                base_url=args.vlm_base_url,
                model=args.vlm_model,
                timeout=args.vlm_timeout,
            )
        (case_dir / "critic_on" / "critic_llm_feedback.json").write_text(
            json.dumps(llm_on, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        disabled["llm_feedback"] = "critic_off/critic_llm_feedback.json"
        (case_dir / "critic_off").mkdir()
        (case_dir / "critic_off" / "critic_llm_feedback.json").write_text(
            json.dumps(llm_off, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        (case_dir / "critic_off.json").write_text(
            json.dumps(disabled, indent=2), encoding="utf-8"
        )
        manifest = {
            "schema_version": "orientation_ablation.manifest.v1",
            "variant": variant.name,
            "title": variant.title,
            "source_state": str(source_state),
            "expected_orientation": variant.expected_orientation,
            "changes": changes,
            "render_paths": render_paths,
            "fragment_render_paths": fragment_render_paths,
            "critic_on": {
                "enabled": True,
                "report": "critic_on/scenebenchmark_critic.json",
                "prompt_context": "critic_on/prompt_context.txt",
                "llm_feedback": "critic_on/critic_llm_feedback.json",
                "summary": scene_summary(payload),
                "orientation_rows": critic_rows(payload),
            },
            "critic_off": disabled,
            "llm_comparison": {
                "same_fragment_renders": fragment_render_paths,
                "critic_on_feedback": "critic_on/critic_llm_feedback.json",
                "critic_off_feedback": "critic_off/critic_llm_feedback.json",
            },
        }
        (case_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        summary_rows.append(
            {
                "name": variant.name,
                "expected_orientation": variant.expected_orientation,
                "render_paths": render_paths,
                "fragment_render_paths": fragment_render_paths,
                "critic_on_summary": scene_summary(payload),
                "llm_on": llm_on,
                "llm_off": llm_off,
            }
        )
        print(f"completed {variant.name}: {case_dir}")
    write_summary(output_root, summary_rows)
    print(f"PPT summary: {output_root / 'ppt_summary.md'}")


if __name__ == "__main__":
    main()
