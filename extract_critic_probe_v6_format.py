#!/usr/bin/env python3
"""Extract critic probe data in v6.2 format with full training context.

This script extracts data from critic_probe runs and formats it to match
the v6.2 training standard with:
- System prompts (from system_prompts.py)
- Tool definitions (from tool_schemas.py)
- Images (extracted from observe_scene outputs)
- Rich metadata (source, agent, scene info)
- Thinking blocks (from agent_thinking table)
"""

from __future__ import annotations

import base64
import hashlib
import json
import re
import sqlite3
import sys
from collections import Counter, defaultdict, deque
from io import BytesIO
from pathlib import Path
from typing import Any

from PIL import Image

# Import system prompts and tool schemas from v2
V2_SRC = Path("/mnt/afs/visitor33/scenesmith/sft_data/v2")
sys.path.insert(0, str(V2_SRC))
import system_prompts as sp_mod
import tool_schemas as ts_mod

# Config
CRITIC_PROBE_BASE = Path("/mnt/afs/visitor33/Task3.2/outputs/critic_probe/cola_gpt55_scenebenchmark_no_code_repair_room_dense_acp_20260826_080514/critic_on")
OUTPUT_DIR = Path("/mnt/afs/visitor33/Task3.2/outputs/critic_probe/extracted_v6_format")
IMAGES_DIR = OUTPUT_DIR / "images"

THINK_RE = re.compile(r"^\s*<think>\s*(.*?)\s*</think>\s*", re.DOTALL)

# Critic tool whitelists (from v6 extractor)
CRITIC_TOOL_WHITELIST: dict[str, list[str]] = {
    "floor_plan_critic": ["observe_scene", "render_ascii", "validate"],
    "furniture_critic": ["observe_scene", "get_current_scene_state", "check_facing_tool"],
    "wall_critic": ["observe_scene", "get_current_scene_state"],
    "ceiling_critic": ["observe_scene", "get_current_scene_state"],
    "manipuland_critic": ["observe_scene", "get_current_scene_state"],
}

CRITIC_TO_DESIGNER: dict[str, str] = {
    "floor_plan_critic": "floor_plan_designer",
    "furniture_critic": "furniture_designer",
    "wall_critic": "wall_designer",
    "ceiling_critic": "ceiling_designer",
    "manipuland_critic": "manipuland_designer",
}

# Map agent DB names to critic labels
AGENT_DB_TO_LABEL = {
    "critic": "floor_plan_critic",
    "planner": "planner",
    "designer": "floor_plan_designer",
}


def _thinking_wrap(thinking: str | None) -> str:
    """Wrap thinking in <think> tags."""
    body = (thinking or "").strip()
    if not body:
        return ""
    return f"<think>\n{body}\n</think>\n\n"


def _read_db_messages_with_thinking(db_path: Path) -> list[dict[str, Any]]:
    """Read messages from DB and attach thinking from agent_thinking table."""
    if not db_path.exists():
        return []

    con = sqlite3.connect(str(db_path))
    try:
        msg_rows = list(
            con.execute(
                "SELECT id, session_id, message_data FROM agent_messages ORDER BY id"
            )
        )

        # Try to read thinking table
        try:
            thinking_rows = list(
                con.execute(
                    "SELECT session_id, thinking, content_preview "
                    "FROM agent_thinking ORDER BY id"
                )
            )
        except sqlite3.OperationalError:
            thinking_rows = []
    finally:
        con.close()

    out: list[dict[str, Any]] = []
    parsed_rows: list[tuple[str, dict[str, Any]]] = []

    for _msg_id, session_id, raw in msg_rows:
        try:
            msg = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            msg = {}
        parsed_rows.append((session_id, msg))
        out.append(msg)

    # Match thinking to messages
    message_events: dict[str, list[tuple[int, str]]] = defaultdict(list)
    call_events: dict[str, list[int]] = defaultdict(list)
    in_call_run: dict[str, bool] = defaultdict(bool)

    for idx, (session_id, msg) in enumerate(parsed_rows):
        rtype = msg.get("type")
        role = msg.get("role")

        if rtype == "function_call":
            if not in_call_run[session_id]:
                call_events[session_id].append(idx)
            in_call_run[session_id] = True
            continue

        in_call_run[session_id] = False

        if role == "assistant" and rtype == "message":
            # Extract text content
            parts = []
            content = msg.get("content")
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "output_text":
                        parts.append(item.get("text", ""))
            elif isinstance(content, str):
                parts.append(content)
            message_events[session_id].append((idx, "".join(parts)))

    used_message_indices: set[int] = set()
    blank_thinking_by_session: dict[str, deque[str]] = defaultdict(deque)

    def _preview_matches(text: str, preview: str) -> bool:
        preview = preview.strip()
        if not preview:
            return False
        return text.startswith(preview)

    for session_id, thinking, content_preview in thinking_rows:
        thinking = thinking or ""
        preview = content_preview or ""

        if preview.strip():
            matched_idx = None
            for idx, text in message_events.get(session_id, []):
                if idx in used_message_indices:
                    continue
                if _preview_matches(text, preview):
                    matched_idx = idx
                    break

            if matched_idx is not None:
                out[matched_idx]["_agent_thinking"] = thinking
                used_message_indices.add(matched_idx)
            else:
                blank_thinking_by_session[session_id].append(thinking)
        else:
            blank_thinking_by_session[session_id].append(thinking)

    # Assign remaining thinking to tool calls
    for session_id, indices in call_events.items():
        queue = blank_thinking_by_session[session_id]
        for idx in indices:
            if not queue:
                break
            out[idx]["_agent_thinking"] = queue.popleft()

    return out


def _extract_images_from_output(output: Any, sample_idx: int, seg_idx: int, img_counter: list[int]) -> tuple[str, list[str]]:
    """Extract base64 images from tool output and save them."""
    images = []
    output_parts = []

    # Handle list format output (multimodal content)
    if isinstance(output, list):
        for item in output:
            if isinstance(item, dict):
                item_type = item.get('type')

                if item_type == 'input_image':
                    # Extract base64 from data URL
                    image_url = item.get('image_url', '')
                    if image_url.startswith('data:image/'):
                        try:
                            # Extract base64 part
                            base64_data = image_url.split(',', 1)[1] if ',' in image_url else ''
                            if base64_data:
                                img_data = base64.b64decode(base64_data)
                                img = Image.open(BytesIO(img_data))

                                # Generate filename
                                img_hash = hashlib.md5(img_data).hexdigest()[:8]
                                filename = f"critic_probe_{sample_idx:07d}_seg{seg_idx}_img{img_counter[0]:02d}_{img_hash}.jpg"
                                img_path = IMAGES_DIR / filename

                                # Save image
                                img.save(img_path, format='JPEG', quality=85)
                                images.append(str(img_path))
                                img_counter[0] += 1

                                # Add <image> placeholder
                                output_parts.append('<image>')
                        except Exception as e:
                            print(f"Warning: Failed to extract image: {e}")
                            output_parts.append('<image>')

                elif item_type == 'output_text':
                    # Regular text content
                    output_parts.append(item.get('text', ''))
                else:
                    # Unknown type, try to get text
                    output_parts.append(str(item.get('text', '')))
            else:
                output_parts.append(str(item))

        output_str = '\n'.join(output_parts)

    else:
        # Handle string format output
        output_str = str(output)
        pattern = r'<image>\s*([A-Za-z0-9+/=\s]+?)\s*</image>'
        matches = re.findall(pattern, output_str, re.DOTALL)

        for match in matches:
            try:
                img_data = base64.b64decode(match.replace('\n', '').replace(' ', ''))
                img = Image.open(BytesIO(img_data))

                img_hash = hashlib.md5(img_data).hexdigest()[:8]
                filename = f"critic_probe_{sample_idx:07d}_seg{seg_idx}_img{img_counter[0]:02d}_{img_hash}.jpg"
                img_path = IMAGES_DIR / filename

                img.save(img_path, format='JPEG', quality=85)
                images.append(str(img_path))
                img_counter[0] += 1
            except Exception as e:
                print(f"Warning: Failed to extract image: {e}")
                continue

        output_str = re.sub(pattern, '<image>', output_str, flags=re.DOTALL)

    return output_str, images


def _convert_to_v6_messages(
    raw_messages: list[dict[str, Any]],
    system_prompt: str,
    sample_idx: int,
    seg_idx: int
) -> tuple[list[dict[str, Any]], list[str]] | None:
    """Convert raw DB messages to v6.2 format with tool_call/tool_response roles."""

    messages = [{"role": "system", "content": system_prompt}]
    images: list[str] = []
    img_counter = [0]

    pending_tool_calls = []
    pending_thinking = ""
    user_seen = False

    for msg in raw_messages:
        msg_type = msg.get("type")
        role = msg.get("role")
        thinking = msg.get("_agent_thinking", "")

        if role == "user":
            messages.append({"role": "user", "content": msg.get("content", "")})
            user_seen = True
            continue

        if not user_seen:
            continue

        if msg_type == "function_call":
            # Collect tool calls
            if thinking and not pending_thinking:
                pending_thinking = thinking

            tool_call = {
                "name": msg.get("name"),
                "arguments": msg.get("arguments", "{}")
            }
            pending_tool_calls.append(tool_call)

        elif msg_type == "function_call_output":
            # Flush pending tool calls as assistant message
            if pending_tool_calls:
                assistant_content = _thinking_wrap(pending_thinking)
                messages.append({"role": "assistant", "content": assistant_content})

                # Add tool_call messages
                for tc in pending_tool_calls:
                    messages.append({
                        "role": "tool_call",
                        "content": json.dumps(tc, ensure_ascii=False)
                    })

                pending_tool_calls = []
                pending_thinking = ""

            # Process tool output
            output = msg.get("output", "")
            output_clean, extracted_images = _extract_images_from_output(
                output, sample_idx, seg_idx, img_counter
            )
            images.extend(extracted_images)

            messages.append({
                "role": "tool_response",
                "content": output_clean
            })

        elif role == "assistant" and msg_type == "message":
            # Final assistant message (critique)
            content_parts = []
            content = msg.get("content")

            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "output_text":
                        content_parts.append(item.get("text", ""))
            elif isinstance(content, str):
                content_parts.append(content)

            text = "".join(content_parts)

            # Add thinking if present
            full_content = _thinking_wrap(thinking) + text
            messages.append({"role": "assistant", "content": full_content})

    # Flush any remaining tool calls
    if pending_tool_calls:
        assistant_content = _thinking_wrap(pending_thinking)
        messages.append({"role": "assistant", "content": assistant_content})

        for tc in pending_tool_calls:
            messages.append({
                "role": "tool_call",
                "content": json.dumps(tc, ensure_ascii=False)
            })

    return messages, images


def _get_critic_tools(agent_label: str) -> str:
    """Get tool schemas for a critic agent."""
    if agent_label not in CRITIC_TO_DESIGNER:
        return "[]"

    designer_label = CRITIC_TO_DESIGNER[agent_label]
    whitelist = CRITIC_TOOL_WHITELIST.get(agent_label, [])

    # Get all designer tools (returns JSON string)
    all_tools_json = ts_mod.get_tools(designer_label)

    # Parse if it's a string
    if isinstance(all_tools_json, str):
        all_tools = json.loads(all_tools_json)
    else:
        all_tools = all_tools_json

    # Filter to whitelist
    tools = []
    for tool in all_tools:
        if isinstance(tool, dict) and "function" in tool:
            tool_name = tool["function"].get("name", "")
            if tool_name in whitelist:
                tools.append(tool)

    return json.dumps(tools, ensure_ascii=False)


def process_scene(batch_name: str, scene_name: str, sample_idx: int) -> list[dict[str, Any]]:
    """Process a single scene and return training samples."""
    scene_path = CRITIC_PROBE_BASE / batch_name / "hydra" / scene_name

    if not scene_path.exists():
        return []

    samples = []

    # Process critic agent
    agent_db_name = "critic"
    agent_label = "floor_plan_critic"
    db_path = scene_path / f"{agent_db_name}.db"

    if not db_path.exists():
        print(f"Warning: {db_path} not found")
        return []

    print(f"Processing {batch_name}/{scene_name}/{agent_db_name}...")

    try:
        # Get system prompt and tools
        system_prompt = sp_mod.get_system_prompt(agent_label)
        tools_json = _get_critic_tools(agent_label)

        # Read messages with thinking
        raw_messages = _read_db_messages_with_thinking(db_path)

        if not raw_messages:
            print(f"  No messages found")
            return []

        # Convert to v6 format
        result = _convert_to_v6_messages(
            raw_messages,
            system_prompt,
            sample_idx,
            seg_idx=0
        )

        if result is None:
            print(f"  Conversion failed")
            return []

        messages, images = result

        # Build sample
        sample = {
            "messages": messages,
            "tools": tools_json,
            "images": images,
            "_metadata": {
                "source_db": str(db_path),
                "agent_label": agent_label,
                "run_label": "cola_gpt55_scenebenchmark_20260826",
                "scene": scene_name,
                "batch": batch_name,
                "db_segment_index": 0,
                "target_kind": "full_conversation",
                "num_messages": len(messages),
                "num_images": len(images),
            }
        }

        samples.append(sample)
        print(f"  ✓ Extracted: {len(messages)} messages, {len(images)} images")

    except Exception as e:
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()

    return samples


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    # Batch to scene mapping
    batch_scene_map = {
        'batch_001': 'scene_000',
        'batch_002': 'scene_001',
        'batch_003': 'scene_002',
        'batch_004': 'scene_003',
    }

    all_samples = []
    sample_idx = 0

    for batch_name, scene_name in batch_scene_map.items():
        samples = process_scene(batch_name, scene_name, sample_idx)
        all_samples.extend(samples)
        sample_idx += len(samples)

    # Write output
    output_file = OUTPUT_DIR / "train_critic_probe.jsonl"
    with open(output_file, 'w', encoding='utf-8') as f:
        for sample in all_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')

    print(f"\n{'=' * 70}")
    print(f"✓ Extraction complete!")
    print(f"  Output: {output_file}")
    print(f"  Samples: {len(all_samples)}")
    print(f"  Images: {IMAGES_DIR}")
    print(f"{'=' * 70}")

    # Print statistics
    if all_samples:
        total_messages = sum(len(s["messages"]) for s in all_samples)
        total_images = sum(len(s["images"]) for s in all_samples)

        print(f"\nStatistics:")
        print(f"  Total messages: {total_messages}")
        print(f"  Total images: {total_images}")
        print(f"  Avg messages/sample: {total_messages / len(all_samples):.1f}")
        print(f"  Avg images/sample: {total_images / len(all_samples):.1f}")
    else:
        print(f"\n⚠ No samples extracted - check errors above")


if __name__ == "__main__":
    main()
