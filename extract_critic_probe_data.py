#!/usr/bin/env python3
"""Extract critic probe data and convert to toolcalling jsonl format."""

import sqlite3
import json
from pathlib import Path
from typing import List, Dict, Any

def extract_conversation(db_path: Path) -> List[Dict[str, Any]]:
    """Extract full conversation from an agent database."""
    if not db_path.exists():
        return []

    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    try:
        cursor.execute("SELECT message_data FROM agent_messages ORDER BY id")
        messages = []
        for row in cursor.fetchall():
            msg = json.loads(row[0])
            messages.append(msg)
        return messages
    finally:
        conn.close()

def convert_to_toolcalling_format(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert from function_call format to OpenAI toolcalling format."""
    converted = []
    pending_tool_calls = []

    for msg in messages:
        msg_type = msg.get('type')

        if msg.get('role') == 'user':
            converted.append({
                'role': 'user',
                'content': msg['content']
            })

        elif msg.get('role') == 'assistant':
            converted.append({
                'role': 'assistant',
                'content': msg.get('content', '')
            })

        elif msg_type == 'function_call':
            # Collect tool calls
            tool_call = {
                'id': msg.get('call_id'),
                'type': 'function',
                'function': {
                    'name': msg.get('name'),
                    'arguments': msg.get('arguments', '{}')
                }
            }
            pending_tool_calls.append(tool_call)

        elif msg_type == 'function_call_output':
            # When we hit an output, flush pending tool_calls as assistant message
            if pending_tool_calls:
                converted.append({
                    'role': 'assistant',
                    'content': None,
                    'tool_calls': pending_tool_calls
                })
                pending_tool_calls = []

            # Add tool result
            converted.append({
                'role': 'tool',
                'tool_call_id': msg.get('call_id'),
                'content': msg.get('output', '')
            })

    # Flush any remaining tool calls
    if pending_tool_calls:
        converted.append({
            'role': 'assistant',
            'content': None,
            'tool_calls': pending_tool_calls
        })

    return converted

def process_agent(scene_path: Path, agent_name: str) -> Dict[str, Any]:
    """Process a single agent's conversation."""
    db_path = scene_path / f"{agent_name}.db"

    if not db_path.exists():
        return None

    raw_messages = extract_conversation(db_path)
    converted_messages = convert_to_toolcalling_format(raw_messages)

    return {
        'agent': agent_name,
        'raw_count': len(raw_messages),
        'converted_count': len(converted_messages),
        'messages': converted_messages
    }

def main():
    base_path = Path("/mnt/afs/visitor33/Task3.2/outputs/critic_probe/cola_gpt55_scenebenchmark_no_code_repair_room_dense_acp_20260826_080514/critic_on")

    # Mapping: batch -> scene
    batch_scene_map = {
        'batch_001': 'scene_000',
        'batch_002': 'scene_001',
        'batch_003': 'scene_002',
        'batch_004': 'scene_003'
    }

    all_data = []

    for batch_name, scene_name in batch_scene_map.items():
        batch_path = base_path / batch_name
        scene_path = batch_path / "hydra" / scene_name

        if not scene_path.exists():
            print(f"Warning: {scene_path} not found")
            continue

        print(f"\nProcessing {batch_name}/{scene_name}...")

        scene_data = {
            'batch': batch_name,
            'scene': scene_name,
            'agents': {}
        }

        for agent_name in ['planner', 'designer', 'critic']:
            agent_data = process_agent(scene_path, agent_name)
            if agent_data:
                scene_data['agents'][agent_name] = agent_data
                print(f"  {agent_name}: {agent_data['raw_count']} raw -> {agent_data['converted_count']} converted")

        all_data.append(scene_data)

    print(f"\n✓ Processed {len(all_data)} scenes")

    # Summary statistics
    print(f"\nSummary:")
    for agent in ['planner', 'designer', 'critic']:
        total_raw = sum(
            scene['agents'].get(agent, {}).get('raw_count', 0)
            for scene in all_data
        )
        total_converted = sum(
            scene['agents'].get(agent, {}).get('converted_count', 0)
            for scene in all_data
        )
        print(f"  {agent}: {total_raw} raw messages -> {total_converted} converted messages")

    return all_data

if __name__ == "__main__":
    data = main()
