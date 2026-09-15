#!/usr/bin/env python3
"""Test script to extract data from critic probe runs."""

import sqlite3
import json
from pathlib import Path

def extract_agent_data(db_path, agent_name):
    """Extract messages from an agent database."""
    if not db_path.exists():
        return None

    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    try:
        cursor.execute("SELECT message_data FROM agent_messages ORDER BY id")
        messages = []
        for row in cursor.fetchall():
            msg = json.loads(row[0])
            messages.append(msg)

        return {
            'agent': agent_name,
            'message_count': len(messages),
            'messages': messages
        }
    except Exception as e:
        return {'agent': agent_name, 'error': str(e)}
    finally:
        conn.close()

def process_scene(scene_path):
    """Process a single scene directory."""
    scene_path = Path(scene_path)

    result = {
        'scene': scene_path.name,
        'agents': {}
    }

    for agent_name in ['planner', 'designer', 'critic']:
        db_path = scene_path / f"{agent_name}.db"
        data = extract_agent_data(db_path, agent_name)
        if data:
            result['agents'][agent_name] = data

    return result

def main():
    # Test on scene_000 from batch_001
    base_path = Path("/mnt/afs/visitor33/Task3.2/outputs/critic_probe/cola_gpt55_scenebenchmark_no_code_repair_room_dense_acp_20260826_080514/critic_on/batch_001/hydra")

    # Process first 3 scenes
    results = []
    for i in range(3):
        scene_path = base_path / f"scene_{i:03d}"
        if scene_path.exists():
            print(f"\nProcessing {scene_path.name}...")
            result = process_scene(scene_path)
            results.append(result)

            # Print summary
            for agent_name, agent_data in result['agents'].items():
                if 'error' in agent_data:
                    print(f"  {agent_name}: ERROR - {agent_data['error']}")
                else:
                    msg_count = agent_data.get('message_count', 0)
                    print(f"  {agent_name}: {msg_count} messages")

                    # Check for tool calls
                    if 'messages' in agent_data:
                        tool_call_count = sum(1 for m in agent_data['messages'] if m.get('tool_calls'))
                        if tool_call_count > 0:
                            print(f"    - {tool_call_count} messages with tool_calls")
        else:
            print(f"Scene {i:03d} not found")
            break

    print(f"\n✓ Successfully extracted data from {len(results)} scenes")
    return results

if __name__ == "__main__":
    results = main()
