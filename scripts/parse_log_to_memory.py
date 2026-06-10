"""Parse experiment.log to extract top-N successful critique iterations per stage
and write them directly as success_cases.jsonl (no Qwen3 call needed).

For the top-1 success case per stage, the script also enriches the record with
the actual object placement state from the corresponding scene_state.json
checkpoint, so future agents know exact positions, rotations, and surface IDs
that produced the highest scores.

Usage:
    python scripts/parse_log_to_memory.py \
        --log outputs/2026-06-03/12-28-58/experiment.log \
        --memory-dir outputs/scene_expert_memory/ablation_4 \
        --scene-states-dir outputs/2026-06-03/12-28-58/scene_000/room_bedroom/scene_states \
        [--top-n 5] [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
import uuid
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ---------------------------------------------------------------------------
# Placement extraction from scene_state.json checkpoints
# ---------------------------------------------------------------------------

# Maps stage name -> scene_states subdirectory that holds the final checkpoint
_STAGE_CHECKPOINT_DIR = {
    'furniture':       'scene_after_furniture',
    'wall_mounted':    'scene_after_wall_objects',
    'ceiling_mounted': 'scene_after_ceiling_objects',
    'manipuland':      'final_scene',
}

# Object types to include per stage
_STAGE_OBJECT_TYPES = {
    'furniture':       {'furniture'},
    'wall_mounted':    {'wall_mounted'},
    'ceiling_mounted': {'ceiling_mounted'},
    'manipuland':      {'manipuland'},
}


def _wxyz_to_yaw_deg(w: float, x: float, y: float, z: float) -> float:
    """Extract yaw (Z-rotation) from quaternion, in degrees."""
    siny = 2.0 * (w * z + x * y)
    cosy = 1.0 - 2.0 * (y * y + z * z)
    return round(math.degrees(math.atan2(siny, cosy)), 2)


def _format_placement(obj_id: str, obj: dict) -> str:
    """Format one object's placement as a compact human-readable string."""
    t = obj.get('transform', {})
    tr = t.get('translation', [0.0, 0.0, 0.0])
    rot = t.get('rotation_wxyz', [1.0, 0.0, 0.0, 0.0])
    yaw = _wxyz_to_yaw_deg(*rot)

    bmin = obj.get('bbox_min', [])
    bmax = obj.get('bbox_max', [])
    size_str = ""
    if bmin and bmax and len(bmin) == 3:
        sx, sy, sz = [round(bmax[i] - bmin[i], 3) for i in range(3)]
        size_str = f", size={sx}×{sy}×{sz}m"

    pinfo = obj.get('placement_info') or {}
    surface = pinfo.get('parent_surface_id', '')
    pos2d = pinfo.get('position_2d', [])

    name = obj.get('name', obj_id)
    otype = obj.get('object_type', '')

    if otype == 'furniture':
        return (
            f"{obj_id} ({name}): x={tr[0]:.3f}, y={tr[1]:.3f}, yaw={yaw}°{size_str}"
        )
    elif otype == 'wall_mounted':
        wall = pinfo.get('parent_surface_id', '?')
        px = round(pos2d[0], 3) if pos2d else '?'
        pz = round(pos2d[1], 3) if pos2d else '?'
        return (
            f"{obj_id} ({name}): wall={wall}, pos2d=({px}, {pz}), "
            f"z={tr[2]:.2f}m{size_str}"
        )
    elif otype == 'ceiling_mounted':
        return (
            f"{obj_id} ({name}): x={tr[0]:.3f}, y={tr[1]:.3f}, "
            f"z={tr[2]:.2f}m (ceiling){size_str}"
        )
    elif otype == 'manipuland':
        px = round(pos2d[0], 3) if pos2d else '?'
        py = round(pos2d[1], 3) if pos2d else '?'
        return (
            f"{obj_id} ({name}): surface={surface}, "
            f"pos2d=({px}, {py}), yaw={yaw}°{size_str}"
        )
    return f"{obj_id} ({name}): x={tr[0]:.3f}, y={tr[1]:.3f}, z={tr[2]:.3f}"


def load_stage_placements(scene_states_dir: Path, stage: str) -> list[str]:
    """Load final object placements for a stage from its checkpoint scene_state.json.

    Returns a list of human-readable placement strings, one per object.
    Returns [] if the checkpoint file does not exist.
    """
    checkpoint_subdir = _STAGE_CHECKPOINT_DIR.get(stage)
    if not checkpoint_subdir:
        return []

    state_path = scene_states_dir / checkpoint_subdir / "scene_state.json"
    if not state_path.exists():
        return []

    with state_path.open() as f:
        state = json.load(f)

    target_types = _STAGE_OBJECT_TYPES.get(stage, set())
    lines = []
    for obj_id, obj in state.get('objects', {}).items():
        if obj.get('object_type') in target_types:
            lines.append(_format_placement(obj_id, obj))

    return lines


# ---------------------------------------------------------------------------
# Stage name detection heuristics
# ---------------------------------------------------------------------------

# Maps log markers to canonical stage names
_STAGE_MARKERS = [
    # explicit stage agent markers
    (re.compile(r'request_initial_design.*furniture|furniture.*request_initial_design|'
                r'add_furniture|PLANNER \(FURNITURE\)', re.I), 'furniture'),
    (re.compile(r'PLANNER \(WALL\)|wall.*request_initial_design|'
                r'place_wall_object|request_initial_design.*wall', re.I), 'wall_mounted'),
    (re.compile(r'PLANNER \(CEILING\)|ceiling.*request_initial_design|'
                r'place_ceiling_object', re.I), 'ceiling_mounted'),
    (re.compile(r'PLANNER \(MANIPULAND\)|manipuland.*request_initial_design|'
                r'place_manipuland', re.I), 'manipuland'),
]

_FINAL_SAVE_RE = re.compile(r'Saved final scores to .*/scene_states/(\w+)/scores\.yaml')
_VERDICT_RE = re.compile(r'Stage (\w+) (PASSED|FAILED)')
_INIT_DESIGN_RE = re.compile(r'Tool called: request_initial_design')
_SCORE_LINE_RE = re.compile(r'^([\w\s]+):\s*(\d+)/10$')
_CRITIQUE_HEADER_RE = re.compile(r'CRITIQUE SCORES')
_RESET_RE = re.compile(r'Scene reset to checkpoint\. Reason: (.+)')
_DESIGN_CHANGE_RE = re.compile(r'Tool called: request_design_change')
_TIMESTAMP_RE = re.compile(r'^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+ - [\w\.]+ - \w+ - (.*)')
_STAGE_DIR_MAP = {
    'furniture': 'furniture',
    'wall': 'wall_mounted',
    'ceiling': 'ceiling_mounted',
    'manipuland_furniture': 'manipuland',
}


def _strip_prefix(line: str) -> str:
    m = _TIMESTAMP_RE.match(line)
    return m.group(2).strip() if m else line.strip()


def _get_timestamp(line: str) -> str:
    m = _TIMESTAMP_RE.match(line)
    return m.group(1) if m else ""


# ---------------------------------------------------------------------------
# Parse log into per-stage critique records
# ---------------------------------------------------------------------------

def parse_log(log_path: Path) -> dict[str, list[dict]]:
    """Parse log and group critique iterations by stage.

    Returns dict: stage_name -> list of critique dicts sorted by total score desc.
    Each dict: {total, scores, comments, summary, reset_reasons, timestamp, line}
    """
    with log_path.open() as f:
        lines = f.readlines()

    # First pass: identify stage boundaries using final_save and verdict lines
    stage_boundaries: list[tuple[int, str]] = []  # (line_idx, stage_name)
    for i, raw in enumerate(lines):
        content = _strip_prefix(raw)
        m = _FINAL_SAVE_RE.search(content)
        if m:
            dir_name = m.group(1)
            # Map dir_name to stage
            if dir_name == 'furniture':
                stage_boundaries.append((i, 'furniture'))
            elif dir_name == 'wall':
                stage_boundaries.append((i, 'wall_mounted'))
            elif dir_name == 'ceiling':
                stage_boundaries.append((i, 'ceiling_mounted'))
            elif dir_name.startswith('manipuland'):
                stage_boundaries.append((i, 'manipuland'))

    # Build stage line ranges
    stage_ranges: list[tuple[str, int, int]] = []  # (stage, start, end)
    # Find request_initial_design calls as stage starts
    init_lines = [i for i, raw in enumerate(lines) if _INIT_DESIGN_RE.search(raw)]

    # Map each final_save to the preceding init_design
    for save_idx, stage_name in stage_boundaries:
        # Find the closest init_design before this save
        preceding = [i for i in init_lines if i < save_idx]
        if preceding:
            start = preceding[-1]
        else:
            start = 0
        stage_ranges.append((stage_name, start, save_idx))

    # Second pass: extract all critique blocks within each stage range
    def extract_critiques_in_range(start: int, end: int) -> list[dict]:
        critiques = []
        i = start
        while i <= end:
            content = _strip_prefix(lines[i])
            if _CRITIQUE_HEADER_RE.search(content) and '=====' not in content:
                # Found a CRITIQUE SCORES block
                scores: dict[str, int] = {}
                comments: dict[str, str] = {}
                last_cat = None
                j = i + 1
                while j < min(i + 30, len(lines)):
                    c = _strip_prefix(lines[j])
                    m = _SCORE_LINE_RE.match(c)
                    if m:
                        cat = m.group(1).strip()
                        grade = int(m.group(2))
                        scores[cat] = grade
                        last_cat = cat
                    elif last_cat and c and not c.startswith('=') and 'INFO' not in c and 'WARNING' not in c:
                        # Line immediately after a score is the comment
                        comments[last_cat] = c
                        last_cat = None
                    if '=====' in c and j > i + 2:
                        break
                    j += 1

                if scores:
                    total = sum(scores.values())
                    critiques.append({
                        'line': i + 1,
                        'timestamp': _get_timestamp(lines[i]),
                        'total': total,
                        'max_possible': len(scores) * 10,
                        'scores': scores,
                        'comments': comments,
                    })
            i += 1
        return critiques

    stage_critiques: dict[str, list[dict]] = defaultdict(list)
    for stage_name, start, end in stage_ranges:
        crits = extract_critiques_in_range(start, end)
        stage_critiques[stage_name].extend(crits)

    # Collect reset reasons per stage for failure pattern extraction
    stage_resets: dict[str, list[str]] = defaultdict(list)
    for stage_name, start, end in stage_ranges:
        for i in range(start, end + 1):
            content = _strip_prefix(lines[i])
            m = _RESET_RE.search(content)
            if m:
                stage_resets[stage_name].append(m.group(1))

    # Sort by total score desc
    result = {}
    for stage, crits in stage_critiques.items():
        crits.sort(key=lambda x: x['total'], reverse=True)
        result[stage] = crits
    result['_resets'] = stage_resets
    return result


# ---------------------------------------------------------------------------
# Build SuccessCase records from top-N critiques
# ---------------------------------------------------------------------------

def _build_success_case(stage: str, critique: dict, room_type: str, style: str,
                        task_signature: list[str], trace_ref: str,
                        placement_lines: list[str] | None = None) -> dict:
    """Convert a high-scoring critique iteration into a SuccessCase dict."""
    scores_normalized = {
        k: round(v / 10.0, 2) for k, v in critique['scores'].items()
    }

    # Build successful_pattern from per-category comments (text only, no coords)
    patterns = []
    for cat, comment in critique['comments'].items():
        if comment and len(comment) > 10:
            patterns.append(f"{cat}: {comment}")

    if not patterns:
        top = sorted(critique['scores'].items(), key=lambda x: x[1], reverse=True)
        patterns = [f"{k} scored {v}/10" for k, v in top[:3]]

    return {
        'case_id': f"{stage}_{str(uuid.uuid4())[:8]}",
        'room_type': room_type,
        'style': style,
        'stage': stage,
        'task_signature': task_signature,
        'successful_pattern': patterns,
        'placement_reference': placement_lines or [],  # dedicated field, not mixed into patterns
        'scores': scores_normalized,
        'trace_ref': trace_ref,
    }


def _build_failure_case(stage: str, reset_reason: str, room_type: str) -> dict:
    """Convert a reset reason into a FailureCase dict."""
    # Extract failure type keywords
    failure_type = 'degradation'
    if 'collision' in reset_reason.lower() or 'collides' in reset_reason.lower():
        failure_type = 'collision'
    elif 'missing' in reset_reason.lower():
        failure_type = 'missing_object'
    elif 'imbalance' in reset_reason.lower() or 'overcrowded' in reset_reason.lower():
        failure_type = 'layout_imbalance'
    elif 'unrealistic' in reset_reason.lower() or 'wrong direction' in reset_reason.lower():
        failure_type = 'placement_error'

    # Extract the fix advice (usually after "Need to reset and")
    repair_action = ""
    m = re.search(r'Need to (?:reset and )?(.{20,200})', reset_reason)
    if m:
        repair_action = m.group(1).rstrip('.')

    return {
        'failure_id': f"fail_{stage}_{str(uuid.uuid4())[:8]}",
        'room_type': room_type,
        'stage': stage,
        'object': '',
        'failure_type': failure_type,
        'bad_pattern': reset_reason[:200],
        'failure_reason': reset_reason[:300],
        'repair_action': repair_action[:200],
        'repair_verified': False,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(log_path: Path, memory_dir: str, top_n: int, dry_run: bool,
        room_type: str, style: str,
        scene_states_dir: Path | None = None) -> None:
    print(f"Parsing {log_path} ...")
    data = parse_log(log_path)
    resets = data.pop('_resets', {})

    success_cases = []
    failure_cases = []

    trace_ref = str(log_path)

    # Task signatures per stage (from the bedroom prompt)
    task_signatures = {
        'furniture': ['bed', 'nightstand', 'wardrobe', 'bedroom'],
        'wall_mounted': ['mirror', 'shelf', 'print', 'artwork', 'bedroom'],
        'ceiling_mounted': ['ceiling light', 'pendant', 'bedroom'],
        'manipuland': ['book', 'lamp', 'alarm clock', 'glasses', 'plant', 'bedroom'],
    }

    for stage, critiques in data.items():
        top = critiques[:top_n]
        print(f"\n[{stage}] total critique rounds: {len(critiques)}, "
              f"selecting top {len(top)} (scores: {[c['total'] for c in top]})")

        for rank, c in enumerate(top):
            # Only enrich the best case (rank 0) with actual placement data
            placement_lines: list[str] = []
            if rank == 0 and scene_states_dir:
                placement_lines = load_stage_placements(scene_states_dir, stage)
                if placement_lines:
                    print(f"  Enriched top case with {len(placement_lines)} object placements")

            sc = _build_success_case(
                stage=stage,
                critique=c,
                room_type=room_type,
                style=style,
                task_signature=task_signatures.get(stage, []),
                trace_ref=trace_ref,
                placement_lines=placement_lines if rank == 0 else None,
            )
            success_cases.append(sc)
            if dry_run:
                prefix = "★" if rank == 0 else " "
                print(f"  {prefix}SUCCESS [{stage}] total={c['total']} patterns={sc['successful_pattern'][:1]}")
                if placement_lines and rank == 0:
                    for pl in placement_lines[:3]:
                        print(f"    placement: {pl}")

        # Top-3 failure cases per stage
        stage_resets = resets.get(stage, [])
        for reason in stage_resets[:3]:
            fc = _build_failure_case(stage=stage, reset_reason=reason, room_type=room_type)
            failure_cases.append(fc)
            if dry_run:
                print(f"  FAILURE [{stage}] type={fc['failure_type']} repair={fc['repair_action'][:80]}")

    if dry_run:
        print(f"\nDry-run: would write {len(success_cases)} success cases, "
              f"{len(failure_cases)} failure cases to {memory_dir}")
        return

    out_dir = Path(memory_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    success_path = out_dir / "success_cases.jsonl"
    failure_path = out_dir / "failure_cases.jsonl"

    with success_path.open('w') as f:
        for sc in success_cases:
            f.write(json.dumps(sc, ensure_ascii=False) + '\n')
    print(f"\nWrote {len(success_cases)} success cases → {success_path}")

    with failure_path.open('w') as f:
        for fc in failure_cases:
            f.write(json.dumps(fc, ensure_ascii=False) + '\n')
    print(f"Wrote {len(failure_cases)} failure cases → {failure_path}")

    # Verify loadable
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from scenesmith.scene_expert.memory.store import FastMemoryStore
    store = FastMemoryStore(memory_dir)
    print(f"\nVerification — FastMemoryStore loaded:")
    print(f"  success_cases: {len(store.success_cases)}")
    print(f"  failure_cases: {len(store.failure_cases)}")
    print(f"  skills:        {len(store.skills)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--log', required=True, help='Path to experiment.log')
    parser.add_argument('--memory-dir', default='outputs/scene_expert_memory/ablation_4')
    parser.add_argument('--scene-states-dir', default=None,
                        help='Path to room_bedroom/scene_states/ directory. '
                             'When provided, the top-1 success case per stage is '
                             'enriched with actual object placements from the '
                             'corresponding checkpoint scene_state.json.')
    parser.add_argument('--top-n', type=int, default=5, help='Top-N success cases per stage')
    parser.add_argument('--room-type', default='bedroom')
    parser.add_argument('--style', default='modern')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()

    scene_states_dir = Path(args.scene_states_dir) if args.scene_states_dir else None

    run(
        log_path=Path(args.log),
        memory_dir=args.memory_dir,
        top_n=args.top_n,
        dry_run=args.dry_run,
        room_type=args.room_type,
        style=args.style,
        scene_states_dir=scene_states_dir,
    )


if __name__ == '__main__':
    main()
