#!/usr/bin/env bash
# Run one L-shaped polygon scene through the normal Task3.2 batch launcher.
# Model, endpoint, GPU, and runtime tuning remain controlled by the existing
# launcher arguments and SCENEEXPERT_* environment variables.

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNNER="${RUNNER:-${PROJECT_DIR}/run_prompt_batch.sh}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/outputs/polygon_smoke}"
CSV_PATH="${CSV_PATH:-${OUTPUT_DIR}/polygon_smoke_prompt.csv}"
NAME="${NAME:-polygon_smoke}"

if [ ! -f "$RUNNER" ]; then
    echo "[ERROR] batch runner not found: $RUNNER" >&2
    exit 1
fi

mkdir -p "$OUTPUT_DIR"
python3 - "$CSV_PATH" <<'PY'
import csv
import sys

prompt = """Create a complete modern living room in one irregular L-shaped room.
Use these ordered floor-boundary vertices in meters:
[[0,0],[7,0],[7,3],[4,3],[4,6],[0,6]].
The rectangular notch [4,7] x [3,6] is outside the room and must remain empty.
Add suitable doors and windows on W-numbered walls, then furnish the room and
complete the wall, ceiling, manipuland, physics, reachability, and export stages.
Every complete object footprint must remain inside the exact polygon."""

with open(sys.argv[1], "w", newline="", encoding="utf-8") as stream:
    writer = csv.writer(stream)
    writer.writerow(["scene_index", "prompt"])
    writer.writerow([0, prompt])
PY

exec bash "$RUNNER" \
    --floor-plan-mode polygon \
    --output-dir "$OUTPUT_DIR" \
    0 1 "$CSV_PATH" "$NAME" "$@"
