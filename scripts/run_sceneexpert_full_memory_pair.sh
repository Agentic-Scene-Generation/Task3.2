#!/usr/bin/env bash
# Run one controlled Full-mode Fast Memory OFF/ON pair against the same reused
# floor-plan checkpoints and the same frozen long-term memory bank.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"

# TODO(user): Point to the canonical Full reuse launcher on the ACP host.
FULL_REUSE_LAUNCHER="${FULL_REUSE_LAUNCHER:-$PROJECT_ROOT/tmp/acp/acp_qwen38_full_reuse.sh}"

# TODO(user): Select exactly one completed shared-base source. The underlying
# reuse launcher resolves SOURCE_RUN_ID below outputs/critic_probe, or accepts
# an absolute REUSED_SHARED_BASE_ROOT.
SOURCE_RUN_ID="${SOURCE_RUN_ID:-}"
REUSED_SHARED_BASE_ROOT="${REUSED_SHARED_BASE_ROOT:-}"

# TODO(user): Use a populated memory bank prepared before this comparison. It
# is opened read-only by SceneExpert in both arms; no record may be promoted or
# receive utility updates during the pair.
FROZEN_MEMORY_DIR="${FROZEN_MEMORY_DIR:-}"

# TODO(user): Give every logical pair a unique, filesystem-safe ID.
PAIR_ID="${PAIR_ID:-full_memory_pair_$(date +%Y%m%d_%H%M%S)}"

# TODO(user): Alternate this value across repeated pairs to avoid a fixed
# first/second-arm timing bias. Supported values: off_on, on_off.
ARM_ORDER="${ARM_ORDER:-off_on}"

# ``both`` preserves the original single-job workflow. The ACP wrappers use
# the other actions so OFF and ON can run as separate jobs while sharing the
# same pair contract and frozen bank. ``metrics`` performs no generation.
PAIR_ACTION="${PAIR_ACTION:-both}"

PYTHON_BIN="${PYTHON_BIN:-$PROJECT_ROOT/.venv/bin/python}"
PAIR_METRICS_DIR="${PAIR_METRICS_DIR:-$PROJECT_ROOT/outputs/critic_probe/${PAIR_ID}_metrics}"

die() {
  echo "ERROR: $*" >&2
  exit 2
}

[[ -d "$PROJECT_ROOT" ]] || die "PROJECT_ROOT does not exist: $PROJECT_ROOT"
[[ -f "$FULL_REUSE_LAUNCHER" ]] || \
  die "Full reuse launcher does not exist: $FULL_REUSE_LAUNCHER"
[[ -n "$SOURCE_RUN_ID" || -n "$REUSED_SHARED_BASE_ROOT" ]] || \
  die "set SOURCE_RUN_ID or REUSED_SHARED_BASE_ROOT"
[[ -n "$FROZEN_MEMORY_DIR" && -d "$FROZEN_MEMORY_DIR" ]] || \
  die "FROZEN_MEMORY_DIR must be an existing memory bank"
[[ "$PAIR_ID" != */* && "$PAIR_ID" != "." && "$PAIR_ID" != ".." ]] || \
  die "PAIR_ID must be one directory name: $PAIR_ID"
[[ "$ARM_ORDER" == "off_on" || "$ARM_ORDER" == "on_off" ]] || \
  die "ARM_ORDER must be off_on or on_off: $ARM_ORDER"
case "$PAIR_ACTION" in
  both|memory_off|memory_on|metrics) ;;
  *) die "PAIR_ACTION must be both, memory_off, memory_on, or metrics: $PAIR_ACTION" ;;
esac
[[ -x "$PYTHON_BIN" ]] || die "PYTHON_BIN is not executable: $PYTHON_BIN"

for required in \
  manifest.json \
  success_cases.jsonl \
  failure_cases.jsonl \
  skills.jsonl \
  events.jsonl; do
  [[ -f "$FROZEN_MEMORY_DIR/$required" ]] || \
    die "frozen memory bank is incomplete: $FROZEN_MEMORY_DIR/$required"
done

BASELINE_RUN_ID="${PAIR_ID}_memory_off"
TREATMENT_RUN_ID="${PAIR_ID}_memory_on"
BASELINE_ROOT="$PROJECT_ROOT/outputs/critic_probe/$BASELINE_RUN_ID"
TREATMENT_ROOT="$PROJECT_ROOT/outputs/critic_probe/$TREATMENT_RUN_ID"

case "$PAIR_ACTION" in
  both)
    [[ ! -e "$BASELINE_ROOT" ]] || \
      die "baseline output already exists: $BASELINE_ROOT"
    [[ ! -e "$TREATMENT_ROOT" ]] || \
      die "treatment output already exists: $TREATMENT_ROOT"
    [[ ! -e "$PAIR_METRICS_DIR" ]] || \
      die "paired metrics output already exists: $PAIR_METRICS_DIR"
    ;;
  memory_off)
    [[ ! -e "$BASELINE_ROOT" ]] || \
      die "baseline output already exists: $BASELINE_ROOT"
    ;;
  memory_on)
    [[ ! -e "$TREATMENT_ROOT" ]] || \
      die "treatment output already exists: $TREATMENT_ROOT"
    ;;
  metrics)
    [[ -d "$BASELINE_ROOT" ]] || \
      die "baseline output does not exist: $BASELINE_ROOT"
    [[ -d "$TREATMENT_ROOT" ]] || \
      die "treatment output does not exist: $TREATMENT_ROOT"
    [[ ! -e "$PAIR_METRICS_DIR" ]] || \
      die "paired metrics output already exists: $PAIR_METRICS_DIR"
    ;;
esac

run_arm() {
  local arm="$1"
  local retrieval_enabled="$2"
  local run_id="$3"

  echo "========== FULL MEMORY PAIR: $arm =========="
  echo "pair_id=$PAIR_ID"
  echo "run_id=$run_id"
  echo "memory_bank=$FROZEN_MEMORY_DIR"
  echo "fast_memory_retrieval=$retrieval_enabled"
  echo "memory_writer=false"

  env \
    PROJECT_ROOT="$PROJECT_ROOT" \
    SOURCE_RUN_ID="$SOURCE_RUN_ID" \
    REUSED_SHARED_BASE_ROOT="$REUSED_SHARED_BASE_ROOT" \
    RUN_ID="$run_id" \
    SCENEEXPERT_MEMORY_DIR="$FROZEN_MEMORY_DIR" \
    SCENEEXPERT_EXPERIMENT=ablation_5_qwen3_full \
    SCENEEXPERT_COMPONENT_FAST_MEMORY_RETRIEVAL_ENABLED="$retrieval_enabled" \
    SCENEEXPERT_COMPONENT_MEMORY_WRITER_ENABLED=false \
    SCENEEXPERT_COMPONENT_SLOW_MEMORY_CAPTURE_ENABLED=true \
    SCENEEXPERT_EVAL_PAIR_ID="$PAIR_ID" \
    SCENEEXPERT_EVAL_DIMENSION=fast_memory_retrieval \
    SCENEEXPERT_EVAL_ARM="$arm" \
    SCENEEXPERT_EVAL_REQUIRE_FROZEN_MEMORY=true \
    bash "$FULL_REUSE_LAUNCHER"
}

generate_pair_metrics() {
  "$PYTHON_BIN" -m scenesmith.scene_expert.paired_metrics \
    --baseline "$BASELINE_ROOT" \
    --treatment "$TREATMENT_ROOT" \
    --output-dir "$PAIR_METRICS_DIR"

  echo "========== FULL MEMORY PAIR COMPLETE =========="
  echo "baseline_metrics=$BASELINE_ROOT/metrics/run_metrics.json"
  echo "treatment_metrics=$TREATMENT_ROOT/metrics/run_metrics.json"
  echo "paired_metrics=$PAIR_METRICS_DIR/paired_metrics.json"
  echo "paired_report=$PAIR_METRICS_DIR/paired_metrics.md"
}

case "$PAIR_ACTION" in
  both)
    if [[ "$ARM_ORDER" == "off_on" ]]; then
      run_arm memory_off false "$BASELINE_RUN_ID"
      run_arm memory_on true "$TREATMENT_RUN_ID"
    else
      run_arm memory_on true "$TREATMENT_RUN_ID"
      run_arm memory_off false "$BASELINE_RUN_ID"
    fi
    generate_pair_metrics
    ;;
  memory_off)
    run_arm memory_off false "$BASELINE_RUN_ID"
    echo "OFF arm complete; run the ON ACP with the same pair configuration."
    echo "baseline_metrics=$BASELINE_ROOT/metrics/run_metrics.json"
    ;;
  memory_on)
    run_arm memory_on true "$TREATMENT_RUN_ID"
    echo "ON arm complete; run PAIR_ACTION=metrics after the OFF arm is complete."
    echo "treatment_metrics=$TREATMENT_ROOT/metrics/run_metrics.json"
    ;;
  metrics)
    generate_pair_metrics
    ;;
esac
