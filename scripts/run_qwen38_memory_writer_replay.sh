#!/usr/bin/env bash
# Self-contained CCI/ACP Writer replay: start Qwen -> readiness -> replay -> cleanup.
# Does not generate scenes, start embedding, train models, or inspect Git.
set -euo pipefail

# TODO(user): Only change the source scene when replaying another batch.
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
SOURCE_RUN="${SOURCE_RUN:-reuse_full_shared_full_sceneeval100_hard_qwen38_p7_20260826_124733_sceneeval100_hard_qwen38_p7_20260909_145714}"
SCENE_EXPERT_DIR="${SCENE_EXPERT_DIR:-$PROJECT_ROOT/outputs/critic_probe/$SOURCE_RUN/critic_on/batch_091/hydra/scene_090/scene_expert}"
# TODO(user): false owns a fresh service in THIS CCI instance; true requires a
# deliberately managed existing service and will never stop that service.
REUSE_EXISTING_MODEL_SERVICES="${REUSE_EXISTING_MODEL_SERVICES:-false}"

TASK3_SHARED_ROOT="${TASK3_SHARED_ROOT:-/mnt/afs/task3_2}"
MODEL_NAME="${MODEL_NAME:-unsloth/Qwen3.8-27B-GGUF}"
MODEL_DIR="${MODEL_DIR:-$TASK3_SHARED_ROOT/share_model/unsloth/Qwen3.8-27B-GGUF}"
MODEL="${MODEL:-$MODEL_DIR/Qwen3.8-27B-UD-Q8_K_XL.gguf}"
MMPROJ="${MMPROJ:-$MODEL_DIR/mmproj-F16.gguf}"
LLM_LAUNCHER="${LLM_LAUNCHER:-$TASK3_SHARED_ROOT/share_scripts/llama.cpp/run_qwen38_27b_llama_cpp.sh}"
CUDA13_LIB_DIR="${CUDA13_LIB_DIR:-$TASK3_SHARED_ROOT/L202500276_lwz/projects/Task3.2/.venv/lib/python3.11/site-packages/nvidia/cu13/lib}"
PYTHON_BIN="${PYTHON_BIN:-$PROJECT_ROOT/.venv/bin/python}"
LLM_PORT="${LLM_PORT:-8002}"
WAIT_TIMEOUT="${WAIT_TIMEOUT:-7200}"
RUN_ID="${RUN_ID:-writer_replay_091_$(date +%Y%m%d_%H%M%S)_$$}"
OUTPUT_DIR="$PROJECT_ROOT/tmp/$RUN_ID"
LOG_DIR="$PROJECT_ROOT/tmp/acp_logs/$RUN_ID"

[[ "$RUN_ID" =~ ^[A-Za-z0-9_.-]+$ && "$RUN_ID" != . && "$RUN_ID" != .. ]] || { echo 'ERROR: invalid RUN_ID'; exit 2; }
[[ "$LLM_PORT" =~ ^[0-9]+$ && "$WAIT_TIMEOUT" =~ ^[0-9]+$ ]] || { echo 'ERROR: invalid port or timeout'; exit 2; }
[[ -x "$PYTHON_BIN" ]] || { echo "ERROR: Python missing: $PYTHON_BIN"; exit 2; }
[[ -d "$SCENE_EXPERT_DIR" ]] || { echo "ERROR: source missing: $SCENE_EXPERT_DIR"; exit 2; }
[[ "$REUSE_EXISTING_MODEL_SERVICES" == true || "$REUSE_EXISTING_MODEL_SERVICES" == false ]] || { echo 'ERROR: reuse must be true or false'; exit 2; }
[[ ! -e "$OUTPUT_DIR" && ! -e "$LOG_DIR" ]] || { echo 'ERROR: RUN_ID already used; choose a new ID'; exit 2; }
command -v curl >/dev/null
command -v timeout >/dev/null
mkdir -p "$PROJECT_ROOT/tmp/acp_logs"
mkdir "$LOG_DIR"
exec > >(tee "$LOG_DIR/terminal.log") 2>&1
cp "${BASH_SOURCE[0]}" "$LOG_DIR/launcher.sh"
cd "$PROJECT_ROOT"

LLM_PID=""
REPLAY_PID=""
cleanup() {
  local code=$?
  trap - EXIT INT TERM
  [[ -z "$REPLAY_PID" ]] || kill "$REPLAY_PID" 2>/dev/null || true
  # Only the process group started by this invocation is owned here.
  if [[ -n "$LLM_PID" ]]; then
    kill -TERM -- "-$LLM_PID" 2>/dev/null || true
    for _ in {1..20}; do
      kill -0 -- "-$LLM_PID" 2>/dev/null || break
      sleep 0.2
    done
    kill -KILL -- "-$LLM_PID" 2>/dev/null || true
    wait "$LLM_PID" 2>/dev/null || true
  fi
  [[ -z "$REPLAY_PID" ]] || wait "$REPLAY_PID" 2>/dev/null || true
  printf 'exit_code=%s\nfinished_at=%s\n' "$code" "$(date --iso-8601=seconds)" > "$LOG_DIR/exit_status.env"
  echo "Replay exit_code=$code; results=$OUTPUT_DIR; logs=$LOG_DIR"
  exit "$code"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
printf 'started_at=%s\nhostname=%s\napi_base_url=http://127.0.0.1:%s/v1\nmodel=%s\n' \
  "$(date --iso-8601=seconds)" "$(hostname)" "$LLM_PORT" "$MODEL_NAME" > "$LOG_DIR/service.env"
echo "phase=starting" > "$LOG_DIR/phase.env"

if [[ "$REUSE_EXISTING_MODEL_SERVICES" == false ]]; then
  command -v setsid >/dev/null
  for required in "$MODEL" "$MMPROJ" "$LLM_LAUNCHER" "$CUDA13_LIB_DIR/libcudart.so.13"; do
    [[ -f "$required" ]] || { echo "ERROR: required file missing: $required"; exit 2; }
  done
  if timeout 2 bash -c "exec 3<>/dev/tcp/127.0.0.1/$LLM_PORT" >/dev/null 2>&1; then
    echo "ERROR: Port $LLM_PORT occupied; will not replace its service. Use REUSE_EXISTING_MODEL_SERVICES=true intentionally."
    exit 2
  fi
  export LD_LIBRARY_PATH="$CUDA13_LIB_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  # Same validated Qwen3.8 p7 profile as the current Full ACP runtime.
  setsid env \
    HOST=0.0.0.0 PORT="$LLM_PORT" CTX_SIZE=458752 PARALLEL=7 N_GPU_LAYERS=999 \
    CACHE_TYPE_K=q8_0 CACHE_TYPE_V=q8_0 VISION=true THREADS=16 THREADS_HTTP=7 \
    BATCH_SIZE=1024 UBATCH_SIZE=256 THINKING=true REASONING=auto \
    REASONING_PRESERVE=true TEMP=1.0 TOP_P=0.95 TOP_K=20 MIN_P=0.00 \
    PRESENCE_PENALTY=0.0 REPEAT_PENALTY=1.0 \
    MODEL_DIR="$MODEL_DIR" MODEL="$MODEL" MMPROJ="$MMPROJ" ALIAS="$MODEL_NAME" \
    bash "$LLM_LAUNCHER" \
    --flash-attn on --spec-type draft-mtp --spec-draft-n-max 3 \
    --no-kv-unified --cache-prompt --cache-ram 49152 --cache-idle-slots \
    --ctx-checkpoints 64 --slot-prompt-similarity 0.5 \
    > "$LOG_DIR/llama_qwen38_27b.log" 2>&1 &
  LLM_PID=$!
  echo "owned_llm_pid=$LLM_PID" >> "$LOG_DIR/service.env"
fi

echo "Waiting for Qwen at http://127.0.0.1:$LLM_PORT in this CCI instance."
deadline=$((SECONDS + WAIT_TIMEOUT))
while ! curl --noproxy '*' -fsS --max-time 3 "http://127.0.0.1:$LLM_PORT/health" >/dev/null 2>&1; do
  if [[ -n "$LLM_PID" ]] && ! kill -0 "$LLM_PID" 2>/dev/null; then
    echo 'ERROR: Qwen exited during startup.'
    tail -n 60 "$LOG_DIR/llama_qwen38_27b.log"
    exit 2
  fi
  if (( SECONDS >= deadline )); then
    echo 'ERROR: model health timeout; no Writer or bank was created. Check this CCI instance and model log.'
    exit 2
  fi
  sleep 2
done

echo "phase=writer_replay" > "$LOG_DIR/phase.env"
OPENAI_API_KEY="${OPENAI_API_KEY:-sk-123}" \
  "$PYTHON_BIN" scripts/replay_sceneexpert_memory_writer.py \
  --scene-expert-dir "$SCENE_EXPERT_DIR" --output-dir "$OUTPUT_DIR" \
  --api-base-url "http://127.0.0.1:$LLM_PORT/v1" --model "$MODEL_NAME" &
REPLAY_PID=$!
if wait "$REPLAY_PID"; then code=0; else code=$?; fi
REPLAY_PID=""
echo "phase=finished" > "$LOG_DIR/phase.env"
exit "$code"
