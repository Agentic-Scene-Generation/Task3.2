#!/bin/bash
# =============================================================================
# 集群单任务提交脚本：在同一个 job 内启动 vLLM + 运行实验
# 用法：
#   本地：bash scripts/run_experiment.sh ablation_2_qwen3_naive
#   SLURM：sbatch scripts/run_experiment.sh ablation_2_qwen3_naive
# =============================================================================
#SBATCH --job-name=scenesmith
#SBATCH --gpus=2
#SBATCH --output=outputs/slurm_%j.log

EXPERIMENT=${1:-"ablation_2_qwen3_naive"}
# Detect number of visible GPUs for tensor parallelism
NUM_GPUS=$(python3 -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo 1)
echo "  可见 GPU 数量: $NUM_GPUS (tensor_parallel_size=$NUM_GPUS)"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
MODEL_DIR="$PROJECT_DIR/models/Qwen3.5-35B-A3B"
MODEL_ID="Qwen/Qwen3.5-35B-A3B"
VLLM_PORT=8000
VLLM_LOG="$PROJECT_DIR/vllm_server.log"

cd "$PROJECT_DIR"

# ── 1. 激活环境 ──────────────────────────────────────────────
source .venv/bin/activate
[ -f .env ] && source .env

export OPENAI_API_KEY="not-needed"
export OPENAI_BASE_URL="http://localhost:${VLLM_PORT}/v1"

# ── 2. 后台启动 vLLM ─────────────────────────────────────────
echo "[$(date)] 启动 vLLM..."
vllm serve "$MODEL_DIR" \
    --served-model-name "$MODEL_ID" \
    --tensor-parallel-size "$NUM_GPUS" \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_xml \
    --reasoning-parser qwen3 \
    --port "$VLLM_PORT" \
    --max-model-len 262144  \
    --gpu-memory-utilization 0.90 \
    > "$VLLM_LOG" 2>&1 &

VLLM_PID=$!
echo "[$(date)] vLLM PID=$VLLM_PID，等待就绪..."

# ── 3. 等待 /health ──────────────────────────────────────────
for i in $(seq 1 360); do
    if curl -sf "http://localhost:${VLLM_PORT}/health" > /dev/null 2>&1; then
        echo "[$(date)] vLLM 就绪 ($((i * 5))s)"
        break
    fi
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
        echo "[$(date)] 错误：vLLM 进程退出，见 $VLLM_LOG"
        exit 1
    fi
    sleep 5
done

# ── 4. 运行实验 ──────────────────────────────────────────────
echo "[$(date)] 运行实验: $EXPERIMENT"
python main.py experiment="$EXPERIMENT" +name="$EXPERIMENT"
# python main.py +name=branch_2 \
#   experiment.pipeline.start_stage=ceiling_mounted \
#   experiment.pipeline.resume_from_path=outputs/2026-05-29/10-07-02
EXIT_CODE=$?

# ── 5. 关闭 vLLM ─────────────────────────────────────────────
echo "[$(date)] 关闭 vLLM (PID=$VLLM_PID)"
kill "$VLLM_PID" 2>/dev/null
wait "$VLLM_PID" 2>/dev/null

exit $EXIT_CODE
