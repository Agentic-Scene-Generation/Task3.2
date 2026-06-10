#!/bin/bash
# =============================================================================
# 仅启动 vLLM 服务（模型已下载的情况下快速启动）
# 用法：bash scripts/start_vllm.sh
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
MODEL_DIR="$PROJECT_DIR/models/Qwen3.5-35B-A3B"
MODEL_ID="Qwen/Qwen3.5-35B-A3B"
VLLM_PORT=8000

if [ ! -d "$MODEL_DIR" ]; then
    echo "错误：模型目录不存在: $MODEL_DIR"
    echo "请先运行 bash scripts/deploy_qwen.sh 下载模型"
    exit 1
fi

echo "启动 vLLM 服务..."
echo "  模型: $MODEL_DIR"
echo "  端口: $VLLM_PORT"

# 设置环境变量
export OPENAI_API_KEY="not-needed"
export OPENAI_BASE_URL="http://localhost:${VLLM_PORT}/v1"
export VLLM_LOGGING_LEVEL=INFO

VLLM_LOG="$PROJECT_DIR/vllm_server.log"

# 后台启动，日志写入文件
vllm serve "$MODEL_DIR" \
    --served-model-name "$MODEL_ID" \
    --tensor-parallel-size 2 \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_xml \
    --reasoning-parser qwen3 \
    --port "$VLLM_PORT" \
    --max-model-len 262144  \
    --gpu-memory-utilization 0.90 \
    > "$VLLM_LOG" 2>&1 &

VLLM_PID=$!
echo "vLLM 进程 PID: $VLLM_PID，日志: $VLLM_LOG"

# 等待服务就绪（最多 30 分钟，首次编译较慢）
echo "等待 vLLM 服务就绪..."
for i in $(seq 1 360); do
    if curl -sf "http://localhost:${VLLM_PORT}/health" > /dev/null 2>&1; then
        echo "vLLM 服务已就绪 (等待了 $((i * 5)) 秒)"
        exit 0
    fi
    # 检查进程是否意外退出
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
        echo "错误：vLLM 进程意外退出，查看日志: $VLLM_LOG"
        exit 1
    fi
    sleep 5
done

echo "错误：vLLM 服务在 30 分钟内未就绪，查看日志: $VLLM_LOG"
exit 1
