#!/usr/bin/env bash
# Start or reuse the local Mihomo proxy used by OpenRouter image requests.
# Source this script so its proxy environment exports remain in the caller:
#   source /mnt/afs/visitor33/Task3.2/start_openrouter_proxy.sh

PROXY_URL="${PROXY_URL:-http://127.0.0.1:7890}"
PROXY_DIR="${PROXY_DIR:-/mnt/afs/visitor33/proxy}"
PROXY_BIN="${PROXY_BIN:-/mnt/afs/visitor33/bin/mihomo}"
PROXY_CONFIG="${PROXY_CONFIG:-$PROXY_DIR/config.yaml}"
PROXY_LOG="${PROXY_LOG:-/tmp/mihomo-openrouter-${HOSTNAME:-acp}.log}"
PROXY_PID_FILE="${PROXY_PID_FILE:-/tmp/mihomo-openrouter-${HOSTNAME:-acp}.pid}"
PROXY_READY_URL="${PROXY_READY_URL:-https://openrouter.ai/api/v1/models}"
PROXY_START_ATTEMPTS="${PROXY_START_ATTEMPTS:-30}"
PROXY_CURL_TIMEOUT="${PROXY_CURL_TIMEOUT:-8}"

openrouter_proxy_is_ready() {
    curl -fsS \
        --max-time "$PROXY_CURL_TIMEOUT" \
        --proxy "$PROXY_URL" \
        "$PROXY_READY_URL" \
        >/dev/null 2>&1
}

for required_command in curl nohup; do
    if ! command -v "$required_command" >/dev/null 2>&1; then
        echo "[ERROR] required command is unavailable: $required_command" >&2
        return 1 2>/dev/null || exit 1
    fi
done

if openrouter_proxy_is_ready; then
    echo "[PROXY] reusing ready proxy: $PROXY_URL"
else
    if [ ! -x "$PROXY_BIN" ]; then
        echo "[ERROR] Mihomo executable is unavailable: $PROXY_BIN" >&2
        return 1 2>/dev/null || exit 1
    fi
    if [ ! -r "$PROXY_CONFIG" ]; then
        echo "[ERROR] Mihomo configuration is unreadable: $PROXY_CONFIG" >&2
        return 1 2>/dev/null || exit 1
    fi

    echo "[PROXY] starting Mihomo on $PROXY_URL"
    nohup "$PROXY_BIN" \
        -d "$PROXY_DIR" \
        -f "$PROXY_CONFIG" \
        >"$PROXY_LOG" 2>&1 &
    OPENROUTER_PROXY_PID=$!
    printf '%s\n' "$OPENROUTER_PROXY_PID" >"$PROXY_PID_FILE"

    proxy_ready=false
    proxy_attempt=1
    while [ "$proxy_attempt" -le "$PROXY_START_ATTEMPTS" ]; do
        if openrouter_proxy_is_ready; then
            proxy_ready=true
            break
        fi
        if ! kill -0 "$OPENROUTER_PROXY_PID" 2>/dev/null; then
            break
        fi
        sleep 1
        proxy_attempt=$((proxy_attempt + 1))
    done

    if [ "$proxy_ready" != true ]; then
        echo "[ERROR] OpenRouter proxy failed to start: $PROXY_URL" >&2
        echo "[ERROR] Mihomo log: $PROXY_LOG" >&2
        if [ -r "$PROXY_LOG" ]; then
            tail -n 50 "$PROXY_LOG" >&2
        fi
        return 1 2>/dev/null || exit 1
    fi
    echo "[PROXY] Mihomo is ready: pid=$OPENROUTER_PROXY_PID"
fi

export HTTP_PROXY="$PROXY_URL"
export HTTPS_PROXY="$PROXY_URL"
export http_proxy="$PROXY_URL"
export https_proxy="$PROXY_URL"

# Keep the local LLM, embedding service, GroundingDINO, and other loopback
# services outside the proxy.
export NO_PROXY="127.0.0.1,localhost,::1"
export no_proxy="$NO_PROXY"

echo "[PROXY] environment configured for OpenRouter requests"
echo "[PROXY] local services bypass proxy via NO_PROXY=$NO_PROXY"

if [ "${BASH_SOURCE[0]}" = "$0" ]; then
    echo "[NOTICE] The proxy is running, but exports cannot modify the parent shell."
    echo "[NOTICE] Before launching an experiment, run:"
    echo "         source ${BASH_SOURCE[0]}"
fi
