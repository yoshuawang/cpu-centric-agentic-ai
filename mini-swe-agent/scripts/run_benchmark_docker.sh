#!/usr/bin/env bash
# Run vLLM and mini-swe-agent in separate Docker containers and capture stats.
#
# Usage:
#   ./scripts/run_benchmark_docker.sh [BENCHMARK_TYPE] [BASE_URL] [MODEL_PATH]
#
# Defaults:
#   BENCHMARK_TYPE  sorting
#   BASE_URL        http://vllm-server:${VLLM_PORT} when START_VLLM_CONTAINER=1
#   MODEL_PATH      Qwen/Qwen2.5-Coder-32B-Instruct
#
# Key environment overrides:
#   VLLM_PORT                    5000
#   VLLM_IMAGE                   vllm/vllm-openai:latest
#   VLLM_DTYPE                   bfloat16
#   VLLM_MAX_MODEL_LEN           8192
#   VLLM_GPU_MEMORY_UTILIZATION  0.90
#   VLLM_ENFORCE_EAGER           1
#   VLLM_EXTRA_ARGS              extra args appended to the vLLM command
#   VLLM_HEALTH_BASE_URL         http://127.0.0.1:${VLLM_PORT}
#   DOCKER_NETWORK_NAME          mswa-bench-net
#   START_VLLM_CONTAINER         1; set to 0 to reuse an already running server
#   KEEP_CONTAINERS              0; set to 1 to leave containers running
#   INTERVAL                     0.05 seconds for cgroup samples
#   GPU_INTERVAL                 0.1 seconds; effective nvidia-smi cadence is ~0.16s
#
# Outputs:
#   benchmark_results/           benchmark logs and JSON
#   stats_log.csv                per-container cgroup stats plus host/GPU stats

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

BENCHMARK_TYPE="${1:-sorting}"
MODEL_PATH="${3:-Qwen/Qwen2.5-Coder-32B-Instruct}"

VLLM_PORT="${VLLM_PORT:-5000}"
VLLM_IMAGE="${VLLM_IMAGE:-vllm/vllm-openai:latest}"
VLLM_DTYPE="${VLLM_DTYPE:-bfloat16}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-8192}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.90}"
VLLM_ENFORCE_EAGER="${VLLM_ENFORCE_EAGER:-1}"
VLLM_EXTRA_ARGS="${VLLM_EXTRA_ARGS:-}"
VLLM_HEALTH_BASE_URL="${VLLM_HEALTH_BASE_URL:-http://127.0.0.1:${VLLM_PORT}}"
VLLM_WAIT_SECONDS="${VLLM_WAIT_SECONDS:-900}"
START_VLLM_CONTAINER="${START_VLLM_CONTAINER:-1}"
KEEP_CONTAINERS="${KEEP_CONTAINERS:-0}"

VLLM_CONTAINER_NAME="${VLLM_CONTAINER_NAME:-vllm-server}"
BENCH_CONTAINER_NAME="${BENCH_CONTAINER_NAME:-mswa-bench}"
DOCKER_NETWORK_NAME="${DOCKER_NETWORK_NAME:-mswa-bench-net}"
BENCH_IMAGE="${BENCH_IMAGE:-mini-swe-agent:local}"
OUTPUT_FILE="${OUTPUT_FILE:-stats_log.csv}"
INTERVAL="${INTERVAL:-0.05}"
GPU_INTERVAL="${GPU_INTERVAL:-0.1}"
HF_CACHE_DIR="${HF_CACHE_DIR:-${HF_HOME:-/data1/joshw/hugging_face}}"

if [ "$#" -ge 2 ]; then
  BASE_URL="$2"
elif [ "$START_VLLM_CONTAINER" = "1" ]; then
  BASE_URL="http://${VLLM_CONTAINER_NAME}:${VLLM_PORT}"
else
  BASE_URL="http://127.0.0.1:${VLLM_PORT}"
fi

DOCKER_CMD=(docker)
if ! docker info &>/dev/null 2>&1; then
  DOCKER_CMD=(sudo docker)
fi

if ! "${DOCKER_CMD[@]}" info >/dev/null 2>&1; then
  echo "Error: Docker is not running or you don't have permissions."
  exit 1
fi

MONITOR_PID=""
CREATED_DOCKER_NETWORK=0

stop_monitor() {
  if [ -n "$MONITOR_PID" ]; then
    kill "$MONITOR_PID" 2>/dev/null || true
    wait "$MONITOR_PID" 2>/dev/null || true
    MONITOR_PID=""
  fi
}

cleanup() {
  stop_monitor
  if [ "$KEEP_CONTAINERS" != "1" ]; then
    "${DOCKER_CMD[@]}" rm -f "$BENCH_CONTAINER_NAME" >/dev/null 2>&1 || true
    if [ "$START_VLLM_CONTAINER" = "1" ]; then
      "${DOCKER_CMD[@]}" rm -f "$VLLM_CONTAINER_NAME" >/dev/null 2>&1 || true
    fi
    if [ "$CREATED_DOCKER_NETWORK" = "1" ]; then
      "${DOCKER_CMD[@]}" network rm "$DOCKER_NETWORK_NAME" >/dev/null 2>&1 || true
    fi
  fi
}

trap cleanup EXIT
trap 'exit 130' INT TERM

container_running() {
  [ "$("${DOCKER_CMD[@]}" inspect -f '{{.State.Running}}' "$1" 2>/dev/null || true)" = "true" ]
}

wait_for_vllm() {
  local health_url="${VLLM_HEALTH_BASE_URL%/}/health"
  local i

  echo "[info] Waiting for vLLM health endpoint: $health_url"
  for i in $(seq 1 "$VLLM_WAIT_SECONDS"); do
    if curl -fsS "$health_url" >/dev/null 2>&1; then
      echo "[info] vLLM is ready."
      return 0
    fi

    if [ "$START_VLLM_CONTAINER" = "1" ] && ! container_running "$VLLM_CONTAINER_NAME"; then
      echo "Error: vLLM container exited before becoming ready."
      "${DOCKER_CMD[@]}" logs --tail 120 "$VLLM_CONTAINER_NAME" 2>/dev/null || true
      return 1
    fi

    sleep 1
  done

  echo "Error: vLLM did not become ready after ${VLLM_WAIT_SECONDS}s."
  if [ "$START_VLLM_CONTAINER" = "1" ]; then
    "${DOCKER_CMD[@]}" logs --tail 120 "$VLLM_CONTAINER_NAME" 2>/dev/null || true
  fi
  return 1
}

mkdir -p "$ROOT_DIR/benchmark_results"
mkdir -p "$ROOT_DIR/outputs"

echo "[info] Benchmark type        : $BENCHMARK_TYPE"
echo "[info] Model path            : $MODEL_PATH"
echo "[info] vLLM image            : $VLLM_IMAGE"
echo "[info] vLLM container        : $VLLM_CONTAINER_NAME"
echo "[info] Benchmark container   : $BENCH_CONTAINER_NAME"
echo "[info] Docker network        : $DOCKER_NETWORK_NAME"
echo "[info] Benchmark base URL    : $BASE_URL"
echo "[info] vLLM health URL       : ${VLLM_HEALTH_BASE_URL%/}/health"
echo "[info] Output file           : $ROOT_DIR/$OUTPUT_FILE"
echo "[info] Sampling interval     : ${INTERVAL}s"
echo "[info] GPU sampling interval : ${GPU_INTERVAL}s"
echo ""

echo "[info] Removing leftover benchmark container, if any."
"${DOCKER_CMD[@]}" rm -f "$BENCH_CONTAINER_NAME" >/dev/null 2>&1 || true

if [ "$START_VLLM_CONTAINER" = "1" ]; then
  echo "[info] Removing leftover vLLM container, if any."
  "${DOCKER_CMD[@]}" rm -f "$VLLM_CONTAINER_NAME" >/dev/null 2>&1 || true
fi

if ! "${DOCKER_CMD[@]}" network inspect "$DOCKER_NETWORK_NAME" >/dev/null 2>&1; then
  echo "[info] Creating Docker network: $DOCKER_NETWORK_NAME"
  "${DOCKER_CMD[@]}" network create "$DOCKER_NETWORK_NAME" >/dev/null
  CREATED_DOCKER_NETWORK=1
fi

echo "[info] Starting resource monitor."
OUTPUT_FILE="$ROOT_DIR/$OUTPUT_FILE" \
INTERVAL="$INTERVAL" \
GPU_INTERVAL="$GPU_INTERVAL" \
bash "$SCRIPT_DIR/monitor_docker_resources.sh" "$VLLM_CONTAINER_NAME" "$BENCH_CONTAINER_NAME" &
MONITOR_PID=$!

if [ "$START_VLLM_CONTAINER" = "1" ]; then
  echo "[info] Starting vLLM container."

  VLLM_DOCKER_ARGS=(
    run -d
    --name "$VLLM_CONTAINER_NAME"
    --gpus all
    --network "$DOCKER_NETWORK_NAME"
    -p "${VLLM_PORT}:${VLLM_PORT}"
    --ipc host
    --env HF_HOME=/root/.cache/huggingface
  )

  if [ -d "$HF_CACHE_DIR" ]; then
    VLLM_DOCKER_ARGS+=(-v "$HF_CACHE_DIR:/root/.cache/huggingface")
  fi
  if [ -n "${HF_TOKEN:-}" ]; then
    VLLM_DOCKER_ARGS+=(--env HF_TOKEN)
  fi
  if [ -n "${HUGGING_FACE_HUB_TOKEN:-}" ]; then
    VLLM_DOCKER_ARGS+=(--env HUGGING_FACE_HUB_TOKEN)
  fi

  read -r -a VLLM_EXTRA_ARGV <<< "$VLLM_EXTRA_ARGS"
  if [ "$VLLM_ENFORCE_EAGER" = "1" ]; then
    VLLM_EXTRA_ARGV+=(--enforce-eager)
  fi

  "${DOCKER_CMD[@]}" "${VLLM_DOCKER_ARGS[@]}" "$VLLM_IMAGE" \
    --model "$MODEL_PATH" \
    --host 0.0.0.0 \
    --port "$VLLM_PORT" \
    --dtype "$VLLM_DTYPE" \
    --max-model-len "$VLLM_MAX_MODEL_LEN" \
    --gpu-memory-utilization "$VLLM_GPU_MEMORY_UTILIZATION" \
    "${VLLM_EXTRA_ARGV[@]}"
else
  echo "[info] START_VLLM_CONTAINER=0; expecting vLLM to already be available."
fi

wait_for_vllm

echo "[info] Starting benchmark container."
BENCH_EXIT=0

BENCH_DOCKER_ARGS=(
  run
  --name "$BENCH_CONTAINER_NAME"
  --network "$DOCKER_NETWORK_NAME"
  --add-host=host.docker.internal:host-gateway
  -v "$ROOT_DIR/benchmark_results:/app/benchmark_results"
  -v "$ROOT_DIR/outputs:/app/outputs"
)

# Optional LangSmith tracing config. When the gitignored env file exists, mount
# its variables into the container so the agent publishes spans (mirrors the
# gpt-researcher LANGCHAIN_TRACING_V2 / LANGCHAIN_API_KEY pattern).
LANGSMITH_ENV_FILE="${LANGSMITH_ENV_FILE-$ROOT_DIR/.env.langsmith}"
if [ -f "$LANGSMITH_ENV_FILE" ]; then
  BENCH_DOCKER_ARGS+=(--env-file "$LANGSMITH_ENV_FILE")
  echo "[info] LangSmith env file mounted: $LANGSMITH_ENV_FILE"
fi

"${DOCKER_CMD[@]}" "${BENCH_DOCKER_ARGS[@]}" \
  "$BENCH_IMAGE" \
  python benchmark_latency.py \
    --benchmark-type "$BENCHMARK_TYPE" \
    --base-url "$BASE_URL" \
    --model-path "$MODEL_PATH" || BENCH_EXIT=$?

stop_monitor

echo ""
echo "[info] Benchmark finished with exit code: $BENCH_EXIT"
echo "[info] Stats written to: $ROOT_DIR/$OUTPUT_FILE"
echo ""
echo "--- Last 10 stats entries ---"
tail -10 "$ROOT_DIR/$OUTPUT_FILE" || true

exit "$BENCH_EXIT"
