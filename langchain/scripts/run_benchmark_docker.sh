#!/usr/bin/env bash
# Run vLLM and the LangChain orchestrator in separate Docker containers and capture stats.
#
# Usage:
#   ./scripts/run_benchmark_docker.sh [BENCHMARK] [BASE_URL] [MODEL_PATH]
#
# Defaults:
#   BENCHMARK   freshQA
#   BASE_URL    http://vllm-server:${VLLM_PORT}/v1 when START_VLLM_CONTAINER=1
#   MODEL_PATH  openai/gpt-oss-20b
#
# Important environment overrides:
#   TAVILY_API_KEY               required unless SKIP_WEB_SEARCH=1
#   SKIP_WEB_SEARCH              0; set to 1 to use static URLs (no Tavily)
#   BATCH_SIZE                   1
#   RUN_ID                       UTC timestamp by default
#   VERBOSE                      0; set to 1 for per-stage timing stats
#   TRACE_OUTPUT                 /app/benchmark_results/trace_${BENCHMARK}_batch${BATCH_SIZE}.json; set empty to disable
#   VLLM_PORT                    5000
#   VLLM_IMAGE                   vllm/vllm-openai:latest
#   VLLM_DTYPE                   bfloat16
#   VLLM_MAX_MODEL_LEN           8192
#   VLLM_GPU_MEMORY_UTILIZATION  0.90
#   VLLM_ENFORCE_EAGER           0; set to 1 to disable CUDA graphs (slower but easier to debug)
#   VLLM_DISABLE_PREFIX_CACHING  1
#   VLLM_EXTRA_ARGS              extra args appended to the vLLM command
#   VLLM_HEALTH_BASE_URL         http://127.0.0.1:${VLLM_PORT}
#   START_VLLM_CONTAINER         1; set to 0 to reuse an already running server
#   KEEP_CONTAINERS              0; set to 1 to leave containers running
#   INTERVAL                     0.05 seconds for cgroup samples
#   GPU_INTERVAL                 0.1 seconds; effective nvidia-smi cadence is ~0.16s
#
# Outputs:
#   benchmark_results/           trace JSON and other benchmark outputs
#   stats_log.csv                per-container cgroup stats plus host/GPU stats

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LANGCHAIN_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
ROOT_DIR="$(cd "$LANGCHAIN_DIR/.." && pwd)"

normalize_openai_base_url() {
  local url="${1%/}"
  if [[ "$url" == */v1 ]]; then
    echo "$url"
  else
    echo "${url}/v1"
  fi
}

resolve_default_hf_cache_dir() {
  local base="/data1/joshw/hugging_face"
  local repo_dir=""

  if [[ "$MODEL_PATH" != /* && "$MODEL_PATH" == */* ]]; then
    repo_dir="models--${MODEL_PATH//\//--}"
    if [ -d "$base/hf_home/hub/$repo_dir" ]; then
      echo "$base/hf_home"
      return
    fi
    if [ -d "$base/hub/$repo_dir" ]; then
      echo "$base"
      return
    fi
  fi

  if [ -d "$base/hf_home/hub" ]; then
    echo "$base/hf_home"
  else
    echo "$base"
  fi
}

BENCHMARK="${1:-freshQA}"
BASE_URL_ARG="${2:-}"
MODEL_PATH="${3:-openai/gpt-oss-20b}"

VLLM_PORT="${VLLM_PORT:-5000}"
VLLM_IMAGE="${VLLM_IMAGE:-vllm/vllm-openai:latest}"
VLLM_DTYPE="${VLLM_DTYPE:-bfloat16}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-8192}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.90}"
VLLM_ENFORCE_EAGER="${VLLM_ENFORCE_EAGER:-0}"
VLLM_DISABLE_PREFIX_CACHING="${VLLM_DISABLE_PREFIX_CACHING:-1}"
VLLM_EXTRA_ARGS="${VLLM_EXTRA_ARGS:-}"
VLLM_HEALTH_BASE_URL="${VLLM_HEALTH_BASE_URL:-http://127.0.0.1:${VLLM_PORT}}"
VLLM_WAIT_SECONDS="${VLLM_WAIT_SECONDS:-900}"
START_VLLM_CONTAINER="${START_VLLM_CONTAINER:-1}"
KEEP_CONTAINERS="${KEEP_CONTAINERS:-0}"

# Load Tavily key from gitignored env file when present (before SKIP_WEB_SEARCH default).
TAVILY_ENV_FILE="${TAVILY_ENV_FILE-$LANGCHAIN_DIR/.env.tavily}"
if [ -f "$TAVILY_ENV_FILE" ]; then
  set -a
  # shellcheck disable=SC1090
  source "$TAVILY_ENV_FILE"
  set +a
fi

SKIP_WEB_SEARCH="${SKIP_WEB_SEARCH:-0}"
BATCH_SIZE="${BATCH_SIZE:-1}"
JOB_ID="${JOB_ID:-1}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
VERBOSE="${VERBOSE:-0}"
TRACE_OUTPUT="${TRACE_OUTPUT-/app/benchmark_results/trace_${BENCHMARK}_batch${BATCH_SIZE}_${RUN_ID}.json}"

VLLM_CONTAINER_NAME="${VLLM_CONTAINER_NAME:-vllm-server}"
LANGCHAIN_CONTAINER_NAME="${LANGCHAIN_CONTAINER_NAME:-langchain-bench}"
DOCKER_NETWORK_NAME="${DOCKER_NETWORK_NAME:-langchain-bench-net}"
LANGCHAIN_IMAGE="${LANGCHAIN_IMAGE:-langchain-agent:local}"
RESULTS_DIR="${RESULTS_DIR:-$LANGCHAIN_DIR/benchmark_results}"
DEFAULT_HF_CACHE_DIR="$(resolve_default_hf_cache_dir)"
HF_CACHE_DIR="${HF_CACHE_DIR:-${HF_HOME:-$DEFAULT_HF_CACHE_DIR}}"
OUTPUT_FILE="${OUTPUT_FILE:-$LANGCHAIN_DIR/stats_log.csv}"
INTERVAL="${INTERVAL:-0.05}"
GPU_INTERVAL="${GPU_INTERVAL:-0.1}"

if [ -n "$BASE_URL_ARG" ]; then
  BASE_URL="$(normalize_openai_base_url "$BASE_URL_ARG")"
elif [ "$START_VLLM_CONTAINER" = "1" ]; then
  BASE_URL="http://${VLLM_CONTAINER_NAME}:${VLLM_PORT}/v1"
else
  BASE_URL="http://host.docker.internal:${VLLM_PORT}/v1"
fi

if [ "$SKIP_WEB_SEARCH" != "1" ] && [ -z "${TAVILY_API_KEY:-}" ]; then
  echo "Error: TAVILY_API_KEY is required unless SKIP_WEB_SEARCH=1."
  echo "Add it to langchain/.env.tavily or export TAVILY_API_KEY in your shell."
  exit 1
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

save_vllm_logs() {
  if [ "$START_VLLM_CONTAINER" != "1" ]; then
    return 0
  fi
  local vllm_log_file="${RESULTS_DIR}/vllm.log"
  if "${DOCKER_CMD[@]}" inspect "$VLLM_CONTAINER_NAME" >/dev/null 2>&1; then
    echo "[info] Saving vLLM logs to: $vllm_log_file"
    "${DOCKER_CMD[@]}" logs "$VLLM_CONTAINER_NAME" > "$vllm_log_file" 2>&1 || true
  fi
}

cleanup() {
  stop_monitor
  save_vllm_logs
  if [ "$KEEP_CONTAINERS" != "1" ]; then
    "${DOCKER_CMD[@]}" rm -f "$LANGCHAIN_CONTAINER_NAME" >/dev/null 2>&1 || true
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

mkdir -p "$RESULTS_DIR"

echo "[info] Benchmark type        : $BENCHMARK"
echo "[info] Batch size            : $BATCH_SIZE"
echo "[info] Model path            : $MODEL_PATH"
echo "[info] vLLM image            : $VLLM_IMAGE"
echo "[info] vLLM container        : $VLLM_CONTAINER_NAME"
echo "[info] LangChain container   : $LANGCHAIN_CONTAINER_NAME"
echo "[info] Docker network        : $DOCKER_NETWORK_NAME"
echo "[info] OpenAI base URL       : $BASE_URL"
echo "[info] vLLM health URL       : ${VLLM_HEALTH_BASE_URL%/}/health"
echo "[info] HF cache dir          : $HF_CACHE_DIR"
echo "[info] Skip web search       : $SKIP_WEB_SEARCH"
echo "[info] Results dir           : $RESULTS_DIR"
echo "[info] Output file           : $OUTPUT_FILE"
echo "[info] Sampling interval     : ${INTERVAL}s"
echo "[info] GPU sampling interval : ${GPU_INTERVAL}s"
echo ""

echo "[info] Removing leftover LangChain container, if any."
"${DOCKER_CMD[@]}" rm -f "$LANGCHAIN_CONTAINER_NAME" >/dev/null 2>&1 || true

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
OUTPUT_FILE="$OUTPUT_FILE" \
INTERVAL="$INTERVAL" \
GPU_INTERVAL="$GPU_INTERVAL" \
bash "$SCRIPT_DIR/monitor_docker_resources.sh" "$VLLM_CONTAINER_NAME" "$LANGCHAIN_CONTAINER_NAME" &
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
  if [ "$VLLM_DISABLE_PREFIX_CACHING" = "1" ]; then
    VLLM_EXTRA_ARGV+=(--no-enable-prefix-caching)
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

LANGCHAIN_DOCKER_ARGS=(
  run
  --name "$LANGCHAIN_CONTAINER_NAME"
  --network "$DOCKER_NETWORK_NAME"
  --add-host=host.docker.internal:host-gateway
  -v "$RESULTS_DIR:/app/benchmark_results"
  --env "VLLM_OPENAI_BASE_URL=$BASE_URL"
  --env "VLLM_MODEL=$MODEL_PATH"
  --env "LANGCHAIN_RUN_ID=$RUN_ID"
  --env "LANGCHAIN_IMAGE=$LANGCHAIN_IMAGE"
  --env "LANGCHAIN_CONTAINER_NAME=$LANGCHAIN_CONTAINER_NAME"
  --env "VLLM_IMAGE=$VLLM_IMAGE"
  --env "VLLM_CONTAINER_NAME=$VLLM_CONTAINER_NAME"
  --env "DOCKER_NETWORK_NAME=$DOCKER_NETWORK_NAME"
  --env "START_VLLM_CONTAINER=$START_VLLM_CONTAINER"
  --env "VLLM_PORT=$VLLM_PORT"
)

if [ -f "$TAVILY_ENV_FILE" ]; then
  LANGCHAIN_DOCKER_ARGS+=(--env-file "$TAVILY_ENV_FILE")
  echo "[info] Tavily env file mounted: $TAVILY_ENV_FILE"
elif [ -n "${TAVILY_API_KEY:-}" ]; then
  LANGCHAIN_DOCKER_ARGS+=(--env TAVILY_API_KEY)
fi
if [ -n "${VLLM_MAX_TOKENS:-}" ]; then
  LANGCHAIN_DOCKER_ARGS+=(--env VLLM_MAX_TOKENS)
fi
if [ -n "${VLLM_TEMPERATURE:-}" ]; then
  LANGCHAIN_DOCKER_ARGS+=(--env VLLM_TEMPERATURE)
fi

# Optional LangSmith tracing config. When the gitignored env file exists, mount
# its variables into the container so the orchestrator publishes spans (mirrors
# the gpt-researcher LANGCHAIN_TRACING_V2 / LANGCHAIN_API_KEY pattern).
LANGSMITH_ENV_FILE="${LANGSMITH_ENV_FILE-$LANGCHAIN_DIR/.env.langsmith}"
if [ -f "$LANGSMITH_ENV_FILE" ]; then
  LANGCHAIN_DOCKER_ARGS+=(--env-file "$LANGSMITH_ENV_FILE")
  echo "[info] LangSmith env file mounted: $LANGSMITH_ENV_FILE"
fi

LANGCHAIN_ARGS=(
  python orchestrator.py
  --benchmark "$BENCHMARK"
  --batch-size "$BATCH_SIZE"
  --job-id "$JOB_ID"
)

if [ "$SKIP_WEB_SEARCH" = "1" ]; then
  LANGCHAIN_ARGS+=(--skip-web-search)
fi
if [ "$VERBOSE" = "1" ]; then
  LANGCHAIN_ARGS+=(--verbose)
fi
if [ -n "$TRACE_OUTPUT" ]; then
  LANGCHAIN_ARGS+=(--trace-output "$TRACE_OUTPUT")
fi

echo "[info] Starting LangChain container."
LANGCHAIN_EXIT=0
"${DOCKER_CMD[@]}" "${LANGCHAIN_DOCKER_ARGS[@]}" "$LANGCHAIN_IMAGE" "${LANGCHAIN_ARGS[@]}" || LANGCHAIN_EXIT=$?

stop_monitor

echo ""
echo "[info] LangChain benchmark finished with exit code: $LANGCHAIN_EXIT"
echo "[info] Stats written to: $OUTPUT_FILE"
if [ -n "$TRACE_OUTPUT" ]; then
  echo "[info] Trace output path in container: $TRACE_OUTPUT"
  echo "[info] Host results dir: $RESULTS_DIR"
fi
echo ""
echo "--- Last 10 stats entries ---"
tail -10 "$OUTPUT_FILE" || true

exit "$LANGCHAIN_EXIT"
