# Running LangChain in Docker (benchmark + resource trace)

This guide covers building the benchmark image, starting vLLM, running `orchestrator.py` inside Docker, and collecting **`stats_log.csv`** (CPU, memory, block I/O, optional host GPU) at configurable intervals.

For extra Docker build notes and troubleshooting, see [docker_setup.md](docker_setup.md).

## Prerequisites

- **Docker** (user in `docker` group, or the scripts will use `sudo docker` when the socket is not accessible).
- **Linux with cgroup v2** (unified hierarchy). The sampler reads `/sys/fs/cgroup/...` from the host.
- **vLLM** reachable from the benchmark container (the script starts a GPU vLLM container by default).
- **Optional:** `nvidia-smi` on the host for GPU columns in `stats_log.csv`.

## 1. Prepare NLTK data and build the image

From the `langchain/` directory:

```bash
cd /path/to/cpu-centric-agentic-ai/langchain
# One-time: copy shared NLTK data into the build context (if missing)
test -d nltk_data || cp -a ../nltk_data ./nltk_data
docker build -t langchain-agent:local .
```

Smoke test:

```bash
docker run --rm langchain-agent:local
```

You should see `orchestrator.py --help`.

## 2. Run the benchmark and collect stats

```bash
cd /path/to/cpu-centric-agentic-ai/langchain
chmod +x scripts/run_benchmark_docker.sh   # once
./scripts/run_benchmark_docker.sh [BENCHMARK] [BASE_URL] [MODEL_PATH]
```

**Positional arguments (all optional):**

| Argument | Default | Meaning |
|----------|---------|---------|
| `BENCHMARK` | `freshQA` | Dataset (`freshQA`, `musique`, `QASC`). |
| `BASE_URL` | `http://vllm-server:5000/v1` | OpenAI-compatible API base URL when vLLM is started by the script. |
| `MODEL_PATH` | `openai/gpt-oss-20b` | Model id passed to vLLM and the orchestrator. |

**Environment variables:**

| Variable | Default | Meaning |
|----------|---------|---------|
| `INTERVAL` | `0.05` | Seconds between container cgroup samples. |
| `GPU_INTERVAL` | `0.1` | Minimum seconds between `nvidia-smi` refreshes. |
| `BATCH_SIZE` | `1` | Queries per orchestrator batch. |
| `SKIP_WEB_SEARCH` | `1` | Use static URLs (no Tavily key). |
| `START_VLLM_CONTAINER` | `1` | Set to `0` to reuse an already running vLLM. |

Example:

```bash
BATCH_SIZE=4 VERBOSE=1 ./scripts/run_benchmark_docker.sh freshQA
```

Reuse existing vLLM on the host:

```bash
START_VLLM_CONTAINER=0 \
  VLLM_HEALTH_BASE_URL=http://127.0.0.1:5000 \
  ./scripts/run_benchmark_docker.sh freshQA
```

## 3. Outputs

| Path | Description |
|------|-------------|
| **`stats_log.csv`** | Per-container CPU/memory/net/block I/O plus host GPU aggregates. |
| **`benchmark_results/`** | Trace JSON and other run artifacts (bind-mounted from the host). |

Plot resource traces:

```bash
python scripts/plot_resource_trace.py stats_log.csv --events benchmark_results/trace_*.json
```

## 4. Host-only benchmark (no Docker)

```bash
python orchestrator.py --batch-size 4 --benchmark freshQA --skip-web-search --verbose
```

Parallel batch sweeps on the host:

```bash
./scripts/run_batch_experiment.sh
```
