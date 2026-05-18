# Running mini-swe-agent in Docker (benchmark + resource trace)

This guide covers building the benchmark image, starting vLLM, running `benchmark_latency.py` inside Docker, and collecting **`stats_log.csv`** (CPU, memory, block I/O, optional host GPU) at configurable intervals.

For extra Docker build notes and troubleshooting, see [docker_setup.md](docker_setup.md).

## Prerequisites

- **Docker** (user in `docker` group, or the scripts will use `sudo docker` when the socket is not accessible).
- **Linux with cgroup v2** (unified hierarchy). The sampler reads `/sys/fs/cgroup/...` from the host; if `0::` is missing in `/proc/<pid>/cgroup`, the script exits with an error.
- **vLLM** (or any OpenAI-compatible HTTP API) reachable from the host at the URL you pass to the benchmark (default `http://127.0.0.1:5000`).
- **Optional:** `nvidia-smi` on the host to append **system-wide** GPU utilization and memory columns to each row (not per-container).

## 1. Build the image

From this directory:

```bash
cd /path/to/cpu-centric-agentic-ai/mini-swe-agent
docker build -t mini-swe-agent:local .
```

Smoke test:

```bash
docker run --rm mini-swe-agent:local
```

You should see `benchmark_latency.py --help`.

## 2. Start the vLLM server (on the host)

The container uses **`--network host`**, so the benchmark talks to **localhost on the host** (not inside an isolated bridge network).

Example (adjust paths and model to your machine):

```bash
HF_HOME=/path/to/huggingface_cache /path/to/venv/bin/python -m vllm.entrypoints.openai.api_server \
  --host 0.0.0.0 \
  --port 5000 \
  --model Qwen/Qwen2.5-Coder-32B-Instruct \
  --dtype bfloat16 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.90 \
  --enforce-eager
```

Keep this process running while you run the benchmark script.

## 3. Run the benchmark and collect stats

```bash
cd /path/to/cpu-centric-agentic-ai/mini-swe-agent
chmod +x scripts/run_benchmark_docker.sh   # once
./scripts/run_benchmark_docker.sh [BENCHMARK_TYPE] [BASE_URL] [MODEL_PATH]
```

**Positional arguments (all optional):**

| Argument | Default | Meaning |
|----------|---------|---------|
| `BENCHMARK_TYPE` | `sorting` | Passed to `--benchmark-type` (e.g. `integration`, `knn`, `swebench`, …). |
| `BASE_URL` | `http://127.0.0.1:5000` | vLLM OpenAI-compatible base URL (no `/v1` suffix required by the benchmark; see `benchmark_latency.py`). |
| `MODEL_PATH` | `Qwen/Qwen2.5-Coder-32B-Instruct` | Model id string sent to the API. |

**Environment variables:**

| Variable | Default | Meaning |
|----------|---------|---------|
| `INTERVAL` | `0.1` | Seconds between **container** samples. CPU% is derived from cgroup v2 `cpu.stat` deltas, so this is the real cadence (not limited by `docker stats`). |
| `GPU_INTERVAL` | `1.0` | Minimum seconds between **`nvidia-smi`** refreshes when GPU stats are enabled. |

Example with a different benchmark and remote vLLM:

```bash
INTERVAL=0.2 GPU_INTERVAL=2.0 \
  ./scripts/run_benchmark_docker.sh integration http://192.168.1.10:5000 Qwen/Qwen2.5-Coder-32B-Instruct
```

## 4. Outputs

| Path | Description |
|------|-------------|
| **`stats_log.csv`** | One row per sample: timestamp, container name, CPU%, memory usage/limit/%, net in/out (see note below), block in/out, then GPU aggregate columns if `nvidia-smi` works. |
| **`benchmark_results/`** | Benchmark JSON/logs from inside the container (bind-mounted from the host). |

**Note on network columns:** With `--network host`, per-container network accounting in cgroup v2 is not used the same way as on a bridge network; the script keeps the CSV schema and may report **`0B`** for net I/O. Use host-level tools (e.g. `iftop`, `nethogs`) if you need network breakdown for that setup.

**Note on GPU columns:** Values come from **`nvidia-smi` on the host** (all GPUs aggregated in the script’s summary). They reflect **GPU activity on the machine**, dominated by vLLM during a run—not an isolated “Docker GPU cgroup” per container.

## 5. Manual Docker run (without the script)

Equivalent to what the script does for the workload (stats collection is only in `scripts/run_benchmark_docker.sh`):

```bash
docker run --rm \
  --name mswa-bench \
  --network host \
  -v "$PWD/benchmark_results:/app/benchmark_results" \
  mini-swe-agent:local \
  python benchmark_latency.py \
    --benchmark-type sorting \
    --base-url http://127.0.0.1:5000 \
    --model-path Qwen/Qwen2.5-Coder-32B-Instruct
```

## 6. Troubleshooting

- **Connection refused to vLLM:** Ensure the server is listening on `0.0.0.0` (or the IP you use in `BASE_URL`) and that the port matches.
- **Permission denied on Docker socket:** Install/configure Docker for your user, or rely on the script’s `sudo docker` fallback.
- **cgroup v2 error:** Upgrade/kernel config must expose unified cgroup; mixed v1/v2-only setups are not supported by this sampler.
- **Empty or header-only `stats_log.csv`:** Usually a race or cgroup path issue on an older script; use the current `scripts/run_benchmark_docker.sh` (loop waits on `cpu.stat` under the container cgroup).

## Files involved

- [../Dockerfile](../Dockerfile) — image definition (`PYTHONPATH=/app/src`, dependencies).
- [../.dockerignore](../.dockerignore) — shrinks build context.
- [../scripts/run_benchmark_docker.sh](../scripts/run_benchmark_docker.sh) — benchmark + cgroup v2 + optional GPU sampling.
- [../scripts/monitor_docker_resources.sh](../scripts/monitor_docker_resources.sh) — cgroup and GPU sampler used by the runner.
- [docker_setup.md](docker_setup.md) — original Docker notes and `docker stats` examples.
