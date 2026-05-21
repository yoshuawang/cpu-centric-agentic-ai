# LangChain Orchestrator

A batch LLM orchestrator built with LangGraph for web-queried QA benchmarks, with per-stage timing, optional NVTX profiling, and Docker-based resource tracing (aligned with the [mini-swe-agent](../mini-swe-agent/) layout).

## Overview

The orchestrator runs queries through a four-stage pipeline and records timing for each stage. It supports parallel batches, JSON trace export, and host-side cgroup/GPU sampling when run via Docker.

```
web_search → fetch_url → summarize → final_answer
```

| Stage | Description |
|-------|-------------|
| **web_search** | Tavily Search API (optional; use `--skip-web-search` for static URLs) |
| **fetch_url** | Download and extract page text (parallel) |
| **summarize** | LexRank extractive summaries (parallel) |
| **llm_inference** | Final answer via vLLM OpenAI-compatible API |

## Architecture

```
User → orchestrator.py (LangGraph) → vLLM (OpenAI API)
              ↓
        benchmark_results/  (traces, batch logs)
        stats_log.csv       (Docker resource monitor)
```

## Directory layout

```
langchain/
├── orchestrator.py              # Main benchmark entry
├── orchestrator_cpu_only.py     # CPU-only variant (no LLM stage)
├── Dockerfile
├── .dockerignore
├── stats_log.csv                # Written by run_benchmark_docker.sh (gitignored)
├── benchmark_results/           # Traces, batch logs, plot outputs
├── docs/
│   ├── docker.md                # Docker run + stats collection
│   └── docker_setup.md          # Build notes and troubleshooting
├── notebooks/
│   └── trace_analysis.ipynb
└── scripts/
    ├── run_benchmark_docker.sh      # vLLM + LangChain containers + monitor
    ├── monitor_docker_resources.sh  # cgroup + nvidia-smi → stats_log.csv
    ├── plot_resource_trace.py       # Plot stats_log.csv time series
    ├── run_batch_experiment.sh        # Host parallel batch sweep
    ├── run_batch_experiment_verbose.sh
    ├── bash_parallel.sh
    ├── cgam_overlap.sh
    ├── parse_stats.py
    ├── print_trace_report.py
    └── plot_*.py
```

## Installation

### Conda environment

```bash
conda activate langchain
```

### Dependencies (host)

```bash
pip install langchain==0.3.27 langgraph==0.6.10 langchain-core==0.3.79 langchain-community==0.3.31
pip install requests beautifulsoup4==4.14.2 sumy==0.11.0
pip install nvtx tavily-python openai
```
 
### External Services
 
- **Tavily Search API**: Requires `TAVILY_API_KEY` when web search is enabled.
    - Keep this key in your shell or secret manager.
    - Do not put it in the Dockerfile, Docker image, or committed scripts.
- **VLLM Server**: Local OpenAI-compatible server running at `http://localhost:5000/v1`
    - The Docker runner starts `vllm/vllm-openai:latest` as a separate GPU container by default.
    - It mounts the local Hugging Face cache and auto-resolves `openai/gpt-oss-20b` under `/data1/joshw/hugging_face/hf_home`.
    - It starts vLLM with `--enforce-eager` and `--no-enable-prefix-caching` by default, matching the mini-swe-agent Docker setup.
 
## Environment Setup
 
```bash
export TAVILY_API_KEY="your-tavily-api-key"
export VLLM_OPENAI_BASE_URL="http://localhost:5000/v1"
export VLLM_MODEL="openai/gpt-oss-20b"
```

## Usage

### Host benchmark

From the `langchain/` directory:

```bash
python orchestrator.py --batch-size 4 --benchmark freshQA --skip-web-search --verbose
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--batch-size` | `1` | Parallel queries per batch |
| `--benchmark` | `freshQA` | Dataset: `freshQA`, `musique`, `QASC` |
| `--skip-web-search` | off | Use built-in static URLs (no Tavily) |
| `--verbose` | off | Print per-stage timing table |
| `--trace-output` | none | Write per-query JSON trace |
| `--job-id` | `1` | Job id prefix in logs |

### Docker + resource trace

Run inside Docker with a separate vLLM GPU container and collect **`stats_log.csv`** (cgroup CPU/memory, block I/O, host GPU via `nvidia-smi`). Full details: **[docs/docker.md](docs/docker.md)**.

```bash
cd langchain

# One-time: NLTK data for the image build context
test -d nltk_data || cp -a ../nltk_data ./nltk_data

docker build -t langchain-agent:local .
./scripts/run_benchmark_docker.sh freshQA
```

**Common overrides:**

```bash
BATCH_SIZE=8 VERBOSE=1 ./scripts/run_benchmark_docker.sh freshQA

# Reuse an already-running vLLM on the host
START_VLLM_CONTAINER=0 \
  VLLM_HEALTH_BASE_URL=http://127.0.0.1:5000 \
  ./scripts/run_benchmark_docker.sh freshQA
```

**Plot resource traces:**

```bash
python scripts/plot_resource_trace.py stats_log.csv
```

### Host batch sweeps (no Docker)

```bash
./scripts/run_batch_experiment.sh
./scripts/run_batch_experiment_verbose.sh   # writes CSV under benchmark_results/
```

## Benchmarks

| Name | Description |
|------|-------------|
| `freshQA` | FreshQA-style queries (default) |
| `musique` | Multi-hop QA |
| `QASC` | Science QA |

## Outputs

| Path | Description |
|------|-------------|
| **`benchmark_results/`** | Trace JSON (`trace_<benchmark>_batch<N>_<run_id>.json`), batch experiment logs, generated figures |
| **`stats_log.csv`** | Per-container resource samples from Docker runs (default: `langchain/stats_log.csv`) |

**Console timing:**

```
<job_id>: [TIMING] start: <timestamp>s
<job_id>: [TIMING] end: <elapsed>s
```

**Trace JSON** (`--trace-output` or Docker `TRACE_OUTPUT`): per-query stage timings and run metadata (model, endpoints, container names). Secrets such as `TAVILY_API_KEY` are never written.

**Print a trace report:**

```bash
python scripts/print_trace_report.py benchmark_results/trace_*.json
```

## Performance monitoring

### Stage timing

With `--verbose`, the orchestrator prints count / avg / min / max per stage: `web_search`, `fetch_url`, `summarize`, `llm_inference`.

### NVTX (optional)

Stages are annotated for NVIDIA Nsight Systems:

```bash
nsys profile -t nvtx,cuda python orchestrator.py --batch-size 8 --skip-web-search
nsys-ui output_profile.nsys-rep
```

### Resource CSV columns

`stats_log.csv` includes per-container cgroup metrics plus host-wide GPU columns (`GPU_Util_Max`, `GPU_Mem_Used`, `GPU_Mem_Perc`). GPU values reflect the whole machine (dominated by vLLM), not per-container VRAM attribution.

### LangSmith tracing (optional)

Mirrors the gpt-researcher setup: when `LANGCHAIN_API_KEY` is present, the orchestrator flips `LANGCHAIN_TRACING_V2=true` and LangGraph publishes per-node spans to [smith.langchain.com](https://smith.langchain.com).

Setup:

1. Copy your key into the gitignored env file `langchain/.env.langsmith`:

   ```bash
   LANGCHAIN_TRACING_V2=true
   LANGCHAIN_API_KEY=ls__...
   LANGCHAIN_PROJECT=langchain-benchmark
   ```

2. Host runs: `source langchain/.env.langsmith && export $(grep -v '^#' langchain/.env.langsmith | cut -d= -f1)`, then run `orchestrator.py` as usual.
3. Docker runs: `scripts/run_benchmark_docker.sh` auto-mounts the file when present (override with `LANGSMITH_ENV_FILE=...`).

Spans you'll see per query:

| Span | Source |
|------|--------|
| `web_search`, `fetch_url`, `summarize`, `final_answer` | LangGraph nodes (auto) |
| `vllm_completion` (`run_type="llm"`) | `@traceable` on `_vllm_completion_non_stream_with_headers` |

Local timing (`--verbose`, `--trace-output` JSON, NVTX, `stats_log.csv`) is unchanged and runs alongside LangSmith.

## Configuration

- **URL fetch limit**: edit the `if len(texts) >= 2` guard in `orchestrator.py`.
- **vLLM**: `VLLM_OPENAI_BASE_URL`, `VLLM_MODEL`, optional `VLLM_MAX_TOKENS`, `VLLM_TEMPERATURE`.
- **Docker runner**: see env vars in `scripts/run_benchmark_docker.sh` header (`INTERVAL`, `GPU_INTERVAL`, `BATCH_SIZE`, `SKIP_WEB_SEARCH`, etc.).

## Error handling

- Missing `TAVILY_API_KEY` without `--skip-web-search` → `RuntimeError`
- Failed URL fetches are skipped; the pipeline continues with available content
- HTTP requests use a 10-second timeout

## Development

To add a pipeline stage:

1. Define a node with `GraphState`
2. Add NVTX markers and timing if needed
3. Register the node and edges in the graph builder

```python
def new_stage(state: GraphState) -> GraphState:
    nvtx.push_range("new_stage")
    # ...
    nvtx.pop_range()
    return {"new_field": result}

builder.add_node("new_stage", new_stage)
builder.add_edge("previous_stage", "new_stage")
```
