# Mini-SWE-Agent Coding Framework
 
A comprehensive coding framework for evaluating LLM-based software engineering agents with detailed latency profiling and performance analysis across multiple computational and coding tasks.
 
## Overview
 
This benchmark measures the performance of autonomous software engineering agents built on large language models (LLMs) using the vLLM inference server. It provides detailed timing breakdowns for LLM inference, bash command execution, and overall task completion across diverse problem domains.
 
## Architecture
 
```
User → LatencyBenchmarker → DefaultAgent → VLLMModel (vLLM Server)
                                    ↓
                              LocalEnvironment (Bash Execution)
                                    ↓
                              Incremental Results Logging
```
 
### Processing Pipeline
 
1. **Task Setup**: Load benchmark configuration and initialize agent
2. **Agent Execution**: Multi-step reasoning with LLM and bash tools
3. **Latency Tracking**: Record timestamps for all LLM calls and bash executions
4. **Result Aggregation**: Compute timing summaries and success metrics
5. **Incremental Saving**: Persist results to prevent data loss
 
## Installation
 
### Requirements

If already setup, activate langchain conda environment-

`conda activate swe`

Or, if you want to setup from scratch-

```bash
pip install datasets vllm pyyaml min
```
 
### Dependencies
 
- **datasets**: HuggingFace datasets library for SWE-bench, SciCode, LiveCodeBench
- **vllm**: GPU-accelerated LLM inference server
- **pyyaml**: YAML configuration parsing
- **Custom modules**: `vllm_model`, `minisweagent` (included in repository)
 
### vLLM Server Setup
 
Start vLLM server with Qwen2.5-Coder or similar coding model:
 
```bash
vllm serve Qwen/Qwen2.5-Coder-32B-Instruct \
    --host 0.0.0.0 \
    --port 5000 \
    --dtype bfloat16 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.95
```
 
## Usage
 
### Basic Benchmark Execution
 
```bash
python benchmark_latency.py --benchmark-type sorting
```

### Docker + resource trace

To run the benchmark inside Docker and write **`stats_log.csv`** (cgroup-based CPU/memory sampling, optional host GPU via `nvidia-smi`), see **[docs/docker.md](docs/docker.md)**.

## Benchmark Types
 
### 1. CPU-Intensive Benchmarks
 
#### Sorting Algorithms
Tests agent's ability to implement and benchmark sorting algorithms (bubble sort on 10K-20K elements).
 
**Metrics**: Implementation correctness, timing accuracy, code quality
 
```bash
python benchmark_latency.py --benchmark-type sorting
```
 
#### Numerical Integration
Implements trapezoidal rule with numpy and scipy for sin(x) integration.
 
**Metrics**: Numerical accuracy, performance comparison, step count handling
 
```bash
python benchmark_latency.py --benchmark-type integration
```
 
#### K-Nearest Neighbors
KNN classifier implementation from scratch or with scikit-learn.
  
```bash
python benchmark_latency.py --benchmark-type knn
```
 
### 2. Software Engineering Benchmarks
 
#### SWE-bench
Real-world GitHub issue resolution from popular repositories.
 
**Dataset**: SWE-bench verified instances
**Source**: HuggingFace `princeton-nlp/SWE-bench_Lite`
 
#### SciCode
Scientific computing problems requiring domain expertise.
 
**Dataset**: SciCode problems from research domains
**Difficulty**: Advanced scientific programming

 
 
## Detailed Logging
 
All benchmarks track:
- **LLM API calls**: Timestamp, duration, prompt/completion tokens, cost
- **Bash executions**: Command, stdout/stderr, exit code, duration
- **Agent messages**: Full conversation history
- **Timing breakdown**: LLM vs. bash vs. overhead percentages

## LangSmith tracing (optional)

Mirrors the gpt-researcher setup: when `LANGCHAIN_API_KEY` is present, `benchmark_latency.py` flips `LANGCHAIN_TRACING_V2=true` and the `@traceable` decorators on the agent loop publish per-stage spans to [smith.langchain.com](https://smith.langchain.com).

Setup:

1. `pip install langsmith` (already in `Dockerfile`).
2. Copy your key into the gitignored env file `mini-swe-agent/.env.langsmith`:

   ```bash
   LANGCHAIN_TRACING_V2=true
   LANGCHAIN_API_KEY=ls__...
   LANGCHAIN_PROJECT=mini-swe-agent
   ```

3. Host runs: export the variables (e.g. `set -a; source mini-swe-agent/.env.langsmith; set +a`) before `python benchmark_latency.py`.
4. Docker runs: `scripts/run_benchmark_docker.sh` auto-mounts the file when present (override with `LANGSMITH_ENV_FILE=...`).

Span hierarchy per task:

| Span | Source |
|------|--------|
| `agent_run` | `DefaultAgent.run` |
| `agent_step` | `DefaultAgent.step` (one per LLM round) |
| `llm_api` (`run_type="llm"`) | `DefaultAgent.query` |
| `vllm_query` (`run_type="llm"`) | `VLLMModel.query` (nested under `llm_api`) |
| `bash_execution` | `LocalEnvironment.execute` |

Span names align with the `usage_time_by_stage` keys already produced by `benchmark_latency.py` (`llm_api`, `bash_execution`, `agent_overhead`). Local JSON / `stats_log.csv` / resource-monitor output is unchanged and runs alongside LangSmith.

After each benchmark run (when `LANGCHAIN_API_KEY` is set), traces are also exported locally under `benchmark_results/`:

| File | Description |
|------|-------------|
| `{benchmark}_langsmith_trace_{timestamp}.csv` | Flattened spans: `name`, `run_type`, `latency_seconds`, `parent_run_id`, … |
| `{benchmark}_langsmith_trace_{timestamp}.json` | Same data plus export metadata |

Example: `benchmark_results/sorting_langsmith_trace_20260520_214500.csv`

 
## Acknowledgments
 
- **SWE-bench Team** at Princeton NLP for the software engineering benchmark
- **Qwen Team** at Alibaba for the Qwen2.5-Coder models
- **vLLM Team** for high-performance inference framework
- **HuggingFace** for datasets infrastructure
- **SciCode** and **LiveCodeBench** contributors
 
 
