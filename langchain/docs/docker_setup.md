# Docker Setup for LangChain

This setup mirrors the mini-swe-agent Docker pattern:

- Build one local image for the LangChain benchmark runner from `langchain/`.
- Run vLLM as a separate GPU container from `vllm/vllm-openai:latest`.
- Connect the LangChain container to vLLM through the OpenAI-compatible HTTP API.
- Sample cgroup and GPU stats into `stats_log.csv` via `scripts/monitor_docker_resources.sh`.

## Build the LangChain Image

From the `langchain/` directory (not the repo root):

```bash
cd /path/to/cpu-centric-agentic-ai/langchain
test -d nltk_data || cp -a ../nltk_data ./nltk_data
docker build -t langchain-agent:local .
```

Confirm the image starts:

```bash
docker run --rm langchain-agent:local
```

## Web Search API

The orchestrator uses Tavily when web search is enabled. The Docker runner defaults to `SKIP_WEB_SEARCH=1`. To enable live search:

```bash
export TAVILY_API_KEY="your-tavily-api-key"
SKIP_WEB_SEARCH=0 ./scripts/run_benchmark_docker.sh freshQA
```

Do not put API keys in the Dockerfile or committed config.

## Run vLLM and LangChain Together

```bash
./scripts/run_benchmark_docker.sh freshQA
```

Defaults for the vLLM container match mini-swe-agent (`bfloat16`, `--enforce-eager`, `--no-enable-prefix-caching`, etc.).

Useful overrides:

```bash
BATCH_SIZE=8 VERBOSE=1 ./scripts/run_benchmark_docker.sh freshQA
HF_CACHE_DIR=/path/to/hf_home ./scripts/run_benchmark_docker.sh freshQA
START_VLLM_CONTAINER=0 ./scripts/run_benchmark_docker.sh freshQA
```

## Outputs

| Path | Description |
|------|-------------|
| `stats_log.csv` | Resource monitor CSV (written next to `langchain/` by default). |
| `benchmark_results/` | Trace JSON from `--trace-output` and other artifacts. |

See [docker.md](docker.md) for the full run workflow and plotting commands.
