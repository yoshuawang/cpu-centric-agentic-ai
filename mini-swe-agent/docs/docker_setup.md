# Docker Setup for mini-swe-agent

This guide explains how to build and run a Docker image for this
`mini-swe-agent` benchmark checkout.

The recommended setup is:

- Run the agent/benchmark code inside Docker.
- Run the vLLM OpenAI-compatible server separately, usually on the host or on a
  GPU machine.
- Connect the agent container to the vLLM server over HTTP.

This keeps the benchmark image lightweight and avoids baking large model weights
or GPU server dependencies into the same image.

## 1. Prerequisites

Install Docker on the machine where you want to run the benchmark container:

```bash
docker --version
docker compose version
```

If you plan to run vLLM inside Docker on the same machine, also install the
NVIDIA Container Toolkit and verify GPU access:

```bash
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

The benchmark container itself does not require GPU access when it only calls an
external vLLM server.

## 2. Add a Dockerfile

Create a file named `Dockerfile` in the repo root:

```dockerfile
FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    git \
    curl \
    build-essential \
    procps \
    && rm -rf /var/lib/apt/lists/*

COPY . /app

RUN python -m pip install --upgrade pip setuptools wheel && \
    pip install \
      datasets \
      requests \
      pyyaml \
      jinja2 \
      python-dotenv \
      platformdirs \
      rich \
      typer \
      prompt_toolkit \
      textual \
      litellm \
      tenacity \
      psutil \
      nvidia-ml-py \
      numpy \
      scipy \
      scikit-learn \
      matplotlib

CMD ["python", "benchmark_latency.py", "--help"]
```

Why this Dockerfile is explicit:

- This checkout does not currently have a root `pyproject.toml` or
  `requirements.txt`.
- `benchmark_latency.py` imports modules from `src/`, so `PYTHONPATH=/app/src`
  is required.
- vLLM is not installed in this image because the benchmark code talks to vLLM
  through an HTTP API.

## 3. Add a .dockerignore

Create a file named `.dockerignore` in the repo root:

```dockerignore
.git
**/__pycache__/
*.pyc
*.pyo
benchmark_results/
notebooks/
.DS_Store
```

This keeps local outputs and cache files out of the image build context.

## 4. Build the Image

From the repo root:

```bash
cd /home/jwang354/cpu-centric-agentic-ai/mini-swe-agent
docker build -t mini-swe-agent:local .
```

Confirm the image starts:

```bash
docker run --rm mini-swe-agent:local
```

You should see the `benchmark_latency.py` help output.

## 5. Start the vLLM Server

Start vLLM outside the benchmark container. Example:

```bash
vllm serve Qwen/Qwen2.5-Coder-32B-Instruct \
  --host 0.0.0.0 \
  --port 5000 \
  --dtype bfloat16 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.95
```

The benchmark script defaults to an OpenAI-compatible API server at
`http://localhost:5000`, but inside a Docker container `localhost` means the
container itself. Use one of the networking setups below.

## 6. Run a Benchmark on Linux

On Linux, the simplest setup is host networking:

```bash
docker run --rm \
  --network host \
  -v "$PWD/benchmark_results:/app/benchmark_results" \
  mini-swe-agent:local \
  python benchmark_latency.py \
    --benchmark-type sorting \
    --base-url http://127.0.0.1:5000 \
    --model-path Qwen/Qwen2.5-Coder-32B-Instruct
```

The volume mount writes results back to the host at:

```text
benchmark_results/
```

## 7. Run a Benchmark on Docker Desktop

On Docker Desktop, use `host.docker.internal`:

```bash
docker run --rm \
  --add-host=host.docker.internal:host-gateway \
  -v "$PWD/benchmark_results:/app/benchmark_results" \
  mini-swe-agent:local \
  python benchmark_latency.py \
    --benchmark-type sorting \
    --base-url http://host.docker.internal:5000 \
    --model-path Qwen/Qwen2.5-Coder-32B-Instruct
```

## 8. Common Benchmark Commands

Run sorting:

```bash
docker run --rm --network host \
  -v "$PWD/benchmark_results:/app/benchmark_results" \
  mini-swe-agent:local \
  python benchmark_latency.py --benchmark-type sorting --base-url http://127.0.0.1:5000
```

Run numerical integration:

```bash
docker run --rm --network host \
  -v "$PWD/benchmark_results:/app/benchmark_results" \
  mini-swe-agent:local \
  python benchmark_latency.py --benchmark-type integration --base-url http://127.0.0.1:5000
```

Run k-NN:

```bash
docker run --rm --network host \
  -v "$PWD/benchmark_results:/app/benchmark_results" \
  mini-swe-agent:local \
  python benchmark_latency.py --benchmark-type knn --base-url http://127.0.0.1:5000
```

## 9. Optional: Run vLLM in Docker

If you want vLLM itself in Docker, run it as a separate GPU container:

```bash
docker run --rm --gpus all \
  --name vllm-server \
  -p 5000:5000 \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  vllm/vllm-openai:latest \
  --model Qwen/Qwen2.5-Coder-32B-Instruct \
  --host 0.0.0.0 \
  --port 5000 \
  --dtype bfloat16 \
  --max-model-len 8192
```

Then run the benchmark container using the same `--base-url` examples above.

## 10. Optional: Collect Resource Stats

For container CPU, memory, network I/O, and block I/O:

```bash
docker stats <container_name_or_id>
```

For GPU utilization and GPU memory:

```bash
gpustat --json
```

For timeline plots, sample both tools at a fixed interval, such as every
`1s`, and use seconds since benchmark start as the x-axis.

## 11. Troubleshooting

If the benchmark cannot connect to vLLM:

- Confirm the vLLM server is running.
- Confirm the server is listening on `0.0.0.0:5000` or the expected host/port.
- On Linux, prefer `--network host` and `--base-url http://127.0.0.1:5000`.
- On Docker Desktop, use `--base-url http://host.docker.internal:5000`.

If Python imports fail:

- Confirm the image was rebuilt after Dockerfile changes.
- Confirm `PYTHONPATH=/app/src` is set in the image.

If benchmark outputs disappear after the container exits:

- Mount `benchmark_results` from the host:

```bash
-v "$PWD/benchmark_results:/app/benchmark_results"
```

If GPU stats are missing:

- The benchmark container does not need GPU access unless it runs GPU code.
- `gpustat` must run on a machine that can see the NVIDIA GPU.
- Verify the host can run `nvidia-smi`.
