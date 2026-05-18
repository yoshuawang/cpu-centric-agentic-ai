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
