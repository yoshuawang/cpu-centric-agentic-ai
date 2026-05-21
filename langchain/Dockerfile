FROM python:3.9-slim

ARG DEBIAN_FRONTEND=noninteractive

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    NLTK_DATA=/app/nltk_data

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    build-essential \
    curl \
    git \
    procps \
    && rm -rf /var/lib/apt/lists/*

COPY . /app

RUN python -m pip install --upgrade pip setuptools wheel && \
    pip install \
      beautifulsoup4==4.14.2 \
      bs4==0.0.2 \
      langchain==0.3.27 \
      langchain-community==0.3.31 \
      langchain-core==0.3.79 \
      langchain-text-splitters==0.3.11 \
      langgraph==0.6.10 \
      langsmith \
      matplotlib==3.9.4 \
      nltk==3.9.2 \
      numpy==2.0.2 \
      nvtx==0.2.13 \
      openai==2.3.0 \
      pandas==2.3.3 \
      requests==2.32.5 \
      sumy==0.11.0 \
      tavily-python

RUN python -m nltk.downloader -d /app/nltk_data punkt_tab

CMD ["python", "orchestrator.py", "--help"]
