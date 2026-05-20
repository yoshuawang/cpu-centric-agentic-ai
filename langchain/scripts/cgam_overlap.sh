#!/bin/bash
# Overlapping parallel orchestrator waves (host).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LANGCHAIN_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
ROOT="$(cd "$LANGCHAIN_DIR/.." && pwd)"
ORCHESTRATOR="$LANGCHAIN_DIR/orchestrator.py"

while [[ $# -gt 0 ]]; do
    case $1 in
        -r|--root)
            ROOT="$2"
            LANGCHAIN_DIR="$ROOT/langchain"
            ORCHESTRATOR="$LANGCHAIN_DIR/orchestrator.py"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [-r|--root REPO_ROOT]"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

RESULTS_DIR="$LANGCHAIN_DIR/benchmark_results"
JOBS_FILE="$SCRIPT_DIR/jobs.txt"
mkdir -p "$RESULTS_DIR"

> "$JOBS_FILE"
for i in $(seq 1 64); do
    echo "python $ORCHESTRATOR --skip-web-search --job-id $i" >> "$JOBS_FILE"
done

cat "$JOBS_FILE" | xargs -P 64 -n 1 -I{} bash -c "{}" > "$RESULTS_DIR/cgam_7a_o1.txt" &
sleep 2.5
cat "$JOBS_FILE" | xargs -P 64 -n 1 -I{} bash -c "{}" > "$RESULTS_DIR/cgam_7a_o2.txt"
wait
