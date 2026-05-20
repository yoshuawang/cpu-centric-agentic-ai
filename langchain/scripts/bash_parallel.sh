#!/bin/bash
# Launch many orchestrator processes in parallel (host).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LANGCHAIN_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
ORCHESTRATOR="$LANGCHAIN_DIR/orchestrator.py"

while [[ $# -gt 0 ]]; do
    case $1 in
        -o|--orchestrator)
            ORCHESTRATOR="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [-o|--orchestrator PATH]"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

export ORCHESTRATOR

for _ in $(seq 1 128); do
    python "$ORCHESTRATOR" --skip-web-search &
done
wait
