#!/bin/bash
# Run orchestrator with different batch sizes and collect timing statistics (host, no Docker).

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

OUTPUT_FILE="$LANGCHAIN_DIR/benchmark_results/batch_timing_results_4b.txt"
LOG_DIR="$LANGCHAIN_DIR/benchmark_results/batch_logs_4b"

mkdir -p "$LOG_DIR" "$(dirname "$OUTPUT_FILE")"
> "$OUTPUT_FILE"

BATCH_SIZES=(1 2 4 8 16 32 64 128)

echo "Starting batch size experiments..."
echo "Results will be saved to $OUTPUT_FILE"
echo "Individual logs will be saved to $LOG_DIR/"
echo ""

run_parallel() {
    local num_processes=$1

    if [ -z "$num_processes" ] || [ "$num_processes" -lt 1 ]; then
        echo "Usage: run_parallel <number_of_processes>"
        return 1
    fi

    for ((i=1; i<num_processes; i++)); do
        python "$ORCHESTRATOR" --skip-web-search &
    done
    python "$ORCHESTRATOR" --skip-web-search
}

for batch_size in "${BATCH_SIZES[@]}"; do
    echo "========================================"
    echo "Running with batch_size=$batch_size"
    echo "========================================"

    echo "Batch Size: $batch_size" >> "$OUTPUT_FILE"
    echo "-------------------" >> "$OUTPUT_FILE"

    run_parallel "$batch_size" 2>&1 >> "$OUTPUT_FILE"
    sleep 4
    echo "Completed batch_size=$batch_size"
    echo ""
done

echo "All experiments completed."
echo "Results: $OUTPUT_FILE"
echo "Logs: $LOG_DIR/"
