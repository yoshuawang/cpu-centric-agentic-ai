#!/bin/bash
# Run orchestrator with different batch sizes and collect verbose timing statistics (host).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LANGCHAIN_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
ROOT="$(cd "$LANGCHAIN_DIR/.." && pwd)"
ORCHESTRATOR="$LANGCHAIN_DIR/orchestrator.py"
PARSE_STATS="$SCRIPT_DIR/parse_stats.py"

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

OUTPUT_FILE="$LANGCHAIN_DIR/benchmark_results/batch_timing_results_4c.txt"
LOG_DIR="$LANGCHAIN_DIR/benchmark_results/batch_logs_4c"
CSV_FILE="$LANGCHAIN_DIR/benchmark_results/batch_timing_results_4c.csv"

mkdir -p "$LOG_DIR" "$(dirname "$OUTPUT_FILE")"
> "$OUTPUT_FILE"

BATCH_SIZES=(1 2 4 8 16 32 64 128)

run_parallel() {
    local num_processes=$1
    for ((i=1; i<num_processes; i++)); do
        python "$ORCHESTRATOR" --skip-web-search --verbose &
    done
    python "$ORCHESTRATOR" --skip-web-search --verbose
}

for batch_size in "${BATCH_SIZES[@]}"; do
    echo "========================================"
    echo "Running with batch_size=$batch_size"
    echo "========================================"

    LOG_FILE_FULL="$LOG_DIR/batch_${batch_size}_full_log.log"
    LOG_FILE_AVG="$LOG_DIR/batch_${batch_size}_average_log.log"

    echo "Batch Size: $batch_size" >> "$OUTPUT_FILE"
    echo "-------------------" >> "$OUTPUT_FILE"

    run_parallel "$batch_size" 2>&1 > "$LOG_FILE_FULL"
    sleep 4

    python "$PARSE_STATS" --input_file "$LOG_FILE_FULL" --batch-size "$batch_size" > "$LOG_FILE_AVG"
    grep -A 10 "TIMING STATISTICS" "$LOG_FILE_AVG" >> "$OUTPUT_FILE"
    echo "" >> "$OUTPUT_FILE"
    echo "" >> "$OUTPUT_FILE"
done

echo "batch_size,stage,count,avg,min,max" > "$CSV_FILE"

python3 << EOF
import re

results_file = "$OUTPUT_FILE"
csv_file = "$CSV_FILE"

with open(results_file, "r") as f:
    content = f.read()

sections = re.split(r"Batch Size: (\d+)", content)
csv_lines = []

for i in range(1, len(sections), 2):
    batch_size = sections[i]
    section_content = sections[i + 1]
    for line in section_content.split("\n"):
        match = re.match(r"(\w+)\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)", line)
        if match:
            stage, count, avg, min_val, max_val = match.groups()
            csv_lines.append(f"{batch_size},{stage},{count},{avg},{min_val},{max_val}")

with open(csv_file, "a") as f:
    for line in csv_lines:
        f.write(line + "\n")

print(f"CSV file created: {csv_file}")
print(f"Total records: {len(csv_lines)}")
EOF

echo "Results: $OUTPUT_FILE"
echo "CSV: $CSV_FILE"
