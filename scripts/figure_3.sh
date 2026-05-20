source /data1/joshw/venv/bin/activate

# Update root directory as per your file structure
export ROOT=/home/jwang354/cpu-centric-agentic-ai

# Update HF_HOME env variable as per your hugging face home location
export HF_HOME=/data1/joshw/hugging_face/hf_home

MODEL=openai/gpt-oss-20b
GPU=0

echo "Starting vLLM server for gpt-oss-20b ..."

CUDA_VISIBLE_DEVICES="$GPU" vllm serve "$MODEL" --no-enable-prefix-caching --port 5000 > vllm.log 2>&1 &
echo $! > vllm.pid



if ! timeout 60 bash -c 'until curl -sf http://localhost:5000/health > /dev/null; do
  sleep 1
done'; then
  echo "vLLM did not become healthy in time. Check logs: vllm.log"
  exit 1
fi

echo "vLLM server started. Running langchain workload ..."

echo "Running Sequential ..."
echo "Running Sequential ..." > "$ROOT/langchain/benchmark_results/latency_figure3.txt"
echo "Batch size 1 ..."
python "$ROOT/langchain/orchestrator.py" --sequential --skip-web-search --batch-size 1 >> "$ROOT/langchain/benchmark_results/latency_figure3.txt"
echo "Batch size 2 ..."
python "$ROOT/langchain/orchestrator.py" --sequential --skip-web-search --batch-size 2 >> "$ROOT/langchain/benchmark_results/latency_figure3.txt"
echo "Batch size 4 ..."
python "$ROOT/langchain/orchestrator.py" --sequential --skip-web-search --batch-size 4 >> "$ROOT/langchain/benchmark_results/latency_figure3.txt" 
echo "Batch size 8 ..."
python "$ROOT/langchain/orchestrator.py" --sequential --skip-web-search --batch-size 8 >> "$ROOT/langchain/benchmark_results/latency_figure3.txt"
echo "Batch size 16 ..."
python "$ROOT/langchain/orchestrator.py" --sequential --skip-web-search --batch-size 16 >> "$ROOT/langchain/benchmark_results/latency_figure3.txt"
echo "Batch size 32 ..."
python "$ROOT/langchain/orchestrator.py" --sequential --skip-web-search --batch-size 32 >> "$ROOT/langchain/benchmark_results/latency_figure3.txt"
echo "Batch size 64 ..."
python "$ROOT/langchain/orchestrator.py" --sequential --skip-web-search --batch-size 64 >> "$ROOT/langchain/benchmark_results/latency_figure3.txt"
echo "Batch size 128 ..."
python "$ROOT/langchain/orchestrator.py" --sequential --skip-web-search --batch-size 128 >> "$ROOT/langchain/benchmark_results/latency_figure3.txt"

echo "Running Multithreading ..."
echo "Running Multithreading ..." >> "$ROOT/langchain/benchmark_results/latency_figure3.txt"
echo "Batch size 1 ..."
python "$ROOT/langchain/orchestrator.py" --skip-web-search --batch-size 1 >> "$ROOT/langchain/benchmark_results/latency_figure3.txt"
echo "Batch size 2 ..."
python "$ROOT/langchain/orchestrator.py" --skip-web-search --batch-size 2 >> "$ROOT/langchain/benchmark_results/latency_figure3.txt"
echo "Batch size 4 ..."
python "$ROOT/langchain/orchestrator.py" --skip-web-search --batch-size 4 >> "$ROOT/langchain/benchmark_results/latency_figure3.txt"
echo "Batch size 8 ..."
python "$ROOT/langchain/orchestrator.py" --skip-web-search --batch-size 8 >> "$ROOT/langchain/benchmark_results/latency_figure3.txt"
echo "Batch size 16 ..."
python "$ROOT/langchain/orchestrator.py" --skip-web-search --batch-size 16 >> "$ROOT/langchain/benchmark_results/latency_figure3.txt"
echo "Batch size 32 ..."
python "$ROOT/langchain/orchestrator.py" --skip-web-search --batch-size 32 >> "$ROOT/langchain/benchmark_results/latency_figure3.txt"
echo "Batch size 64 ..."
python "$ROOT/langchain/orchestrator.py" --skip-web-search --batch-size 64 >> "$ROOT/langchain/benchmark_results/latency_figure3.txt"
echo "Batch size 128 ..."
python "$ROOT/langchain/orchestrator.py" --skip-web-search --batch-size 128 >> "$ROOT/langchain/benchmark_results/latency_figure3.txt"


echo "Running Multiprocessing ..."
echo "Running Multiprocessing ..." >> "$ROOT/langchain/benchmark_results/latency_figure3.txt"
run_parallel() {
    local num_processes=$1
    
    if [ -z "$num_processes" ] || [ "$num_processes" -lt 1 ]; then
        echo "Usage: run_parallel <number_of_processes>"
        echo "Number of processes must be >= 1"
        return 1
    fi
    # Run first (num_processes - 1) in background
    for ((i=1; i<num_processes; i++)); do
        python "$ROOT/langchain/orchestrator.py" --skip-web-search & 
    done
    
    # Run the last one in foreground
    python "$ROOT/langchain/orchestrator.py" --skip-web-search
}

echo "Batch size 1 ..."
run_parallel 1 >> "$ROOT/langchain/benchmark_results/latency_figure3.txt"
echo "Batch size 2 ..."
run_parallel 2 >> "$ROOT/langchain/benchmark_results/latency_figure3.txt"
echo "Batch size 4 ..."
run_parallel 4 >> "$ROOT/langchain/benchmark_results/latency_figure3.txt"
echo "Batch size 8 ..."
run_parallel 8 >> "$ROOT/langchain/benchmark_results/latency_figure3.txt"
echo "Batch size 16 ..."
run_parallel 16 >> "$ROOT/langchain/benchmark_results/latency_figure3.txt"
echo "Batch size 32 ..."
run_parallel 32 >> "$ROOT/langchain/benchmark_results/latency_figure3.txt"
echo "Batch size 64 ..."
run_parallel 64 >> "$ROOT/langchain/benchmark_results/latency_figure3.txt"
echo "Batch size 128 ..."
run_parallel 128 >> "$ROOT/langchain/benchmark_results/latency_figure3.txt"

sleep 5

python "$ROOT/langchain/scripts/plot_multiprocessing.py" --input "$ROOT/langchain/benchmark_results/latency_figure3.txt" --output "$ROOT/figures/figure_3.png"
kill -TERM "$(cat vllm.pid)"

wait "$(cat vllm.pid)" 2>/dev/null || true

rm -f vllm.pid