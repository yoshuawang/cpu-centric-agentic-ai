#!/usr/bin/env bash
# Sample Docker container cgroup stats plus host and GPU stats.

set -euo pipefail

if [ "$#" -gt 0 ]; then
  CONTAINERS=("$@")
else
  CONTAINERS=("vllm-server" "mswa-bench")
fi

OUTPUT_FILE="${OUTPUT_FILE:-stats_log.csv}"
INTERVAL="${INTERVAL:-0.05}"
GPU_INTERVAL="${GPU_INTERVAL:-0.1}"

DOCKER="docker"
if ! docker info &>/dev/null 2>&1; then
  DOCKER="sudo docker"
fi

if ! $DOCKER info >/dev/null 2>&1; then
  echo "Error: Docker is not running or you don't have permissions."
  exit 1
fi

NVIDIA_SMI="$(command -v nvidia-smi || true)"
GPU_STATS_AVAILABLE=0
if [ -n "$NVIDIA_SMI" ] && "$NVIDIA_SMI" --query-gpu=name --format=csv,noheader,nounits >/dev/null 2>&1; then
  GPU_STATS_AVAILABLE=1
fi

mkdir -p "$(dirname "$OUTPUT_FILE")"
echo "Timestamp,Container,CPU_Perc,Host_CPU_Perc,Mem_Usage,Mem_Limit,Mem_Perc,Host_Mem_Usage,Host_Mem_Total,Host_Mem_Perc,Net_Input,Net_Output,Block_Input,Block_Output,GPU_Count,GPU_Util_Max,GPU_Mem_Used,GPU_Mem_Total,GPU_Mem_Perc" > "$OUTPUT_FILE"

echo "[monitor] containers          : ${CONTAINERS[*]}"
echo "[monitor] output              : $OUTPUT_FILE"
echo "[monitor] interval            : ${INTERVAL}s"
echo "[monitor] GPU interval        : ${GPU_INTERVAL}s"
if [ "$GPU_STATS_AVAILABLE" -eq 1 ]; then
  echo "[monitor] GPU stats           : enabled via $NVIDIA_SMI"
elif [ -n "$NVIDIA_SMI" ]; then
  echo "[monitor] GPU stats           : disabled ($NVIDIA_SMI cannot read the NVIDIA driver)"
else
  echo "[monitor] GPU stats           : disabled (nvidia-smi not found)"
fi

fmt_bytes() {
  awk -v b="$1" 'BEGIN {
    if (b == "max" || b == "") { print b; exit }
    b = b + 0
    if (b >= 1073741824) printf "%.4gGiB", b/1073741824
    else if (b >= 1048576) printf "%.4gMiB", b/1048576
    else if (b >= 1024)   printf "%.4gkB",  b/1024
    else                  printf "%dB",     b
  }'
}

read_host_cpu_stat() {
  awk '/^cpu / {
    idle = $5 + $6
    total = 0
    for (i=2; i<=NF; i++) total += $i
    printf "%d %d\n", idle, total
    exit
  }' /proc/stat
}

read_host_mem_stat() {
  awk '
    /^MemTotal:/ { total = $2 * 1024 }
    /^MemAvailable:/ { available = $2 * 1024 }
    END {
      if (total <= 0) total = 0
      if (available < 0) available = 0
      printf "%d %d\n", total - available, total
    }
  ' /proc/meminfo
}

resolve_pid() {
  local name="$1"
  local pid

  pid=$($DOCKER inspect -f '{{.State.Pid}}' "$name" 2>/dev/null || echo 0)
  if [ -z "$pid" ] || [ "$pid" = "0" ]; then
    return 1
  fi
  printf "%s\n" "$pid"
}

resolve_cgroup() {
  local pid="$1"
  local cg_rel cg

  cg_rel=$(awk -F'::' '/^0::/ {print $2; exit}' "/proc/$pid/cgroup" 2>/dev/null || true)
  if [ -z "$cg_rel" ]; then
    return 1
  fi

  cg="/sys/fs/cgroup${cg_rel}"
  if [ ! -r "$cg/cpu.stat" ] || [ ! -r "$cg/memory.current" ]; then
    return 1
  fi

  printf "%s\n" "$cg"
}

read_container_net_stat() {
  local pid="$1"
  awk -F'[: ]+' '
    NR > 2 {
      iface = $2
      if (iface != "lo") {
        rx += $3
        tx += $11
      }
    }
    END { printf "%d %d\n", rx+0, tx+0 }
  ' "/proc/$pid/net/dev" 2>/dev/null || printf "0 0\n"
}

GPU_COUNT=0
GPU_UTIL_MAX="N/A"
GPU_MEM_USED="N/A"
GPU_MEM_TOTAL="N/A"
GPU_MEM_PCT="N/A"

read_gpu_stats() {
  GPU_COUNT=0
  GPU_UTIL_MAX="N/A"
  GPU_MEM_USED="N/A"
  GPU_MEM_TOTAL="N/A"
  GPU_MEM_PCT="N/A"

  if [ "$GPU_STATS_AVAILABLE" -ne 1 ]; then
    return 0
  fi

  local out parsed used_b total_b
  if ! out=$("$NVIDIA_SMI" \
      --query-gpu=utilization.gpu,memory.used,memory.total \
      --format=csv,noheader,nounits 2>/dev/null); then
    return 0
  fi

  if [ -z "$out" ]; then
    return 0
  fi

  if ! parsed=$(awk -F',' '
    function trim(s) {
      gsub(/^[ \t]+|[ \t]+$/, "", s)
      return s
    }
    function num(s) {
      s = trim(s)
      return (s ~ /^[0-9.]+$/) ? s + 0 : 0
    }
    {
      util = num($1)
      used = num($2)
      total = num($3)
      if (total > 0) {
        if (util > util_max) util_max = util
        used_sum += used
        total_sum += total
        count += 1
      }
    }
    END {
      if (count == 0 || total_sum <= 0) exit 1
      printf "%d,%.2f%%,%.0f,%.0f,%.2f%%",
        count, util_max, used_sum * 1048576, total_sum * 1048576,
        used_sum / total_sum * 100
    }
  ' <<< "$out"); then
    return 0
  fi

  IFS=',' read -r GPU_COUNT GPU_UTIL_MAX used_b total_b GPU_MEM_PCT <<< "$parsed"
  GPU_MEM_USED=$(fmt_bytes "$used_b")
  GPU_MEM_TOTAL=$(fmt_bytes "$total_b")
}

declare -A CG_PATH
declare -A CONTAINER_PID
declare -A PREV_USEC
declare -A PREV_NS

HOST_MEM_B=$(( $(awk '/^MemTotal:/ {print $2}' /proc/meminfo) * 1024 ))
read -r PREV_HOST_IDLE PREV_HOST_TOTAL < <(read_host_cpu_stat)

GPU_INTERVAL_NS=$(awk -v s="$GPU_INTERVAL" 'BEGIN { printf "%.0f", s * 1000000000 }')
if [ "$GPU_INTERVAL_NS" -le 0 ]; then
  GPU_INTERVAL_NS=1
fi
LAST_GPU_NS=0
read_gpu_stats

trap "echo -e '\n[monitor] stopped.'; exit" SIGINT SIGTERM

while true; do
  sleep "$INTERVAL"

  NOW_NS=$(date +%s%N)
  TIMESTAMP=$(date +"%Y-%m-%d %H:%M:%S.%3N")
  read -r NOW_HOST_IDLE NOW_HOST_TOTAL < <(read_host_cpu_stat)
  read -r HOST_MEM_USED_B HOST_MEM_TOTAL_B < <(read_host_mem_stat)

  HOST_CPU_PCT=$(awk -v pi="$PREV_HOST_IDLE" -v pt="$PREV_HOST_TOTAL" \
    -v ni="$NOW_HOST_IDLE" -v nt="$NOW_HOST_TOTAL" \
    'BEGIN {
      dt = nt - pt
      di = ni - pi
      if (dt > 0) printf "%.2f%%", (dt - di) / dt * 100
      else printf "0.00%%"
    }')

  HOST_MEM_PCT=$(awk -v u="$HOST_MEM_USED_B" -v l="$HOST_MEM_TOTAL_B" \
    'BEGIN { if (l+0 > 0) printf "%.2f%%", u/l*100; else printf "0.00%%" }')
  HOST_MEM_USAGE=$(fmt_bytes "$HOST_MEM_USED_B")
  HOST_MEM_TOTAL=$(fmt_bytes "$HOST_MEM_TOTAL_B")

  if [ "$GPU_STATS_AVAILABLE" -eq 1 ] && [ $((NOW_NS - LAST_GPU_NS)) -ge "$GPU_INTERVAL_NS" ]; then
    read_gpu_stats
    LAST_GPU_NS=$NOW_NS
  fi

  for name in "${CONTAINERS[@]}"; do
    pid="${CONTAINER_PID[$name]:-}"
    if [ -z "$pid" ] || [ ! -r "/proc/$pid/net/dev" ]; then
      if ! pid=$(resolve_pid "$name"); then
        unset "CONTAINER_PID[$name]" "CG_PATH[$name]" "PREV_USEC[$name]" "PREV_NS[$name]"
        continue
      fi
      CONTAINER_PID[$name]="$pid"
    fi

    cg="${CG_PATH[$name]:-}"
    if [ -z "$cg" ] || [ ! -r "$cg/cpu.stat" ] || [ ! -r "$cg/memory.current" ]; then
      if ! cg=$(resolve_cgroup "$pid"); then
        unset "CONTAINER_PID[$name]" "CG_PATH[$name]" "PREV_USEC[$name]" "PREV_NS[$name]"
        continue
      fi
      CG_PATH[$name]="$cg"
    fi

    NOW_USEC=$(awk '$1=="usage_usec" {print $2; exit}' "$cg/cpu.stat" 2>/dev/null || echo 0)
    if [ -z "${PREV_USEC[$name]:-}" ] || [ -z "${PREV_NS[$name]:-}" ]; then
      CPU_PCT="0.00%"
    else
      CPU_PCT=$(awk -v p="${PREV_USEC[$name]}" -v n="$NOW_USEC" -v pt="${PREV_NS[$name]}" -v nt="$NOW_NS" \
        'BEGIN {
          dt_us = (nt - pt) / 1000.0
          if (dt_us > 0) printf "%.2f%%", (n - p) / dt_us * 100
          else printf "0.00%%"
        }')
    fi

    read -r MEM_B < "$cg/memory.current" 2>/dev/null || MEM_B=0
    read -r MEM_LIMIT_RAW < "$cg/memory.max" 2>/dev/null || MEM_LIMIT_RAW=max
    if [ "$MEM_LIMIT_RAW" = "max" ]; then
      MEM_LIMIT_B=$HOST_MEM_B
    else
      MEM_LIMIT_B=$MEM_LIMIT_RAW
    fi

    MEM_PCT=$(awk -v u="$MEM_B" -v l="$MEM_LIMIT_B" \
      'BEGIN { if (l+0 > 0) printf "%.2f%%", u/l*100; else printf "0.00%%" }')

    read -r BLOCK_IN_B BLOCK_OUT_B < <(awk '
      {
        for (i=2; i<=NF; i++) {
          split($i, kv, "=")
          if (kv[1] == "rbytes") r += kv[2]
          if (kv[1] == "wbytes") w += kv[2]
        }
      }
      END { printf "%d %d", r+0, w+0 }
    ' "$cg/io.stat" 2>/dev/null) || { BLOCK_IN_B=0; BLOCK_OUT_B=0; }

    MEM_USAGE=$(fmt_bytes "$MEM_B")
    MEM_LIMIT=$(fmt_bytes "$MEM_LIMIT_B")
    read -r NET_IN_B NET_OUT_B < <(read_container_net_stat "$pid")
    NET_IN=$(fmt_bytes "$NET_IN_B")
    NET_OUT=$(fmt_bytes "$NET_OUT_B")
    BLOCK_IN=$(fmt_bytes "$BLOCK_IN_B")
    BLOCK_OUT=$(fmt_bytes "$BLOCK_OUT_B")

    echo "$TIMESTAMP,$name,$CPU_PCT,$HOST_CPU_PCT,$MEM_USAGE,$MEM_LIMIT,$MEM_PCT,$HOST_MEM_USAGE,$HOST_MEM_TOTAL,$HOST_MEM_PCT,$NET_IN,$NET_OUT,$BLOCK_IN,$BLOCK_OUT,$GPU_COUNT,$GPU_UTIL_MAX,$GPU_MEM_USED,$GPU_MEM_TOTAL,$GPU_MEM_PCT" >> "$OUTPUT_FILE"

    PREV_USEC[$name]=$NOW_USEC
    PREV_NS[$name]=$NOW_NS
  done

  PREV_HOST_IDLE=$NOW_HOST_IDLE
  PREV_HOST_TOTAL=$NOW_HOST_TOTAL
done
