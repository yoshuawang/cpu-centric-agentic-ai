#!/usr/bin/env python3
"""Plot resource time series from scripts/run_benchmark_docker.sh stats_log.csv."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import warnings
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / ".matplotlib-cache"))
warnings.filterwarnings("ignore", message="Unable to import Axes3D.*")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


PERCENT_RE = re.compile(r"^\s*([+-]?\d+(?:\.\d+)?)%?\s*$")
BYTE_RE = re.compile(r"^\s*([+-]?\d+(?:\.\d+)?)([A-Za-z]+)?\s*$")
STAGE_COLORS = {
    "vLLM": "#f97316",
    "Code": "#16a34a",
    "Startup": "#64748b",
}


def parse_percent(value: object) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.upper() == "N/A":
        return None
    match = PERCENT_RE.match(text)
    return float(match.group(1)) if match else None


def parse_bytes_to_gib(value: object) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.upper() == "N/A" or text == "max":
        return None
    match = BYTE_RE.match(text)
    if not match:
        return None

    amount = float(match.group(1))
    unit = (match.group(2) or "B").lower()
    multipliers = {
        "b": 1,
        "kb": 1024,
        "kib": 1024,
        "mb": 1024**2,
        "mib": 1024**2,
        "gb": 1024**3,
        "gib": 1024**3,
        "tb": 1024**4,
        "tib": 1024**4,
    }
    multiplier = multipliers.get(unit)
    if multiplier is None:
        return None
    return amount * multiplier / 1024**3


def parse_bytes_to_mib(value: object) -> float | None:
    gib = parse_bytes_to_gib(value)
    if gib is None:
        return None
    return gib * 1024


def save_line_plot(
    elapsed_s: pd.Series,
    values: pd.Series,
    title: str,
    ylabel: str,
    output_path: Path,
    color: str,
    stage_events: list[dict[str, object]] | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(11, 5.5), constrained_layout=True)
    ax.plot(elapsed_s, values, linewidth=1.8, color=color)
    ax.set_title(title)
    ax.set_xlabel("Elapsed time (s)")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)
    set_zero_based_ylim(ax, values)
    add_stage_annotations(ax, stage_events or [])
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def set_zero_based_ylim(ax: plt.Axes, values: pd.Series) -> None:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if clean.empty:
        ax.set_ylim(-0.1, 1.0)
        ax.axhline(0, color="#111827", linewidth=0.8, alpha=0.35)
        return

    ymax = clean.max()
    if ymax <= 0:
        ax.set_ylim(-0.1, 1.0)
        ax.axhline(0, color="#111827", linewidth=0.8, alpha=0.35)
        return

    ax.set_ylim(-(ymax * 0.025), ymax * 1.08)
    ax.axhline(0, color="#111827", linewidth=0.8, alpha=0.35)


def load_stage_events(events_path: Path | None, trace_start: pd.Timestamp) -> list[dict[str, object]]:
    if events_path is None or not events_path.exists():
        return []

    with events_path.open() as f:
        data = json.load(f)

    if isinstance(data, list):
        records = data
    else:
        records = [data]

    events: list[dict[str, object]] = []

    def epoch_to_elapsed(value: object) -> float | None:
        if not isinstance(value, (int, float)):
            return None
        event_time = pd.Timestamp(dt.datetime.fromtimestamp(float(value)))
        return (event_time - trace_start).total_seconds()

    for record in records:
        if not isinstance(record, dict):
            continue
        detailed_logs = record.get("detailed_logs") or {}
        if not isinstance(detailed_logs, dict):
            continue

        for idx, call in enumerate(detailed_logs.get("model_api_calls") or [], start=1):
            start = epoch_to_elapsed(call.get("timestamp"))
            duration = call.get("duration_total_seconds") or 0.0
            if start is None:
                continue
            end = start + float(duration)
            events.append({"start": start, "end": end, "stage": "vLLM", "label": f"vLLM {idx}"})

        for idx, execution in enumerate(detailed_logs.get("bash_executions") or [], start=1):
            start = epoch_to_elapsed(execution.get("timestamp_start"))
            end = epoch_to_elapsed(execution.get("timestamp_end"))
            if start is None or end is None:
                continue
            events.append({"start": start, "end": end, "stage": "Code", "label": f"Code {idx}"})

    events.sort(key=lambda event: (float(event["start"]), float(event["end"])))
    if events and float(events[0]["start"]) > 1.0:
        events.insert(
            0,
            {
                "start": 0.0,
                "end": float(events[0]["start"]),
                "stage": "Startup",
                "label": "vLLM startup",
            },
        )
    return events


def add_stage_annotations(ax: plt.Axes, stage_events: list[dict[str, object]]) -> None:
    if not stage_events:
        return

    _, ymax = ax.get_ylim()
    label_y = ymax * 0.94
    for event in stage_events:
        start = max(0.0, float(event["start"]))
        end = max(start, float(event["end"]))
        if end <= 0:
            continue
        stage = str(event["stage"])
        label = str(event["label"])
        color = STAGE_COLORS.get(stage, "#64748b")
        ax.axvspan(start, end, color=color, alpha=0.10, linewidth=0)

        width = end - start
        x = start + width / 2 if width >= 0.5 else start
        rotation = 0 if width >= 2.0 else 90
        ha = "center" if width >= 0.5 else "left"
        ax.text(
            x,
            label_y,
            label,
            color=color,
            fontsize=8,
            rotation=rotation,
            ha=ha,
            va="top",
            clip_on=True,
        )


def save_container_plot(
    df: pd.DataFrame,
    value_column: str,
    title: str,
    ylabel: str,
    output_path: Path,
    stage_events: list[dict[str, object]],
) -> None:
    fig, ax = plt.subplots(figsize=(11, 5.5), constrained_layout=True)
    for container, data in df.groupby("Container", sort=False):
        ax.plot(data["Elapsed_Time"], data[value_column], linewidth=1.8, label=container)
    ax.set_title(title)
    ax.set_xlabel("Elapsed time (s)")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)
    set_zero_based_ylim(ax, df[value_column])
    add_stage_annotations(ax, stage_events)
    ax.legend()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", nargs="?", default="stats_log.csv")
    parser.add_argument(
        "--outdir",
        default="benchmark_results/resource_trace_plots",
        help="Directory to write PNG plots into.",
    )
    parser.add_argument(
        "--events",
        default=None,
        help="Benchmark JSON containing detailed_logs.model_api_calls and bash_executions.",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    events_path = Path(args.events) if args.events else csv_path.parent / "benchmark_results" / "sorting_benchmark.json"

    df = pd.read_csv(csv_path)
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    start_time = df["Timestamp"].min()
    stage_events = load_stage_events(events_path, start_time)
    df["Elapsed_Time"] = (df["Timestamp"] - start_time).dt.total_seconds()
    df["CPU_Perc_Val"] = df["CPU_Perc"].map(parse_percent)
    df["Mem_Usage_GiB"] = df["Mem_Usage"].map(parse_bytes_to_gib)
    df["Net_Input_MiB"] = df["Net_Input"].map(parse_bytes_to_mib)
    df["Net_Output_MiB"] = df["Net_Output"].map(parse_bytes_to_mib)
    df["Block_Input_MiB"] = df["Block_Input"].map(parse_bytes_to_mib)
    df["Block_Output_MiB"] = df["Block_Output"].map(parse_bytes_to_mib)
    df["GPU_Util_Perc"] = df["GPU_Util_Max"].map(parse_percent)
    df["GPU_Mem_Used_GiB"] = df["GPU_Mem_Used"].map(parse_bytes_to_gib)

    host_columns = {}
    if "Host_CPU_Perc" in df.columns:
        df["Host_CPU_Perc_Val"] = df["Host_CPU_Perc"].map(parse_percent)
        host_columns["Host_CPU_Perc_Val"] = "mean"
    if "Host_Mem_Usage" in df.columns:
        df["Host_Mem_Usage_GiB"] = df["Host_Mem_Usage"].map(parse_bytes_to_gib)
        host_columns["Host_Mem_Usage_GiB"] = "mean"

    machine_trace = df.groupby("Elapsed_Time", as_index=False).agg(
        {"GPU_Util_Perc": "max", "GPU_Mem_Used_GiB": "max", **host_columns}
    )

    save_container_plot(
        df,
        "CPU_Perc_Val",
        "Container CPU Utilization vs Time",
        "CPU utilization (%)",
        outdir / "cpu_util_vs_time.png",
        stage_events,
    )
    save_container_plot(
        df,
        "Mem_Usage_GiB",
        "Container Memory Usage vs Time",
        "Memory used (GiB)",
        outdir / "container_memory_vs_time.png",
        stage_events,
    )
    fig, ax = plt.subplots(figsize=(11, 5.5), constrained_layout=True)
    for container, data in df.groupby("Container", sort=False):
        ax.plot(data["Elapsed_Time"], data["Net_Input_MiB"], linewidth=1.8, label=f"{container} input")
        ax.plot(data["Elapsed_Time"], data["Net_Output_MiB"], linewidth=1.8, linestyle="--", label=f"{container} output")
    ax.set_title("Network I/O vs Time")
    ax.set_xlabel("Elapsed time (s)")
    ax.set_ylabel("Network I/O (MiB)")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)
    set_zero_based_ylim(ax, pd.concat([df["Net_Input_MiB"], df["Net_Output_MiB"]]))
    add_stage_annotations(ax, stage_events)
    ax.legend()
    fig.savefig(outdir / "network_io_vs_time.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 5.5), constrained_layout=True)
    for container, data in df.groupby("Container", sort=False):
        ax.plot(data["Elapsed_Time"], data["Block_Input_MiB"], linewidth=1.8, label=f"{container} read")
        ax.plot(data["Elapsed_Time"], data["Block_Output_MiB"], linewidth=1.8, linestyle="--", label=f"{container} write")
    ax.set_title("Block I/O vs Time")
    ax.set_xlabel("Elapsed time (s)")
    ax.set_ylabel("Block I/O (MiB)")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)
    set_zero_based_ylim(ax, pd.concat([df["Block_Input_MiB"], df["Block_Output_MiB"]]))
    add_stage_annotations(ax, stage_events)
    ax.legend()
    fig.savefig(outdir / "block_io_vs_time.png", dpi=180)
    plt.close(fig)

    if "Host_CPU_Perc_Val" in machine_trace.columns:
        save_line_plot(
            machine_trace["Elapsed_Time"],
            machine_trace["Host_CPU_Perc_Val"],
            "Host CPU Utilization vs Time",
            "Host CPU utilization (%)",
            outdir / "host_cpu_vs_time.png",
            "#2563eb",
            stage_events,
        )
    if "Host_Mem_Usage_GiB" in machine_trace.columns:
        save_line_plot(
            machine_trace["Elapsed_Time"],
            machine_trace["Host_Mem_Usage_GiB"],
            "System Memory Usage vs Time",
            "Host memory used (GiB)",
            outdir / "system_memory_vs_time.png",
            "#059669",
            stage_events,
        )
    save_line_plot(
        machine_trace["Elapsed_Time"],
        machine_trace["GPU_Util_Perc"],
        "GPU Utilization vs Time",
        "GPU utilization (%)",
        outdir / "gpu_util_vs_time.png",
        "#dc2626",
        stage_events,
    )
    save_line_plot(
        machine_trace["Elapsed_Time"],
        machine_trace["GPU_Mem_Used_GiB"],
        "VRAM Usage vs Time",
        "VRAM used (GiB)",
        outdir / "vram_vs_time.png",
        "#7c3aed",
        stage_events,
    )

    fig, axes = plt.subplots(4, 1, figsize=(12, 13), sharex=True, constrained_layout=True)
    for container, data in df.groupby("Container", sort=False):
        axes[0].plot(data["Elapsed_Time"], data["CPU_Perc_Val"], linewidth=1.6, label=container)
        axes[1].plot(data["Elapsed_Time"], data["Mem_Usage_GiB"], linewidth=1.6, label=container)
    axes[0].set_ylabel("CPU (%)")
    axes[0].set_title("Container CPU Utilization vs Time")
    axes[0].legend()
    axes[1].set_ylabel("Memory (GiB)")
    axes[1].set_title("Container Memory Usage vs Time")
    axes[1].legend()
    axes[2].plot(machine_trace["Elapsed_Time"], machine_trace["GPU_Util_Perc"], linewidth=1.6, color="#dc2626")
    axes[2].set_ylabel("GPU (%)")
    axes[2].set_title("GPU Utilization vs Time")
    axes[3].plot(machine_trace["Elapsed_Time"], machine_trace["GPU_Mem_Used_GiB"], linewidth=1.6, color="#7c3aed")
    axes[3].set_ylabel("VRAM (GiB)")
    axes[3].set_title("VRAM Usage vs Time")
    axes[3].set_xlabel("Elapsed time (s)")
    for ax in axes:
        ax.grid(True, alpha=0.3)
        ax.set_xlim(left=0)
    set_zero_based_ylim(axes[0], df["CPU_Perc_Val"])
    set_zero_based_ylim(axes[1], df["Mem_Usage_GiB"])
    set_zero_based_ylim(axes[2], machine_trace["GPU_Util_Perc"])
    set_zero_based_ylim(axes[3], machine_trace["GPU_Mem_Used_GiB"])
    for ax in axes:
        add_stage_annotations(ax, stage_events)
    fig.savefig(outdir / "resource_trace_overview.png", dpi=180)
    plt.close(fig)

    print(f"Wrote plots to {outdir}")
    print(f"Rows: {len(df)}")
    print(f"Duration: {df['Elapsed_Time'].max():.2f}s")
    print(f"Containers: {', '.join(df['Container'].dropna().unique())}")
    print(f"Stage events: {len(stage_events)} from {events_path if events_path.exists() else 'none'}")


if __name__ == "__main__":
    main()
