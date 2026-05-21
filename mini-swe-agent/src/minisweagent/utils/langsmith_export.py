"""Export LangSmith trace spans to local JSON and CSV under benchmark_results/."""

from __future__ import annotations

import csv
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _latency_seconds(run: Any) -> float | None:
    lat = getattr(run, "latency", None)
    if lat is None:
        return None
    lat = float(lat)
    # LangSmith may report seconds (typical for agent runs) or milliseconds.
    if lat > 10_000:
        return lat / 1000.0
    return lat


def _run_to_row(run: Any, *, trace_id: str, root_run_id: str) -> dict[str, Any]:
    start = getattr(run, "start_time", None)
    end = getattr(run, "end_time", None)
    return {
        "trace_id": trace_id,
        "root_run_id": root_run_id,
        "run_id": str(getattr(run, "id", "")),
        "parent_run_id": str(getattr(run, "parent_run_id", "") or ""),
        "name": getattr(run, "name", "") or "",
        "run_type": getattr(run, "run_type", "") or "",
        "status": getattr(run, "status", "") or "",
        "start_time": start.isoformat() if start else "",
        "end_time": end.isoformat() if end else "",
        "latency_seconds": _latency_seconds(run),
    }


def fetch_langsmith_trace_rows(
    *,
    project_name: str,
    run_started_at: datetime | None = None,
    limit_roots: int = 5,
) -> list[dict[str, Any]]:
    """Pull recent root traces and flatten all spans in each trace."""
    from langsmith import Client

    client = Client()
    kwargs: dict[str, Any] = {
        "project_name": project_name,
        "is_root": True,
        "limit": limit_roots,
    }
    if run_started_at is not None:
        kwargs["start_time"] = run_started_at

    rows: list[dict[str, Any]] = []
    for root in client.list_runs(**kwargs):
        trace_id = str(getattr(root, "trace_id", None) or root.id)
        root_run_id = str(root.id)
        rows.append(_run_to_row(root, trace_id=trace_id, root_run_id=root_run_id))
        for child in client.list_runs(project_name=project_name, trace_id=trace_id, limit=100):
            if str(child.id) == root_run_id:
                continue
            rows.append(_run_to_row(child, trace_id=trace_id, root_run_id=root_run_id))
    return rows


def _write_plot_csv(rows: list[dict[str, Any]], out_dir: Path, benchmark_type: str) -> Path:
    """
    Write a second CSV under outputs/ using the column names expected by
    plot_resources_cropped.py: 'Start Time', 'End Time', 'Run Type'.
    """
    outputs_dir = out_dir.parent / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    plot_path = outputs_dir / f"{benchmark_type}_langsmith_stats.csv"

    plot_fieldnames = ["trace_id", "root_run_id", "run_id", "parent_run_id",
                       "name", "Run Type", "status", "Start Time", "End Time",
                       "latency_seconds"]
    with plot_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=plot_fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "trace_id": row["trace_id"],
                "root_run_id": row["root_run_id"],
                "run_id": row["run_id"],
                "parent_run_id": row["parent_run_id"],
                "name": row["name"],
                "Run Type": row["run_type"],
                "status": row["status"],
                "Start Time": row["start_time"],
                "End Time": row["end_time"],
                "latency_seconds": row["latency_seconds"],
            })
    return plot_path


def export_langsmith_traces(
    output_dir: str | Path,
    benchmark_type: str,
    run_started_at: datetime | None = None,
    *,
    limit_roots: int = 5,
) -> dict[str, str | int] | None:
    """
    Write LangSmith spans for recent runs to two locations:

    - benchmark_results/{benchmark}_langsmith_trace_{ts}.{json,csv}  (full detail)
    - outputs/{benchmark}_langsmith_stats.csv  (column names expected by
      plot_resources_cropped.py: 'Start Time', 'End Time', 'Run Type')

    Returns paths written, or None if tracing is disabled / langsmith unavailable.
    """
    if not os.environ.get("LANGCHAIN_API_KEY"):
        return None

    try:
        from langsmith import Client  # noqa: F401
    except ImportError:
        print("[langsmith] langsmith package not installed; skipping local trace export.")
        return None

    project_name = os.environ.get("LANGCHAIN_PROJECT", "mini-swe-agent")
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    try:
        rows = fetch_langsmith_trace_rows(
            project_name=project_name,
            run_started_at=run_started_at,
            limit_roots=limit_roots,
        )
    except Exception as exc:
        print(f"[langsmith] Failed to fetch traces from project '{project_name}': {exc}")
        return None

    if not rows:
        print(f"[langsmith] No traces found in project '{project_name}' since benchmark start.")
        return None

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    base = out / f"{benchmark_type}_langsmith_trace_{stamp}"
    json_path = base.with_suffix(".json")
    csv_path = base.with_suffix(".csv")

    payload = {
        "project": project_name,
        "benchmark_type": benchmark_type,
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "run_started_at": run_started_at.isoformat() if run_started_at else None,
        "span_count": len(rows),
        "spans": rows,
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    fieldnames = list(rows[0].keys())
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # Also write the plot-compatible CSV to outputs/
    plot_path = _write_plot_csv(rows, out, benchmark_type)

    print(f"[langsmith] Trace JSON saved to:  {json_path}")
    print(f"[langsmith] Trace CSV saved to:   {csv_path}")
    print(f"[langsmith] Plot CSV saved to:    {plot_path}")
    return {
        "json_path": str(json_path),
        "csv_path": str(csv_path),
        "plot_csv_path": str(plot_path),
        "span_count": len(rows),
        "project": project_name,
    }
