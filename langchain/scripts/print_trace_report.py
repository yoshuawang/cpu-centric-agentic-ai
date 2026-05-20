#!/usr/bin/env python3
"""
Print timing report tables from orchestrator --trace-output JSON.

Example:
  python print_trace_report.py trace_haystack_8_web.json
  python print_trace_report.py trace_haystack_8_web.json trace_haystack_32_web.json
  python print_trace_report.py trace_haystack_8_web.json --extra llm_ttft llm_prefill llm_decode llm_e2e llm_queue
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable


DEFAULT_STAGES = ["web_search", "fetch_url", "summarize", "llm_inference"]
DEFAULT_EXTRA = ["llm_ttft", "llm_prefill", "llm_decode", "llm_e2e", "llm_queue"]


def _fmt_row(stage: str, count: int, avg: float, mn: float, mx: float) -> str:
    return f"{stage:<20} {count:<10} {avg:<12.4f} {mn:<12.4f} {mx:<12.4f}"


def _stats(values: list[float]) -> tuple[int, float, float, float]:
    c = len(values)
    if c == 0:
        return 0, 0.0, 0.0, 0.0
    s = sum(values)
    return c, s / c, min(values), max(values)


def _collect(series: Iterable[dict], key: str) -> list[float]:
    out: list[float] = []
    for t in series:
        v = t.get(key)
        if isinstance(v, (int, float)):
            out.append(float(v))
    return out


def print_report(path: Path, *, stages: list[str], extra: list[str]) -> None:
    data = json.loads(path.read_text())
    traces = data.get("traces") or []
    if not isinstance(traces, list):
        raise ValueError(f"Bad trace file (traces not a list): {path}")

    bench = data.get("benchmark", "<unknown>")
    batch_size = data.get("batch_size", "<unknown>")

    print()
    print("=" * 70)
    print(f"BENCHMARK: {bench} | batch_size={batch_size} | file={path.name}")
    print("=" * 70)

    print()
    print("TIMING STATISTICS (across all batches)")
    print("=" * 70)
    print(f"{'Stage':<20} {'Count':<10} {'Avg (s)':<12} {'Min (s)':<12} {'Max (s)':<12}")
    print("-" * 70)

    for s in stages:
        vals = _collect(traces, s)
        c, avg, mn, mx = _stats(vals)
        print(_fmt_row(s, c, avg, mn, mx))

    if extra:
        print("-" * 70)
        for s in extra:
            vals = _collect(traces, s)
            if not vals:
                continue
            c, avg, mn, mx = _stats(vals)
            print(_fmt_row(s, c, avg, mn, mx))

    print("=" * 70)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("trace_files", nargs="+", help="Paths to --trace-output JSON files")
    p.add_argument(
        "--stages",
        nargs="*",
        default=DEFAULT_STAGES,
        help=f"Stage keys to summarize (default: {DEFAULT_STAGES})",
    )
    p.add_argument(
        "--extra",
        nargs="*",
        default=[],
        help=(
            "Additional metric keys to summarize if present "
            f"(suggested: {DEFAULT_EXTRA}). If omitted, nothing extra is printed."
        ),
    )
    p.add_argument(
        "--extra-default",
        action="store_true",
        help="Print the default extra llm_* metrics if present.",
    )
    args = p.parse_args()

    extra = list(args.extra)
    if args.extra_default:
        for k in DEFAULT_EXTRA:
            if k not in extra:
                extra.append(k)

    for f in args.trace_files:
        print_report(Path(f), stages=list(args.stages), extra=extra)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

