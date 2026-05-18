"""Background system resource sampler used by benchmark_latency.

Samples CPU, memory, GPU utilization, VRAM, disk, and network counters at a
fixed cadence on a daemon thread. `summarize(start_t, end_t)` returns the
peaks and cumulative deltas observed within an arbitrary time window so the
benchmark can derive per-call and per-stage resource breakdowns from the
existing call/bash timestamp logs.
"""

from __future__ import annotations

import bisect
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass
from typing import Any

try:
    import psutil
except ImportError:
    psutil = None

import warnings as _warnings

with _warnings.catch_warnings():
    _warnings.simplefilter("ignore")
    try:
        import pynvml

        try:
            pynvml.nvmlInit()
            _PYNVML_HANDLES = [
                pynvml.nvmlDeviceGetHandleByIndex(i)
                for i in range(pynvml.nvmlDeviceGetCount())
            ]
            _PYNVML_OK = bool(_PYNVML_HANDLES)
        except Exception:
            _PYNVML_HANDLES = []
            _PYNVML_OK = False
    except Exception:
        pynvml = None
        _PYNVML_HANDLES = []
        _PYNVML_OK = False


_NVIDIA_SMI = shutil.which("nvidia-smi") if not _PYNVML_OK else None


@dataclass
class _Sample:
    t: float
    cpu_pct: float | None
    mem_used_b: int | None
    mem_pct: float | None
    gpu_util_pct: float | None
    vram_used_b: int | None
    disk_read_b: int | None
    disk_write_b: int | None
    net_sent_b: int | None
    net_recv_b: int | None


def _read_gpu() -> tuple[float | None, int | None]:
    """Return (gpu_util_percent, vram_used_bytes), max across visible GPUs."""
    if _PYNVML_OK:
        try:
            util_max = 0.0
            vram_max = 0
            for h in _PYNVML_HANDLES:
                util = pynvml.nvmlDeviceGetUtilizationRates(h)
                mem = pynvml.nvmlDeviceGetMemoryInfo(h)
                util_max = max(util_max, float(util.gpu))
                vram_max = max(vram_max, int(mem.used))
            return util_max, vram_max
        except Exception:
            return None, None
    if _NVIDIA_SMI:
        try:
            out = subprocess.check_output(
                [
                    _NVIDIA_SMI,
                    "--query-gpu=utilization.gpu,memory.used",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
                timeout=2,
            )
            util_max = 0.0
            vram_max_mib = 0
            for line in out.strip().splitlines():
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 2:
                    util_max = max(util_max, float(parts[0]))
                    vram_max_mib = max(vram_max_mib, int(parts[1]))
            return util_max, vram_max_mib * 1024 * 1024
        except Exception:
            return None, None
    return None, None


class ResourceMonitor:
    """Daemon-thread sampler. Start at construction time; query via summarize()."""

    def __init__(self, interval_s: float = 0.2) -> None:
        self.interval_s = float(interval_s)
        self._samples: list[_Sample] = []
        self._lock = threading.Lock()
        self._stop_evt = threading.Event()
        self._thread: threading.Thread | None = None
        self._started = False

        if psutil is not None:
            try:
                psutil.cpu_percent(None)
            except Exception:
                pass

    def start(self) -> None:
        if self._started:
            return
        self._started = True
        self._thread = threading.Thread(
            target=self._run, name="resource-monitor", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop_evt.set()
        if self._thread is not None:
            self._thread.join(timeout=2 * self.interval_s + 1.0)

    def _run(self) -> None:
        next_t = time.time()
        while not self._stop_evt.is_set():
            sample = self._take_sample()
            with self._lock:
                self._samples.append(sample)
            next_t += self.interval_s
            sleep_for = next_t - time.time()
            if sleep_for > 0:
                self._stop_evt.wait(sleep_for)
            else:
                next_t = time.time()

    def _take_sample(self) -> _Sample:
        now = time.time()
        cpu_pct = mem_used = mem_pct = None
        disk_r = disk_w = net_s = net_r = None
        if psutil is not None:
            try:
                cpu_pct = float(psutil.cpu_percent(None))
            except Exception:
                cpu_pct = None
            try:
                vm = psutil.virtual_memory()
                mem_used = int(vm.used)
                mem_pct = float(vm.percent)
            except Exception:
                pass
            try:
                d = psutil.disk_io_counters()
                if d is not None:
                    disk_r = int(d.read_bytes)
                    disk_w = int(d.write_bytes)
            except Exception:
                pass
            try:
                n = psutil.net_io_counters()
                if n is not None:
                    net_s = int(n.bytes_sent)
                    net_r = int(n.bytes_recv)
            except Exception:
                pass
        gpu_util, vram_used = _read_gpu()
        return _Sample(
            t=now,
            cpu_pct=cpu_pct,
            mem_used_b=mem_used,
            mem_pct=mem_pct,
            gpu_util_pct=gpu_util,
            vram_used_b=vram_used,
            disk_read_b=disk_r,
            disk_write_b=disk_w,
            net_sent_b=net_s,
            net_recv_b=net_r,
        )

    def _samples_in_window(self, start_t: float, end_t: float) -> list[_Sample]:
        with self._lock:
            ts = [s.t for s in self._samples]
            lo = bisect.bisect_left(ts, start_t)
            hi = bisect.bisect_right(ts, end_t)
            window = list(self._samples[lo:hi])
            if not window and self._samples:
                idx = min(max(bisect.bisect_left(ts, end_t) - 1, 0), len(self._samples) - 1)
                window = [self._samples[idx]]
        return window

    def summarize(self, start_t: float, end_t: float) -> dict[str, Any]:
        """Return peaks and totals for the [start_t, end_t] window."""
        if end_t < start_t:
            start_t, end_t = end_t, start_t
        window = self._samples_in_window(start_t, end_t)
        return _summarize_samples(window, end_t - start_t, self.interval_s)


def _max_or_none(values: list[float | int | None]) -> float | int | None:
    vals = [v for v in values if v is not None]
    return max(vals) if vals else None


def _peak_rate_mbps(values: list[int | None], interval_s: float) -> float | None:
    """Peak per-sample rate in MB/s from cumulative byte counters."""
    cleaned = [v for v in values if v is not None]
    if len(cleaned) < 2 or interval_s <= 0:
        return None
    peak = 0.0
    for prev, cur in zip(cleaned, cleaned[1:]):
        delta = max(0, cur - prev)
        rate = delta / interval_s
        if rate > peak:
            peak = rate
    return peak / (1024 * 1024)


def _cumulative_bytes(values: list[int | None]) -> int | None:
    cleaned = [v for v in values if v is not None]
    if len(cleaned) < 2:
        return 0 if cleaned else None
    return max(0, cleaned[-1] - cleaned[0])


def _summarize_samples(
    window: list[_Sample], duration_s: float, interval_s: float
) -> dict[str, Any]:
    if not window:
        return {
            "sample_count": 0,
            "interval_s": interval_s,
            "duration_seconds": duration_s,
            "cpu_max_percent": None,
            "memory_max_bytes": None,
            "memory_max_percent": None,
            "gpu_max_percent": None,
            "vram_max_bytes": None,
            "disk_read_bytes_total": None,
            "disk_write_bytes_total": None,
            "disk_read_peak_mbps": None,
            "disk_write_peak_mbps": None,
            "net_sent_bytes_total": None,
            "net_recv_bytes_total": None,
            "net_sent_peak_mbps": None,
            "net_recv_peak_mbps": None,
        }
    return {
        "sample_count": len(window),
        "interval_s": interval_s,
        "duration_seconds": duration_s,
        "cpu_max_percent": _max_or_none([s.cpu_pct for s in window]),
        "memory_max_bytes": _max_or_none([s.mem_used_b for s in window]),
        "memory_max_percent": _max_or_none([s.mem_pct for s in window]),
        "gpu_max_percent": _max_or_none([s.gpu_util_pct for s in window]),
        "vram_max_bytes": _max_or_none([s.vram_used_b for s in window]),
        "disk_read_bytes_total": _cumulative_bytes([s.disk_read_b for s in window]),
        "disk_write_bytes_total": _cumulative_bytes([s.disk_write_b for s in window]),
        "disk_read_peak_mbps": _peak_rate_mbps([s.disk_read_b for s in window], interval_s),
        "disk_write_peak_mbps": _peak_rate_mbps([s.disk_write_b for s in window], interval_s),
        "net_sent_bytes_total": _cumulative_bytes([s.net_sent_b for s in window]),
        "net_recv_bytes_total": _cumulative_bytes([s.net_recv_b for s in window]),
        "net_sent_peak_mbps": _peak_rate_mbps([s.net_sent_b for s in window], interval_s),
        "net_recv_peak_mbps": _peak_rate_mbps([s.net_recv_b for s in window], interval_s),
    }


_RESOURCE_METRIC_FIELDS_MAX = (
    "cpu_max_percent",
    "memory_max_bytes",
    "memory_max_percent",
    "gpu_max_percent",
    "vram_max_bytes",
    "disk_read_peak_mbps",
    "disk_write_peak_mbps",
    "net_sent_peak_mbps",
    "net_recv_peak_mbps",
)
_RESOURCE_METRIC_FIELDS_SUM = (
    "disk_read_bytes_total",
    "disk_write_bytes_total",
    "net_sent_bytes_total",
    "net_recv_bytes_total",
)


def empty_resource_metrics() -> dict[str, Any]:
    """Return a zero/None metrics dict for stages that had no windows."""
    metrics: dict[str, Any] = {field: None for field in _RESOURCE_METRIC_FIELDS_MAX}
    for field in _RESOURCE_METRIC_FIELDS_SUM:
        metrics[field] = 0
    metrics["sample_count"] = 0
    metrics["window_count"] = 0
    return metrics


def aggregate_resource_metrics(
    metrics_list: list[dict[str, Any] | None],
) -> dict[str, Any]:
    """Roll up multiple per-window resource_metrics dicts.

    Maxes are taken across windows; cumulative byte counters are summed.
    """
    cleaned = [m for m in metrics_list if m]
    if not cleaned:
        return empty_resource_metrics()

    out: dict[str, Any] = {}
    for field in _RESOURCE_METRIC_FIELDS_MAX:
        values = [m.get(field) for m in cleaned if m.get(field) is not None]
        out[field] = max(values) if values else None
    for field in _RESOURCE_METRIC_FIELDS_SUM:
        values = [m.get(field) for m in cleaned if m.get(field) is not None]
        out[field] = sum(values) if values else 0
    out["sample_count"] = sum(int(m.get("sample_count", 0) or 0) for m in cleaned)
    out["window_count"] = len(cleaned)
    return out
