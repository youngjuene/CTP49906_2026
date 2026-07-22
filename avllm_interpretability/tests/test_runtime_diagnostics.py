"""CPU-only contract tests for bounded host/CUDA memory instrumentation."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.runtime_diagnostics import (  # noqa: E402
    RuntimeMemoryMonitor,
    sample_cuda_memory,
)


class FakeCuda:
    def __init__(self, *, available: bool = True) -> None:
        self.available = available
        self.reset_calls: list[object] = []

    def is_available(self) -> bool:
        return self.available

    def reset_peak_memory_stats(self, device=None) -> None:
        self.reset_calls.append(device)

    def memory_allocated(self, device=None) -> int:
        return 100

    def memory_reserved(self, device=None) -> int:
        return 200

    def max_memory_allocated(self, device=None) -> int:
        return 300

    def max_memory_reserved(self, device=None) -> int:
        return 400


def test_cuda_snapshot_is_explicit_when_cuda_is_unavailable() -> None:
    snapshot = sample_cuda_memory(cuda_backend=FakeCuda(available=False))

    assert snapshot.available is False
    assert snapshot.allocated_bytes is None
    assert snapshot.reserved_bytes is None
    assert snapshot.peak_allocated_bytes is None
    assert snapshot.peak_reserved_bytes is None
    assert snapshot.unavailable_reason == "cuda_not_available"


def test_monitor_resets_cuda_peaks_and_records_allocated_and_reserved() -> None:
    cuda = FakeCuda()
    rss_values = iter((1_000, 1_500, 1_250))
    monitor = RuntimeMemoryMonitor(
        cuda_backend=cuda,
        rss_reader=lambda: next(rss_values),
        poll_interval_seconds=None,
        device="cuda:0",
    )

    with monitor:
        middle = monitor.checkpoint("after-model-load")

    report = monitor.report
    assert cuda.reset_calls == ["cuda:0"]
    assert middle.host_rss_bytes == 1_500
    assert middle.cuda.allocated_bytes == 100
    assert middle.cuda.reserved_bytes == 200
    assert report is not None
    assert report.start.host_rss_bytes == 1_000
    assert report.end.host_rss_bytes == 1_250
    assert report.peak_host_rss_bytes == 1_500
    assert report.peak_cuda_allocated_bytes == 300
    assert report.peak_cuda_reserved_bytes == 400
    assert report.elapsed_ns == report.end.monotonic_ns - report.start.monotonic_ns
    assert report.elapsed_seconds == report.elapsed_ns / 1_000_000_000
    assert report.steady_host_rss_bytes == 1_250
    assert report.steady_cuda_allocated_bytes == 100
    assert report.steady_cuda_reserved_bytes == 200


def test_monitor_finishes_and_preserves_report_when_body_raises() -> None:
    rss_values = iter((10, 30, 20))
    monitor = RuntimeMemoryMonitor(
        cuda_backend=FakeCuda(available=False),
        rss_reader=lambda: next(rss_values),
        poll_interval_seconds=None,
    )

    with pytest.raises(RuntimeError, match="boom"):
        with monitor:
            monitor.checkpoint("before-error")
            raise RuntimeError("boom")

    assert monitor.report is not None
    assert monitor.report.completed is False
    assert monitor.report.error_type == "RuntimeError"
    assert monitor.report.peak_host_rss_bytes == 30
    payload = monitor.report.to_dict()
    assert payload["elapsed_ns"] >= 0
    assert payload["steady_host_rss_bytes"] == 20
    assert payload["steady_cuda_allocated_bytes"] is None
    assert payload["steady_cuda_reserved_bytes"] is None
    assert payload["steady_cuda_unavailable_reason"] == "cuda_not_available"


def test_report_dict_is_json_safe_and_keeps_checkpoint_labels() -> None:
    rss_values = iter((100, 120, 110))
    monitor = RuntimeMemoryMonitor(
        cuda_backend=FakeCuda(),
        rss_reader=lambda: next(rss_values),
        poll_interval_seconds=None,
    )

    with monitor:
        monitor.checkpoint("route-complete")

    payload = monitor.report.to_dict() if monitor.report else {}
    assert payload["schema_version"] == "runtime-memory-report/1.0.0"
    assert payload["completed"] is True
    assert [item["label"] for item in payload["checkpoints"]] == [
        "start",
        "route-complete",
        "end",
    ]
    assert payload["peak_host_rss_bytes"] == 120
    assert payload["peak_cuda_allocated_bytes"] == 300
    assert payload["peak_cuda_reserved_bytes"] == 400
    assert payload["elapsed_seconds"] >= 0
    assert payload["steady_host_rss_bytes"] == 110
    assert payload["steady_cuda_allocated_bytes"] == 100
    assert payload["steady_cuda_reserved_bytes"] == 200


def test_invalid_poll_interval_is_rejected() -> None:
    with pytest.raises(ValueError, match="poll_interval_seconds"):
        RuntimeMemoryMonitor(poll_interval_seconds=0)
