"""Bounded host RSS and CUDA memory instrumentation for classroom routes.

The module is intentionally import-safe on CPU-only hosts.  PyTorch is loaded
only when CUDA metrics are requested, and callers may inject a CUDA-compatible
backend for tests or constrained runtimes.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import os
from pathlib import Path
import resource
import sys
import threading
import time
from typing import Any, Callable, Literal


RUNTIME_MEMORY_REPORT_SCHEMA = "runtime-memory-report/1.0.0"


@dataclass(frozen=True)
class CudaMemorySnapshot:
    available: bool
    allocated_bytes: int | None
    reserved_bytes: int | None
    peak_allocated_bytes: int | None
    peak_reserved_bytes: int | None
    unavailable_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class MemoryCheckpoint:
    label: str
    monotonic_ns: int
    host_rss_bytes: int
    cuda: CudaMemorySnapshot

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "monotonic_ns": self.monotonic_ns,
            "host_rss_bytes": self.host_rss_bytes,
            "cuda": self.cuda.to_dict(),
        }


@dataclass(frozen=True)
class RuntimeMemoryReport:
    start: MemoryCheckpoint
    end: MemoryCheckpoint
    checkpoints: tuple[MemoryCheckpoint, ...]
    peak_host_rss_bytes: int
    peak_cuda_allocated_bytes: int | None
    peak_cuda_reserved_bytes: int | None
    completed: bool
    error_type: str | None
    sampling_errors: tuple[str, ...]
    schema_version: str = RUNTIME_MEMORY_REPORT_SCHEMA

    @property
    def elapsed_ns(self) -> int:
        return max(0, self.end.monotonic_ns - self.start.monotonic_ns)

    @property
    def elapsed_seconds(self) -> float:
        return self.elapsed_ns / 1_000_000_000

    @property
    def steady_host_rss_bytes(self) -> int:
        return self.end.host_rss_bytes

    @property
    def steady_cuda_allocated_bytes(self) -> int | None:
        return self.end.cuda.allocated_bytes

    @property
    def steady_cuda_reserved_bytes(self) -> int | None:
        return self.end.cuda.reserved_bytes

    @property
    def steady_cuda_unavailable_reason(self) -> str | None:
        return self.end.cuda.unavailable_reason

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "completed": self.completed,
            "error_type": self.error_type,
            "start": self.start.to_dict(),
            "end": self.end.to_dict(),
            "checkpoints": [item.to_dict() for item in self.checkpoints],
            "peak_host_rss_bytes": self.peak_host_rss_bytes,
            "peak_cuda_allocated_bytes": self.peak_cuda_allocated_bytes,
            "peak_cuda_reserved_bytes": self.peak_cuda_reserved_bytes,
            "elapsed_ns": self.elapsed_ns,
            "elapsed_seconds": self.elapsed_seconds,
            "steady_host_rss_bytes": self.steady_host_rss_bytes,
            "steady_cuda_allocated_bytes": self.steady_cuda_allocated_bytes,
            "steady_cuda_reserved_bytes": self.steady_cuda_reserved_bytes,
            "steady_cuda_unavailable_reason": self.steady_cuda_unavailable_reason,
            "sampling_errors": list(self.sampling_errors),
        }


def read_host_rss_bytes() -> int:
    """Return the current process RSS without requiring psutil.

    Linux exposes current resident pages through ``/proc/self/statm``.  The
    fallback is the process high-water mark; that is conservative but not a
    substitute for the scope-local polling peak captured by
    :class:`RuntimeMemoryMonitor`.
    """

    try:
        fields = Path("/proc/self/statm").read_text(encoding="ascii").split()
        resident_pages = int(fields[1])
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
        return resident_pages * page_size
    except (FileNotFoundError, IndexError, OSError, ValueError):
        high_water = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        return high_water if sys.platform == "darwin" else high_water * 1024


def _default_cuda_backend() -> tuple[Any | None, str | None]:
    try:
        import torch
    except ImportError:
        return None, "torch_not_installed"
    return torch.cuda, None


def sample_cuda_memory(
    *, cuda_backend: Any | None = None, device: Any | None = None
) -> CudaMemorySnapshot:
    """Sample current and peak CUDA allocation/reservation counters."""

    backend = cuda_backend
    missing_reason = None
    if backend is None:
        backend, missing_reason = _default_cuda_backend()
    if backend is None:
        return CudaMemorySnapshot(False, None, None, None, None, missing_reason)
    try:
        if not backend.is_available():
            return CudaMemorySnapshot(
                False, None, None, None, None, "cuda_not_available"
            )
        return CudaMemorySnapshot(
            True,
            int(backend.memory_allocated(device)),
            int(backend.memory_reserved(device)),
            int(backend.max_memory_allocated(device)),
            int(backend.max_memory_reserved(device)),
            None,
        )
    except (AttributeError, RuntimeError) as exc:
        return CudaMemorySnapshot(
            False,
            None,
            None,
            None,
            None,
            f"cuda_query_failed:{type(exc).__name__}",
        )


class RuntimeMemoryMonitor:
    """Capture checkpoints plus a bounded polling peak for one route scope."""

    def __init__(
        self,
        *,
        cuda_backend: Any | None = None,
        device: Any | None = None,
        rss_reader: Callable[[], int] = read_host_rss_bytes,
        poll_interval_seconds: float | None = 0.05,
    ) -> None:
        if poll_interval_seconds is not None and poll_interval_seconds <= 0:
            raise ValueError("poll_interval_seconds must be positive or None")
        self._cuda_backend = cuda_backend
        self._device = device
        self._rss_reader = rss_reader
        self._poll_interval = poll_interval_seconds
        self._checkpoints: list[MemoryCheckpoint] = []
        self._peak_host_rss_bytes = 0
        self._sampling_errors: list[str] = []
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._active = False
        self.report: RuntimeMemoryReport | None = None

    def _cuda(self) -> Any | None:
        if self._cuda_backend is not None:
            return self._cuda_backend
        backend, _ = _default_cuda_backend()
        self._cuda_backend = backend
        return backend

    def _reset_cuda_peaks(self) -> None:
        backend = self._cuda()
        if backend is None:
            return
        try:
            if backend.is_available():
                backend.reset_peak_memory_stats(self._device)
        except (AttributeError, RuntimeError) as exc:
            self._sampling_errors.append(
                f"cuda_peak_reset_failed:{type(exc).__name__}"
            )

    def _observe_rss(self) -> int:
        value = int(self._rss_reader())
        if value < 0:
            raise ValueError("rss_reader returned a negative value")
        with self._lock:
            self._peak_host_rss_bytes = max(self._peak_host_rss_bytes, value)
        return value

    def _poll(self) -> None:
        assert self._poll_interval is not None
        while not self._stop_event.wait(self._poll_interval):
            try:
                self._observe_rss()
            except (OSError, RuntimeError, ValueError) as exc:
                with self._lock:
                    self._sampling_errors.append(
                        f"host_rss_poll_failed:{type(exc).__name__}"
                    )
                return

    def start(self) -> MemoryCheckpoint:
        if self._active:
            raise RuntimeError("runtime memory monitor is already active")
        self.report = None
        self._checkpoints.clear()
        self._sampling_errors.clear()
        self._peak_host_rss_bytes = 0
        self._stop_event.clear()
        self._reset_cuda_peaks()
        self._active = True
        start = self.checkpoint("start")
        if self._poll_interval is not None:
            self._thread = threading.Thread(
                target=self._poll,
                name="runtime-memory-monitor",
                daemon=True,
            )
            self._thread.start()
        return start

    def checkpoint(self, label: str) -> MemoryCheckpoint:
        if not self._active:
            raise RuntimeError("runtime memory monitor is not active")
        if not isinstance(label, str) or not label.strip():
            raise ValueError("checkpoint label must be non-empty")
        checkpoint = MemoryCheckpoint(
            label=label.strip(),
            monotonic_ns=time.monotonic_ns(),
            host_rss_bytes=self._observe_rss(),
            cuda=sample_cuda_memory(
                cuda_backend=self._cuda_backend, device=self._device
            ),
        )
        self._checkpoints.append(checkpoint)
        return checkpoint

    def finish(self, error: BaseException | None = None) -> RuntimeMemoryReport:
        if not self._active:
            raise RuntimeError("runtime memory monitor is not active")
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, (self._poll_interval or 0) * 4))
            self._thread = None
        end = self.checkpoint("end")
        self._active = False
        cuda_peaks_allocated = [
            item.cuda.peak_allocated_bytes
            for item in self._checkpoints
            if item.cuda.peak_allocated_bytes is not None
        ]
        cuda_peaks_reserved = [
            item.cuda.peak_reserved_bytes
            for item in self._checkpoints
            if item.cuda.peak_reserved_bytes is not None
        ]
        self.report = RuntimeMemoryReport(
            start=self._checkpoints[0],
            end=end,
            checkpoints=tuple(self._checkpoints),
            peak_host_rss_bytes=self._peak_host_rss_bytes,
            peak_cuda_allocated_bytes=(
                max(cuda_peaks_allocated) if cuda_peaks_allocated else None
            ),
            peak_cuda_reserved_bytes=(
                max(cuda_peaks_reserved) if cuda_peaks_reserved else None
            ),
            completed=error is None,
            error_type=type(error).__name__ if error is not None else None,
            sampling_errors=tuple(self._sampling_errors),
        )
        return self.report

    def __enter__(self) -> "RuntimeMemoryMonitor":
        self.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: Any,
    ) -> Literal[False]:
        self.finish(exc)
        return False


__all__ = [
    "CudaMemorySnapshot",
    "MemoryCheckpoint",
    "RUNTIME_MEMORY_REPORT_SCHEMA",
    "RuntimeMemoryMonitor",
    "RuntimeMemoryReport",
    "read_host_rss_bytes",
    "sample_cuda_memory",
]
