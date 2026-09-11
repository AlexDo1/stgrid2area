"""
Small utilities related to worker memory:

- ``format_bytes`` / ``get_worker_memory_limit``: reporting helpers used to
  log how much memory each worker actually has available.
- ``log_memory_pressure_warning``: instead of trying to predict OOM crashes
  upfront (unreliable -- the actual memory used while clipping/aggregating an
  area depends heavily on its size and on downstream library overhead, e.g.
  exact_extract/GDAL, that can't be estimated from array byte counts alone),
  this reacts to *actual* worker deaths observed during a run and logs one
  clear, actionable warning recommending a smaller time chunk.
"""

import logging
from typing import Optional, Union

import xarray as xr


def format_bytes(n: float) -> str:
    """Format a byte count as a human-readable string, e.g. ``1536`` -> ``'1.50 KiB'``."""
    n = float(n)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(n) < 1024.0:
            return f"{n:.2f} {unit}"
        n /= 1024.0
    return f"{n:.2f} PiB"


def get_worker_memory_limit(client) -> Optional[int]:
    """
    Return the smallest per-worker memory limit (bytes) reported by the
    scheduler, or ``None`` if no worker reports a (nonzero) limit.
    """
    try:
        workers = client.scheduler_info().get("workers", {})
    except Exception:
        return None

    limits = [w.get("memory_limit") for w in workers.values() if w.get("memory_limit")]
    return min(limits) if limits else None


def _current_time_chunk_length(stgrids: list, time_dim: str = "time") -> Optional[int]:
    """Largest chunk length along *time_dim* across all given stgrids, or ``None``."""
    for stgrid in stgrids:
        arrays = [stgrid] if isinstance(stgrid, xr.DataArray) else list(stgrid.data_vars.values())
        for da in arrays:
            if time_dim not in da.dims:
                continue
            chunks = getattr(da.data, "chunks", None)
            if not chunks:
                continue
            axis = da.dims.index(time_dim)
            return max(chunks[axis])
    return None


def log_memory_pressure_warning(
    n_worker_deaths: int,
    stgrids: list,
    logger: logging.Logger,
    time_dim: str = "time",
) -> None:
    """
    Log a single, actionable warning if one or more tasks failed because
    their worker(s) were killed while running them (almost always caused by
    running out of memory).
    """
    if not n_worker_deaths:
        return

    chunk_hint = ""
    current_len = _current_time_chunk_length(stgrids, time_dim)
    if current_len and current_len > 1:
        chunk_hint = (
            f" Your stgrid's current '{time_dim}' chunk length is {current_len}; "
            f"try roughly halving it, e.g. "
            f"stgrid.chunk({{'{time_dim}': {max(1, current_len // 2)}}}), and resubmit."
        )

    logger.warning(
        f"{n_worker_deaths} task(s) failed because their worker(s) ran out of "
        f"memory and were killed while processing them. This is most likely "
        f"for large/complex areas. If this keeps happening, reduce your "
        f"stgrid's time chunk size before resubmitting (with skip_exist=True, "
        f"already-completed areas will simply be skipped)." + chunk_hint
    )

