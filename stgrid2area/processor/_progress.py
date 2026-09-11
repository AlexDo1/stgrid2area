"""
Progress/heartbeat logging for the shared processing loop in ``_engine.py``.
"""

import logging
import threading

from ..area import Area
from ..tasks import check_area_needs_processing


def _is_area_fully_done_on_disk(
    area: Area, total_stgrids: int, save_nc: bool, save_csv: bool
) -> bool:
    """Check on disk (ground truth) whether every stgrid stage of *area* is done."""
    try:
        existing_files = (
            {f.name for f in area.output_path.iterdir()} if area.output_path.exists() else set()
        )
    except OSError:
        return False
    return all(
        not check_area_needs_processing(
            area, n, total_stgrids, save_nc, save_csv, existing_files=existing_files
        )
        for n in range(1, total_stgrids + 1)
    )


def _count_areas_done_on_disk(
    areas: list[Area], total_stgrids: int, save_nc: bool, save_csv: bool
) -> int:
    """
    Count areas whose output files are all already present on disk.

    This is a ground-truth cross-check for the heartbeat's future-based
    ``areas_done`` counter, which only increments once a completed task's
    result has actually been collected by the client. Under heavy worker
    churn (workers repeatedly killed/restarted due to memory pressure), a
    task's output file can be written to disk long before (or even without)
    its result ever reaching the client, since a dead worker's already-
    computed-but-uncollected results are lost and the task is silently
    resubmitted/recomputed elsewhere.
    """
    return sum(
        _is_area_fully_done_on_disk(area, total_stgrids, save_nc, save_csv) for area in areas
    )


def _heartbeat_logger(
    progress_state: dict,
    n_total: int,
    n_areas_total: int,
    client,
    logger: logging.Logger,
    stop_event: threading.Event,
    interval: int = 300,
    areas: list[Area] = None,
    total_stgrids: int = None,
    save_nc: bool = None,
    save_csv: bool = None,
) -> None:
    """Background thread: logs true task progress using atomic counters."""
    while not stop_event.wait(interval):
        try:
            n_done = progress_state["n_done"]
            n_err = progress_state["n_err"]
            areas_done = progress_state["areas_done"]

            pct_tasks = n_done / n_total * 100 if n_total else 0.0
            pct_areas = areas_done / n_areas_total * 100 if n_areas_total else 0.0

            try:
                n_workers = len(client.nthreads())
            except Exception as e:
                n_workers = "?"
                logger.debug(f"[heartbeat] Could not fetch worker count: {e}")

            w_t = len(str(n_total))
            w_a = len(str(n_areas_total))

            on_disk_str = ""
            if areas is not None:
                try:
                    areas_done_on_disk = _count_areas_done_on_disk(
                        areas, total_stgrids, save_nc, save_csv
                    )
                    on_disk_str = f"  │  on-disk: {areas_done_on_disk:>{w_a}} / {n_areas_total} areas"
                except Exception as e:
                    logger.debug(f"[heartbeat] Could not scan output directories: {e}")

            logger.info(
                f"[heartbeat]"
                f"  tasks:  {n_done:>{w_t}} / {n_total}  ({pct_tasks:4.1f}%)"
                f"  │  areas:  {areas_done:>{w_a}} / {n_areas_total}  ({pct_areas:4.1f}%)"
                f"  │  {n_err} errors"
                f"  │  {n_workers} workers"
                f"{on_disk_str}"
            )

        except Exception as e:
            logger.debug(f"[heartbeat] Encountered an error: {e}")
