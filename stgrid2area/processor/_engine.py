"""
Core execution loop shared by :class:`~.local.LocalDaskProcessor` and
:class:`~.mpi.MPIDaskProcessor`.
"""

import gc
import logging
import threading
import time
from typing import Union

from dask.distributed import Client, as_completed
from distributed.scheduler import KilledWorker

from ..area import Area
from ..memory_guard import format_bytes, get_worker_memory_limit, log_memory_pressure_warning
from ..tasks import process_area, check_area_needs_processing
from ._progress import _heartbeat_logger


def _run_with_client(
    client: Client,
    stgrids: list,
    areas: list[Area],
    variables: Union[str, list[str]],
    method: str,
    operations: list[str],
    skip_exist: bool,
    save_nc: bool,
    save_csv: bool,
    logger: logging.Logger,
    task_retries: int = 2,
    clip_only_variables: bool = True,
) -> dict:
    """
    Core execution loop shared by :class:`LocalDaskProcessor` and
    :class:`MPIDaskProcessor`.

    Strategy — scatter-all, submit-all (area-first)
    ------------------------------------------------
    All stgrids are scattered upfront as lazy (dask-backed) xarray datasets,
    then every ``area x stgrid`` task is submitted in **area-first** order.

    Parameters
    ----------
    client : dask.distributed.Client
    stgrids : list of xr.Dataset or xr.DataArray
        Lazy (dask-backed) spatiotemporal grids.
    areas : list of Area
    variables : str or list[str]
    method : str
    operations : list[str]
    skip_exist : bool
    save_nc : bool
    save_csv : bool
    logger : logging.Logger
    task_retries : int, optional
        Number of times a failed/killed task is automatically retried by Dask.
    clip_only_variables : bool, optional
        If ``True`` (default), only the requested ``variables`` are read and
        clipped instead of the full stgrid, reducing memory use and runtime
        when the source grid contains additional, unused variables.

    Returns
    -------
    dict
        ``successful_areas``, ``total_areas``, ``failed_areas``,
        ``success_rate``, ``area_success_details``.
    """
    logger.info(f"Dask dashboard: {client.dashboard_link}")

    total_areas = len(areas)
    total_stgrids = len(stgrids)
    area_success = {area.id: 0 for area in areas}
    processed_areas = 0

    n_workers_now = len(client.nthreads())
    worker_memory_bytes = get_worker_memory_limit(client)
    if worker_memory_bytes:
        logger.info(f"Worker memory limit: {format_bytes(worker_memory_bytes)} per worker.")
    else:
        logger.warning("Could not determine per-worker memory limit.")

    logger.info(
        f"Processing {total_areas} areas x {total_stgrids} stgrids. "
        f"Active workers: {n_workers_now}"
    )

    ac = as_completed()
    n_submitted = 0
    scattered: dict = {}
    _all_collected = False
    _stop_heartbeat: threading.Event | None = None
    n_worker_deaths = 0

    try:
        area_file_cache: dict = {}
        if skip_exist:
            t_scan = time.perf_counter()
            for _area in areas:
                try:
                    area_file_cache[_area.id] = (
                        {f.name for f in _area.output_path.iterdir()}
                        if _area.output_path.exists()
                        else set()
                    )
                except OSError:
                    area_file_cache[_area.id] = set()
            logger.info(
                f"Scanned {len(areas)} output dirs in "
                f"{time.perf_counter() - t_scan:.1f}s."
            )

        t_scatter = time.perf_counter()
        for n_idx, stgrid in enumerate(stgrids):
            stgrid.close()
            scattered[n_idx + 1] = client.scatter(stgrid, broadcast=False)
        logger.info(
            f"Scattered {total_stgrids} stgrids in "
            f"{time.perf_counter() - t_scatter:.1f}s."
        )

        t_submit = time.perf_counter()
        for area in areas:
            for n_stgrid, stgrid_fut in scattered.items():
                if skip_exist and not check_area_needs_processing(
                    area, n_stgrid, total_stgrids, save_nc, save_csv,
                    existing_files=area_file_cache.get(area.id),
                ):
                    area_success[area.id] += 1
                    if area_success[area.id] == total_stgrids:
                        processed_areas += 1
                    continue

                fut = client.submit(
                    process_area,
                    area, stgrid_fut,
                    variables, method, operations,
                    skip_exist, n_stgrid, total_stgrids,
                    save_nc, save_csv,
                    clip_only_variables,
                    key=f"{area.id}_{n_stgrid}",
                    pure=False,
                    retries=task_retries,
                )
                ac.add(fut)
                n_submitted += 1

        if n_submitted == 0:
            logger.info(f"All {total_areas} areas already processed, skipping.")
            _all_collected = True
        else:
            n_tasks_skipped = total_areas * total_stgrids - n_submitted
            logger.info(
                f"Submitted {n_submitted} tasks "
                f"({processed_areas} areas already fully processed, "
                f"{n_tasks_skipped} tasks skipped total) "
                f"in {time.perf_counter() - t_submit:.1f}s."
            )

            progress_state = {
                "n_done": 0,
                "n_err": 0,
                "n_worker_deaths": 0,
                "areas_done": processed_areas
            }

            _stop_heartbeat = threading.Event()
            _hb_thread = threading.Thread(
                target=_heartbeat_logger,
                args=(progress_state, n_submitted, total_areas, client, logger, _stop_heartbeat, 300),
                kwargs={
                    "areas": areas,
                    "total_stgrids": total_stgrids,
                    "save_nc": save_nc,
                    "save_csv": save_csv,
                },
                daemon=True,
            )
            _hb_thread.start()

            try:
                for future in ac:
                    area_id, n_stgrid_str = future.key.rsplit("_", 1)
                    n_stgrid = int(n_stgrid_str)

                    try:
                        result = future.result()
                        if isinstance(result, dict) and result.get("status") == "success":
                            area_success[area_id] += 1
                            if area_success[area_id] == total_stgrids:
                                progress_state["areas_done"] += 1
                    except KilledWorker as e:
                        progress_state["n_err"] += 1
                        progress_state["n_worker_deaths"] += 1
                        logger.error(f"{area_id}, stgrid {n_stgrid} --- Error: {e}")
                    except Exception as e:
                        progress_state["n_err"] += 1
                        logger.error(f"{area_id}, stgrid {n_stgrid} --- Error: {e}")
                    finally:
                        progress_state["n_done"] += 1
                        future.release()

            finally:
                _stop_heartbeat.set()
                _hb_thread.join(timeout=5)

            _all_collected = True
            n_worker_deaths = progress_state["n_worker_deaths"]

    except Exception:
        logger.exception("Error during processing:")

    finally:
        if _stop_heartbeat is not None:
            _stop_heartbeat.set()

        if not _all_collected:
            try:
                client.cancel(list(ac.futures))
            except Exception:
                pass

        if scattered:
            try:
                client.cancel(list(scattered.values()))
            except Exception:
                pass
        scattered.clear()

        gc.collect()

    successful_areas = sum(1 for count in area_success.values() if count == total_stgrids)
    failed_areas = [area_id for area_id, count in area_success.items() if count < total_stgrids]

    result = {
        "successful_areas": successful_areas,
        "total_areas": total_areas,
        "failed_areas": failed_areas,
        "success_rate": (successful_areas / total_areas) * 100 if total_areas > 0 else 0,
        "area_success_details": area_success,
    }

    logger.info(
        f"Processing completed: {successful_areas}/{total_areas} areas processed successfully."
    )
    log_memory_pressure_warning(n_worker_deaths, stgrids, logger)
    return result
