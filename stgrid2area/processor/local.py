"""Single-machine / single-node processing using a local Dask cluster."""

import logging
import warnings
from typing import Union

import xarray as xr
from dask.distributed import Client, LocalCluster

from ..area import Area
from ._cluster import _detect_available_cpus, _resolve_memory_limit
from ._engine import _run_with_client


class LocalDaskProcessor:
    def __init__(
        self,
        areas: list[Area],
        stgrid: Union[xr.Dataset, list[xr.Dataset]],
        variables: Union[str, list[str]],
        method: str,
        operations: list[str],
        n_workers: int = None,
        skip_exist: bool = False,
        batch_size: int = None,
        save_nc: bool = True,
        save_csv: bool = True,
        logger: logging.Logger = None,
        memory_limit: Union[str, int, None] = "auto",
        task_retries: int = 2,
        clip_only_variables: bool = True,
    ) -> None:
        """
        Initialize a LocalDaskProcessor.

        Parameters
        ----------
        areas : list of Area
        stgrid : xr.Dataset or list of xr.Dataset
            Opened lazily (e.g. ``xr.open_dataset(..., chunks="auto")``).
            Do **not** call ``.load()`` first.
        variables : str or list[str]
        method : str
            ``"exact_extract"``, ``"xarray"``, or ``"fallback_xarray"``.
        operations : list[str]
        n_workers : int, optional
            Number of local Dask workers. Default: the number of CPUs
            actually available to this process (respecting SLURM/cgroup
            affinity), *not* ``os.cpu_count()``, which would overcount on
            hyperthreaded/shared nodes and starve workers of memory.
        skip_exist : bool, optional
        batch_size : int, optional
            Deprecated, will be removed in a future release.
        save_nc : bool, optional
        save_csv : bool, optional
        logger : logging.Logger, optional
        memory_limit : str, int, or None, optional
            Memory limit per worker, forwarded to ``LocalCluster``. Default
            ``"auto"``, which stgrid2area resolves to total system memory
            divided by ``n_workers`` -- *not* distributed's own ``"auto"``,
            which divides by its own (potentially wrong, see
            ``_detect_available_cpus``) idea of the CPU count instead of the
            actual number of workers started.
        task_retries : int, optional
            Number of times a failed/killed task is automatically retried.
            Default ``2``. If a task fails because its worker ran out of
            memory and was killed, a warning recommending a smaller time
            chunk is logged once processing completes (see
            ``memory_guard.log_memory_pressure_warning``).
        clip_only_variables : bool, optional
            If ``True`` (default), only the requested ``variables`` are
            read/clipped instead of the full stgrid.
        """
        if batch_size is not None:
            warnings.warn(
                "batch_size is deprecated and will be removed in a future release.",
                DeprecationWarning,
                stacklevel=2,
            )

        self.areas = areas
        if isinstance(stgrid, (xr.Dataset, xr.DataArray)):
            self.stgrids = [stgrid]
        elif isinstance(stgrid, list):
            self.stgrids = stgrid
        else:
            raise ValueError(
                "stgrid must be an xr.Dataset, xr.DataArray, or a list thereof."
            )
        self.variables = variables
        self.method = method
        self.operations = operations
        self.n_workers = n_workers or _detect_available_cpus()
        self.skip_exist = skip_exist
        self.save_nc = save_nc
        self.save_csv = save_csv
        self.logger = logger
        self.memory_limit = memory_limit
        self.task_retries = task_retries
        self.clip_only_variables = clip_only_variables

        if not self.logger:
            self.logger = logging.getLogger(__name__)
            self.logger.setLevel(logging.INFO)
            self.logger.addHandler(logging.StreamHandler())

    def run(self) -> dict:
        """
        Run the parallel processing of areas using a local Dask cluster.

        Returns
        -------
        dict
            ``successful_areas``, ``total_areas``, ``failed_areas``,
            ``success_rate``, ``area_success_details``.
        """
        self.logger.info(
            f"Starting processing with LocalDaskProcessor ({self.n_workers} workers)."
        )
        resolved_memory_limit = _resolve_memory_limit(self.memory_limit, self.n_workers)
        with LocalCluster(
            n_workers=self.n_workers,
            threads_per_worker=1,
            memory_limit=resolved_memory_limit,
        ) as cluster:
            with Client(cluster) as client:
                return _run_with_client(
                    client=client,
                    stgrids=self.stgrids,
                    areas=self.areas,
                    variables=self.variables,
                    method=self.method,
                    operations=self.operations,
                    skip_exist=self.skip_exist,
                    save_nc=self.save_nc,
                    save_csv=self.save_csv,
                    logger=self.logger,
                    task_retries=self.task_retries,
                    clip_only_variables=self.clip_only_variables,
                )
