"""Multi-node HPC processing using an externally provided dask-mpi cluster."""

import logging
import warnings
from typing import Union

import xarray as xr
from dask.distributed import Client

from ..area import Area
from ._engine import _run_with_client


class MPIDaskProcessor:
    def __init__(
        self,
        areas: list[Area],
        stgrid: Union[xr.Dataset, list[xr.Dataset]],
        variables: Union[str, list[str]],
        method: str,
        operations: list[str],
        skip_exist: bool = False,
        batch_size: int = None,
        save_nc: bool = True,
        save_csv: bool = True,
        logger: logging.Logger = None,
        task_retries: int = 2,
        clip_only_variables: bool = True,
    ) -> None:
        """
        Initialize an MPIDaskProcessor for parallel processing using dask-mpi.

        This processor is intended for multi-node HPC workloads with resources
        allocated via SLURM and launched using ``mpirun`` / ``srun``.

        Parameters
        ----------
        areas : list of Area
        stgrid : xr.Dataset or list of xr.Dataset
            Opened lazily (e.g. ``xr.open_dataset(..., chunks="auto")``).
            Do **not** call ``.load()`` first — workers materialise only the
            spatial subsets they need.
        variables : str or list[str]
        method : str
            ``"exact_extract"``, ``"xarray"``, or ``"fallback_xarray"``.
        operations : list[str]
        skip_exist : bool, optional
        batch_size : int, optional
            Deprecated, will be removed in a future release.
        save_nc : bool, optional
        save_csv : bool, optional
        logger : logging.Logger, optional
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
        self.skip_exist = skip_exist
        self.save_nc = save_nc
        self.save_csv = save_csv
        self.logger = logger
        self.task_retries = task_retries
        self.clip_only_variables = clip_only_variables

        if not self.logger:
            self.logger = logging.getLogger(__name__)
            self.logger.setLevel(logging.INFO)
            self.logger.addHandler(logging.StreamHandler())

    def run(self, client: Client = None) -> dict:
        """
        Run the parallel processing of areas using a Dask cluster.

        Parameters
        ----------
        client : dask.distributed.Client, optional
            Pass an existing client (e.g. one created after
            ``dask_mpi.initialize()``).  If *None*, ``Client()`` is used which
            connects to the scheduler started by dask-mpi.

        Returns
        -------
        dict
            ``successful_areas``, ``total_areas``, ``failed_areas``,
            ``success_rate``, ``area_success_details``.
        """
        self.logger.info("Starting processing with MPIDaskProcessor.")
        if client is None:
            client = Client()
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
