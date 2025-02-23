import os
from typing import Union
import gc
from dask import delayed
from dask.distributed import Client, LocalCluster, as_completed
from dask_jobqueue import SLURMCluster
import pandas as pd
import xarray as xr
import rioxarray
import logging
from pathlib import Path
import numpy as np

from .area import Area


def process_area(area: Area, stgrid: Union[xr.Dataset, xr.DataArray], variable: str, method: str, operations: list[str], skip_exist: bool, 
                 n_stgrid: int, total_stgrids: int, save_nc: bool = True, save_csv: bool = True) -> pd.DataFrame:
    """
    Standalone function to process (clip and aggregate) a single area.  
    This cannot be a method of the Processor class because it is used in parallel processing with Dask and
    `self.process_area` would serialize the entire Processor object, with all its data, meaning that the
    all stgrids and areas would be copied to each worker, which is very inefficient and consumes a lot of memory.

    Parameters
    ----------
    area : Area
        The area to process.
    stgrid : xr.Dataset or xr.DataArray
        The spatiotemporal grid to clip to the area.
    variable : str
        The variable in stgrid to aggregate.
    method : str
        The method to use for aggregation.
    operations : list of str
        List of aggregation operations to apply.
    skip_exist : bool
        If True, skip processing areas that already have clipped grids or aggregated in their output directories.
    n_stgrid : int
        The index of the spatiotemporal grid in the list of stgrids.
    total_stgrids : int
        The total number of spatiotemporal grids to process.
    save_nc : bool, optional
        If True, save the clipped grid to a NetCDF file in the output directory of the area.
    save_csv : bool, optional
        If True, save the aggregated variable to a CSV file in the output directory of the area. 

    Returns
    -------
    pd.DataFrame
        The aggregated variable as a pandas DataFrame.
    
    """
    filename_clip = f"{area.id}_{n_stgrid}_clipped.nc" if total_stgrids > 1 else f"{area.id}_clipped.nc"
    filename_aggr = f"{area.id}_{n_stgrid}_aggregated.csv" if total_stgrids > 1 else f"{area.id}_aggregated.csv"
    
    try:
        clipped = area.clip(stgrid, save_result=save_nc, skip_exist=skip_exist, filename=filename_clip)
    
        if method in ["exact_extract", "xarray"]:
            result = area.aggregate(clipped, variable, method, operations, 
                                    save_result=save_csv, skip_exist=skip_exist, filename=filename_aggr)
        elif method == "fallback_xarray":
            try:
                result = area.aggregate(clipped, variable, "exact_extract", operations, 
                                        save_result=save_csv, skip_exist=skip_exist, filename=filename_aggr)
            except ValueError:
                # Indicate fallback was used
                Path(area.output_path, "fallback_xarray").touch()
                result = area.aggregate(clipped, variable, "xarray", operations, 
                                        save_result=save_csv, skip_exist=skip_exist, filename=filename_aggr)
        return result
    except Exception as e:
        raise e
    finally:
        clipped.close()
        stgrid.close()        


def check_area_needs_processing(area: Area, n_stgrid: int, total_stgrids: int, save_nc: bool, save_csv: bool) -> bool:
    """
    Helper function to check if an area needs processing based on existing files.
    This function checks if the clipped and aggregated files already exist in the output directory of the area.
    This is only used when skip_exist=True.
    
    Parameters
    ----------
    area : Area
        The area to check.
    n_stgrid : int
        The index of the spatiotemporal grid in the list of stgrids.
    total_stgrids : int
        The total number of spatiotemporal grids to process.
    save_nc : bool
        If True, save the clipped grid to a NetCDF file in the output directory of the area.
    save_csv : bool
        If True, save the aggregated variable to a CSV file in the output directory of the area.
    
    Returns
    -------
    bool
        True if the area needs processing, False otherwise. 
    
    """
     # Check for merged files if they exist
    if save_csv:
        merged_csv = Path(area.output_path) / f"{area.id}_aggregated.csv"
        if merged_csv.exists():
            return False
            
    if save_nc:
        merged_nc = Path(area.output_path) / f"{area.id}_clipped.nc"
        if merged_nc.exists():
            return False

    filename_clip = f"{area.id}_{n_stgrid}_clipped.nc" if total_stgrids > 1 else f"{area.id}_clipped.nc"
    filename_aggr = f"{area.id}_{n_stgrid}_aggregated.csv" if total_stgrids > 1 else f"{area.id}_aggregated.csv"
    
    clip_path = Path(area.output_path) / filename_clip
    aggr_path = Path(area.output_path) / filename_aggr
    
    # If only saving NetCDF files
    if save_nc and not save_csv:
        return not clip_path.exists()
        
    # If only saving CSV files
    if save_csv and not save_nc:
        return not aggr_path.exists()
        
    # If saving both
    if save_nc and save_csv:
        return not (clip_path.exists() and aggr_path.exists())
    
    # If neither file type is being saved, always process
    return True

class LocalDaskProcessor:
    def __init__(self, areas: list[Area], stgrid: Union[Union[xr.Dataset, xr.DataArray], list[Union[xr.Dataset, xr.DataArray]]], 
                 variable: str, method: str, operations: list[str], n_workers: int = None, skip_exist: bool = False, batch_size: int = None, 
                 save_nc: bool = True, save_csv: bool = True, logger: logging.Logger = None) -> None:
        """
        Initialize a LocalDaskProcessor for efficient parallel processing on a single machine.

        Parameters
        ----------
        areas : list of Area
            List of area objects to process.
        stgrid : xr.Dataset or xr.DataArray
            The spatiotemporal data to process.  
            If stgrid is a list of xr.Dataset or xr.DataArray, the processor will process each one in turn. Splitting the data into multiple
            xr.Dataset or xr.DataArray objects can be useful when the spatiotemporal data is too large to fit into memory.
        variable : str
            The variable in stgrid to aggregate.
        method : str, optional
            The method to use for aggregation.  
            Can be "exact_extract", "xarray" or "fallback_xarray".  
            "fallback_xarray" will first try to use the exact_extract method, and if this raises a ValueError, it will fall back to 
            the xarray method.
        operations : list of str
            List of aggregation operations to apply.
        n_workers : int, optional
            Number of parallel workers to use (default: os.cpu_count()).
        skip_exist : bool, optional
            If True, skip processing areas that already have clipped grids or aggregated in their output directories.
        batch_size : int, optional
            Number of areas to process in each batch. Default: process all areas at once.  
            If the number of areas is large, it may be necessary to process them in smaller batches to avoid memory issues.
        save_nc : bool, optional
            If True, save the clipped grids to NetCDF files in the output directories of the areas.
        save_csv : bool, optional
            If True, save the aggregated variables to CSV files in the output directories of the areas.
        logger : logging.Logger, optional
            Logger to use for logging. If None, a basic logger will be set up.

        """
        self.areas = areas
        if isinstance(stgrid, xr.Dataset) or isinstance(stgrid, xr.DataArray):
            self.stgrid = [stgrid]
        elif isinstance(stgrid, list):
            self.stgrid = stgrid
        else:
            raise ValueError("stgrid must be an xr.Dataset, xr.DataArray or a list of xr.Dataset or xr.DataArray.")
        self.variable = variable
        self.method = method
        self.operations = operations
        self.n_workers = n_workers or os.cpu_count()
        self.skip_exist = skip_exist
        self.save_nc = save_nc
        self.save_csv = save_csv
        self.logger = logger
        self.batch_size = batch_size or len(areas)  # Default: process all areas at once

        # Set up basic logging if no handler is configured
        if not self.logger:
            self.logger = logging.getLogger(__name__)
            self.logger.setLevel(logging.INFO)
            self.logger.addHandler(logging.StreamHandler())
        
    def run(self) -> None:
        """
        Run the parallel processing of areas using Dask with batching.
        
        """
        self.logger.info("Starting processing with LocalDaskProcessor.")

        with LocalCluster(n_workers=self.n_workers, threads_per_worker=1) as cluster:
            with Client(cluster) as client:
                try:
                    self.logger.info(f"Dask dashboard address: {client.dashboard_link}")
                    
                    # Split areas into batches
                    area_batches = np.array_split(self.areas, max(1, len(self.areas) // self.batch_size))
                    self.logger.info(f"Processing {len(self.areas)} areas in {len(area_batches)} batches.")

                    total_areas = len(self.areas)
                    area_success = {area.id: 0 for area in self.areas}  # Track success count per area
                    total_stgrids = len(self.stgrid)
                    processed_areas = 0

                    # Process each batch of areas
                    for i, batch in enumerate(area_batches, start=1):
                        self.logger.info(f"Processing batch {i}/{len(area_batches)} with {len(batch)} areas.")

                        for n_stgrid, stgrid in enumerate(self.stgrid, start=1):
                            # First check which areas need processing
                            if self.skip_exist:
                                areas_to_process = []
                                for area in batch:
                                    if check_area_needs_processing(area, n_stgrid, total_stgrids, 
                                                                   self.skip_exist, self.save_nc, self.save_csv):
                                        areas_to_process.append(area)
                                    else:
                                        area_success[area.id] += 1
                                        if area_success[area.id] == total_stgrids:
                                            processed_areas += 1
                                            self.logger.info(f"[{processed_areas}/{total_areas}]: {area.id} --- Already processed.")

                                if not areas_to_process:
                                    self.logger.info(f"Skipping batch {i}, stgrid {n_stgrid} - all areas already processed.")
                                    continue
                            else:
                                # No skipping, process all areas in the batch
                                areas_to_process = batch

                            self.logger.info(f"Processing {len(areas_to_process)} areas in batch {i}.")

                            try:
                                # Pre-clip individually for each area
                                area_stgrids = {
                                    area.id: stgrid.rio.clip(
                                        area.geometry.geometry.to_crs(stgrid.rio.crs), 
                                        all_touched=True
                                    ).persist() 
                                    for area in areas_to_process
                                }

                                # Create tasks with area-specific pre-clipped grids
                                tasks = [
                                    delayed(process_area)(
                                        area,
                                        area_stgrids[area.id], # Use area-specific grid
                                        self.variable,
                                        self.method,
                                        self.operations,
                                        self.skip_exist,
                                        n_stgrid,
                                        total_stgrids,
                                        self.save_nc,
                                        self.save_csv,
                                        dask_key_name=f"{area.id}_{n_stgrid}"
                                    ) for area in areas_to_process
                                ]

                                futures = client.compute(tasks)

                                for future in as_completed(futures):
                                    area_id = future.key.split('_')[0]  # Extract area ID from the key
                                    try:
                                        result = future.result()

                                        if isinstance(result, pd.DataFrame):
                                            area_success[area_id] += 1
                                            area_stgrids[area_id].close()
                                            # Only log success when all stgrids for an area are processed
                                            if area_success[area_id] == total_stgrids:
                                                processed_areas += 1
                                                self.logger.info(f"[{processed_areas}/{total_areas}]: {area_id} --- Processing completed.")
                                    except Exception as e:
                                        self.logger.error(f"{area_id}, stgrid {n_stgrid} --- Error occurred: {e}")

                                # Cleanup futures and persisted data
                                client.cancel(futures)
                                for grid in area_stgrids.values():
                                    if hasattr(grid, 'close'):
                                        grid.close()
                                    client.cancel(grid)
                                del area_stgrids, tasks, futures
                                gc.collect()

                            except Exception as e:
                                self.logger.error(f"Error during batch {i}, stgrid {n_stgrid}: {e}")

                        # Restart the Dask client and cluster after the batch
                        client.restart()
                        self.logger.info(f"Finished batch {i}/{len(area_batches)}. Restarted Dask client and cluster for the next batch.")

                    # Final summary
                    successful_areas = sum(1 for count in area_success.values() if count == total_stgrids)
                    self.logger.info(f"Processing completed: {successful_areas}/{total_areas} areas processed successfully.")
                except Exception as e:
                    self.logger.error(f"An error occurred: {e}")
                finally:
                    self.logger.info("Shutting down Dask client and cluster.")

class SLURMDaskProcessor:
    def __init__(self,
                 areas: list[Area],
                 stgrid: Union[Union[xr.Dataset, xr.DataArray], list[Union[xr.Dataset, xr.DataArray]]],
                 variable: str,
                 method: str,
                 operations: list[str],
                 desired_jobs: int,      # Total number of worker jobs to launch
                 queue: str,             # SLURM queue name (e.g., "multiple_il")
                 cores: int,             # Cores per worker job (should match --ntasks-per-node)
                 memory: str,            # Total memory per worker job (e.g., "124800MB")
                 walltime: str,          # Walltime for each worker job (e.g., "01:00:00")
                 skip_exist: bool = False,
                 batch_size: int = None,
                 save_nc: bool = True,
                 save_csv: bool = True,
                 logger: logging.Logger = None,
                 threads_per_worker: int = 1,
                 cluster_kwargs: dict = None):
        """
        Initialize a SLURMDaskProcessor for parallel processing on a SLURM HPC.

        Explicitly provide the configuration for the SLURMCluster, e.g.:
            desired_jobs = 8       # Use 8 nodes (worker jobs)
            queue = "multiple"     # SLURM queue name
            cores = 64             # 64 tasks per node
            memory = "124800MB"    # 64 * 1950MB
            walltime = "01:00:00"
        
        Parameters
        ----------
        areas : list of Area
            List of area objects to process.
        stgrid : xr.Dataset, xr.DataArray or list of them
            The spatiotemporal data to process.
        variable : str
            The variable in stgrid to aggregate.
        method : str
            Aggregation method ("exact_extract", "xarray", or "fallback_xarray").
        operations : list of str
            List of aggregation operations to apply.
        desired_jobs : int
            The total number of worker jobs (nodes or job units) to submit.
        queue : str
            The SLURM queue name (e.g., "multiple_il").
        cores : int
            Number of cores per worker job (should match your HPC’s ntasks-per-node).
        memory : str
            Total memory per worker job (e.g., "124800MB").
        walltime : str
            Walltime for each worker job (e.g., "01:00:00").
        skip_exist : bool, optional
            If True, skip processing areas that already have outputs.
        batch_size : int, optional
            Number of areas to process in each batch.
        save_nc : bool, optional
            If True, save clipped grids to NetCDF.
        save_csv : bool, optional
            If True, save aggregated variables to CSV.
        logger : logging.Logger, optional
            Logger for logging.
        threads_per_worker : int, optional
            Number of threads per worker (default is 1 for CPU-bound tasks).
        cluster_kwargs : dict, optional
            Additional keyword arguments to pass to SLURMCluster.

        """
        self.areas = areas
        if isinstance(stgrid, xr.Dataset) or isinstance(stgrid, xr.DataArray):
            self.stgrid = [stgrid]
        elif isinstance(stgrid, list):
            self.stgrid = stgrid
        else:
            raise ValueError("stgrid must be an xr.Dataset, xr.DataArray or a list of them.")
        self.variable = variable
        self.method = method
        self.operations = operations
        self.desired_jobs = desired_jobs
        self.queue = queue
        self.cores = cores
        self.memory = memory
        self.walltime = walltime
        self.skip_exist = skip_exist
        self.save_nc = save_nc
        self.save_csv = save_csv
        self.batch_size = batch_size or len(areas)
        self.threads_per_worker = threads_per_worker
        self.cluster_kwargs = cluster_kwargs or {}

        # Set up logging.
        self.logger = logger
        if not self.logger:
            self.logger = logging.getLogger(__name__)
            self.logger.setLevel(logging.INFO)
            self.logger.addHandler(logging.StreamHandler())

    def _log_memory_usage(self, client: Client, message: str = "Memory usage"):
        try:
            worker_info = client.scheduler_info()["workers"]
            total_memory = sum(worker["memory"] for worker in worker_info.values())
            self.logger.info(f"{message}: {total_memory / (1024**2):.2f} MB")
        except Exception as e:
            self.logger.error(f"Could not log memory usage: {e}")

    def run(self) -> None:
        """
        Run the parallel processing of areas using a Dask cluster created with SLURMCluster.
        
        Worker jobs are submitted based on the explicit cluster configuration provided.
        """
        self.logger.info("Starting processing with SLURMDaskProcessor (SLURM HPC).")

        # Create the SLURMCluster using the explicit configuration.
        cluster = SLURMCluster(
            queue=self.queue,
            cores=self.cores,
            memory=self.memory,
            walltime=self.walltime,
            processes=self.cores,  # Typically one process per core (with threads_per_worker=1)
            **self.cluster_kwargs
        )
        self.logger.info(f"Scaling cluster to {self.desired_jobs} jobs.")
        cluster.scale(jobs=self.desired_jobs)

        client = Client(cluster)
        try:
            self.logger.info(f"Dask dashboard address: {client.dashboard_link}")

            # Split areas into batches.
            area_batches = np.array_split(self.areas, max(1, len(self.areas) // self.batch_size))
            self.logger.info(f"Processing {len(self.areas)} areas in {len(area_batches)} batches.")

            total_areas = len(self.areas)
            area_success = {area.id: 0 for area in self.areas}
            total_stgrids = len(self.stgrid)
            processed_areas = 0

            for i, batch in enumerate(area_batches, start=1):
                self.logger.info(f"Processing batch {i}/{len(area_batches)} with {len(batch)} areas.")

                for n_stgrid, stgrid in enumerate(self.stgrid, start=1):
                    try:
                        # First check which areas need processing
                        if self.skip_exist:
                            areas_to_process = []
                            for area in batch:
                                if check_area_needs_processing(area, n_stgrid, total_stgrids, 
                                                            self.skip_exist, self.save_nc, self.save_csv):
                                    areas_to_process.append(area)
                                else:
                                    area_success[area.id] += 1
                                    if area_success[area.id] == total_stgrids:
                                        processed_areas += 1
                                        self.logger.info(f"[{processed_areas}/{total_areas}]: {area.id} --- Already processed.")

                            if not areas_to_process:
                                self.logger.info(f"Skipping batch {i}, stgrid {n_stgrid} - all areas already processed.")
                                continue
                        else:
                            # No skipping, process all areas in the batch
                            areas_to_process = batch

                        # Pre-clip each area individually using its own geometry.
                        area_stgrids = {
                            area.id: stgrid.rio.clip(
                                area.geometry.geometry.to_crs(stgrid.rio.crs),
                                all_touched=True
                            ).persist()
                            for area in areas_to_process
                        }

                        tasks = [
                            delayed(process_area)(
                                area,
                                area_stgrids[area.id],
                                self.variable,
                                self.method,
                                self.operations,
                                self.skip_exist,
                                n_stgrid,
                                total_stgrids,
                                self.save_nc,
                                self.save_csv,
                                dask_key_name=f"{area.id}_{n_stgrid}"
                            ) for area in areas_to_process
                        ]

                        futures = client.compute(tasks)
                        for future in as_completed(futures):
                            area_id = future.key.split('_')[0]
                            try:
                                result = future.result()
                                if isinstance(result, pd.DataFrame):
                                    area_success[area_id] += 1
                                    try:
                                        area_stgrids[area_id].close()
                                    except Exception:
                                        pass
                                    if area_success[area_id] == total_stgrids:
                                        processed_areas += 1
                                        self.logger.info(f"[{processed_areas}/{total_areas}]: {area_id} --- Processing completed.")
                            except Exception as e:
                                self.logger.error(f"{area_id}, stgrid {n_stgrid} --- Error occurred: {e}")

                        client.cancel(futures)
                        for grid in area_stgrids.values():
                            if hasattr(grid, 'close'):
                                try:
                                    grid.close()
                                except Exception:
                                    pass
                            client.cancel(grid)
                        del area_stgrids, tasks, futures
                        gc.collect()
                    except Exception as e:
                        self.logger.error(f"Error during batch {i}, stgrid {n_stgrid}: {e}")

                self._log_memory_usage(client, message=f"Memory usage before restart for batch {i}")
                client.restart()
                self.logger.info(f"Finished batch {i}/{len(area_batches)}. Restarted Dask client and cluster for the next batch.\n")
                self._log_memory_usage(client, message=f"Memory usage after restart for batch {i}")

            successful_areas = sum(1 for count in area_success.values() if count == total_stgrids)
            self.logger.info(f"Processing completed: {successful_areas}/{total_areas} areas processed successfully.")
        except Exception as e:
            self.logger.error(f"An error occurred: {e}")
        finally:
            self.logger.info("Shutting down Dask client and cluster.")
            client.close()
            cluster.close()

class MPIDaskProcessor:
    def __init__(self, 
                 areas: list[Area], 
                 stgrid: Union[Union[xr.Dataset, xr.DataArray], list[Union[xr.Dataset, xr.DataArray]]], 
                 variable: str, 
                 method: str, 
                 operations: list[str],
                 skip_exist: bool = False, 
                 batch_size: int = None, 
                 save_nc: bool = True, 
                 save_csv: bool = True, 
                 logger: logging.Logger = None) -> None:
        """
        Initialize a MPIDaskProcessor for parallel processing using dask-mpi.
        
        This processor is intended for static workloads running on a multi-node HPC 
        with resources allocated via SLURM and launched using mpirun/srun.
        
        Parameters
        ----------
        areas : list of Area
            List of area objects to process.
        stgrid : xr.Dataset or xr.DataArray
            The spatiotemporal data to process.  
            If stgrid is a list of xr.Dataset or xr.DataArray, the processor will process each one in turn. Splitting the data into multiple
            xr.Dataset or xr.DataArray objects can be useful when the spatiotemporal data is too large to fit into memory.
        variable : str
            The variable in stgrid to aggregate.
        method : str, optional
            The method to use for aggregation.  
            Can be "exact_extract", "xarray" or "fallback_xarray".  
            "fallback_xarray" will first try to use the exact_extract method, and if this raises a ValueError, it will fall back to 
            the xarray method.
        operations : list of str
            List of aggregation operations to apply.
        skip_exist : bool, optional
            If True, skip processing areas that already have clipped grids or aggregated in their output directories.
        batch_size : int, optional
            Number of areas to process in each batch. Default: process all areas at once.  
            If the number of areas is large, it may be necessary to process them in smaller batches to avoid memory issues.
        save_nc : bool, optional
            If True, save the clipped grids to NetCDF files in the output directories of the areas.
        save_csv : bool, optional
            If True, save the aggregated variables to CSV files in the output directories of the areas.
        logger : logging.Logger, optional
            Logger to use for logging. If None, a basic logger will be set up.
        """
        self.areas = areas
        if isinstance(stgrid, xr.Dataset) or isinstance(stgrid, xr.DataArray):
            self.stgrid = [stgrid]
        elif isinstance(stgrid, list):
            self.stgrid = stgrid
        else:
            raise ValueError("stgrid must be an xr.Dataset, xr.DataArray or a list of them.")
            
        self.variable = variable
        self.method = method
        self.operations = operations
        # n_workers is not used here because dask-mpi will launch as many processes as allocated.
        self.skip_exist = skip_exist
        self.save_nc = save_nc
        self.save_csv = save_csv
        self.batch_size = batch_size or len(areas)
        self.logger = logger

        # Set up basic logging if no handler is configured
        if not self.logger:
            self.logger = logging.getLogger(__name__)
            self.logger.setLevel(logging.INFO)
            self.logger.addHandler(logging.StreamHandler())

    def run(self, client: Client = None) -> None:
        """
        Run the parallel processing of areas using a Dask cluster initialized with dask-mpi.
        
        This assumes that your job was launched via an MPI launcher (e.g., mpirun or srun)
        and that dask_mpi.initialize() will start the scheduler and workers across all allocated nodes.

        Parameters
        ----------
        client : dask.distributed.Client, optional
            A Dask client to use for processing. When using dask-mpi, the client should be created with 
            `dask_mpi.initialize()`.  
            If None, a new client will be created.

        """
        self.logger.info("Starting processing with MPIDaskProcessor (using dask-mpi).")

        # Use the provided client or create a new one
        if client is None:
            client = Client()

        self.logger.info(f"Dask dashboard address: {client.dashboard_link}")
        
        # Split areas into batches
        area_batches = np.array_split(self.areas, max(1, len(self.areas) // self.batch_size))
        self.logger.info(f"Processing {len(self.areas)} areas in {len(area_batches)} batches.")

        total_areas = len(self.areas)
        area_success = {area.id: 0 for area in self.areas}  # Track success count per area
        total_stgrids = len(self.stgrid)
        processed_areas = 0

        for i, batch in enumerate(area_batches, start=1):
            self.logger.info(f"Processing batch {i}/{len(area_batches)} with {len(batch)} areas.")

            for n_stgrid, stgrid in enumerate(self.stgrid, start=1):
                try:
                    # First check which areas need processing
                    if self.skip_exist:
                        areas_to_process = []
                        for area in batch:
                            if check_area_needs_processing(area, n_stgrid, total_stgrids, 
                                                            self.skip_exist, self.save_nc, self.save_csv):
                                areas_to_process.append(area)
                            else:
                                area_success[area.id] += 1
                                if area_success[area.id] == total_stgrids:
                                    processed_areas += 1
                                    self.logger.info(f"[{processed_areas}/{total_areas}]: {area.id} --- Already processed.")

                        if not areas_to_process:
                            self.logger.info(f"Skipping batch {i}, stgrid {n_stgrid} - all areas already processed.")
                            continue
                    else:
                        # No skipping, process all areas in the batch
                        areas_to_process = batch

                    # Pre-clip individually for each area.
                    area_stgrids = {
                        area.id: stgrid.rio.clip(
                            area.geometry.geometry.to_crs(stgrid.rio.crs),
                            all_touched=True
                        ).persist()
                        for area in areas_to_process
                    }

                    # Create tasks using area-specific pre-clipped grids.
                    tasks = [
                        delayed(process_area)(
                            area,
                            area_stgrids[area.id],
                            self.variable,
                            self.method,
                            self.operations,
                            self.skip_exist,
                            n_stgrid,
                            total_stgrids,
                            self.save_nc,
                            self.save_csv,
                            dask_key_name=f"{area.id}_{n_stgrid}"
                        ) for area in areas_to_process
                    ]

                    futures = client.compute(tasks)

                    for future in as_completed(futures):
                        area_id = future.key.split('_')[0]
                        try:
                            result = future.result()
                            if isinstance(result, pd.DataFrame):
                                area_success[area_id] += 1
                                try:
                                    area_stgrids[area_id].close()
                                except Exception:
                                    pass
                                if area_success[area_id] == total_stgrids:
                                    processed_areas += 1
                                    self.logger.info(f"[{processed_areas}/{total_areas}]: {area_id} --- Processing completed.")
                        except Exception as e:
                            self.logger.error(f"{area_id}, stgrid {n_stgrid} --- Error occurred: {e}")

                    client.cancel(futures)
                    for grid in area_stgrids.values():
                        if hasattr(grid, 'close'):
                            try:
                                grid.close()
                            except Exception:
                                pass
                        client.cancel(grid)
                    del area_stgrids, tasks, futures
                    gc.collect()

                except Exception as e:
                    self.logger.error(f"Error during batch {i}, stgrid {n_stgrid}: {e}")

            # For dask-mpi, it might be simpler to run all batches in one go and not restart the client between batches.
            self.logger.info(f"Finished batch {i}/{len(area_batches)}.")

        successful_areas = sum(1 for count in area_success.values() if count == total_stgrids)
        self.logger.info(f"Processing completed: {successful_areas}/{total_areas} areas processed successfully.")
        client.close()