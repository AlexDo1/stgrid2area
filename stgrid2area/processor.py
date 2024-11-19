import os
from typing import Union, List
from dask import delayed
from dask.distributed import Client, LocalCluster, as_completed
import pandas as pd
import xarray as xr
import rioxarray
import logging
from pathlib import Path
import numpy as np

from .area import Area


class LocalDaskProcessor:
    def __init__(self, areas: List[Area], stgrid: xr.Dataset, variable: str, method: str, operations: List[str], n_workers: int = None, skip_exist: bool = False, batch_size: int = None, logger: logging.Logger = None):
        """
        Initialize a LocalDaskProcessor for efficient parallel processing on a single machine.

        Parameters
        ----------
        areas : list of Area
            List of area objects to process.
        stgrid : xr.Dataset or xr.DataArray
            The spatiotemporal data to process.
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
        logger : logging.Logger, optional
            Logger to use for logging. If None, a basic logger will be set up.

        """
        self.areas = areas
        self.stgrid = stgrid
        self.variable = variable
        self.method = method
        self.operations = operations
        self.n_workers = n_workers or os.cpu_count()
        self.skip_exist = skip_exist
        self.logger = logger
        self.batch_size = batch_size or len(areas)  # Default: process all areas at once

        # Set up basic logging if no handler is configured
        if not self.logger:
            self.logger = logging.getLogger(__name__)
            self.logger.setLevel(logging.INFO)
            self.logger.addHandler(logging.StreamHandler())

    def clip_and_aggregate(self, area: Area, stgrid: xr.Dataset) -> Union[pd.DataFrame, Exception]:
        """
        Process an area by clipping the spatiotemporal grid to the area and aggregating the variable.  
        When clipping the grid, the all_touched parameter is set to True, as the variable is aggregated with
        the exact_extract method, which requires all pixels that are partially in the area.
        The clipped grid and the aggregated variable are saved in the output directory of the area.  
                
        Parameters
        ----------
        area : Area
            The area to process.

        
        Returns
        -------
        pd.DataFrame or None
            The aggregated variable, or None if an error occurred.
        
        """
        # Clip the spatiotemporal grid to the area
        clipped = area.clip(stgrid, save_result=True)

        # Aggregate the variable
        if self.method in ["exact_extract", "xarray"]:
            return area.aggregate(clipped, self.variable, self.method, self.operations, save_result=True, skip_exist=self.skip_exist)
        elif self.method == "fallback_xarray":
            try:
                return area.aggregate(clipped, self.variable, "exact_extract", self.operations, save_result=True, skip_exist=self.skip_exist)
            except ValueError:
                self.logger.warning(f"Method 'exact_extract' failed for area {area.id}. Falling back to 'xarray' method.")
                # add a file "fallback_xarray" to the output directory to indicate that the fallback method was used
                Path(area.output_path, "fallback_xarray").touch()
                return area.aggregate(clipped, self.variable, "xarray", self.operations, save_result=True, skip_exist=self.skip_exist)
        else:
            raise ValueError("Invalid method. Use 'exact_extract', 'xarray' or 'fallback_xarray'.")
        
    def run(self) -> None:
        """
        Run the parallel processing of areas using Dask with batching.

        """
        self.logger.info("Starting processing with LocalDaskProcessor.")
        
        with LocalCluster(n_workers=self.n_workers, threads_per_worker=1) as cluster:
            with Client(cluster) as client:
                try:
                    # Log the Dask dashboard address
                    self.logger.info(f"Dask dashboard address: {client.dashboard_link}")

                    # Split areas into batches
                    area_batches = np.array_split(self.areas, max(1, len(self.areas) // self.batch_size))
                    self.logger.info(f"Processing {len(self.areas)} areas in {len(area_batches)} batches.")

                    total_areas = len(self.areas)
                    counter, success = 0, 0

                    for i, batch in enumerate(area_batches, start=1):
                        self.logger.info(f"Processing batch {i}/{len(area_batches)} with {len(batch)} areas.")

                        # Pre-clip the stgrid to the area batch before persisting to be memory efficient
                        stgrid_chunk = self.stgrid.rio.clip(pd.concat([area.geometry for area in batch]).geometry.to_crs(self.stgrid.rio.crs), all_touched=True).persist()
                        
                        # Create tasks for this batch
                        tasks = [delayed(self.clip_and_aggregate)(area, stgrid_chunk, dask_key_name=f"{area.id}") for area in batch]

                        # Compute the tasks for this batch
                        futures = client.compute(tasks)
                        
                        for future in as_completed(futures):
                            try:
                                result = future.result()
                                if isinstance(result, pd.DataFrame):
                                    success += 1
                                    self.logger.info(f"[{success}/{total_areas}]: {future.key} --- Processing completed.")
                            except Exception as e:
                                self.logger.error(f"[{counter + 1}/{total_areas}]: {future.key} --- Error occurred: {e}")
                            finally:
                                counter += 1
                            
                        # cleanup memory
                        del stgrid_chunk

                    self.logger.info(f"Processing completed: {success}/{total_areas} areas processed successfully.")
                finally:
                    self.logger.info("Shutting down Dask client and cluster.")


class DistributedDaskProcessor:
    def __init__(self, areas: List[Area], stgrid: Union[xr.Dataset, xr.DataArray], variable: Union[str, None], operations: List[str], n_workers: int = None, skip_exist: bool = False, log_file: str = None, log_level: str = "INFO"):
        """
        Initialize a DistributedDaskProcessor object.

        Deprecation Warning
        -------
        This processor class was developed for use in a HPC environment, development has been discontinued and it is recommended to use the LocalDaskProcessor class for local processing.


        Parameters
        ----------
        areas : list[Area]
            The list of areas to process.
        stgrid : Union[xr.Dataset, xr.DataArray]
            The spatiotemporal grid to clip to the areas.
        variable : Union[str, None]
            The variable in st_grid to aggregate temporally. Required if stgrid is an xr.Dataset.
        operations : list[str]
            The list of operations to aggregate the variable.
        n_workers : int, optional
            The number of workers to use for parallel processing.  
            If None, the number of workers will be set to the number of CPUs on the machine.
        skip_exist : bool, optional
            If True, skip processing areas that already have clipped grids or aggregated variables in their output directories. 
            If False, process all areas regardless of whether they already have clipped grids or aggregated variables.
        log_file : str, optional
            The path to save the log file. If None, the log will be printed to the console.
        log_level : str, optional
            The logging level to use for the processing. Use 'DEBUG' for more detailed error messages.
        
        """
        self.areas = areas
        self.stgrid = stgrid
        self.variable = variable
        self.operations = operations
        self.n_workers = n_workers if n_workers is not None else os.cpu_count()
        self.skip_exist = skip_exist
        self.log_file = Path(log_file) if log_file else None
        self.log_level = log_level

        # Check if variable is provided when stgrid is an xr.Dataset
        if isinstance(stgrid, xr.Dataset) and variable is None:
            raise ValueError("The variable must be defined if stgrid is an xr.Dataset.")
        
    def configure_logging(self) -> None:
        """
        Configure logging dynamically based on log_file.  
        Note that you have to restart your local kernel if you want to change logging from file to console or vice versa.  
        Also note that Dask logging is not captured by this configuration, Dask logs are printed to the console.
        
        """
        # Set up the new log handler (either file or stream)
        if self.log_file:
            # Create log file path if it does not exist
            self.log_file.parent.mkdir(parents=True, exist_ok=True)
            log_handler = logging.FileHandler(self.log_file)
        else:
            log_handler = logging.StreamHandler()

        # Set up the log format
        log_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

        # Add the new handler
        logging.getLogger().addHandler(log_handler)

        # Set the logging level
        logging.getLogger().setLevel(self.log_level)

    def clip_and_aggregate(self, area: Area) -> Union[pd.DataFrame, Exception]:
        """
        Process an area by clipping the spatiotemporal grid to the area and aggregating the variable.  
        When clipping the grid, the all_touched parameter is set to True, as the variable is aggregated with
        the exact_extract method, which requires all pixels that are partially in the area.
        The clipped grid and the aggregated variable are saved in the output directory of the area.  
                
        Parameters
        ----------
        area : Area
            The area to process.
        
        Returns
        -------
        pd.DataFrame or None
            The aggregated variable, or None if an error occurred.
        
        """
        # Clip the spatiotemporal grid to the area
        clipped = area.clip(self.stgrid, save_result=True)

        # Check if clipped is a xarray Dataset or DataArray
        if isinstance(clipped, xr.Dataset):
            return area.aggregate(clipped[self.variable], self.operations, save_result=True, skip_exist=self.skip_exist)
        elif isinstance(clipped, xr.DataArray):
            return area.aggregate(clipped, self.operations, save_result=True, skip_exist=self.skip_exist)

    def run(self, client: Client = None) -> None:
        """
        Run the parallel processing of the areas using the distributed scheduler.
        Results are saved in the output directories of the areas.

        Parameters
        ----------
        client : dask.distributed.Client, optional
            The Dask client to use for parallel processing. If None, a local client will be created.  
            For HPC clusters, the client should be created with the appropriate configuration.

            Example using a SLURMCluster:
            ```python
            from jobqueue_features import SLURMCluster
            from dask.distributed import Client
            
            cluster = SLURMCluster(
                queue='your_queue',
                project='your_project',
                cores=24,
                memory='32GB',
                walltime='02:00:00'
            )

            client = Client(cluster)
            ```

            Example using MPI:
            ```python
            from dask.distributed import Client
            from dask_mpi import initialize

            initialize()

            client = Client()
            ```
        
        """
        success = 0

        # Configure logging
        self.configure_logging()

        # Use the passed client or create a local one
        client = client or Client(LocalCluster(n_workers=self.n_workers, threads_per_worker=1, dashboard_address=':8787'))
        
        # Log the Dask dashboard address
        logging.info(f"Dask dashboard address: {client.dashboard_link}")

        # Process the areas in parallel and keep track of futures
        tasks = [delayed(self.clip_and_aggregate)(area, dask_key_name=f"{area.id}") for area in self.areas]
        futures = client.compute(tasks)
        
        # Wait for the tasks to complete
        for future in as_completed(futures):
            try:
                # Get the result of the task
                result = future.result()
                if isinstance(result, pd.DataFrame):
                    logging.info(f"{future.key} --- Processing completed")
                    success += 1
            except Exception as e:
                if logging.getLogger().level == logging.DEBUG:
                    logging.exception(f"{future.key} --- An error occurred: {e}")
                else:
                    logging.error(f"{future.key} --- An error occurred: {e}")

        client.close()

        logging.info(f"Processing completed and was successful for [{success} / {len(self.areas)}] areas" if self.log_file else "Processing completed.")