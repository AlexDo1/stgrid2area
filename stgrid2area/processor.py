import os
from typing import Union
import gc
from dask import delayed
from dask.distributed import Client, LocalCluster, as_completed
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
                Path(area.output_path, "fallback_xarray").touch()
                result = area.aggregate(clipped, variable, "xarray", operations, 
                                        save_result=save_csv, skip_exist=skip_exist, filename=filename_aggr)
        return result
    except Exception as e:
        raise e
    finally:
        clipped.close()
        stgrid.close()        

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
                            try:
                                # Pre-clip individually for each area
                                area_stgrids = {
                                    area.id: stgrid.rio.clip(
                                        area.geometry.geometry.to_crs(stgrid.rio.crs), 
                                        all_touched=True
                                    ).persist() 
                                    for area in batch
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
                                    ) for area in batch
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
                        self.logger.info(f"Finished batch {i}/{len(area_batches)}. Restarted Dask client and cluster for the next batch.\n")

                    # Final summary
                    successful_areas = sum(1 for count in area_success.values() if count == total_stgrids)
                    self.logger.info(f"Processing completed: {successful_areas}/{total_areas} areas processed successfully.")
                except Exception as e:
                    self.logger.error(f"An error occurred: {e}")
                finally:
                    self.logger.info("Shutting down Dask client and cluster.")



class DistributedDaskProcessor:
    def __init__(self, areas: list[Area], stgrid: Union[xr.Dataset, xr.DataArray], variable: Union[str, None], operations: list[str], n_workers: int = None, skip_exist: bool = False, log_file: str = None, log_level: str = "INFO"):
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