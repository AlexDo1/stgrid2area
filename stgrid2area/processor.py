import os
from typing import Union

from dask import delayed, compute
import pandas as pd
import xarray as xr
import rioxarray

from .area import Area

class ParallelDaskProcessor:
    def __init__(self, areas: list[Area], stgrid: Union[xr.Dataset, xr.DataArray], variable: Union[str, None], operations: list[str], n_workers: int = None, skip_exist: bool = False):
        """
        Initialize a ParallelProcessor object.

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
        
        """
        self.areas = areas
        self.stgrid = stgrid
        self.variable = variable
        self.operations = operations
        self.n_workers = n_workers if n_workers is not None else os.cpu_count()
        self.skip_exist = skip_exist

        # Check if variable is provided when stgrid is an xr.Dataset
        if isinstance(stgrid, xr.Dataset) and variable is None:
            raise ValueError("The variable must be defined if stgrid is an xr.Dataset.")

    def clip_and_aggregate(self, area: Area) -> pd.DataFrame:
        """
        Process an area by clipping the spatiotemporal grid to the area and aggregating the variable.  
        When clipping the grid, the all_touched parameter is set to True, as the variable is aggregated with
        the exact_extract method, which requires all pixels that are partially in the area.
        The clipped grid and the aggregated variable are saved in the output directory of the area.  
                
        Parameters
        ----------
        area : Area
            The area to process.
        variable : str
            The variable to aggregate.
        
        Returns
        -------
        pd.DataFrame
            The aggregated variable.
        
        """
        # clip the spatiotemporal grid to the area
        clipped = area.clip(self.stgrid, save_result=True)

        # check if clipped is a xarray Dataset or DataArray
        if isinstance(clipped, xr.Dataset):
            return area.aggregate(clipped[self.variable], self.operations, save_result=True, skip_exist=self.skip_exist)
        elif isinstance(clipped, xr.DataArray):
            return area.aggregate(clipped, self.operations, save_result=True, skip_exist=self.skip_exist)

    def run(self) -> None:
        """
        Run the parallel processing of the areas.
        Results are saved in the output directories of the areas.
        
        """
        tasks = [delayed(self.clip_and_aggregate)(area) for area in self.areas]
        compute(*tasks, scheduler='processes', num_workers=self.n_workers)