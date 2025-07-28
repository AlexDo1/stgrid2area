import os
from pathlib import Path
from typing import Union
import warnings

import geopandas as gpd
import pandas as pd
import xarray as xr
from exactextract import exact_extract


class Area():
    def __init__(self, geometry: Union[gpd.GeoDataFrame, gpd.GeoSeries], id: str, output_dir: str):
        """
        Initialize an Area object.

        Parameters
        ----------
        id : str
            The unique identifier of the area.
        geometry : gpd.GeoDataFrame
            The GeoDataFrame containing the geometry of the area.
        output_dir : str
            The output directory where results will be saved.  
            Will always be a subdirectory of this directory, named after the area's id.

        """
        self.id = str(id)

        # Check if the geometry is a GeoDataFrame
        if isinstance(geometry, gpd.GeoDataFrame):
            self.geometry = geometry
        else:
            raise TypeError(f"{self.id}: The geometry must be a GeoDataFrame.")
        
        # Make output_dir a Path
        output_dir = Path(output_dir)

        # Set the output path of the area: output_dir/id
        self.output_path = output_dir / self.id

    def __repr__(self):
        return f"Area(id={self.id})"
    
    def __str__(self):
        return f"Area with id {self.id}"

    def clip(self, stgrid: Union[xr.Dataset, xr.DataArray], all_touched: bool = True, save_result: bool = False, skip_exist: bool = False, filename: str = None) -> xr.Dataset:
        """
        Clip the spatiotemporal grid to the area's geometry.

        Parameters
        ----------
        stgrid : xr.Dataset
            The spatiotemporal grid to clip.
        all_touched : bool, optional
            If True, all pixels that are at least partially in the catchment are returned.  
            If False, only pixels whose center is within the polygon or that are selected by Bresenham's line algorithm are selected.  
            Note that you should set `all_touched=True` if you want to calculate weighted statistics with the `aggregate` method later.  
            The default is True, as the aggregation uses exact_extract by default.
        save_result : bool, optional
            If True, the clipped grid will be saved to the output directory of the area.  
            The default is False.
        skip_exist : bool, optional
            If True, the clipping will be skipped if the clipped grid already exists. 
            In this case, the existing clipped grid will be returned.
            The default is False.
        filename : str, optional
            The filename of the clipped grid when written to disk.  
            If None, the filename will be the id of the area + "_clipped.nc".  
            The default is None.

        Returns
        -------
        xr.Dataset
            The clipped spatiotemporal grid.

        """
        # Parse the filename
        if filename is None:
            filename = f"{self.id}_clipped.nc"
        elif not filename.endswith(".nc"):
            filename = f"{filename}.nc"

        # Check if the clipping should be skipped if the clipped grid already exists
        if skip_exist and (self.output_path / filename).exists():
            return xr.open_dataset(self.output_path / filename)
        
        # Check if the stgrid is a xarray Dataset or DataArray
        if not isinstance(stgrid, (xr.Dataset, xr.DataArray)):
            raise TypeError(f"{self.id}: The stgrid must be a xarray Dataset or DataArray.")
        
        # Set the crs of the geometry to the crs of the stgrid
        geometry = self.geometry.to_crs(stgrid.rio.crs.to_string())

        # Clip the stgrid to the geometry, all_touched=True to get all pixels that are at least partially in the catchment
        clipped = stgrid.rio.clip(geometry.geometry, all_touched=all_touched)

        # Save the clipped grid to the output directory of the area
        if save_result:
            # Create the output directory if it does not exist
            self.output_path.mkdir(parents=True, exist_ok=True)
            
            try:
                # Save the clipped grid to the output directory
                clipped.to_netcdf(self.output_path / filename, engine="h5netcdf")
            # xarray PermissionError: delete existing file before saving
            except PermissionError:
                os.remove(self.output_path / filename)
                clipped.to_netcdf(self.output_path / filename, engine="h5netcdf")
        
        return clipped
    
    def aggregate(self, stgrid: xr.Dataset, variables: Union[str, list[str]], method: str, operations: list[str], save_result: bool = False, skip_exist: bool = False, filename: str = None) -> pd.DataFrame:
        """
        Wrapper function to aggregate the spatiotemporal grid to the area's geometry using either the exact_extract or xarray method.  
        Usually, you first perform the `clip` and then aggregate the clipped stgrid. Using the clipped
        raster data also results in much faster aggregation.  
        The aggregation is spatially (e.g. the spatial mean), so the time dimension is preserved and
        the result is a time series DataFrame with the same time dimension as the input grid.  
        Depending on the method, the aggregation is done using exact_extract or xarray. The exact_extract
        method yields weighted spatial statistics based on the fraction of the pixel that is covered by the geometry,
        while the xarray method yields unweighted spatial statistics based on the pixel values. As weighted
        statistics are only possible with spatial dimensions >= (2, 2), the exact_extract method will raise an error
        if the spatial dimensions are 1 in at least one direction. In this case, the xarray method can be used as a fallback.

        Parameters
        ----------
        stgrid : xr.Dataset
            The spatiotemporal grid to aggregate. Must be a xr.Dataset.
        variables : str or list[str]
            The variable(s) in stgrid to aggregate. Can be a single variable name (str) or a list of variable names.
        method : str
            The method to use for aggregation.  
            Can be "exact_extract", "xarray".
        operations : list[str]
            The operations to use for aggregation.
            Can be "mean", "min", "median", "max", "stdev", "quantile(q=0.XX)" and all other operations that are 
            supported by the [exact_extract](https://github.com/isciences/exactextract) package.
        save_result : bool, optional
            If True, the aggregated timeseries will be saved to the output directory of the area.  
            The default is False.
        skip_exist : bool, optional
            If True, the aggregation will be skipped if the aggregated timeseries already exists.  
            In this case, the existing timeseries will be returned.  
            The default is False.
        filename : str, optional
            The filename of the aggregated results when written to disk.  
            If None, the filename will be the id of the area + "_aggregated.csv". 
            The default is None.

        Returns
        -------
        pd.DataFrame
            The aggregated spatiotemporal grid with columns for each variable and operation combination.

        """
        # Convert single variable to list for consistent processing
        if isinstance(variables, str):
            variables = [variables]
        
        # Parse the filename
        if filename is None:
            filename = f"{self.id}_aggregated.csv"
        elif not filename.endswith(".csv"):
            filename = f"{filename}.csv"

        # Check if the aggregation should be skipped if the aggregated results already exist
        if skip_exist and (self.output_path / filename).exists():
            return pd.read_csv(self.output_path / filename, index_col="time")
        
        # Check if the stgrid is a xarray Dataset
        if not isinstance(stgrid, xr.Dataset):
            raise TypeError(f"{self.id}: The stgrid must be a xarray Dataset.")
        
        # Validate that all variables exist in the dataset
        missing_vars = [var for var in variables if var not in stgrid.data_vars]
        if missing_vars:
            raise ValueError(f"{self.id}: Variables {missing_vars} not found in the dataset. Available variables: {list(stgrid.data_vars)}")

        # Check if operations is a list
        if not isinstance(operations, list):
            operations = [operations]

        # Load the clipped grid, as this yielded the fastest computation times
        stgrid = stgrid.load()

        if method == "exact_extract":
            df_timeseries = self._aggregate_exact_extract(stgrid, variables, operations)
        elif method == "xarray":
            df_timeseries = self._aggregate_xarray(stgrid, variables, operations)
        else:
            raise ValueError(f"{self.id}: The method {method} is not supported. Use 'exact_extract' or 'xarray'.")
    
        if save_result:
            # Create the output directory if it does not exist
            self.output_path.mkdir(parents=True, exist_ok=True)

            # Save the aggregated timeseries to the output directory
            df_timeseries.to_csv(self.output_path / filename, index_label="time")

        return df_timeseries

    def _aggregate_exact_extract(self, stgrid: xr.Dataset, variables: list[str], operations: list[str]) -> pd.DataFrame:
        """
        Aggregate the spatiotemporal grid to the area's geometry.  
        Usually, you first perform the `clip` and then aggregate the clipped stgrid. Using the clipped  
        raster data also results in much faster aggregation.  
        The aggregation is spatially (e.g. the spatial mean), so the time dimension is preserved and 
        the result is a time series DataFrame with the same time dimension as the input grid.  
        This method uses the exact_extract package for the aggregation, which yields weighted spatial
        statistics based on the fraction of the pixel that is covered by the geometry.

        Parameters
        ----------
        stgrid : xr.Dataset
            The spatiotemporal grid to aggregate. Must be a xr.Dataset.
        variables : list[str]
            The variables in stgrid to aggregate.
        operations : list[str]
            The operations to use for aggregation.  
            Can be "mean", "min", "median", "max", "stdev", "quantile(q=0.XX)" and all other operations that are 
            supported by the [exact_extract](https://github.com/isciences/exactextract) package.

        Returns
        -------
        pd.DataFrame
            The aggregated spatiotemporal grid with columns for each variable and operation combination.
        """
        # Set the crs of the geometry to the crs of the stgrid
        geometry = self.geometry.to_crs(stgrid.rio.crs.to_string())
        
        # Store results for all variables
        all_dfs = []
        
        # Process each variable
        for variable in variables:
            # Get the DataArray for this variable
            stgrid_var = stgrid[variable]
            
            # Check dimensionality of gridded data, calculating weighted statistics with exactaxtract can only be done if shape is >= (2, 2)
            if 1 in stgrid_var.isel(time=0).shape:
                raise ValueError(f"{self.id}: Gridded data for variable '{variable}' has spatial dimensionality of 1 in at least one direction, aggregation for 1-D data is not possible with the exact_extract method, you can use method='xarray' instead.")
            
            # Aggregate the clipped grid to the geometry using exact_extract
            with warnings.catch_warnings():
                # Suppress the warning that the spatial reference system of the input features does not exactly match the raster
                warnings.filterwarnings("ignore", message="Spatial reference system of input features does not exactly match raster.")
                
                try:
                    df = exact_extract(stgrid_var, geometry, operations, output="pandas")
                    
                    # Transpose dataframe
                    df = df.T
                    
                    # Get the time index from the xarray dataset
                    time_index = stgrid_var.time.values
                    
                    # Create a list of dataframes, each dataframe contains the timeseries for one statistic
                    sliced_dfs = [df.iloc[i:i+len(time_index)].copy() for i in range(0, len(df), len(time_index))]
                    
                    # Set the index to the time values and rename the columns
                    for i, df_slice in enumerate(sliced_dfs):
                        df_slice.index = time_index
                        df_slice.columns = [f"{variable}_{operations[i]}"]
                    
                        # Replace quantile column names to not include brackets, equal sign and points
                        for col in df_slice.columns:
                            if "quantile" in col:
                                # get the quantile value
                                q = int(float(col.split('=')[1].split(')')[0]) * 100)
                                
                                # replace the column name
                                df_slice.rename(columns={col: f"{variable}_quantile{q}"}, inplace=True)
                    
                    # Concatenate the dataframes for this variable's operations
                    var_df = pd.concat(sliced_dfs, axis=1)
                    all_dfs.append(var_df)
                
                except Exception as e:
                    raise ValueError(f"{self.id}: Error processing variable '{variable}': {str(e)}")
        
        # Ensure we have at least one successful variable processed
        if not all_dfs:
            raise ValueError(f"{self.id}: No variables were successfully processed")
        
        # Concatenate all variables' dataframes
        df_timeseries = pd.concat(all_dfs, axis=1)
        
        # Label the index
        df_timeseries.index.name = "time"
        
        return df_timeseries
    
    def _aggregate_xarray(self, stgrid: xr.Dataset, variables: list[str], operations: list[str]) -> pd.DataFrame:
        """
        Aggregate the spatiotemporal grid to the area's geometry.  
        Usually, you first perform the `clip` and then aggregate the clipped stgrid. Using the clipped  
        raster data also results in much faster aggregation.  
        The aggregation is spatially (e.g. the spatial mean), so the time dimension is preserved and 
        the result is a time series DataFrame with the same time dimension as the input grid.  
        This method uses xarray directly for the aggregation, which yields unweighted spatial
        statistics based on the pixel values.

        Parameters
        ----------
        stgrid : xr.Dataset
            The spatiotemporal grid to aggregate.
        variables : list[str]
            The variables in stgrid to aggregate.
        operations : list[str]
            The operations to use for aggregation.  
            Can be "mean", "min", "median", "max", "stdev", "quantile(q=0.XX)" and all other operations that are 
            supported by xarray.

        Returns
        -------
        pd.DataFrame
            The aggregated spatiotemporal grid with columns for each variable and operation combination.

        """
        # Infer the spatial dimensions of the grid based on common dimension names TODO: improve this / spatial dims as input?
        spatial_dims = [dim for dim in stgrid.dims if dim in ["latitude", "longitude", "lat", "lon", "x", "y", "X", "Y"]]        

        # Store results for all variables
        all_dfs = []

        # Process each variable
        for variable in variables:
            # Get the DataArray for this variable
            stgrid_var = stgrid[variable]

            # Store dataframes for each operation for this variable
            variable_dfs = []
            
            try:
                # Apply each operation to this variable
                for operation in operations:
                    if operation == "mean":
                        result = stgrid_var.mean(dim=spatial_dims)
                    elif operation == "min":
                        result = stgrid_var.min(dim=spatial_dims)
                    elif operation == "median":
                        result = stgrid_var.median(dim=spatial_dims)
                    elif operation == "max":
                        result = stgrid_var.max(dim=spatial_dims)
                    elif operation == "stdev":
                        result = stgrid_var.std(dim=spatial_dims)
                    elif "quantile" in operation:
                        q = float(operation.split("=")[1].split(")")[0])
                        result = stgrid_var.quantile(q, dim=spatial_dims)
                    else:
                        raise ValueError(f"{self.id}: The operation {operation} is not supported.")
                    
                    # Compute the result
                    result = result.compute()
                    
                    # Convert the result to a DataFrame
                    df = result.to_dataframe()
                    
                    # Format column name
                    if "quantile" in operation:
                        q = float(operation.split("=")[1].split(")")[0])
                        column_name = f"{variable}_quantile{int(q*100)}"
                    else:
                        column_name = f"{variable}_{operation}"
                    
                    # Select only the variable column and rename it
                    df = df[[variable]].rename(columns={variable: column_name})
                    
                    # Append to list of dataframes for this variable
                    variable_dfs.append(df)
                
                # Concatenate all operations for this variable
                if variable_dfs:
                    var_df = pd.concat(variable_dfs, axis=1)
                    all_dfs.append(var_df)
            except Exception as e:
                raise ValueError(f"{self.id}: Error processing variable '{variable}': {str(e)}")

        # Ensure we have at least one successful variable processed
        if not all_dfs:
            raise ValueError(f"{self.id}: No variables were successfully processed")
            
        # Concatenate all variables
        df_timeseries = pd.concat(all_dfs, axis=1)

        # Label the index
        df_timeseries.index.name = "time"
        
        return df_timeseries