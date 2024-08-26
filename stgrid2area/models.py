import geopandas as gpd
import xarray as xr
import pandas as pd
from typing import Union
from exactextract import exact_extract
from pathlib import Path


class Area():
    def __init__(self, geometry: gpd.GeoDataFrame, id: str, output_dir: str):
        """
        Initialize an Area object.

        Parameters
        ----------
        id : str
            The unique identifier of the area.
        geometry : gpd.GeoDataFrame
            The geometry of the area.
        output_dir : str
            The output directory where results will be saved.  
            Will always be a subdirectory of this directory, named after the area's id.

        """
        self.id = str(id)

        # Check if the geometry is a GeoDataFrame
        if isinstance(geometry, gpd.GeoDataFrame):
            self.geometry = geometry
        else:
            raise TypeError("The geometry must be a GeoDataFrame.")
        
        # Make output_dir a Path
        output_dir = Path(output_dir)

        # Set the output path of the area: output_dir/id
        self.output_path = output_dir / self.id

    @property
    def has_clip(self) -> bool:
        """
        Check if the area already has a clipped grid in the output path.
        
        Returns
        -------
        bool
            True if the area has a clipped grid, False otherwise.

        """
        return (self.output_path / f"{self.id}_clipped.nc").exists()

    def clip(self, stgrid: Union[xr.Dataset, xr.DataArray], all_touched: bool = True, save_result: bool = False) -> xr.Dataset:
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

        Returns
        -------
        xr.Dataset
            The clipped spatiotemporal grid.

        """
        # Check if the stgrid is a xarray Dataset or DataArray
        if not isinstance(stgrid, (xr.Dataset, xr.DataArray)):
            raise TypeError("The stgrid must be a xarray Dataset or DataArray.")
        
        # Set the crs of the geometry to the crs of the stgrid
        geometry = self.geometry.to_crs(stgrid.rio.crs)

        # Clip the stgrid to the geometry, all_touched=True to get all pixels that are at least partially in the catchment
        clipped = stgrid.rio.clip(geometry.geometry, all_touched=all_touched)

        # Save the clipped grid to the output directory of the area
        if save_result:
            # Create the output directory if it does not exist
            self.output_path.mkdir(parents=True, exist_ok=True)

            # Save the clipped grid to the output directory
            clipped.to_netcdf(self.output_path / f"{self.id}_clipped.nc")
        
        return clipped
    
    def aggregate(self, stgrid:  xr.DataArray, operations: list[str], save_result: bool = False) -> pd.DataFrame:
        """
        Aggregate the spatiotemporal grid to the area's geometry.  
        Usually, you first perform the `clip` and then aggregate the clipped stgrid. Using the clipped  
        raster data also results in much faster aggregation.  
        The aggregation is spatially (e.g. the spatial mean), so the time dimension is preserved and 
        the result is a time series DataFrame with the same time dimension as the input grid.

        Parameters
        ----------
        stgrid : xr.DataArray
            The spatiotemporal grid to aggregate. Must be a xr.DataArray, as only one variable  
            can be aggregated.
        operations : list[str]
            The operations to use for aggregation.  
            Can be "mean", "min", "median", "max", "stdev", "quantile(q=0.XX)" and all other operations that are 
            supported by the [exact_extract](https://github.com/isciences/exactextract) package.
            The default is "mean".
        save_result : bool, optional
            If True, the aggregated grid will be saved to the output directory of the area.  
            The default is False.

        Returns
        -------
        pd.DataFrame
            The aggregated spatiotemporal grid.

        """
        # Check dimensionality of gridded data, calculating weighted statistics with exactaxtract can only be done if shape is >= (2, 2)
        if 1 in stgrid.isel(time=0).shape:
            raise NotImplementedError("Gridded data has spatial dimensionality of 1 in at least one direction, aggregation for 1-D data is not supported at the moment.")

        # Check if the stgrid is a xarray Dataset or DataArray
        if not isinstance(stgrid, xr.DataArray):
            raise TypeError("The stgrid must be a xarray DataArray.")
        
        # Check if operations is a list
        if not isinstance(operations, list):
            operations = [operations]

        # Set the crs of the geometry to the crs of the stgrid
        geometry = self.geometry.to_crs(stgrid.rio.crs)

        # Aggregate the clipped grid to the geometry
        df = exact_extract(stgrid, geometry, operations, output="pandas")

        # Transpose dataframe
        df = df.T

        # Get the time index from the xarray dataset
        time_index = stgrid.time.values
        
        # Create a list of dataframes, each dataframe contains the timeseries for one statistic
        sliced_dfs = [df.iloc[i:i+len(time_index)] for i in range(0, len(df), len(time_index))]

        # Set the index to the time values and rename the columns
        for i, df in enumerate(sliced_dfs):
            df.index = time_index
            df.columns = [f"{stgrid.name}_{operations[i]}"]
        
            # Replace quantile column names to not include brackets, equal sign and points
            for col in df.columns:
                if "quantile" in col:
                    # get the quantile value
                    q = int(float(col.split('=')[1].split(')')[0]) * 100)

                    # replace the column name
                    df.rename(columns={col: f"{stgrid.name}_quantile{q}"}, inplace=True)                

        # Concatenate the dataframes
        df_timeseries = pd.concat(sliced_dfs, axis=1)

        # Save the aggregated grid to the output directory of the area
        if save_result:
             # Create the output directory if it does not exist
            self.output_path.mkdir(parents=True, exist_ok=True)

            # Save the aggregated timeseries to the output directory
            df_timeseries.to_csv(self.output_path / f"{self.id}_aggregated.csv", index_label="time")
        
        return df_timeseries