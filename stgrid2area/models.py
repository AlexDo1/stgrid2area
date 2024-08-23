import geopandas as gpd
import xarray as xr
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

    def clip(self, stgrid: Union[xr.Dataset, xr.DataArray], all_touched: bool = False, save_result: bool = False) -> xr.Dataset:
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
            The default is False.
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