from .area import Area

import geopandas as gpd


def geodataframe_to_areas(areas: gpd.GeoDataFrame, id_column: str, output_dir: str, sort_by_proximity: bool = True, sort_by_area: bool = True) -> list[Area]:
    """
    Convert a GeoDataFrame of areas to a list of Area objects to be used as input for the ParallelProcessor.

    Parameters
    ----------
    areas : gpd.GeoDataFrame
        The GeoDataFrame of areas.
    id_column : str
        The name of the column in the GeoDataFrame that contains the unique identifier for each area.
    output_dir : str
        The output directory where results will be saved.  
        Will always be a subdirectory of this directory, named after the area's id.
    sort_by_proximity : bool, optional
        Whether to sort the areas by proximity. 
        Default is True.
        This is especially useful when using the ParallelProcessor with batches of areas, as this makes
        sure that batched areas are close to each other, which is more efficient, as a smaller portion of the 
        stgrid will be loaded into memory.
    sort_by_area : bool, optional
        Whether to sort the areas by area size. 
        Default is True.
        This is especially useful when using the ParallelProcessor with batches of areas, as this makes
        sure that batched areas are of similar size, which is more efficient, as a similar amount of work 
        will be done for each area in the each batch.

    Returns
    -------
    list[Area]
        The list of Area objects.

    """
    # Sort the areas by proximity
    if sort_by_proximity:
        areas = areas.sort_values(by="geometry")

    if sort_by_area:
        # Sort areas by area
        areas["area"] = areas.to_crs("EPSG:25832").geometry.area / 1e6
        areas = areas.sort_values("area")
        # drop the area column
        areas = areas.drop(columns=["area"])

    areas_list = []

    for idx in areas.index:
        # Make sure to pass the row as a GeoDataFrame
        area_gdf = areas.iloc[[idx]]

        # Create an Area object
        area = Area(geometry=area_gdf.iloc[[0]].reset_index(), id=area_gdf.iloc[[0]][id_column].values[0], output_dir=output_dir)

        # Append the Area object to the list
        areas_list.append(area)

    return areas_list
