from .area import Area

import geopandas as gpd


def geodataframe_to_areas(areas: gpd.GeoDataFrame, id_column: str, output_dir: str) -> list[Area]:
    """
    Convert a GeoDataFrame of areas to a list of Area objects to be used as input for the ParallelProcessor.

    Parameters
    ----------
    areas : gpd.GeoDataFrame
        The GeoDataFrame of areas.
    output_dir : str
        The output directory where results will be saved.  
        Will always be a subdirectory of this directory, named after the area's id.

    Returns
    -------
    list[Area]
        The list of Area objects.

    """
    areas_list = []

    for idx in areas.index:
        # Make sure to pass the row as a GeoDataFrame
        area_gdf = areas.iloc[[idx]]

        # Create an Area object
        area = Area(geometry=area_gdf.iloc[[0]].reset_index(), id=area_gdf.iloc[[0]][id_column].values[0], output_dir=output_dir)

        # Append the Area object to the list
        areas_list.append(area)

    return areas_list
