import warnings

import geopandas as gpd

from .area import Area


def geodataframe_to_areas(
    areas: gpd.GeoDataFrame,
    id_column: str,
    output_dir: str,
    sort_by_proximity: bool = None,
    sort_by_area: bool = True,
) -> list[Area]:
    """
    Convert a GeoDataFrame of areas to a list of Area objects to be used as input for
    the processor classes.

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
        Deprecated, will be removed in a future release. Used to group nearby areas
        together for a shared batch-wise bounding-box pre-clip, which no longer
        exists (each area is now clipped independently), so this has no effect
        anymore.
    sort_by_area : bool, optional
        Whether to sort the areas ascending by area size (smallest first).
        Default is True.
        The processors submit tasks to Dask in this list order, and Dask schedules
        equally-ready tasks in submission order, so smaller areas tend to start (and
        finish) earlier. This means that if a large/complex area causes a worker to
        run out of memory, it is more likely to happen towards the end of a run,
        once most smaller areas are already done, rather than at a random point.

    Returns
    -------
    list[Area]
        The list of Area objects.

    """
    if sort_by_proximity is not None:
        warnings.warn(
            "sort_by_proximity is deprecated and no longer has any effect (the "
            "batch-wise bounding-box pre-clip it was meant to help with no longer "
            "exists), it will be removed in a future release.",
            DeprecationWarning,
            stacklevel=2,
        )

    if sort_by_area:
        # Sort areas by area
        areas["area"] = areas.to_crs("EPSG:25832").geometry.area / 1e6
        areas = areas.sort_values("area")
        # drop the area column
        areas = areas.drop(columns=["area"])

    areas_list = []

    for i in range(len(areas)):
        # Positional indexing so the loop follows the current (e.g. sorted) row order
        area_gdf = areas.iloc[[i]]

        # Create an Area object
        area = Area(geometry=area_gdf.reset_index(), id=area_gdf[id_column].values[0], output_dir=output_dir)

        # Append the Area object to the list
        areas_list.append(area)

    return areas_list
