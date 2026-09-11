import gc
from typing import Union
from pathlib import Path

import dask

import pandas as pd
import xarray as xr
import rioxarray

from .area import Area


def process_area(
    area: Area,
    stgrid: Union[xr.Dataset, xr.DataArray],
    variables: Union[str, list[str]],
    method: str,
    operations: list[str],
    skip_exist: bool,
    n_stgrid: int,
    total_stgrids: int,
    save_nc: bool = True,
    save_csv: bool = True,
    clip_only_variables: bool = True,
) -> dict:
    """
    Clip and aggregate a single area against one spatiotemporal grid.

    This is a standalone module-level function so that Dask workers can import and
    execute it without serialising the entire processor object (which would copy all
    stgrids to every worker).

    Parameters
    ----------
    area : Area
        The area to process.
    stgrid : xr.Dataset or xr.DataArray
        The spatiotemporal grid (or a persisted pre-clip of it) to process.
    variables : str or list[str]
        Variable name(s) in *stgrid* to aggregate.
    method : str
        Aggregation method: ``"exact_extract"``, ``"xarray"``, or
        ``"fallback_xarray"``.
    operations : list[str]
        Aggregation operations to apply (e.g. ``["mean", "max"]``).
    skip_exist : bool
        When ``True``, skip areas whose output files already exist.
    n_stgrid : int
        1-based index of this stgrid in the full list, used to build output
        filenames when there is more than one stgrid.
    total_stgrids : int
        Total number of stgrids being processed.
    save_nc : bool, optional
        Save the clipped grid to a NetCDF file.  Default ``True``.
    save_csv : bool, optional
        Save the aggregated result to a CSV file.  Default ``True``.
    clip_only_variables : bool, optional
        If ``True`` (default), only *variables* are read/clipped instead of
        every variable in *stgrid*, reducing worker memory use and runtime
        when the source grid contains additional, unused variables.

    Returns
    -------
    dict
        Execution status and metadata (e.g., rows processed). Returning a tiny 
        dict instead of a full pd.DataFrame prevents Dask memory leaks.
    """
    if isinstance(variables, str):
        variables = [variables]

    filename_clip = (
        f"{area.id}_{n_stgrid}_clipped.nc" if total_stgrids > 1 else f"{area.id}_clipped.nc"
    )
    filename_aggr = (
        f"{area.id}_{n_stgrid}_aggregated.csv"
        if total_stgrids > 1
        else f"{area.id}_aggregated.csv"
    )

    try:
        # Use the synchronous scheduler to avoid Dask's memory leak when returning large objects.
        with dask.config.set(scheduler="synchronous"):
            stgrid_source = stgrid
            if clip_only_variables and isinstance(stgrid, xr.Dataset):
                present = [v for v in variables if v in stgrid.data_vars]
                if present:
                    stgrid_source = stgrid[present]

            # Pre-clip lazily to bounding box of area first
            minx, miny, maxx, maxy = area.geometry.geometry.to_crs(stgrid_source.rio.crs).total_bounds
            stgrid_bbox = stgrid_source.rio.clip_box(minx=minx, miny=miny, maxx=maxx, maxy=maxy, allow_one_dimensional_raster=True).compute() # compute before using rio.clip

            clipped = area.clip(
                stgrid_bbox,
                save_result=save_nc,
                skip_exist=skip_exist,
                filename=filename_clip,
            )

            stgrid_bbox.close()
            del stgrid_bbox

            if method in ["exact_extract", "xarray"]:
                result = area.aggregate(
                    clipped,
                    variables,
                    method,
                    operations,
                    save_result=save_csv,
                    skip_exist=skip_exist,
                    filename=filename_aggr,
                )
            elif method == "fallback_xarray":
                try:
                    result = area.aggregate(
                        clipped,
                        variables,
                        "exact_extract",
                        operations,
                        save_result=save_csv,
                        skip_exist=skip_exist,
                        filename=filename_aggr,
                    )
                except ValueError as e:
                    if (
                        "spatial dimensionality of 1" in str(e)
                        or "aggregation for 1-D data is not possible" in str(e)
                    ):
                        if not area.output_path.exists():
                            area.output_path.mkdir(parents=True, exist_ok=True)
                        Path(area.output_path, "fallback_xarray").touch()
                        result = area.aggregate(
                            clipped,
                            variables,
                            "xarray",
                            operations,
                            save_result=save_csv,
                            skip_exist=skip_exist,
                            filename=filename_aggr,
                        )
                    else:
                        raise
            else:
                raise ValueError(
                    f"Unknown method: {method!r}. "
                    "Use 'exact_extract', 'xarray', or 'fallback_xarray'."
                )

        # Return dict for tracking process
        n_rows = len(result) if isinstance(result, pd.DataFrame) else 0
        return {"status": "success", "rows_written": n_rows}

    except Exception:
        raise

    finally:
        try:
            clipped.close()
            del clipped
        except NameError:
            pass
        gc.collect()


def check_area_needs_processing(
    area: Area,
    n_stgrid: int,
    total_stgrids: int,
    save_nc: bool,
    save_csv: bool,
    existing_files: set = None,
) -> bool:
    """
    Return ``True`` if *area* still needs to be processed for stgrid *n_stgrid*.

    Checks for both the per-stgrid output files and the merged output files that
    are written after all stgrids are concatenated.

    Parameters
    ----------
    area : Area
    n_stgrid : int
        1-based stgrid index.
    total_stgrids : int
        Total number of stgrids.
    save_nc : bool
    save_csv : bool
    existing_files : set, optional
        Pre-scanned set of filenames already present in *area.output_path*.
        When provided, all existence checks are done against this in-memory set
        (no filesystem calls).

    Returns
    -------
    bool
        ``True`` if processing is required, ``False`` if outputs already exist.
    """
    def _exists(filename: str) -> bool:
        if existing_files is not None:
            return filename in existing_files
        return (Path(area.output_path) / filename).exists()

    # If merged (post-concatenation) files already exist, nothing more to do.
    if save_csv and _exists(f"{area.id}_aggregated.csv"):
        return False
    if save_nc and _exists(f"{area.id}_clipped.nc"):
        return False

    filename_clip = (
        f"{area.id}_{n_stgrid}_clipped.nc" if total_stgrids > 1 else f"{area.id}_clipped.nc"
    )
    filename_aggr = (
        f"{area.id}_{n_stgrid}_aggregated.csv"
        if total_stgrids > 1
        else f"{area.id}_aggregated.csv"
    )

    if save_nc and not save_csv:
        return not _exists(filename_clip)
    if save_csv and not save_nc:
        return not _exists(filename_aggr)
    if save_nc and save_csv:
        return not (_exists(filename_clip) and _exists(filename_aggr))

    # Neither file type is being saved — always process.
    return True