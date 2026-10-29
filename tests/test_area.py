import warnings

import pytest
import geopandas as gpd
import xarray as xr
import pandas as pd
import numpy as np
from shapely.geometry import Polygon
from pathlib import Path
from stgrid2area import Area


@pytest.fixture
def raster_data():
    """
    Create an example xarray Dataset with a gradient pattern over multiple time steps.
    
    """
    # Create data for 3 time steps
    data = np.array([[[1, 2, 3, 4],
                      [5, 6, 7, 8],
                      [9, 10, 11, 12],
                      [13, 14, 15, 16]],

                     [[2, 3, 4, 5],
                      [6, 7, 8, 9],
                      [10, 11, 12, 13],
                      [14, 15, 16, 17]],

                     [[3, 4, 5, 6],
                      [7, 8, 9, 10],
                      [11, 12, 13, 14],
                      [15, 16, 17, 18]]])

    # Define the coordinates
    times = pd.date_range("2023-01-01", periods=3)
    coords = {
        "time": times,
        "x": np.array([49.0, 49.1, 49.2, 49.3]),
        "y": np.array([8.4, 8.5, 8.6, 8.7])
    }

    # Create a xarray Dataset
    ds = xr.Dataset(
        {
            "var": (["time", "y", "x"], data)
        },
        coords=coords
    )

    # Add a CRS to the dataset
    ds.rio.write_crs("EPSG:4326", inplace=True)

    return ds


@pytest.fixture
def area_data():
    """
    Create a GeoDataFrame covering parts of the raster data.  
    All the raster cells that are covered are covered entirely, so using exact_extract 
    does not make a difference in this case.
    
    """
    # Create a polygon covering a square area that overlaps with the raster data.
    poly = Polygon([(48.95, 8.65), (49.25, 8.65), (49.25, 8.3), (48.95, 8.3)])
    gdf = gpd.GeoDataFrame({"id": ["test_area"], "geometry": [poly]}, crs="EPSG:4326")

    return gdf


@pytest.fixture
def raster_data_with_nan():
    """
    Create an example xarray Dataset like `raster_data`, but with two NaN pixels in the first
    timestep (within the area that will be clipped) and `rio.nodata` set to NaN on the variable,
    for exercising the "nan_percentage" aggregation operation.

    """
    # Create data for 3 time steps, with two NaN values in the first timestep
    data = np.array([[[1.0, 2.0, 3.0, 4.0],
                      [5.0, np.nan, 7.0, 8.0],
                      [9.0, 10.0, np.nan, 12.0],
                      [13.0, 14.0, 15.0, 16.0]],

                     [[2.0, 3.0, 4.0, 5.0],
                      [6.0, 7.0, 8.0, 9.0],
                      [10.0, 11.0, 12.0, 13.0],
                      [14.0, 15.0, 16.0, 17.0]],

                     [[3.0, 4.0, 5.0, 6.0],
                      [7.0, 8.0, 9.0, 10.0],
                      [11.0, 12.0, 13.0, 14.0],
                      [15.0, 16.0, 17.0, 18.0]]])

    # Define the coordinates
    times = pd.date_range("2023-01-01", periods=3)
    coords = {
        "time": times,
        "x": np.array([49.0, 49.1, 49.2, 49.3]),
        "y": np.array([8.4, 8.5, 8.6, 8.7])
    }

    # Create a xarray Dataset
    ds = xr.Dataset(
        {
            "var": (["time", "y", "x"], data)
        },
        coords=coords
    )

    # Add a CRS to the dataset and mark NaN as the nodata value
    ds.rio.write_crs("EPSG:4326", inplace=True)
    ds["var"].rio.write_nodata(np.nan, inplace=True)

    return ds


@pytest.fixture
def nan_area_data():
    """
    Create a GeoDataFrame whose bounds align exactly with the pixel edges of the 3x3 region that
    `raster_data_with_nan` clips to (x: 48.95-49.25, y: 8.35-8.65), so that geometry.area / pixel_area
    is exactly 9, for a deterministic test of the area-based "nan_percentage" exact_extract aggregation.

    """
    poly = Polygon([(48.95, 8.65), (49.25, 8.65), (49.25, 8.35), (48.95, 8.35)])
    gdf = gpd.GeoDataFrame({"id": ["test_area"], "geometry": [poly]}, crs="EPSG:4326")

    return gdf


@pytest.fixture
def multi_var_raster_data():
    """
    Create an example xarray Dataset with multiple variables over multiple time steps.
    
    """
    # Create data for 3 time steps for variable 1
    data1 = np.array([[[1, 2, 3, 4],
                      [5, 6, 7, 8],
                      [9, 10, 11, 12],
                      [13, 14, 15, 16]],

                     [[2, 3, 4, 5],
                      [6, 7, 8, 9],
                      [10, 11, 12, 13],
                      [14, 15, 16, 17]],

                     [[3, 4, 5, 6],
                      [7, 8, 9, 10],
                      [11, 12, 13, 14],
                      [15, 16, 17, 18]]])
    
    # Create data for 3 time steps for variable 2 (just adding 10 to all values)
    data2 = data1 + 10

    # Define the coordinates
    times = pd.date_range("2023-01-01", periods=3)
    coords = {
        "time": times,
        "x": np.array([49.0, 49.1, 49.2, 49.3]),
        "y": np.array([8.4, 8.5, 8.6, 8.7])
    }

    # Create a xarray Dataset with multiple variables
    ds = xr.Dataset(
        {
            "var1": (["time", "y", "x"], data1),
            "var2": (["time", "y", "x"], data2)
        },
        coords=coords
    )

    # Add a CRS to the dataset
    ds.rio.write_crs("EPSG:4326", inplace=True)

    return ds


def test_area_initialization(area_data):
    """
    Test the initialization of an Area object.
    
    """
    # Create an Area object
    area = Area(area_data, "test_id", "/tmp/output")

    assert area.id == "test_id"
    assert area.output_path == Path("/tmp/output/test_id")


def test_clip(raster_data, area_data):
    """
    Test the clip method of the Area object.
    
    """
    # Create an Area object
    area = Area(geometry=area_data, id="test", output_dir="/tmp/output")
    
    # Clip the raster data to the area
    clipped_ds = area.clip(raster_data)

    assert isinstance(clipped_ds, xr.Dataset)

    # Expected clipped values for all time steps
    expected_clipped_data = np.array([[[1,  2,  3],
                                       [5,  6,  7],
                                       [9, 10, 11]],

                                      [[2,  3,  4],
                                       [6,  7,  8],
                                       [10, 11, 12]],

                                      [[3,  4,  5],
                                       [7,  8,  9],
                                       [11, 12, 13]]])

    assert clipped_ds["var"].values.shape == expected_clipped_data.shape
    assert np.array_equal(clipped_ds["var"].values, expected_clipped_data)


def test_aggregate(raster_data, area_data):
    """
    Test the aggregate method.
    
    """
    # Create an Area object
    area = Area(geometry=area_data, id="test", output_dir="/tmp/output")

    # First, clip the data to the area
    clipped_ds = area.clip(raster_data)

    # Aggregate the clipped data spatially
    aggregated_df = area.aggregate(clipped_ds, variables="var", method="exact_extract", operations=["mean", "min"])

    assert isinstance(aggregated_df, pd.DataFrame)
    assert not aggregated_df.empty
    assert aggregated_df.shape == (3, 2)

    # Expected clipped values for all time steps
    expected_clipped_data = np.array([[[1,  2,  3],
                                       [5,  6,  7],
                                       [9, 10, 11]],

                                      [[2,  3,  4],
                                       [6,  7,  8],
                                       [10, 11, 12]],

                                      [[3,  4,  5],
                                       [7,  8,  9],
                                       [11, 12, 13]]])

    # Expected temporal mean across the 3 time steps
    expected_means = [expected_clipped_data[0].mean(), expected_clipped_data[1].mean(), expected_clipped_data[2].mean()]

    # Expected temporal min across the 3 time steps
    expected_mins = [expected_clipped_data[0].min(), expected_clipped_data[1].min(), expected_clipped_data[2].min()]

    # Make a dataframe with the expected values
    expected_df = pd.DataFrame({"var_mean": expected_means, "var_min": expected_mins}, index=clipped_ds.time.values)
    expected_df.index.name = "time"

    assert pd.testing.assert_frame_equal(aggregated_df, expected_df, check_dtype=False) is None


def test_aggregate_multiple_variables(multi_var_raster_data, area_data):
    """
    Test aggregating multiple variables at once.
    
    """
    # Create an Area object
    area = Area(geometry=area_data, id="test", output_dir="/tmp/output")

    # First, clip the data to the area
    clipped_ds = area.clip(multi_var_raster_data)

    # Aggregate multiple variables in the clipped data spatially
    aggregated_df = area.aggregate(clipped_ds, variables=["var1", "var2"], method="exact_extract", operations=["mean", "max"])

    assert isinstance(aggregated_df, pd.DataFrame)
    assert not aggregated_df.empty
    # Should have 4 columns (2 variables × 2 operations)
    assert aggregated_df.shape == (3, 4)
    
    # Check column names follow expected pattern: {variable}_{operation}
    expected_columns = ["var1_mean", "var1_max", "var2_mean", "var2_max"]
    for col in expected_columns:
        assert col in aggregated_df.columns
    
    # Expected clipped values for all time steps for var1
    expected_clipped_var1 = np.array([[[1,  2,  3],
                                      [5,  6,  7],
                                      [9, 10, 11]],

                                     [[2,  3,  4],
                                      [6,  7,  8],
                                      [10, 11, 12]],

                                     [[3,  4,  5],
                                      [7,  8,  9],
                                      [11, 12, 13]]])
    
    # Expected clipped values for all time steps for var2 (var1 + 10)
    expected_clipped_var2 = expected_clipped_var1 + 10

    # Calculate expected values
    expected_means_var1 = [expected_clipped_var1[i].mean() for i in range(3)]
    expected_maxes_var1 = [expected_clipped_var1[i].max() for i in range(3)]
    expected_means_var2 = [expected_clipped_var2[i].mean() for i in range(3)]
    expected_maxes_var2 = [expected_clipped_var2[i].max() for i in range(3)]

    # Make a dataframe with the expected values
    expected_df = pd.DataFrame({
        "var1_mean": expected_means_var1,
        "var1_max": expected_maxes_var1,
        "var2_mean": expected_means_var2,
        "var2_max": expected_maxes_var2
    }, index=clipped_ds.time.values)
    expected_df.index.name = "time"

    assert pd.testing.assert_frame_equal(aggregated_df, expected_df, check_dtype=False) is None


def test_aggregate_xarray_multiple_variables(multi_var_raster_data, area_data):
    """
    Test aggregating multiple variables at once using the xarray method.
    
    """
    # Create an Area object
    area = Area(geometry=area_data, id="test", output_dir="/tmp/output")

    # First, clip the data to the area
    clipped_ds = area.clip(multi_var_raster_data)

    # Aggregate multiple variables in the clipped data spatially using xarray method
    aggregated_df = area.aggregate(clipped_ds, variables=["var1", "var2"], method="xarray", operations=["mean", "max"])

    assert isinstance(aggregated_df, pd.DataFrame)
    assert not aggregated_df.empty
    # Should have 4 columns (2 variables × 2 operations)
    assert aggregated_df.shape == (3, 4)

    # Check column names follow expected pattern: {variable}_{operation}
    expected_columns = ["var1_mean", "var1_max", "var2_mean", "var2_max"]
    for col in expected_columns:
        assert col in aggregated_df.columns


def test_aggregate_nan_percentage_exact_extract(raster_data_with_nan, nan_area_data):
    """
    Test the "nan_percentage" operation with the exact_extract method.

    """
    # Create an Area object whose geometry aligns exactly with the clipped pixel grid
    area = Area(geometry=nan_area_data, id="test", output_dir="/tmp/output")

    # First, clip the data to the area
    clipped_ds = area.clip(raster_data_with_nan)

    # Aggregate the clipped data spatially
    aggregated_df = area.aggregate(clipped_ds, variables="var", method="exact_extract", operations=["nan_percentage"])

    assert isinstance(aggregated_df, pd.DataFrame)
    assert list(aggregated_df.columns) == ["var_nan_percentage"]

    # The first timestep has 2 NaN cells out of 9 clipped cells, the other two timesteps have none
    expected = np.array([2 / 9 * 100, 0.0, 0.0])
    np.testing.assert_allclose(aggregated_df["var_nan_percentage"].values, expected, rtol=1e-6)


def test_aggregate_nan_percentage_exact_extract_keeps_requested_count(raster_data_with_nan, nan_area_data):
    """
    Test that explicitly requesting "count" alongside "nan_percentage" keeps both columns in the result.

    """
    # Create an Area object whose geometry aligns exactly with the clipped pixel grid
    area = Area(geometry=nan_area_data, id="test", output_dir="/tmp/output")

    # First, clip the data to the area
    clipped_ds = area.clip(raster_data_with_nan)

    # Aggregate the clipped data spatially, requesting the helper count explicitly too
    aggregated_df = area.aggregate(
        clipped_ds, variables="var", method="exact_extract", operations=["nan_percentage", "count"],
    )

    assert set(aggregated_df.columns) == {"var_nan_percentage", "var_count"}
    np.testing.assert_allclose(aggregated_df["var_count"].values, [7.0, 9.0, 9.0])


def test_aggregate_nan_percentage_exact_extract_warns_without_nodata(raster_data_with_nan, nan_area_data):
    """
    Test that a UserWarning is raised for the exact_extract method if the variable has no
    `rio.nodata` value set, since "nan_percentage" then cannot correctly identify NaN pixels.

    """
    # Create an Area object
    area = Area(geometry=nan_area_data, id="test", output_dir="/tmp/output")

    # Remove the nodata value to simulate a dataset where it was never set
    raster_data_with_nan["var"].rio.write_nodata(None, inplace=True)

    # First, clip the data to the area
    clipped_ds = area.clip(raster_data_with_nan)

    with pytest.warns(UserWarning, match="rio.nodata"):
        area.aggregate(clipped_ds, variables="var", method="exact_extract", operations=["nan_percentage"])


def test_aggregate_nan_percentage_xarray(raster_data_with_nan, area_data):
    """
    Test the "nan_percentage" operation with the xarray method.

    """
    # Create an Area object
    area = Area(geometry=area_data, id="test", output_dir="/tmp/output")

    # First, clip the data to the area
    clipped_ds = area.clip(raster_data_with_nan)

    # Aggregate the clipped data spatially
    aggregated_df = area.aggregate(clipped_ds, variables="var", method="xarray", operations=["nan_percentage"])

    assert isinstance(aggregated_df, pd.DataFrame)
    assert list(aggregated_df.columns) == ["var_nan_percentage"]

    # The first timestep has 2 NaN cells out of 9 clipped cells, the other two timesteps have none
    expected = np.array([2 / 9 * 100, 0.0, 0.0])
    np.testing.assert_allclose(aggregated_df["var_nan_percentage"].values, expected, rtol=1e-6)