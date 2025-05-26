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