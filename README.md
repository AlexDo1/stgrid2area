# stgrid2area

Extract and aggregate spatio-temporal gridded data in netCDF or GRIB format to a specified area.

## Overview

`stgrid2area` is a Python package that simplifies the workflow of extracting and aggregating spatio-temporal gridded data (e.g., climate data, weather forecasts) to geometrically defined areas like catchments, administrative boundaries, or any other geographical region. The package handles the process of clipping gridded data to the specified area's boundaries and then calculating spatial statistics across the area.

![Clip and aggregate](docs/images/workflow_image.svg)

*Figure: Clip and aggregate spatio-temporal gridded data to areas. Image from [https://doi.org/10.5194/essd-16-5625-2024](https://doi.org/10.5194/essd-16-5625-2024)*

## Features

- **Clip** spatio-temporal gridded data to area boundaries &rarr; `xarray.Dataset` or save to `.nc`
- **Aggregate** variables using various spatial statistics (mean, min, max, stdev, quantiles) &rarr; `pd.DataFrame` or save to `.csv`
- **Process multiple variables simultaneously** in a single operation
- **Efficient parallel processing** options for large datasets and many areas
- Support for both **weighted** ([exactextract](https://github.com/isciences/exactextract)) and **unweighted** (xarray) spatial statistics
- **HPC integration** with single-node and MPI multi-node support for large-scale processing using [Dask](https://docs.dask.org/en/stable/)

## Installation

```bash
pip install stgrid2area
```

## Basic Usage

The workflow for clipping and aggregating spatio-temporal gridded data to specified areas usually involves the following steps:
1. **Read the spatio-temporal gridded (stgrid) data** (e.g., NetCDF or GRIB files) using `xarray`.
2. **Read the geometries** (e.g., catchment boundaries, administrative boundaries) using `geopandas`.
3. Do preprocessing if necessary, e.g. set the coordinate reference system (CRS) of the stgrid data (`stgrid.rio.write_crs("EPSG:4326")`) and reproject the geometries to match the CRS of the stgrid data using `geometries.to_crs(stgrid.crs)` or reformat the stgrid data if necessary.
4. **Create an `Area` object** for each geometry, which encapsulates the clipping and aggregation logic. You can use the helper function `geodataframe_to_areas` to create a list of `Area` objects from a GeoDataFrame.
5. **Clip and aggregate the gridded data** to the areas. You can either use `area.clip()` and `area.aggregate()` directly or use a processor class for parallel processing.

### Example 1: Basic Clipping and Aggregation

```python
import xarray as xr
import geopandas as gpd
from stgrid2area import Area

# Read the spatio-temporal gridded data
stgrid = xr.open_dataset("path/to/climate_data.nc")
stgrid = stgrid.rio.set_crs("EPSG:4326")  # Set the CRS if not already set

# Read the geometry data (e.g., a catchment boundary)
geometry = gpd.read_file("path/to/catchments.gpkg")
geometry = geometry.to_crs(stgrid.crs)  # Ensure CRS matches the gridded data

# Create an Area object
area = Area(geometry=geometry, id="catchment1", output_dir="/path/to/output")

# Clip the gridded data to the area
clipped_data = area.clip(stgrid)

# Aggregate a single variable across the area
aggregated = area.aggregate(
    clipped_data, 
    variables="temperature", 
    method="exact_extract", 
    operations=["mean", "min", "max"]
)

# Save the aggregated data
aggregated.to_csv("/path/to/output/temperature_stats.csv")
```

### Example 2: Processing Multiple Variables

```python
# Aggregate multiple variables at once
aggregated = area.aggregate(
    clipped_data, 
    variables=["temperature", "precipitation", "wind_speed"], 
    method="exact_extract", 
    operations=["mean", "max"]
)

# The result will contain columns like: temperature_mean, temperature_max, precipitation_mean, precipitation_max, wind_speed_mean, wind_speed_max
```

### Example 3: Processing Multiple Areas in Parallel

```python
import geopandas as gpd
from stgrid2area import Area, LocalDaskProcessor, geodataframe_to_areas

# Read the spatio-temporal gridded data
stgrid = xr.open_dataset('path/to/climate_data.nc')

# Read multiple geometries (e.g., multiple catchments)
geometries = gpd.read_file('path/to/catchments.gpkg')

# Create a list of Area objects
areas = geodataframe_to_areas(
    geometries, 
    output_dir="/path/to/output", 
    id_column="catchment_id"  # Column containing unique identifiers for each area
)

# Create a single-node processor with 4 workers
processor = LocalDaskProcessor(
    areas=areas,
    stgrid=stgrid,
    variables=["temperature", "precipitation"],
    method="exact_extract",
    operations=["mean", "max"],
    n_workers=4,
    save_nc=True,  # Save clipped NetCDF files
    save_csv=True  # Save aggregated CSV files
)

# Run the processing
processor.run()
```

## Documentation

### The `Area` Class

The `Area` class represents a geographical area (e.g., a catchment or administrative boundary) and provides methods to clip and aggregate gridded data to this area.

```python
from stgrid2area import Area

# Create an Area object
area = Area(
    geometry=geometry_gdf,  # geopandas GeoDataFrame containing the geometry
    id="unique_identifier",  # Unique identifier for this area
    output_dir="/path/to/output"  # Directory to save output files
)
```

#### Clipping Data

```python
# Clip the gridded data to the area's geometry
clipped_data = area.clip(
    stgrid,  # xarray.Dataset: The gridded data to clip
    all_touched=True,  # Whether to include cells that are partially within the area
    save_result=True,  # Whether to save the clipped data to disk
    skip_exist=False,  # Whether to skip clipping if the output file already exists
    filename=None  # Optional filename for the output file
)
```

#### Aggregating Data

```python
# Aggregate the clipped data spatially
aggregated_data = area.aggregate(
    stgrid,  # xarray.Dataset: The gridded data to aggregate
    variables=["temperature", "precipitation"],  # Variable(s) to aggregate
    method="exact_extract",  # Aggregation method: "exact_extract" or "xarray"
    operations=["mean", "min", "max", "stdev"],  # Statistical operations
    save_result=True,  # Whether to save the aggregated data to disk
    skip_exist=False,  # Whether to skip aggregation if the output file already exists
    filename=None  # Optional filename for the output file
)
```

### Aggregation Methods

stgrid2area supports two aggregation methods:

1. **`exact_extract`**: Uses the [exactextract](https://github.com/isciences/exactextract) library to calculate weighted statistics based on cell coverage. This provides more accurate results, especially for areas with irregular shapes and with cells that are partially within the area.

2. **`xarray`**: Uses xarray's built-in statistical functions to calculate unweighted statistics.

### Processor Classes

For processing multiple areas efficiently, stgrid2area provides several processor classes:

#### LocalDaskProcessor

For processing on a single machine / a single-node HPC batch queue with multiple cores. This processor class creates a Dask cluster locally and distributes the workload across multiple workers.

```python
from stgrid2area import LocalDaskProcessor

processor = LocalDaskProcessor(
    areas=areas,  # List of Area objects
    stgrid=stgrid,  # The gridded data
    variables="temperature",  # Variable(s) to aggregate
    method="exact_extract",  # Aggregation method
    operations=["mean", "max"],  # Statistical operations
    n_workers=4,  # Number of parallel workers
    skip_exist=True,  # Skip areas that already have output files
    batch_size=10,  # Process areas in batches of this size
    save_nc=True,  # Save clipped NetCDF files
    save_csv=True  # Save aggregated CSV files
)

processor.run()
```

#### MPIDaskProcessor

For processing with MPI (Message Passing Interface) on HPC systems. This processor is designed to run on distributed systems with multiple nodes, allowing for efficient parallel processing of large datasets.  
To use this method you need to initialize a Dask MPI client yourself and pass it to `processor.run()`. On the HPC system, you would typically launch the Python processing script using `mpirun` inside a SLURM job script or similar.

```python
from stgrid2area import MPIDaskProcessor
from dask.distributed import Client
import dask_mpi

# Initialize dask_mpi
dask_mpi.initialize()
client = Client()

processor = MPIDaskProcessor(
    areas=areas,  # List of Area objects
    stgrid=stgrid,  # The gridded data
    variables=["temperature", "precipitation"],  # Variable(s) to aggregate
    method="exact_extract",  # Aggregation method
    operations=["mean", "max"],  # Statistical operations
    skip_exist=True,  # Skip areas that already have output files
    batch_size=100  # Process areas in batches of this size
)

processor.run(client=client)
```

### When to Use Which Processor

- **LocalDaskProcessor**: Use locally on a single machine / a single-node HPC batch queue with multiple cores.
- **MPIDaskProcessor**: Use for large datasets on HPC systems using MPI for multi-node queues and distributed computing.

## Advanced Usage

### Processing Multiple Time Periods

If your stgrid data is too large to fit in memory, you can process it in chunks. The `LocalDaskProcessor` and `MPIDaskProcessor` classes can handle a list of xarray datasets, the clipped and aggregated results are saved for each input dataset separately, so you have to care about post-processing the results if you want to combine them later.

```python
import xarray as xr
from stgrid2area import LocalDaskProcessor

# Read multiple files, each containing data for a different time period
stgrid_files = [
    'path/to/data_2020.nc',
    'path/to/data_2021.nc',
    'path/to/data_2022.nc'
]

# Create a list of xarray datasets
stgrids = [xr.open_dataset(file) for file in stgrid_files]

# Create a processor with multiple stgrids
processor = LocalDaskProcessor(
    areas=areas,
    stgrid=stgrids,  # List of xarray datasets
    variables="temperature",
    method="exact_extract",
    operations=["mean", "max"],
    n_workers=4
)

# Run the processing
processor.run()
```

## stgrid2area-workflows

The [stgrid2area-workflows](https://github.com/CAMELS-DE/stgrid2area-workflows) repository is a collection of implemented data processing workflows using the `stgrid2area` package. It is mainly used for processing meteorological input data for the CAMELS-DE dataset. It also includes Python scripts and SLURM job scripts for running the workflows on HPC systems.  
Hence, it can serve as a reference for how to use `stgrid2area` in practice, even though the code is not always perfectly documented and under development.