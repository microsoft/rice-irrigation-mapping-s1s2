# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import ee
import pandas as pd
from datetime import datetime
import xarray as xr
import rioxarray
from pathlib import Path
from typing import Dict, Optional, List
from shapely.geometry import Polygon, shape
from shapely.validation import make_valid
import re
from concurrent.futures import ProcessPoolExecutor
from ricemapper.utils.utils import gdf_to_single_ee_geometry
import os
from tqdm import tqdm
import psutil
import time


class S1TimeseriesProcessor:
    def __init__(
        self,
        aoi_gdf,
        start_date: str,
        end_date: str,
        local_data_dir: str,
        orbits: list = ["ASCENDING"],
        max_workers: int = -1,
    ):
        """
        Initialize parallel processor for S1 timeseries

        Args:
            aoi_gdf: Area of interest GeoDataFrame
                should have the same bounds as gdf, but this gdf only has the district polygons
            start_date: Overall start date
            end_date: Overall end date
            local_data_dir: Directory with S1 data
            date_range: Optional specific date range for processing
            orbits: List of orbits to process
            max_workers: Number of parallel processes
        """
        self.start_date = start_date
        self.end_date = end_date
        self.data_dir = local_data_dir
        self.orbits = orbits

        self.max_workers = os.cpu_count() if max_workers == -1 else max_workers
        aoi = gdf_to_single_ee_geometry(aoi_gdf)

        # Initialize S1 processor once with combined geometry
        self.s1_processor = Sentinel1RowCollections(
            aoi=aoi,
            start_date=start_date,
            end_date=end_date,
            local_data_dir=local_data_dir,
        )

        self.s1_processor.describe_collections()

    def visualize_collection(self, Map, point=None):
        """Visualize the collection boundaries using geemap"""
        self.s1_processor.visualize_collection(Map, point=point)

    def _process_single_row(self, row_data: Dict) -> Dict:
        """Process a single row with all orbits, maintaining temporal order"""
        idx, row = row_data["idx"], row_data["row"]
        results = {"idx": idx}

        for orbit in self.orbits:
            try:
                # Get temporally ordered timeseries
                df = self.s1_processor.get_timeseries_polygon_pandas(
                    geometry=row.geometry, orbit=orbit
                )

                if not df.empty:
                    for col in df.columns:
                        results[col] = df[col].iloc[0]
            except Exception as e:
                print(f"Error processing {orbit} for geometry {idx}: {str(e)}")

        return results

    def __call__(self, gdf):
        """Process all polygons in parallel while maintaining temporal order within each polygon
        gdf: Input GeoDataFrame with plot-level polygons

        """
        # Prepare row data
        row_data = [{"idx": idx, "row": row} for idx, row in gdf.iterrows()]

        # Process rows in parallel
        results = []
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all row processing jobs
            futures = [
                executor.submit(self._process_single_row, data) for data in row_data
            ]

            # Collect results maintaining input order
            for future in futures:
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    print(f"Error in future: {str(e)}")

        # Update GeoDataFrame
        for result in results:
            idx = result.pop("idx")
            for col, value in result.items():
                gdf.at[idx, col] = value

        return gdf


class Sentinel1RowCollections:
    def __init__(
        self,
        aoi,
        start_date: str,
        end_date: str,
        local_data_dir: str,
        orbits=["ASCENDING"],
    ):
        """
        Initialize with GEE bounds and lazy loading of local data

        Args:
            aoi: Earth Engine geometry for the area of interest
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            local_data_dir: Directory containing processed S1 GeoTIFFs organized as:
                          orbit/row/slice/tiff_files
        """
        self.aoi = aoi
        self.start_date = start_date
        self.end_date = end_date
        self.data_dir = Path(local_data_dir)
        self.orbits = orbits

        # Get GEE information including slice metadata
        self.rows_dict = self._get_row_collections_with_slices()

        # discard extra rows
        self.discard_rows()

        print("=== Sentinel-1 Data Collection ===")
        print(f"Start Date: {start_date}")
        print(f"End Date: {end_date}")
        print(f"Local Data Directory: {local_data_dir}")
        print(
            f"Rows: {len(self.rows_dict['ASCENDING'])} ASCENDING, {len(self.rows_dict['DESCENDING'])} DESCENDING"
        )
        for orbit in self.rows_dict:
            for row in self.rows_dict[orbit]:
                print(f"  Row {row}: {len(self.rows_dict[orbit][row])} slices")
        print("==================================\n")

        # Find intersecting slices with AOI and setup lazy loading
        self._setup_lazy_loading()

    def _get_row_collections_with_slices(self) -> Dict:
        """Get row collections from GEE with slice information"""
        s1_collection = ee.ImageCollection("COPERNICUS/S1_GRD_FLOAT")

        filtered_collection = (
            s1_collection.filterDate(self.start_date, self.end_date)
            .filterBounds(self.aoi)
            .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
            .filter(ee.Filter.eq("instrumentMode", "IW"))
            .filter(ee.Filter.eq("resolution_meters", 10))
        )

        rows_dict = {"ASCENDING": {}, "DESCENDING": {}}

        for orbit in self.orbits:  # ['ASCENDING', 'DESCENDING']:
            orbit_collection = filtered_collection.filter(
                ee.Filter.eq("orbitProperties_pass", orbit)
            )

            # Get unique combinations of row and slice numbers
            def get_row_slice(image):
                return ee.Feature(
                    None,
                    {
                        "row": image.get("relativeOrbitNumber_start"),
                        "slice": image.get("sliceNumber"),
                    },
                )

            row_slices = (
                orbit_collection.map(get_row_slice)
                .distinct(["row", "slice"])
                .getInfo()["features"]
            )

            for feature in row_slices:
                row = feature["properties"]["row"]
                slice_num = feature["properties"]["slice"]

                if row not in rows_dict[orbit]:
                    rows_dict[orbit][row] = {}

                # Get geometry for this specific row/slice combination
                slice_collection = orbit_collection.filter(
                    ee.Filter.eq("relativeOrbitNumber_start", row)
                ).filter(ee.Filter.eq("sliceNumber", slice_num))

                slice_geom = slice_collection.geometry().getInfo()

                rows_dict[orbit][row][slice_num] = {
                    "bounds": shape(slice_geom),
                    "collection": None,  # Will be lazily loaded if needed
                }

        return rows_dict

    def discard_rows(self, discard_rows={"ASCENDING": 173, "DESCENDING": 107}):
        """
        Given a list of row numbers, discard the corresponding rows from the collection.
        discard_rows : dict with keys as orbit and values as list of row numbers to discard
            e.g.: discard_rows = {'ASCENDING':173, 'DESCENDING': 107}
        """
        for orbit, row in discard_rows.items():
            self.rows_dict[orbit].pop(row, None)

        print(f"[ASCENDING] rows: {self.rows_dict['ASCENDING'].keys()}")
        print(f"[DESCENDING] rows: {self.rows_dict['DESCENDING'].keys()}")

    def _setup_lazy_loading(self):
        """Set up lazy loading for slices that intersect with AOI and validate geometries"""
        from shapely.validation import make_valid, explain_validity

        # Clean and validate AOI geometry once
        aoi_shape = shape(self.aoi.getInfo())
        if not aoi_shape.is_valid:
            print(f"AOI geometry invalid. Issue: {explain_validity(aoi_shape)}")
            aoi_shape = make_valid(aoi_shape)
            if not aoi_shape.is_valid:
                aoi_shape = aoi_shape.buffer(0)
            if not aoi_shape.is_valid:
                raise ValueError("Could not fix AOI geometry")

        self.intersecting_slices = {orbit: {} for orbit in ["ASCENDING", "DESCENDING"]}

        # Clean and validate slice geometries once during initialization
        for orbit in self.rows_dict:
            for row in self.rows_dict[orbit]:
                for slice_num, slice_data in self.rows_dict[orbit][row].items():
                    try:
                        # Clean and validate slice geometry
                        slice_geom = slice_data["bounds"]
                        if not slice_geom.is_valid:
                            print(
                                f"Cleaning geometry for {orbit} row {row} slice {slice_num}"
                            )
                            slice_geom = make_valid(slice_geom)
                            if not slice_geom.is_valid:
                                slice_geom = slice_geom.buffer(0)
                            if not slice_geom.is_valid:
                                print(
                                    f"Could not fix geometry for {orbit} row {row} slice {slice_num}"
                                )
                                continue

                            # Update the cleaned geometry
                            self.rows_dict[orbit][row][slice_num]["bounds"] = slice_geom

                        # Check intersection with AOI
                        if slice_geom.intersects(aoi_shape):
                            # Set up lazy loading for this slice
                            slice_dir = (
                                self.data_dir
                                / orbit.lower()
                                / str(row)
                                / str(slice_num)
                            )
                            if slice_dir.exists():
                                if row not in self.intersecting_slices[orbit]:
                                    self.intersecting_slices[orbit][row] = []
                                self.intersecting_slices[orbit][row].append(slice_num)

                                # Store directory path and file list for lazy loading
                                tiff_files = list(
                                    slice_dir.glob("*_COG_Orb_NR_Cal_Spk_TC.tif")
                                )
                                if tiff_files:
                                    self.rows_dict[orbit][row][slice_num][
                                        "collection"
                                    ] = (slice_dir, tiff_files)

                    except Exception as e:
                        print(
                            f"Error processing {orbit} row {row} slice {slice_num}: {str(e)}"
                        )
                        continue

    def _parse_s1_filename(self, filename: str) -> datetime:
        """Extract datetime from S1 filename"""
        pattern = r"\d{8}T\d{6}"
        match = re.search(pattern, str(filename))
        if match:
            return datetime.strptime(match.group(), "%Y%m%dT%H%M%S")
        raise ValueError(f"Could not parse datetime from filename: {filename}")

    def _load_slice_data(self, slice_dir, tiff_files):
        """Lazy load a single slice when needed"""
        datasets = []
        times = []

        for file_path in tiff_files:
            print(f"Loading {file_path}")
            ds = rioxarray.open_rasterio(file_path)

            # Rename bands properly
            ds = ds.assign_coords(band=["Sigma0_VV", "Sigma0_VH"])  # , 'angle'])

            ds.attrs["time"] = self._parse_s1_filename(file_path)
            datasets.append(ds)
            times.append(ds.attrs["time"])
        print(f"Concatenating {len(datasets)} datasets")
        return xr.concat(datasets, dim=pd.Index(times, name="time"))

    def _process_polygon_stats(self, geotiff_path, geometry):
        """Process a single GeoTIFF file sequentially"""
        try:
            with rioxarray.open_rasterio(geotiff_path) as ds:
                # Clip to polygon
                clipped = ds.rio.clip([geometry.__geo_interface__], all_touched=True)

                # Calculate statistics sequentially
                means = clipped.mean(dim=["x", "y"], skipna=True)
                stds = clipped.std(dim=["x", "y"], skipna=True)
                medians = clipped.median(dim=["x", "y"], skipna=True)

                return {
                    "time": self._parse_s1_filename(geotiff_path),
                    "VV_mean": float(means.sel(band=1).values),
                    "VH_mean": float(means.sel(band=2).values),
                    "angle_mean": float(means.sel(band=3).values),
                    "VV_std": float(stds.sel(band=1).values),
                    "VH_std": float(stds.sel(band=2).values),
                    "VV_median": float(medians.sel(band=1).values),
                    "VH_median": float(medians.sel(band=2).values),
                }

        except Exception as e:
            print(f"Error processing file {geotiff_path}: {str(e)}")
            return None

    def get_timeseries_polygon_pandas(
        self,
        geometry: Polygon,
        orbit: str = "ASCENDING",
        date_range: Optional[tuple] = None,
    ) -> pd.DataFrame:
        """Extract time series statistics for a polygon - now with minimal geometry processing"""
        # Find intersecting slices using pre-cleaned geometries
        relevant_slices = []
        for row in self.intersecting_slices[orbit]:
            for slice_num in self.intersecting_slices[orbit][row]:
                if self.rows_dict[orbit][row][slice_num]["bounds"].intersects(geometry):
                    relevant_slices.append((row, slice_num))

        if not relevant_slices:
            return pd.DataFrame()

        # Process each relevant slice
        all_stats = []
        all_dates = set()

        for row, slice_num in relevant_slices:
            slice_data = self.rows_dict[orbit][row][slice_num]
            if not isinstance(slice_data["collection"], tuple):
                continue

            # Lazy load the data only when needed
            slice_dir, tiff_files = slice_data["collection"]

            # Filter by date range if provided
            filtered_files = tiff_files
            if date_range:
                start_date = pd.to_datetime(date_range[0])
                end_date = pd.to_datetime(date_range[1])
                filtered_files = [
                    f
                    for f in tiff_files
                    if start_date <= self._parse_s1_filename(f) <= end_date
                ]

            # Sort files by timestamp
            filtered_files.sort(key=self._parse_s1_filename)

            # Process each file
            for tiff_file in filtered_files:
                stats = self._process_polygon_stats(tiff_file, geometry)
                if stats is not None:
                    all_stats.append(stats)
                    all_dates.add(stats["time"].strftime("%Y-%m-%d"))

        # Create output DataFrame
        if not all_stats:
            return pd.DataFrame()

        stats_dict = {
            f"VV_{orbit}_mean": [[s["VV_mean"] for s in all_stats]],
            f"VH_{orbit}_mean": [[s["VH_mean"] for s in all_stats]],
            f"angle_{orbit}_mean": [[s["angle_mean"] for s in all_stats]],
            f"VV_{orbit}_std": [[s["VV_std"] for s in all_stats]],
            f"VH_{orbit}_std": [[s["VH_std"] for s in all_stats]],
            f"VV_{orbit}_median": [[s["VV_median"] for s in all_stats]],
            f"VH_{orbit}_median": [[s["VH_median"] for s in all_stats]],
            f"dates_{orbit}": [sorted(list(all_dates))],
        }

        return pd.DataFrame(stats_dict)

    def save_slice_bounds_to_parquet(self, output_path: str):
        """
        Save the bounds of each slice to a GeoParquet file.

        Args:
            output_path: Path where the GeoParquet file will be saved

        Returns:
            None
        """
        import geopandas as gpd

        # Create lists to store the data
        geometries = []
        orbits = []
        rows = []
        slice_numbers = []

        # Collect data from all slices
        for orbit in self.rows_dict:
            for row in self.rows_dict[orbit]:
                for slice_num, slice_data in self.rows_dict[orbit][row].items():
                    # Get the geometry
                    slice_geom = slice_data["bounds"]

                    # Append data
                    geometries.append(slice_geom)
                    orbits.append(orbit)
                    rows.append(row)
                    slice_numbers.append(slice_num)

        # Create a GeoDataFrame
        gdf = gpd.GeoDataFrame(
            {
                "orbit": orbits,
                "row": rows,
                "slice": slice_numbers,
                "geometry": geometries,
            }
        )

        # Add CRS if not already present
        if not gdf.crs:
            gdf.set_crs("EPSG:4326", inplace=True)

        # Save to GeoParquet
        gdf.to_parquet(output_path)

        print(f"Saved slice bounds to {output_path}")
        print(f"Total slices saved: {len(gdf)}")

    def describe_collections(self):
        """Print information about available data and intersecting slices"""
        print("\n=== Sentinel-1 Collection Structure ===\n")

        for orbit in self.orbits:  # ['ASCENDING', 'DESCENDING']:
            print(f"\n{orbit} Orbit:")
            print("=" * (len(orbit) + 7))

            if not self.intersecting_slices[orbit]:
                print("No intersecting slices found.")
                continue

            for row in sorted(self.intersecting_slices[orbit].keys()):
                print(f"\nRow {row}:")
                print("-" * (len(str(row)) + 4))

                for slice_num in sorted(self.intersecting_slices[orbit][row]):
                    slice_data = self.rows_dict[orbit][row][slice_num]
                    print(f"\nSlice {slice_num}:")

                    if isinstance(slice_data["collection"], tuple):
                        _, tiff_files = slice_data["collection"]
                        print(f"  Number of files: {len(tiff_files)}")
                        print(f"  Geometry: {slice_data['bounds']}")
                    else:
                        print("  No data available")

    def visualize_collection(self, Map, point=None):
        """
        Visualize the collection boundaries using geemap

        Args:
            Map: geemap Map object
            point: Optional point to visualize
            orbits: List of orbits to visualize ['ASCENDING', 'DESCENDING']
        """
        import random

        def random_hex_color():
            """Generate a random hex color"""
            r = random.randint(0, 255)
            g = random.randint(0, 255)
            b = random.randint(0, 255)
            return f"#{r:02x}{g:02x}{b:02x}"

        for orbit in self.orbits:
            for row in self.rows_dict[orbit]:
                print(f"Visualizing Row {row} ({orbit})")

                # Visualize each slice separately with its own random color
                for slice_num, slice_data in self.rows_dict[orbit][row].items():
                    # Generate random color
                    color = random_hex_color()

                    # Convert shapely geometry to ee.Geometry
                    slice_geom = ee.Geometry(slice_data["bounds"].__geo_interface__)

                    # Create feature with properties for labeling
                    feature = ee.Feature(
                        slice_geom,
                        {
                            "row": str(row),
                            "slice": str(slice_num),
                            "label": f"Row {row} Slice {slice_num}",
                        },
                    )

                    # Create feature collection for this slice
                    fc = ee.FeatureCollection([feature])

                    # Style with random color and transparency
                    styled_slice = fc.style(
                        color=color,  # Border color
                        fillColor="00000000",  # Fill color with 20% opacity
                        width=2,
                    )

                    # Add slice layer
                    Map.addLayer(styled_slice, {}, f"R:{row}_S:{slice_num} {orbit} ")

                # Add labels for slices
                # Create text properties
                # text_props = {
                #     'textColor': orbit_colors[orbit]['color'],
                #     'fontSize': 14,
                #     'outlineColor': 'white',
                #     'outlineWidth': 2
                # }

                # # Add labels at the centroid of each slice
                # labels = fc.map(lambda f: f.setGeometry(f.geometry().centroid())) \
                #         .style(**text_props, textField='label')

                # Map.addLayer(labels, {}, f'{orbit} Row {row} Labels')

        if point is not None:
            Map.addLayer(point, {"color": "yellow"}, "POI")

        # Center map on AOI
        Map.centerObject(self.aoi, 7)


class Sentinel1RowCollectionMem:
    def __init__(
        self,
        aoi,
        start_date: str,
        end_date: str,
        local_data_dir: str,
        orbits: list = ["ASCENDING"],
        bounds_file: str = None,
    ):
        """
        Initialize with data loaded and clipped to AOI

        Args:
            aoi: Earth Engine geometry for area of interest
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            local_data_dir: Directory containing S1 data organized as:
                          orbit/row/slice/tiff_files
            bounds_file: Optional path to geoparquet file containing saved bounds

        """
        self.aoi = aoi
        self.start_date = start_date
        self.end_date = end_date
        self.data_dir = Path(local_data_dir)
        self.orbits = orbits

        if bounds_file is not None:
            # Load from saved bounds
            import geopandas as gpd

            gdf = gpd.read_parquet(bounds_file)

            # Convert GeoDataFrame to rows_dict structure
            self.rows_dict = {}
            for orbit in gdf["orbit"].unique():
                self.rows_dict[orbit] = {}
                orbit_data = gdf[gdf["orbit"] == orbit]

                for row in orbit_data["row"].unique():
                    self.rows_dict[orbit][row] = {}
                    row_data = orbit_data[orbit_data["row"] == row]

                    for _, record in row_data.iterrows():
                        self.rows_dict[orbit][row][record["slice_num"]] = {
                            "bounds": record["geometry"],
                            "data": None,
                        }

            # Initialize data only
            self._initialize_data_only()
        else:
            # Get row/slice information from GEE and load data
            self.rows_dict = self._initialize_and_load_data()

        # self.describe_collections()

    def _get_row_collections_with_slices(self) -> Dict:
        """Get row/slice information from GEE"""
        s1_collection = ee.ImageCollection("COPERNICUS/S1_GRD_FLOAT")

        filtered_collection = (
            s1_collection.filterDate(self.start_date, self.end_date)
            .filterBounds(self.aoi)
            .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
            .filter(ee.Filter.eq("instrumentMode", "IW"))
            .filter(ee.Filter.eq("resolution_meters", 10))
        )

        rows_dict = {"ASCENDING": {}, "DESCENDING": {}}

        for orbit in self.orbits:
            orbit_collection = filtered_collection.filter(
                ee.Filter.eq("orbitProperties_pass", orbit)
            )

            def get_row_slice(image):
                return ee.Feature(
                    None,
                    {
                        "row": image.get("relativeOrbitNumber_start"),
                        "slice": image.get("sliceNumber"),
                    },
                )

            row_slices = (
                orbit_collection.map(get_row_slice)
                .distinct(["row", "slice"])
                .getInfo()["features"]
            )

            for feature in row_slices:
                row = feature["properties"]["row"]
                slice_num = feature["properties"]["slice"]

                if row not in rows_dict[orbit]:
                    rows_dict[orbit][row] = {}

                # Get geometry for this specific row/slice combination
                slice_collection = orbit_collection.filter(
                    ee.Filter.eq("relativeOrbitNumber_start", row)
                ).filter(ee.Filter.eq("sliceNumber", slice_num))

                slice_geom = slice_collection.geometry().getInfo()
                slice_geom = make_valid(shape(slice_geom))

                if not slice_geom.is_valid:
                    print(
                        f"Warning: Invalid geometry for {orbit} row {row} slice {slice_num}"
                    )
                    continue

                rows_dict[orbit][row][slice_num] = {
                    "bounds": slice_geom,
                    "data": None,  # Will be filled with xarray dataset
                }

        return rows_dict

    def _parse_s1_filename(self, filename: str) -> datetime:
        """Extract datetime from S1 filename"""
        pattern = r"\d{8}T\d{6}"
        match = re.search(pattern, str(filename))
        if match:
            return datetime.strptime(match.group(), "%Y%m%dT%H%M%S")
        raise ValueError(f"Could not parse datetime from filename: {filename}")

    # #parallel read of slices

    # version that uses xr.open_mfdataset
    def _load_slice_data(
        self, tiff_files: List[Path], slice_geom: Polygon, aoi_geojson: dict
    ) -> xr.Dataset:
        """Load slice data using xr.open_mfdataset with parallel processing"""
        import xarray as xr
        import rioxarray
        import numpy as np
        from functools import partial

        # Sort files and create mapping of files to times
        sorted_files = sorted(tiff_files, key=self._parse_s1_filename)
        file_times = {str(f): self._parse_s1_filename(f) for f in sorted_files}

        def preprocess(ds, filename):
            """Preprocess function that receives both dataset and filename"""
            # Get intersection of slice geometry and AOI
            aoi_shape = shape(aoi_geojson)
            clip_geom = slice_geom.intersection(aoi_shape)

            # Clip to geometry
            ds = ds.rio.clip([clip_geom.__geo_interface__], all_touched=True)

            # Rename bands
            ds = ds.assign_coords(band=["Sigma0_VV", "Sigma0_VH", "angle"])

            # Add time coordinate from our mapping
            ds = ds.assign_coords(time=file_times[filename])

            return ds

        # Open all files as a single dataset with parallel processing
        ds = xr.open_mfdataset(
            sorted_files,
            engine="rasterio",
            concat_dim="time",
            combine="nested",
            preprocess=lambda ds: preprocess(ds, ds.encoding.get("source", "")),
            parallel=False,
            chunks={"time": 1},  # Chunk along time dimension
        )

        return ds

    def _initialize_and_load_data(self) -> Dict:
        """Initialize structure and load all relevant data into memory"""
        # Get row/slice information from GEE
        self.rows_dict = self._get_row_collections_with_slices()

        # Use _initialize_data_only to handle data loading
        self._initialize_data_only()

        return self.rows_dict

    def _initialize_data_only(self) -> Dict:
        """Initialize data loading without getting bounds from GEE"""

        def get_memory_usage():
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024 / 1024

        print("\nInitial rows_dict structure:")
        for orbit in self.rows_dict:
            print(f"{orbit}:", list(self.rows_dict[orbit].keys()))

        # Convert AOI to GeoJSON once
        aoi_geojson = self.aoi.getInfo()
        aoi_shape = shape(aoi_geojson)

        # Initialize intersecting_slices dictionary
        self.intersecting_slices = {orbit: {} for orbit in self.orbits}

        total_slices = sum(
            sum(len(self.rows_dict[orbit][row]) for row in self.rows_dict[orbit])
            for orbit in self.rows_dict
        )

        initial_memory = get_memory_usage()
        print(f"Initial memory usage: {initial_memory:.2f} GB")

        with tqdm(total=total_slices, desc="Loading slices") as pbar:
            for orbit in self.rows_dict:
                for row in self.rows_dict[orbit]:
                    for slice_num in self.rows_dict[orbit][row]:
                        print(f"Processing row {row} : slice {slice_num}")
                        slice_data = self.rows_dict[orbit][row][slice_num]

                        # Check if slice intersects AOI
                        if slice_data["bounds"].intersects(aoi_shape):
                            slice_dir = (
                                self.data_dir
                                / orbit.lower()
                                / str(row)
                                / str(slice_num)
                            )
                            if slice_dir.exists():
                                # Add to intersecting_slices
                                if row not in self.intersecting_slices[orbit]:
                                    self.intersecting_slices[orbit][row] = []
                                self.intersecting_slices[orbit][row].append(slice_num)

                                tiff_files = list(
                                    slice_dir.glob("*_COG_Orb_NR_Cal_Spk_TC.tif")
                                )
                                print(f"Found {len(tiff_files)} files")
                                if tiff_files:
                                    try:
                                        print("Attempting to load data...")
                                        ds = self._load_slice_data(
                                            tiff_files,
                                            slice_data["bounds"],
                                            aoi_geojson,
                                        )
                                        # Print dataset info
                                        print(f"Successfully loaded dataset:")
                                        print(f"Dimensions: {dict(ds.dims)}")
                                        print(f"Variables: {list(ds.variables)}")

                                        slice_data["data"] = ds
                                        print(
                                            f"Stored data for {orbit} row {row} slice {slice_num}"
                                        )

                                        # Print memory usage
                                        current_memory = get_memory_usage()
                                        memory_delta = current_memory - initial_memory
                                        print(
                                            f"Current memory usage: {current_memory:.2f} GB"
                                        )
                                        print(
                                            f"Memory increase: {current_memory - initial_memory:.2f} GB"
                                        )
                                        pbar.set_postfix(
                                            {
                                                "orbit": orbit,
                                                "row": row,
                                                "slice": slice_num,
                                                "memory": f"{current_memory:.2f}GB",
                                                "memory_delta": f"+{memory_delta:.2f}GB",
                                            }
                                        )
                                    except Exception as e:
                                        print(
                                            f"\nError loading {orbit} row {row} slice {slice_num}: {str(e)}"
                                        )
                                else:
                                    print("No TIFF files found")
                            else:
                                print("Directory does not exist")
                        else:
                            print("Does not intersect with AOI")

                        pbar.update(1)

        final_memory = get_memory_usage()
        print(f"\nFinal memory usage: {final_memory:.2f} GB")
        print(f"Total memory increase: {final_memory - initial_memory:.2f} GB")

        print("\nFinal intersecting_slices structure:")
        for orbit in self.intersecting_slices:
            print(f"{orbit}:", self.intersecting_slices[orbit])

    def get_timeseries_polygon_pandas(
        self, geometry: Polygon, orbit: str = "ASCENDING"
    ) -> pd.DataFrame:
        """Extract time series statistics for a polygon"""
        # Find intersecting slices
        relevant_slices = []

        time00 = time.time()
        for row in self.intersecting_slices[orbit]:
            for slice_num in self.intersecting_slices[orbit][row]:
                if self.rows_dict[orbit][row][slice_num]["bounds"].intersects(geometry):
                    relevant_slices.append((row, slice_num))
        time01 = time.time()
        print(f"Finding relevant slices took {time01 - time00:.2f} seconds")
        print(f"Relevant slices: {relevant_slices}")

        if not relevant_slices:
            return pd.DataFrame()

        # Process each relevant slice
        all_stats = []
        all_dates = set()

        for row, slice_num in relevant_slices:
            print(f"Processing row {row} : slice {slice_num}")
            slice_data = self.rows_dict[orbit][row][slice_num]

            # Debug print
            print(f"Data keys: {slice_data.keys()}")
            print(f"Data type: {type(slice_data['data'])}")

            ds = slice_data["data"]
            print(f"Dataset dimensions: {dict(ds.dims)}")
            print(f"Dataset variables: {list(ds.variables)}")

            t0 = time.time()

            # Clip to polygon
            try:
                clipped = ds.rio.clip([geometry.__geo_interface__], all_touched=True)
                print(f"Clipped shape: {dict(clipped.dims)}")

                t1 = time.time()
                print(f"Clipping took {t1 - t0:.2f} seconds")

                # Get the actual data array
                data = clipped["band_data"]

                # Calculate statistics for all times at once
                means = data.mean(dim=["y", "x"], skipna=True)
                t2 = time.time()
                print(f"Means calculated in {t2 - t1:.2f} seconds")
                stds = data.std(dim=["y", "x"], skipna=True)
                t3 = time.time()
                print(f"Stds calculated in {t3 - t2:.2f} seconds")
                medians = data.median(dim=["y", "x"], skipna=True)
                t4 = time.time()
                print(f"Medians calculated in {t4 - t3:.2f} seconds")

                # Process each timestamp
                for t_idx, t in enumerate(ds.time.values):
                    stats_dict = {
                        "time": pd.to_datetime(t),
                        "VV_mean": float(means.isel(time=t_idx, band=0)),
                        "VH_mean": float(means.isel(time=t_idx, band=1)),
                        "angle_mean": float(means.isel(time=t_idx, band=2)),
                        "VV_std": float(stds.isel(time=t_idx, band=0)),
                        "VH_std": float(stds.isel(time=t_idx, band=1)),
                        "VV_median": float(medians.isel(time=t_idx, band=0)),
                        "VH_median": float(medians.isel(time=t_idx, band=1)),
                    }
                    all_stats.append(stats_dict)
                    all_dates.add(pd.to_datetime(t).strftime("%Y-%m-%d"))
                t5 = time.time()
                print(f"Processing took {t5 - t0:.2f} seconds")

            except Exception as e:
                print(f"Error processing slice: {str(e)}")
                import traceback

                traceback.print_exc()
                continue

        # Create output DataFrame
        if not all_stats:
            return pd.DataFrame()

        stats_dict = {
            f"VV_{orbit}_mean": [[s["VV_mean"] for s in all_stats]],
            f"VH_{orbit}_mean": [[s["VH_mean"] for s in all_stats]],
            f"angle_{orbit}_mean": [[s["angle_mean"] for s in all_stats]],
            f"VV_{orbit}_std": [[s["VV_std"] for s in all_stats]],
            f"VH_{orbit}_std": [[s["VH_std"] for s in all_stats]],
            f"VV_{orbit}_median": [[s["VV_median"] for s in all_stats]],
            f"VH_{orbit}_median": [[s["VH_median"] for s in all_stats]],
            f"dates_{orbit}": [sorted(list(all_dates))],
        }

        return pd.DataFrame(stats_dict)

    def describe_collections(self):
        """Print information about available data and intersecting slices"""
        print("\n=== Sentinel-1 Collection Structure ===\n")

        for orbit in self.orbits:
            print(f"\n{orbit} Orbit:")
            print("=" * (len(orbit) + 7))

            if not self.intersecting_slices[orbit]:
                print("No intersecting slices found.")
                continue

            for row in sorted(self.intersecting_slices[orbit].keys()):
                print(f"\nRow {row}:")
                print("-" * (len(str(row)) + 4))

                for slice_num in sorted(self.intersecting_slices[orbit][row]):
                    slice_data = self.rows_dict[orbit][row][slice_num]
                    print(f"\nSlice {slice_num}:")

                    ds = slice_data.get("data")
                    if ds is not None:
                        print(f"  Dimensions: {dict(ds.dims)}")
                        print(f"  Variables: {list(ds.variables)}")
                        print(f"  Coordinates: {list(ds.coords)}")

                        # Calculate memory usage
                        memory_usage = ds.nbytes / (1024**2)  # Convert to MB
                        print(f"  Memory Usage: {memory_usage:.2f} MB")

                        # Print bounds
                        print(f"  Geometry bounds: {slice_data['bounds'].bounds}")

                        # Try to print time range if available
                        if "time" in ds.coords:
                            print(
                                f"  Time range: {ds.time.values[0]} to {ds.time.values[-1]}"
                            )
                            print(f"  Number of timesteps: {len(ds.time)}")
                    else:
                        print("  No data available")

    def visualize_collection(self, Map, point=None):
        """
        Visualize the collection boundaries using geemap

        Args:
            Map: geemap Map object
            point: Optional point to visualize
            orbits: List of orbits to visualize ['ASCENDING', 'DESCENDING']
        """
        import random

        def random_hex_color():
            """Generate a random hex color"""
            r = random.randint(0, 255)
            g = random.randint(0, 255)
            b = random.randint(0, 255)
            return f"#{r:02x}{g:02x}{b:02x}"

        for orbit in self.orbits:
            for row in self.rows_dict[orbit]:
                print(f"Visualizing Row {row} ({orbit})")

                # Visualize each slice separately with its own random color
                for slice_num, slice_data in self.rows_dict[orbit][row].items():
                    # Generate random color
                    color = random_hex_color()

                    # Convert shapely geometry to ee.Geometry
                    slice_geom = ee.Geometry(slice_data["bounds"].__geo_interface__)

                    # Create feature with properties for labeling
                    feature = ee.Feature(
                        slice_geom,
                        {
                            "row": str(row),
                            "slice": str(slice_num),
                            "label": f"Row {row} Slice {slice_num}",
                        },
                    )

                    # Create feature collection for this slice
                    fc = ee.FeatureCollection([feature])

                    # Style with random color and transparency
                    styled_slice = fc.style(
                        color=color,  # Border color
                        fillColor="00000000",  # Fill color with 20% opacity
                        width=2,
                    )

                    # Add slice layer
                    Map.addLayer(styled_slice, {}, f"R:{row}_S:{slice_num} {orbit} ")

        if point is not None:
            Map.addLayer(point, {"color": "yellow"}, "POI")

        # Center map on AOI
        Map.centerObject(self.aoi, 7)

    def save_row_bounds(self, output_file: str):
        """Save row/slice bounds to a geoparquet file"""
        import geopandas as gpd

        # Create list of records
        records = []
        for orbit in self.rows_dict:
            for row in self.rows_dict[orbit]:
                for slice_num, slice_data in self.rows_dict[orbit][row].items():
                    records.append(
                        {
                            "orbit": orbit,
                            "row": row,
                            "slice_num": slice_num,
                            "geometry": slice_data["bounds"],
                        }
                    )

        # Convert to GeoDataFrame
        gdf = gpd.GeoDataFrame(records)
        gdf.to_parquet(output_file)
        print(f"Saved bounds to {output_file}")
