# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import ee
import pandas as pd
from typing import Tuple
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist

import matplotlib.pyplot as plt
from typing import Tuple, Optional
from ricemapper.utils.utils import convert_tuple2dates, stringify_date
from dateutil.relativedelta import relativedelta
import ricemapper.utils.s1_processing.bnc as bnc
import ricemapper.utils.s1_processing.speckle as sf
import ricemapper.utils.s1_processing.terrain as trf
from shapely.geometry import Polygon


class Sentinel1RowCollections:
    def __init__(
        self,
        aoi,
        start_date,
        end_date,
        save_dir=None,
        focal_median_radius_px=None,
        # Preprocessing parameters
        border_noise_correction=True,
        terrain_correction=True,
        speckle_filter_framework="MULTI",
        speckle_kernel_size=7,
        speckle_filter_type="REFINED LEE",
        speckle_filter_num_images=10,
    ):
        self.aoi = aoi
        self.start_date = start_date
        self.end_date = end_date

        self.save_dir = save_dir
        self.border_noise_correction = border_noise_correction
        self.terrain_correction = terrain_correction

        # Make a projection to discover the scale in degrees.
        self.proj = ee.Projection("EPSG:4326").atScale(10).getInfo()

        # Get scales out of the transform.
        self.scale_x = self.proj["transform"][0]
        self.scale_y = -self.proj["transform"][4]

        self.focal_median_radius_px = focal_median_radius_px

        frame_needed = ["MONO", "MULTI", None]
        if speckle_filter_framework not in frame_needed:
            raise ValueError("ERROR!!! SPECKLE_FILTER_FRAMEWORK not correctly defined")
        else:
            self.speckle_filter_framework = speckle_filter_framework
            self.speckle_kernel_size = speckle_kernel_size
            self.speckle_filter_num_images = speckle_filter_num_images
            format_sfilter = ["BOXCAR", "LEE", "GAMMA MAP", "REFINED LEE", "LEE SIGMA"]

            if speckle_filter_type not in format_sfilter:
                raise ValueError(
                    f"Invalid speckle filter type. Choose from {format_sfilter}"
                )
            else:
                self.speckle_filter_type = speckle_filter_type

        # Initialize the collection
        self.rows_dict = self._get_row_collections()
        # print the number of keys in self.rows_dict['ASCENDING']
        print(
            f"Number of keys in self.rows_dict['ASCENDING']: {len(self.rows_dict['ASCENDING'].keys())}"
        )
        # print the number of keys in self.rows_dict['DESCENDING']
        print(
            f"Number of keys in self.rows_dict['DESCENDING']: {len(self.rows_dict['DESCENDING'].keys())}"
        )

    def _apply_preprocessing(self, s1):
        ###########################################
        # 2. ADDITIONAL BORDER NOISE CORRECTION
        ###########################################

        if self.border_noise_correction:
            s1_1 = s1.map(bnc.f_mask_edges)
            print("Additional border noise correction is completed")
        else:
            s1_1 = s1
        ########################
        # 3. SPECKLE FILTERING
        #######################

        if self.speckle_filter_framework is not None:
            if self.speckle_filter_framework == "MONO":
                s1_1 = ee.ImageCollection(
                    sf.MonoTemporal_Filter(
                        s1_1, self.speckle_kernel_size, self.speckle_filter_type
                    )
                )
                print("Mono-temporal speckle filtering is completed")
            else:
                s1_1 = ee.ImageCollection(
                    sf.MultiTemporal_Filter(
                        s1_1,
                        self.speckle_kernel_size,
                        self.speckle_filter_type,
                        self.speckle_filter_num_images,
                    )
                )
                print("Multi-temporal speckle filtering is completed")

        ########################
        # 4. TERRAIN CORRECTION
        #######################

        if self.terrain_correction:
            s1_1 = trf.slope_correction(s1_1, "DIRECT", ee.Image("CGIAR/SRTM90_V4"), 0)

            print("Radiometric terrain normalization is completed")

        return s1_1

    def _get_row_collections(self):
        # Define the Sentinel-1 collection
        s1_collection = ee.ImageCollection("COPERNICUS/S1_GRD_FLOAT")

        # Filter the collection by date and AOI
        filtered_collection = (
            s1_collection.filterDate(self.start_date, self.end_date)
            .filterBounds(self.aoi)
            .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
            .filter(ee.Filter.eq("instrumentMode", "IW"))
            .filter(ee.Filter.eq("resolution_meters", 10))
        )

        self.processed_collection = self._apply_preprocessing(filtered_collection)

        # Print size of collection
        print(
            f"Number of items in collection: {self.processed_collection.size().getInfo()}"
        )

        print("Generating orbit rows ...")

        rows_dict = {"ASCENDING": {}, "DESCENDING": {}}

        for orbit in ["ASCENDING", "DESCENDING"]:
            orbit_collection = self.processed_collection.filter(
                ee.Filter.eq("orbitProperties_pass", orbit)
            )
            rows = (
                orbit_collection.aggregate_array("relativeOrbitNumber_start")
                .distinct()
                .getInfo()
            )

            for row in rows:
                row_collection = orbit_collection.filter(
                    ee.Filter.eq("relativeOrbitNumber_start", row)
                )

                def round_date(image):
                    return image.set(
                        "group_date",
                        ee.Date(
                            ee.Date(image.get("system:time_start")).format("YYYY-MM-dd")
                        ).millis(),
                    )

                grouped_collection = row_collection.map(round_date)
                distinct_dates = grouped_collection.aggregate_array(
                    "group_date"
                ).distinct()

                def create_daily_mosaic(date):
                    daily_images = grouped_collection.filter(
                        ee.Filter.eq("group_date", date)
                    )
                    return daily_images.mosaic().set("system:time_start", date)

                daily_mosaics = ee.ImageCollection(
                    distinct_dates.map(create_daily_mosaic)
                )

                # Merge the geometry of the row_collection
                row_geometry = row_collection.geometry()

                rows_dict[orbit][row] = {
                    "collection": daily_mosaics,
                    "bounds": row_geometry,
                }

        return rows_dict

    def query_point(self, point, orbit="ASCENDING", start_date=None, end_date=None):
        # Find the row that contains the point
        containing_row = None
        for row, data in self.rows_dict[orbit].items():
            if data["bounds"].contains(point).getInfo():
                containing_row = row
                break

        if containing_row is None:
            return None

        # Extract time series for the point
        collection = self.rows_dict[orbit][containing_row]["collection"]

        # Filter by date range if provided
        if start_date and end_date:
            collection = collection.filterDate(start_date, end_date)

        def extract_values(image):
            values = image.select(["VV", "VH"]).reduceRegion(
                reducer=ee.Reducer.first(), geometry=point, scale=10
            )
            return ee.Feature(
                None,
                {
                    "date": image.date().format("YYYY-MM-dd"),
                    "VV": values.get("VV"),
                    "VH": values.get("VH"),
                    "orbit": orbit,
                },
            )

        time_series = collection.map(extract_values).getInfo()

        # Convert to pandas DataFrame
        df = pd.DataFrame(
            [feature["properties"] for feature in time_series["features"]]
        )
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()

        return df

    def visualize_collection(
        self, Map, point=None, orbits=["ASCENDING", "DESCENDING"], error_margin=1
    ):

        colors = {"ASCENDING": "red", "DESCENDING": "blue"}
        for orbit in orbits:
            for row, data in self.rows_dict[orbit].items():
                print(f"Visualizing Row {row} ({orbit})")

                # Add row bounds
                bounds_feature = ee.Feature(data["bounds"])
                styled_bounds = ee.FeatureCollection([bounds_feature]).style(
                    color=colors[orbit],  # Red color
                    fillColor="00000000",  # Transparent fill
                    width=2,  # Line width
                )
                Map.addLayer(styled_bounds, {}, f"Bounds Row {row} ({orbit})")

        if point is not None:
            Map.addLayer(point, {}, "POI")

        Map.centerObject(self.aoi, 7)

    def get_timeseries_image(
        self,
        point: Tuple[float, float],
        size_px: int = 64,
        orbit: str = "ASCENDING",
        index: int = 1,
        label: str = "test",
        file_format="GEO_TIFF",
    ):
        # file format can be 'GEO_TIFF' or 'NPY'
        # Create an ee.Geometry.Point from the input coordinates
        ee_point = ee.Geometry.Point(point)

        # Find the row that contains the point
        containing_row = None
        for row, data in self.rows_dict[orbit].items():
            if data["bounds"].contains(ee_point).getInfo():
                containing_row = row
                break

        if containing_row is None:
            print(f"No data found for the given point in {orbit} orbit.")
            return

        # Get the collection for the containing row
        collection = self.rows_dict[orbit][containing_row]["collection"]

        file_postfix = "tif" if file_format == "GEO_TIFF" else "npy"

        # Function to process and save each image
        def process_and_save_image(image):
            date = (
                ee.Date(image.get("system:time_start")).format("YYYY-MM-dd").getInfo()
            )

            if self.focal_median_radius_px is not None:
                expr = image.select(["VV", "VH"]).focal_median(
                    self.focal_median_radius_px
                )
            else:
                expr = image.select(["VV", "VH"])

            # Set up the request for ee.data.computePixels
            request = {
                "expression": expr,
                "fileFormat": file_format,
                "bandIds": ["VV", "VH"],
                "grid": {
                    "dimensions": {"width": size_px, "height": size_px},
                    "affineTransform": {
                        "scaleX": self.scale_x,
                        "shearX": 0,
                        "translateX": point[0],
                        "shearY": 0,
                        "scaleY": self.scale_y,
                        "translateY": point[1],
                    },
                    "crsCode": self.proj["crs"],
                },
            }

            try:
                # Compute pixels
                image_data = ee.data.computePixels(request)

                # Save the image
                filename = (
                    f"s1_{index}_{label}_{date}_{orbit}_{containing_row}.{file_postfix}"
                )

                # Save to self.save_dir
                if self.save_dir:
                    filename = f"{self.save_dir}/{filename}"

                with open(filename, "wb") as f:
                    f.write(image_data)
                print(f"Saved {filename}")
                return True
            except Exception as e:
                print(f"Failed to compute or save image for {date}: {str(e)}")
                return False

        # Process all images in the collection
        images = collection.toList(collection.size())
        num_images = images.size().getInfo()
        successful_saves = 0

        for i in range(num_images):
            image = ee.Image(images.get(i))
            if process_and_save_image(image):
                successful_saves += 1

        print(f"Processed {num_images} images, successfully saved {successful_saves}.")

    def get_aggregated_timeseries(
        self,
        ee_geometry: ee.Geometry,
        collection: ee.ImageCollection,
        orbit,
        scale: int = 10,
        max_pixels: int = 1e8,
        chunk_size: int = 100,
    ):
        """
        Memory-efficient function to compute mean and standard deviation of VV and VH polarizations
        for a single orbit (ascending or descending) within a polygon for each time step, using a chunking strategy.

        Args:
        geometry (Polygon): A shapely Polygon object defining the area of interest.
        collection (ee.ImageCollection): The Sentinel-1 image collection for a single orbit (ascending or descending).
        date_range (Optional[Tuple[str, str]]): A tuple of start and end dates in 'YYYY-MM-dd' format.
        scale (int): The scale in meters of the projection to work in. Default is 10 (Sentinel-1 resolution).
        max_pixels (int): The maximum number of pixels to reduce in reduceRegion. Default is 1e8.
        chunk_size (int): The number of images to process in each chunk. Default is 100.

        Returns:
        pd.DataFrame: A single-row DataFrame with columns for mean and standard deviation of VV and VH as a timeseries.
        """

        # Filter the collection to include only images that intersect with the geometry
        collection = collection.filterBounds(ee_geometry)

        # Function to calculate mean and standard deviation for VV and VH bands
        def calculate_stats(image):
            reduced = image.select(["VV", "VH", "angle"]).reduceRegion(
                reducer=ee.Reducer.mean()
                .combine(ee.Reducer.stdDev(), None, True)
                .combine(ee.Reducer.median(), None, True),
                geometry=ee_geometry,
                scale=scale,
                maxPixels=max_pixels,
                bestEffort=True,
            )
            return ee.Feature(
                None,
                {
                    "VV_mean": reduced.get("VV_mean"),
                    "VH_mean": reduced.get("VH_mean"),
                    "VV_stdDev": reduced.get("VV_stdDev"),
                    "VH_stdDev": reduced.get("VH_stdDev"),
                    "angle_mean": reduced.get("angle_mean"),
                    "VH_median": reduced.get("VH_median"),
                    "VV_median": reduced.get("VV_median"),
                    "date": image.date().format("YYYY-MM-dd"),
                },
            )

        # Function to process a chunk of the collection
        def process_chunk(chunk):
            return chunk.map(calculate_stats).getInfo()["features"]

        try:
            # Get the total number of images
            total_images = collection.size().getInfo()

            if total_images == 0:
                print("No images found in the collection for the given parameters.")
                return pd.DataFrame()

            # Process the collection in chunks
            all_features = []
            for i in range(0, total_images, chunk_size):
                chunk = collection.toList(chunk_size, i)
                chunk_features = process_chunk(ee.ImageCollection(chunk))
                all_features.extend(chunk_features)

            # Function to extract time series data
            def extract_timeseries(stats_list, band, stat):
                return [
                    s["properties"][f"{band}_{stat}"]
                    for s in stats_list
                    if f"{band}_{stat}" in s["properties"]
                ], [
                    s["properties"]["date"]
                    for s in stats_list
                    if "date" in s["properties"]
                ]

            # Extract time series for each polarization and statistic
            vv_mean, dates = extract_timeseries(all_features, "VV", "mean")
            vh_mean, _ = extract_timeseries(all_features, "VH", "mean")
            vv_std, _ = extract_timeseries(all_features, "VV", "stdDev")
            vh_std, _ = extract_timeseries(all_features, "VH", "stdDev")
            vv_median, _ = extract_timeseries(all_features, "VV", "median")
            vh_median, _ = extract_timeseries(all_features, "VH", "median")
            angle_mean, _ = extract_timeseries(all_features, "angle", "mean")

            try:
                if len(angle_mean) > 1:
                    angle_mean = angle_mean[0]  # first in the timeseries
            except:
                angle_mean = 0

            # Create the DataFrame
            df = pd.DataFrame(
                {
                    f"VV_{orbit}_mean": [vv_mean],
                    f"VH_{orbit}_mean": [vh_mean],
                    f"VV_{orbit}_std": [vv_std],
                    f"VH_{orbit}_std": [vh_std],
                    f"VV_{orbit}_median": [vv_median],
                    f"VH_{orbit}_median": [vh_median],
                    f"angle_{orbit}_mean": angle_mean,
                    f"dates_{orbit}": [dates],
                }
            )

            return df

        except ee.EEException as e:
            print(f"An Earth Engine error occurred: {str(e)}")
            return pd.DataFrame()

    def get_timeseries_polygon_pandas(
        self,
        date_range: Optional[tuple],
        geometry: Polygon,
        orbit: str = "ASCENDING",
        buffer=-10,
    ):

        ee_geometry = ee.Geometry.Polygon(list(geometry.exterior.coords))
        ee_geometry = ee_geometry.buffer(distance=buffer)

        assert (
            len(ee_geometry.getInfo()["coordinates"]) > 0
        ), f"Invalid Polygon geometry: {geometry}"

        # Find the row that contains the point
        containing_row = None
        for row, data in self.rows_dict[orbit].items():
            C = self.rows_dict[orbit][row]["collection"]
            # if data['bounds'].contains(ee_centroid).getInfo():
            if (
                data["bounds"].contains(ee_geometry).getInfo()
                and C.filterBounds(ee_geometry).size().getInfo() > 0
            ):
                containing_row = row
                break

        if containing_row is None:
            print(
                f"ERROR! No data found for the given Polygon geom: {geometry} | ee_geom: {ee_geometry.getInfo()}"
            )
            return None

        # Get the collection for the containing row
        assert (
            C.filterBounds(ee_geometry).size().getInfo() > 0
        ), f"No data found for the given Polygon in row: {containing_row} for geometry {ee_geometry.getInfo()}"

        if date_range is not None:
            start_date, end_date = convert_tuple2dates(date_range)
            start_date_str, end_date_str = stringify_date([start_date, end_date])
            C = C.filterDate(start_date_str, end_date_str)

        return self.get_aggregated_timeseries(ee_geometry, C, orbit)

    def get_timeseries_image_pandas(
        self,
        date_range: Optional[tuple],
        point: Tuple[float, float],
        size_px: int = 64,
        orbit: str = "ASCENDING",
    ):
        """
        Given a point, fetch the timeseries data for the given point and return a pandas DataFrame.
        """

        ee_point = ee.Geometry.Point(point)

        # Find the row that contains the point
        containing_row = None
        for row, data in self.rows_dict[orbit].items():
            if data["bounds"].contains(ee_point).getInfo():
                containing_row = row
                break

        if containing_row is None:
            print(f"No data found for the given point in {orbit} orbit.")
            return None

        # Get the collection for the containing row
        collection = self.rows_dict[orbit][containing_row]["collection"]

        if date_range is not None:
            start_date, end_date = convert_tuple2dates(date_range)
            start_date_str, end_date_str = stringify_date([start_date, end_date])
            collection = collection.filterDate(start_date_str, end_date_str)

        # Initialize dict
        pixel_data = {"VV": {}, "VH": {}}
        timestamps = []

        # Function to process each image
        def process_image(image):
            date = (
                ee.Date(image.get("system:time_start")).format("YYYY-MM-dd").getInfo()
            )

            if self.focal_median_radius_px is not None:
                expr = image.select(["VV", "VH"]).focal_median(
                    self.focal_median_radius_px
                )
            else:
                expr = image.select(["VV", "VH"])

            request = {
                "expression": expr,
                "fileFormat": "NUMPY_NDARRAY",
                "bandIds": ["VV", "VH"],
                "grid": {
                    "dimensions": {"width": size_px, "height": size_px},
                    "affineTransform": {
                        "scaleX": self.scale_x,
                        "shearX": 0,
                        "translateX": point[0],
                        "shearY": 0,
                        "scaleY": self.scale_y,
                        "translateY": point[1],
                    },
                    "crsCode": self.proj["crs"],
                },
            }

            try:
                npy_arr = ee.data.computePixels(request)
                vv_data, vh_data = npy_arr["VV"], npy_arr["VH"]

                for i in range(size_px):
                    for j in range(size_px):
                        pixel_key = (i, j)
                        if pixel_key not in pixel_data["VV"]:
                            pixel_data["VV"][pixel_key] = []
                            pixel_data["VH"][pixel_key] = []
                        pixel_data["VV"][pixel_key].append(float(vv_data[i, j]))
                        pixel_data["VH"][pixel_key].append(float(vh_data[i, j]))

                # timestamps.append(pd.to_datetime(date, format='%Y-%m-%d'))
                timestamps.append(date)
                return True
            except Exception as e:
                print(f"Failed to compute image for {date}: {str(e)}")
                return False

        # Process all images in the collection
        images = collection.toList(collection.size())
        num_images = images.size().getInfo()

        for i in range(num_images):
            image = ee.Image(images.get(i))
            process_image(image)

        # Create DataFrame
        df_rows = []
        for i in range(size_px):
            for j in range(size_px):
                lat = point[1] + j * self.scale_y
                lon = point[0] + i * self.scale_x
                pixel_key = (i, j)

                df_rows.append(
                    {
                        "latitude": lat,
                        "longitude": lon,
                        "ts_VV": pixel_data["VV"][pixel_key],
                        "ts_VH": pixel_data["VH"][pixel_key],
                        "dates_s1": timestamps,
                        "orbit": orbit,
                    }
                )

        return pd.DataFrame(df_rows)

    def get_longterm_avg(
        self,
        date_range: tuple,
        point: tuple,
        size_px: int = 1,
        orbit: str = "ASCENDING",
        seasons: int = 5,
    ):
        """
        Given a point, fetch the long-term average data for the given point and return a pandas DataFrame.
        Inputs:
         - date_range: Tuple of start and end dates for the seasonal average: should be season dates
            - point: Tuple of (longitude, latitude) coordinates
            - size_px: Size of the pixel grid -> focal_median radius
            - orbit: 'ASCENDING' or 'DESCENDING'
            - seasons: Number of seasons to consider for the long-term average: this is the number of years __before__ the current year
        """

        ee_point = ee.Geometry.Point(point)

        # Find the row that contains the point
        containing_row = None
        for row, data in self.rows_dict[orbit].items():
            if data["bounds"].contains(ee_point).getInfo():
                containing_row = row
                break

        if containing_row is None:
            print(f"No data found for the given point in {orbit} orbit.")
            return None

        # Get the collection for the containing row
        collection = self.rows_dict[orbit][containing_row]["collection"]

        # Calculate date ranges for each season
        start_date, end_date = convert_tuple2dates(date_range)

        date_ranges = [
            (
                start_date - relativedelta(years=i + 1),
                end_date - relativedelta(years=i + 1),
            )
            for i in range(seasons)
        ]

        print(
            f"Getting long term avg values for {point[1]}, {point[0]} | from {date_ranges[-1][0]} to {date_ranges[0][1]}"
        )
        # Initialize dictionaries to store sum and count of values
        pixel_data_sum = {"VV": {}, "VH": {}}
        pixel_data_count = {"VV": {}, "VH": {}}

        # Iterate over each date range
        for sdate, edate in date_ranges:
            # Convert dates back to strings
            start_date_str, end_date_str = stringify_date([sdate, edate])

            # Filter the collection by the provided date range
            filtered_collection = collection.filterDate(start_date_str, end_date_str)

            # Calculate the mean image over the date range
            mean_image = filtered_collection.mean()

            if self.focal_median_radius_px is not None:
                mean_image = mean_image.focal_median(self.focal_median_radius_px)

            # Extract pixel values
            request = {
                "expression": mean_image.select(["VV", "VH"]),
                "fileFormat": "NUMPY_NDARRAY",
                "bandIds": ["VV", "VH"],
                "grid": {
                    "dimensions": {"width": size_px, "height": size_px},
                    "affineTransform": {
                        "scaleX": self.scale_x,
                        "shearX": 0,
                        "translateX": point[0],
                        "shearY": 0,
                        "scaleY": self.scale_y,
                        "translateY": point[1],
                    },
                    "crsCode": self.proj["crs"],
                },
            }

            try:
                npy_arr = ee.data.computePixels(request)
                vv_data, vh_data = npy_arr["VV"], npy_arr["VH"]

                for i in range(size_px):
                    for j in range(size_px):
                        pixel_key = (i, j)
                        if pixel_key not in pixel_data_sum["VV"]:
                            pixel_data_sum["VV"][pixel_key] = 0
                            pixel_data_sum["VH"][pixel_key] = 0
                            pixel_data_count["VV"][pixel_key] = 0
                            pixel_data_count["VH"][pixel_key] = 0
                        pixel_data_sum["VV"][pixel_key] += float(vv_data[i, j])
                        pixel_data_sum["VH"][pixel_key] += float(vh_data[i, j])
                        pixel_data_count["VV"][pixel_key] += 1
                        pixel_data_count["VH"][pixel_key] += 1

            except Exception as e:
                print(
                    f"Failed to compute mean image for date range {start_date_str} to {end_date_str}: {str(e)}"
                )

        # Calculate averages
        pixel_data_avg = {"VV": {}, "VH": {}}
        for key in pixel_data_sum["VV"]:
            pixel_data_avg["VV"][key] = (
                pixel_data_sum["VV"][key] / pixel_data_count["VV"][key]
            )
            pixel_data_avg["VH"][key] = (
                pixel_data_sum["VH"][key] / pixel_data_count["VH"][key]
            )

        # Now calculate the average for this season
        start_date_str = start_date.strftime("%Y-%m-%d")
        end_date_str = end_date.strftime("%Y-%m-%d")
        filtered_collection = collection.filterDate(start_date_str, end_date_str)

        # Calculate the mean image over the date range
        mean_image = filtered_collection.mean()

        if self.focal_median_radius_px is not None:
            mean_image = mean_image.focal_median(self.focal_median_radius_px)

        # Extract pixel values
        request = {
            "expression": mean_image.select(["VV", "VH"]),
            "fileFormat": "NUMPY_NDARRAY",
            "bandIds": ["VV", "VH"],
            "grid": {
                "dimensions": {"width": size_px, "height": size_px},
                "affineTransform": {
                    "scaleX": self.scale_x,
                    "shearX": 0,
                    "translateX": point[0],
                    "shearY": 0,
                    "scaleY": self.scale_y,
                    "translateY": point[1],
                },
                "crsCode": self.proj["crs"],
            },
        }

        try:
            npy_arr = ee.data.computePixels(request)
            vv_data, vh_data = npy_arr["VV"], npy_arr["VH"]

        except Exception as e:
            print(f"Failed to compute mean image for this season: {str(e)}")
            return None

        # Create DataFrame
        df_rows = []
        for i in range(size_px):
            for j in range(size_px):
                lat = point[1] + j * self.scale_y
                lon = point[0] + i * self.scale_x
                pixel_key = (i, j)

                df_rows.append(
                    {
                        "latitude": lat,
                        "longitude": lon,
                        "longterm_avg_VV": pixel_data_avg["VV"][pixel_key],
                        "longterm_avg_VH": pixel_data_avg["VH"][pixel_key],
                        "seasonal_avg_VV": float(vv_data[i, j]),
                        "seasonal_avg_VH": float(vh_data[i, j]),
                    }
                )

        return pd.DataFrame(df_rows)

    def group_and_combine(self, ts_df):
        def custom_aggregation(group):
            # Define the columns to keep all values
            keep_all_columns = ["Timeseries", "Band", "Orbit"]

            # Initialize a dictionary to hold the aggregated data
            aggregated_data = {}

            for column in group.columns:
                if column in keep_all_columns:
                    # Keep all values as a list
                    aggregated_data[column] = group[column].tolist()
                else:
                    # Keep the first occurrence
                    aggregated_data[column] = group[column].iloc[0]

            return pd.Series(aggregated_data)

        # Group by 'Latitude' and 'Longitude'
        grouped_df = (
            ts_df.groupby(["Latitude", "Longitude"])
            .apply(custom_aggregation)
            .reset_index(drop=True)
        )
        return grouped_df

    def reshape_dataframe(self, ts_df, feature_combinations=None):

        if feature_combinations is None:
            feature_combinations = [
                ("VV", "ASCENDING"),
                ("VV", "DESCENDING"),
                ("VH", "ASCENDING"),
                ("VH", "DESCENDING"),
            ]
        # Get unique combinations of 'Latitude' and 'Longitude' in the order they appear
        unique_samples = ts_df[["Latitude", "Longitude"]].drop_duplicates()

        # Initialize lists to hold the reshaped data
        reshaped_data = []

        for _, sample in unique_samples.iterrows():
            lat, lon = sample["Latitude"], sample["Longitude"]
            group = ts_df[(ts_df["Latitude"] == lat) & (ts_df["Longitude"] == lon)]

            # Initialize a list to hold the timeseries for each feature combination
            sample_data = []

            for band, orbit in feature_combinations:
                # Extract the timeseries for the current feature combination
                timeseries = group[(group["Band"] == band) & (group["Orbit"] == orbit)][
                    "Timeseries"
                ].values

                if len(timeseries) > 0:
                    # Use the first timeseries if multiple are found
                    sample_data.append(timeseries[0])
                else:
                    # If no timeseries is found, append an array of NaNs with the same length as the first found timeseries
                    sample_data.append(
                        np.full_like(group["Timeseries"].values[0], np.nan)
                    )

            # Stack the timeseries to form an array of shape (n_timesteps, n_features)
            combined_series = np.stack(sample_data, axis=-1)

            # Append to the reshaped data list
            reshaped_data.append(combined_series)

        # convert reshaped_data to numpy array
        reshaped_data = np.stack(reshaped_data, axis=0)

        return reshaped_data, self.group_and_combine(ts_df)

    def cluster_timeseries(
        self,
        df: pd.DataFrame,
        method: str = "kmeans",
        n_clusters: int = 5,
        eps: float = 0.5,
        min_samples: int = 5,
        n_jobs=-1,
        metric="dtw",
        feature_combinations=None,
    ):
        """
        Perform clustering on the time series data.

        :param df: DataFrame returned by get_timeseries_image_pandas
        :param method: Clustering method ('kmeans', 'hierarchical', or 'dbscan')
        :param n_clusters: Number of clusters for KMeans and Hierarchical clustering
        :param eps: The maximum distance between two samples for DBSCAN
        :param min_samples: The number of samples in a neighborhood for a point to be considered as a core point in DBSCAN
        :return: DataFrame with an additional 'Cluster' column
        """

        X, df_clustered = self.reshape_dataframe(
            df, feature_combinations=feature_combinations
        )

        if method == "kmeans":
            # Use TimeSeriesKMeans from tslearn for DTW-based clustering
            clusterer = TimeSeriesKMeans(
                n_clusters=n_clusters, metric=metric, random_state=42, n_jobs=n_jobs
            )
            df_clustered["Cluster"] = clusterer.fit_predict(X)
            score = silhouette_score(X, df_clustered["Cluster"])

        elif method == "hierarchical":
            # Compute the distance matrix using DTW
            distance_matrix = pdist(X, metric=lambda u, v: dtw(u, v))
            linkage_matrix = linkage(distance_matrix, method="ward")
            df_clustered["Cluster"] = pd.cut(
                dendrogram(linkage_matrix, no_plot=True)["leaves"],
                bins=n_clusters,
                labels=range(n_clusters),
            )
            score = None
        elif method == "dbscan":
            # Standardize the data
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # Apply DBSCAN
            clusterer = DBSCAN(eps=eps, min_samples=min_samples)
            df_clustered["Cluster"] = clusterer.fit_predict(X_scaled)
            score = None

        else:
            raise ValueError(
                "Invalid clustering method. Choose 'kmeans', 'hierarchical', or 'dbscan'."
            )

        return df_clustered, score

    def plot_cluster_results(self, df: pd.DataFrame):
        """
        Plot the clustering results.

        :param df: DataFrame with clustering results
        :param band: Band to plot ('VV' or 'VH')
        """

        n_clusters = len(df["Cluster"].unique())

        fig, axes = plt.subplots(
            n_clusters, 1, figsize=(15, 5 * n_clusters), sharex=True
        )
        if n_clusters == 1:
            axes = [axes]

        for cluster, ax in enumerate(axes):
            cluster_data = df[df["Cluster"] == cluster]
            for _, row in cluster_data.iterrows():
                ax.plot(row["Dates"], row["Timeseries"], alpha=0.5)
            ax.set_title(f"Cluster {cluster}")
            ax.set_ylabel("Backscatter")

        plt.xlabel("Date")
        plt.tight_layout()
        plt.show()

    def plot_cluster_results_spatial(
        self, df: pd.DataFrame, figsize: Tuple[int, int] = (10, 10)
    ):
        """
        Plot the clustering results spatially.

        :param df: DataFrame with clustering results (pre-filtered for desired band)
        :param figsize: Size of the figure (width, height)
        """
        # Create a colormap
        n_clusters = len(df["Cluster"].unique())
        cmap = plt.cm.get_cmap("tab10")  # You can change 'tab10' to any other colormap

        plt.figure(figsize=figsize)

        # Plot each point
        scatter = plt.scatter(
            df["Longitude"], df["Latitude"], c=df["Cluster"], cmap=cmap, s=50, alpha=0.7
        )

        # Add colorbar
        plt.colorbar(scatter, label="Cluster")

        plt.title(f"Spatial Distribution of Clusters")
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")

        # Add cluster centroids
        centroids = df.groupby("Cluster")[["Longitude", "Latitude"]].mean()
        plt.scatter(
            centroids["Longitude"],
            centroids["Latitude"],
            c="red",
            s=200,
            alpha=1,
            marker="*",
            label="Cluster Centroid",
        )

        for idx, centroid in centroids.iterrows():
            plt.annotate(
                f"Cluster {idx}",
                (centroid["Longitude"], centroid["Latitude"]),
                xytext=(5, 5),
                textcoords="offset points",
            )

        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
