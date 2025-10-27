# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import ee
import pandas as pd
from typing import Optional
from ricemapper.utils.utils import convert_tuple2dates, stringify_date


class Sentinel2Collections:

    def __init__(
        self,
        start_date,
        end_date,
        aoi,
        cloud_filter_perc=80,
        cld_prb_threshold=50,
        bands=[
            "B1",
            "B2",
            "B3",
            "B3",
            "B4",
            "B5",
            "B6",
            "B7",
            "B8",
            "B8A",
            "B11",
            "B12",
        ],
    ):
        """
        Args:
        start_date (str): Start date of the data collection
        end_date (str): End date of the data collection
        aoi (ee.Geometry): Area of interest
        cloud_filter_perc (int): Cloud filter percentage - filter out images with cloud percentage greater than this value for the entire collection
        cld_prb_threshold (int): Cloud probability threshold - filter out clouds with probability greater than this value - at a polygon level
        bands (list): List of bands to be included in the collection

        """

        self.start_date = start_date
        self.end_date = end_date
        self.aoi = aoi
        self.bands = bands
        self.cld_prb_threshold = cld_prb_threshold

        self.collection = self.get_s2_sr_cld_col(
            self.aoi, self.start_date, self.end_date, cloud_filter_perc
        )
        # map the cloud shadow mask function over the collection to get the mask for each image
        self.collection = self.collection.filterDate(start_date, end_date)

    def get_s2_sr_cld_col(self, aoi, start_date, end_date, cloud_filter_perc):
        # Import and filter S2 SR.
        s2_sr_col = (
            ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
            .filterBounds(aoi)
            .filterDate(start_date, end_date)
            .filter(ee.Filter.lte("CLOUDY_PIXEL_PERCENTAGE", cloud_filter_perc))
        )

        # Import and filter s2cloudless.
        s2_cloudless_col = (
            ee.ImageCollection("COPERNICUS/S2_CLOUD_PROBABILITY")
            .filterBounds(aoi)
            .filterDate(start_date, end_date)
        )

        # Join the filtered s2cloudless collection to the SR collection by the 'system:index' property.
        return ee.ImageCollection(
            ee.Join.saveFirst("s2cloudless").apply(
                **{
                    "primary": s2_sr_col,
                    "secondary": s2_cloudless_col,
                    "condition": ee.Filter.equals(
                        **{"leftField": "system:index", "rightField": "system:index"}
                    ),
                }
            )
        )

    def add_cloud_bands(self, img):
        # Get s2cloudless image, subset the probability band.
        cld_prb = ee.Image(img.get("s2cloudless")).select("probability")

        # Condition s2cloudless by the probability threshold value.
        is_cloud = cld_prb.gt(self.cld_prb_threshold).rename("clouds")

        # Add the cloud probability layer and cloud mask as image bands.
        return img.addBands(ee.Image([cld_prb, is_cloud]))

    def add_shadow_bands(self, img):
        # Identify water pixels from the SCL band.
        not_water = img.select("SCL").neq(6)

        # Identify dark NIR pixels that are not water (potential cloud shadow pixels).
        SR_BAND_SCALE = 1e4
        NIR_DRK_THRESH = 0.15
        CLD_PRJ_DIST = 1
        dark_pixels = (
            img.select("B8")
            .lt(NIR_DRK_THRESH * SR_BAND_SCALE)
            .multiply(not_water)
            .rename("dark_pixels")
        )

        # Determine the direction to project cloud shadow from clouds (assumes UTM projection).
        shadow_azimuth = ee.Number(90).subtract(
            ee.Number(img.get("MEAN_SOLAR_AZIMUTH_ANGLE"))
        )

        # Project shadows from clouds for the distance specified by the CLD_PRJ_DIST input.
        cld_proj = (
            img.select("clouds")
            .directionalDistanceTransform(shadow_azimuth, CLD_PRJ_DIST * 10)
            .reproject(**{"crs": img.select(0).projection(), "scale": 100})
            .select("distance")
            .mask()
            .rename("cloud_transform")
        )

        # Identify the intersection of dark pixels with cloud shadow projection.
        shadows = cld_proj.multiply(dark_pixels).rename("shadows")

        # Add dark pixels, cloud projection, and identified shadows as image bands.
        return img.addBands(ee.Image([dark_pixels, cld_proj, shadows]))

    def add_cld_shdw_mask(self, img):
        BUFFER = 50
        # Add cloud component bands.
        img_cloud = self.add_cloud_bands(img)

        # Add cloud shadow component bands.
        img_cloud_shadow = self.add_shadow_bands(img_cloud)

        # Combine cloud and shadow mask, set cloud and shadow as value 1, else 0.
        is_cld_shdw = (
            img_cloud_shadow.select("clouds")
            .add(img_cloud_shadow.select("shadows"))
            .gt(0)
        )

        # Remove small cloud-shadow patches and dilate remaining pixels by BUFFER input.
        # 20 m scale is for speed, and assumes clouds don't require 10 m precision.
        is_cld_shdw = (
            is_cld_shdw.focalMin(2)
            .focalMax(BUFFER * 2 / 20)
            .reproject(**{"crs": img.select([0]).projection(), "scale": 20})
            .rename("cloudmask")
        )

        # Add the final cloud-shadow mask to the image.
        # return img_cloud_shadow.addBands(is_cld_shdw)
        # if you prefer to include only the final cloud/cloud shadow mask along with the original image bands:
        return img.addBands(is_cld_shdw)

    def get_aggregated_timeseries(
        self,
        ee_geometry: ee.Geometry,
        collection: ee.ImageCollection,
        scale: int = 10,
        max_pixels: int = 1e8,
        chunk_size: int = 100,
    ):
        """
        Memory-efficient function to compute mean and standard deviation of Sentinel-2 bands
        within a polygon for each time step, using a chunking strategy.

        Args:
            ee_geometry (ee.Geometry): Earth Engine geometry defining the area of interest
            collection (ee.ImageCollection): The Sentinel-2 image collection
            scale (int): The scale in meters of the projection to work in. Default is 10 (Sentinel-2 resolution)
            max_pixels (int): The maximum number of pixels to reduce in reduceRegion. Default is 1e8
            chunk_size (int): The number of images to process in each chunk. Default is 100

        Returns:
            pd.DataFrame: A single-row DataFrame with columns for mean and standard deviation of each band as a timeseries
        """
        # Filter the collection to include only images that intersect with the geometry
        collection = collection.filterBounds(ee_geometry)

        def calculate_stats(image):
            image = self.add_cld_shdw_mask(image)  # add cloudmask to image
            unique_bands = list(set(self.bands))
            print(f"Sampling S2 bands: {unique_bands}")
            reduced = image.select(unique_bands).reduceRegion(
                reducer=ee.Reducer.mean()
                .combine(ee.Reducer.stdDev(), None, True)
                .combine(ee.Reducer.median(), None, True),
                geometry=ee_geometry,
                scale=scale,
                maxPixels=max_pixels,
                bestEffort=True,
            )

            # Calculate some common vegetation indices
            ndvi = image.normalizedDifference(
                ["B8", "B4"]
            )  # NIR and Red -> returned image band name is always 'nd'
            lswi = image.normalizedDifference(
                ["B8A", "B11"]
            )  # Narrow NIR and SWIR1 -> returned image band name is always 'nd'

            # Calculate indices statistics
            indices_stats = ndvi.addBands(lswi).reduceRegion(
                reducer=ee.Reducer.mean()
                .combine(ee.Reducer.stdDev(), None, True)
                .combine(ee.Reducer.median(), None, True),
                geometry=ee_geometry,
                scale=scale,
                maxPixels=max_pixels,
                bestEffort=True,
            )

            # Combine all properties
            properties = {"date": image.date().format("YYYY-MM-dd")}

            # Add band statistics
            for band in self.bands:
                properties.update(
                    {
                        f"{band}_mean": reduced.get(f"{band}_mean"),
                        f"{band}_stdDev": reduced.get(f"{band}_stdDev"),
                        f"{band}_median": reduced.get(f"{band}_median"),
                    }
                )

            # Add indices statistics
            properties.update(
                {
                    "NDVI_mean": indices_stats.get("nd_mean"),
                    "NDVI_stdDev": indices_stats.get("nd_stdDev"),
                    "NDVI_median": indices_stats.get("nd_median"),
                    "LSWI_mean": indices_stats.get("nd_1_mean"),
                    "LSWI_stdDev": indices_stats.get("nd_1_stdDev"),
                    "LSWI_median": indices_stats.get("nd_1_median"),
                }
            )

            cloudmask_stats = image.select("cloudmask").reduceRegion(
                reducer=ee.Reducer.max(),
                geometry=ee_geometry,
                scale=scale,
                maxPixels=max_pixels,
                bestEffort=True,
            )

            properties["cloudmask"] = cloudmask_stats.get("cloudmask")

            return ee.Feature(None, properties)

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
            def extract_timeseries(stats_list, metric, stat):
                return [
                    s["properties"][f"{metric}_{stat}"]
                    for s in stats_list
                    if f"{metric}_{stat}" in s["properties"]
                ], [
                    s["properties"]["date"]
                    for s in stats_list
                    if "date" in s["properties"]
                ]

            # Create dictionary to store all time series
            timeseries_dict = {
                "dates_s2": [
                    s["properties"]["date"]
                    for s in all_features
                    if "date" in s["properties"]
                ]
            }

            # Extract time series for each band and statistic
            for band in self.bands:
                for stat in ["mean", "stdDev", "median"]:
                    values, _ = extract_timeseries(all_features, band, stat)
                    timeseries_dict[f"{band}_{stat}"] = values

            # Extract time series for indices
            for index in ["NDVI", "LSWI"]:
                for stat in ["mean", "stdDev", "median"]:
                    values, _ = extract_timeseries(all_features, index, stat)
                    timeseries_dict[f"{index}_{stat}"] = values

            cloudmask_values = [
                s["properties"]["cloudmask"]
                for s in all_features
                if "cloudmask" in s["properties"]
            ]
            print(f"Cloudmask debug values: {cloudmask_values}")
            timeseries_dict["cloudmask"] = cloudmask_values

            # Create the DataFrame with all time series
            df = pd.DataFrame({key: [value] for key, value in timeseries_dict.items()})

            return df

        except ee.EEException as e:
            print(f"An Earth Engine error occurred: {str(e)}")
            return pd.DataFrame()

    def get_timeseries_polygon_pandas(
        self, geometry, date_range: Optional[tuple] = None
    ):
        """
        Get time series data for a polygon from Sentinel-2 imagery.

        Args:
            date_range (Optional[tuple]): Tuple of (start_date, end_date) in any format that can be converted to dates
            geometry (Polygon): Shapely Polygon object defining the area of interest

        Returns:
            pd.DataFrame: DataFrame containing time series data for all bands and indices
        """
        ee_geometry = ee.Geometry.Point(geometry.centroid.x, geometry.centroid.y)

        print(f"Getting Sentinel-2 timeseries for Polygon: {ee_geometry.getInfo()}")

        if date_range is not None:
            start_date, end_date = convert_tuple2dates(date_range)
            start_date_str, end_date_str = stringify_date([start_date, end_date])
            collection = self.collection.filterDate(start_date_str, end_date_str)
        else:
            collection = self.collection

        return self.get_aggregated_timeseries(ee_geometry, collection)
