# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import ee
import pandas as pd


class Era5Collections:
    def __init__(
        self,
        aoi,
        start_date,
        end_date,
        cadence=["monthly", "daily"],
        vars=["temperature_2m", "total_precipitation_sum"],
    ):
        self.aoi = aoi
        self.start_date = start_date
        self.end_date = end_date
        self.vars = vars

        self.collection = {}

        if "monthly" in cadence:
            collection = (
                ee.ImageCollection("ECMWF/ERA5_LAND/MONTHLY_AGGR")
                .filterDate(self.start_date, self.end_date)
                .filterBounds(self.aoi)
                .select(vars)
            )
            self.collection["monthly"] = collection

        if "daily" in cadence:
            collection = (
                ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR")
                .filterDate(self.start_date, self.end_date)
                .filterBounds(self.aoi)
                .select(vars)
            )
            self.collection["daily"] = collection
        else:
            raise ValueError("Invalid cadence. Choose from: monthly, daily")

        self.scale = 11132  # m

    def get_timeseries_point(self, point, max_pixels: int = 1e3, chunk_size: int = 100):
        """
        Memory-efficient function to extract ERA5 temperature and precipitation data
        for a point location over time, using a chunking strategy.

        Args:
            point (shapely Point): A point geometry defining the location of interest
            collection (ee.ImageCollection): The ERA5 image collection (ECMWF/ERA5_LAND/DAILY_AGGR)
            start_date (str): Start date in 'YYYY-MM-dd' format
            end_date (str): End date in 'YYYY-MM-dd' format
            max_pixels (int): The maximum number of pixels to reduce in reduceRegion. Default is 1e8
            chunk_size (int): The number of images to process in each chunk. Default is 100

        Returns:
            pd.DataFrame: A single-row DataFrame with columns for temperature and precipitation as timeseries
        """
        # Set default variables if not defined
        if not hasattr(self, "vars"):
            self.vars = ["temperature_2m", "total_precipitation_sum"]

        ee_geometry = ee.Geometry.Point(point)
        collection = self.collection["daily"]  # force for now

        # Filter the collection by date range and location
        collection = collection.filterBounds(
            ee_geometry
        )  # not sure this is needed, or might need to buffer to make > point

        # Function to calculate values for each variable
        def calculate_stats(image):
            reduced = image.select(self.vars).reduceRegion(
                reducer=ee.Reducer.first(),  # Use first() since we're dealing with a point
                geometry=ee_geometry,
                scale=self.scale,
                maxPixels=max_pixels,
                bestEffort=True,
            )

            # Create a dictionary with the date and all variables
            properties = {"date": image.date().format("YYYY-MM-dd")}

            # Add each variable to the properties
            for var in self.vars:
                properties[var] = reduced.get(var)

            return ee.Feature(None, properties)

        # Function to process a chunk of the collection
        def process_chunk(chunk):
            return chunk.map(calculate_stats).getInfo()["features"]

        try:
            # Get the total number of images
            total_images = collection.size().getInfo()

            if total_images == 0:
                print("No images found in the collection for the given date range.")
                return pd.DataFrame()

            # Process the collection in chunks
            all_features = []
            for i in range(0, total_images, chunk_size):
                chunk = collection.toList(chunk_size, i)
                chunk_features = process_chunk(ee.ImageCollection(chunk))
                all_features.extend(chunk_features)

            # Function to extract time series data
            def extract_timeseries(stats_list, var):
                return [
                    s["properties"][var] for s in stats_list if var in s["properties"]
                ], [
                    s["properties"]["date"]
                    for s in stats_list
                    if "date" in s["properties"]
                ]

            # Create a dictionary to store all timeseries
            timeseries_dict = {}

            # Extract dates once
            _, dates = extract_timeseries(all_features, self.vars[0])
            timeseries_dict["dates_era5"] = [dates]

            # Extract timeseries for each variable
            for var in self.vars:
                values, _ = extract_timeseries(all_features, var)
                timeseries_dict[var] = [values]

            # Create the DataFrame
            df = pd.DataFrame(timeseries_dict)

            return df

        except ee.EEException as e:
            print(f"An Earth Engine error occurred: {str(e)}")
            return pd.DataFrame()
