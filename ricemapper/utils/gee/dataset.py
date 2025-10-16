# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import sys

sys.path.append("../utils")
import pandas as pd
import geopandas as gpd
from concurrent.futures import ProcessPoolExecutor
from ricemapper.utils.utils import find_gdf_ROI, gdf_to_single_ee_geometry
from ricemapper.utils.s1 import Sentinel1RowCollections
from ricemapper.utils.era5 import Era5Collections
from ricemapper.utils.s2 import Sentinel2Collections
from datetime import datetime
from dateutil.relativedelta import relativedelta
from typing import Optional, Dict, List
import os
import time
import shapely.wkt
from shapely.geometry import Polygon, MultiPolygon


class DatasetGenerator:

    def __init__(
        self,
        ##----- Input Params -----##
        source_gdf,
        aoi_use_bounds: bool = True,
        start_date: str = "2024-06-01",
        end_date: str = "2024-08-31",
        years: int = None,
        ##----- S1 Parameters -----##
        s1_focal_median_radius_px: float = None,
        save_dir: str = None,
        # processing params
        orbits=["ASCENDING"],
        border_noise_correction=True,
        terrain_correction=True,
        speckle_filter_framework="MULTI",
        speckle_kernel_size=7,
        speckle_filter_type="REFINED LEE",
        speckle_filter_num_images=10,
        # ------- ERA5 Parameters -----##
        era5_daily=True,
    ) -> None:
        """
        Initialize the DatasetGenerator class
        Input parameters
        source_gdf: GeoDataFrame that has attributes that you want to preserve, and a 'Latitude' and 'Longitude' column
        start_date: Start date for the dataset - format 'YYYY-MM-DD'
        end_date: End date for the dataset - format 'YYYY-MM-DD'
        years: Number of years of historical data to use, if None, no historical data is generated
        s1_focal_median_radius_px: Radius for focal median filter in pixels
        """

        self.aoi = (
            find_gdf_ROI(source_gdf)
            if aoi_use_bounds
            else gdf_to_single_ee_geometry(source_gdf)
        )

        if years is not None:
            # convert start_date to datetime
            start_date = datetime.strptime(start_date, "%Y-%m-%d")
            # Adjust start date by the number of years
            start_date = start_date - relativedelta(years=years)
            # convert back to string
            start_date = start_date.strftime("%Y-%m-%d")
            print(
                f"Setting up dataset to include {years} years of historical data from {start_date} to {end_date}"
            )

        self.start_date = start_date
        self.end_date = end_date
        self.source_gdf = source_gdf
        self.years = years
        self.orbits = orbits

        self.s1Collection = Sentinel1RowCollections(
            aoi=self.aoi,
            start_date=start_date,
            end_date=end_date,
            focal_median_radius_px=s1_focal_median_radius_px,
            border_noise_correction=border_noise_correction,
            terrain_correction=terrain_correction,
            speckle_filter_framework=speckle_filter_framework,
            speckle_kernel_size=speckle_kernel_size,
            speckle_filter_type=speckle_filter_type,
            speckle_filter_num_images=speckle_filter_num_images,
        )

        self.era5Collection = Era5Collections(
            aoi=self.aoi, start_date=start_date, end_date=end_date
        )
        self.s2Collection = Sentinel2Collections(
            aoi=self.aoi, start_date=start_date, end_date=end_date
        )
        self.save_dir = save_dir
        self.columns = [
            "VV_ASCENDING_mean",
            "VH_ASCENDING_mean",
            "VV_ASCENDING_std",
            "VH_ASCENDING_std",
            "VV_DESCENDING_mean",
            "VH_DESCENDING_mean",
            "VV_DESCENDING_mean",
            "VH_ASCENDING_std",
            "dates_ASCENDING",
            "dates_DESCENDING",
        ]

        self.df = pd.DataFrame(columns=self.columns)

    def _process_row_s1(
        self, row, square_px, orbit, keep_cols="all", date_range: Optional[tuple] = None
    ):
        """
        Process a single row of the dataset.
        Takes in generic column names as inputs

        """

        if square_px is not None:
            # Fetch the timeseries data for polygons
            df = self.s1Collection.get_timeseries_image_pandas(
                date_range=date_range,
                point=(row["Longitude"], row["Latitude"]),
                size_px=row["square_px"],
                orbit=orbit,
            )
        else:
            df = self.s1Collection.get_timeseries_polygon_pandas(
                date_range=date_range, geometry=row["geometry"], orbit=orbit
            )

        # Save all other column values
        # FIX ME: put it back
        if keep_cols == "all":
            for col_name in row.index.tolist():
                if col_name not in ["Latitude", "Longitude"]:
                    df[col_name] = row[col_name]

        else:
            for col_name in row.index:
                if col_name not in ["Latitude", "Longitude"] and col_name in keep_cols:
                    df[col_name] = row[col_name]

        if self.years is not None and self.years > 0:
            df_historical = self.s1Collection.get_longterm_avg(
                date_range=date_range,
                point=(row["Longitude"], row["Latitude"]),
                size_px=square_px,
                orbit=orbit,
                seasons=self.years,
            )

            # merge df and df_historical on [Latitude, Longitude]
            merged_df = df.merge(
                df_historical,
                on=["latitude", "longitude"],
                how="left",
                suffixes=("", "_drop"),
            )
            # drop columns with _drop suffix
            merged_df = merged_df.loc[:, ~merged_df.columns.str.endswith("_drop")]
            # Update df with the merged DataFrame
            df = merged_df

        return df

    def _process_row_era5(self, row, index_col=None, keep_cols=None):
        """
        Process a single row of the dataset.
        Takes in generic column names as inputs

        """

        # set the point to be the centroid of the geometry of the row
        point = row["geometry"].centroid

        # Fetch the timeseries data
        df = self.era5Collection.get_timeseries_point(point=(point.x, point.y))
        if index_col is not None:
            df[index_col] = row[index_col]
            df.set_index(index_col, inplace=True)
        else:
            index = None

        if keep_cols == "all":
            for col_name in row.index.tolist():
                if col_name not in ["latitude", "longitude"]:
                    df[col_name] = row[col_name]

        elif keep_cols is not None:  # mainly for copying over the index
            for col_name in row.index:
                if col_name not in ["latitude", "longitude"] and col_name in keep_cols:
                    df[col_name] = row[col_name]

        return df

    def _process_row_s2(self, row, keep_cols=None):
        """
        Process a single row of the dataset.
        Takes in generic column names as inputs

        """
        df = self.s2Collection.get_timeseries_polygon_pandas(geometry=row["geometry"])

        if keep_cols == "all":
            for col_name in row.index.tolist():
                if col_name not in ["Latitude", "Longitude"]:
                    df[col_name] = row[col_name]

        else:
            for col_name in row.index:
                if col_name not in ["Latitude", "Longitude"] and col_name in keep_cols:
                    df[col_name] = row[col_name]

        return df

    def process_s1_row(self, args):
        _, row, square_px, orbit, keep_cols, date_range = args
        return self._process_row_s1(row, square_px, orbit, keep_cols, date_range)

    def process_era5_row(self, args):
        _, row, keep_cols = args
        return self._process_row_era5(row, keep_cols=keep_cols)

    def process_s2_row(self, args):
        _, row, keep_cols = args
        return self._process_row_s2(row, keep_cols)

    def _process_batch(
        self,
        batch_data: List[Dict],
        modalities: List[str],
        square_px: Optional[int],
        keep_cols: str,
        date_range: Optional[tuple],
        index_col: str,
        orbit: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Process a single batch of data for all requested modalities
        """
        batch_results = {}

        print(
            f'Processing batch: {batch_data[0]["idx"]} to {batch_data[-1]["idx"]} for modalities: {modalities}'
        )

        # Process each modality for the batch
        for modality in modalities:
            if modality == "s1" and orbit:
                s1_time_start = time.time()
                print(
                    f"Processing S1 data for orbit: {orbit} with {len(batch_data)} workers"
                )
                futures = []
                with ProcessPoolExecutor(max_workers=len(batch_data)) as executor:
                    for item in batch_data:
                        futures.append(
                            executor.submit(
                                self.process_s1_row,
                                (
                                    item["idx"],
                                    item["row"],
                                    square_px,
                                    orbit,
                                    keep_cols,
                                    date_range,
                                ),
                            )
                        )
                results = [future.result() for future in futures]
                # Add idx to each result DataFrame
                for idx, result_df in zip(
                    [item["idx"] for item in batch_data], results
                ):
                    result_df["_idx"] = idx
                batch_results["s1_" + orbit] = pd.concat(results).reset_index(drop=True)
                print(f"S1 processing time: {time.time() - s1_time_start:.2f} seconds")

            elif modality == "era5":
                era5_time_start = time.time()
                print(f"Processing ERA5 data with {len(batch_data)} workers")
                futures = []
                with ProcessPoolExecutor(max_workers=len(batch_data)) as executor:
                    for item in batch_data:
                        futures.append(
                            executor.submit(
                                self.process_era5_row,
                                (item["idx"], item["row"], keep_cols),
                            )
                        )
                results = [future.result() for future in futures]
                # Add idx to each result DataFrame
                for idx, result_df in zip(
                    [item["idx"] for item in batch_data], results
                ):
                    result_df["_idx"] = idx
                batch_results["era5"] = pd.concat(results).reset_index(drop=True)
                print(
                    f"ERA5 processing time: {time.time() - era5_time_start:.2f} seconds"
                )

            elif modality == "s2":
                s2_time_start = time.time()
                print(f"Processing S2 data with {len(batch_data)} workers")
                futures = []
                with ProcessPoolExecutor(max_workers=len(batch_data)) as executor:
                    for item in batch_data:
                        futures.append(
                            executor.submit(
                                self.process_s2_row,
                                (item["idx"], item["row"], keep_cols),
                            )
                        )
                results = [future.result() for future in futures]
                # Add idx to each result DataFrame
                for idx, result_df in zip(
                    [item["idx"] for item in batch_data], results
                ):
                    result_df["_idx"] = idx
                batch_results["s2"] = pd.concat(results).reset_index(drop=True)
                print(f"S2 processing time: {time.time() - s2_time_start:.2f} seconds")

        # Merge results from all modalities using idx
        merged_df = None
        for modality, df in batch_results.items():
            if merged_df is None:
                merged_df = df
            else:
                # Use _idx for merging
                merged_df = merged_df.merge(
                    df, on="_idx", how="outer", suffixes=("", "_drop")
                )
                merged_df = merged_df.loc[:, ~merged_df.columns.str.endswith("_drop")]

        # Add processed_row column and drop temporary _idx
        merged_df["processed_row"] = batch_data[-1]["idx"]

        # Add index_col if it doesn't exist in the merged results
        if index_col not in merged_df.columns and merged_df is not None:
            idx_values = {
                item["idx"]: item["row"][index_col]
                for item in batch_data
                if index_col in item["row"]
            }
            if idx_values:
                merged_df[index_col] = merged_df["_idx"].map(idx_values)

        # Drop the temporary _idx column
        if merged_df is not None and "_idx" in merged_df.columns:
            merged_df = merged_df.drop("_idx", axis=1)

        return merged_df

    def _geometry_to_wkt(self, df):
        """Convert geometry column to WKT string format"""
        if "geometry" in df.columns:
            df = df.copy()
            df["geometry"] = df["geometry"].apply(
                lambda x: x.wkt if x is not None and not isinstance(x, str) else x
            )
        return df

    def _wkt_to_geometry(self, df):
        """Convert WKT string format back to shapely geometry"""
        if "geometry" in df.columns:
            df = df.copy()
            df["geometry"] = df["geometry"].apply(
                lambda x: (
                    x
                    if isinstance(x, (Polygon, MultiPolygon))
                    else (shapely.wkt.loads(x) if isinstance(x, str) else None)
                )
            )
        return df

    def _process_with_retry(
        self,
        data: List[Dict],
        modalities: List[str],
        output_file: str,
        batch_size: int,
        square_px: Optional[int],
        keep_cols: str,
        date_range: Optional[tuple],
        index_col: str,
        max_retries: int = 3,
        retry_delay: int = 60,
    ) -> pd.DataFrame:
        """
        Process data in batches with retry mechanism, appending results to a parquet file
        """

        # Check if output file exists and get last processed row
        if os.path.exists(output_file):
            existing_df = pd.read_parquet(output_file)
            existing_df = self._wkt_to_geometry(
                existing_df
            )  # Convert WKT back to geometry
            last_processed = existing_df["processed_row"].max()
            print(f"Resuming processing from row: {last_processed}")
        else:
            last_processed = -1
            existing_df = None

        retries = 0

        while retries < max_retries:
            try:
                for batch_start in range(last_processed + 1, len(data), batch_size):
                    batch_end = min(batch_start + batch_size, len(data))
                    batch_data = data[batch_start:batch_end]

                    print(f"Processing batch {batch_start//batch_size + 1}")
                    print(
                        f"Processing rows {batch_start} to {batch_end}..., last processed row: {last_processed}"
                    )

                    # Process the batch and get results
                    batch_df = self._process_batch(
                        batch_data,
                        modalities,
                        square_px,
                        keep_cols,
                        date_range,
                        index_col,
                        orbit=self.orbits[0],
                    )  # hack for now, it will only do the first orbit specified

                    # Convert geometry to WKT before saving
                    batch_df_to_save = self._geometry_to_wkt(batch_df)

                    # Append batch results to parquet file
                    if existing_df is None:
                        batch_df_to_save.to_parquet(output_file, index=False)
                        existing_df = batch_df
                    else:
                        existing_df = pd.concat(
                            [existing_df, batch_df], ignore_index=True
                        )
                        # Convert to WKT before saving
                        self._geometry_to_wkt(existing_df).to_parquet(
                            output_file, index=False
                        )

                    print(f"Processed and saved rows up to {batch_end}")

                # Verify row count
                final_df = pd.read_parquet(output_file)
                final_df = self._wkt_to_geometry(
                    final_df
                )  # Convert WKT back to geometry
                expected_rows = len(data)
                actual_rows = len(final_df)

                if actual_rows == expected_rows:
                    print(
                        f"Processing completed successfully! All {expected_rows} rows processed."
                    )
                else:
                    print(
                        f"Warning: Processed {actual_rows} rows but expected {expected_rows} rows."
                    )

                return final_df

            except Exception as e:
                retries += 1
                print(f"\nAn error occurred during processing: {str(e)}\n")
                print(f"Retry attempt {retries} of {max_retries}")
                if retries < max_retries:
                    print(f"Waiting for {retry_delay} seconds before retrying...")
                    time.sleep(retry_delay)
                else:
                    print("Max retries reached. Progress saved in output file.")
                    raise

    def _contains_polygons(self, gdf: gpd.GeoDataFrame) -> bool:
        if "geometry" not in gdf.columns:
            return False
        return gdf.geom_type.isin(["Polygon", "MultiPolygon"]).any()

    def generate_timeseries(
        self,
        modalities=["s1", "era5"],
        keep_cols="all",
        index_col="SN",
        date_range: Optional[tuple] = None,
        square_px: Optional[int] = None,
        batch_size: int = 25,
        max_retries: int = 5,
        retry_delay: int = 10,
    ) -> pd.DataFrame:
        """
        Generate a dataset from the given dataframe for all requested modalities
        """
        if square_px is None and not self._contains_polygons(self.source_gdf):
            raise ValueError(
                "No square pixel size OR polygons found in the dataframe. ABORT"
            )

        # Prepare data for processing
        data = [{"idx": idx, "row": row} for idx, row in self.source_gdf.iterrows()]

        today = datetime.today().strftime("%Y-%m-%d")

        mods_string = f'{"-".join(modalities)}'
        fname = f"ts_{mods_string}_{today}.parquet"

        # Set output file path
        output_file = os.path.join(self.save_dir, fname) if self.save_dir else fname

        # Process all modalities together in batches
        self.df = self._process_with_retry(
            data=data,
            modalities=modalities,
            output_file=output_file,
            batch_size=batch_size,
            square_px=square_px,
            keep_cols=keep_cols,
            date_range=date_range,
            index_col=index_col,
            max_retries=max_retries,
            retry_delay=retry_delay,
        )
        if "geometry" in self.df.columns:
            self.df = self._wkt_to_geometry(self.df)
        return self.df
