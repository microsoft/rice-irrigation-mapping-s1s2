# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
import numpy as np
import geopandas as gpd
from pathlib import Path
import subprocess
import pandas as pd
from tqdm import tqdm
from exactextract import exact_extract
import time
from concurrent.futures import ProcessPoolExecutor
import warnings
import argparse
from dotenv import load_dotenv

# Suppress FutureWarning
warnings.simplefilter(action="ignore", category=FutureWarning)

BAND_VV = "Sigma0_VV"
BAND_VH = "Sigma0_VH"

# For Gamma0
BAND_VV = "Gamma0_VV"
BAND_VH = "Gamma0_VH"


def get_band_order(tif_file):
    """Determine the order of bands in the TIFF file."""
    gdalinfo_command = ["gdalinfo", str(tif_file)]
    try:
        output = subprocess.run(
            gdalinfo_command, capture_output=True, text=True, check=True
        ).stdout
        bands = {}
        current_band = None

        for line in output.splitlines():
            if line.strip().startswith("Band "):
                current_band = int(line.split("Band ")[1].split()[0])
            if "Description =" in line and current_band is not None:
                description = line.split("Description =")[1].strip()
                if description == BAND_VV:
                    bands["VV"] = current_band
                    print(f"Found VV {current_band} in {description}")
                elif description == BAND_VH:
                    bands["VH"] = current_band
                    print(f"Found VH {current_band} in {description}")
                current_band = None

        if "VV" in bands and "VH" in bands:
            print(f'Found VV {bands["VV"]} and VH {bands["VH"]} in {tif_file}')
            return [bands["VV"], bands["VH"]]
        else:
            print(
                f"Warning: VV or VH band description missing in {tif_file}. Assuming band order is [VV, VH]."
            )
            return [1, 2]
    except subprocess.CalledProcessError as e:
        print(f"Error running gdalinfo on {tif_file}: {e}")
        return None


def process_single_tif(clipped_tif, user_geometries, progress=True):
    try:
        start_time = time.time()
        # Compute statistics using exactextract
        out_gdf = exact_extract(
            str(clipped_tif),  # Raster file path
            user_geometries,  # GeoDataFrame with geometries
            ops=["mean"],  # Operation to compute
            include_geom=True,  # Include geometry column
            include_cols=["ID"],  # Include ID column
            output="pandas",  # Return as GeoDataFrame
            progress=progress,  # Disable progress bar for parallel tasks
        )
        elapsed_time = time.time() - start_time
        # print(f"Processed {clipped_tif} in {elapsed_time:.2f} seconds.")

        # Remove rows with NaN values in band_1_mean or band_2_mean
        out_gdf = out_gdf.dropna(subset=["band_1_mean", "band_2_mean"])

        # Drop unnecessary band_3_mean column
        if "band_3_mean" in out_gdf.columns:
            out_gdf = out_gdf.drop(columns=["band_3_mean"])

        # Convert band means to numpy arrays
        out_gdf["band_1_mean"] = out_gdf["band_1_mean"].apply(np.array)
        out_gdf["band_2_mean"] = out_gdf["band_2_mean"].apply(np.array)

        # print(f'Saving intermediate results to {clipped_tif}.pkl')
        out_gdf.to_pickle(f"{clipped_tif}.pkl")

        return out_gdf
    except Exception as e:
        print(f"Error processing {clipped_tif}: {e}")
        return None


def clean_bounds(slice_bounds):
    # Get the outer hull (convex hull) of all geometries and simplify it
    outer_hull = slice_bounds.union_all().convex_hull
    # Simplify the hull with a tolerance of 0.001 degrees (approximately 100m)
    simplified_hull = outer_hull.simplify(0.001, preserve_topology=True)

    # Create a new GeoDataFrame with just the simplified hull
    slice_bounds = gpd.GeoDataFrame(geometry=[simplified_hull], crs="EPSG:4326")
    return slice_bounds


def clean_geometries(input_geoparquet):
    print("Loading user-provided geometries from the GeoParquet file.")
    user_geometries = gpd.read_parquet(input_geoparquet)
    user_geometries = user_geometries.to_crs("EPSG:4326")  # Ensure CRS is consistent
    user_geometries["ID"] = range(len(user_geometries))
    return user_geometries


def process_slices(input_root, input_geoparquet, output_dir, save_geojson=False):
    input_root = Path(input_root)
    combined_gdf = gpd.GeoDataFrame()  # init an empty geodataframe

    user_geometries = clean_geometries(input_geoparquet)

    if "Class" in user_geometries.columns:
        train = True
        print(f"\n\nTraining data found in {input_geoparquet}\n\n")
    else:
        train = False
        print(f"\n\nInference data found in {input_geoparquet}\n\n")

    print(f"Loaded {len(user_geometries)} geometries from {input_geoparquet}.")

    for row in tqdm(input_root.iterdir(), desc="Processing rows"):
        if not row.is_dir():
            continue

        print(f"\n-----------Processing Row: {row.name}---------------------\n")
        row_start_time = time.time()

        row_dirs = [d.name for d in row.iterdir() if d.is_dir()]
        print(f"Processing slices: {row_dirs}\n")
        for slice_dir in tqdm(row.iterdir(), desc=f"[{row.name}]Processing slices"):

            if not slice_dir.is_dir():
                continue
            print(
                f"\n-----------Processing Slice: {slice_dir.name}---------------------\n"
            )

            geometry_file = slice_dir / f"bounds_{row.name}_{slice_dir.name}.geojson"
            print(f"Parsing geometry file: {geometry_file}")
            if not geometry_file.exists():
                print(
                    f"Geometry file {geometry_file} not found. Skipping slice {slice_dir.name}."
                )
                continue

            # Load slice bounds and ensure CRS is consistent
            slice_bounds = gpd.read_file(geometry_file)
            slice_bounds = slice_bounds.to_crs("EPSG:4326")

            slice_bounds = clean_bounds(slice_bounds)

            # clip the slice_bounds to the user_geometries
            slice_plots = user_geometries.copy()
            slice_plots = slice_plots.clip(slice_bounds)
            # check to remove all rows with geometry == 'Polygon' or 'MultiPolygon'
            slice_plots = slice_plots[
                slice_plots.geometry.geom_type.isin(["Polygon", "MultiPolygon"])
            ].reset_index(drop=True)

            if len(slice_plots) == 0:
                print(
                    f"No geometries are within slice bounds. Skipping slice {slice_dir.name}."
                )
                continue
            else:
                print(
                    f"[{row.name}:{slice_dir.name}] Processing {len(slice_plots)} geometries that are within slice bounds out of {len(user_geometries)} total geometries."
                )

            # Clip each TIFF file in the slice directory using the bounds
            for tif_file in tqdm(
                slice_dir.glob("S1A*.tif"), desc=f"Clipping TIFFs in {slice_dir.name}"
            ):
                clipped_tif = slice_dir / f"clipped_{tif_file.name}"
                if clipped_tif.exists():
                    # print(f"Clipped file {clipped_tif} already exists. Skipping clipping.")
                    continue
                band_order = get_band_order(tif_file)
                print(f"Band order: {band_order}")
                if not band_order:
                    continue

                clip_command = [
                    "gdalwarp",
                    "-cutline",
                    str(geometry_file),
                    "-crop_to_cutline",
                    "-dstalpha",
                    "-b",
                    str(band_order[0]),
                    "-b",
                    str(band_order[1]),
                    "-co",
                    "COMPRESS=DEFLATE",
                    "-co",
                    "PREDICTOR=3",
                    "-co",
                    "BIGTIFF=YES",
                    str(tif_file),
                    str(clipped_tif),
                ]

                try:
                    print(f"Clipping {tif_file} to {clipped_tif}.")
                    subprocess.run(clip_command, check=True)
                except subprocess.CalledProcessError as e:
                    print(f"Error clipping {tif_file}: {e}")
                    continue

            # Process clipped TIFFs for the current slice
            clipped_tiffs = list(slice_dir.glob("clipped_*.tif"))
            if not clipped_tiffs:
                print(
                    f"No clipped TIFF files found for slice {slice_dir.name}. Skipping."
                )
                continue

            # Sort TIFFs by date
            sorted_data = sorted(
                [(f, f.name.split("_")[5][:8]) for f in clipped_tiffs],
                key=lambda x: x[1],
            )
            clipped_tiffs, dates_array = zip(*sorted_data)
            dates_array = np.array(dates_array)

            # Initialize aggregation dictionary for the slice
            aggregated_data = {
                id: {"band_1_mean": [], "band_2_mean": []} for id in slice_plots["ID"]
            }

            print(
                f"[{row.name}]Computing zonal statistics for slice {slice_dir.name} in parallel."
            )
            with ProcessPoolExecutor() as executor:
                results = list(
                    executor.map(
                        process_single_tif,
                        clipped_tiffs,
                        [slice_plots] * len(clipped_tiffs),
                    )
                )
                # repeat the user_geometries list for each tif file in the slice - since map function only accepts one iterable

            print(
                f"\n\n[{row.name}:{slice_dir.name}]Finished processing slice; now aggregating timeseries for the slice.\n\n"
            )

            # Aggregate timeseries for the slice
            for out_gdf in tqdm(results, desc=f"Slice:{slice_dir.name} timesteps"):
                if out_gdf is not None and not out_gdf.empty:
                    for _, R in out_gdf.iterrows():
                        id_ = R["ID"]
                        # Ensure the ID exists in aggregated_data
                        if id_ not in aggregated_data:
                            print(
                                f"Warning: ID {id_} not found in aggregated_data. Skipping."
                            )
                            continue

                        # Validate and append band_1_mean
                        if (
                            isinstance(R["band_1_mean"], (np.ndarray, list))
                            and len(R["band_1_mean"]) > 0
                        ):
                            if R["band_1_mean"] < 0:
                                print(
                                    f"Warning1: band_1_mean for ID {id_} is negative. Skipping."
                                )
                                continue
                            aggregated_data[id_]["band_1_mean"].append(R["band_1_mean"])
                            if R["band_1_mean"].all() == 0:
                                print(f"\n!!!XX{id_} has all zeros!!!\n")
                        elif not np.isnan(R["band_1_mean"]):
                            if R["band_1_mean"] < 0:
                                print(
                                    f"Warning1: band_1_mean for ID {id_} is negative. Skipping."
                                )
                                continue
                            aggregated_data[id_]["band_1_mean"].append(
                                np.array([R["band_1_mean"]])
                            )
                            # print(f'VV_mean: {R["band_1_mean"]}')
                            # if R["band_1_mean"] == 0:
                            #     print(f'\n!!!XX{id_} has all zeros!!!\n')

                        # Validate and append band_2_mean
                        if (
                            isinstance(R["band_2_mean"], (np.ndarray, list))
                            and len(R["band_2_mean"]) > 0
                        ):
                            if R["band_2_mean"] < 0:
                                print(
                                    f"Warning2: band_2_mean for ID {id_} is negative. Skipping."
                                )
                                continue
                            aggregated_data[id_]["band_2_mean"].append(R["band_2_mean"])
                        elif not np.isnan(R["band_2_mean"]):
                            if R["band_2_mean"] < 0:
                                print(
                                    f"Warning2: band_2_mean for ID {id_} is negative. Skipping."
                                )
                                continue
                            aggregated_data[id_]["band_2_mean"].append(
                                np.array([R["band_2_mean"]])
                            )
                else:
                    print("Warning: Empty or invalid out_gdf encountered. Skipping.")

            print(
                f"[{row.name}:{slice_dir.name}]Aggregated data len for slice: {len(aggregated_data)}\n\n"
            )

            # Remove IDs with no data
            aggregated_data = {
                id_: data
                for id_, data in aggregated_data.items()
                if data["band_1_mean"] and data["band_2_mean"]
            }

            # Flatten the aggregated timeseries into arrays
            aggregated_data = {
                id_: {
                    "band_1_mean": np.concatenate(data["band_1_mean"]),
                    "band_2_mean": np.concatenate(data["band_2_mean"]),
                }
                for id_, data in aggregated_data.items()
            }

            # Convert aggregated data to GeoDataFrame for the slice
            slice_combined_gdf = pd.DataFrame(
                [
                    {
                        "ID": id_,
                        "geometry": slice_plots.loc[
                            slice_plots["ID"] == id_, "geometry"
                        ].values[0],
                        "band_1_mean": aggregated_data[id_]["band_1_mean"],
                        "band_2_mean": aggregated_data[id_]["band_2_mean"],
                        **(
                            {
                                "Class": slice_plots.loc[
                                    slice_plots["ID"] == id_, "Class"
                                ].values[0]
                            }
                            if train
                            else {}
                        ),
                    }
                    for id_ in aggregated_data
                ]
            )
            # Add dates for the slice
            if not slice_combined_gdf.empty:
                slice_combined_gdf["dates_ASCENDING"] = [dates_array] * len(
                    slice_combined_gdf
                )

            num_zeros = len(
                slice_combined_gdf[
                    slice_combined_gdf["band_1_mean"].apply(
                        lambda x: all(value == 0.0 for value in x)
                    )
                ]
            )
            print(
                f"[{row.name}:{slice_dir.name}] All zeros:All rows = {num_zeros}:{len(slice_combined_gdf)}"
            )
            slice_combined_gdf = gpd.GeoDataFrame(
                slice_combined_gdf, geometry="geometry", crs="EPSG:4326"
            )

            # TEST: remove all rows with all zeros
            if num_zeros > 0:
                slice_combined_gdf = slice_combined_gdf[
                    ~slice_combined_gdf["band_1_mean"].apply(
                        lambda x: all(value == 0.0 for value in x)
                    )
                ]
                print(
                    f"[{row.name}:{slice_dir.name}] Removing all rows with all zeros. Slice {slice_dir.name} has {len(slice_combined_gdf)} geometries after removing all zeros."
                )

            # Merge slice results into the row's combined_gdf
            if combined_gdf.empty:
                combined_gdf = slice_combined_gdf
            else:
                combined_gdf = (
                    combined_gdf.set_index("ID")
                    .combine_first(slice_combined_gdf.set_index("ID"))
                    .reset_index()
                )

            print(
                f"[{row.name}:{slice_dir.name}] Combined GDF length: {len(combined_gdf)}\n\n"
            )

        row_elapsed_time = time.time() - row_start_time
        if not combined_gdf.empty:
            print(
                f"\n***Finished processing row {row.name} in {row_elapsed_time:.2f} s :{row_elapsed_time/len(combined_gdf)} s/polygon***\n"
            )

    # Save to Parquet
    if not combined_gdf.empty:
        if "ID" in combined_gdf.columns:
            combined_gdf = combined_gdf.drop(
                columns=["ID"]
            )  # Drop the ID column if present

        # rename columns:
        combined_gdf = combined_gdf.rename(
            columns={
                "band_1_mean": "VV_ASCENDING_mean",
                "band_2_mean": "VH_ASCENDING_mean",
            }
        )

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        output_parquet = os.path.join(output_dir, f"{dist_name}_s1.parquet")
        print(f"Output file: {output_parquet}")
        print(f"Processing complete. Results saved to {output_parquet}.")

        combined_gdf.to_parquet(output_parquet, index=False)
        if save_geojson:
            output_geojson = os.path.join(output_dir, f"{dist_name}_s1.geojson")
            combined_gdf.to_file(output_geojson, driver="GeoJSON")
            print(f"GeoJSON file saved to {output_geojson}")

    else:
        print("No data to save.")


if __name__ == "__main__":

    # Load .env file
    load_dotenv()
    DATA_DIR = os.getenv("DATA_DIR")

    # Set up argument parser
    parser = argparse.ArgumentParser(description="Process S1 data for districts")
    parser.add_argument(
        "--input_dir",
        type=str,
        default=f"{DATA_DIR}/s1/gamma0",
        help="Directory containing raw S1 data",
    )
    parser.add_argument(
        "--polys_dir",
        type=str,
        default=f"{DATA_DIR}/dataset/plots_clean",
        help="Directory containing polygon files",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=f"{DATA_DIR}/dataset/s1/s1_gamma0",
        help="Directory for output files",
    )
    parser.add_argument(
        "--inference",
        action="store_true",
        help="Use inference mode with different default paths",
    )

    args = parser.parse_args()

    # Override defaults for inference mode
    if args.inference:
        args.output_dir = f"{DATA_DIR}/inference/districts/s1_gamma0_3"
        args.polys_dir = f"{DATA_DIR}/ftw/polygons"

    input_directory = args.input_dir
    polys_dir = Path(args.polys_dir)
    output_directory = args.output_dir

    ignore_districts = [
        "_".join(parq_file.stem.split("_")[:-1])
        for parq_file in Path(output_directory).glob("*.parquet")
    ]
    do_districts = []

    print(f"Ignoring districts: {ignore_districts}")

    for parq_file in sorted(polys_dir.glob("*.parquet"), key=lambda x: x.stem):

        dist_name = parq_file.stem
        if dist_name in ignore_districts:
            print(
                f"\n-----------------Ignoring District: {dist_name}-----------------\n"
            )
            continue
        if len(do_districts) > 0 and dist_name not in do_districts:
            continue

        print(f"-----------------Processing District: {dist_name}-----------------")
        output_dir = Path(output_directory)

        process_slices(input_directory, parq_file, output_dir, save_geojson=True)
        print(
            f"-----------------Finished Processing District: {dist_name}-----------------"
        )
