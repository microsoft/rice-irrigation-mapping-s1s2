# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import geopandas as gpd
import rasterio
from exactextract import exact_extract
from glob import glob
import tqdm
import os
from dotenv import load_dotenv


def filter_polygons_inside_mask_fast(gdf, mask_path):
    """
    Fastest approach using exactextract for bulk polygon filtering.
    Keeps only polygons entirely within mask=1 regions.
    """
    with rasterio.open(mask_path) as src:
        mask_crs = src.crs

    # Reproject if needed
    if gdf.crs != mask_crs:
        gdf = gdf.to_crs(mask_crs)

    # Use exactextract to get stats for each polygon
    stats = exact_extract(
        mask_path,
        gdf,
        ops=["min", "count"],  # min=0 means polygon touches 0 values
        output="pandas",
    )

    # Keep only polygons where minimum value is 1 (entirely in positive region)
    # and count > 0 (polygon actually intersects raster)
    mask = (stats["min"] == 1) & (stats["count"] > 0)

    return gdf[mask].reset_index(drop=True)


def process_folders(input_path, output_path, mask_path):
    """
    Process all parquet files in a folder and filter them based on a mask.
    """

    parquet_files = glob(os.path.join(input_path, "*.parquet"))
    # skip any files that already exist in the output path
    parquet_files = [
        file
        for file in parquet_files
        if not os.path.exists(os.path.join(output_path, os.path.basename(file)))
    ]
    print(f"Found {len(parquet_files)} parquet files to process")

    for file in tqdm.tqdm(parquet_files, desc="Processing parquet files"):
        print(f"Processing {file}")
        gdf = gpd.read_parquet(file)
        len_before = len(gdf)
        filtered_gdf = filter_polygons_inside_mask_fast(gdf, mask_path)
        len_after = len(filtered_gdf)
        output_file = os.path.join(output_path, os.path.basename(file))
        filtered_gdf.to_parquet(output_file)
        print(
            f"Filtered {len_before-len_after} polygons from {file} and saved to {output_file} ({len_after} polygons remaining)"
        )


if __name__ == "__main__":
    """
    Pass in input path containing a folder of geoparquet files, and a mask path.
    The script will filter the polygons in the input path based on the mask (note only where mask==1)
    The filtered polygons will be saved to the output path.
    """
    load_dotenv()
    data_dir = os.getenv("DATA_DIR")
    repo_dir = os.getenv("REPO_DIR", "/data/panopticon/tnc")
    input_path = os.path.join(
        data_dir, "inference/districts/features/06-01_09-05_weeks_gamma0_3_clean_class/"
    )
    output_path = os.path.join(
        data_dir,
        "inference/districts/features/06-01_09-05_weeks_gamma0_3_clean_masked/",
    )
    mask_path = os.path.join(repo_dir, "data/misc/Han2022-paddyRice2021.tif")
    process_folders(input_path, output_path, mask_path)
