# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import geopandas as gpd
import os
from dotenv import load_dotenv

load_dotenv()
DATA_DIR = os.getenv("DATA_DIR")

"""
This script splits the FTW polygons into district-specific polygons.

Input:
- districts_path: GeoParquet file with district boundaries
- polygons_path: GeoParquet file with polygons
- output_folder: Output folder to save district-specific GeoParquet files


"""

# Input paths
districts_path = (
    f"{DATA_DIR}/assets/punjab.geojson"  # GeoParquet file with district boundaries
)
polygons_path = f"{DATA_DIR}/ftw/ftw_punjab.parquet"  # GeoParquet file with polygons
output_folder = f"{DATA_DIR}/ftw/polygons"  # Output folder to save district-specific GeoParquet files

# Ensure output directory exists
os.makedirs(output_folder, exist_ok=True)

# Load district boundaries and polygons
districts = gpd.read_file(districts_path)
polygons = gpd.read_parquet(polygons_path)

# Reproject polygons to match districts' CRS if needed
if polygons.crs != districts.crs:
    polygons = polygons.to_crs(districts.crs)

# Loop through each district and extract polygons within it
for idx, district in districts.iterrows():
    # Extract district name, replace whitespaces with '_'
    district_name = (
        district["Dist_Name"].replace(" ", "_")
        if "Dist_Name" in district
        else f"district_{idx}"
    )

    # Create a GeoDataFrame for the single district
    district_gdf = gpd.GeoDataFrame([district], crs=districts.crs)

    # Spatial join: Extract polygons that intersect with the current district
    polygons_in_district = gpd.overlay(polygons, district_gdf, how="intersection")

    # Skip empty results
    if polygons_in_district.empty:
        print(f"No polygons found in district: {district_name}")
        continue

    # Save polygons to a new GeoParquet file
    output_path = os.path.join(output_folder, f"{district_name}.parquet")
    polygons_in_district.to_parquet(output_path)

    print(f"Saved polygons for district '{district_name}' to: {output_path}")

print("Processing complete.")
