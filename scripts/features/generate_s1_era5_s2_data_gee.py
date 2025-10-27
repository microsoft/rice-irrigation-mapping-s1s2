# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import geopandas as gpd
import os
from ricemapper.utils.dataset import DatasetGenerator
from dotenv import load_dotenv
import fire
from datetime import datetime
import ee

load_dotenv()
private_key = os.getenv("EE_KEY")
service_account = os.getenv("EE_SERVICE_ACCOUNT")

credentials = ee.ServiceAccountCredentials(service_account, private_key)
ee.Initialize(credentials)


def preprocess(dir_name, geojson_fname, SEED=13, random_subset=None):
    gdf = gpd.read_file(os.path.join(os.environ["DATA_DIR"], dir_name, geojson_fname))

    # find the centroid of each geometry and save into a new column
    gdf["Latitude"] = gdf["geometry"].centroid.x
    gdf["Longitude"] = gdf["geometry"].centroid.y

    if random_subset is not None:
        # subset if needed: subsample 10 rows randomly from df
        gdf = gdf.sample(random_subset, random_state=SEED)

    print(f"Total available plots: {len(gdf)}")

    ptr, dsr, awd = (
        gdf[gdf["Class"] == "PTR"].copy(),
        gdf[gdf["Class"] == "DSR"].copy(),
        gdf[gdf["Class"] == "AWD"].copy(),
    )
    # Convert to cartesian coordinates (web mercator)
    ptr, dsr, awd = (
        ptr.to_crs("EPSG:3857"),
        dsr.to_crs("EPSG:3857"),
        awd.to_crs("EPSG:3857"),
    )
    ptr["area"] = ptr["geometry"].area * 0.0002471054  # acres
    dsr["area"] = dsr["geometry"].area * 0.0002471054
    awd["area"] = awd["geometry"].area * 0.0002471054

    if len(ptr) != 0:
        # calculate the area of geometry in ptr
        print(
            f'PTR plots: Number of plots: {len(ptr)}, Total area: {sum(ptr["area"])} acres, Avg acres/plot: {sum(ptr["area"])/len(ptr):.1f}'
        )
    if len(dsr) != 0:
        print(
            f'DSR plots: Number of plots: {len(dsr)}, Total area: {sum(dsr["area"])} acres, Avg acres/plot: {sum(dsr["area"])/len(dsr):.1f}'
        )
    if len(awd) != 0:
        print(
            f'AWD plots: Number of plots: {len(awd)}, Total area: {sum(awd["area"])} acres, Avg acres/plot: {sum(awd["area"])/len(awd):.1f}'
        )

    return gdf


def generate_from_geojson(
    dir_name="november",
    geojson_fname="TNC_plots_fix.geojson",
    SEED=13,
    random_subset=None,
    start_date="2024-04-15",
    end_date="2024-10-01",
    modalities=["s1", "era5", "s2"],
    orbits=["ASCENDING"],
    num_workers=15,
    checkpoint_interval=15,
    max_retries=2,
    retry_delay=10,
):

    checkpoint_interval = max(checkpoint_interval, num_workers)
    gdf = preprocess(dir_name, geojson_fname, SEED, random_subset)

    dsg = DatasetGenerator(
        source_gdf=gdf,
        start_date=start_date,
        end_date=end_date,
        years=None,
        num_workers=num_workers,
        s1_focal_median_radius_px=None,
        orbits=orbits,
        terrain_correction=False,
    )
    ts_df = dsg.generate_timeseries(
        modalities=modalities,
        weekly_align=False,
        gap_fill=False,
        checkpoint_interval=checkpoint_interval,
        max_retries=max_retries,
        retry_delay=retry_delay,
    )

    # #flatten the list orbits into a single string
    orbits = "-".join(orbits)
    # unroll modalities list into a single string
    modalities = "-".join(modalities)

    save_df(ts_df, f"{modalities}-{orbits}-mean-median-stdev-latlon", fname=None)


def save_df(df, meta, fname=None):

    if fname is None:
        today = datetime.today().strftime("%Y-%m-%d")
        fname = f"ts_s1_era5_longterm_{today}_{meta}.parquet"

    path = os.path.join(os.environ["DATA_DIR"], "dataset")
    # assert if path exists
    if os.path.exists(path):
        path = os.path.join(path, fname)
        df.to_parquet(path)
        print(f"Saved dataset to {path}")
    else:
        print(f"WARNING: File {path} does not exist, saving locally...")
        df.to_parquet(fname)


if __name__ == "__main__":
    fire.Fire(generate_from_geojson)
