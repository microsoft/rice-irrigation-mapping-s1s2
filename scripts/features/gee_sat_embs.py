# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import argparse
import os
import geopandas as gpd
from ricemapper.utils.gee.sat_embs import (
    SatelliteEmbeddingExtractor,
    flatten_sat_embeddings,
)
import ee


def main():
    parser = argparse.ArgumentParser(
        description="Generate Google Satellite Embeddings for georeferenced polygons."
    )
    parser.add_argument(
        "--input-parquet",
        type=str,
        required=True,
        help="Path to input train_features.parquet file",
    )
    parser.add_argument(
        "--output-dir", type=str, required=True, help="Directory to save output files"
    )
    parser.add_argument(
        "--year", type=int, default=2024, help="Year for satellite embeddings"
    )
    parser.add_argument(
        "--head",
        type=int,
        default=None,
        help="Number of rows to use from input (optional)",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize Earth Engine
    try:
        ee.Initialize()
    except Exception:
        try:
            ee.Authenticate()
            ee.Initialize()
        except Exception as e:
            raise Exception(f"Failed to initialize Google Earth Engine: {e}")

    # Load polygons
    train_plots = gpd.read_parquet(args.input_parquet)
    if args.head:
        train_plots = train_plots.head(args.head)

    # Generate satellite embeddings
    sat_embs_func = SatelliteEmbeddingExtractor(year=args.year, polys=train_plots)
    embs = sat_embs_func.extract_embeddings()

    # Keep relevant columns
    cols_to_keep = ["Class", "label", "sat_embs", "geometry"]
    sat_embs = embs[cols_to_keep].copy()

    # Flatten embeddings
    sat_embs = flatten_sat_embeddings(sat_embs, "sat_embs", "sat_emb_", True)

    # Save outputs
    output_parquet = os.path.join(args.output_dir, "satellite_embeddings.parquet")
    sat_embs.to_parquet(output_parquet)
    print(f"Saved flattened satellite embeddings to {output_parquet}")


if __name__ == "__main__":
    main()
