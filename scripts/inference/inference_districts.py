# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

#!/usr/bin/env python3
"""
Inference script for district-level rice irrigation mapping predictions.

This script runs inference using trained models (single or ensemble) on district features
and calculates rice growing areas with optional masking.

Copyright (c) Microsoft Corporation. All rights reserved.
"""

import argparse
import glob
import os
import warnings
from pathlib import Path

import geopandas as gpd
import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import rasterio
from exactextract import exact_extract
from rasterio.warp import calculate_default_transform, reproject, Resampling
from tqdm.auto import tqdm

warnings.filterwarnings("ignore")


def reproject_raster(src_path, dst_path, dst_crs):
    """
    Reproject a raster to a new coordinate system.

    Parameters:
    src_path (str): Path to source raster
    dst_path (str): Path to output reprojected raster
    dst_crs (str): Target coordinate system
    """
    with rasterio.open(src_path) as src:
        transform, width, height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds
        )

        kwargs = src.meta.copy()
        kwargs.update(
            {"crs": dst_crs, "transform": transform, "width": width, "height": height}
        )

        with rasterio.open(dst_path, "w", **kwargs) as dst:
            reproject(
                source=rasterio.band(src, 1),
                destination=rasterio.band(dst, 1),
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=transform,
                dst_crs=dst_crs,
                resampling=Resampling.nearest,
            )


def run_ensemble_inference(
    model_dir: Path, feature_dir: Path, output_dir: Path, skip_existing: bool = True
) -> dict:
    """Run inference using ensemble of models on all parquet files."""
    print(f"\nInitializing inference...")
    print(f"Model directory: {model_dir}")
    print(f"Feature directory: {feature_dir}")
    print(f"Output directory: {output_dir}")

    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nCreated output directory: {output_dir}")

    # Load all models
    print("\nLoading models...")
    models = {}
    for model_path in model_dir.glob("*"):
        if model_path.suffix == ".joblib":
            print(f"Loading RF model: {model_path.name}")
            models[f"{model_path.stem}"] = joblib.load(model_path)
        elif model_path.suffix == ".txt":
            print(f"Loading LGB model: {model_path.name}")
            model = lgb.Booster(model_file=str(model_path))
            models[f"{model_path.stem}"] = model

    print(f"Loaded {len(models)} models: {list(models.keys())}")

    if not models:
        raise ValueError(f"No models found in {model_dir}")

    # Find all parquet files
    feature_files = list(feature_dir.glob("*.parquet"))
    print(f"\nFound {len(feature_files)} parquet files to process")

    # Process files serially
    summary_stats = {}

    # Get list of districts to skip
    ignore_districts = []
    if skip_existing:
        ignore_districts = [
            "_".join(parq_file.stem.split("_")[:-1])
            for parq_file in output_dir.glob("*.parquet")
        ]

    for feature_file in tqdm(feature_files, desc="Processing districts"):
        if skip_existing and feature_file.stem in ignore_districts:
            print(f"\n{'='*20} Skipping {feature_file.name} {'='*20}")
            continue

        print(f"\n{'='*20} Processing {feature_file.name} {'='*20}")
        try:
            # Load features
            X_pred = gpd.read_parquet(feature_file)
            num_plots = len(X_pred)
            print(f"Loaded {num_plots} plots")

            X_pred.columns = X_pred.columns.astype(str)

            # Create output GeoDataFrame with only geometry
            out_gdf = X_pred[["geometry"]].copy()

            # Remove label and geometry from features if they exist
            drop_features = [
                "VV_ASCENDING_mean_spline",
                "VH_ASCENDING_mean_spline",
                "ASCENDING_spline_ratio",
                "dates_spline",
                "VV_DESCENDING_mean_spline",
                "VH_DESCENDING_mean_spline",
                "DESCENDING_spline_ratio",
                "SN",
                "temperature_2m",
                "total_precipitation_sum",
                "dates_era5",
                "geometry",
                "label",
                "Latitude",
                "Longitude",
            ]

            feature_cols = X_pred.columns.difference(drop_features)
            X_pred = X_pred[feature_cols]

            print(f"Number of features: {len(feature_cols)}")

            results = {"num_plots": num_plots}

            # Run prediction for each model
            for model_name, model in models.items():
                print(f"\nRunning prediction with {model_name}")

                try:
                    # Ensure all feature names are strings
                    X_pred.columns = X_pred.columns.astype(str)

                    if hasattr(model, "feature_names_in_"):
                        # RF model - align the order of inputs
                        X_pred = X_pred[model.feature_names_in_]
                        missing_features = set(model.feature_names_in_) - set(
                            X_pred.columns
                        )
                        print(
                            f"[RF] model features: {len(set(model.feature_names_in_))}"
                        )

                        if missing_features:
                            print(
                                f"Warning: Missing features for {model_name}: {missing_features}"
                            )

                    elif hasattr(model, "feature_name"):
                        # LGB model
                        required_features = model.feature_name()
                        X_pred = X_pred[required_features]
                        missing_features = set(required_features) - set(X_pred.columns)

                        print(f"[LGB] model features: {len(set(required_features))}")

                        if missing_features:
                            print(
                                f"Warning: Missing features for {model_name}: {missing_features}"
                            )

                    print("Generating predictions...")
                    predictions = model.predict(X_pred)

                    # Handle multiclass LightGBM models that return probability arrays
                    if predictions.ndim == 2:
                        # For multiclass models, get the class with highest probability
                        predictions = np.argmax(predictions, axis=1)
                        predictions = (predictions >= 0.5).astype(int)

                    print(f"predictions.shape: {predictions.shape}")

                    dsr_count = sum(predictions == 1)

                    # Store results
                    results[f"{model_name}_dsr"] = dsr_count
                    out_gdf[f"label_{model_name}"] = predictions

                    print(
                        f"{model_name} predictions: {dsr_count}/{num_plots} DSR plots ({dsr_count/num_plots*100:.2f}%)"
                    )

                except Exception as e:
                    print(f"\n\n!!Error with model {model_name}: {str(e)}!!\n\n")
                    continue

            # Calculate ensemble predictions: use mode of predictions across label columns
            label_cols = [col for col in out_gdf.columns if col.startswith("label_")]
            ensemble_predictions = out_gdf[label_cols].mode(axis=1)
            # Extract the mode value from the DataFrame (first column)
            ensemble_predictions = ensemble_predictions.iloc[:, 0]
            dsr_count = sum(ensemble_predictions == 1)
            results["ensemble_dsr"] = dsr_count
            out_gdf["label_ensemble"] = ensemble_predictions

            # Save predictions if we have any
            if len(results) > 1:  # More than just num_plots
                out_path = output_dir / f"{feature_file.stem}_predictions.parquet"
                out_gdf.to_parquet(out_path)
                print(f"Saved predictions to {out_path}")

            # Store summary stats
            summary_stats[feature_file.stem] = results

            # Print district summary
            print(f"\nDistrict Summary for {feature_file.stem}:")
            print(f"Total plots: {num_plots}")
            for key, value in results.items():
                if key.endswith("_dsr"):
                    model_name = key.replace("_dsr", "")
                    pct = (value / num_plots) * 100
                    print(f"{model_name}: {pct:.2f}% DSR ({value}/{num_plots})")

        except Exception as e:
            print(f"Failed to process {feature_file.stem}: {str(e)}")
            summary_stats[feature_file.stem] = {"error": str(e)}
            continue

    print("\nProcessing complete!")
    print(f"Summary stats collected for {len(summary_stats)} districts")
    return summary_stats


def calculate_district_label_areas(
    folder_path, mask_path=None, label_cols=["label_ensemble"]
):
    """
    Calculate total area per label type across all districts, processing one label at a time.
    Optionally applies a raster mask where calculations are only done for mask value = 1.

    Parameters:
    folder_path (str): Path to folder containing geoparquet files
    mask_path (str, optional): Path to raster mask file (0/1 values)
    label_cols (list): List of label columns to process

    Returns:
    tuple: (detailed DataFrame, pivot table m2, pivot table acres)
    """
    # Initialize list to store results
    results_list = []

    # Get all geoparquet files
    parquet_files = glob.glob(os.path.join(folder_path, "*.parquet"))

    # First, read one file to get label columns
    if not parquet_files:
        print("No parquet files found in the specified folder")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    sample_gdf = gpd.read_parquet(parquet_files[0])

    if label_cols is None:
        label_cols = [col for col in sample_gdf.columns if col.startswith("label_")]

    if not label_cols:
        print("No label columns found in the files")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    # Open mask if provided
    mask_src = None
    if mask_path:
        print(f"Using mask from: {mask_path}")
        mask_src = rasterio.open(mask_path)

    # Process each label type serially
    for label_col in label_cols:
        print(f"\nProcessing {label_col}...")

        # Process each district for this label
        for file_path in parquet_files:
            district_name = Path(file_path).stem
            print(f"  District: {district_name}")

            # Read geoparquet file
            gdf = gpd.read_parquet(file_path)

            # Get appropriate UTM CRS for accurate area calculation
            utm_crs = gdf.estimate_utm_crs()
            print(f"    Using {utm_crs.name} for area calculation")

            # Reproject to UTM
            gdf_utm = gdf.to_crs(utm_crs)

            # Filter for positive labels
            positive_features = gdf_utm[gdf_utm[label_col] == 1]

            if positive_features.empty:
                print(f"    No features found for {label_col}")
                continue

            feature_count = len(positive_features)

            if mask_src:
                # Calculate masked area using exactextract
                stats = exact_extract(
                    mask_src, positive_features.to_crs(mask_src.crs), "sum"
                )

                # Get pixel area in UTM coordinates
                utm_transform = rasterio.transform.from_bounds(
                    *positive_features.total_bounds, mask_src.width, mask_src.height
                )
                pixel_area = abs(utm_transform[0] * utm_transform[4])

                total_area_m2 = sum(stats) * pixel_area
            else:
                # Calculate unmasked area
                total_area_m2 = positive_features.geometry.area.sum()

            if total_area_m2 > 0:
                total_area_acres = total_area_m2 * 0.000247105

                results_list.append(
                    {
                        "district": district_name,
                        "label": label_col,
                        "total_area_m2": total_area_m2,
                        "total_area_acres": total_area_acres,
                        "count": feature_count,
                    }
                )

    # Close mask raster if it was opened
    if mask_src:
        mask_src.close()

    if not results_list:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    # Convert to DataFrame
    results = pd.DataFrame(results_list)

    # Sort by label and area
    results = results.sort_values(["label", "total_area_m2"], ascending=[True, False])

    # Create pivot tables
    pivot_m2 = pd.pivot_table(
        results,
        values="total_area_m2",
        index="district",
        columns="label",
        aggfunc="first",
    ).fillna("0.00")

    pivot_acres = pd.pivot_table(
        results,
        values=["total_area_acres", "count"],
        index="district",
        columns="label",
        aggfunc={"total_area_acres": "first", "count": "first"},
    ).fillna("0.00")

    return results, pivot_m2, pivot_acres


def merge_predictions_state(folder_path, key="label_ensemble"):
    """
    Merge all district predictions into a single state-level GeoDataFrame.

    Parameters:
    folder_path (str): Path to folder containing geoparquet files
    key (str): Label column to filter by (default: 'label_ensemble')

    Returns:
    GeoDataFrame: Combined predictions for all districts
    """
    # Get all geoparquet files
    parquet_files = glob.glob(os.path.join(folder_path, "*.parquet"))

    # Empty list for gdfs
    gdfs = []

    for file_path in parquet_files:
        district_name = Path(file_path).stem
        print(f"  District: {district_name}")

        # Read geoparquet file
        gdf = gpd.read_parquet(file_path)
        gdf["district"] = file_path.split("/")[-1].split("_")[0]

        gdf = gdf[gdf[key] == 1].reset_index(drop=True)
        gdf = gdf[["geometry", "district"]]
        # Append to list
        gdfs.append(gdf)

    combined_gdf = gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True))

    return combined_gdf


def save_dsr_predictions_geojson(predicts_dir):
    """
    Save DSR predictions as separate GeoJSON files per district.

    Parameters:
    predicts_dir (str): Directory containing prediction parquet files
    """
    # Iterate over all parquet files in the directory
    for file in Path(predicts_dir).glob("*.parquet"):
        # Read the file
        df = gpd.read_parquet(file)
        df = df[df["label_ensemble"] == 1].reset_index(drop=True)
        df = df[["geometry"]]

        # Save as geojson
        output_filename = Path(
            os.path.join(
                predicts_dir,
                file.stem.replace("features_predictions", "dsr_predictions"),
            )
        )

        df.to_file(output_filename.with_suffix(".geojson"), driver="GeoJSON")
        print(f"Saved {output_filename.with_suffix('.geojson')}")


def main():
    parser = argparse.ArgumentParser(
        description="Run inference on district features using trained rice irrigation models"
    )

    # Required arguments
    parser.add_argument(
        "--model-dir",
        type=str,
        required=True,
        help="Path to directory containing trained model files (.joblib or .txt)",
    )
    parser.add_argument(
        "--feature-dir",
        type=str,
        required=True,
        help="Path to directory containing district feature parquet files",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Path to directory where predictions and results will be saved",
    )

    # Optional arguments
    parser.add_argument(
        "--mask-path",
        type=str,
        default=None,
        help="Path to raster mask file for area calculations (optional)",
    )
    parser.add_argument(
        "--label-cols",
        type=str,
        nargs="+",
        default=["label_ensemble"],
        help="Label columns to use for area calculations (default: label_ensemble)",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip districts that already have predictions",
    )
    parser.add_argument(
        "--save-geojson",
        action="store_true",
        help="Save DSR predictions as separate GeoJSON files per district",
    )
    parser.add_argument(
        "--merge-state",
        action="store_true",
        help="Merge all district predictions into a single state-level file",
    )
    parser.add_argument(
        "--merge-key",
        type=str,
        default="label_ensemble",
        help="Label column to use when merging state predictions (default: label_ensemble)",
    )

    args = parser.parse_args()

    # Convert to Path objects
    model_dir = Path(args.model_dir)
    feature_dir = Path(args.feature_dir)
    output_dir = Path(args.output_dir)

    # Validate inputs
    if not model_dir.exists():
        raise ValueError(f"Model directory does not exist: {model_dir}")
    if not feature_dir.exists():
        raise ValueError(f"Feature directory does not exist: {feature_dir}")

    # Run inference
    print("=" * 80)
    print("STEP 1: Running inference on districts")
    print("=" * 80)
    summary_stats = run_ensemble_inference(
        model_dir=model_dir,
        feature_dir=feature_dir,
        output_dir=output_dir,
        skip_existing=args.skip_existing,
    )

    # Calculate areas
    print("\n" + "=" * 80)
    print("STEP 2: Calculating district areas")
    print("=" * 80)
    results, pivot_m2, pivot_acres = calculate_district_label_areas(
        folder_path=str(output_dir),
        mask_path=args.mask_path,
        label_cols=args.label_cols,
    )

    if not results.empty:
        print("\nDetailed Results:")
        print(results)

        print("\nAreas by District (square meters):")
        print(pivot_m2)

        print("\nAreas by District (acres):")
        print(pivot_acres)

        # Save results
        results.to_csv(output_dir / "district_areas_detailed.csv", index=False)
        pivot_acres.to_csv(output_dir / "district_areas_acres.csv")
        print(f"\nSaved area calculations to {output_dir}")

        # Print as markdown table
        print("\nMarkdown table:")
        print(pivot_acres.to_markdown())

    # Save GeoJSON files
    if args.save_geojson:
        print("\n" + "=" * 80)
        print("STEP 3: Saving DSR predictions as GeoJSON")
        print("=" * 80)
        save_dsr_predictions_geojson(str(output_dir))

    # Merge state predictions
    if args.merge_state:
        print("\n" + "=" * 80)
        print("STEP 4: Merging state-level predictions")
        print("=" * 80)
        state_gdf = merge_predictions_state(str(output_dir), key=args.merge_key)
        output_path = output_dir / "Punjab_predictions.geojson"
        state_gdf.to_file(output_path, driver="GeoJSON")
        print(f"Saved state-level predictions to {output_path}")

    print("\n" + "=" * 80)
    print("COMPLETE!")
    print("=" * 80)


if __name__ == "__main__":
    main()
