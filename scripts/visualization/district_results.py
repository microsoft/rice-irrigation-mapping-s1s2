# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

#!/usr/bin/env python3
"""
District Level Predictions and Comparison to Government Records

This script visualizes rice irrigation predictions at the district level and
compares them to government estimates. It generates bar plots, scatter plots,
and statistical analyses including correlation coefficients, Jaccard similarity,
and Rank Biased Overlap (RBO).

Copyright (c) Microsoft Corporation. All rights reserved.
Licensed under the MIT License.
"""

import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np


def thousands(x, pos):
    """Format axis labels in thousands."""
    return f"{x/1000:,.0f}"


def jaccard_similarity(list1, list2, k):
    """
    Calculate Jaccard similarity between two lists up to depth k.

    Args:
        list1: First ranked list
        list2: Second ranked list
        k: Depth to consider

    Returns:
        Jaccard similarity score
    """
    set1 = set(list1[:k])
    set2 = set(list2[:k])
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union > 0 else 0


def calculate_rbo(list1, list2, p=0.9):
    """
    Calculate Rank Biased Overlap between two lists.

    Args:
        list1: First ranked list
        list2: Second ranked list
        p: Persistence parameter (default 0.9)

    Returns:
        RBO score
    """
    max_length = max(len(list1), len(list2))
    score = 0
    weight = 1

    for d in range(1, max_length + 1):
        set1 = set(list1[:d])
        set2 = set(list2[:d])
        intersection = set1.intersection(set2)
        agreement = len(intersection) / d
        score += weight * agreement
        weight *= p

    norm_factor = (1 - p) / (1 - p**max_length)
    return score * norm_factor


def visualise_results(df, comparison_cols, threshold, output_dir, output_prefix=""):
    """
    Visualize results for multiple comparison columns.
    Only the first comparison column gets a scatter plot.
    Up to 4 comparison columns are supported for color distinction.

    Args:
        df: DataFrame with district data
        comparison_cols: List of column names to compare
        threshold: Threshold for percentage difference annotations
        output_dir: Directory to save output files
        output_prefix: Prefix for output filenames
    """
    # Set general font sizes
    plt.rcParams["font.size"] = 10
    plt.rcParams["axes.labelsize"] = 12
    plt.rcParams["axes.titlesize"] = 14
    plt.rcParams["legend.fontsize"] = 10

    # Color palette for up to 4 comparison columns
    color_palette = ["lightcoral", "mediumseagreen", "goldenrod", "slateblue"]
    bar_labels = []
    if len(comparison_cols) == 1:
        bar_labels = ["Model Prediction"]
    else:
        bar_labels = [col.replace("_", " ").title() for col in comparison_cols]

    # First figure: Original plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(30, 12))

    # Sort data by Government Estimates for the bar plot
    df_sorted = df.sort_values("Govt. Estimates", ascending=False)

    # Plot 1: Bar plot
    x = range(len(df_sorted))
    width = 0.8 / (len(comparison_cols) + 1)  # dynamic width for multiple bars

    # Government Estimates bar
    ax1.barh(
        x,
        df_sorted["Govt. Estimates"],
        width,
        label="Government Estimates",
        color="skyblue",
        alpha=0.7,
    )

    # Model predictions bars
    for idx, col in enumerate(comparison_cols):
        offset = width * (idx + 1)
        color = color_palette[idx % len(color_palette)]
        ax1.barh(
            [i + offset for i in x],
            df_sorted[col],
            width,
            label=bar_labels[idx],
            color=color,
            alpha=0.7,
        )

    # Customize bar plot
    ax1.set_yticks([i + width * (len(comparison_cols) / 2) for i in x])
    ax1.set_yticklabels(df_sorted["District"], fontsize=20)
    ax1.tick_params(axis="x", labelsize=20)
    ax1.xaxis.set_major_formatter(FuncFormatter(thousands))
    ax1.set_xlabel("Area (acres, thousands)", fontsize=22)
    ax1.set_title("Government Estimates vs Model Predictions", fontsize=22)
    ax1.legend(fontsize=18)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Scatter plot with log scales (only for the first comparison column)
    comp_col0 = comparison_cols[0]
    ax2.scatter(
        df["Govt. Estimates"], df[comp_col0], alpha=0.6, color=color_palette[0], s=200
    )
    ax2.set_xscale("log")
    ax2.set_yscale("log")

    # Add diagonal line for reference
    min_val = min(df["Govt. Estimates"].min(), df[comp_col0].min())
    max_val = max(df["Govt. Estimates"].max(), df[comp_col0].max())
    ax2.plot(
        [min_val, max_val],
        [min_val, max_val],
        "k--",
        alpha=0.5,
        label="Perfect Agreement",
    )

    # Customize scatter plot
    ax2.set_xlabel("Government Estimates (acres, log scale)", fontsize=22)
    ax2.set_ylabel("Model Predictions (acres, log scale)", fontsize=22)
    ax2.set_title("Scatter Plot (Log Scales)", fontsize=22)
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(axis="both", which="major", labelsize=20)
    ax2.legend(fontsize=18)

    # Collect points to annotate first
    points_to_annotate = []
    for i, row in df.iterrows():
        if (
            abs(row["Govt. Estimates"] - row[comp_col0]) / row["Govt. Estimates"]
        ) > threshold:
            points_to_annotate.append(row)

    # Add jittered annotations
    for row in points_to_annotate:
        angle = np.random.uniform(0, 2 * np.pi)
        radius = np.random.uniform(80, 120)  # pixels
        dx = radius * np.cos(angle)
        dy = radius * np.sin(angle)
        ax2.annotate(
            row["District"],
            xy=(row["Govt. Estimates"], row[comp_col0]),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=18,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.7),
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.2", alpha=0.6),
        )

    plt.tight_layout()

    # Save figure
    cols_str = "_".join(comparison_cols)
    if output_prefix:
        output_file = os.path.join(output_dir, f"{output_prefix}_{cols_str}.png")
    else:
        output_file = os.path.join(output_dir, f"comparison_{cols_str}.png")
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    print(f"\nSaved figure to: {output_file}")
    plt.close()

    # Redirect statistics to file
    if output_prefix:
        stats_file = os.path.join(output_dir, f"{output_prefix}_{cols_str}_stats.txt")
    else:
        stats_file = os.path.join(output_dir, f"comparison_{cols_str}_stats.txt")

    with open(stats_file, "w") as f:
        # Print statistics for each comparison column
        for idx, col in enumerate(comparison_cols):
            f.write(f"\n{'='*50}\n")
            f.write(f"Statistics for {col}\n")
            f.write(f"{'='*50}\n")

            correlation = df["Govt. Estimates"].corr(df[col])
            f.write(f"\nPearson correlation coefficient: {correlation:.3f}\n")

            df_copy = df.copy()
            df_copy["Abs_Diff"] = abs(df_copy["Govt. Estimates"] - df_copy[col])
            f.write("\nDistricts with largest absolute differences:\n")
            top_diff = df_copy.nlargest(5, "Abs_Diff")[
                ["District", "Govt. Estimates", col, "Abs_Diff"]
            ]
            f.write(top_diff.to_string(index=False))
            f.write("\n")

            # Get rankings
            govt_ranking = df.sort_values("Govt. Estimates", ascending=False)[
                "District"
            ].tolist()
            pred_ranking = df.sort_values(col, ascending=False)["District"].tolist()

            # Calculate RBO for different p values
            p_values = [0.8, 0.85, 0.9, 0.95]
            f.write("\nRank Biased Overlap (RBO) scores:\n")
            for p in p_values:
                rbo = calculate_rbo(govt_ranking, pred_ranking, p)
                f.write(f"  RBO (p={p}): {rbo:.3f}\n")

            # Calculate Jaccard similarity at different depths
            depths = [5, 10, 15, len(govt_ranking)]
            f.write("\nJaccard Similarity at different depths:\n")
            for k in depths:
                similarity = jaccard_similarity(govt_ranking, pred_ranking, k)
                f.write(f"  Top-{k} Jaccard Similarity: {similarity:.3f}\n")

    print(f"Saved statistics to: {stats_file}")

    # Also print to console
    print(f"\n{'='*50}")
    print(f"Statistics for {', '.join(comparison_cols)}")
    print(f"{'='*50}")
    with open(stats_file, "r") as f:
        print(f.read())


def create_hybrid_column(df, districts_to_unmask):
    """
    Create a hybrid masked column that uses unmasked values for specified districts.

    Args:
        df: DataFrame with ensemble and ensemble_masked columns
        districts_to_unmask: List of district names to use unmasked values for

    Returns:
        DataFrame with new 'ensemble_masked_hybrid' column
    """
    df = df.copy()
    df["ensemble_masked_hybrid"] = df["ensemble_masked"]
    df.loc[df["District"].isin(districts_to_unmask), "ensemble_masked_hybrid"] = df.loc[
        df["District"].isin(districts_to_unmask), "ensemble"
    ]
    return df


def main():
    parser = argparse.ArgumentParser(
        description="Visualize district-level rice irrigation predictions and compare to government estimates.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with single comparison column
  python district_results.py --input data.csv --output-dir ./results --comparison-cols ensemble
  
  # Compare multiple models
  python district_results.py --input data.csv --output-dir ./results --comparison-cols ensemble ensemble_masked
  
  # Create hybrid column with unmasked districts
  python district_results.py --input data.csv --output-dir ./results --comparison-cols ensemble_masked_hybrid --unmask-districts "Sri Muktsar Sahib" Fazilka
        """,
    )

    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input CSV file with district predictions and government estimates",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to save output figures and statistics",
    )
    parser.add_argument(
        "--comparison-cols",
        type=str,
        nargs="+",
        default=["ensemble"],
        help="Column name(s) to compare against government estimates (default: ensemble)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.7,
        help="Threshold for percentage difference to annotate on scatter plot (default: 0.7)",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="",
        help="Prefix for output filenames (optional)",
    )
    parser.add_argument(
        "--unmask-districts",
        type=str,
        nargs="+",
        default=None,
        help="Districts to unmask when creating hybrid column (requires ensemble_masked column)",
    )

    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Read the CSV file
    print(f"Reading data from: {args.input}")
    df = pd.read_csv(args.input)

    # Sort alphabetically by district
    df = df.sort_values("District").reset_index(drop=True)

    print(f"Loaded {len(df)} districts")
    print(f"Available columns: {', '.join(df.columns)}")

    # Create hybrid column if requested
    if args.unmask_districts:
        if "ensemble_masked" not in df.columns or "ensemble" not in df.columns:
            print(
                "Warning: Cannot create hybrid column - 'ensemble' or 'ensemble_masked' not found in data"
            )
        else:
            print(
                f"\nCreating hybrid column with unmasked districts: {', '.join(args.unmask_districts)}"
            )
            df = create_hybrid_column(df, args.unmask_districts)
            print("Created 'ensemble_masked_hybrid' column")

    # Validate comparison columns exist
    missing_cols = [col for col in args.comparison_cols if col not in df.columns]
    if missing_cols:
        print(
            f"Error: The following comparison columns are not in the data: {', '.join(missing_cols)}"
        )
        print(f"Available columns: {', '.join(df.columns)}")
        return 1

    # Validate required column exists
    if "Govt. Estimates" not in df.columns:
        print("Error: 'Govt. Estimates' column not found in data")
        return 1

    # Generate visualizations
    print(f"\nGenerating visualizations for: {', '.join(args.comparison_cols)}")
    print(f"Threshold for annotations: {args.threshold}")

    visualise_results(
        df=df,
        comparison_cols=args.comparison_cols,
        threshold=args.threshold,
        output_dir=args.output_dir,
        output_prefix=args.output_prefix,
    )

    print(f"\nAll outputs saved to: {args.output_dir}")
    return 0


if __name__ == "__main__":
    exit(main())
