# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
from dotenv import load_dotenv
from ricemapper.scripts.train import RiceClassifier
import time
import warnings
from pathlib import Path
import gc
import argparse

# Suppress all warnings
warnings.filterwarnings("ignore")

load_dotenv()


def featurize(
    src_fname,
    save_path,
    inference=True,
    date_in_weeks=False,
    start_date=None,
    end_date=None,
    device=None,
    frequency=7,
):

    rc = RiceClassifier(
        src_fname=src_fname,
        date_range=(start_date, end_date),
        task="DSR_vs_Rest",
        batch_size=2000,
        device=device,
        inference=inference,
        date_in_weeks=date_in_weeks,
    )

    print(f"Size of input dataframe: {rc.src_df.shape}")

    print("\nBegin Featurizing.....\n")

    rc.featurize(save_path=save_path, frequency=frequency)
    del rc
    gc.collect()


def main():
    parser = argparse.ArgumentParser(
        description="Featurize training data for rice classification"
    )
    parser.add_argument("--start_date", type=str, default="2024-06-01")
    parser.add_argument("--end_date", type=str, default="2024-08-30")
    parser.add_argument(
        "--input_path",
        type=str,
        default=os.path.join(
            os.environ["DATA_DIR"],
            "dataset/s1/s1_gamma0/train_2025-06-16_plots_final_s1.parquet",
        ),
        help="Path to input parquet file",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.path.join(
            os.environ["DATA_DIR"], "dataset/features/06-06_08-30_weeks_gamma0_2_clean"
        ),
        help="Directory to save output features",
    )
    parser.add_argument(
        "--date_in_weeks",
        action="store_true",
        default=True,
        help="Whether to use date in weeks (default: True)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help='Device to use for computation (e.g. "cuda:0", "cuda:1", "cpu")',
    )
    parser.add_argument("--frequency", type=int, default=7)

    args = parser.parse_args()

    input_s1_path = args.input_path
    output_dir = Path(args.output_dir)
    date_in_weeks = args.date_in_weeks
    start_date = args.start_date
    end_date = args.end_date
    device = args.device
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"\n-----------------Processing train-----------------\n")

    output_file = os.path.join(output_dir, f"train_features.parquet")
    start_time = time.time()
    featurize(
        input_s1_path,
        output_file,
        inference=False,
        date_in_weeks=date_in_weeks,
        start_date=start_date,
        end_date=end_date,
        device=device,
        frequency=args.frequency,
    )
    print(
        f"Output saved in {output_file}, completed in {time.time()-start_time} seconds"
    )
    # Delete objects and force garbage collection
    gc.collect()


if __name__ == "__main__":
    main()
