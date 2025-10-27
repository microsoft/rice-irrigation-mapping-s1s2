# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
from dotenv import load_dotenv
from ricemapper.scripts.train import RiceClassifier
import time
import warnings
from tqdm import tqdm
from pathlib import Path
import gc
import argparse
from datetime import datetime

warnings.filterwarnings("ignore")

load_dotenv()


def featurize(
    src_fname,
    save_path,
    date_range,
    device,
    inference=True,
    date_in_weeks=False,
    frequency=7,
):

    rc = RiceClassifier(
        src_fname=src_fname,
        date_range=date_range,
        task="DSR_vs_Rest",
        batch_size=10000,
        device=device,
        inference=inference,
        date_in_weeks=date_in_weeks,
    )

    print(f"Size of input dataframe: {rc.src_df.shape}")

    print("\nBegin Featurizing.....\n")
    try:
        rc.featurize(save_path=save_path, frequency=frequency)
    except Exception as e:
        print(f"Error featurizing district {src_fname}: {e}")
        del rc
        gc.collect()
        raise e

    del rc
    gc.collect()


def main():
    parser = argparse.ArgumentParser(description="Featurize rice data for districts")
    parser.add_argument(
        "--start_date",
        type=str,
        default="2024-06-01",
        help="Start date in YYYY-MM-DD format (default: 2024-06-01)",
    )
    parser.add_argument(
        "--end_date",
        type=str,
        default="2024-09-05",
        help="End date in YYYY-MM-DD format (default: 2024-09-05)",
    )
    parser.add_argument(
        "--input_path",
        type=str,
        default=os.path.join(
            os.environ.get("DATA_DIR", ""), "inference/districts/s1_gamma0"
        ),
        help="Input directory path containing parquet files",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=os.path.join(
            os.environ.get("DATA_DIR", ""),
            "inference/districts/features/06-01_09-05_weeks_gamma0_3_clean_class",
        ),
        help="Output directory path for feature files",
    )
    parser.add_argument(
        "--device", type=str, default=None, help="GPU device number (default: cuda:0)"
    )
    parser.add_argument(
        "--frequency", type=int, default=7, help="Frequency in days (default: 7)"
    )

    args = parser.parse_args()

    # Validate date format
    try:
        datetime.strptime(args.start_date, "%Y-%m-%d")
        datetime.strptime(args.end_date, "%Y-%m-%d")
    except ValueError:
        raise ValueError("Date must be in YYYY-MM-DD format")

    date_range = (args.start_date, args.end_date)
    input_dir = Path(args.input_path)
    output_dir = Path(args.output_path)
    ignore_districts = [
        "_".join(parq_file.stem.split("_")[:-1])
        for parq_file in output_dir.glob("*.parquet")
    ]
    date_in_weeks = True

    # check if output_dir exists, if not, create it
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"Processing files from: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Date range: {date_range}")
    print(f"Device: {args.device}")

    for parq_file in tqdm(
        input_dir.glob("*.parquet"), desc=f"Acquiring features for {input_dir.name}"
    ):

        dist_name = parq_file.stem.split("_")[0]

        if dist_name in ignore_districts:
            print(
                f"\n-----------------Ignoring District: {dist_name}-----------------\n"
            )
            continue
        print(f"\n-----------------Processing District: {dist_name}-----------------\n")

        output_file = os.path.join(output_dir, f"{dist_name}_features.parquet")
        start_time = time.time()
        featurize(
            parq_file,
            output_file,
            date_range,
            args.device,
            inference=True,
            date_in_weeks=date_in_weeks,
            frequency=args.frequency,
        )
        print(
            f"Output saved in {output_file}, completed in {time.time()-start_time} seconds"
        )
        print(f"\n-----------------Completed District: {dist_name}-----------------\n")
        gc.collect()


if __name__ == "__main__":
    main()
