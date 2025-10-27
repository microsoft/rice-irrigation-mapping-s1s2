# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
import pandas as pd
import numpy as np
import geopandas as gpd
from datetime import datetime
from typing import List, Dict, Union, Optional, Tuple
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import optuna
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
from concurrent.futures import ProcessPoolExecutor
import warnings
from ricemapper.utils.utils import (
    sample_columns_with_aggregation,
    generate_column_names,
    count_inflection_points,
    find_min_max_mean_stdev,
    find_troughs_and_crests,
)
from ricemapper.utils.utils import (
    generate_s1_spline,
    flatten_2d_features,
    RandomBaseline,
)
from ricemapper.models.presto.presto import Presto
from ricemapper.models.presto.presto_ds import PrestoDataset
from tqdm import tqdm
from dotenv import load_dotenv


load_dotenv()


input_feature_set = [
    "VV_ASCENDING_mean_spline",
    "VH_ASCENDING_mean_spline",
    "ASCENDING_spline_ratio",
    "VV_DESCENDING_mean_spline",
    "VH_DESCENDING_mean_spline",
    "DESCENDING_spline_ratio",
    "dates_spline",
    "SN",
    "label",
    "Latitude",
    "Longitude",
    "ASCENDING_ratio_gaussian_a",
    "ASCENDING_ratio_gaussian_b",
    "ASCENDING_ratio_gaussian_c",
    "ASCENDING_ratio_gaussian_r2",
    "dates_era5",
    "temperature_2m",
    "total_precipitation_sum",
]
vars_asc = [
    "VV_ASCENDING_mean_spline",
    "VH_ASCENDING_mean_spline",
    "ASCENDING_spline_ratio",
    "ASCENDING_ratio_gaussian_a",
    "ASCENDING_ratio_gaussian_b",
    "ASCENDING_ratio_gaussian_c",
    "ASCENDING_ratio_gaussian_r2",
]
vars_desc = [
    "VV_DESCENDING_mean_spline",
    "VH_DESCENDING_mean_spline",
    "DESCENDING_spline_ratio",
    "DESCENDING_ratio_gaussian_a",
    "DESCENDING_ratio_gaussian_b",
    "DESCENDING_ratio_gaussian_c",
    "DESCENDING_ratio_gaussian_r2",
]
era5_vars = ["temperature_2m", "total_precipitation_sum", "dates_era5"]
base_vars = ["dates_spline", "Latitude", "Longitude", "label", "Class"]  #'SN']


def _predict_batch_rf(batch_df, model):
    """Standalone prediction function for Random Forest"""
    return model.predict(batch_df)


def _predict_batch_xgb(batch_df, model):
    """Standalone prediction function for XGBoost"""
    return model.predict(batch_df)


class RiceClassifier:
    """A class to handle rice field classification tasks using S-1 & ERA5"""

    def __init__(
        self,
        src_fname: str = "dataset/new_plots/ts_s1-era5_2024-12-10_ASC_DESC.parquet",
        date_range: Tuple[str, str] = ("2024-04-24", "2024-08-22"),
        task: str = "All",
        orbits: List[str] = ["ASCENDING"],
        do_era5=False,
        batch_size: int = 256,
        device: str = None,
        seed: int = 42,
        k: int = 3,
        inference=False,
        date_in_weeks=False,
    ):
        """
        Initialize the RiceClassifier.

        Args:
            src_df: Input dataframe with required columns
            date_range: Tuple of (start_date, end_date) strings
            task: Classification task ('All', 'DSR_vs_Rest', 'AWD_vs_PTR')
            orbits: List of orbits to use ['ASCENDING', 'DESCENDING']
            batch_size: Batch size for processing
            device: Device to use for Presto (defaults to cuda if available)
            seed: Random seed
            k: number of troughs/crests to detect
            GPU: GPU number to use
            inference: True if running inference
        """

        fname = os.path.join(os.environ["DATA_DIR"], src_fname)
        self.src_df = gpd.read_parquet(fname)
        self.src_df.set_crs(epsg=4326, inplace=True)
        self.inference = inference
        if inference:
            print("Inference mode!")
            self._clean_inference()
        else:
            self._clean_inputs()

        self.start_date, self.end_date = date_range
        self.task = task
        self.orbits = orbits
        self.do_era5 = do_era5
        self.batch_size = batch_size
        # Set up device properly
        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Only set CUDA device if using CUDA
        if self.device.type == "cuda":
            torch.cuda.set_device(self.device)

        print(f"DEBUG: Device: {self.device}")

        # Clear any existing CUDA cache to start fresh
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.seed = seed
        self.k = k
        self.date_in_weeks = date_in_weeks

        # Validate inputs
        self._validate_inputs()

        # Initialize model components
        self.presto_model = None
        self.trained_models = {}
        self.feature_df = None

        # Set up class mappings based on task
        self._setup_class_mappings()
        self._setup_input_feature_set()
        # Initialize Presto model
        self._init_presto_model()

    def _init_presto_model(self):
        """Initialize the Presto model and move to device"""

        self.presto_model = Presto.load_pretrained().to(self.device)
        self.presto_model.eval()

    def _setup_input_feature_set(self):
        self.input_feature_set = base_vars

        if "ASCENDING" in self.orbits:
            self.input_feature_set += vars_asc

        if "DESCENDING" in self.orbits:
            self.input_feature_set += vars_desc

        if self.do_era5:
            self.input_feature_set += era5_vars

        if self.inference:
            self.src_df["label"] = None

    def _remove_short_timeseries(self, df, columns_to_check, min_length=5):
        # Create a mask for rows to keep
        valid_rows = df.apply(
            lambda row: all(len(row[col]) >= min_length for col in columns_to_check),
            axis=1,
        )
        print(f"DEBUG: Removed short timeseries: #{len(df[~valid_rows])}")
        return df[valid_rows]

    def _remove_invalid_rows(self, df, columns_to_check=None):
        def has_invalid_values(x):

            if isinstance(x, list):
                return any(pd.isna(val) or np.isinf(val) for val in x)
            elif isinstance(x, (np.ndarray, pd.Series)):
                return pd.isna(x).any() or np.isinf(x).any()
            elif isinstance(x, (int, float)):
                return pd.isna(x) or np.isinf(x)
            else:  # strings/floats or other types
                print(f"Unknown type: {x}:{type(x)}")
                return pd.isna(x)

        # If no columns specified, check all columns
        columns_to_check = columns_to_check or df.columns
        valid_rows = ~df[columns_to_check].apply(
            lambda row: any(has_invalid_values(val) for val in row), axis=1
        )
        return df[valid_rows]

    def _remove_nans(self, df, columns_to_check=None):
        # If no columns specified, check all columns
        columns_to_check = columns_to_check or df.columns
        valid_rows = ~df[columns_to_check].apply(
            lambda row: any(pd.isna(val.any()) for val in row), axis=1
        )
        return df[valid_rows]

    def _remove_non_timeseries(self, df, columns_to_check):
        # Create a mask for rows to keep
        valid_rows = df.apply(
            lambda row: all(
                isinstance(row[col], (list, np.ndarray)) for col in columns_to_check
            ),
            axis=1,
        )
        # print indexes of removed rows
        removed_rows = df[~valid_rows].index
        print(f"DEBUG: Removed non-timeseriesrows: {removed_rows}")
        return df[valid_rows]

    def _clean_inputs(self):
        columns_to_check = [
            "VV_ASCENDING_mean",
            "VH_ASCENDING_mean",
            "VV_ASCENDING_std",
            "VH_ASCENDING_std",
            "VV_ASCENDING_median",
            "VH_ASCENDING_median",
            "VV_DESCENDING_mean",
            "VH_DESCENDING_mean",
            "VV_DESCENDING_std",
            "VH_DESCENDING_std",
            "VV_DESCENDING_median",
            "VH_DESCENDING_median",
            "dates_ASCENDING",
        ]

        # check if columns Latitude and Longitude are present
        if (
            "Latitude" not in self.src_df.columns
            or "Longitude" not in self.src_df.columns
        ):
            self.src_df["Latitude"] = self.src_df.geometry.centroid.y
            self.src_df["Longitude"] = self.src_df.geometry.centroid.x
        num_rows = len(self.src_df)

        # ensure that all columns_to_check is in df.columns, otherwise remove
        columns_to_check = [
            col for col in columns_to_check if col in self.src_df.columns
        ]
        self.src_df = self._remove_non_timeseries(self.src_df, columns_to_check)
        self.src_df = self._remove_short_timeseries(
            self.src_df, ["VV_ASCENDING_mean", "VH_ASCENDING_mean"]
        )
        print(
            f"DEBUG: Removed {num_rows - len(self.src_df)} rows. Dataframe shape: {self.src_df.shape}"
        )

    def _clean_inference(self):
        # remove all rows that have any nans
        self.src_df = self.src_df.dropna()
        if (
            "Latitude" not in self.src_df.columns
            or "Longitude" not in self.src_df.columns
        ):
            self.src_df["Latitude"] = self.src_df.geometry.centroid.y
            self.src_df["Longitude"] = self.src_df.geometry.centroid.x

    def _validate_inputs(self):
        """Validate input parameters"""
        if not isinstance(self.src_df, (pd.DataFrame, gpd.GeoDataFrame)):
            raise ValueError("src_df must be a pandas DataFrame or GeoDataFrame")

        valid_tasks = ["All", "DSR_vs_Rest", "AWD_vs_PTR"]
        if self.task not in valid_tasks:
            raise ValueError(f"Task must be one of {valid_tasks}")

        valid_orbits = ["ASCENDING", "DESCENDING"]
        if not all(orbit in valid_orbits for orbit in self.orbits):
            raise ValueError(f"Orbits must be from {valid_orbits}")

    def _setup_class_mappings(self):
        """Set up class mappings based on classification task"""
        if self.task == "All":
            self.class_map = {"PTR": 0, "DSR": 1, "AWD": 2}
        elif self.task == "DSR_vs_Rest":
            self.class_map = {"PTR_AWD": 0, "DSR": 1}
        else:  # AWD_vs_PTR
            self.class_map = {"PTR": 0, "AWD": 1}

    def _prepare_data(self):
        """Prepare data based on classification task"""
        df = self.src_df.copy()

        if self.task == "DSR_vs_Rest":
            ptr = df[df["Class"] == "PTR"].copy()
            dsr = df[df["Class"] == "DSR"].copy()
            awd = df[df["Class"] == "AWD"].copy()

            ptr["label"] = "PTR_AWD"
            dsr["label"] = "DSR"
            awd["label"] = "PTR_AWD"

            # Remove overlapping geometries
            if "geometry" in df.columns:
                self._remove_overlapping_geometries(dsr, ptr)
                self._remove_overlapping_geometries(dsr, awd)

            return pd.concat([ptr, dsr, awd], axis=0).reset_index(drop=True)

        elif self.task == "AWD_vs_PTR":
            df = df[df["Class"].isin(["AWD", "PTR"])].copy()
            df["label"] = df["Class"]
            return df

        else:  # All classes
            df["label"] = df["Class"]
            return df

    @staticmethod
    def _remove_overlapping_geometries(gdf1: gpd.GeoDataFrame, gdf2: gpd.GeoDataFrame):
        """Remove overlapping geometries between two GeoDataFrames"""
        overlap_mask_1 = gdf1["geometry"].apply(
            lambda geom: gdf2["geometry"].intersects(geom).any()
        )
        overlap_mask_2 = gdf2["geometry"].apply(
            lambda geom: gdf1["geometry"].intersects(geom).any()
        )

        gdf1.drop(gdf1[overlap_mask_1].index, inplace=True)
        gdf2.drop(gdf2[overlap_mask_2].index, inplace=True)

    # preserves indices
    @staticmethod
    def _generate_handcrafted_features_batch(args) -> pd.DataFrame:
        """Generate handcrafted features for a batch of data"""
        df, input_feature_set, start_date, end_date, orbits, k = args

        # Keep track of original indices
        batch_df = df.copy()
        original_indices = batch_df.index

        # Generate spline features
        batch_df = generate_s1_spline(
            batch_df,
            start_date=start_date,
            end_date=end_date,
            mode="temporal_smoothing",
            sigma=0.5,
            prefix=None,
            variable="mean",
            convert_to_db=False,
            orbits=orbits,
        )
        # fileter input_feature_set to only include columns that are in batch_df
        input_feature_set = [
            col for col in input_feature_set if col in batch_df.columns
        ]
        batch_df = batch_df[input_feature_set]

        variables = []
        if "ASCENDING" in orbits:
            variables.extend(
                [
                    "VV_ASCENDING_mean_spline",
                    "VH_ASCENDING_mean_spline",
                    "ASCENDING_spline_ratio",
                ]
            )

        if "DESCENDING" in orbits:
            variables.extend(
                [
                    "VV_DESCENDING_mean_spline",
                    "VH_DESCENDING_mean_spline",
                    "DESCENDING_spline_ratio",
                ]
            )

        # PRE-FILTER: Remove rows with invalid time series data
        valid_rows = batch_df.apply(
            lambda row: all(
                isinstance(row[v], (list, np.ndarray))
                and len(row[v]) > 1
                and not pd.isna(row[v]).any()
                and not np.isinf(row[v]).any()
                for v in variables
            ),
            axis=1,
        )

        batch_df = batch_df[valid_rows].copy()

        if len(batch_df) == 0:
            return pd.DataFrame()  # Return empty DataFrame if no valid rows

        trough_crest_indices = generate_column_names(variables, k=k)

        # Generate features only for valid rows while preserving their original indices
        batch_df[trough_crest_indices] = batch_df.apply(
            find_troughs_and_crests,
            args=(start_date, "dates_spline", variables, k),
            axis=1,
        )

        for v in variables:
            batch_df[f"inflection_points_{v}"] = batch_df[v].apply(
                count_inflection_points
            )
            (
                batch_df[f"min_{v}"],
                batch_df[f"max_{v}"],
                batch_df[f"mean_{v}"],
                batch_df[f"std_{v}"],
            ) = zip(*batch_df[v].apply(find_min_max_mean_stdev))

        # Verify indices are preserved
        # check how many indices are lost
        print(f"Indices lost: {len(original_indices) - len(batch_df.index)}")

        if "Class" in batch_df.columns:
            print(f'Unique classes: {batch_df["Class"].unique()}')

        return batch_df

    def _generate_presto_embeddings(
        self, df: pd.DataFrame, frequency: int = 7
    ) -> np.ndarray:
        """Generate Presto embeddings for a batch of data"""
        # Implementation of Presto feature generation from notebook
        results_df = df.copy()

        presto_vars = []
        if "ASCENDING" in self.orbits:
            presto_vars.extend(vars_asc[0:2])
        if "DESCENDING" in self.orbits:
            presto_vars.extend(vars_desc[0:2])
        if self.do_era5:
            presto_vars.extend(era5_vars[0:2])

        print(f"[PRESTO] frequency: {frequency}\n")

        # Resample
        sampled_df = sample_columns_with_aggregation(
            df=results_df,
            variables=presto_vars,
            date_col="dates_spline",
            frequency=frequency,
            convert_to_db=True,
        )

        # convert labels to int
        # check if lable column is present
        if "label" in sampled_df.columns:
            sampled_df["label"] = sampled_df["label"].map(self.class_map)
        else:
            sampled_df["label"] = 9999  # dummy label

        # Check if the DataFrame is empty before accessing the last row
        if len(sampled_df) == 0:
            print("Warning: No valid data after sampling. Returning empty array.")
            return np.array([])

        print(f'DEBUG: NUMBER OF TIMESTEPS: {len(sampled_df["dates_spline"].iloc[-1])}')
        assert (
            len(sampled_df["dates_spline"].iloc[-1]) < 25
        ), "more than 24 timesteps is not allowed for Presto"

        # labels = []

        for orbit in self.orbits:

            if orbit == "ASCENDING":
                input_s1_bands = [
                    "VV_ASCENDING_mean_spline",
                    "VH_ASCENDING_mean_spline",
                ]
            else:
                input_s1_bands = [
                    "VV_DESCENDING_mean_spline",
                    "VH_DESCENDING_mean_spline",
                ]

            input_era5_bands = ["temperature_2m", "total_precipitation_sum"]

            # Create dataset for full data
            dataset = PrestoDataset(
                source=sampled_df,
                input_s1_bands=input_s1_bands,
                input_era5_bands=input_era5_bands,
                date_col="dates_spline",
                lat_col="Latitude",
                lon_col="Longitude",
                label_col="label",
                do_era5=self.do_era5,
                date_in_weeks=self.date_in_weeks,
            )

            # Create dataloader with built-in batching
            dataloader = DataLoader(
                dataset,
                batch_size=self.batch_size * 2,
                shuffle=False,
                pin_memory=False if self.device.type == "cpu" else True,
            )

            features_list = []

            # Process all batches
            with torch.no_grad():
                for (x, mask, dw, latlons, month), _ in tqdm(
                    dataloader, desc=f"Generating {orbit} embeddings"
                ):
                    # Move tensors to device
                    x = x.to(self.device)
                    mask = mask.to(self.device)
                    dw = dw.to(self.device)
                    latlons = latlons.to(self.device)
                    month = month.to(self.device)

                    # Generate encodings
                    encodings, _, _ = self.presto_model.encoder(
                        x,
                        dynamic_world=dw,
                        mask=mask,
                        latlons=latlons,
                        month=month,
                        eval_task=False,
                    )

                    features_list.append(encodings.cpu().numpy())

        # Concatenate all features
        return np.concatenate(features_list)

    # properly handing indices to make sure geometries are not lost
    def featurize(
        self,
        save_path: Optional[str] = None,
        n_jobs: int = -1,
        batch_size: Optional[int] = None,
        frequency: int = 7,
    ) -> pd.DataFrame:
        """
        Generate features for the dataset.

        Args:
            save_path: Optional path to save features
            n_jobs: Number of CPU jobs for parallel processing
            batch_size: Optional override for batch size

        Returns:
            DataFrame with generated features
        """
        batch_size = batch_size or self.batch_size

        # Store original geometry and crs
        original_geometry = self.src_df.geometry
        original_crs = self.src_df.crs

        if self.inference:
            prepared_data = self.src_df.copy()
        else:
            prepared_data = self._prepare_data()

        print(f"Data points: {len(prepared_data)}, cols: {prepared_data.columns}")

        n_jobs = n_jobs if n_jobs > 0 else os.cpu_count()

        # Prepare batches for handcrafted features
        n_batches = len(prepared_data) // batch_size + (
            1 if len(prepared_data) % batch_size != 0 else 0
        )
        idx_batches = np.array_split(prepared_data.index, n_batches)
        data_batches = [prepared_data.loc[idx_batch] for idx_batch in idx_batches]

        print(f"[FEATURIZE] Number of batches: {n_batches}")

        # Create argument tuples for each batch
        feature_args = [
            (
                batch,
                self.input_feature_set,
                self.start_date,
                self.end_date,
                self.orbits,
                self.k,
            )
            for batch in data_batches
        ]

        # Generate handcrafted features in parallel using the static method
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            handcrafted_features = list(
                tqdm(
                    executor.map(
                        RiceClassifier._generate_handcrafted_features_batch,
                        feature_args,
                    ),
                    total=len(feature_args),
                    desc="Generating handcrafted features",
                )
            )

        # Concatenate while preserving original indices
        self.feature_df = pd.concat(handcrafted_features, axis=0, ignore_index=False)

        # Print indices before and after for verification
        print("Unique indices in feature_df:", len(self.feature_df.index.unique()))
        print("Sample of indices:", self.feature_df.index[:5])

        # Generate Presto features for filtered dataset
        presto_features = self._generate_presto_embeddings(
            self.feature_df, frequency=frequency
        )

        # Add Presto features to the dataframe
        self.feature_df["presto_features_a"] = list(presto_features)
        print(f"Presto features shape: {len(self.feature_df)}")
        self.feature_df = flatten_2d_features(self.feature_df, "presto_features_a")
        print(f"Presto features shape: {len(self.feature_df)}")

        illegal_features = ["SN"]
        columns_to_drop = illegal_features
        existing_columns_to_drop = [
            col for col in columns_to_drop if col in self.feature_df.columns
        ]
        self.feature_df = self.feature_df.drop(columns=existing_columns_to_drop, axis=1)

        # Get preserved indices after feature generation
        preserved_indices = self.feature_df.index

        # First check if indices exist in original geometry
        valid_indices = preserved_indices[preserved_indices.isin(self.src_df.index)]

        if len(valid_indices) < len(preserved_indices):
            print(
                f"Warning: {len(preserved_indices) - len(valid_indices)} indices were not found in original geometry"
            )
            # Filter feature_df to only include valid indices
            self.feature_df = self.feature_df.loc[valid_indices]

        # Use only the geometries from preserved rows
        preserved_geometry = original_geometry.loc[valid_indices]

        # Verify alignment
        assert len(preserved_geometry) == len(
            self.feature_df
        ), "Geometry and feature lengths don't match"
        assert all(
            preserved_geometry.index == self.feature_df.index
        ), "Indices don't align"

        # Convert to GeoDataFrame with correct geometries
        self.feature_df = gpd.GeoDataFrame(
            self.feature_df, geometry=preserved_geometry, crs=original_crs
        )

        # save the original labels
        if "Class" in self.feature_df.columns:
            self.feature_df["label"] = self.feature_df["Class"]

        print("Features Shape:", self.feature_df.shape)
        print("Preserved rows:", len(valid_indices))
        print("Dropped rows:", len(original_geometry) - len(valid_indices))

        if save_path:
            self.feature_df.to_parquet(save_path)

        return self.feature_df

    def train(
        self,
        models: List[str] = ["RF"],
        test_size: float = 0.2,
        param_ranges: Optional[Dict] = None,
        metrics: List[str] = ["accuracy", "f1_macro", "f1_weighted"],
        opt_metric: str = "f1_weighted",
        n_trials: int = 100,
        feature_path: Optional[str] = None,
        cv_folds: Optional[int] = None,
        rand_K: int = 1000,
    ) -> Dict:
        """
        Train specified models and optimize hyperparameters.
        Stores training data splits and results as class variables.

        Args:
            models: List of models to train ('RF', 'XGB', 'Random', 'All')
            test_size: Train/test split ratio
            param_ranges: Dictionary of parameter ranges for optimization
            metrics: List of metrics to report
            opt_metric: Metric to optimize for
            n_trials: Number of optimization trials
            feature_path: Optional path to load precomputed features from
            rand_K: Number of random trials for random baseline model

        Returns:
            Dictionary of best models and their parameters
        """
        # Load features if path provided
        if feature_path is not None:
            if feature_path.endswith(".parquet"):
                self.feature_df = pd.read_parquet(feature_path)
            elif feature_path.endswith(".csv"):
                self.feature_df = pd.read_csv(feature_path)
            else:
                raise ValueError("feature_path must end in .parquet or .csv")

        # Check if features are available
        if self.feature_df is None:
            raise ValueError("Must either run featurize() or provide feature_path")

        if "geometry" in self.feature_df.columns:
            self.feature_df.drop("geometry", axis=1, inplace=True)

        X = self.feature_df.drop(["label"], axis=1)
        y = self.feature_df["label"].map(self.class_map)

        # Store splits as class variables for later use
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.seed, stratify=y
        )

        if models == ["All"]:
            models = ["RF", "XGB", "Random"]

        results = {}
        for model_name in models:

            print(f"\n\n-------------TRAINING {model_name}------------------\n\n")
            if model_name == "Random":
                results[model_name] = {
                    "params": None,
                    "model": None,
                    "metrics": self._train_random_baseline(
                        self.X_train,
                        self.y_train,
                        self.X_test,
                        self.y_test,
                        metrics,
                        rand_K,
                    ),
                }
            else:
                best_params, best_model = self._optimize_model(
                    model_name,
                    self.X_train,
                    self.y_train,
                    self.X_test,
                    self.y_test,
                    param_ranges.get(model_name, None) if param_ranges else None,
                    opt_metric,
                    n_trials,
                    cv_folds,
                )
                results[model_name] = {
                    "params": best_params,
                    "model": best_model,
                    "metrics": self._evaluate_model(
                        best_model, self.X_test, self.y_test, metrics
                    ),
                }
            print(f"\n\n-------------TRAINING {model_name} DONE------------------\n\n")

        # Store results as class variable
        self.best_results = results
        return results

    def train_fit(
        self,
        models: List[str] = ["RF"],
        test_size: float = 0.2,
        param_ranges: Optional[Dict] = None,
        save_path: Optional[str] = None,
        **train_kwargs,
    ) -> str:
        """
        Train models, fit them, and report results in a table-friendly format.

        Args:
            models: List of models to try
            test_size: Train/test split ratio
            param_ranges: Parameter ranges for optimization
            save_path: Optional path to save models
            **train_kwargs: Additional arguments for train()

        Returns:
            String containing results in a table-friendly format
        """
        # Train all models
        self.train(models, test_size, param_ranges, **train_kwargs)

        # Format results for table
        results_table = []
        header = ["Model", "Accuracy", "F1 (macro)", "F1 (weighted)"]
        results_table.append(" | ".join(header))
        results_table.append("|".join(["-" * len(h) for h in header]))

        # Add results for each model
        for model_name, result in self.best_results.items():
            if model_name == "Random":
                metrics = result["metrics"]
            else:
                metrics = result["metrics"]

                # Save model if path provided
                if save_path:
                    save_dir = os.path.join(
                        save_path, datetime.now().strftime("%Y%m%d")
                    )
                    os.makedirs(save_dir, exist_ok=True)
                    self._save_model(result["model"], model_name, save_dir)

            # Format metrics
            row = [
                model_name,
                f"{metrics['accuracy']:.3f}",
                f"{metrics['f1_macro']:.3f}",
                f"{metrics['f1_weighted']:.3f}",
            ]
            results_table.append(" | ".join(row))

        # Add dataset info
        info_lines = [
            f"\nDataset Info:",
            f"Total samples: {len(self.feature_df)}",
            f"Train samples: {len(self.X_train)}",
            f"Test samples: {len(self.X_test)}",
            f"Number of features: {self.X_train.shape[1]}",
            f"Class distribution:",
        ]

        # Add class distribution
        for class_name, class_idx in self.class_map.items():
            train_count = sum(self.y_train == class_idx)
            test_count = sum(self.y_test == class_idx)
            info_lines.append(
                f"  {class_name}: {train_count} (train), {test_count} (test)"
            )

        # Combine all output
        full_output = "\n".join(results_table + [""] + info_lines)

        return full_output

    @staticmethod
    def _save_model(model, name: str, path: str):
        """Save a trained model"""
        if isinstance(model, RandomForestClassifier):
            import joblib

            joblib.dump(model, os.path.join(path, f"{name}.joblib"))
        elif isinstance(model, xgb.XGBClassifier):
            model.save_model(os.path.join(path, f"{name}.json"))

    def _train_random_baseline(self, X_train, y_train, X_test, y_test, metrics, K=1000):
        random_classifier = RandomBaseline()
        random_classifier.fit(X_train, y_train)

        results = {}

        # Lists to store the performance metrics
        accuracies = []
        f1_macros = []
        f1_globals = []

        # Repeat the random classifier K times
        for _ in range(K):
            y_pred = random_classifier.predict(X_test)

            for metric in metrics:
                if metric == "accuracy":
                    acc = accuracy_score(y_test, y_pred)
                elif metric == "f1_macro":
                    f1_macro = f1_score(y_test, y_pred, average="macro")
                elif metric == "f1_weighted":
                    f1_weighted = f1_score(y_test, y_pred, average="weighted")
                else:
                    warnings.warn(f"Unsupported metric: {metric}. Skipping.")

            # Append the metrics to the lists
            accuracies.append(acc)
            f1_macros.append(f1_macro)
            f1_globals.append(f1_weighted)

        mean_accuracy = np.mean(accuracies)
        std_accuracy = np.std(accuracies)
        mean_f1_macro = np.mean(f1_macros)
        std_f1_macro = np.std(f1_macros)
        mean_f1_global = np.mean(f1_globals)
        std_f1_global = np.std(f1_globals)

        results = {
            "accuracy": mean_accuracy,
            "accuracy_stdev": std_accuracy,
            "f1_macro": mean_f1_macro,
            "f1_macro_stdev": std_f1_macro,
            "f1_weighted": mean_f1_global,
            "f1_weighted_stdev": std_f1_global,
        }
        return results

    def _optimize_model(
        self,
        model_type: str,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        param_ranges: Optional[Dict] = None,
        opt_metric: str = "f1_weighted",
        n_trials: int = 100,
        cv_folds: Optional[int] = None,
    ) -> Tuple[Dict, Union[RandomForestClassifier, xgb.XGBClassifier]]:
        """
        Optimize hyperparameters using Optuna with optional cross-validation.

        Args:
            model_type: Type of model ('RF' or 'XGB')
            X_train: Training features
            y_train: Training labels
            X_test: Test features
            y_test: Test labels
            param_ranges: Optional dictionary of parameter ranges
            opt_metric: Metric to optimize for
            n_trials: Number of optimization trials
            cv_folds: Optional number of CV folds (None, 5, or 10)

        Returns:
            Tuple of (best parameters, best model)
        """
        if model_type not in ["RF", "XGB"]:
            raise ValueError("model_type must be 'RF' or 'XGB'")

        # Default parameter ranges if none provided
        if param_ranges is None:
            if model_type == "RF":
                param_ranges = {
                    "n_estimators": (300, 1800),
                    "max_depth": (2, 20),
                    "min_samples_split": (2, 20),
                    "min_samples_leaf": (1, 20),
                }
            else:  # XGB
                param_ranges = {
                    "max_depth": (3, 8),
                    "learning_rate": (0.01, 0.1),
                    "n_estimators": (100, 500),
                    "subsample": (0.5, 0.8),
                    "colsample_bytree": (0.1, 0.5),
                    "min_child_weight": (3, 10),
                }

        def objective_rf(trial):
            params = {
                "n_estimators": trial.suggest_int(
                    "n_estimators",
                    param_ranges["n_estimators"][0],
                    param_ranges["n_estimators"][1],
                    log=True,
                ),
                "max_depth": trial.suggest_int(
                    "max_depth",
                    param_ranges["max_depth"][0],
                    param_ranges["max_depth"][1],
                    log=True,
                ),
                "min_samples_split": trial.suggest_int(
                    "min_samples_split",
                    param_ranges["min_samples_split"][0],
                    param_ranges["min_samples_split"][1],
                ),
                "min_samples_leaf": trial.suggest_int(
                    "min_samples_leaf",
                    param_ranges["min_samples_leaf"][0],
                    param_ranges["min_samples_leaf"][1],
                ),
                "random_state": self.seed,
                "class_weight": "balanced",
            }

            model = RandomForestClassifier(**params)

            if cv_folds:
                # Use cross-validation score
                cv_scores = cross_val_score(
                    model,
                    X_train,
                    y_train,
                    cv=cv_folds,
                    scoring=opt_metric,
                    n_jobs=1,  # Avoid nested parallelism
                )
                score = np.mean(cv_scores)
            else:
                # Use single train-test split
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                score = self._get_metric_score(y_test, y_pred, opt_metric)

            return score

        def objective_xgb(trial):
            # Calculate class weights
            classes = np.unique(y_train)
            class_weights = len(y_train) / (len(classes) * np.bincount(y_train))
            weight_dict = dict(zip(classes, class_weights))
            sample_weights = np.array([weight_dict[i] for i in y_train])

            params = {
                "max_depth": trial.suggest_int(
                    "max_depth",
                    param_ranges["max_depth"][0],
                    param_ranges["max_depth"][1],
                ),
                "learning_rate": trial.suggest_float(
                    "learning_rate",
                    param_ranges["learning_rate"][0],
                    param_ranges["learning_rate"][1],
                    log=True,
                ),
                "n_estimators": trial.suggest_int(
                    "n_estimators",
                    param_ranges["n_estimators"][0],
                    param_ranges["n_estimators"][1],
                ),
                "subsample": trial.suggest_float(
                    "subsample",
                    param_ranges["subsample"][0],
                    param_ranges["subsample"][1],
                ),
                "colsample_bytree": trial.suggest_float(
                    "colsample_bytree",
                    param_ranges["colsample_bytree"][0],
                    param_ranges["colsample_bytree"][1],
                ),
                "min_child_weight": trial.suggest_int(
                    "min_child_weight",
                    param_ranges["min_child_weight"][0],
                    param_ranges["min_child_weight"][1],
                ),
                "random_state": self.seed,
                "tree_method": "hist",
            }

            model = xgb.XGBClassifier(**params)

            if cv_folds:
                # Use cross-validation score with sample weights
                cv_scores = cross_val_score(
                    model,
                    X_train,
                    y_train,
                    cv=cv_folds,
                    scoring=opt_metric,
                    fit_params={"sample_weight": sample_weights},
                    n_jobs=1,  # Avoid nested parallelism
                )
                score = np.mean(cv_scores)
            else:
                # Use single train-test split
                model.fit(X_train, y_train, sample_weight=sample_weights)
                y_pred = model.predict(X_test)
                score = self._get_metric_score(y_test, y_pred, opt_metric)

            return score

        def _get_metric_score(y_true, y_pred, metric):
            """Helper function to calculate metric scores"""
            if metric == "f1_weighted":
                return f1_score(y_true, y_pred, average="weighted")
            elif metric == "f1_macro":
                return f1_score(y_true, y_pred, average="macro")
            else:  # accuracy
                return accuracy_score(y_true, y_pred)

        # Create and run optimization study
        study = optuna.create_study(direction="maximize")
        objective = objective_rf if model_type == "RF" else objective_xgb
        study.optimize(
            objective, n_trials=n_trials, n_jobs=n_trials, show_progress_bar=True
        )

        # Get best parameters
        best_params = study.best_params

        # Add fixed parameters
        if model_type == "RF":
            best_params.update({"random_state": self.seed, "class_weight": "balanced"})
        else:  # XGB
            best_params.update({"random_state": self.seed, "tree_method": "hist"})

        # Train final model with best parameters
        best_model = self._get_model_instance(model_type, best_params)

        if model_type == "RF":
            if cv_folds:
                cv_scores = cross_val_score(
                    best_model,
                    X_train,
                    y_train,
                    cv=cv_folds,
                    scoring=opt_metric,
                    n_jobs=-1,  # Can use parallel now
                )
                print(
                    f"{model_type} CV Scores ({cv_folds}-fold): {cv_scores.mean():.3f} ± {cv_scores.std():.3f}"
                )

            best_model.fit(X_train, y_train)

        else:  # XGB
            classes = np.unique(y_train)
            class_weights = len(y_train) / (len(classes) * np.bincount(y_train))
            weight_dict = dict(zip(classes, class_weights))
            sample_weights = np.array([weight_dict[i] for i in y_train])

            if cv_folds:
                cv_scores = cross_val_score(
                    best_model,
                    X_train,
                    y_train,
                    cv=cv_folds,
                    scoring=opt_metric,
                    fit_params={"sample_weight": sample_weights},
                    n_jobs=-1,  # Can use parallel now
                )
                print(
                    f"{model_type} CV Scores ({cv_folds}-fold): {cv_scores.mean():.3f} ± {cv_scores.std():.3f}"
                )

            best_model.fit(X_train, y_train, sample_weight=sample_weights)

        return best_params, best_model

    def _get_model_instance(
        self, model_type: str, params: Dict
    ) -> Union[RandomForestClassifier, xgb.XGBClassifier]:
        """
        Get a model instance with specified parameters.

        Args:
            model_type: Type of model ('RF' or 'XGB')
            params: Dictionary of model parameters

        Returns:
            Initialized model instance
        """
        if model_type == "RF":
            # Ensure class_weight is set for RF
            if "class_weight" not in params:
                params["class_weight"] = "balanced"
            return RandomForestClassifier(**params)

        elif model_type == "XGB":
            # Ensure tree_method is set for efficiency
            if "tree_method" not in params:
                params["tree_method"] = "hist"
            return xgb.XGBClassifier(**params)

        else:
            raise ValueError(
                f"Unsupported model type: {model_type}. Use 'RF' or 'XGB'."
            )

    def _evaluate_model(
        self,
        model: Union[RandomForestClassifier, xgb.XGBClassifier],
        X_test: pd.DataFrame,
        y_test: pd.Series,
        metrics: List[str] = ["accuracy", "f1_macro", "f1_weighted"],
    ) -> Dict[str, float]:
        """
        Evaluate a model using specified metrics.

        Args:
            model: Trained model instance
            X_test: Test features
            y_test: True labels
            metrics: List of metrics to compute

        Returns:
            Dictionary of metric names and values
        """
        y_pred = model.predict(X_test)
        results = {}

        # Calculate requested metrics
        for metric in metrics:
            if metric == "accuracy":
                results[metric] = accuracy_score(y_test, y_pred)
            elif metric == "f1_macro":
                results[metric] = f1_score(y_test, y_pred, average="macro")
            elif metric == "f1_weighted":
                results[metric] = f1_score(y_test, y_pred, average="weighted")
            else:
                warnings.warn(f"Unsupported metric: {metric}. Skipping.")

        # Always include confusion matrix and classification report
        results["confusion_matrix"] = confusion_matrix(y_test, y_pred).tolist()
        results["classification_report"] = classification_report(y_test, y_pred)

        # Include feature importances if available
        if hasattr(model, "feature_importances_"):
            results["feature_importances"] = dict(
                zip(X_test.columns, model.feature_importances_)
            )

        return results

    def run_inference(
        self,
        feature_path: str,
        model_path: str,
        batch_size: int = 1024,
        n_jobs: int = -1,
    ) -> gpd.GeoDataFrame:
        """
        Run batched parallel inference on pre-featurized data using a trained model.

        Args:
            feature_path: Path to featurized parquet file
            model_path: Path to saved model (.joblib for RF or .json for XGB)
            batch_size: Size of batches for prediction
            n_jobs: Number of parallel jobs (-1 for all cores)

        Returns:
            GeoDataFrame with predictions added in 'label' column
        """
        # Load featurized data
        gdf = gpd.read_parquet(feature_path)

        # Load model based on file extension
        if model_path.endswith(".joblib"):
            import joblib

            model = joblib.load(model_path)
            model_type = "RF"
            predict_func = _predict_batch_rf
        elif model_path.endswith(".json"):
            model = xgb.XGBClassifier()
            model.load_model(model_path)
            model_type = "XGB"
            predict_func = _predict_batch_xgb
        else:
            raise ValueError("Model path must end in .joblib (RF) or .json (XGB)")

        # Get feature names from model
        if model_type == "RF":
            required_features = model.feature_names_in_
        else:  # XGB
            required_features = model.get_booster().feature_names

        # Verify all required features are present
        missing_features = set(required_features) - set(gdf.columns)
        if missing_features:
            raise ValueError(f"Missing required features: {missing_features}")

        # Extract features
        X = gdf[required_features]

        def predict_batch(batch_df):
            """Helper function to predict on a batch"""
            return model.predict(batch_df)

        # Create batches
        n_batches = len(X) // batch_size + (1 if len(X) % batch_size != 0 else 0)
        batches = np.array_split(X, n_batches)

        # Run parallel prediction with progress bar
        from concurrent.futures import ProcessPoolExecutor
        from tqdm.auto import tqdm

        n_jobs = n_jobs if n_jobs > 0 else os.cpu_count()

        predictions = []
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            # Submit all batches
            futures = [executor.submit(predict_func, batch, model) for batch in batches]

            # Process results with progress bar
            for future in tqdm(
                futures, total=len(futures), desc="Running inference", unit="batch"
            ):
                batch_predictions = future.result()
                predictions.extend(batch_predictions)

        # Create reverse mapping for class labels
        reverse_class_map = {v: k for k, v in self.class_map.items()}

        # Update GeoDataFrame with predictions
        gdf["label"] = [reverse_class_map[pred] for pred in predictions]

        print(f"\nPredictions summary:")
        print(gdf["label"].value_counts())

        return gdf

    def save_predictions(
        self, gdf: gpd.GeoDataFrame, output_path: str, save_all: bool = False
    ) -> None:
        """
        Save predictions to file.

        Args:
            gdf: GeoDataFrame with predictions
            output_path: Path to save results
            save_all: If True, saves all columns; if False, saves only geometry and label
        """
        if not save_all:
            gdf = gdf[["geometry", "label"]]

        if output_path.endswith(".parquet"):
            gdf.to_parquet(output_path)
        elif output_path.endswith(".geojson"):
            gdf.to_file(output_path, driver="GeoJSON")
        else:
            raise ValueError("Output path must end in .parquet, .geojson")
