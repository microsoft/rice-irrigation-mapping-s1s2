# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import pandas as pd
import os
import pandas as pd
import numpy as np
import os
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
import optuna
import torch
from typing import List
import pandas as pd
import numpy as np
from ricemapper.utils.utils import linear_to_db
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.decomposition import PCA
from ricemapper.utils.utils import RandomBaseline
import lightgbm as lgb
import joblib
from sklearn.model_selection import StratifiedKFold
from scipy import stats
from sklearn.metrics import precision_score, recall_score
import json
import argparse
from dotenv import load_dotenv

load_dotenv()

DATE_IN_WEEKS = True
BATCH_SIZE = 256
SEED = 42


def serialize_argument(arg):
    """Convert pandas objects and other non-serializable types to serializable formats."""
    if isinstance(arg, (pd.Index, pd.Series)):
        return arg.tolist()
    if isinstance(arg, np.ndarray):
        return arg.tolist()
    return arg


def sample_columns_with_aggregation(
    df: pd.DataFrame,
    variables: List[str],
    date_col: str,
    frequency: int,
    sum_variables: List[str] = None,
    convert_to_db: bool = True,
) -> pd.DataFrame:
    """
    Sample the specified columns of a DataFrame at a given frequency using the date column.
    Some variables can be summed over the period instead of sampled.
    Ensures all output lists have exactly the same length.

    Args:
        df (pd.DataFrame): The input DataFrame.
        variables (List[str]): List of all columns to process.
        date_col (str): The name of the date column.
        frequency (int): The frequency for resampling (keep every nth value in lists)
        sum_variables (List[str], optional): List of variables that should be summed over the period
            rather than sampled. These variables should also be in the variables list.
        convert_to_db (bool, optional): Whether to convert radar variables to dB. Defaults to True.

    Returns:
        pd.DataFrame: A new DataFrame with the processed data in lists.
    """

    variables = [var for var in variables if var in df.columns]

    def resample_list(
        lst: List[float], freq: int, max_samples: int = None
    ) -> List[float]:
        """
        Take every nth value from the list, ensuring the output doesn't exceed max_samples.
        """
        sampled = lst[::freq]
        if max_samples is not None:
            sampled = sampled[:max_samples]
        return sampled

    def sum_by_groups(lst: List[float], freq: int, max_samples: int) -> List[float]:
        """
        Sum values in groups of size freq, ensuring the output matches max_samples.
        """
        arr = np.array(lst)
        result = []

        for i in range(max_samples):
            start_idx = i * freq
            end_idx = min(start_idx + freq, len(arr))

            if start_idx < len(arr):
                group_sum = arr[start_idx:end_idx].sum()
                result.append(group_sum)
            else:
                break

        return result

    # Create a copy of the DataFrame to avoid modifying the original
    processed_df = df.copy()
    sum_variables = sum_variables or []

    def calculate_max_samples(row):
        """Calculate the maximum number of samples possible for this row"""
        return min(
            len(row[date_col]) // frequency,
            *(len(row[var]) // frequency for var in variables),
        )

    # First calculate the number of samples we should have for each row
    processed_df["_max_samples"] = processed_df.apply(calculate_max_samples, axis=1)

    # Process each variable ensuring consistent lengths
    # Start with the date column
    processed_df[date_col] = processed_df.apply(
        lambda row: resample_list(row[date_col], frequency, row["_max_samples"]), axis=1
    )

    # Process other variables
    for var in variables:
        if var in sum_variables:
            processed_df[var] = processed_df.apply(
                lambda row: sum_by_groups(row[var], frequency, row["_max_samples"]),
                axis=1,
            )
        else:
            processed_df[var] = processed_df.apply(
                lambda row: resample_list(row[var], frequency, row["_max_samples"]),
                axis=1,
            )
            # Convert radar variables to dB if needed
            if (
                convert_to_db
                and "ratio" not in var
                and ("DESCENDING" in var or "ASCENDING" in var)
            ):
                processed_df[var] = processed_df[var].apply(
                    lambda x: [linear_to_db(v) for v in x]
                )

    # Validate that all lists have the same length
    def validate_lengths(row):
        lengths = [len(row[date_col])] + [len(row[var]) for var in variables]
        assert len(set(lengths)) == 1, f"Inconsistent lengths found: {lengths}"
        return True

    try:
        processed_df.apply(validate_lengths, axis=1)
    except AssertionError as e:
        raise ValueError(f"Validation failed: {str(e)}")

    # Remove the temporary column used for max_samples
    processed_df = processed_df.drop("_max_samples", axis=1)

    return processed_df


def apply_pca_to_column(df, column_name, n_components):
    """
    Apply PCA to the values in a column that contains lists and reduce it to the first N components.

    Args:
    df (pd.DataFrame): The input DataFrame.
    column_name (str): The name of the column containing lists.
    n_components (int): The number of components to reduce to.

    Returns:
    pd.DataFrame: A new DataFrame with the reduced components.
    """
    # Extract the column with lists
    data = df[column_name].tolist()

    # Convert the list of lists to a 2D array
    data_matrix = pd.DataFrame(data).values

    # Apply PCA
    pca = PCA(n_components=n_components)
    reduced_data = pca.fit_transform(data_matrix)
    pca_names = [f"{column_name}_PC{i+1}" for i in range(n_components)]

    # Create a new DataFrame with the reduced components
    reduced_df = pd.DataFrame(reduced_data, columns=pca_names)

    # Concatenate the reduced components with the original DataFrame
    result_df = pd.concat([df.drop(columns=[column_name]), reduced_df], axis=1)

    return result_df


def flatten_features(df, column_name):
    # Extract the 'presto_features' column
    features = df[column_name].apply(pd.Series)

    # Rename columns to have a prefix 'feature_'
    features.columns = [f"{column_name}_{i}" for i in range(features.shape[1])]

    # Concatenate the new columns back to the original DataFrame
    df = pd.concat([df.drop(columns=[column_name]), features], axis=1)

    return df


def flatten_2d_features(df, column_name, era5=False, prefix=""):
    # Check if the column exists in the DataFrame
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found in the DataFrame.")

    # Get the shape of the 2D array (assuming all rows have the same shape)
    sample_array = df[column_name].iloc[0]
    if not isinstance(sample_array, np.ndarray) or sample_array.ndim != 2:
        raise ValueError(f"Column '{column_name}' must contain 2D numpy arrays.")

    rows, cols = sample_array.shape  # weeks, features

    if era5:
        num_temp_features = int((rows - 1) * 2 / 3)
        num_era5_features = rows - num_temp_features - 1

        print(num_temp_features, num_era5_features)
    else:
        num_temp_features = int((rows - 1))
        print(num_temp_features)

    # Create new column names

    time_features = [
        f"{prefix}_time_{i}_{j}" for i in range(num_temp_features) for j in range(cols)
    ]
    if era5:
        era5_features = [
            f"{prefix}_era5_{i}_{j}"
            for i in range(num_temp_features, num_temp_features + num_era5_features)
            for j in range(cols)
        ]
        location_features = [
            f"{prefix}_location_{i}_{j}"
            for i in range(2 * num_temp_features, rows)
            for j in range(cols)
        ]
        # Create new column names
        new_columns = time_features + location_features + era5_features
    else:
        location_features = [
            f"{prefix}_location_{i}_{j}"
            for i in range(rows - 1, rows)
            for j in range(cols)
        ]

        # Create new column names
        new_columns = time_features + location_features

    # Flatten the 2D arrays and create a new DataFrame
    flattened_data = np.vstack(df[column_name].apply(lambda x: x.flatten()))

    # Create a new DataFrame with flattened columns
    flattened_df = pd.DataFrame(flattened_data, columns=new_columns, index=df.index)

    # Concatenate the original DataFrame (excluding the original column) with the flattened DataFrame
    result_df = pd.concat([df.drop(columns=[column_name]), flattened_df], axis=1)

    return result_df


def apply_pca_to_2d_column(df, column_name, n_components):
    """
    Apply PCA to a 2D variable stored in a DataFrame column.

    Args:
    df (pd.DataFrame): The input DataFrame.
    column_name (str): The name of the column containing 2D data.
    n_components (int): The number of principal components to keep.

    Returns:
    pd.DataFrame: A new DataFrame with the PCA results added as new columns.
    """
    # Extract the 2D data from the DataFrame column
    data_2d = np.array(df[column_name].tolist())

    # Reshape the data if necessary
    # Assuming each element in the column is a 2D array of shape (m, n)
    # We reshape to (-1, n)
    reshaped_data = data_2d.reshape(-1, data_2d.shape[2])

    # Initialize and fit PCA
    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(reshaped_data)
    print(pca.explained_variance_ratio_)

    # Reshape the result back to (num_samples, m, n_components)
    pca_reshaped = pca_result.reshape(data_2d.shape[0], data_2d.shape[1], n_components)

    # Create a new DataFrame with the PCA results
    result_df = df.copy()
    for i in range(n_components):
        result_df[f"{column_name}_PC{i+1}"] = [pc[:, i] for pc in pca_reshaped]

    return result_df


def apply_pca_to_2d_column_row_wise(df, column_name, n_components):
    """
    Apply PCA to each 2D array in a DataFrame column independently.

    Args:
    df (pd.DataFrame): The input DataFrame.
    column_name (str): The name of the column containing 2D data.
    n_components (int): The number of principal components to keep.

    Returns:
    pd.DataFrame: A new DataFrame with the PCA results added as new columns.
    """

    def pca_transform(array_2d):
        pca = PCA(n_components=n_components)
        y = pca.fit_transform(array_2d)
        return y

    # Apply PCA to each 2D array in the column
    pca_results = df[column_name].apply(pca_transform)

    # Create a new DataFrame with the PCA results
    result_df = df.copy()
    for i in range(n_components):
        result_df[f"{column_name}_PC{i+1}"] = pca_results.apply(lambda x: x[:, i])

    return result_df


def remove_short_timeseries(df, columns_to_check, min_length=12):
    # Create a mask for rows to keep
    valid_rows = df[columns_to_check].apply(
        lambda row: sum(~pd.isna(row)) >= min_length, axis=1
    )

    return df[valid_rows]


def remove_invalid_rows(df, columns_to_check=None):
    def has_invalid_values(x):
        if isinstance(x, list):
            return any(pd.isna(val) or np.isinf(val) for val in x)
        elif isinstance(x, (np.ndarray, pd.Series)):
            return pd.isna(x).any() or np.isinf(x).any()
        elif isinstance(x, (int, float)):
            return pd.isna(x) or np.isinf(x)
        else:  # strings or other types
            return pd.isna(x)

    # If no columns specified, check all columns
    columns_to_check = columns_to_check or df.columns

    # Create a mask for rows to keep
    valid_rows = ~df[columns_to_check].apply(
        lambda row: any(has_invalid_values(val) for val in row), axis=1
    )

    return df[valid_rows]


def create_ensemble_model(rf_models, lgb_models):
    """
    Create an ensemble model that combines RF and LGB models using majority voting.

    Args:
        rf_models: List of trained RandomForest models
        lgb_models: List of trained LightGBM models

    Returns:
        A callable ensemble model that predicts using majority voting
    """

    class EnsembleModel:
        def __init__(self, rf_models, lgb_models):
            self.rf_models = rf_models
            self.lgb_models = lgb_models
            self.all_models = rf_models + lgb_models

        def predict(self, X):
            """Predict using majority voting across all models"""
            predictions = []

            # Get predictions from all RF models
            for model in self.rf_models:
                pred = model.predict(X)
                predictions.append(pred)

            # Get predictions from all LGB models
            for model in self.lgb_models:
                pred = model.predict(X)
                predictions.append(pred)

            # Convert to numpy array for easier manipulation
            predictions = np.array(predictions)

            # Use majority voting (mode) for each sample
            ensemble_predictions = []
            for i in range(predictions.shape[1]):
                sample_predictions = predictions[:, i]
                # Use scipy.stats.mode to get the most frequent prediction
                mode_result = stats.mode(sample_predictions, keepdims=True)
                ensemble_predictions.append(mode_result.mode[0])

            return np.array(ensemble_predictions)

        def predict_proba(self, X):
            """Predict probabilities by averaging across all models"""
            probabilities = []

            # Get probabilities from all RF models
            for model in self.rf_models:
                proba = model.predict_proba(X)
                probabilities.append(proba)

            # Get probabilities from all LGB models
            for model in self.lgb_models:
                proba = model.predict_proba(X)
                probabilities.append(proba)

            # Average probabilities across all models
            avg_probabilities = np.mean(probabilities, axis=0)
            return avg_probabilities

    return EnsembleModel(rf_models, lgb_models)


def main(
    SPLIT=0.2,
    SPLIT_VAL=0.0,
    TASK=None,
    MODELS=["RF", "GB", "Random"],
    GPU=1,
    k=3,
    output_dir=".",
    train_ft_path=None,
    num_models=3,
    cv=False,
):

    if TASK not in ["DSR", "AWD", "ALL"]:
        raise ValueError("TASK must be one of ['DSR', 'AWD']")

    device = torch.device(f"cuda:{GPU}" if torch.cuda.is_available() else "cpu")

    df = pd.read_parquet(train_ft_path)

    assert "label" in df.columns, "label column is not in df.columns"

    if TASK == "DSR":
        label_map = {"AWD": 0, "PTR": 0, "DSR": 1}
    elif TASK == "AWD":
        label_map = {"AWD": 1, "PTR": 0, "DSR": 0}
    elif TASK == "ALL":
        label_map = {"AWD": 0, "PTR": 1, "DSR": 2}
    else:
        print("Unknown task")

    df["label"] = df["label"].map(label_map)
    print(df["label"].value_counts())

    X = df.copy()
    y = X["label"]

    print("Label mapping:", label_map, " ; unique labels:", X["label"].unique())
    print(X["label"].value_counts())
    drop_features = [
        "VV_ASCENDING_mean_spline",
        "VH_ASCENDING_mean_spline",
        "ASCENDING_spline_ratio",
        "dates_spline",
        "VV_DESCENDING_mean_spline",
        "VH_DESCENDING_mean_spline",
        "DESCENDING_spline_ratio",
        "SN",
        "Class",
        "temperature_2m",
        "total_precipitation_sum",
        "dates_era5",
        "geometry",
        "label",
        "Latitude",
        "Longitude",
    ]

    # check if ts_features are present in X.columns
    ts_features = [f for f in drop_features if f in X.columns]
    X = X.drop(columns=ts_features, axis=1)

    if SPLIT_VAL > 0:
        # First split off the test set
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=SPLIT, random_state=42, stratify=y
        )
        # Then split remaining data into train and validation
        val_size = SPLIT_VAL / (
            1 - SPLIT
        )  # Adjust validation size relative to remaining data
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size, random_state=42, stratify=y_temp
        )
        print(
            f"Training set size: {X_train.shape}, Validation set size: {X_val.shape}, Test set size: {X_test.shape}"
        )
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=SPLIT, random_state=42, stratify=y
        )
        X_val, y_val = None, None
        print(f"Training set size: {X_train.shape}, Test set size: {X_test.shape}")

    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

    assert "label" not in X_train.columns, "label is in X_train.columns"
    assert "Class" not in X_train.columns, "Class is in X_train.columns"
    assert "Planting_date" not in X_train.columns, "Planting_date is in X_train.columns"
    assert "district" not in X_train.columns, "district is in X_train.columns"
    assert "stratify_key" not in X_train.columns, "stratify_key is in X_train.columns"
    assert "geometry" not in X_train.columns, "geometry is in X_train.columns"

    results = {}

    results["num_features"] = len(X.columns)
    results["test_indices"] = serialize_argument(X_test.index)
    print(f'Number of features: {results["num_features"]}')

    print(f"\n\n--------------------TASK == {TASK}-----------------------------\n")

    # Dictionary to store all models and their validation scores
    all_models = {}
    model_scores = {}

    if "RF" in MODELS:
        print(f"\n\n--------------------RF-----------------------------\n")

        # Store all RF models and their validation scores
        rf_models = []
        rf_scores = []

        def objective(trial):
            # Define the hyperparameters to optimize
            n_estimators = trial.suggest_int("n_estimators", 300, 1800, log=True)
            max_depth = trial.suggest_int("max_depth", 2, 20, log=True)
            min_samples_split = trial.suggest_int("min_samples_split", 2, 20)
            min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 20)

            # Create and train the model
            model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                random_state=SEED,
                class_weight="balanced",
            ).fit(X_train, y_train)

            # Always use validation set for optimization if it exists
            if X_val is not None:
                y_pred = model.predict(X_val)
                score = f1_score(y_val, y_pred, average="weighted")
            else:
                # Only use test set if no validation set exists
                y_pred = model.predict(X_test)
                score = f1_score(y_test, y_pred, average="weighted")

            # Train the model on the full training set for later use
            model.fit(X_train, y_train)

            # Store the model and its score
            rf_models.append(model)
            rf_scores.append(score)

            return score

        def objective_cv(trial):
            # Define the hyperparameters to optimize
            n_estimators = trial.suggest_int("n_estimators", 300, 1800, log=True)
            max_depth = trial.suggest_int("max_depth", 2, 20, log=True)
            min_samples_split = trial.suggest_int("min_samples_split", 2, 20)
            min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 20)

            # Add regularization parameters
            max_features = trial.suggest_categorical(
                "max_features", ["sqrt", "log2", None]
            )
            bootstrap = trial.suggest_categorical("bootstrap", [True, False])
            oob_score = bootstrap  # Only use OOB score if bootstrap is True

            # Use stratified k-fold instead of regular cross_val_score
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
            cv_scores = []

            for train_idx, val_idx in skf.split(X_train, y_train):
                X_fold_train = X_train.iloc[train_idx]
                X_fold_val = X_train.iloc[val_idx]
                y_fold_train = (
                    y_train.iloc[train_idx]
                    if hasattr(y_train, "iloc")
                    else y_train[train_idx]
                )
                y_fold_val = (
                    y_train.iloc[val_idx]
                    if hasattr(y_train, "iloc")
                    else y_train[val_idx]
                )

                model = RandomForestClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    min_samples_split=min_samples_split,
                    min_samples_leaf=min_samples_leaf,
                    max_features=max_features,
                    bootstrap=bootstrap,
                    oob_score=oob_score,
                    random_state=SEED,
                    class_weight="balanced",
                )

                model.fit(X_fold_train, y_fold_train)
                y_pred_val = model.predict(X_fold_val)
                fold_score = f1_score(y_fold_val, y_pred_val, average="weighted")
                cv_scores.append(fold_score)

            val_score = np.mean(cv_scores)

            # Train the model on the full training set for later use
            model.fit(X_train, y_train)

            # Store the model and its score
            rf_models.append(model)
            rf_scores.append(val_score)

            return val_score

        # Create a study object and optimize the objective function
        study = optuna.create_study(direction="maximize")
        if cv:
            study.optimize(
                objective_cv, n_trials=200, n_jobs=50, show_progress_bar=True
            )
        else:
            study.optimize(objective, n_trials=200, n_jobs=100, show_progress_bar=True)

        # Get the best hyperparameters
        best_params = study.best_params
        best_params

        # Sort models by validation score and keep the top num_models
        sorted_indices = np.argsort(rf_scores)[::-1]
        top_rf_models = [rf_models[i] for i in sorted_indices[:num_models]]
        top_rf_scores = [rf_scores[i] for i in sorted_indices[:num_models]]

        # Store the top models and their scores
        all_models["RF"] = top_rf_models
        model_scores["RF"] = top_rf_scores

        # Calculate test scores for all top models
        rf_test_scores = []
        rf_test_accuracies = []
        rf_test_f1_macros = []

        for model in top_rf_models:
            y_pred = model.predict(X_test)
            rf_test_scores.append(f1_score(y_test, y_pred, average="weighted"))
            rf_test_accuracies.append(accuracy_score(y_test, y_pred))
            rf_test_f1_macros.append(f1_score(y_test, y_pred, average="macro"))

        # Calculate averages and standard deviations
        avg_test_score = np.mean(rf_test_scores)
        std_test_score = np.std(rf_test_scores)
        avg_test_accuracy = np.mean(rf_test_accuracies)
        std_test_accuracy = np.std(rf_test_accuracies)
        avg_test_f1_macro = np.mean(rf_test_f1_macros)
        std_test_f1_macro = np.std(rf_test_f1_macros)

        avg_val_score = np.mean(top_rf_scores)
        std_val_score = np.std(top_rf_scores)

        # Use the best model for final evaluation on test set
        best_rf_model = top_rf_models[0]
        y_pred = best_rf_model.predict(X_test)

        # Analyze feature importance
        feature_importance = pd.DataFrame(
            {
                "feature": X_train.columns,
                "importance": best_rf_model.feature_importances_,
            }
        ).sort_values("importance", ascending=False)

        # Save feature importance to results
        results["RF"] = {
            "best_model_test_accuracy": round(accuracy_score(y_test, y_pred) * 100, 2),
            "best_model_test_f1_weighted": round(
                f1_score(y_test, y_pred, average="weighted") * 100, 2
            ),
            "best_model_test_f1_macro": round(
                f1_score(y_test, y_pred, average="macro") * 100, 2
            ),
            "best_model_test_precision": round(
                precision_score(y_test, y_pred, average="weighted") * 100, 2
            ),
            "best_model_test_recall": round(
                recall_score(y_test, y_pred, average="weighted") * 100, 2
            ),
            "val_scores": [round(score * 100, 2) for score in top_rf_scores],
            "avg_val_score": round(avg_val_score * 100, 2),
            "std_val_score": round(std_val_score * 100, 2),
            "avg_test_score": round(avg_test_score * 100, 2),
            "std_test_score": round(std_test_score * 100, 2),
            "avg_test_accuracy": round(avg_test_accuracy * 100, 2),
            "std_test_accuracy": round(std_test_accuracy * 100, 2),
            "avg_test_f1_macro": round(avg_test_f1_macro * 100, 2),
            "std_test_f1_macro": round(std_test_f1_macro * 100, 2),
            "feature_importance": serialize_argument(
                feature_importance.head(20).to_dict("records")
            ),
        }

        print(classification_report(y_test, y_pred))
        print(
            f"Best model - OA, Global weighted F1 score, Macro F1 score (test set):  {accuracy_score(y_test, y_pred)*100:.2f}% ,  {f1_score(y_test, y_pred, average='weighted')*100:.2f}%, {f1_score(y_test, y_pred, average='macro')*100:.2f}%"
        )
        print(
            f"Best model - Precision, Recall (test set):  {precision_score(y_test, y_pred, average='weighted')*100:.2f}%, {recall_score(y_test, y_pred, average='weighted')*100:.2f}%"
        )
        print(
            f"Top {num_models} RF models validation scores: {[round(score*100, 2) for score in top_rf_scores]}"
        )
        print(
            f"RF models - Avg val score: {avg_val_score*100:.2f}% ± {std_val_score*100:.2f}%"
        )
        print(
            f"RF models - Avg test score: {avg_test_score*100:.2f}% ± {std_test_score*100:.2f}%"
        )
        print(
            f"RF models - Avg test accuracy: {avg_test_accuracy*100:.2f}% ± {std_test_accuracy*100:.2f}%"
        )
        print(
            f"RF models - Avg test macro F1: {avg_test_f1_macro*100:.2f}% ± {std_test_f1_macro*100:.2f}%"
        )
        print(f"RF models - Final test performance (across all top models):")
        print(
            f"  Accuracy: {avg_test_accuracy*100:.2f}% ± {std_test_accuracy*100:.2f}%"
        )
        print(f"  Weighted F1: {avg_test_score*100:.2f}% ± {std_test_score*100:.2f}%")
        print(
            f"  Macro F1: {avg_test_f1_macro*100:.2f}% ± {std_test_f1_macro*100:.2f}%"
        )

        # Print top 10 features
        print("\nTop 10 most important features:")
        print(feature_importance.head(10))

    if "GB" in MODELS:
        print("\n\n--------------------GB-----------------------------\n\n")

        # Calculate class weights if dealing with imbalanced classes
        classes = np.unique(y_train)
        class_weights = len(y_train) / (len(classes) * np.bincount(y_train))
        weight_dict = dict(zip(classes, class_weights))
        sample_weights = np.array([weight_dict[i] for i in y_train])

        # Store all GB models and their validation scores
        xgb_models = []
        xgb_scores = []

        def objective(trial):
            import gc  # Add at the top of the file if not already there

            params = {
                "objective": "multiclass",
                "num_class": len(classes),
                "boosting_type": "gbdt",
                "max_depth": trial.suggest_int("max_depth", 3, 8),
                "learning_rate": trial.suggest_float(
                    "learning_rate", 0.001, 0.01, log=True
                ),
                "n_estimators": trial.suggest_int("n_estimators", 100, 500),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "min_child_samples": trial.suggest_int("min_child_samples", 5, 30),
                "min_split_gain": trial.suggest_float("min_split_gain", 1e-8, 0.1),
                "num_leaves": trial.suggest_int("num_leaves", 20, 3000),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 0.5, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 0.5, log=True),
                "random_state": SEED,
                "verbose": -1,
                # Add these memory-saving parameters
                "max_bin": 63,  # Reduce number of bins used in histogram
                "bin_construct_sample_cnt": 100000,  # Reduce number of samples used to construct bins
                "is_provide_training_metric": False,  # Don't calculate metric on training data
                "two_round": True,  # Use two-round loading for memory optimization
                "force_row_wise": True,  # Use row-wise processing (can be slower but uses less memory)
            }

            model = lgb.LGBMClassifier(**params)
            model.fit(X_train, y_train, sample_weight=sample_weights)

            if X_val is not None:
                y_pred = model.predict(X_val)
                score = f1_score(y_val, y_pred, average="weighted")
            else:
                y_pred = model.predict(X_test)
                score = f1_score(y_test, y_pred, average="weighted")

            model.fit(
                X_train,
                y_train,
                sample_weight=sample_weights,
                # eval_metric='multi_logloss',
            )

            # Store the model and its score
            xgb_models.append(model)
            xgb_scores.append(score)
            return score

        def objective_cv(trial):
            params = {
                "objective": "multiclass",
                "num_class": len(classes),
                "boosting_type": trial.suggest_categorical(
                    "boosting_type", ["gbdt", "dart"]
                ),
                "max_depth": trial.suggest_int("max_depth", 3, 6),
                "learning_rate": trial.suggest_float(
                    "learning_rate", 0.01, 0.1, log=False
                ),
                "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "min_child_samples": trial.suggest_int("min_child_samples", 10, 50),
                "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 0.2),
                "num_leaves": trial.suggest_int("num_leaves", 2, 256),
                "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 2.0),
                "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 2.0),
                "random_state": SEED,
                "verbose": -1,
                # Add these memory-saving parameters
                "max_bin": 63,  # Reduce number of bins used in histogram
                "bin_construct_sample_cnt": 100000,  # Reduce number of samples used to construct bins
                "is_provide_training_metric": False,  # Don't calculate metric on training data
                "two_round": True,  # Use two-round loading for memory optimization
                "force_row_wise": True,  # Use row-wise processing (can be slower but uses less memory)
            }

            # Use StratifiedKFold instead of regular KFold
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

            cv_scores = []
            for train_idx, val_idx in skf.split(X_train, y_train):
                X_fold_train = X_train.iloc[train_idx]
                X_fold_val = X_train.iloc[val_idx]
                y_fold_train = (
                    y_train.iloc[train_idx]
                    if hasattr(y_train, "iloc")
                    else y_train[train_idx]
                )
                y_fold_val = (
                    y_train.iloc[val_idx]
                    if hasattr(y_train, "iloc")
                    else y_train[val_idx]
                )
                sample_weights_fold = sample_weights[train_idx]

                model = lgb.LGBMClassifier(**params)

                # Use early stopping to prevent overfitting
                eval_set = [(X_fold_val, y_fold_val)]

                # Check if boosting type is dart - early stopping not available in dart mode
                callbacks = []
                if params["boosting_type"] != "dart":
                    callbacks = [lgb.early_stopping(stopping_rounds=20)]

                model.fit(
                    X_fold_train,
                    y_fold_train,
                    sample_weight=sample_weights_fold,
                    eval_set=eval_set,
                    eval_metric="multi_logloss",
                    callbacks=callbacks,
                )

                y_pred_val = model.predict(X_fold_val)
                fold_score = f1_score(y_fold_val, y_pred_val, average="weighted")
                cv_scores.append(fold_score)

            val_score = np.mean(cv_scores)

            # Train the model on the full training set for later use
            model = lgb.LGBMClassifier(**params)

            # Split while maintaining DataFrame structure and stratification
            X_train_final, X_val_final, y_train_final, y_val_final = train_test_split(
                X_train, y_train, test_size=0.1, random_state=SEED, stratify=y_train
            )
            sample_weights_final = sample_weights[: len(X_train_final)]

            # Use early stopping on the final model
            eval_set = [(X_val_final, y_val_final)]

            callbacks = []
            if params["boosting_type"] != "dart":
                callbacks = [lgb.early_stopping(stopping_rounds=20)]

            model.fit(
                X_train_final,
                y_train_final,
                sample_weight=sample_weights_final,
                eval_set=eval_set,
                eval_metric="multi_logloss",
                callbacks=callbacks,
            )

            # Store the model and its score
            xgb_models.append(model)
            xgb_scores.append(val_score)

            return val_score

        # Create study and optimize
        study = optuna.create_study(direction="maximize")
        if cv:
            study.optimize(
                objective_cv, n_trials=200, n_jobs=10, show_progress_bar=True
            )
        else:
            study.optimize(objective, n_trials=200, n_jobs=10, show_progress_bar=True)

        # Get the best hyperparameters
        best_params = study.best_params

        # Sort models by validation score and keep the top num_models
        sorted_indices = np.argsort(xgb_scores)[::-1]
        top_xgb_models = [xgb_models[i] for i in sorted_indices[:num_models]]
        top_xgb_scores = [xgb_scores[i] for i in sorted_indices[:num_models]]

        # Store the top models and their scores
        all_models["GB"] = top_xgb_models
        model_scores["GB"] = top_xgb_scores

        # Calculate test scores for all top models
        xgb_test_scores = []
        xgb_test_accuracies = []
        xgb_test_f1_macros = []

        for model in top_xgb_models:
            y_pred = model.predict(X_test)
            xgb_test_scores.append(f1_score(y_test, y_pred, average="weighted"))
            xgb_test_accuracies.append(accuracy_score(y_test, y_pred))
            xgb_test_f1_macros.append(f1_score(y_test, y_pred, average="macro"))

        # Calculate averages and standard deviations
        avg_test_score = np.mean(xgb_test_scores)
        std_test_score = np.std(xgb_test_scores)
        avg_test_accuracy = np.mean(xgb_test_accuracies)
        std_test_accuracy = np.std(xgb_test_accuracies)
        avg_test_f1_macro = np.mean(xgb_test_f1_macros)
        std_test_f1_macro = np.std(xgb_test_f1_macros)

        avg_val_score = np.mean(top_xgb_scores)
        std_val_score = np.std(top_xgb_scores)

        # Use the best model for final evaluation on test set
        best_xgb_model = top_xgb_models[0]
        y_pred = best_xgb_model.predict(X_test)

        # Analyze feature importance
        feature_importance = pd.DataFrame(
            {
                "feature": X_train.columns,
                "importance": best_xgb_model.feature_importances_,
            }
        ).sort_values("importance", ascending=False)

        # Save feature importance to results
        results["GB"] = {
            "best_model_accuracy": round(accuracy_score(y_test, y_pred) * 100, 2),
            "best_model_f1_weighted": round(
                f1_score(y_test, y_pred, average="weighted") * 100, 2
            ),
            "best_model_f1_macro": round(
                f1_score(y_test, y_pred, average="macro") * 100, 2
            ),
            "best_model_precision": round(
                precision_score(y_test, y_pred, average="weighted") * 100, 2
            ),
            "best_model_recall": round(
                recall_score(y_test, y_pred, average="weighted") * 100, 2
            ),
            "val_scores": [round(score * 100, 2) for score in top_xgb_scores],
            "avg_val_score": round(avg_val_score * 100, 2),
            "std_val_score": round(std_val_score * 100, 2),
            "avg_test_score": round(avg_test_score * 100, 2),
            "std_test_score": round(std_test_score * 100, 2),
            "avg_test_accuracy": round(avg_test_accuracy * 100, 2),
            "std_test_accuracy": round(std_test_accuracy * 100, 2),
            "avg_test_f1_macro": round(avg_test_f1_macro * 100, 2),
            "std_test_f1_macro": round(std_test_f1_macro * 100, 2),
            "feature_importance": serialize_argument(
                feature_importance.head(20).to_dict("records")
            ),
        }

        print(classification_report(y_test, y_pred))
        print(
            f"Best model - OA, Global weighted F1 score, Macro F1 score:  {accuracy_score(y_test, y_pred)*100:.2f}% ,  {f1_score(y_test, y_pred, average='weighted')*100:.2f}%, {f1_score(y_test, y_pred, average='macro')*100:.2f}%"
        )
        print(
            f"Best model - Precision, Recall (test set):  {precision_score(y_test, y_pred, average='weighted')*100:.2f}%, {recall_score(y_test, y_pred, average='weighted')*100:.2f}%"
        )
        print(
            f"Top {num_models} GB models validation scores: {[round(score*100, 2) for score in top_xgb_scores]}"
        )
        print(
            f"GB models - Avg val score: {avg_val_score*100:.2f}% ± {std_val_score*100:.2f}%"
        )
        print(
            f"GB models - Avg test score: {avg_test_score*100:.2f}% ± {std_test_score*100:.2f}%"
        )
        print(
            f"GB models - Avg test accuracy: {avg_test_accuracy*100:.2f}% ± {std_test_accuracy*100:.2f}%"
        )
        print(
            f"GB models - Avg test macro F1: {avg_test_f1_macro*100:.2f}% ± {std_test_f1_macro*100:.2f}%"
        )
        print(f"GB models - Final test performance (across all top models):")
        print(
            f"  Accuracy: {avg_test_accuracy*100:.2f}% ± {std_test_accuracy*100:.2f}%"
        )
        print(f"  Weighted F1: {avg_test_score*100:.2f}% ± {std_test_score*100:.2f}%")
        print(
            f"  Macro F1: {avg_test_f1_macro*100:.2f}% ± {std_test_f1_macro*100:.2f}%"
        )

        # Print top 10 features
        print("\nTop 10 most important features:")
        print(feature_importance.head(10))

    if "Random" in MODELS:
        print(f"\n\n--------------------Random-----------------------------")
        random_classifier = RandomBaseline()
        random_classifier.fit(X_train, y_train)
        # Number of repetitions
        K = 1000

        # Lists to store the performance metrics
        accuracies = []
        f1_macros = []
        f1_globals = []

        # Repeat the random classifier K times
        for _ in range(K):
            y_pred_random = random_classifier.predict(X_test)

            accuracy_random = accuracy_score(y_test, y_pred_random)
            f1_random_macro = f1_score(y_test, y_pred_random, average="macro")
            f1_macro_global = f1_score(y_test, y_pred_random, average="weighted")

            # Append the metrics to the lists
            accuracies.append(accuracy_random)
            f1_macros.append(f1_random_macro)
            f1_globals.append(f1_macro_global)

        # Calculate mean and standard deviation
        mean_accuracy = np.mean(accuracies)
        std_accuracy = np.std(accuracies)
        mean_f1_macro = np.mean(f1_macros)
        std_f1_macro = np.std(f1_macros)
        mean_f1_global = np.mean(f1_globals)
        std_f1_global = np.std(f1_globals)

        results["Random"] = {
            "overall_accuracy": round(mean_accuracy * 100, 2),
            "overall_accuracy_std": round(std_accuracy * 100, 2),
            "f1_weighted": round(mean_f1_global * 100, 2),
            "f1_weighted_std": round(std_f1_global * 100, 2),
            "f1_macro": round(mean_f1_macro * 100, 2),
            "f1_macro_std": round(std_f1_macro * 100, 2),
        }

    # Create ensemble model if both RF and GB models are available
    if "RF" in MODELS and "GB" in MODELS and "RF" in all_models and "GB" in all_models:
        print(
            f"\n\n--------------------ENSEMBLE (RF + LGB)-----------------------------\n"
        )

        # Create ensemble model
        ensemble_model = create_ensemble_model(all_models["RF"], all_models["GB"])

        # Get ensemble predictions
        y_pred_ensemble = ensemble_model.predict(X_test)
        y_proba_ensemble = ensemble_model.predict_proba(X_test)

        # Calculate ensemble metrics
        ensemble_accuracy = accuracy_score(y_test, y_pred_ensemble)
        ensemble_f1_weighted = f1_score(y_test, y_pred_ensemble, average="weighted")
        ensemble_f1_macro = f1_score(y_test, y_pred_ensemble, average="macro")
        # add precision and recall
        ensemble_precision = precision_score(
            y_test, y_pred_ensemble, average="weighted"
        )
        ensemble_recall = recall_score(y_test, y_pred_ensemble, average="weighted")

        # Store ensemble results
        results["Ensemble"] = {
            "test_accuracy": round(ensemble_accuracy * 100, 2),
            "test_f1_weighted": round(ensemble_f1_weighted * 100, 2),
            "test_f1_macro": round(ensemble_f1_macro * 100, 2),
            "num_rf_models": len(all_models["RF"]),
            "num_lgb_models": len(all_models["GB"]),
            "total_models": len(all_models["RF"]) + len(all_models["GB"]),
            "precision": round(ensemble_precision * 100, 2),
            "recall": round(ensemble_recall * 100, 2),
        }

        print(f"Ensemble Model Performance:")
        print(
            f"  Models used: {len(all_models['RF'])} RF + {len(all_models['GB'])} LGB = {len(all_models['RF']) + len(all_models['GB'])} total"
        )
        print(f"  Test Accuracy: {ensemble_accuracy*100:.2f}%")
        print(f"  Test Weighted F1: {ensemble_f1_weighted*100:.2f}%")
        print(f"  Test Macro F1: {ensemble_f1_macro*100:.2f}%")
        print("\nEnsemble Classification Report:")
        print(classification_report(y_test, y_pred_ensemble))

        # Note: Ensemble model can be recreated from individual saved models using majority voting
        print(
            f"\nNote: Ensemble uses majority voting across all {len(all_models['RF']) + len(all_models['GB'])} models."
        )
        print(
            "To recreate: load all individual models and use majority voting on their predictions."
        )

    # Add arguments to the results
    results["arguments"] = {"SPLIT": SPLIT, "TASK": TASK, "num_models": num_models}

    # Create experiment directory name
    exp_dir_name = f"experiment-{TASK}"
    exp_dir_path = os.path.join(output_dir, exp_dir_name)

    # Create the directory if it doesn't exist
    os.makedirs(exp_dir_path, exist_ok=True)

    # Save results to JSON file
    json_filename = "results.json"
    json_path = os.path.join(exp_dir_path, json_filename)
    with open(json_path, "w") as json_file:
        json.dump(results, json_file, indent=4)

    # Save the top models if they were trained
    if "RF" in MODELS:
        for i, model in enumerate(all_models["RF"]):
            rf_model_path = os.path.join(exp_dir_path, f"rf_model_{i+1}.joblib")
            joblib.dump(model, rf_model_path)

    if "GB" in MODELS:
        for i, model in enumerate(all_models["GB"]):
            lgb_model_path = os.path.join(exp_dir_path, f"lgb_model_{i+1}.txt")
            model.booster_.save_model(lgb_model_path)

    # Note: Individual models can be loaded and combined into an ensemble using the create_ensemble_model function
    if "RF" in MODELS and "GB" in MODELS and "RF" in all_models and "GB" in all_models:
        print(
            f"\nAll individual models saved. Ensemble can be recreated by loading models and using create_ensemble_model()."
        )

    # Print results
    print(f"\nResults saved in: {exp_dir_path}")
    print(json.dumps(results, indent=4))


if __name__ == "__main__":
    load_dotenv()
    DATA_DIR = os.getenv("DATA_DIR")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train_ft_path",
        type=str,
        default=f"{DATA_DIR}/dataset/features/06-01_08-30_f=7days/train_features.parquet",
    )
    parser.add_argument("--SPLIT", type=float, default=0.1)
    parser.add_argument(
        "--TASK", type=str, default="DSR", choices=["DSR", "AWD", "ALL"]
    )
    parser.add_argument("--MODELS", type=str, nargs="+", default=["RF", "GB"])
    parser.add_argument("--CV", action="store_true", default=False)
    parser.add_argument("--output_dir", type=str, default=".")  # New argument
    parser.add_argument(
        "--num_models", type=int, default=5
    )  # New argument for number of top models to keep
    parser.add_argument("--SPLIT_VAL", type=float, default=0.0)  # New argument

    args = parser.parse_args()

    # Print all args
    print("---------------------------------")
    print(args)
    print("---------------------------------")

    main(
        SPLIT=args.SPLIT,
        SPLIT_VAL=args.SPLIT_VAL,
        TASK=args.TASK,
        MODELS=args.MODELS,
        output_dir=args.output_dir,
        train_ft_path=args.train_ft_path,
        num_models=args.num_models,
        cv=args.CV,
    )  # Pass the new argument
