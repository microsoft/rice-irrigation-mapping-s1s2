# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from pathlib import Path
import joblib
import lightgbm as lgb
from sklearn.metrics import classification_report, confusion_matrix


def main():
    parser = argparse.ArgumentParser(
        description="Error analysis for rice irrigation mapping models."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Base data directory (for models, features, etc)",
        default="../../data",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to save results and plots",
    )
    parser.add_argument(
        "--task",
        type=str,
        choices=["DSR", "AWD"],
        default="DSR",
        help="Classification task",
    )
    parser.add_argument("--split", type=float, default=0.1, help="Test split ratio")
    parser.add_argument(
        "--split-val", type=float, default=0.0, help="Validation split ratio"
    )
    parser.add_argument("--lgb", action="store_true", help="Use LightGBM model")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Set paths and labels
    if args.task == "DSR":
        model_dir = (
            f"{args.data_dir}/models/20250526_Jun1-Sep5_90-10_f=4day/experiment-DSR"
        )
        train_ft_path = f"{args.data_dir}/dataset/features/06-01_09-05_f=4days_gamma0_3_clean_class/train_features.parquet"
        cf_labels = ["PTR", "DSR"]
    elif args.task == "AWD":
        model_dir = (
            f"{args.data_dir}/models/20250606_May1-Dec15_90-10_f=10day/experiment-AWD"
        )
        train_ft_path = f"{args.data_dir}/dataset/features/05-01_12-15_f=10days_gamma0_3_clean_class/train_features.parquet"
        cf_labels = ["CF", "AWD"]
    else:
        raise ValueError("Unknown task")

    # Load features
    df = pd.read_parquet(train_ft_path)
    assert "label" in df.columns, "label column is not in df.columns"

    # Map labels
    if args.task == "DSR":
        label_map = {"AWD": 0, "PTR": 0, "DSR": 1}
    elif args.task == "AWD":
        label_map = {"AWD": 1, "PTR": 0, "DSR": 0}
    else:
        raise ValueError("Unknown task")
    df["label"] = df["label"].map(label_map)

    X = df.copy()
    y = X["label"]

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
    ts_features = [f for f in drop_features if f in X.columns]
    X = X.drop(columns=ts_features, axis=1)

    # Split data
    if args.split_val > 0:
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=args.split, random_state=42, stratify=y
        )
        val_size = args.split_val / (1 - args.split)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size, random_state=42, stratify=y_temp
        )
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=args.split, random_state=42, stratify=y
        )
        X_val, y_val = None, None
    test_indices = X_test.index

    # Load model
    model_dir = Path(model_dir)
    models = {}
    model = None
    for model_path in model_dir.glob("*"):
        if model_path.suffix == ".joblib" and not args.lgb:
            models[f"{model_path.stem}"] = joblib.load(model_path)
            model = models[f"{model_path.stem}"]
        elif model_path.suffix == ".txt":
            model = lgb.Booster(model_file=str(model_path))
            models[f"{model_path.stem}"] = model
    if not models:
        raise ValueError(f"No models found in {model_dir}")

    # Prediction
    X_pred = X_test.copy()
    X_pred.columns = X_pred.columns.astype(str)
    feature_cols = X_pred.columns.difference(drop_features)
    X_pred = X_pred[feature_cols]
    required_features = model.feature_name()
    assert "location" not in required_features, "location feature is not allowed"
    X_pred = X_pred[required_features]
    predictions = model.predict(X_pred)
    if isinstance(predictions, np.ndarray) and predictions.ndim == 2:
        predictions = np.argmax(predictions, axis=1)
    X_pred["predictions"] = predictions
    X_pred["label"] = y_test

    # Add original Class column
    test_classes = df.loc[test_indices, "Class"]
    X_pred["Class"] = list(test_classes)
    X_pred = X_pred[["Class", "predictions", "label"]]
    X_pred.to_csv(os.path.join(args.output_dir, "test_predictions.csv"), index=False)

    # Stacked bar plot for correct/incorrect by class
    labels = ["DSR", "PTR", "AWD"]
    correct_counts = []
    incorrect_counts = []
    for cls in labels:
        class_mask = X_pred["Class"] == cls
        class_preds = X_pred.loc[class_mask, "predictions"]
        class_labels = X_pred.loc[class_mask, "label"]
        correct = (class_preds == class_labels).sum()
        incorrect = (class_preds != class_labels).sum()
        correct_counts.append(correct)
        incorrect_counts.append(incorrect)
    x = np.arange(len(labels))
    width = 0.6
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.bar(x, correct_counts, width, label="Correct", color="#3498db")
    ax.bar(
        x,
        incorrect_counts,
        width,
        bottom=correct_counts,
        label="Incorrect",
        color="#2c3e50",
        hatch="//",
    )
    ax.set_xlabel("Original Classes")
    ax.set_ylabel("Counts")
    ax.set_title(f"{args.task} Classification Contributions [test set]")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    total_counts = [c + i for c, i in zip(correct_counts, incorrect_counts)]
    if any(total_counts):
        plt.ylim(0, max(total_counts) * 1.15)
    else:
        plt.ylim(0, 100)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "class_contributions.png"))
    plt.close()

    # Print analysis results
    with open(os.path.join(args.output_dir, "class_contributions.txt"), "w") as f:
        for i, (correct, incorrect) in enumerate(zip(correct_counts, incorrect_counts)):
            total = correct + incorrect
            if total > 0:
                ratio = incorrect / total * 100
                f.write(
                    f"{labels[i]}: correct: {correct}, incorrect: {incorrect}, total: {total}, ratio: {ratio:.2f}%\n"
                )
            else:
                f.write(f"{labels[i]}: No samples in test set\n")

    # Confusion matrix
    cm = confusion_matrix(np.array(y_test), predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        annot_kws={"size": 16},
        cmap=sns.color_palette("Blues", as_cmap=True),
        xticklabels=cf_labels,
        yticklabels=cf_labels,
    )
    plt.title(f"{args.task} Confusion Matrix (Test Set)")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.savefig(os.path.join(args.output_dir, "confusion_matrix.png"))
    plt.close()

    # Classification report
    report = classification_report(np.array(y_test), predictions, output_dict=True)
    pd.DataFrame(report).transpose().to_csv(
        os.path.join(args.output_dir, "classification_report.csv")
    )

    # Feature importance
    if args.lgb:
        lgb.plot_importance(
            model,
            importance_type="gain",
            figsize=(7, 6),
            max_num_features=30,
            title=f"{args.task}: LightGBM Feature Importance (Gain) ",
        )
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, "feature_importance_gain.png"))
        plt.close()
        lgb.plot_importance(
            model,
            importance_type="split",
            figsize=(7, 6),
            max_num_features=30,
            title=f"{args.task}: LightGBM Feature Importance (Split) ",
        )
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, "feature_importance_split.png"))
        plt.close()


if __name__ == "__main__":
    main()
