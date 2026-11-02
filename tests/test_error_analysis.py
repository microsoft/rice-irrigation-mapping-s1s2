# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import joblib
import lightgbm as lgb
import matplotlib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from scripts.visualization import error_analysis

matplotlib.use("Agg")


class UntrustedModel:
    def __init__(self, marker):
        self.marker = marker

    def __reduce__(self):
        return Path.write_text, (self.marker, "deserialized")


class ErrorAnalysisModelTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.data_dir = Path(self.temp_dir.name)
        self.model_dir = (
            self.data_dir / "models/20250526_Jun1-Sep5_90-10_f=4day/experiment-DSR"
        )
        self.model_dir.mkdir(parents=True)
        self.model_path = self.model_dir / "rf_model.joblib"
        self.marker = self.data_dir / "deserialized"
        self.features = pd.DataFrame(
            {"feature": [0.0, 1.0] * 20, "label": ["PTR", "DSR"] * 20}
        )
        self.features["Class"] = self.features["label"]
        self.argv = [
            "error_analysis.py",
            "--data-dir",
            str(self.data_dir),
            "--output-dir",
            str(self.data_dir / "output"),
        ]

    def test_cli_rejects_untrusted_joblib_before_deserialization(self):
        joblib.dump(UntrustedModel(self.marker), self.model_path)

        for options, message in (
            ([], "require --trusted-model-sha256"),
            (["--trusted-model-sha256", "0" * 64], "SHA-256 verification failed"),
        ):
            with self.subTest(options=options):
                with (
                    patch("sys.argv", self.argv + options),
                    patch.object(
                        error_analysis.pd,
                        "read_parquet",
                        return_value=self.features.copy(),
                    ),
                    patch.object(error_analysis.joblib, "load") as deserialize,
                    self.assertRaisesRegex(ValueError, message),
                ):
                    error_analysis.main()
                deserialize.assert_not_called()
                self.assertFalse(self.marker.exists())

    def test_approved_model_loads_with_matching_digest(self):
        model = RandomForestClassifier(n_estimators=2, random_state=42).fit(
            self.features[["feature"]], self.features["label"]
        )
        joblib.dump(model, self.model_path)
        digest = hashlib.sha256(self.model_path.read_bytes()).hexdigest()

        restored = error_analysis.load_verified_joblib(self.model_path, [digest])

        np.testing.assert_array_equal(
            restored.predict(self.features[["feature"]]),
            model.predict(self.features[["feature"]]),
        )

    def test_tampered_model_cannot_use_original_digest(self):
        joblib.dump({"approved": True}, self.model_path)
        digest = hashlib.sha256(self.model_path.read_bytes()).hexdigest()
        joblib.dump(UntrustedModel(self.marker), self.model_path)

        with self.assertRaisesRegex(ValueError, "SHA-256 verification failed"):
            error_analysis.load_verified_joblib(self.model_path, [digest])

        self.assertFalse(self.marker.exists())

    def test_deserializes_verified_bytes_even_if_file_is_replaced(self):
        joblib.dump({"approved": True}, self.model_path)
        digest = hashlib.sha256(self.model_path.read_bytes()).hexdigest()
        sha256 = hashlib.sha256

        def replace_file_after_read(model_bytes):
            joblib.dump(UntrustedModel(self.marker), self.model_path)
            return sha256(model_bytes)

        with patch.object(
            error_analysis.hashlib, "sha256", side_effect=replace_file_after_read
        ):
            restored = error_analysis.load_verified_joblib(self.model_path, [digest])

        self.assertEqual(restored, {"approved": True})
        self.assertFalse(self.marker.exists())

    def test_lightgbm_analysis_ignores_joblib_without_trusted_digests(self):
        joblib.dump(UntrustedModel(self.marker), self.model_path)
        model = lgb.train(
            {
                "objective": "multiclass",
                "num_class": 2,
                "min_data_in_leaf": 1,
                "num_threads": 1,
                "verbosity": -1,
            },
            lgb.Dataset(
                self.features[["feature"]],
                label=(self.features["label"] == "DSR").astype(int),
            ),
            num_boost_round=2,
        )
        model.save_model(str(self.model_dir / "lgb_model.txt"))

        with (
            patch("sys.argv", self.argv + ["--lgb"]),
            patch.object(
                error_analysis.pd, "read_parquet", return_value=self.features.copy()
            ),
            patch.object(error_analysis.joblib, "load") as deserialize,
        ):
            error_analysis.main()

        deserialize.assert_not_called()
        self.assertFalse(self.marker.exists())
        self.assertTrue((self.data_dir / "output/test_predictions.csv").is_file())
        self.assertTrue(
            (self.data_dir / "output/feature_importance_gain.png").is_file()
        )


if __name__ == "__main__":
    unittest.main()
