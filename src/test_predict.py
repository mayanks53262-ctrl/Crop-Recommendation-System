from pathlib import Path
import unittest

import numpy as np
import pandas as pd

if __package__:
    from .predict import (
        PredictionInputError,
        check_feature_range_warnings,
        get_observed_feature_ranges,
        load_model_artifact,
        predict_crop,
    )
else:
    from predict import (
        PredictionInputError,
        check_feature_range_warnings,
        get_observed_feature_ranges,
        load_model_artifact,
        predict_crop,
    )

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "raw" / "Crop_recommendation.csv"


class PredictModuleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.artifact = load_model_artifact()
        cls.features = cls.artifact["feature_names"]
        cls.class_labels = cls.artifact["class_labels"]
        dataset = pd.read_csv(DATA_PATH)
        cls.valid_sample = dataset[cls.features].iloc[0].to_dict()
        cls.training_indices = tuple(
            cls.artifact["split_configuration"]["train_row_indices"]
        )
        cls.ranges = get_observed_feature_ranges(
            cls.artifact["dataset_sha256"], cls.training_indices
        )

    def check_ranges(self, values: dict[str, object]) -> list[dict[str, float | str]]:
        return check_feature_range_warnings(
            values,
            self.artifact["dataset_sha256"],
            self.training_indices,
        )

    def test_valid_dataset_sample_returns_known_crop(self) -> None:
        result = predict_crop(self.valid_sample, self.artifact)
        self.assertIn(result["predicted_crop"], self.class_labels)

    def test_missing_feature_is_rejected(self) -> None:
        incomplete = dict(self.valid_sample)
        incomplete.pop("humidity")
        with self.assertRaisesRegex(PredictionInputError, "Missing required feature"):
            predict_crop(incomplete, self.artifact)

    def test_non_numeric_value_is_rejected(self) -> None:
        invalid = dict(self.valid_sample)
        invalid["N"] = "not-numeric"
        with self.assertRaisesRegex(PredictionInputError, "must be a numeric"):
            predict_crop(invalid, self.artifact)

    def test_nan_and_infinity_are_rejected(self) -> None:
        for invalid_value in (np.nan, np.inf, -np.inf):
            with self.subTest(value=invalid_value):
                invalid = dict(self.valid_sample)
                invalid["rainfall"] = invalid_value
                with self.assertRaisesRegex(PredictionInputError, "must be finite"):
                    predict_crop(invalid, self.artifact)

    def test_negative_nutrient_and_rainfall_are_rejected(self) -> None:
        for feature in ("N", "P", "K", "rainfall"):
            with self.subTest(feature=feature):
                invalid = dict(self.valid_sample)
                invalid[feature] = -0.1
                with self.assertRaisesRegex(PredictionInputError, "cannot be negative"):
                    predict_crop(invalid, self.artifact)

    def test_negative_humidity_and_ph_are_rejected(self) -> None:
        for feature in ("humidity", "ph"):
            with self.subTest(feature=feature):
                invalid = dict(self.valid_sample)
                invalid[feature] = -0.1
                with self.assertRaisesRegex(PredictionInputError, "cannot be negative"):
                    predict_crop(invalid, self.artifact)

    def test_negative_temperature_is_allowed(self) -> None:
        below_freezing = dict(self.valid_sample)
        below_freezing["temperature"] = -5
        result = predict_crop(below_freezing, self.artifact)
        self.assertIn(result["predicted_crop"], self.class_labels)

    def test_output_format_and_optional_top_three_scores(self) -> None:
        result = predict_crop(self.valid_sample, self.artifact)
        self.assertEqual(result["predicted_crop"] in self.class_labels, True)

        model = self.artifact["model"]
        if not callable(getattr(model, "predict_proba", None)):
            self.assertEqual(set(result), {"predicted_crop"})
            return

        self.assertEqual(set(result), {"predicted_crop", "top_three", "score_type"})
        top_three = result["top_three"]
        self.assertEqual(len(top_three), min(3, len(self.class_labels)))
        scores = [item["score"] for item in top_three]
        self.assertTrue(np.isfinite(scores).all())
        self.assertTrue(all(0.0 <= score <= 1.0 for score in scores))
        self.assertEqual(scores, sorted(scores, reverse=True))
        self.assertTrue(all(item["crop"] in self.class_labels for item in top_three))
        self.assertIn("not agricultural success probability", result["score_type"])

    def test_in_range_values_return_no_range_warnings(self) -> None:
        self.assertEqual(self.check_ranges(self.valid_sample), [])

    def test_values_below_observed_ranges_warn_when_input_remains_valid(self) -> None:
        for feature, (minimum, _) in self.ranges.items():
            if minimum == 0:
                continue
            with self.subTest(feature=feature):
                values = dict(self.valid_sample)
                values[feature] = 0
                warnings = self.check_ranges(values)
                self.assertEqual([warning["feature"] for warning in warnings], [feature])
                self.assertEqual(warnings[0]["observed_min"], minimum)

    def test_n_zero_is_at_observed_minimum_and_negative_n_is_invalid(self) -> None:
        values = dict(self.valid_sample)
        values["N"] = 0
        self.assertEqual(self.check_ranges(values), [])
        values["N"] = -0.1
        with self.assertRaisesRegex(PredictionInputError, "cannot be negative"):
            self.check_ranges(values)

    def test_values_above_each_observed_range_warn(self) -> None:
        for feature, (_, maximum) in self.ranges.items():
            with self.subTest(feature=feature):
                values = dict(self.valid_sample)
                values[feature] = maximum + 1
                warnings = self.check_ranges(values)
                self.assertEqual([warning["feature"] for warning in warnings], [feature])
                self.assertEqual(warnings[0]["observed_max"], maximum)

    def test_multiple_out_of_range_values_return_multiple_warnings(self) -> None:
        values = {
            feature: self.ranges[feature][1] + 1
            for feature in self.features
        }
        warnings = self.check_ranges(values)
        self.assertEqual(
            [warning["feature"] for warning in warnings], self.features
        )

    def test_invalid_values_are_rejected_not_returned_as_range_warnings(self) -> None:
        invalid_cases = [
            ("missing", {key: value for key, value in self.valid_sample.items() if key != "N"}),
            ("nonnumeric", {**self.valid_sample, "P": "not-numeric"}),
            ("nan", {**self.valid_sample, "K": np.nan}),
            ("infinity", {**self.valid_sample, "rainfall": np.inf}),
            ("negative_ph", {**self.valid_sample, "ph": -1}),
        ]
        for case_name, values in invalid_cases:
            with self.subTest(case=case_name):
                with self.assertRaises(PredictionInputError):
                    self.check_ranges(values)

if __name__ == "__main__":
    print(
        "Input rules: reject negative N/P/K, humidity, pH, and rainfall; allow "
        "negative temperature. No arbitrary upper bounds are set. Scores, when "
        "available, are model output scores and not probabilities of crop survival, "
        "yield, or agricultural success."
    )
    unittest.main(verbosity=2)
