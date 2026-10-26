from pathlib import Path
import unittest

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "raw" / "Crop_recommendation.csv"
MODEL_PATH = ROOT / "models" / "random_forest_baseline.joblib"


class PredictionInputError(ValueError):
    """Raised when application input does not match the prediction schema."""


def validate_application_input(
    values: dict[str, object], feature_names: list[str]
) -> pd.DataFrame:
    missing = [name for name in feature_names if name not in values]
    if missing:
        raise PredictionInputError(f"Missing required features: {missing}")

    unexpected = [name for name in values if name not in feature_names]
    if unexpected:
        raise PredictionInputError(f"Unexpected features: {unexpected}")

    validated: dict[str, float] = {}
    non_negative_features = {"N", "P", "K", "humidity", "rainfall"}
    for name in feature_names:
        value = values[name]
        if isinstance(value, bool):
            raise PredictionInputError(f"Feature {name!r} must be numeric, not boolean.")
        try:
            numeric_value = float(value)
        except (TypeError, ValueError) as exc:
            raise PredictionInputError(f"Feature {name!r} must be numeric.") from exc
        if not np.isfinite(numeric_value):
            raise PredictionInputError(f"Feature {name!r} must be finite.")
        if name in non_negative_features and numeric_value < 0:
            raise PredictionInputError(
                f"Feature {name!r} cannot be negative by its measurement scale."
            )
        validated[name] = numeric_value

    return pd.DataFrame([validated], columns=feature_names)


class SavedModelPredictionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Saved model not found at {MODEL_PATH}")
        artifact = joblib.load(MODEL_PATH)
        cls.model = artifact["model"]
        cls.feature_names = artifact["feature_names"]
        cls.class_labels = artifact["class_labels"]
        if not DATA_PATH.exists():
            raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")
        cls.sample = pd.read_csv(DATA_PATH)[cls.feature_names].iloc[0].to_dict()

    def test_valid_dataset_row_predicts_known_label_with_expected_shape(self) -> None:
        prediction = self.model.predict(pd.DataFrame([self.sample]))
        self.assertEqual(prediction.shape, (1,))
        self.assertIn(prediction[0], self.class_labels)

    def test_prediction_probabilities_are_valid_when_available(self) -> None:
        if not hasattr(self.model, "predict_proba"):
            self.skipTest("This model does not expose predict_proba.")
        probabilities = self.model.predict_proba(pd.DataFrame([self.sample]))
        self.assertEqual(probabilities.shape, (1, len(self.class_labels)))
        self.assertTrue(np.isfinite(probabilities).all())
        self.assertTrue(((probabilities >= 0) & (probabilities <= 1)).all())
        self.assertAlmostEqual(float(probabilities.sum()), 1.0, places=6)

    def test_application_input_rejects_missing_feature(self) -> None:
        incomplete = dict(self.sample)
        incomplete.pop(self.feature_names[0])
        with self.assertRaisesRegex(PredictionInputError, "Missing required features"):
            validate_application_input(incomplete, self.feature_names)

    def test_application_input_rejects_non_numeric_value(self) -> None:
        invalid = dict(self.sample)
        invalid["N"] = "not-a-number"
        with self.assertRaisesRegex(PredictionInputError, "must be numeric"):
            validate_application_input(invalid, self.feature_names)

    def test_application_input_rejects_negative_nonnegative_measurement(self) -> None:
        invalid = dict(self.sample)
        invalid["N"] = -1
        with self.assertRaisesRegex(PredictionInputError, "cannot be negative"):
            validate_application_input(invalid, self.feature_names)

    def test_application_input_allows_subzero_temperature(self) -> None:
        below_freezing = dict(self.sample)
        below_freezing["temperature"] = -5
        validated = validate_application_input(below_freezing, self.feature_names)
        self.assertEqual(validated.loc[0, "temperature"], -5)

    def test_application_input_rejects_non_finite_value(self) -> None:
        invalid = dict(self.sample)
        invalid["rainfall"] = float("inf")
        with self.assertRaisesRegex(PredictionInputError, "must be finite"):
            validate_application_input(invalid, self.feature_names)


if __name__ == "__main__":
    print(
        "Input validation policy: N/P/K, humidity, and rainfall are checked only "
        "for non-negativity based on their measurement scales; no dataset-specific "
        "minimum or maximum is imposed on pH or temperature, and no upper bound is "
        "imposed on the other inputs. Temperature may be below zero. These checks "
        "are application-layer checks and are not applied by raw model.predict()."
    )
    print(
        "Model probabilities are normalized output scores, not guaranteed "
        "probabilities of agricultural success."
    )
    unittest.main(verbosity=2)
