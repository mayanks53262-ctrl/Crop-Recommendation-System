from functools import lru_cache
import hashlib
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "selected_model.joblib"
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "Crop_recommendation.csv"
REQUIRED_FEATURES = [
    "N",
    "P",
    "K",
    "temperature",
    "humidity",
    "ph",
    "rainfall",
]
NON_NEGATIVE_FEATURES = {"N", "P", "K", "humidity", "ph", "rainfall"}


class PredictionInputError(ValueError):
    """Raised when prediction inputs are missing or invalid."""


class ModelArtifactError(RuntimeError):
    """Raised when the selected model artifact is missing or invalid."""


@lru_cache(maxsize=1)
def get_observed_feature_ranges(
    expected_dataset_sha256: str | None = None,
    training_row_indices: tuple[int, ...] | None = None,
) -> dict[str, tuple[float, float]]:
    """Return observed min/max values from the selected model's training rows."""
    if not DATA_PATH.is_file():
        raise FileNotFoundError(f"Range reference dataset not found: {DATA_PATH}")

    dataset_bytes = DATA_PATH.read_bytes()
    if expected_dataset_sha256 is not None:
        actual_fingerprint = hashlib.sha256(dataset_bytes).hexdigest()
        if actual_fingerprint != expected_dataset_sha256:
            raise ModelArtifactError(
                "Range reference dataset does not match the selected model dataset."
            )
        if training_row_indices is None:
            raise ModelArtifactError(
                "Selected model metadata does not contain training row indices."
            )

    data = pd.read_csv(DATA_PATH)
    missing = [feature for feature in REQUIRED_FEATURES if feature not in data.columns]
    if missing:
        raise ValueError(f"Range reference dataset is missing features: {missing}")
    if training_row_indices is not None:
        indices = np.asarray(training_row_indices, dtype=int)
        if (
            indices.size == 0
            or np.unique(indices).size != indices.size
            or indices.min() < 0
            or indices.max() >= len(data)
        ):
            raise ModelArtifactError(
                "Selected model training row indices are invalid for the range dataset."
            )
        data = data.iloc[indices]

    ranges: dict[str, tuple[float, float]] = {}
    for feature in REQUIRED_FEATURES:
        values = pd.to_numeric(data[feature], errors="coerce")
        if values.isna().any() or not np.isfinite(values.to_numpy()).all():
            raise ValueError(
                f"Range reference feature {feature!r} must contain finite numbers."
            )
        ranges[feature] = (float(values.min()), float(values.max()))
    return ranges


def load_model_artifact(model_path: Path = MODEL_PATH) -> dict[str, Any]:
    """Load and check the expected selected-model artifact."""
    if not model_path.is_file():
        raise FileNotFoundError(f"Selected model artifact not found: {model_path}")

    try:
        artifact = joblib.load(model_path)
    except (OSError, EOFError, ValueError, KeyError, ImportError) as exc:
        raise ModelArtifactError(
            f"Could not load selected model artifact at {model_path}: {exc}"
        ) from exc

    if not isinstance(artifact, dict):
        raise ModelArtifactError("Selected model artifact must contain a metadata dictionary.")
    if "model" not in artifact:
        raise ModelArtifactError("Selected model artifact is missing the 'model' entry.")
    feature_names = artifact.get("feature_names")
    if feature_names != REQUIRED_FEATURES:
        raise ModelArtifactError(
            f"Selected model feature names must be in this order: {REQUIRED_FEATURES}"
        )
    class_labels = artifact.get("class_labels")
    if not isinstance(class_labels, list) or not class_labels:
        raise ModelArtifactError("Selected model artifact has no valid class label list.")

    model = artifact["model"]
    if not callable(getattr(model, "predict", None)):
        raise ModelArtifactError("Selected model does not provide a predict method.")
    fitted_feature_names = getattr(model, "feature_names_in_", None)
    if fitted_feature_names is not None and list(fitted_feature_names) != feature_names:
        raise ModelArtifactError(
            "Saved estimator feature order does not match its artifact metadata."
        )
    return artifact


def validate_inputs(
    values: dict[str, object], feature_names: list[str] | None = None
) -> pd.DataFrame:
    """Validate application inputs and arrange them in the artifact feature order."""
    ordered_features = REQUIRED_FEATURES if feature_names is None else feature_names
    missing = [feature for feature in ordered_features if feature not in values]
    if missing:
        raise PredictionInputError(f"Missing required feature(s): {', '.join(missing)}.")

    unexpected = [feature for feature in values if feature not in ordered_features]
    if unexpected:
        raise PredictionInputError(
            f"Unexpected feature(s): {', '.join(unexpected)}."
        )

    clean_values: dict[str, float] = {}
    for feature in ordered_features:
        value = values[feature]
        if isinstance(value, (bool, np.bool_)):
            raise PredictionInputError(f"{feature} must be a numeric value, not boolean.")
        try:
            numeric_value = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise PredictionInputError(
                f"{feature} must be a numeric value; received {value!r}."
            ) from exc
        if not np.isfinite(numeric_value):
            raise PredictionInputError(f"{feature} must be finite, not NaN or infinity.")
        if feature in NON_NEGATIVE_FEATURES and numeric_value < 0:
            raise PredictionInputError(
                f"{feature} cannot be negative; its measurement is non-negative."
            )
        clean_values[feature] = numeric_value

    return pd.DataFrame([clean_values], columns=ordered_features)


def check_feature_range_warnings(
    values: dict[str, object],
    expected_dataset_sha256: str | None = None,
    training_row_indices: tuple[int, ...] | None = None,
) -> list[dict[str, float | str]]:
    """Describe valid inputs outside observed training ranges without rejecting them."""
    validated = validate_inputs(values)
    observed_ranges = get_observed_feature_ranges(
        expected_dataset_sha256, training_row_indices
    )
    warnings: list[dict[str, float | str]] = []

    for feature in REQUIRED_FEATURES:
        value = float(validated.loc[0, feature])
        minimum, maximum = observed_ranges[feature]
        if value < minimum or value > maximum:
            warnings.append(
                {
                    "feature": feature,
                    "value": value,
                    "observed_min": minimum,
                    "observed_max": maximum,
                    "message": (
                        f"{feature} is outside the observed training-data range "
                        f"[{minimum:g}, {maximum:g}]. The model may be less reliable "
                        "for inputs outside its observed data."
                    ),
                }
            )
    return warnings


def predict_crop(
    values: dict[str, object],
    artifact: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Predict a crop and, where possible, return its top three model scores.

    The returned scores describe the model's output distribution. They are not
    probabilities of crop survival, yield, or agricultural success.
    """
    loaded_artifact = load_model_artifact() if artifact is None else artifact
    feature_names = loaded_artifact["feature_names"]
    model = loaded_artifact["model"]
    model_input = validate_inputs(values, feature_names)

    prediction = model.predict(model_input)
    if np.asarray(prediction).shape != (1,):
        raise ModelArtifactError(
            f"Expected one predicted label for one input row; received shape "
            f"{np.asarray(prediction).shape}."
        )
    predicted_label = prediction[0]
    if predicted_label not in loaded_artifact["class_labels"]:
        raise ModelArtifactError(
            f"Model predicted unknown crop label: {predicted_label!r}."
        )

    result: dict[str, Any] = {"predicted_crop": predicted_label}
    predict_proba = getattr(model, "predict_proba", None)
    if callable(predict_proba):
        probabilities = np.asarray(predict_proba(model_input), dtype=float)
        model_classes = list(getattr(model, "classes_", []))
        if probabilities.shape != (1, len(model_classes)):
            raise ModelArtifactError(
                "Model probability output does not match the fitted class labels."
            )
        if not np.isfinite(probabilities).all() or (
            (probabilities < 0) | (probabilities > 1)
        ).any():
            raise ModelArtifactError(
                "Model returned probability scores outside the finite [0, 1] range."
            )
        if not np.isclose(probabilities.sum(), 1.0, atol=1e-6):
            raise ModelArtifactError(
                "Model probability scores do not sum approximately to 1."
            )
        if set(model_classes) != set(loaded_artifact["class_labels"]):
            raise ModelArtifactError(
                "Estimator classes do not match the artifact's saved class labels."
            )

        top_indices = np.argsort(probabilities[0])[::-1][:3]
        result["top_three"] = [
            {
                "crop": model_classes[index],
                "score": float(probabilities[0][index]),
            }
            for index in top_indices
        ]
        result["score_type"] = "model probability output; not agricultural success probability"
    return result
