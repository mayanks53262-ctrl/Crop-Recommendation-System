from io import StringIO
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "raw" / "Crop_recommendation.csv"
MODEL_PATH = ROOT / "models" / "random_forest_baseline.joblib"
BASELINE_REPORT_PATH = ROOT / "reports" / "baseline_metrics.txt"
CV_REPORT_PATH = ROOT / "reports" / "cross_validation_results.txt"
MISCLASSIFIED_PATH = ROOT / "reports" / "misclassified_samples.csv"
PER_CLASS_PATH = ROOT / "reports" / "per_class_metrics.csv"
SPLIT_PATH = ROOT / "reports" / "baseline_split_indices.csv"
FIGURES_DIR = ROOT / "reports" / "figures"
CONFUSION_MATRIX_PATH = FIGURES_DIR / "test_confusion_matrix.png"

FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
TARGET = "label"
RANDOM_STATE = 42
TEST_SIZE = 0.20
CLASS_LABELS = [
    "apple",
    "banana",
    "blackgram",
    "chickpea",
    "coconut",
    "coffee",
    "cotton",
    "grapes",
    "jute",
    "kidneybeans",
    "lentil",
    "maize",
    "mango",
    "mothbeans",
    "mungbean",
    "muskmelon",
    "orange",
    "papaya",
    "pigeonpeas",
    "pomegranate",
    "rice",
    "watermelon",
]


def load_and_validate_data() -> tuple[pd.DataFrame, pd.Series]:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")

    data = pd.read_csv(DATA_PATH)
    required = FEATURES + [TARGET]
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    if data.empty:
        raise ValueError("Dataset has no rows.")
    if data[required].isna().any().any():
        missing_values = data[required].isna().sum()
        raise ValueError(
            f"Dataset has missing values: "
            f"{missing_values[missing_values > 0].to_dict()}"
        )

    X = data[FEATURES].copy()
    for feature in FEATURES:
        converted = pd.to_numeric(X[feature], errors="coerce")
        if converted.isna().any() or not np.isfinite(converted.to_numpy()).all():
            raise ValueError(f"Feature {feature!r} must contain finite numeric values.")
        X[feature] = converted

    y = data[TARGET].copy()
    if not y.map(lambda value: isinstance(value, str)).all():
        raise ValueError(f"Target {TARGET!r} must contain crop names as text.")
    if y.str.strip().eq("").any():
        raise ValueError(f"Target {TARGET!r} contains blank crop names.")
    if sorted(y.unique().tolist()) != CLASS_LABELS:
        raise ValueError("Dataset crop classes do not match the saved model classes.")
    return X, y


def verify_saved_baseline(
    X_test: pd.DataFrame, y_test: pd.Series, artifact: dict
) -> tuple[np.ndarray, dict[str, float]]:
    model = artifact.get("model")
    if not isinstance(model, RandomForestClassifier):
        raise TypeError("Saved artifact does not contain a RandomForestClassifier.")
    if artifact.get("feature_names") != FEATURES:
        raise ValueError("Saved model feature names or their order do not match.")
    if artifact.get("class_labels") != CLASS_LABELS:
        raise ValueError("Saved model crop labels do not match the dataset.")
    if artifact.get("random_state") != RANDOM_STATE or artifact.get("test_size") != TEST_SIZE:
        raise ValueError("Saved model split metadata does not match the baseline split.")
    if model.n_estimators != 100 or model.random_state != RANDOM_STATE:
        raise ValueError("Saved Random Forest configuration differs from the baseline.")
    if model.n_jobs != 1:
        raise ValueError("Saved Random Forest n_jobs setting differs from the baseline.")
    if model.feature_names_in_.tolist() != FEATURES or model.classes_.tolist() != CLASS_LABELS:
        raise ValueError("Fitted model feature or class metadata does not match.")

    predictions = model.predict(X_test[artifact["feature_names"]])
    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "macro_f1": f1_score(
            y_test, predictions, labels=CLASS_LABELS, average="macro", zero_division=0
        ),
    }

    if not BASELINE_REPORT_PATH.exists():
        raise FileNotFoundError(
            f"Cannot verify the original baseline split without {BASELINE_REPORT_PATH}"
        )
    report = BASELINE_REPORT_PATH.read_text(encoding="utf-8")
    try:
        random_forest_section = report.split("RandomForestClassifier\n", 1)[1]
        matrix_text = random_forest_section.split(
            "Confusion matrix (rows=actual, columns=predicted):\n", 1
        )[1]
        matrix_text = matrix_text.split("\n\nInterpretation note:", 1)[0]
        reported_matrix = pd.read_csv(
            StringIO(matrix_text), sep=r"\s+", index_col=0
        )
        reported_accuracy = float(
            next(
                line.split(": ", 1)[1]
                for line in random_forest_section.splitlines()
                if line.startswith("Accuracy:")
            )
        )
        reported_macro_f1 = float(
            next(
                line.split(": ", 1)[1]
                for line in random_forest_section.splitlines()
                if line.startswith("Macro-F1:")
            )
        )
    except (IndexError, StopIteration, ValueError, pd.errors.ParserError) as exc:
        raise ValueError("Could not parse the saved Random Forest baseline report.") from exc

    expected_matrix = confusion_matrix(y_test, predictions, labels=CLASS_LABELS)
    if reported_matrix.index.tolist() != CLASS_LABELS:
        raise ValueError("Baseline report confusion-matrix row labels do not match.")
    if reported_matrix.columns.tolist() != CLASS_LABELS:
        raise ValueError("Baseline report confusion-matrix column labels do not match.")
    if not np.array_equal(reported_matrix.to_numpy(), expected_matrix):
        raise ValueError(
            "Recreated test split does not reproduce the saved baseline confusion matrix."
        )
    if f"{metrics['accuracy']:.4f}" != f"{reported_accuracy:.4f}":
        raise ValueError("Recreated split accuracy does not match the baseline report.")
    if f"{metrics['macro_f1']:.4f}" != f"{reported_macro_f1:.4f}":
        raise ValueError("Recreated split macro-F1 does not match the baseline report.")

    return predictions, metrics


def save_test_confusion_matrix(matrix: np.ndarray) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(14, 12))
    image = ax.imshow(matrix, interpolation="nearest", cmap="Blues")
    fig.colorbar(image, ax=ax, label="Number of samples")
    ax.set(
        title="Random Forest Held-Out Test Confusion Matrix",
        xlabel="Predicted crop",
        ylabel="Actual crop",
        xticks=np.arange(len(CLASS_LABELS)),
        yticks=np.arange(len(CLASS_LABELS)),
        xticklabels=CLASS_LABELS,
        yticklabels=CLASS_LABELS,
    )
    plt.setp(ax.get_xticklabels(), rotation=90, ha="center")
    fig.tight_layout()
    fig.savefig(CONFUSION_MATRIX_PATH, dpi=300)
    plt.close(fig)


def main() -> None:
    X, y = load_and_validate_data()
    train_indices, test_indices = train_test_split(
        np.arange(len(X)),
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    X_train, y_train = X.iloc[train_indices], y.iloc[train_indices]
    X_test, y_test = X.iloc[test_indices], y.iloc[test_indices]

    cross_validation_model = RandomForestClassifier(
        n_estimators=100,
        random_state=RANDOM_STATE,
        n_jobs=1,
    )
    cross_validation = StratifiedKFold(
        n_splits=5, shuffle=True, random_state=RANDOM_STATE
    )
    scores = cross_validate(
        cross_validation_model,
        X_train,
        y_train,
        cv=cross_validation,
        scoring={"accuracy": "accuracy", "macro_f1": "f1_macro"},
        return_train_score=False,
        n_jobs=1,
    )
    accuracy_scores = scores["test_accuracy"]
    macro_f1_scores = scores["test_macro_f1"]

    artifact = joblib.load(MODEL_PATH)
    predictions, held_out_metrics = verify_saved_baseline(X_test, y_test, artifact)

    misclassified = X_test.copy()
    misclassified["actual_crop"] = y_test
    misclassified["predicted_crop"] = predictions
    misclassified["correct"] = misclassified["actual_crop"].eq(
        misclassified["predicted_crop"]
    )
    misclassified.to_csv(MISCLASSIFIED_PATH, index_label="row_index")

    per_class = classification_report(
        y_test,
        predictions,
        labels=CLASS_LABELS,
        target_names=CLASS_LABELS,
        output_dict=True,
        zero_division=0,
    )
    per_class_frame = pd.DataFrame.from_dict(
        {label: per_class[label] for label in CLASS_LABELS}, orient="index"
    )
    per_class_frame = per_class_frame.loc[CLASS_LABELS, ["precision", "recall", "f1-score", "support"]]
    per_class_frame.index.name = "crop"
    per_class_frame = per_class_frame.rename(columns={"f1-score": "f1"})
    per_class_frame.to_csv(PER_CLASS_PATH)

    matrix = confusion_matrix(y_test, predictions, labels=CLASS_LABELS)
    save_test_confusion_matrix(matrix)

    split_manifest = pd.DataFrame(
        {
            "row_index": np.concatenate((train_indices, test_indices)),
            "partition": ["train"] * len(train_indices) + ["test"] * len(test_indices),
            "crop": pd.concat(
                (y.iloc[train_indices], y.iloc[test_indices]), ignore_index=True
            ),
        }
    ).sort_values("row_index")
    split_manifest.to_csv(SPLIT_PATH, index=False)

    report_lines = [
        "Crop Recommendation System - 5-Fold Cross-Validation",
        "=" * 58,
        f"Dataset rows: {len(X)}",
        f"Training partition rows used for CV: {len(X_train)}",
        f"Held-out test rows kept out of CV: {len(X_test)}",
        "CV: StratifiedKFold(n_splits=5, shuffle=True, random_state=42)",
        "Estimator: RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=1)",
        "Preprocessing: none required for these numeric features.",
        "The held-out test set was not used during cross-validation or model selection.",
        "",
        "Fold results:",
        "Fold | Accuracy | Macro-F1",
    ]
    for fold, (accuracy, macro_f1) in enumerate(
        zip(accuracy_scores, macro_f1_scores), start=1
    ):
        report_lines.append(f"{fold:>4} | {accuracy:.4f}   | {macro_f1:.4f}")
    report_lines.extend(
        [
            "",
            f"Accuracy mean: {accuracy_scores.mean():.4f}",
            f"Accuracy standard deviation (sample, ddof=1): {accuracy_scores.std(ddof=1):.4f}",
            f"Macro-F1 mean: {macro_f1_scores.mean():.4f}",
            f"Macro-F1 standard deviation (sample, ddof=1): {macro_f1_scores.std(ddof=1):.4f}",
            "",
            "Held-out split verification:",
            "The saved artifact has matching split metadata and model configuration. "
            "Recreating the stratified split also reproduces the complete per-class "
            "confusion matrix and rounded metrics in reports/baseline_metrics.txt.",
            "The original artifact did not save row IDs or a dataset fingerprint; "
            "reports/baseline_split_indices.csv records the row assignments recreated "
            "from the current dataset and baseline split settings.",
            "",
            "Saved-model held-out test results:",
            f"Accuracy: {held_out_metrics['accuracy']:.4f}",
            f"Macro-F1: {held_out_metrics['macro_f1']:.4f}",
            f"Misclassified samples: {int((~misclassified['correct']).sum())} / {len(y_test)}",
            "",
            "Interpretation note:",
            "Cross-validation and held-out results measure performance on this dataset "
            "only. They do not establish suitability for real farms or generalization "
            "to other locations or conditions.",
        ]
    )
    CV_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    CV_REPORT_PATH.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    print("\n".join(report_lines))
    print(f"\nSaved fold and evaluation report to: {CV_REPORT_PATH}")
    print(f"Saved split row assignments to: {SPLIT_PATH}")
    print(f"Saved test error details to: {MISCLASSIFIED_PATH}")
    print(f"Saved per-class metrics to: {PER_CLASS_PATH}")
    print(f"Saved held-out confusion matrix to: {CONFUSION_MATRIX_PATH}")


if __name__ == "__main__":
    main()
