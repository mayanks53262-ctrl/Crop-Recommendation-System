from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "raw" / "Crop_recommendation.csv"
REPORT_PATH = ROOT / "reports" / "baseline_metrics.txt"
FIGURES_DIR = ROOT / "reports" / "figures"
MODEL_PATH = ROOT / "models" / "random_forest_baseline.joblib"

FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
TARGET = "label"
RANDOM_STATE = 42
TEST_SIZE = 0.20


def load_and_validate_data() -> tuple[pd.DataFrame, pd.Series]:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    required_columns = FEATURES + [TARGET]
    missing_columns = [column for column in required_columns if column not in df]
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {missing_columns}")
    if df.empty:
        raise ValueError("Dataset has no rows.")
    if df[required_columns].isna().any().any():
        missing = df[required_columns].isna().sum()
        missing = missing[missing > 0].to_dict()
        raise ValueError(f"Dataset has missing required values: {missing}")

    X = df[FEATURES].copy()
    for feature in FEATURES:
        converted = pd.to_numeric(X[feature], errors="coerce")
        if converted.isna().any():
            raise ValueError(f"Feature {feature!r} contains non-numeric values.")
        if not np.isfinite(converted.to_numpy()).all():
            raise ValueError(f"Feature {feature!r} contains infinite values.")
        X[feature] = converted

    y = df[TARGET].copy()
    if not y.map(lambda value: isinstance(value, str)).all():
        raise ValueError(f"Target column {TARGET!r} must contain crop names as text.")
    if y.str.strip().eq("").any():
        raise ValueError(f"Target column {TARGET!r} contains blank crop names.")

    return X, y


def save_confusion_matrix(
    matrix: np.ndarray, class_labels: list[str], model_name: str, filename: str
) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(14, 12))
    image = ax.imshow(matrix, interpolation="nearest", cmap="Blues")
    fig.colorbar(image, ax=ax, label="Number of samples")
    ax.set(
        title=f"{model_name} Confusion Matrix",
        xlabel="Predicted crop",
        ylabel="Actual crop",
        xticks=np.arange(len(class_labels)),
        yticks=np.arange(len(class_labels)),
        xticklabels=class_labels,
        yticklabels=class_labels,
    )
    plt.setp(ax.get_xticklabels(), rotation=90, ha="center")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / filename, dpi=300)
    plt.close(fig)


def evaluate_model(
    model,
    model_name: str,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    class_labels: list[str],
    figure_filename: str,
) -> tuple[dict[str, float], str, np.ndarray]:
    predictions = model.predict(X_test)
    metrics = {
        "Accuracy": accuracy_score(y_test, predictions),
        "Macro precision": precision_score(
            y_test, predictions, labels=class_labels, average="macro", zero_division=0
        ),
        "Macro recall": recall_score(
            y_test, predictions, labels=class_labels, average="macro", zero_division=0
        ),
        "Macro-F1": f1_score(
            y_test, predictions, labels=class_labels, average="macro", zero_division=0
        ),
    }
    report = classification_report(
        y_test,
        predictions,
        labels=class_labels,
        target_names=class_labels,
        zero_division=0,
    )
    matrix = confusion_matrix(y_test, predictions, labels=class_labels)
    save_confusion_matrix(matrix, class_labels, model_name, figure_filename)
    return metrics, report, matrix


def main() -> None:
    X, y = load_and_validate_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    class_labels = sorted(y.unique().tolist())

    dummy_model = DummyClassifier(strategy="most_frequent")
    random_forest_model = RandomForestClassifier(
        n_estimators=100,
        random_state=RANDOM_STATE,
        n_jobs=1,
    )

    dummy_model.fit(X_train, y_train)
    random_forest_model.fit(X_train, y_train)

    evaluations = {}
    for model, name, filename in [
        (dummy_model, "DummyClassifier", "dummy_confusion_matrix.png"),
        (
            random_forest_model,
            "RandomForestClassifier",
            "random_forest_confusion_matrix.png",
        ),
    ]:
        metrics, report, matrix = evaluate_model(
            model, name, X_test, y_test, class_labels, filename
        )
        evaluations[name] = (metrics, report, matrix)

    report_lines = [
        "Crop Recommendation System - Baseline Evaluation",
        "=" * 54,
        f"Dataset: {DATA_PATH.relative_to(ROOT)}",
        f"Rows: {len(X)}",
        f"Train samples: {len(X_train)}",
        f"Test samples: {len(X_test)}",
        f"Features: {FEATURES}",
        f"Class labels: {class_labels}",
        f"Split: test_size={TEST_SIZE}, random_state={RANDOM_STATE}, stratified",
        "Preprocessing: none required; both estimators use the validated numeric features directly.",
        "Model selection/tuning: not performed; the held-out test set is for evaluation only.",
        "",
        "Held-out test metrics:",
    ]
    for model_name, (metrics, class_report, matrix) in evaluations.items():
        report_lines.append("")
        report_lines.append(model_name)
        report_lines.append("-" * len(model_name))
        report_lines.extend(f"{name}: {value:.4f}" for name, value in metrics.items())
        report_lines.append("")
        report_lines.append("Classification report:")
        report_lines.append(class_report.rstrip())
        report_lines.append("")
        report_lines.append("Confusion matrix (rows=actual, columns=predicted):")
        report_lines.append(
            pd.DataFrame(matrix, index=class_labels, columns=class_labels).to_string()
        )
    report_lines.extend(
        [
            "",
            "Interpretation note:",
            "These scores describe performance on this dataset's held-out split only. "
            "They do not establish suitability for real farms or generalization to "
            "other locations or growing conditions.",
        ]
    )

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    joblib.dump(
        {
            "model": random_forest_model,
            "feature_names": FEATURES,
            "class_labels": class_labels,
            "random_state": RANDOM_STATE,
            "test_size": TEST_SIZE,
        },
        MODEL_PATH,
    )

    print("\n".join(report_lines))
    print(f"\nSaved evaluation report to: {REPORT_PATH}")
    print(f"Saved confusion matrix plots to: {FIGURES_DIR}")
    print(f"Saved Random Forest model and metadata to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
