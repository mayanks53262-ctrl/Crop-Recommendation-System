import hashlib
from pathlib import Path
import time

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.base import clone
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "raw" / "Crop_recommendation.csv"
SPLIT_PATH = ROOT / "reports" / "baseline_split_indices.csv"
CV_CSV_PATH = ROOT / "reports" / "model_comparison_cv.csv"
CV_REPORT_PATH = ROOT / "reports" / "model_comparison_cv.txt"
CV_PLOT_PATH = ROOT / "reports" / "figures" / "model_comparison_cv.png"
TEST_REPORT_PATH = ROOT / "reports" / "model_comparison_test.txt"
TEST_CONFUSION_PATH = ROOT / "reports" / "figures" / "model_comparison_test_confusion_matrix.png"
SELECTED_MODEL_PATH = ROOT / "models" / "selected_model.joblib"

FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
TARGET = "label"
RANDOM_STATE = 42
TEST_SIZE = 0.20
N_SPLITS = 5
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
    if not DATA_PATH.is_file():
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")

    data = pd.read_csv(DATA_PATH)
    required_columns = FEATURES + [TARGET]
    missing_columns = [
        column for column in required_columns if column not in data.columns
    ]
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {missing_columns}")
    if data.empty:
        raise ValueError("Dataset has no rows.")
    if data[required_columns].isna().any().any():
        missing = data[required_columns].isna().sum()
        raise ValueError(
            f"Dataset has missing required values: {missing[missing > 0].to_dict()}"
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
        raise ValueError("Dataset crop classes do not match the expected crop labels.")

    return X, y


def load_verified_split(y: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    if not SPLIT_PATH.is_file():
        raise FileNotFoundError(
            f"Expected baseline split manifest not found at {SPLIT_PATH}"
        )

    manifest = pd.read_csv(SPLIT_PATH)
    if manifest.columns.tolist() != ["row_index", "partition", "crop"]:
        raise ValueError("Baseline split manifest has an unexpected schema.")
    if len(manifest) != len(y) or manifest["row_index"].duplicated().any():
        raise ValueError("Baseline split manifest row count or row IDs are invalid.")

    manifest = manifest.sort_values("row_index").reset_index(drop=True)
    expected_indices = np.arange(len(y))
    if not np.array_equal(manifest["row_index"].to_numpy(), expected_indices):
        raise ValueError("Baseline split manifest does not cover current dataset rows.")
    if not manifest["crop"].equals(y.reset_index(drop=True).rename("crop")):
        raise ValueError(
            "Baseline split manifest crop labels do not match the current dataset."
        )

    train_indices, test_indices = train_test_split(
        expected_indices,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    expected_partition = np.full(len(y), "train", dtype=object)
    expected_partition[test_indices] = "test"
    if not np.array_equal(manifest["partition"].to_numpy(), expected_partition):
        raise ValueError(
            "Baseline split manifest does not match the prescribed stratified split."
        )
    return train_indices, test_indices


def build_candidates() -> dict[str, Pipeline]:
    return {
        "Logistic Regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        C=1.0,
                        solver="lbfgs",
                        max_iter=2000,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "Random Forest": Pipeline(
            [
                ("scaler", "passthrough"),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=100,
                        criterion="gini",
                        max_depth=None,
                        min_samples_split=2,
                        min_samples_leaf=1,
                        random_state=RANDOM_STATE,
                        n_jobs=1,
                    ),
                ),
            ]
        ),
        "Support Vector Machine": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    SVC(
                        C=1.0,
                        kernel="rbf",
                        gamma="scale",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
    }


def run_cross_validation(
    candidates: dict[str, Pipeline],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    folds: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[pd.DataFrame, dict[str, dict[str, object]]]:
    rows: list[dict[str, object]] = []
    summaries: dict[str, dict[str, object]] = {}

    for model_name, estimator in candidates.items():
        accuracy_scores: list[float] = []
        macro_f1_scores: list[float] = []
        fit_times: list[float] = []

        for fold_number, (fold_train, fold_validation) in enumerate(folds, start=1):
            fold_model = clone(estimator)
            start = time.perf_counter()
            fold_model.fit(X_train.iloc[fold_train], y_train.iloc[fold_train])
            fit_time = time.perf_counter() - start
            predictions = fold_model.predict(X_train.iloc[fold_validation])
            accuracy = accuracy_score(y_train.iloc[fold_validation], predictions)
            macro_f1 = f1_score(
                y_train.iloc[fold_validation],
                predictions,
                labels=CLASS_LABELS,
                average="macro",
                zero_division=0,
            )
            accuracy_scores.append(float(accuracy))
            macro_f1_scores.append(float(macro_f1))
            fit_times.append(float(fit_time))
            rows.append(
                {
                    "model": model_name,
                    "result_type": "fold",
                    "fold": fold_number,
                    "accuracy": accuracy,
                    "macro_f1": macro_f1,
                    "fit_time_seconds": fit_time,
                }
            )

        summary = {
            "accuracy_mean": float(np.mean(accuracy_scores)),
            "accuracy_std": float(np.std(accuracy_scores, ddof=1)),
            "macro_f1_mean": float(np.mean(macro_f1_scores)),
            "macro_f1_std": float(np.std(macro_f1_scores, ddof=1)),
            "fit_time_mean_seconds": float(np.mean(fit_times)),
            "fit_time_std_seconds": float(np.std(fit_times, ddof=1)),
        }
        summaries[model_name] = {
            **summary,
            "accuracy_folds": accuracy_scores,
            "macro_f1_folds": macro_f1_scores,
            "fit_time_folds": fit_times,
        }
        rows.append(
            {
                "model": model_name,
                "result_type": "mean",
                "fold": "",
                "accuracy": summary["accuracy_mean"],
                "macro_f1": summary["macro_f1_mean"],
                "fit_time_seconds": summary["fit_time_mean_seconds"],
            }
        )
        rows.append(
            {
                "model": model_name,
                "result_type": "std",
                "fold": "",
                "accuracy": summary["accuracy_std"],
                "macro_f1": summary["macro_f1_std"],
                "fit_time_seconds": summary["fit_time_std_seconds"],
            }
        )

    return pd.DataFrame(rows), summaries


def select_model(summaries: dict[str, dict[str, object]]) -> str:
    # Highest mean CV macro-F1 wins; ties prefer lower fold variability, then faster fit.
    return max(
        summaries,
        key=lambda name: (
            summaries[name]["macro_f1_mean"],
            -summaries[name]["macro_f1_std"],
            -summaries[name]["fit_time_mean_seconds"],
        ),
    )


def save_cv_plot(summaries: dict[str, dict[str, object]], selected_name: str) -> None:
    CV_PLOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    model_names = list(summaries)
    means = [summaries[name]["macro_f1_mean"] for name in model_names]
    standard_deviations = [summaries[name]["macro_f1_std"] for name in model_names]
    colors = [
        "seagreen" if name == selected_name else "steelblue" for name in model_names
    ]

    fig, ax = plt.subplots(figsize=(10, 6))
    positions = np.arange(len(model_names))
    ax.bar(
        positions,
        means,
        yerr=standard_deviations,
        capsize=6,
        color=colors,
        edgecolor="black",
    )
    ax.set(
        title="Model Comparison: 5-Fold Training CV Macro-F1",
        ylabel="Macro-F1 (mean ± sample standard deviation)",
        xticks=positions,
        xticklabels=model_names,
        ylim=(max(0.0, min(means) - 0.08), 1.02),
    )
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(CV_PLOT_PATH, dpi=300)
    plt.close(fig)


def save_test_confusion_matrix(matrix: np.ndarray, model_name: str) -> None:
    TEST_CONFUSION_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(14, 12))
    image = ax.imshow(matrix, interpolation="nearest", cmap="Blues")
    fig.colorbar(image, ax=ax, label="Number of samples")
    ax.set(
        title=f"Selected Model Held-Out Test Confusion Matrix ({model_name})",
        xlabel="Predicted crop",
        ylabel="Actual crop",
        xticks=np.arange(len(CLASS_LABELS)),
        yticks=np.arange(len(CLASS_LABELS)),
        xticklabels=CLASS_LABELS,
        yticklabels=CLASS_LABELS,
    )
    plt.setp(ax.get_xticklabels(), rotation=90, ha="center")
    fig.tight_layout()
    fig.savefig(TEST_CONFUSION_PATH, dpi=300)
    plt.close(fig)


def write_reports(
    summaries: dict[str, dict[str, object]],
    selected_name: str,
    test_metrics: dict[str, float],
    test_classification_report: str,
    test_confusion_matrix: np.ndarray,
    dataset_sha256: str,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
) -> None:
    lines = [
        "Crop Recommendation System - Model Comparison",
        "=" * 52,
        f"Dataset: {DATA_PATH.relative_to(ROOT)}",
        f"Dataset SHA-256: {dataset_sha256}",
        f"Rows: {len(train_indices) + len(test_indices)}",
        f"Training rows: {len(train_indices)}",
        f"Held-out test rows: {len(test_indices)}",
        "Split: stratified train_test_split(test_size=0.20, random_state=42)",
        "Cross-validation: StratifiedKFold(n_splits=5, shuffle=True, random_state=42)",
        "All candidates use the same five CV folds from the training partition.",
        "The held-out test partition is not used in CV or model selection.",
        "",
        "Candidate configurations:",
        "- Logistic Regression: StandardScaler + LogisticRegression(C=1.0, solver='lbfgs', max_iter=2000, random_state=42).",
        "- Random Forest: passthrough + RandomForestClassifier(n_estimators=100, criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1, random_state=42, n_jobs=1).",
        "- Support Vector Machine: StandardScaler + SVC(C=1.0, kernel='rbf', gamma='scale', random_state=42); probability estimates are disabled by default.",
        "Scaler fitting occurs within each Pipeline training fold.",
        "",
        "Cross-validation results (mean ± sample standard deviation):",
        "Model | Accuracy | Macro-F1 | Mean fit time (s) | Fit-time std (s)",
    ]
    for name, summary in summaries.items():
        lines.append(
            f"{name} | {summary['accuracy_mean']:.4f} ± {summary['accuracy_std']:.4f} "
            f"| {summary['macro_f1_mean']:.4f} ± {summary['macro_f1_std']:.4f} "
            f"| {summary['fit_time_mean_seconds']:.4f} "
            f"| {summary['fit_time_std_seconds']:.4f}"
        )
        lines.append(
            f"  Fold macro-F1: "
            + ", ".join(f"{value:.4f}" for value in summary["macro_f1_folds"])
        )
        lines.append(
            f"  Fold accuracy: "
            + ", ".join(f"{value:.4f}" for value in summary["accuracy_folds"])
        )

    lines.extend(
        [
            "",
            "Selection rule:",
            "Select the highest mean CV macro-F1; for exact ties, prefer lower CV "
            "macro-F1 standard deviation, then lower mean fit time. This rule uses "
            "training-partition cross-validation only.",
            f"Selected candidate: {selected_name}",
            "",
            "Limitations:",
            "These benchmark results are measured on this dataset and split. High "
            "scores do not prove performance on real farms, in different regions, "
            "or under different growing conditions. No hyperparameter tuning was "
            "performed.",
        ]
    )
    CV_REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")

    test_lines = [
        "Selected Model - Held-Out Test Evaluation",
        "=" * 43,
        f"Selected by training CV: {selected_name}",
        f"Test rows: {len(test_indices)}",
        "The test partition was evaluated once, after candidate selection.",
        f"Accuracy: {test_metrics['accuracy']:.4f}",
        f"Macro-F1: {test_metrics['macro_f1']:.4f}",
        "",
        "Classification report:",
        test_classification_report.rstrip(),
        "",
        "Confusion matrix (rows=actual, columns=predicted):",
        pd.DataFrame(
            test_confusion_matrix, index=CLASS_LABELS, columns=CLASS_LABELS
        ).to_string(),
        "",
        "These test scores describe this held-out portion of the dataset only; "
        "they do not prove generalization to real farms or different regions.",
    ]
    TEST_REPORT_PATH.write_text("\n".join(test_lines) + "\n", encoding="utf-8")


def main() -> None:
    X, y = load_and_validate_data()
    train_indices, test_indices = load_verified_split(y)
    X_train, y_train = X.iloc[train_indices], y.iloc[train_indices]
    X_test, y_test = X.iloc[test_indices], y.iloc[test_indices]

    fold_splitter = StratifiedKFold(
        n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE
    )
    folds = list(fold_splitter.split(X_train, y_train))
    candidates = build_candidates()
    cv_results, summaries = run_cross_validation(
        candidates, X_train, y_train, folds
    )
    selected_name = select_model(summaries)

    selected_model = candidates[selected_name]
    selected_model.fit(X_train, y_train)
    predictions = selected_model.predict(X_test)
    test_metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "macro_f1": float(
            f1_score(
                y_test,
                predictions,
                labels=CLASS_LABELS,
                average="macro",
                zero_division=0,
            )
        ),
    }
    test_class_report = classification_report(
        y_test,
        predictions,
        labels=CLASS_LABELS,
        target_names=CLASS_LABELS,
        zero_division=0,
    )
    test_matrix = confusion_matrix(y_test, predictions, labels=CLASS_LABELS)

    dataset_sha256 = hashlib.sha256(DATA_PATH.read_bytes()).hexdigest()
    split_configuration = {
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE,
        "stratified": True,
        "train_row_indices": train_indices.tolist(),
        "test_row_indices": test_indices.tolist(),
        "row_index_basis": "zero-based data-row positions, excluding CSV header",
        "split_manifest": str(SPLIT_PATH.relative_to(ROOT)),
    }
    cv_configuration = {
        "n_splits": N_SPLITS,
        "shuffle": True,
        "random_state": RANDOM_STATE,
        "selection_metric": "mean macro-F1",
        "tie_breakers": [
            "lower sample standard deviation of CV macro-F1",
            "lower mean CV fit time",
        ],
    }
    artifact = {
        "model": selected_model,
        "feature_names": FEATURES,
        "class_labels": CLASS_LABELS,
        "model_name": selected_name,
        "random_seed": RANDOM_STATE,
        "split_configuration": split_configuration,
        "cv_configuration": cv_configuration,
        "dataset_sha256": dataset_sha256,
        "dataset_path": str(DATA_PATH.relative_to(ROOT)),
        "training_fit_rows": len(train_indices),
        "held_out_test_rows": len(test_indices),
        "cv_results": {
            name: {
                key: value
                for key, value in summary.items()
                if not key.endswith("_folds")
            }
            for name, summary in summaries.items()
        },
        "selected_model_test_metrics": test_metrics,
    }

    CV_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    cv_results.to_csv(CV_CSV_PATH, index=False)
    CV_PLOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    save_cv_plot(summaries, selected_name)
    save_test_confusion_matrix(test_matrix, selected_name)
    write_reports(
        summaries,
        selected_name,
        test_metrics,
        test_class_report,
        test_matrix,
        dataset_sha256,
        train_indices,
        test_indices,
    )
    SELECTED_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, SELECTED_MODEL_PATH)

    print("\nModel comparison: 5-fold stratified CV on training partition")
    for name, summary in summaries.items():
        print(
            f"{name}: accuracy={summary['accuracy_mean']:.4f} "
            f"+/- {summary['accuracy_std']:.4f}; "
            f"macro-F1={summary['macro_f1_mean']:.4f} "
            f"+/- {summary['macro_f1_std']:.4f}; "
            f"mean fit time={summary['fit_time_mean_seconds']:.4f}s "
            f"(std={summary['fit_time_std_seconds']:.4f}s)"
        )
        print(
            "  Fold macro-F1: "
            + ", ".join(f"{value:.4f}" for value in summary["macro_f1_folds"])
        )
    print(f"\nSelected by CV: {selected_name}")
    print(f"Held-out test accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Held-out test macro-F1: {test_metrics['macro_f1']:.4f}")
    print("\nHeld-out classification report:")
    print(test_class_report)
    print("Held-out confusion matrix (rows=actual, columns=predicted):")
    print(pd.DataFrame(test_matrix, index=CLASS_LABELS, columns=CLASS_LABELS))
    print(f"\nSaved CV results CSV: {CV_CSV_PATH}")
    print(f"Saved CV summary: {CV_REPORT_PATH}")
    print(f"Saved CV comparison plot: {CV_PLOT_PATH}")
    print(f"Saved held-out test report: {TEST_REPORT_PATH}")
    print(f"Saved held-out confusion plot: {TEST_CONFUSION_PATH}")
    print(f"Saved selected model and metadata: {SELECTED_MODEL_PATH}")


if __name__ == "__main__":
    main()
