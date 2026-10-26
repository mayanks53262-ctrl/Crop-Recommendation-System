from __future__ import annotations

from itertools import combinations
import hashlib
from pathlib import Path
import time
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import t
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

if __package__:
    from .compare_models import (
        CLASS_LABELS,
        FEATURES,
        RANDOM_STATE,
        TEST_SIZE,
        build_candidates,
        load_and_validate_data,
        load_verified_split,
    )
else:
    from compare_models import (
        CLASS_LABELS,
        FEATURES,
        RANDOM_STATE,
        TEST_SIZE,
        build_candidates,
        load_and_validate_data,
        load_verified_split,
    )

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "raw" / "Crop_recommendation.csv"
MODEL_PATH = ROOT / "models" / "selected_model.joblib"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
REPEATED_CV_PATH = REPORTS_DIR / "repeated_cv_results.csv"
CALIBRATION_PATH = REPORTS_DIR / "calibration_metrics.csv"
NEAR_DUPLICATES_PATH = REPORTS_DIR / "near_duplicate_pairs.csv"
ANALYSIS_REPORT_PATH = REPORTS_DIR / "reliability_analysis.txt"
REPEATED_CV_PLOT_PATH = FIGURES_DIR / "repeated_cv_macro_f1.png"
CALIBRATION_PLOT_PATH = FIGURES_DIR / "calibration_reliability.png"

N_SPLITS = 5
N_REPEATS = 10
REPEAT_SEEDS = tuple(range(42, 42 + N_REPEATS))
CALIBRATION_SEED_BASE = 1042
CALIBRATION_INNER_SPLITS = 3
CALIBRATION_BINS = 10
NEAR_DUPLICATE_THRESHOLD = 0.25
NEAR_DUPLICATE_SENSITIVITY_THRESHOLDS = (0.10, 0.25, 0.50)


def corrected_repeated_cv_interval(
    scores: list[float] | np.ndarray,
    validation_to_training_ratio: float,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Return a corrected resampled-t interval for repeated-CV fold scores."""
    values = np.asarray(scores, dtype=float)
    if values.ndim != 1 or values.size < 2 or not np.isfinite(values).all():
        raise ValueError("At least two finite fold scores are required.")
    if validation_to_training_ratio < 0:
        raise ValueError("The validation-to-training ratio cannot be negative.")
    if not 0 < confidence < 1:
        raise ValueError("Confidence must be between 0 and 1.")

    mean = float(values.mean())
    variance = float(values.var(ddof=1))
    corrected_standard_error = np.sqrt(
        (1.0 / values.size + validation_to_training_ratio) * variance
    )
    critical_value = float(t.ppf((1.0 + confidence) / 2.0, df=values.size - 1))
    margin = critical_value * corrected_standard_error
    return mean - margin, mean + margin


def _expected_calibration_error(
    probabilities: np.ndarray,
    observed_frequency: np.ndarray,
    n_bins: int,
) -> tuple[float, list[dict[str, float | int]]]:
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_indices = np.minimum(
        np.searchsorted(edges, probabilities, side="right") - 1,
        n_bins - 1,
    )
    points: list[dict[str, float | int]] = []
    total_error = 0.0
    for bin_index in range(n_bins):
        mask = bin_indices == bin_index
        count = int(mask.sum())
        if count == 0:
            continue
        mean_probability = float(probabilities[mask].mean())
        observed = float(observed_frequency[mask].mean())
        total_error += count / probabilities.size * abs(mean_probability - observed)
        points.append(
            {
                "bin": bin_index,
                "count": count,
                "mean_probability": mean_probability,
                "observed_frequency": observed,
            }
        )
    return float(total_error), points


def multiclass_calibration_metrics(
    y_true: pd.Series | np.ndarray,
    probabilities: np.ndarray,
    class_labels: list[str],
    n_bins: int = CALIBRATION_BINS,
) -> tuple[dict[str, float | int], dict[str, list[dict[str, float | int]]]]:
    """Measure multiclass score calibration and return reliability-plot bins."""
    labels = np.asarray(y_true)
    scores = np.asarray(probabilities, dtype=float)
    if scores.shape != (labels.size, len(class_labels)):
        raise ValueError("Probability dimensions do not match labels and classes.")
    if not np.isfinite(scores).all() or (scores < 0).any() or (scores > 1).any():
        raise ValueError("Calibration scores must be finite and in [0, 1].")
    if not np.allclose(scores.sum(axis=1), 1.0, atol=1e-6):
        raise ValueError("Each multiclass score row must sum approximately to 1.")
    if not set(labels).issubset(class_labels):
        raise ValueError("Observed labels must be present in class_labels.")
    if n_bins < 1:
        raise ValueError("n_bins must be at least 1.")

    class_to_index = {label: index for index, label in enumerate(class_labels)}
    true_indices = np.asarray([class_to_index[label] for label in labels])
    predicted_indices = scores.argmax(axis=1)
    one_hot = np.eye(len(class_labels), dtype=float)[true_indices]
    top_confidence = scores.max(axis=1)
    top_correct = (predicted_indices == true_indices).astype(float)
    top_ece, top_bins = _expected_calibration_error(
        top_confidence, top_correct, n_bins
    )

    classwise_eces: list[float] = []
    for class_index in range(len(class_labels)):
        class_ece, _ = _expected_calibration_error(
            scores[:, class_index], one_hot[:, class_index], n_bins
        )
        classwise_eces.append(class_ece)
    classwise_ece, classwise_bins = _expected_calibration_error(
        scores.ravel(), one_hot.ravel(), n_bins
    )

    metrics: dict[str, float | int] = {
        "n_samples": int(labels.size),
        "accuracy": float(accuracy_score(labels, np.asarray(class_labels)[predicted_indices])),
        "macro_f1": float(
            f1_score(
                labels,
                np.asarray(class_labels)[predicted_indices],
                labels=class_labels,
                average="macro",
                zero_division=0,
            )
        ),
        "multiclass_log_loss": float(
            log_loss(labels, scores, labels=class_labels)
        ),
        "multiclass_brier_score": float(np.mean(np.sum((scores - one_hot) ** 2, axis=1))),
        "top_label_ece": top_ece,
        "classwise_macro_ece": float(np.mean(classwise_eces)),
        "classwise_micro_ece": classwise_ece,
        "calibration_bins": n_bins,
    }
    return metrics, {"top_label": top_bins, "classwise_ovr": classwise_bins}


def investigate_similar_rows(
    X: pd.DataFrame,
    y: pd.Series,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    threshold: float = NEAR_DUPLICATE_THRESHOLD,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Find exact feature duplicates and standardized-distance near pairs."""
    if threshold <= 0:
        raise ValueError("The near-duplicate threshold must be positive.")
    if len(X) != len(y):
        raise ValueError("Feature and target row counts must match.")
    train_set = set(np.asarray(train_indices, dtype=int).tolist())
    test_set = set(np.asarray(test_indices, dtype=int).tolist())
    if train_set & test_set or train_set | test_set != set(range(len(X))):
        raise ValueError("Train/test indices must partition all dataset rows.")

    partition = np.full(len(X), "test", dtype=object)
    partition[list(train_set)] = "train"
    pair_records: list[dict[str, Any]] = []
    exact_group_count = 0
    exact_affected_rows: set[int] = set()
    exact_cross_partition_pairs = 0
    exact_conflicting_label_groups = 0

    grouped = X.groupby(FEATURES, sort=False, dropna=False).indices
    for group_positions in grouped.values():
        row_indices = [int(index) for index in group_positions]
        if len(row_indices) < 2:
            continue
        exact_group_count += 1
        exact_affected_rows.update(row_indices)
        if y.iloc[row_indices].nunique() > 1:
            exact_conflicting_label_groups += 1
        for left, right in combinations(row_indices, 2):
            crosses_partition = partition[left] != partition[right]
            exact_cross_partition_pairs += int(crosses_partition)
            pair_records.append(
                {
                    "similarity_type": "exact_feature_match",
                    "row_index_1": left,
                    "row_index_2": right,
                    "standardized_euclidean_distance": 0.0,
                    "label_1": y.iloc[left],
                    "label_2": y.iloc[right],
                    "same_label": bool(y.iloc[left] == y.iloc[right]),
                    "partition_1": partition[left],
                    "partition_2": partition[right],
                    "crosses_train_test": bool(crosses_partition),
                }
            )

    scaler = StandardScaler().fit(X.iloc[np.asarray(train_indices, dtype=int)])
    standardized = scaler.transform(X)
    tree = cKDTree(standardized)
    neighbor_distances, _ = tree.query(standardized, k=2)
    nearest_distances = neighbor_distances[:, 1]
    max_threshold = max(NEAR_DUPLICATE_SENSITIVITY_THRESHOLDS + (threshold,))
    candidate_pairs = tree.query_pairs(max_threshold, output_type="ndarray")
    near_records: list[dict[str, Any]] = []
    pair_distance_records: list[tuple[float, int, int]] = []
    sensitivity_counts = {value: 0 for value in NEAR_DUPLICATE_SENSITIVITY_THRESHOLDS}

    for left_value, right_value in candidate_pairs:
        left, right = int(left_value), int(right_value)
        distance = float(np.linalg.norm(standardized[left] - standardized[right]))
        if distance <= np.finfo(float).eps:
            continue
        pair_distance_records.append((distance, left, right))
        for radius in sensitivity_counts:
            sensitivity_counts[radius] += int(distance <= radius)
        if distance > threshold:
            continue
        crosses_partition = partition[left] != partition[right]
        near_records.append(
            {
                "similarity_type": "near_feature_match",
                "row_index_1": left,
                "row_index_2": right,
                "standardized_euclidean_distance": distance,
                "label_1": y.iloc[left],
                "label_2": y.iloc[right],
                "same_label": bool(y.iloc[left] == y.iloc[right]),
                "partition_1": partition[left],
                "partition_2": partition[right],
                "crosses_train_test": bool(crosses_partition),
            }
        )

    pair_records.extend(near_records)
    columns = [
        "similarity_type",
        "row_index_1",
        "row_index_2",
        "standardized_euclidean_distance",
        "label_1",
        "label_2",
        "same_label",
        "partition_1",
        "partition_2",
        "crosses_train_test",
    ]
    pairs = pd.DataFrame(pair_records, columns=columns)
    if not pairs.empty:
        pairs = pairs.sort_values(
            ["similarity_type", "standardized_euclidean_distance"]
        ).reset_index(drop=True)

    near_affected = {
        int(row)
        for _, left, right in pair_distance_records
        if np.linalg.norm(standardized[left] - standardized[right]) <= threshold
        for row in (left, right)
    }
    near_cross_partition = sum(
        int(record["crosses_train_test"]) for record in near_records
    )
    near_label_agreement = (
        float(np.mean([record["same_label"] for record in near_records]))
        if near_records
        else float("nan")
    )
    summary: dict[str, Any] = {
        "rows_examined": len(X),
        "exact_duplicate_feature_groups": exact_group_count,
        "exact_duplicate_affected_rows": len(exact_affected_rows),
        "exact_duplicate_unordered_pairs": sum(
            len(group) * (len(group) - 1) // 2
            for group in grouped.values()
            if len(group) > 1
        ),
        "exact_pairs_crossing_train_test": exact_cross_partition_pairs,
        "exact_groups_with_conflicting_labels": exact_conflicting_label_groups,
        "near_similarity_threshold": threshold,
        "near_duplicate_pairs": len(near_records),
        "near_duplicate_affected_rows": len(near_affected),
        "near_pairs_crossing_train_test": near_cross_partition,
        "near_pair_label_agreement_rate": near_label_agreement,
        "nearest_neighbor_distance_min": float(nearest_distances.min()),
        "nearest_neighbor_distance_p10": float(np.quantile(nearest_distances, 0.10)),
        "nearest_neighbor_distance_median": float(np.median(nearest_distances)),
        "near_duplicate_sensitivity_pair_counts": sensitivity_counts,
        "distance_standardization": (
            "Euclidean distance after StandardScaler fitted only on the training partition."
        ),
    }
    return pairs, summary


def _aligned_predict_proba(
    estimator: Any,
    X: pd.DataFrame,
    class_labels: list[str],
) -> np.ndarray:
    raw = np.asarray(estimator.predict_proba(X), dtype=float)
    estimator_classes = list(estimator.classes_)
    if raw.shape != (len(X), len(estimator_classes)):
        raise ValueError("Estimator returned an unexpected probability matrix shape.")
    if set(estimator_classes) != set(class_labels):
        raise ValueError("Estimator classes do not match selected-model metadata.")
    class_order = [estimator_classes.index(label) for label in class_labels]
    return raw[:, class_order]


def build_repeated_fold_specs(y_train: pd.Series) -> list[dict[str, Any]]:
    """Build a deterministic shared outer-fold schedule for every candidate."""
    fold_specs: list[dict[str, Any]] = []
    row_indices = np.arange(len(y_train))
    for repeat, seed in enumerate(REPEAT_SEEDS):
        splitter = StratifiedKFold(
            n_splits=N_SPLITS, shuffle=True, random_state=seed
        )
        for fold, (fit_indices, validation_indices) in enumerate(
            splitter.split(row_indices, y_train), start=1
        ):
            fold_specs.append(
                {
                    "repeat": repeat + 1,
                    "repeat_seed": seed,
                    "fold": fold,
                    "fit_indices": fit_indices,
                    "validation_indices": validation_indices,
                }
            )
    return fold_specs


def run_repeated_evaluation(
    X_train: pd.DataFrame, y_train: pd.Series, class_labels: list[str]
) -> tuple[pd.DataFrame, dict[str, np.ndarray], list[dict[str, Any]]]:
    candidates = build_candidates()
    if "Random Forest" not in candidates:
        raise ValueError("Existing candidate configuration has no Random Forest.")
    random_forest = candidates["Random Forest"]
    models = dict(candidates)
    models["Random Forest (sigmoid calibrated)"] = None

    n_samples = len(X_train)
    probability_sums = {
        "Random Forest": np.zeros((n_samples, len(class_labels)), dtype=float),
        "Random Forest (sigmoid calibrated)": np.zeros(
            (n_samples, len(class_labels)), dtype=float
        ),
    }
    probability_counts = np.zeros(n_samples, dtype=int)
    records: list[dict[str, Any]] = []
    fold_specs = build_repeated_fold_specs(y_train)

    for fold_spec in fold_specs:
        repeat = fold_spec["repeat"]
        fold = fold_spec["fold"]
        fit_indices = fold_spec["fit_indices"]
        validation_indices = fold_spec["validation_indices"]
        X_fit, y_fit = X_train.iloc[fit_indices], y_train.iloc[fit_indices]
        X_validation = X_train.iloc[validation_indices]
        y_validation = y_train.iloc[validation_indices]

        for model_name, candidate in models.items():
            if model_name == "Random Forest (sigmoid calibrated)":
                inner_seed = CALIBRATION_SEED_BASE + (repeat - 1) * N_SPLITS + fold
                inner_cv = StratifiedKFold(
                    n_splits=CALIBRATION_INNER_SPLITS,
                    shuffle=True,
                    random_state=inner_seed,
                )
                estimator = CalibratedClassifierCV(
                    estimator=clone(random_forest),
                    method="sigmoid",
                    cv=inner_cv,
                    ensemble=True,
                    n_jobs=1,
                )
            else:
                estimator = clone(candidate)

            start_time = time.perf_counter()
            estimator.fit(X_fit, y_fit)
            fit_seconds = time.perf_counter() - start_time
            predictions = estimator.predict(X_validation)
            accuracy = float(accuracy_score(y_validation, predictions))
            macro_f1 = float(
                f1_score(
                    y_validation,
                    predictions,
                    labels=class_labels,
                    average="macro",
                    zero_division=0,
                )
            )
            records.append(
                {
                    "model": model_name,
                    "result_type": "fold",
                    "repeat": repeat,
                    "repeat_seed": fold_spec["repeat_seed"],
                    "fold": fold,
                    "accuracy": accuracy,
                    "macro_f1": macro_f1,
                    "fit_time_seconds": fit_seconds,
                    "accuracy_ci_lower": np.nan,
                    "accuracy_ci_upper": np.nan,
                    "macro_f1_ci_lower": np.nan,
                    "macro_f1_ci_upper": np.nan,
                }
            )

            if model_name in probability_sums:
                probabilities = _aligned_predict_proba(
                    estimator, X_validation, class_labels
                )
                probability_sums[model_name][validation_indices] += probabilities

        probability_counts[validation_indices] += 1

    if not np.all(probability_counts == N_REPEATS):
        raise RuntimeError("Repeated CV did not produce one held-out prediction per row and repeat.")

    fold_results = pd.DataFrame(records)
    summary_rows: list[dict[str, Any]] = []
    validation_to_training_ratio = 1.0 / (N_SPLITS - 1)
    for model_name, model_rows in fold_results.groupby("model", sort=False):
        summary: dict[str, Any] = {
            "model": model_name,
            "result_type": "summary_mean",
            "repeat": np.nan,
            "repeat_seed": np.nan,
            "fold": np.nan,
            "fit_time_seconds": float(model_rows["fit_time_seconds"].mean()),
        }
        std_row: dict[str, Any] = {
            "model": model_name,
            "result_type": "summary_std",
            "repeat": np.nan,
            "repeat_seed": np.nan,
            "fold": np.nan,
            "fit_time_seconds": float(model_rows["fit_time_seconds"].std(ddof=1)),
        }
        interval_row: dict[str, Any] = {
            "model": model_name,
            "result_type": "corrected_95_percent_CI",
            "repeat": np.nan,
            "repeat_seed": np.nan,
            "fold": np.nan,
            "fit_time_seconds": np.nan,
        }
        for metric in ("accuracy", "macro_f1"):
            scores = model_rows[metric].to_numpy()
            lower, upper = corrected_repeated_cv_interval(
                scores, validation_to_training_ratio
            )
            summary[metric] = float(scores.mean())
            std_row[metric] = float(scores.std(ddof=1))
            interval_row[f"{metric}_ci_lower"] = lower
            interval_row[f"{metric}_ci_upper"] = upper
            summary[f"{metric}_ci_lower"] = np.nan
            summary[f"{metric}_ci_upper"] = np.nan
            std_row[f"{metric}_ci_lower"] = np.nan
            std_row[f"{metric}_ci_upper"] = np.nan
        summary_rows.extend([summary, std_row, interval_row])

    result_table = pd.concat(
        [fold_results, pd.DataFrame(summary_rows)], ignore_index=True
    )
    averaged_probabilities = {
        model_name: probability_sums[model_name] / probability_counts[:, None]
        for model_name in probability_sums
    }
    return result_table, averaged_probabilities, fold_specs


def write_calibration_plot(
    reliability: dict[str, dict[str, list[dict[str, float | int]]]],
) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    colors = {
        "Random Forest": "steelblue",
        "Random Forest (sigmoid calibrated)": "darkorange",
    }
    for model_name, bins in reliability.items():
        top = bins["top_label"]
        axes[0].plot(
            [point["mean_probability"] for point in top],
            [point["observed_frequency"] for point in top],
            marker="o",
            label=model_name,
            color=colors[model_name],
        )
        classwise = bins["classwise_ovr"]
        axes[1].plot(
            [point["mean_probability"] for point in classwise],
            [point["observed_frequency"] for point in classwise],
            marker="o",
            label=model_name,
            color=colors[model_name],
        )
    for ax, title in zip(
        axes,
        ("Top-label reliability", "Classwise one-vs-rest reliability"),
    ):
        ax.plot([0, 1], [0, 1], linestyle="--", color="black", label="Ideal")
        ax.set(
            title=title,
            xlabel="Mean model score in bin",
            ylabel="Observed accuracy / frequency",
            xlim=(0, 1),
            ylim=(0, 1),
        )
        ax.grid(alpha=0.3)
    axes[0].legend(loc="best", fontsize="small")
    axes[1].legend(loc="best", fontsize="small")
    fig.suptitle("Training-Partition Repeated Out-of-Fold Calibration")
    fig.tight_layout()
    fig.savefig(CALIBRATION_PLOT_PATH, dpi=250)
    plt.close(fig)


def write_repeated_cv_plot(results: pd.DataFrame) -> None:
    fold_results = results[results["result_type"] == "fold"]
    model_names = fold_results["model"].drop_duplicates().tolist()
    means = [
        float(fold_results.loc[fold_results["model"] == model, "macro_f1"].mean())
        for model in model_names
    ]
    standard_deviations = [
        float(fold_results.loc[fold_results["model"] == model, "macro_f1"].std(ddof=1))
        for model in model_names
    ]
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11, 6))
    positions = np.arange(len(model_names))
    ax.bar(
        positions,
        means,
        yerr=standard_deviations,
        capsize=5,
        color=["darkorange" if "calibrated" in name else "steelblue" for name in model_names],
        edgecolor="black",
    )
    ax.set(
        title="Repeated Stratified 5-Fold CV: Macro-F1",
        ylabel="Mean macro-F1 ± sample standard deviation across folds",
        xticks=positions,
        xticklabels=model_names,
        ylim=(max(0.0, min(means) - 0.08), 1.02),
    )
    ax.tick_params(axis="x", labelrotation=12)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(REPEATED_CV_PLOT_PATH, dpi=250)
    plt.close(fig)


def write_analysis_report(
    results: pd.DataFrame,
    calibration_table: pd.DataFrame,
    duplicate_summary: dict[str, Any],
    dataset_sha256: str,
    train_count: int,
    test_count: int,
    threshold: float,
) -> None:
    fold_results = results[results["result_type"] == "fold"]
    lines = [
        "Crop Recommendation System - Phase 8B Reliability Analysis",
        "=" * 66,
        f"Dataset: {DATA_PATH.relative_to(ROOT)}",
        f"Dataset SHA-256: {dataset_sha256}",
        f"Partition: stratified train_test_split(test_size={TEST_SIZE:.2f}, random_state={RANDOM_STATE})",
        f"Training rows used for repeated CV and calibration: {train_count}",
        f"Held-out rows excluded from repeated CV, calibration, and model selection: {test_count}",
        "Existing saved split indices and split manifest were verified against the dataset.",
        "",
        "Repeated cross-validation",
        "-------------------------",
        f"Outer folds: StratifiedKFold(n_splits={N_SPLITS}, shuffle=True), repeated {N_REPEATS} times.",
        f"Repeat random_state values: {list(REPEAT_SEEDS)}.",
        "Each candidate uses the exact same fold indices within every repeat.",
        "Candidate configurations are reused from src/compare_models.py: Logistic Regression(StandardScaler, C=1, lbfgs, max_iter=2000, random_state=42), Random Forest(100 trees, gini, unlimited depth, min_samples_split=2, min_samples_leaf=1, random_state=42, n_jobs=1), and SVM(StandardScaler, RBF, C=1, gamma='scale', random_state=42).",
        "Sigmoid-calibrated Random Forest is an additional evaluation candidate; it is not automatically selected.",
        "Fold-level accuracy, macro-F1, and fit time are saved in repeated_cv_results.csv.",
        "Mean and sample standard deviation use the 50 fold scores for each candidate.",
        "Reported 95% intervals are corrected resampled-t intervals: mean +/- t(df=49) * sqrt((1/50 + 1/4) * sample_variance).",
        "The 1/4 term is the validation-to-fit row ratio in 5-fold CV. This correction acknowledges dependence among overlapping CV folds; intervals remain approximate and are not independent-sample guarantees.",
        "",
        "Repeated-CV summary (mean; sample SD; corrected 95% interval):",
        "Model | Accuracy | Macro-F1 | Mean fit seconds",
    ]
    for model_name in fold_results["model"].drop_duplicates():
        model_rows = fold_results[fold_results["model"] == model_name]
        summary_rows = results[results["model"] == model_name].set_index("result_type")
        mean_row = summary_rows.loc["summary_mean"]
        std_row = summary_rows.loc["summary_std"]
        ci_row = summary_rows.loc["corrected_95_percent_CI"]
        lines.append(
            f"{model_name} | {mean_row['accuracy']:.6f} ± {std_row['accuracy']:.6f} "
            f"[{ci_row['accuracy_ci_lower']:.6f}, {ci_row['accuracy_ci_upper']:.6f}] | "
            f"{mean_row['macro_f1']:.6f} ± {std_row['macro_f1']:.6f} "
            f"[{ci_row['macro_f1_ci_lower']:.6f}, {ci_row['macro_f1_ci_upper']:.6f}] | "
            f"{model_rows['fit_time_seconds'].mean():.4f}"
        )
    lines.extend(
        [
            "",
            "Calibration analysis",
            "---------------------",
            "The selected Random Forest is compared with CalibratedClassifierCV(method='sigmoid', ensemble=True).",
            "The Random Forest base settings remain 100 trees and random_state=42; every estimator uses n_jobs=1.",
            f"Each calibrated outer-fold estimator uses an inner StratifiedKFold(n_splits={CALIBRATION_INNER_SPLITS}, shuffle=True) fit only on that outer fold's training rows.",
            f"Inner seeds are {CALIBRATION_SEED_BASE} + (repeat - 1) * {N_SPLITS} + fold; the outer validation fold is never used to fit its calibrator.",
            "Calibration metrics use per-row probabilities averaged across that row's 10 out-of-fold repeats from the 1,760-row training partition.",
            "Multiclass log loss and Brier score use all 22 classes. Brier is mean per-row sum of squared class-probability errors.",
            f"Top-label ECE uses {CALIBRATION_BINS} equal-width confidence bins and compares top-score confidence with top-class correctness.",
            "Classwise macro ECE averages one-vs-rest ECE across 22 classes; micro ECE pools all sample/class cells. Bin-based metrics depend on bin count and can hide within-bin behavior.",
            "With 22 classes, probabilities are sparse and individual-class reliability estimates can be noisy. Repeated out-of-fold calibration does not establish external or real-world calibration.",
            "Calibration scores are model output distributions, not probabilities of crop survival, yield, or agricultural success.",
            "",
            calibration_table.to_string(index=False, float_format=lambda value: f"{value:.6f}"),
            "",
            "Near-duplicate investigation",
            "----------------------------",
            "Exact duplicates are rows with identical values for all seven input features, regardless of label.",
            f"Near-similar pairs are distinct feature vectors with standardized Euclidean distance <= {threshold:.2f}; feature standardization was fit on training rows only.",
            "The cutoff is an exploratory screening definition, not a validated domain threshold. Sensitivity counts at 0.10, 0.25, and 0.50 standardized-distance units are included below.",
            "The investigation inspects all dataset rows only to flag possible train/test-neighbor overlap; these results do not alter training, predictions, or the raw CSV.",
            f"Rows examined: {duplicate_summary['rows_examined']}",
            f"Exact duplicate feature groups / affected rows / unordered pairs: {duplicate_summary['exact_duplicate_feature_groups']} / {duplicate_summary['exact_duplicate_affected_rows']} / {duplicate_summary['exact_duplicate_unordered_pairs']}",
            f"Exact pairs crossing train/test: {duplicate_summary['exact_pairs_crossing_train_test']}; exact groups with conflicting labels: {duplicate_summary['exact_groups_with_conflicting_labels']}",
            f"Near-similar pairs / affected rows / pairs crossing train/test at threshold: {duplicate_summary['near_duplicate_pairs']} / {duplicate_summary['near_duplicate_affected_rows']} / {duplicate_summary['near_pairs_crossing_train_test']}",
            f"Near-pair label agreement rate: {duplicate_summary['near_pair_label_agreement_rate']:.6f}",
            f"Nearest-neighbor standardized-distance min / p10 / median: {duplicate_summary['nearest_neighbor_distance_min']:.6f} / {duplicate_summary['nearest_neighbor_distance_p10']:.6f} / {duplicate_summary['nearest_neighbor_distance_median']:.6f}",
            f"Threshold sensitivity pair counts: {duplicate_summary['near_duplicate_sensitivity_pair_counts']}",
            "Similar rows alone do not prove leakage, duplicated collection events, or synthetic data. No rows were removed and no causal claim is made.",
            "",
            "Limitations",
            "-----------",
            "All reported CV evidence is internal to this dataset and uses only the original training partition. The held-out test partition was not evaluated by this analysis.",
            "High benchmark scores or apparently good calibration do not establish generalization to farms, locations, seasons, or measurement procedures not represented here.",
        ]
    )
    ANALYSIS_REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    X, y = load_and_validate_data()
    train_indices, test_indices = load_verified_split(y)
    artifact = joblib.load(MODEL_PATH)
    dataset_sha256 = hashlib.sha256(DATA_PATH.read_bytes()).hexdigest()
    if artifact.get("dataset_sha256") != dataset_sha256:
        raise ValueError("Dataset fingerprint does not match selected model metadata.")
    split = artifact.get("split_configuration", {})
    if not np.array_equal(
        np.asarray(split.get("train_row_indices", []), dtype=int), train_indices
    ) or not np.array_equal(
        np.asarray(split.get("test_row_indices", []), dtype=int), test_indices
    ):
        raise ValueError("Artifact row indices do not match the verified split manifest.")
    if artifact.get("feature_names") != FEATURES or artifact.get("class_labels") != CLASS_LABELS:
        raise ValueError("Selected model feature or class metadata is inconsistent.")
    if artifact.get("model_name") != "Random Forest":
        raise ValueError("Phase 8B expects the saved selected Random Forest.")
    selected_classifier = artifact["model"].named_steps.get("classifier")
    if (
        selected_classifier is None
        or selected_classifier.__class__.__name__ != "RandomForestClassifier"
        or selected_classifier.n_estimators != 100
        or selected_classifier.random_state != RANDOM_STATE
    ):
        raise ValueError("Saved estimator does not match the selected Random Forest configuration.")
    if not callable(getattr(artifact["model"], "predict_proba", None)):
        raise ValueError("The selected Random Forest does not expose predict_proba.")

    X_train = X.iloc[train_indices].reset_index(drop=True)
    y_train = y.iloc[train_indices].reset_index(drop=True)
    results, averaged_probabilities, _ = run_repeated_evaluation(
        X_train, y_train, CLASS_LABELS
    )
    calibration_rows: list[dict[str, Any]] = []
    reliability: dict[str, dict[str, list[dict[str, float | int]]]] = {}
    for model_name, probabilities in averaged_probabilities.items():
        metrics, bins = multiclass_calibration_metrics(
            y_train, probabilities, CLASS_LABELS
        )
        calibration_rows.append({"model": model_name, **metrics})
        reliability[model_name] = bins
    calibration_table = pd.DataFrame(calibration_rows)

    duplicate_pairs, duplicate_summary = investigate_similar_rows(
        X, y, train_indices, test_indices
    )
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(REPEATED_CV_PATH, index=False)
    calibration_table.to_csv(CALIBRATION_PATH, index=False)
    duplicate_pairs.to_csv(NEAR_DUPLICATES_PATH, index=False)
    write_repeated_cv_plot(results)
    write_calibration_plot(reliability)
    write_analysis_report(
        results,
        calibration_table,
        duplicate_summary,
        dataset_sha256,
        len(train_indices),
        len(test_indices),
        NEAR_DUPLICATE_THRESHOLD,
    )

    print("Repeated CV and reliability analysis complete.")
    print(results[results["result_type"].isin(["summary_mean", "summary_std", "corrected_95_percent_CI"])].to_string(index=False))
    print("\nCalibration metrics:")
    print(calibration_table.to_string(index=False))
    print("\nNear-duplicate summary:")
    for key, value in duplicate_summary.items():
        print(f"{key}: {value}")
    print("\nGenerated:")
    for path in (
        REPEATED_CV_PATH,
        CALIBRATION_PATH,
        NEAR_DUPLICATES_PATH,
        ANALYSIS_REPORT_PATH,
        REPEATED_CV_PLOT_PATH,
        CALIBRATION_PLOT_PATH,
    ):
        print(path.relative_to(ROOT))


if __name__ == "__main__":
    main()
