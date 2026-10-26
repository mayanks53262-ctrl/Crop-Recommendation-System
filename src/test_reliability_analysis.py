import unittest

import numpy as np
import pandas as pd

if __package__:
    from .reliability_analysis import (
        build_repeated_fold_specs,
        corrected_repeated_cv_interval,
        investigate_similar_rows,
        multiclass_calibration_metrics,
    )
else:
    from reliability_analysis import (
        build_repeated_fold_specs,
        corrected_repeated_cv_interval,
        investigate_similar_rows,
        multiclass_calibration_metrics,
    )


class ReliabilityAnalysisTests(unittest.TestCase):
    def test_repeated_folds_are_fixed_and_shared_schedule_covers_each_row(self) -> None:
        labels = pd.Series(np.repeat(["apple", "banana"], 25))
        first = build_repeated_fold_specs(labels)
        second = build_repeated_fold_specs(labels)
        self.assertEqual(len(first), 50)
        for spec_a, spec_b in zip(first, second):
            self.assertEqual(
                (spec_a["repeat"], spec_a["repeat_seed"], spec_a["fold"]),
                (spec_b["repeat"], spec_b["repeat_seed"], spec_b["fold"]),
            )
            np.testing.assert_array_equal(spec_a["fit_indices"], spec_b["fit_indices"])
            np.testing.assert_array_equal(
                spec_a["validation_indices"], spec_b["validation_indices"]
            )
            self.assertFalse(
                set(spec_a["fit_indices"]) & set(spec_a["validation_indices"])
            )
        for repeat in range(1, 11):
            validations = np.concatenate(
                [
                    spec["validation_indices"]
                    for spec in first
                    if spec["repeat"] == repeat
                ]
            )
            np.testing.assert_array_equal(np.sort(validations), np.arange(len(labels)))

    def test_corrected_interval_contains_mean_and_is_ordered(self) -> None:
        scores = [0.80, 0.82, 0.79, 0.84, 0.81]
        lower, upper = corrected_repeated_cv_interval(scores, 0.25)
        self.assertLess(lower, np.mean(scores))
        self.assertGreater(upper, np.mean(scores))
        self.assertLess(lower, upper)

    def test_corrected_interval_rejects_invalid_scores(self) -> None:
        with self.assertRaises(ValueError):
            corrected_repeated_cv_interval([0.8, np.nan], 0.25)
        with self.assertRaises(ValueError):
            corrected_repeated_cv_interval([0.8, 0.9], -0.1)

    def test_multiclass_calibration_metrics_and_bins(self) -> None:
        labels = np.array(["apple", "banana", "apple", "banana"])
        probabilities = np.array(
            [
                [0.8, 0.2],
                [0.3, 0.7],
                [0.6, 0.4],
                [0.4, 0.6],
            ]
        )
        metrics, bins = multiclass_calibration_metrics(
            labels, probabilities, ["apple", "banana"], n_bins=5
        )
        self.assertEqual(metrics["n_samples"], 4)
        self.assertEqual(metrics["accuracy"], 1.0)
        self.assertEqual(metrics["macro_f1"], 1.0)
        for metric in (
            "multiclass_log_loss",
            "multiclass_brier_score",
            "top_label_ece",
            "classwise_macro_ece",
            "classwise_micro_ece",
        ):
            self.assertTrue(np.isfinite(metrics[metric]))
        self.assertGreater(len(bins["top_label"]), 0)
        self.assertGreater(len(bins["classwise_ovr"]), 0)

    def test_calibration_rejects_malformed_probability_rows(self) -> None:
        labels = np.array(["a", "b"])
        with self.assertRaises(ValueError):
            multiclass_calibration_metrics(
                labels, np.array([[0.5, 0.5], [0.2, 0.2]]), ["a", "b"]
            )
        with self.assertRaises(ValueError):
            multiclass_calibration_metrics(
                labels, np.array([[np.nan, 0.0], [0.0, 1.0]]), ["a", "b"]
            )

    def test_near_duplicate_investigation_reports_pairs_and_partition_overlap(self) -> None:
        features = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
        X = pd.DataFrame(
            [
                [1, 1, 1, 1.0, 1, 1, 1],
                [1, 1, 1, 1.0, 1, 1, 1],
                [1, 1, 1, 1.01, 1, 1, 1],
                [10, 10, 10, 10.0, 10, 10, 10],
            ],
            columns=features,
        )
        y = pd.Series(["a", "a", "b", "b"])
        pairs, summary = investigate_similar_rows(
            X,
            y,
            train_indices=np.array([0, 2, 3]),
            test_indices=np.array([1]),
            threshold=0.1,
        )
        self.assertEqual(summary["exact_duplicate_feature_groups"], 1)
        self.assertEqual(summary["exact_duplicate_affected_rows"], 2)
        self.assertEqual(summary["exact_pairs_crossing_train_test"], 1)
        self.assertTrue(pairs["crosses_train_test"].any())
        self.assertTrue(
            set(pairs["similarity_type"]).issubset(
                {"exact_feature_match", "near_feature_match"}
            )
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
