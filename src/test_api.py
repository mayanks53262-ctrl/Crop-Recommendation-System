from pathlib import Path
import json
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

if __package__:
    from . import api
else:
    import api

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "raw" / "Crop_recommendation.csv"


class CropRecommendationApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.dataset = pd.read_csv(DATA_PATH)
        cls.sample = cls.dataset.iloc[0]

    def setUp(self) -> None:
        self.client_context = TestClient(api.app)
        self.client = self.client_context.__enter__()

    def tearDown(self) -> None:
        self.client_context.__exit__(None, None, None)

    def test_root_status(self) -> None:
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "ok")

    def test_interactive_documentation_is_available(self) -> None:
        response = self.client.get("/docs")
        self.assertEqual(response.status_code, 200)

    def test_health_reports_ready_model(self) -> None:
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"model_ready": True})

    def test_assessment_workflow_is_explicitly_not_connected(self) -> None:
        response = self.client.get("/assessment/workflow-status")
        self.assertEqual(response.status_code, 200)
        status = response.json()
        self.assertEqual(status["workflow_name"], "Agricultural Field Assessment")
        self.assertEqual(status["status"], "not_connected")
        self.assertFalse(status["connected"])
        self.assertEqual(status["integration"], "Gemini Enterprise streamAssist")
        self.assertIn("configuration", status)
        self.assertIn("missing_configuration", status)
        self.assertTrue(status["blockers"])
        self.assertIn("No workflow request has been sent", status["message"])

    def test_workflow_status_reports_configuration_without_exposing_values(self) -> None:
        if __package__:
            from .workflow_config import WORKFLOW_CONFIGURATION, get_workflow_status
        else:
            from workflow_config import WORKFLOW_CONFIGURATION, get_workflow_status

        configured_values = {
            variable: f"example-{field}-value"
            for field, variable in WORKFLOW_CONFIGURATION.items()
        }
        with patch.dict("os.environ", configured_values, clear=True):
            status = get_workflow_status()

        self.assertTrue(all(status["configuration"].values()))
        self.assertEqual(status["missing_configuration"], [])
        self.assertFalse(status["connected"])
        serialized = json.dumps(status)
        for value in configured_values.values():
            self.assertNotIn(value, serialized)

    def test_workflow_status_lists_missing_non_secret_identifiers(self) -> None:
        if __package__:
            from .workflow_config import WORKFLOW_CONFIGURATION, get_workflow_status
        else:
            from workflow_config import WORKFLOW_CONFIGURATION, get_workflow_status

        status = get_workflow_status({})
        self.assertEqual(
            status["missing_configuration"],
            list(WORKFLOW_CONFIGURATION.values()),
        )
        self.assertFalse(any(status["configuration"].values()))
        self.assertFalse(status["connected"])

    def test_health_reports_unavailable_model_without_internal_details(self) -> None:
        with patch.object(
            api,
            "load_model_artifact",
            side_effect=FileNotFoundError("C:\\private\\selected_model.joblib"),
        ), patch.object(api.logger, "exception"):
            with TestClient(api.app) as client:
                response = client.get("/health")
        self.assertEqual(response.status_code, 503)
        self.assertEqual(
            response.json(), {"detail": "The crop prediction model is not ready."}
        )
        self.assert_safe_error(response)

    def test_features_return_exact_artifact_order(self) -> None:
        response = self.client.get("/features")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json()["features"],
            [
                "N",
                "P",
                "K",
                "temperature",
                "humidity",
                "ph",
                "rainfall",
            ],
        )

    def test_crop_catalog_marks_unverified_knowledge_only_entries(self) -> None:
        response = self.client.get("/crop-catalog")
        self.assertEqual(response.status_code, 200, response.text)
        catalog = response.json()
        self.assertEqual(catalog["research_status"]["status"], "missing")
        crops = {crop["crop_name"]: crop for crop in catalog["crops"]}
        self.assertEqual(set(crops), {"wheat", "mustard"})
        for crop in crops.values():
            self.assertEqual(crop["verification_status"], "unverified")
            self.assertEqual(crop["model_support"], "knowledge_base_only")
            self.assertEqual(crop["sources"], [])
            self.assertIsNone(crop["season"])

    def test_valid_request_returns_known_crop_and_sorted_top_three_scores(self) -> None:
        body = {
            feature: float(self.sample[feature])
            for feature in api.REQUIRED_FEATURES
        }
        with patch.object(
            api, "load_model_artifact", side_effect=AssertionError("model reloaded")
        ):
            response = self.client.post("/predict", json=body)
        self.assertEqual(response.status_code, 200, response.text)
        result = response.json()

        artifact = self.client.app.state.model_artifact
        self.assertEqual(
            set(result),
            {"predicted_crop", "top_three", "score_type", "warnings"},
        )
        self.assertIn(result["predicted_crop"], artifact["class_labels"])
        self.assertEqual(result["warnings"], [])
        if hasattr(artifact["model"], "predict_proba"):
            self.assertEqual(len(result["top_three"]), 3)
            scores = [item["score"] for item in result["top_three"]]
            self.assertTrue(np.isfinite(scores).all())
            self.assertTrue(all(0.0 <= score <= 1.0 for score in scores))
            self.assertEqual(scores, sorted(scores, reverse=True))
            self.assertTrue(
                all(item["crop"] in artifact["class_labels"] for item in result["top_three"])
            )

    def test_valid_out_of_range_value_is_predicted_with_warning(self) -> None:
        artifact = self.client.app.state.model_artifact
        values = {
            feature: float(self.sample[feature])
            for feature in artifact["feature_names"]
        }
        values["N"] = 141
        response = self.client.post("/predict", json=values)

        self.assertEqual(response.status_code, 200, response.text)
        result = response.json()
        self.assertIn(result["predicted_crop"], artifact["class_labels"])
        self.assertEqual(
            set(result),
            {"predicted_crop", "top_three", "score_type", "warnings"},
        )
        self.assertEqual(len(result["warnings"]), 1)
        self.assertEqual(result["warnings"][0]["feature"], "N")
        self.assertEqual(result["warnings"][0]["observed_max"], 140)
        self.assertIn(
            "outside the observed training-data range",
            result["warnings"][0]["message"],
        )

    def test_multiple_out_of_range_api_values_return_multiple_warnings(self) -> None:
        artifact = self.client.app.state.model_artifact
        values = {
            feature: float(self.sample[feature])
            for feature in artifact["feature_names"]
        }
        values["N"] = 141
        values["temperature"] = 50
        response = self.client.post("/predict", json=values)

        self.assertEqual(response.status_code, 200, response.text)
        warnings = response.json()["warnings"]
        self.assertEqual(
            [warning["feature"] for warning in warnings], ["N", "temperature"]
        )

    def test_missing_feature_returns_safe_422(self) -> None:
        body = {feature: 1 for feature in api.REQUIRED_FEATURES}
        body.pop("humidity")
        response = self.client.post("/predict", json=body)
        self.assertEqual(response.status_code, 422)
        self.assertIn("humidity", str(response.json()["detail"]))
        self.assert_safe_error(response)

    def test_invalid_string_returns_safe_422(self) -> None:
        body = {feature: 1 for feature in api.REQUIRED_FEATURES}
        body["N"] = "not-a-number"
        response = self.client.post("/predict", json=body)
        self.assertEqual(response.status_code, 422)
        self.assert_safe_error(response)

    def test_nan_and_infinity_return_safe_422(self) -> None:
        for value in (np.nan, np.inf, -np.inf):
            with self.subTest(value=repr(value)):
                body = {feature: 1 for feature in api.REQUIRED_FEATURES}
                body["rainfall"] = value
                response = self.client.post(
                    "/predict",
                    content=json.dumps(body, allow_nan=True),
                    headers={"Content-Type": "application/json"},
                )
                self.assertEqual(response.status_code, 422)
                self.assert_safe_error(response)

    def test_negative_nutrients_and_rainfall_return_safe_422(self) -> None:
        for feature in ("N", "P", "K", "rainfall"):
            with self.subTest(feature=feature):
                body = {name: 1 for name in api.REQUIRED_FEATURES}
                body[feature] = -1
                response = self.client.post("/predict", json=body)
                self.assertEqual(response.status_code, 422)
                self.assertIn("cannot be negative", str(response.json()["detail"]))
                self.assert_safe_error(response)

    def test_negative_humidity_and_ph_return_safe_422(self) -> None:
        for feature in ("humidity", "ph"):
            with self.subTest(feature=feature):
                body = {name: 1 for name in api.REQUIRED_FEATURES}
                body[feature] = -1
                response = self.client.post("/predict", json=body)
                self.assertEqual(response.status_code, 422)
                self.assertIn("cannot be negative", str(response.json()["detail"]))
                self.assert_safe_error(response)

    def test_negative_temperature_is_allowed(self) -> None:
        body = {
            feature: float(self.sample[feature])
            for feature in api.REQUIRED_FEATURES
        }
        body["temperature"] = -5
        response = self.client.post("/predict", json=body)
        self.assertEqual(response.status_code, 200, response.text)

    def test_internal_prediction_error_is_sanitized(self) -> None:
        body = {feature: 1 for feature in api.REQUIRED_FEATURES}
        with patch.object(
            api,
            "predict_crop",
            side_effect=RuntimeError(
                "C:\\private\\workspace\\secret\\traceback details"
            ),
        ), patch.object(api.logger, "exception"):
            with TestClient(api.app, raise_server_exceptions=False) as client:
                response = client.post("/predict", json=body)
        self.assertEqual(response.status_code, 500)
        self.assertEqual(
            response.json(), {"detail": "The request could not be completed."}
        )
        self.assert_safe_error(response)

    def assert_safe_error(self, response) -> None:
        body = response.text.lower()
        self.assertNotIn(str(ROOT).lower(), body)
        self.assertNotIn("traceback", body)
        self.assertNotIn("secret", body)
        self.assertNotIn("selected_model.joblib", body)


if __name__ == "__main__":
    unittest.main(verbosity=2)
