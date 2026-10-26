from contextlib import asynccontextmanager
import json
import logging
import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator

if __package__:
    from .predict import (
        REQUIRED_FEATURES,
        ModelArtifactError,
        PredictionInputError,
        check_feature_range_warnings,
        load_model_artifact,
        predict_crop,
    )
    from .workflow_config import get_workflow_status
else:
    from predict import (
        REQUIRED_FEATURES,
        ModelArtifactError,
        PredictionInputError,
        check_feature_range_warnings,
        load_model_artifact,
        predict_crop,
    )
    from workflow_config import get_workflow_status

logger = logging.getLogger(__name__)
NON_NEGATIVE_FEATURES = {"N", "P", "K", "humidity", "ph", "rainfall"}
CROP_CATALOG_PATH = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "crop_catalog"
    / "crops.json"
)


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    N: float
    P: float
    K: float
    temperature: float
    humidity: float
    ph: float
    rainfall: float

    @field_validator(*sorted(NON_NEGATIVE_FEATURES))
    @classmethod
    def require_non_negative_measurements(cls, value: float, info: Any) -> float:
        if value < 0:
            raise ValueError(f"{info.field_name} cannot be negative.")
        return value


class CropScore(BaseModel):
    crop: str
    score: float


class PredictionResponse(BaseModel):
    predicted_crop: str
    top_three: list[CropScore] | None = None
    score_type: str | None = None
    warnings: list[dict[str, Any]] = Field(default_factory=list)


def _allowed_cors_origins() -> list[str]:
    configured = os.getenv("CROP_RECOMMENDATION_CORS_ORIGINS", "")
    origins = [origin.strip() for origin in configured.split(",") if origin.strip()]
    if "*" in origins:
        raise RuntimeError(
            "Wildcard CORS origins are not permitted; configure explicit frontend origins."
        )
    return origins


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model_artifact = None
    try:
        app.state.model_artifact = load_model_artifact()
        logger.info("Selected crop model loaded and ready.")
    except Exception:
        logger.exception("Selected crop model failed to load.")
    yield


app = FastAPI(
    title="Crop Recommendation API",
    description="Predicts crop labels from soil and climate inputs.",
    version="1.0.0",
    lifespan=lifespan,
)

cors_origins = _allowed_cors_origins()
if cors_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )


@app.exception_handler(RequestValidationError)
async def safe_validation_error_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    _ = request
    safe_detail = []
    for error in exc.errors():
        location = error.get("loc", ())
        field = location[-1] if location else None
        if field in REQUIRED_FEATURES:
            safe_location = ["body", field]
        elif error.get("type") == "missing" and field in REQUIRED_FEATURES:
            safe_location = ["body", field]
        else:
            safe_location = ["body"]

        error_type = error.get("type", "")
        if error_type == "missing":
            message = "This feature is required."
        elif error_type in {"float_parsing", "float_type"}:
            message = "Provide a numeric value."
        elif error_type in {"finite_number", "value_error"}:
            message = error.get("msg", "Invalid value.")
            if not isinstance(message, str) or "cannot be negative" not in message:
                message = "Provide a finite value that meets the feature constraints."
        elif error_type == "extra_forbidden":
            message = "Unexpected input field."
        else:
            message = "Invalid request value."

        safe_detail.append({"loc": safe_location, "msg": message})
    return JSONResponse(status_code=422, content={"detail": safe_detail})


@app.exception_handler(Exception)
async def safe_unexpected_error_handler(
    request: Request, exc: Exception
) -> JSONResponse:
    _ = request, exc
    logger.exception("Unexpected API request failure.")
    return JSONResponse(
        status_code=500,
        content={"detail": "The request could not be completed."},
    )


@app.get("/")
async def root() -> dict[str, str]:
    return {"service": "Crop Recommendation API", "status": "ok"}


@app.get("/health")
async def health(request: Request) -> dict[str, bool]:
    if request.app.state.model_artifact is None:
        raise HTTPException(
            status_code=503,
            detail="The crop prediction model is not ready.",
        )
    return {"model_ready": True}


@app.get("/assessment/workflow-status")
async def assessment_workflow_status() -> dict[str, Any]:
    return get_workflow_status()


@app.get("/features")
async def features(request: Request) -> dict[str, list[str]]:
    artifact = request.app.state.model_artifact
    if artifact is None:
        raise HTTPException(
            status_code=503,
            detail="The crop prediction model is not ready.",
        )
    return {"features": artifact["feature_names"]}


@app.get("/crop-catalog")
async def crop_catalog() -> dict[str, Any]:
    """Return the separately maintained, source-status-labelled crop knowledge base."""
    with CROP_CATALOG_PATH.open(encoding="utf-8") as catalog_file:
        catalog = json.load(catalog_file)
    if not isinstance(catalog, dict) or not isinstance(catalog.get("crops"), list):
        raise ValueError("Crop catalog must contain a crops list.")
    return catalog


@app.post("/predict", response_model=PredictionResponse)
async def predict(
    prediction_request: PredictionRequest, request: Request
) -> dict[str, Any]:
    artifact = request.app.state.model_artifact
    if artifact is None:
        raise HTTPException(
            status_code=503,
            detail="The crop prediction model is not ready.",
        )
    try:
        values = prediction_request.model_dump()
        result = predict_crop(values, artifact)
        split_configuration = artifact.get("split_configuration", {})
        training_row_indices = split_configuration.get("train_row_indices")
        result["warnings"] = check_feature_range_warnings(
            values,
            artifact.get("dataset_sha256"),
            tuple(training_row_indices) if training_row_indices is not None else None,
        )
        return result
    except PredictionInputError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except ModelArtifactError as exc:
        logger.exception("Loaded model artifact produced an invalid prediction.")
        raise HTTPException(
            status_code=500,
            detail="The crop prediction could not be completed.",
        ) from exc
    except (OSError, ValueError) as exc:
        logger.exception("Could not check prediction inputs against observed data ranges.")
        raise HTTPException(
            status_code=500,
            detail="The crop prediction could not be completed.",
        ) from exc
