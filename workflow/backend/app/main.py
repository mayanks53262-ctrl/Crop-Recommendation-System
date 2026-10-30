from fastapi import FastAPI, HTTPException

from app.schemas import (
    ClarificationPayload,
    ClarificationResult,
    CompletenessEvaluation,
    HistoryPreferencesIntake,
    LanguageLocationIntake,
    SessionResults,
    SoilWaterIntake,
    WorkflowSessionCreated,
    WorkflowStatus,
)
from app.state import WorkflowState, session_store
from app.workflow import (
    InvalidTransitionError,
    get_results,
    submit_clarification,
    submit_history_preferences,
    submit_language_location,
    submit_soil_water,
)

app = FastAPI(title="Agricultural Field Assessment Workflow", version="1.0.0")
API_PREFIX = "/api/workflow/v1"


def _session_or_404(session_id: str) -> WorkflowState:
    state = session_store.get(session_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Workflow session not found.")
    return state


def _transition(callable_, *args):
    try:
        return callable_(*args)
    except InvalidTransitionError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


@app.get("/api/workflow/v1/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post(f"{API_PREFIX}/sessions", response_model=WorkflowSessionCreated, status_code=201)
def start_session() -> WorkflowSessionCreated:
    state = session_store.create()
    return WorkflowSessionCreated(
        session_id=state.session_id,
        status=state.status,
        current_node=state.current_node,
    )


@app.post(f"{API_PREFIX}/sessions/{{session_id}}/language-location", status_code=204)
def language_location(session_id: str, payload: LanguageLocationIntake) -> None:
    _transition(submit_language_location, _session_or_404(session_id), payload)


@app.post(f"{API_PREFIX}/sessions/{{session_id}}/soil-water", status_code=204)
def soil_water(session_id: str, payload: SoilWaterIntake) -> None:
    _transition(submit_soil_water, _session_or_404(session_id), payload)


@app.post(
    f"{API_PREFIX}/sessions/{{session_id}}/history-preferences",
    response_model=CompletenessEvaluation,
)
def history_preferences(session_id: str, payload: HistoryPreferencesIntake) -> CompletenessEvaluation:
    return _transition(submit_history_preferences, _session_or_404(session_id), payload)


@app.get(
    f"{API_PREFIX}/sessions/{{session_id}}/evaluation",
    response_model=CompletenessEvaluation,
)
def evaluation(session_id: str) -> CompletenessEvaluation:
    state = _session_or_404(session_id)
    if state.evaluation is None:
        raise HTTPException(status_code=409, detail="Completeness evaluation is not available yet.")
    return state.evaluation


@app.post(
    f"{API_PREFIX}/sessions/{{session_id}}/clarification",
    response_model=ClarificationResult,
)
def clarification(session_id: str, payload: ClarificationPayload) -> ClarificationResult:
    return _transition(submit_clarification, _session_or_404(session_id), payload)


@app.get(
    f"{API_PREFIX}/sessions/{{session_id}}/assessment",
    response_model=SessionResults,
)
def assessment(session_id: str) -> SessionResults:
    state = _session_or_404(session_id)
    return _transition(get_results, state)


@app.get(
    f"{API_PREFIX}/sessions/{{session_id}}/status",
    response_model=WorkflowStatus,
)
def workflow_status(session_id: str) -> WorkflowStatus:
    state = _session_or_404(session_id)
    return WorkflowStatus(
        session_id=state.session_id,
        status=state.status,
        current_node=state.current_node,
        execution_history=state.execution_history,
        routing_decisions=state.routing_decisions,
        evaluation=state.evaluation,
    )
