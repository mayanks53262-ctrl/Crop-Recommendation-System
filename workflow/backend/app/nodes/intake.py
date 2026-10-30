from app.schemas import (
    HistoryPreferencesIntake,
    LanguageLocationIntake,
    SoilWaterIntake,
)
from app.state import WorkflowState


def record_language_location(state: WorkflowState, intake: LanguageLocationIntake) -> None:
    state.intake.update(intake.model_dump(mode="json", exclude_unset=True))
    state.execution_history.append("request_language_and_location")
    state.current_node = "request_soil_and_water"
    state.status = "awaiting_soil_water"


def record_soil_water(state: WorkflowState, intake: SoilWaterIntake) -> None:
    state.intake.update(intake.model_dump(mode="json", exclude_unset=True))
    state.execution_history.append("request_soil_and_water")
    state.current_node = "request_history_and_preferences"
    state.status = "awaiting_history_preferences"


def record_history_preferences(state: WorkflowState, intake: HistoryPreferencesIntake) -> None:
    state.intake.update(intake.model_dump(mode="json", exclude_unset=True))
    state.execution_history.append("request_history_and_preferences")
