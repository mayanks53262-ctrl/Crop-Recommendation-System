from app.nodes.compiler import (
    compile_final_assessment_clarified,
    compile_final_assessment_direct,
)
from app.nodes.evaluator import (
    evaluate_intake,
    explicitly_selected_conflict_source,
    validate_routing_status,
)
from app.nodes.intake import (
    record_history_preferences,
    record_language_location,
    record_soil_water,
)
from app.nodes.verifier import (
    verify_agricultural_context_clarified,
    verify_agricultural_context_direct,
)
from app.schemas import (
    AssessmentOutput,
    ClarificationResult,
    ClarificationPayload,
    CompletenessEvaluation,
    HistoryPreferencesIntake,
    LanguageLocationIntake,
    RoutingDecision,
    SessionResults,
    SoilWaterIntake,
    VerifierOutput,
)
from app.state import WorkflowState


NODE_NAMES = {
    "manual_trigger": "Manual Launch",
    "request_language_and_location": "Language And Location Intake",
    "request_soil_and_water": "Soil And Water Intake",
    "request_history_and_preferences": "History And Preference Intake",
    "assess_intake_completeness": "Completeness Evaluator",
    "check_critical_data": "Check Critical Data",
    "request_critical_clarification": "Clarification Intake",
    "verify_agricultural_context_clarified": "Agricultural Research Verifier Clarified",
    "verify_agricultural_context_direct": "Agricultural Research Verifier Direct",
    "compile_final_assessment_clarified": "Assessment Compiler Clarified",
    "compile_final_assessment_direct": "Assessment Compiler Direct",
}


class InvalidTransitionError(Exception):
    pass


def submit_language_location(state: WorkflowState, payload: LanguageLocationIntake) -> None:
    if state.status != "started":
        raise InvalidTransitionError("Language and location intake is not expected at this stage.")
    record_language_location(state, payload)


def submit_soil_water(state: WorkflowState, payload: SoilWaterIntake) -> None:
    if state.status != "awaiting_soil_water":
        raise InvalidTransitionError("Soil and water intake is not expected at this stage.")
    record_soil_water(state, payload)


def submit_history_preferences(state: WorkflowState, payload: HistoryPreferencesIntake) -> CompletenessEvaluation:
    if state.status != "awaiting_history_preferences":
        raise InvalidTransitionError("History and preferences intake is not expected at this stage.")
    record_history_preferences(state, payload)
    evaluation = evaluate_intake(state)
    route = validate_routing_status(evaluation.evaluation_status)
    if route == "verify_agricultural_context_direct":
        state.routing_decisions.append(
            RoutingDecision(
                from_node="check_critical_data",
                to_node=route,
                reason="evaluation_status=complete; no unresolved critical fields or irrigation conflict.",
            )
        )
        state.current_node = route
        state.verifier = verify_agricultural_context_direct(state).model_dump()
        state.assessment = compile_final_assessment_direct(state).model_dump()
        state.status = "completed"
        state.current_node = "compile_final_assessment_direct"
    else:
        state.routing_decisions.append(
            RoutingDecision(
                from_node="check_critical_data",
                to_node=route,
                reason="evaluation_status=needs_clarification; critical fields are missing or an irrigation conflict is unresolved.",
            )
        )
        state.status = "awaiting_clarification"
        state.current_node = route
        state.execution_history.append(route)
    return evaluation


def submit_clarification(state: WorkflowState, payload: ClarificationPayload) -> ClarificationResult:
    if state.status != "awaiting_clarification":
        raise InvalidTransitionError("Clarification is not expected at this stage.")
    state.clarification = payload.model_dump(mode="json")
    state.clarification["addressed_field"] = state.target_field
    state.clarification_history.append(state.clarification.copy())
    if state.target_field == "irrigation_source_conflict":
        selected = explicitly_selected_conflict_source(
            payload.clarification_response,
            state.intake.get("irrigation_source"),
            state.intake.get("primary_irrigation_source"),
        )
        if selected is not None:
            state.clarified_answers["irrigation_source_conflict"] = selected
    elif state.target_field:
        state.clarified_answers[state.target_field] = payload.clarification_response.strip()

    evaluation = evaluate_intake(
        state,
        state.clarified_answers,
        record_nodes=False,
    )
    state.evaluation = evaluation
    if evaluation.evaluation_status == "needs_clarification":
        state.routing_decisions.append(
            RoutingDecision(
                from_node="request_critical_clarification",
                to_node="request_critical_clarification",
                reason=(
                    "The clarification did not resolve all critical fields or explicitly identify "
                    "one of the conflicting irrigation sources; requesting a focused follow-up."
                ),
            )
        )
        state.current_node = "request_critical_clarification"
        state.execution_history.append("request_critical_clarification")
        return ClarificationResult(
            status="awaiting_clarification",
            evaluation=evaluation,
        )

    state.routing_decisions.append(
        RoutingDecision(
            from_node="request_critical_clarification",
            to_node="verify_agricultural_context_clarified",
            reason="The outstanding critical field(s) were clarified; the original intake remains unchanged.",
        )
    )
    state.verifier = verify_agricultural_context_clarified(state).model_dump()
    state.assessment = compile_final_assessment_clarified(state).model_dump()
    state.status = "completed"
    state.current_node = "compile_final_assessment_clarified"
    return ClarificationResult(
        status="completed",
        evaluation=evaluation,
        results=get_results(state),
    )


def get_results(state: WorkflowState) -> SessionResults:
    if state.status != "completed" or not state.verifier or not state.assessment:
        raise InvalidTransitionError("The final assessment is not available until the workflow completes.")
    return SessionResults(
        session_id=state.session_id,
        original_intake=state.intake,
        clarification=state.clarification,
        clarification_history=state.clarification_history,
        verifier=VerifierOutput(**state.verifier),
        assessment=AssessmentOutput(**state.assessment),
    )
