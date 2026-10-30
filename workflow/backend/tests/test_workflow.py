import json
from pathlib import Path

from fastapi.testclient import TestClient
from pydantic import ValidationError
import pytest

from app.main import app
from app.nodes.compiler import (
    compile_final_assessment_clarified,
    compile_final_assessment_direct,
)
from app.nodes.evaluator import evaluate_intake, validate_routing_status
from app.nodes.verifier import (
    DISCLAIMER,
    verify_agricultural_context_clarified,
    verify_agricultural_context_direct,
)
from app.schemas import (
    ClarificationPayload,
    HistoryPreferencesIntake,
    LanguageLocationIntake,
    SoilWaterIntake,
)
from app.state import WorkflowState, session_store
from app.workflow import (
    InvalidTransitionError,
    NODE_NAMES,
    submit_clarification,
    submit_history_preferences,
    submit_language_location,
    submit_soil_water,
)

client = TestClient(app)


def start_complete_state() -> WorkflowState:
    state = session_store.create()
    submit_language_location(
        state,
        LanguageLocationIntake(
            language_preference="English",
            district="Example district",
            field_area_and_unit="2 acres",
            village_or_location="Example village",
        ),
    )
    submit_soil_water(
        state,
        SoilWaterIntake(
            soil_feel="Unknown (ज्ञात नहीं)",
            irrigation_source="Unknown",
            water_reliability_drainage="Unknown",
        ),
    )
    return state


def test_complete_path_visits_exact_direct_nodes():
    state = start_complete_state()
    evaluation = submit_history_preferences(
        state,
        HistoryPreferencesIntake(
            soil_test_report_available="Unknown",
            previous_crops="Not specified",
            intended_sowing_month="Not decided",
        ),
    )
    assert evaluation.evaluation_status == "complete"
    assert state.execution_history == [
        "manual_trigger",
        "request_language_and_location",
        "request_soil_and_water",
        "request_history_and_preferences",
        "assess_intake_completeness",
        "check_critical_data",
        "verify_agricultural_context_direct",
        "compile_final_assessment_direct",
    ]
    assert state.routing_decisions[0].model_dump() == {
        "from_node": "check_critical_data",
        "to_node": "verify_agricultural_context_direct",
        "reason": "evaluation_status=complete; no unresolved critical fields or irrigation conflict.",
    }
    assert state.clarified_answers == {}
    assert "Clarifications recorded separately" not in state.assessment["information_collected_and_source"]


def test_direct_compiler_ignores_clarification_only_state():
    state = start_complete_state()
    state.intake["district"] = "Original district"
    state.clarified_answers["district"] = "Clarified district"
    state.clarification_history.append(
        {
            "addressed_field": "district",
            "clarification_response": "Clarified district",
            "additional_notes": None,
        }
    )
    assessment = compile_final_assessment_direct(state)
    assert "Original district" in assessment.field_summary
    assert "Clarified district" not in assessment.information_collected_and_source


def test_missing_critical_data_routes_to_clarification():
    state = session_store.create()
    submit_language_location(
        state,
        LanguageLocationIntake(
            language_preference="Hindi",
            district="ज़िला",
            field_area_and_unit="2 एकड़",
            village_or_location="गाँव",
        ),
    )
    submit_soil_water(
        state,
        SoilWaterIntake(
            soil_feel="Unknown (ज्ञात नहीं)",
            irrigation_source="Unknown",
            water_reliability_drainage="Unknown",
        ),
    )
    state.intake["district"] = ""
    result = submit_history_preferences(
        state,
        HistoryPreferencesIntake(
            soil_test_report_available="Unknown",
            previous_crops="Not specified",
            intended_sowing_month="Not decided",
        ),
    )
    assert result.evaluation_status == "needs_clarification"
    assert state.current_node == "request_critical_clarification"
    assert state.execution_history[-1] == "request_critical_clarification"
    assert state.routing_decisions[0].to_node == "request_critical_clarification"
    assert "needs_clarification" in state.routing_decisions[0].reason
    reply = submit_clarification(state, ClarificationPayload(clarification_response="Lucknow"))
    assert reply.status == "completed"
    assert state.execution_history[-3:] == [
        "request_critical_clarification",
        "verify_agricultural_context_clarified",
        "compile_final_assessment_clarified",
    ]


def test_explicit_unknown_and_none_are_not_missing():
    state = start_complete_state()
    evaluation = submit_history_preferences(
        state,
        HistoryPreferencesIntake(
            soil_test_report_available="No",
            previous_crops="Unknown",
            intended_sowing_month="Not specified",
        ),
    )
    assert evaluation.evaluation_status == "complete"


def test_conflicting_irrigation_fields_trigger_clarification():
    state = start_complete_state()
    state.intake["irrigation_source"] = "Canal"
    state.intake["primary_irrigation_source"] = "Tubewell / Borewell"
    evaluation = submit_history_preferences(
        state,
        HistoryPreferencesIntake(
            soil_test_report_available="Unknown",
            previous_crops="Unknown",
            intended_sowing_month="March",
        ),
    )
    assert evaluation.evaluation_status == "needs_clarification"
    assert "irrigation source" in evaluation.clarification_prompt.casefold()
    assert state.intake["irrigation_source"] == "Canal"
    assert state.intake["primary_irrigation_source"] == "Tubewell / Borewell"


def test_unrelated_or_ambiguous_clarification_keeps_conflict_open():
    state = start_complete_state()
    state.intake["irrigation_source"] = "Canal"
    state.intake["primary_irrigation_source"] = "Tubewell / Borewell"
    submit_history_preferences(
        state,
        HistoryPreferencesIntake(
            soil_test_report_available="Unknown",
            previous_crops="Unknown",
            intended_sowing_month="March",
        ),
    )
    reply = submit_clarification(
        state,
        ClarificationPayload(clarification_response="I visited Canal yesterday."),
    )
    assert reply.status == "awaiting_clarification"
    reply = submit_clarification(
        state,
        ClarificationPayload(clarification_response="It may be Canal or Tubewell / Borewell."),
    )
    assert reply.status == "awaiting_clarification"
    assert reply.evaluation.evaluation_status == "needs_clarification"
    assert "irrigation_source_conflict" not in state.clarified_answers
    assert state.status == "awaiting_clarification"
    assert state.verifier is None
    assert state.assessment is None
    assert state.execution_history.count("request_critical_clarification") == 3
    assert state.routing_decisions[-1].from_node == "request_critical_clarification"
    assert state.routing_decisions[-1].to_node == "request_critical_clarification"


def test_explicit_conflict_answer_resolves_only_conflict_and_preserves_inputs():
    state = start_complete_state()
    state.intake["irrigation_source"] = "Canal"
    state.intake["primary_irrigation_source"] = "Tubewell / Borewell"
    submit_history_preferences(
        state,
        HistoryPreferencesIntake(
            soil_test_report_available="No",
            previous_crops="Wheat",
            intended_sowing_month="March",
        ),
    )
    reply = submit_clarification(
        state,
        ClarificationPayload(clarification_response="The correct source is Canal."),
    )
    assert reply.status == "completed"
    assert state.clarified_answers == {"irrigation_source_conflict": "Canal"}
    assert state.intake["irrigation_source"] == "Canal"
    assert state.intake["primary_irrigation_source"] == "Tubewell / Borewell"
    assert state.clarification_history[0]["addressed_field"] == "irrigation_source_conflict"
    assert "Original irrigation answers differed" in state.assessment["information_collected_and_source"]
    assert "irrigation_source_conflict" not in state.assessment["missing_information"]
    assert state.assessment["clarifying_questions"] == "None"


def test_unknown_irrigation_value_does_not_create_a_conflict():
    state = start_complete_state()
    state.intake["intended_sowing_month"] = "March"
    state.intake["irrigation_source"] = "Unknown"
    state.intake["primary_irrigation_source"] = "Canal"
    assert evaluate_intake(state).evaluation_status == "complete"


def test_clarification_preserves_original_and_unrelated_values():
    state = start_complete_state()
    state.intake["district"] = "Original district"
    state.intake["field_area_and_unit"] = None
    submit_history_preferences(
        state,
        HistoryPreferencesIntake(
            soil_test_report_available="No",
            previous_crops="Wheat",
            intended_sowing_month="March",
        ),
    )
    reply = submit_clarification(state, ClarificationPayload(clarification_response="3 acres"))
    assert reply.status == "completed"
    assert state.intake["district"] == "Original district"
    assert state.intake["field_area_and_unit"] is None
    assert state.clarified_answers == {"field_area_and_unit": "3 acres"}
    assert state.clarification["clarification_response"] == "3 acres"
    assert state.assessment["clarifying_questions"] == "None"


def test_node_outputs_have_exact_shapes_and_string_values():
    state = start_complete_state()
    verifier = verify_agricultural_context_direct(state).model_dump()
    assessment = compile_final_assessment_direct(state).model_dump()
    clarified_verifier = verify_agricultural_context_clarified(state).model_dump()
    clarified_assessment = compile_final_assessment_clarified(state).model_dump()
    assert set(verifier) == {
        "verified_agronomic_context",
        "crop_advisory_references",
        "unverified_parameters",
    }
    assert all(isinstance(value, str) for value in verifier.values())
    assert set(clarified_verifier) == set(verifier)
    assert all(isinstance(value, str) for value in clarified_verifier.values())
    assert set(assessment) == {
        "field_summary",
        "information_collected_and_source",
        "preliminary_agricultural_assessment",
        "clarifying_questions",
        "missing_information",
        "checks_required_before_recommendation",
        "limitations_and_confidence",
    }
    assert all(isinstance(value, str) for value in assessment.values())
    assert set(clarified_assessment) == set(assessment)
    assert all(isinstance(value, str) for value in clarified_assessment.values())


def test_disclaimer_and_unverified_claims_are_explicit():
    verifier = verify_agricultural_context_direct(start_complete_state()).model_dump()
    assessment = compile_final_assessment_direct(start_complete_state()).model_dump()
    assert DISCLAIMER == "No external institutional source was directly verified during this workflow run."
    assert DISCLAIMER in verifier["crop_advisory_references"]
    clarified = verify_agricultural_context_clarified(start_complete_state()).model_dump()
    assert DISCLAIMER in clarified["crop_advisory_references"]
    assert "UNVERIFIED" in verifier["unverified_parameters"]
    assert "UNVERIFIED" in clarified["unverified_parameters"]
    assert "UNVERIFIED" in assessment["limitations_and_confidence"]
    hindi_state = start_complete_state()
    hindi_state.intake["language_preference"] = "Hindi"
    hindi_verifier = verify_agricultural_context_direct(hindi_state).model_dump()
    assert DISCLAIMER in hindi_verifier["crop_advisory_references"]
    assert "ICAR" in hindi_verifier["crop_advisory_references"]
    assert "UNVERIFIED" in hindi_verifier["unverified_parameters"]


def test_invalid_routing_status_is_rejected():
    with pytest.raises(ValueError, match="Invalid evaluation status"):
        validate_routing_status("unexpected")


def test_required_node_ids_and_names_are_unchanged():
    assert NODE_NAMES == {
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


def test_sessions_are_isolated():
    first, second = session_store.create(), session_store.create()
    first.intake["district"] = "One"
    assert second.intake == {}
    assert first.session_id != second.session_id


def test_invalid_transitions_are_controlled():
    state = session_store.create()
    with pytest.raises(InvalidTransitionError):
        submit_soil_water(
            state,
            SoilWaterIntake(
                soil_feel="Unknown (ज्ञात नहीं)",
                irrigation_source="Unknown",
                water_reliability_drainage="Unknown",
            ),
        )


def test_api_validates_inputs_and_transitions():
    created = client.post("/api/workflow/v1/sessions")
    assert created.status_code == 201
    session_id = created.json()["session_id"]
    invalid_enum = client.post(
        f"/api/workflow/v1/sessions/{session_id}/soil-water",
        json={"soil_feel": "Mud"},
    )
    assert invalid_enum.status_code == 422
    wrong_transition = client.post(
        f"/api/workflow/v1/sessions/{session_id}/soil-water",
        json={
            "soil_feel": "Unknown (ज्ञात नहीं)",
            "irrigation_source": "Unknown",
            "water_reliability_drainage": "Unknown",
        },
    )
    assert wrong_transition.status_code == 409
    unavailable = client.get(f"/api/workflow/v1/sessions/{session_id}/assessment")
    assert unavailable.status_code == 409
    with pytest.raises(ValidationError):
        ClarificationPayload(clarification_response=" ")


def test_api_complete_path_returns_contract_shapes():
    session_id = client.post("/api/workflow/v1/sessions").json()["session_id"]
    root = f"/api/workflow/v1/sessions/{session_id}"
    assert client.post(
        f"{root}/language-location",
        json={
            "language_preference": "English",
            "district": "District",
            "field_area_and_unit": "2 acres",
            "village_or_location": "Village",
        },
    ).status_code == 204
    assert client.post(
        f"{root}/soil-water",
        json={
            "soil_feel": "Unknown (ज्ञात नहीं)",
            "irrigation_source": "Unknown",
            "water_reliability_drainage": "Unknown",
        },
    ).status_code == 204
    evaluation = client.post(
        f"{root}/history-preferences",
        json={
            "soil_test_report_available": "Unknown",
            "previous_crops": "Unknown",
            "intended_sowing_month": "Not decided",
        },
    )
    assert evaluation.status_code == 200
    assert set(evaluation.json()) == {
        "evaluation_status",
        "clarification_prompt",
        "missing_details",
    }
    assessment = client.get(f"{root}/assessment")
    assert assessment.status_code == 200
    assert set(assessment.json()["assessment"]) == {
        "field_summary",
        "information_collected_and_source",
        "preliminary_agricultural_assessment",
        "clarifying_questions",
        "missing_information",
        "checks_required_before_recommendation",
        "limitations_and_confidence",
    }
    assert set(assessment.json()["verifier"]) == {
        "verified_agronomic_context",
        "crop_advisory_references",
        "unverified_parameters",
    }


def test_api_rejects_omitted_required_fields_but_allows_explicit_unknown():
    session_id = client.post("/api/workflow/v1/sessions").json()["session_id"]
    root = f"/api/workflow/v1/sessions/{session_id}"
    missing_location = client.post(
        f"{root}/language-location",
        json={
            "language_preference": "English",
            "district": "District",
            "field_area_and_unit": "2 acres",
        },
    )
    assert missing_location.status_code == 422
    missing_language = client.post(
        f"{root}/language-location",
        json={
            "district": "District",
            "field_area_and_unit": "2 acres",
            "village_or_location": "Village",
        },
    )
    assert missing_language.status_code == 422
    valid_location = client.post(
        f"{root}/language-location",
        json={
            "language_preference": "English",
            "district": "District",
            "field_area_and_unit": "2 acres",
            "village_or_location": "Village",
        },
    )
    assert valid_location.status_code == 204
    missing_soil = client.post(
        f"{root}/soil-water",
        json={
            "soil_feel": "Unknown (ज्ञात नहीं)",
            "irrigation_source": "Unknown",
        },
    )
    assert missing_soil.status_code == 422
    valid_soil = client.post(
        f"{root}/soil-water",
        json={
            "soil_feel": "Unknown (ज्ञात नहीं)",
            "irrigation_source": "None / Unirrigated",
            "water_reliability_drainage": "Unknown",
        },
    )
    assert valid_soil.status_code == 204
    missing_history = client.post(
        f"{root}/history-preferences",
        json={
            "soil_test_report_available": "Unknown",
            "previous_crops": "Not specified",
        },
    )
    assert missing_history.status_code == 422
    valid_history = client.post(
        f"{root}/history-preferences",
        json={
            "soil_test_report_available": "No",
            "previous_crops": "Unknown",
            "intended_sowing_month": "Not decided",
        },
    )
    assert valid_history.status_code == 200
    assert valid_history.json()["evaluation_status"] == "complete"


def test_api_keeps_ambiguous_conflict_open_and_accepts_explicit_resolution():
    session_id = client.post("/api/workflow/v1/sessions").json()["session_id"]
    root = f"/api/workflow/v1/sessions/{session_id}"
    assert client.post(
        f"{root}/language-location",
        json={
            "language_preference": "English",
            "district": "District",
            "field_area_and_unit": "2 acres",
            "village_or_location": "Village",
        },
    ).status_code == 204
    assert client.post(
        f"{root}/soil-water",
        json={
            "soil_feel": "Unknown (ज्ञात नहीं)",
            "irrigation_source": "Canal",
            "primary_irrigation_source": "Tubewell / Borewell",
            "water_reliability_drainage": "Unknown",
        },
    ).status_code == 204
    assert client.post(
        f"{root}/history-preferences",
        json={
            "soil_test_report_available": "Unknown",
            "previous_crops": "Unknown",
            "intended_sowing_month": "Unknown",
        },
    ).json()["evaluation_status"] == "needs_clarification"
    ambiguous = client.post(
        f"{root}/clarification",
        json={"clarification_response": "Maybe Canal or Tubewell / Borewell"},
    )
    assert ambiguous.status_code == 200
    assert ambiguous.json()["status"] == "awaiting_clarification"
    assert ambiguous.json()["results"] is None
    resolved = client.post(
        f"{root}/clarification",
        json={"clarification_response": "The correct source is Canal"},
    )
    assert resolved.status_code == 200
    assert resolved.json()["status"] == "completed"
    assert resolved.json()["results"]["original_intake"]["irrigation_source"] == "Canal"
    assert resolved.json()["results"]["original_intake"]["primary_irrigation_source"] == "Tubewell / Borewell"
    status = client.get(f"{root}/status")
    assert status.status_code == 200
    assert status.json()["routing_decisions"][-1]["to_node"] == "verify_agricultural_context_clarified"


def test_incomplete_critical_data_blocks_crop_recommendations():
    state = start_complete_state()
    state.intake["district"] = ""
    assessment = compile_final_assessment_direct(state)
    assert "remains undetermined" in assessment.preliminary_agricultural_assessment
    assert "No crops are recommended" in assessment.preliminary_agricultural_assessment
    assert "district" in assessment.missing_information
    assert "not recorded in the intake" in assessment.missing_information


def test_hindi_and_english_locales_cover_all_interface_messages():
    locale_path = Path(__file__).parents[2] / "frontend" / "src" / "locales.json"
    locale_data = json.loads(locale_path.read_text(encoding="utf-8"))
    assert set(locale_data) == {"English", "Hindi"}
    assert locale_data["English"].keys() == locale_data["Hindi"].keys()
    expected_messages = {
        "appTitle",
        "intro",
        "preferredLanguage",
        "requiredError",
        "validationError",
        "loading",
        "clarificationTitle",
        "submitClarification",
        "fieldSummary",
        "preliminaryAssessment",
        "limitationsConfidence",
        "verificationDetails",
        "newSession",
        "footer",
    }
    assert expected_messages <= locale_data["English"].keys()
    assert all(
        locale_data["English"][key] and locale_data["Hindi"][key]
        for key in locale_data["English"]
    )
    assert "अज्ञात" in locale_data["Hindi"]["unknownSoil"]


def test_clarified_route_is_recorded_only_after_conflict_is_resolved():
    state = start_complete_state()
    state.intake["irrigation_source"] = "Canal"
    state.intake["primary_irrigation_source"] = "Tubewell / Borewell"
    submit_history_preferences(
        state,
        HistoryPreferencesIntake(
            soil_test_report_available="Unknown",
            previous_crops="Unknown",
            intended_sowing_month="Unknown",
        ),
    )
    submit_clarification(state, ClarificationPayload(clarification_response="Canal"))
    assert [item.to_node for item in state.routing_decisions] == [
        "request_critical_clarification",
        "verify_agricultural_context_clarified",
    ]
    assert state.execution_history[-2:] == [
        "verify_agricultural_context_clarified",
        "compile_final_assessment_clarified",
    ]


def test_status_unknown_is_not_treated_as_missing():
    state = start_complete_state()
    evaluation = submit_history_preferences(
        state,
        HistoryPreferencesIntake(
            soil_test_report_available="Unknown",
            previous_crops="Unknown",
            intended_sowing_month="Unknown",
        ),
    )
    assert evaluation.evaluation_status == "complete"
