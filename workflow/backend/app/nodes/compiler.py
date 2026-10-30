from app.nodes.evaluator import FIELD_LABELS, irrigation_conflicts, normalize_irrigation
from app.schemas import AssessmentOutput
from app.state import WorkflowState


CRITICAL_FIELDS = (
    "district",
    "field_area_and_unit",
    "soil_feel",
    "water_reliability_drainage",
    "intended_sowing_month",
)
INTAKE_FIELDS = (
    "language_preference",
    "farmer_name",
    "district",
    "village_or_location",
    "field_area_and_unit",
    "soil_feel",
    "soil_visual_notes",
    "irrigation_source",
    "primary_irrigation_source",
    "backup_irrigation_source",
    "water_reliability_drainage",
    "water_availability_reliability",
    "canal_release_schedule",
    "soil_test_report_available",
    "previous_crops",
    "intended_sowing_month",
    "fertilizer_manure_history",
    "farming_objectives_budget",
)
FIELD_LABELS_HI = {
    "district": "ज़िला",
    "field_area_and_unit": "खेत का क्षेत्रफल और इकाई",
    "soil_feel": "मिट्टी का स्पर्श/बनावट",
    "water_reliability_drainage": "पानी की विश्वसनीयता और जल-निकास",
    "intended_sowing_month": "बुवाई का इच्छित समय",
    "primary_irrigation_source": "मुख्य सिंचाई स्रोत",
    "irrigation_source_conflict": "सही सिंचाई स्रोत",
}


def _is_recorded(value: object) -> bool:
    return value is not None and (not isinstance(value, str) or bool(value.strip()))


def _display(value: object, hindi: bool) -> str:
    if _is_recorded(value):
        return str(value)
    if hindi:
        return "इनटेक में दर्ज नहीं (not recorded in the intake)"
    return "not recorded in the intake"


def _field_label(key: str, hindi: bool) -> str:
    if hindi:
        return FIELD_LABELS_HI.get(key, key)
    return FIELD_LABELS.get(key, (key, key))[0]


def _resolved_conflict(state: WorkflowState, source: object, primary: object) -> bool:
    selected = state.clarified_answers.get("irrigation_source_conflict")
    return bool(
        selected
        and source
        and primary
        and normalize_irrigation(selected)
        in {normalize_irrigation(source), normalize_irrigation(primary)}
    )


def _clarifying_questions(unresolved: list[str], hindi: bool) -> str:
    if not unresolved:
        return "कोई नहीं" if hindi else "None"
    labels = ", ".join(_field_label(key, hindi) for key in unresolved)
    if hindi:
        return f"कृपया शेष महत्वपूर्ण जानकारी स्पष्ट करें: {labels}।"
    return f"Please clarify the remaining critical details: {labels}."


def _compile(state: WorkflowState, clarified: bool) -> AssessmentOutput:
    intake = state.intake
    hindi = str(intake.get("language_preference", "English")).casefold() == "hindi"
    effective = dict(intake)
    if clarified:
        effective.update(
            (key, value)
            for key, value in state.clarified_answers.items()
            if key != "irrigation_source_conflict"
        )
    unresolved = [key for key in CRITICAL_FIELDS if not _is_recorded(effective.get(key))]
    if not effective.get("irrigation_source") and not effective.get("primary_irrigation_source"):
        unresolved.append("primary_irrigation_source")
    source = intake.get("irrigation_source")
    primary = intake.get("primary_irrigation_source")
    conflict = irrigation_conflicts(source, primary)
    conflict_resolved = _resolved_conflict(state, source, primary) if clarified else False
    if conflict and not conflict_resolved:
        unresolved.append("irrigation_source_conflict")

    unrecorded = [key for key in INTAKE_FIELDS if not _is_recorded(effective.get(key))]
    unrecorded_text = [
        f"{key}: {_display(effective.get(key), hindi)}"
        for key in unrecorded
    ]
    unresolved_conflicts = []
    if conflict and not conflict_resolved:
        unresolved_conflicts.append(
            "सिंचाई स्रोत में अनसुलझा विरोध: " if hindi else "Unresolved irrigation source conflict: "
        )
        unresolved_conflicts[-1] += (
            f"irrigation_source={source!r}; primary_irrigation_source={primary!r}."
        )
    resolved_conflict_note = ""
    if conflict_resolved:
        resolved_conflict_note = (
            (
                f"मूल सिंचाई उत्तर अलग थे; स्पष्टीकरण में चुना गया स्रोत: "
                f"{state.clarified_answers['irrigation_source_conflict']}।"
            )
            if hindi
            else (
                "Original irrigation answers differed; the clarification explicitly selected: "
                f"{state.clarified_answers['irrigation_source_conflict']}."
            )
        )
    missing_information = "; ".join(unrecorded_text + unresolved_conflicts)
    if not missing_information:
        missing_information = "कोई नहीं" if hindi else "None"

    if unresolved:
        preliminary = (
            "महत्वपूर्ण जानकारी अनुपलब्ध या अनसुलझी है; खेत-विशिष्ट फसल उपयुक्तता अभी निर्धारित नहीं है। "
            "कोई फसल सुझाई, क्रमबद्ध, वर्गीकृत या खारिज नहीं की गई।"
            if hindi
            else (
                "Critical information is missing or unresolved; field-specific crop suitability remains "
                "undetermined. No crops are recommended, ranked, categorized, or eliminated."
            )
        )
    else:
        preliminary = (
            "आवश्यक जानकारी दर्ज है, लेकिन खेत-विशिष्ट कृषि साक्ष्य सत्यापित नहीं हुए हैं। "
            "खेत-विशिष्ट फसल उपयुक्तता अभी निर्धारित नहीं है; कोई फसल सुझाव, क्रमबद्ध, वर्गीकृत "
            "या खारिज नहीं की गई।"
            if hindi
            else (
                "Required intake responses are present, but field-specific agronomic evidence has not "
                "been verified. Field-specific crop suitability remains undetermined; no crops are "
                "recommended, ranked, categorized, or eliminated."
            )
        )
    if conflict and not conflict_resolved:
        preliminary += (
            f" मूल irrigation_source={source!r} और primary_irrigation_source={primary!r} "
            "में अनसुलझा विरोध है।"
            if hindi
            else (
                f" The original irrigation entries remain in unresolved conflict: "
                f"irrigation_source={source!r}; primary_irrigation_source={primary!r}."
            )
        )

    answers = "; ".join(
        f"{key}: {_display(intake.get(key), hindi)}"
        for key in INTAKE_FIELDS
    )
    clarification_records = ""
    if clarified and state.clarification_history:
        clarification_records = "; ".join(
            f"{entry.get('addressed_field')}: clarification_response="
            f"{entry.get('clarification_response')!r}"
            + (
                f", additional_notes={entry['additional_notes']!r}"
                if entry.get("additional_notes")
                else ""
            )
            for entry in state.clarification_history
        )
        answers += (
            f"; स्पष्टीकरण अलग से दर्ज (मूल उत्तर अपरिवर्तित): {clarification_records}"
            if hindi
            else f"; Clarifications recorded separately (original answers unchanged): {clarification_records}"
        )
    if resolved_conflict_note:
        answers += f"; {resolved_conflict_note}"

    if hindi:
        return AssessmentOutput(
            field_summary=(
                f"किसान: {_display(intake.get('farmer_name'), hindi)}; "
                f"ज़िला: {_display(intake.get('district'), hindi)}; "
                f"स्थान: {_display(intake.get('village_or_location'), hindi)}; "
                f"क्षेत्रफल: {_display(intake.get('field_area_and_unit'), hindi)}।"
            ),
            information_collected_and_source=f"किसान द्वारा बताए गए मूल उत्तर: {answers}।",
            preliminary_agricultural_assessment=preliminary,
            clarifying_questions=_clarifying_questions(unresolved, hindi),
            missing_information=missing_information,
            checks_required_before_recommendation=(
                "सिंचाई स्रोत और उपलब्धता की पुष्टि करें; खेत के जल-निकास का निरीक्षण करें; "
                "निर्णय के अनुरूप मिट्टी और पानी की जांच पर विचार करें; स्थानीय कृषि मार्गदर्शन "
                "को संबंधित संस्थानों से पुष्ट करें। जांच को सार्वभौमिक रूप से अनिवार्य या किसी "
                "मौजूदा मॉडल को चलाने के लिए तकनीकी रूप से आवश्यक नहीं बताया जा रहा है।"
            ),
            limitations_and_confidence=(
                "विश्वास सीमित है: यह प्रारंभिक जानकारी-सारांश है, फसल-सिफारिश नहीं। मिट्टी का "
                "स्पर्श केवल गुणात्मक अवलोकन है। क्षेत्रीय वर्गीकरण, वर्षा, मिट्टी प्रोफ़ाइल, "
                "सिंचाई कार्यक्रम और संस्थागत मार्गदर्शन UNVERIFIED हैं। "
                "No external institutional source was directly verified during this workflow run."
            ),
        )
    return AssessmentOutput(
        field_summary=(
            f"Farmer: {_display(intake.get('farmer_name'), hindi)}; "
            f"district: {_display(intake.get('district'), hindi)}; "
            f"location: {_display(intake.get('village_or_location'), hindi)}; "
            f"area: {_display(intake.get('field_area_and_unit'), hindi)}."
        ),
        information_collected_and_source=f"Original farmer-reported responses: {answers}.",
        preliminary_agricultural_assessment=preliminary,
        clarifying_questions=_clarifying_questions(unresolved, hindi),
        missing_information=missing_information,
        checks_required_before_recommendation=(
            "Verify irrigation source and availability; inspect field drainage; consider soil and "
            "water testing appropriate to the decision; corroborate local agronomic guidance with "
            "relevant agricultural institutions. Testing is not asserted to be universally mandatory "
            "or technically required to run any existing model."
        ),
        limitations_and_confidence=(
            "Confidence is limited: this is a preliminary intake summary, not a crop recommendation. "
            "Soil feel is qualitative only. Regional zoning, rainfall, soil profiles, irrigation "
            "schedules, and institutional guidance are UNVERIFIED. "
            "No external institutional source was directly verified during this workflow run."
        ),
    )


def compile_final_assessment_clarified(state: WorkflowState) -> AssessmentOutput:
    state.execution_history.append("compile_final_assessment_clarified")
    return _compile(state, clarified=True)


def compile_final_assessment_direct(state: WorkflowState) -> AssessmentOutput:
    state.execution_history.append("compile_final_assessment_direct")
    return _compile(state, clarified=False)
