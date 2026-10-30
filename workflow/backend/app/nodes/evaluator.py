import re

from app.schemas import CompletenessEvaluation
from app.state import WorkflowState


FIELD_LABELS = {
    "district": ("district", "ज़िला"),
    "field_area_and_unit": ("field area and unit", "खेत का क्षेत्रफल और इकाई"),
    "soil_feel": ("soil feel/texture response", "मिट्टी का स्पर्श/बनावट"),
    "irrigation_source": ("irrigation source", "सिंचाई का स्रोत"),
    "primary_irrigation_source": ("primary irrigation source", "मुख्य सिंचाई स्रोत"),
    "water_reliability_drainage": ("water reliability", "पानी की विश्वसनीयता"),
    "intended_sowing_month": ("intended sowing timeline", "बुवाई का इच्छित समय"),
}
def _is_unanswered(value: object) -> bool:
    if value is None:
        return True
    if not isinstance(value, str):
        return False
    normalized = value.strip().casefold()
    return not normalized


def irrigation_conflicts(source: object, primary: object) -> bool:
    if not source or not primary:
        return False

    def normalize(value: object) -> str:
        normalized = str(value).casefold().replace(" ", "")
        return normalized.replace("tubewell/borewell", "tubewell").replace("borewell", "tubewell")

    source_value, primary_value = normalize(source), normalize(primary)
    if source_value == "unknown" or primary_value == "unknown":
        return False
    return source_value != primary_value


def evaluate_intake(
    state: WorkflowState,
    answered_fields: dict[str, str] | None = None,
    *,
    record_nodes: bool = True,
) -> CompletenessEvaluation:
    intake = dict(state.intake)
    if answered_fields:
        intake.update(
            (key, value)
            for key, value in answered_fields.items()
            if key != "irrigation_source_conflict"
        )
    missing: list[str] = []
    for key in (
        "district",
        "field_area_and_unit",
        "soil_feel",
        "water_reliability_drainage",
        "intended_sowing_month",
    ):
        if _is_unanswered(intake.get(key)):
            missing.append(key)

    source = intake.get("irrigation_source")
    primary = intake.get("primary_irrigation_source")
    if _is_unanswered(source) and _is_unanswered(primary):
        missing.append("primary_irrigation_source")
    conflict = irrigation_conflicts(source, primary)
    conflict_answer = (answered_fields or {}).get("irrigation_source_conflict")
    if conflict and conflict_answer and normalize_irrigation(conflict_answer) in {
        normalize_irrigation(source),
        normalize_irrigation(primary),
    }:
        conflict = False
    if conflict:
        missing.append("irrigation_source_conflict")

    if not missing:
        result = CompletenessEvaluation(
            evaluation_status="complete",
            clarification_prompt="Data is complete",
            missing_details="None",
        )
        state.target_field = None
    else:
        state.target_field = missing[0]
        language = str(intake.get("language_preference", "English"))
        hindi = language.casefold() == "hindi"
        def label_for(key: str) -> str:
            if key == "irrigation_source_conflict":
                return "सही सिंचाई स्रोत" if hindi else "irrigation source conflict"
            return FIELD_LABELS.get(key, (key, key))[1 if hindi else 0]

        label = label_for(missing[0])
        if missing[0] == "irrigation_source_conflict":
            prompt = (
                "कृपया स्पष्ट करें कि सही सिंचाई स्रोत कौन सा है।"
                if hindi else
                "Please clarify which irrigation source is correct."
            )
        else:
            prompt = (
                f"कृपया {label} बताएं।"
                if hindi else
                f"Please provide {label}."
            )
        result = CompletenessEvaluation(
            evaluation_status="needs_clarification",
            clarification_prompt=prompt,
            missing_details=", ".join(label_for(key) for key in missing),
        )
    state.evaluation = result
    if record_nodes:
        state.execution_history.append("assess_intake_completeness")
        state.execution_history.append("check_critical_data")
    return result


def normalize_irrigation(value: object) -> str:
    normalized = str(value).casefold().replace(" ", "")
    return normalized.replace("tubewell/borewell", "tubewell").replace("borewell", "tubewell")


def explicitly_selected_conflict_source(
    response: str,
    source: object,
    primary: object,
) -> str | None:
    """Return a conflicting intake value only when the response identifies one unambiguously."""
    candidates = {normalize_irrigation(str(source)): str(source), normalize_irrigation(str(primary)): str(primary)}
    uncertainty = (
        r"\bmaybe\b",
        r"\bperhaps\b",
        r"\bmight\b",
        r"\bnot\s+sure\b",
        r"\bunsure\b",
        r"\b(?:i\s+)?don'?t\s+know\b",
        r"\beither\b",
        r"\bor\b",
        r"\bunclear\b",
        r"\bnot\s+decided\b",
        r"\bcould\s+be\b",
        r"\bthink\b",
        r"\bprobably\b",
    )
    if any(re.search(pattern, response, re.IGNORECASE) for pattern in uncertainty):
        return None
    source_aliases = {
        "tubewell": (r"tubewell", r"borewell", r"ट्यूबवेल", r"बोरवेल"),
        "canal": (r"canal", r"नहर"),
        "rainfed/pond": (r"rainfed", r"pond", r"तालाब", r"वर्षा\s+आधारित"),
        "river/lift": (r"river", r"lift", r"नदी"),
        "other": (r"other", r"अन्य"),
        "none/unirrigated": (r"none", r"unirrigated", r"कोई\s+नहीं", r"असिंचित"),
        "unknown": (r"unknown", r"अज्ञात"),
    }
    matches = {
        normalized: original
        for normalized, original in candidates.items()
        if normalized
        and any(
            re.search(rf"(?<![a-z]){alias}(?![a-z])", response, re.IGNORECASE)
            for alias in source_aliases.get(normalized, (re.escape(normalized),))
        )
    }
    if len(matches) != 1:
        return None

    selected_normalized, selected_value = next(iter(matches.items()))
    exact_value = normalize_irrigation(response.strip()) == selected_normalized
    exact_alias = any(
        re.fullmatch(alias, response.strip(), re.IGNORECASE)
        for alias in source_aliases.get(selected_normalized, ())
    )
    if exact_value or exact_alias:
        return selected_value
    negative = (
        r"\bnot\b",
        r"\bno\b",
        r"\bnever\b",
        r"\bwrong\b",
        r"\bincorrect\b",
        r"नहीं",
        r"मत",
    )
    if any(re.search(pattern, response, re.IGNORECASE) for pattern in negative):
        return None
    decision = (
        r"\b(?:correct|right|actual|choose|chosen|selected|select|use|using|source)\b",
        r"(?:सही|स्रोत|चुन|उपयोग)",
    )
    return (
        selected_value
        if any(re.search(pattern, response, re.IGNORECASE) for pattern in decision)
        else None
    )


def validate_routing_status(evaluation_status: str) -> str:
    if evaluation_status == "complete":
        return "verify_agricultural_context_direct"
    if evaluation_status == "needs_clarification":
        return "request_critical_clarification"
    raise ValueError(f"Invalid evaluation status: {evaluation_status}")
