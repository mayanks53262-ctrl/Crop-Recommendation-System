from app.schemas import VerifierOutput
from app.state import WorkflowState

DISCLAIMER = "No external institutional source was directly verified during this workflow run."


def _verify(state: WorkflowState) -> VerifierOutput:
    hindi = str(state.intake.get("language_preference", "English")).casefold() == "hindi"
    if hindi:
        context = (
            "किसान द्वारा दी गई जानकारी को उसी रूप में दर्ज किया गया है। मिट्टी का स्पर्श केवल "
            "प्रारंभिक गुणात्मक अवलोकन है; इससे औपचारिक मिट्टी वर्ग या अन्य मिट्टी गुण स्थापित "
            "नहीं होते। किसी क्षेत्रीय कृषि निष्कर्ष की स्वतंत्र पुष्टि नहीं हुई।"
        )
        references = (
            f"{DISCLAIMER} ICAR, संबंधित KVK और उत्तर प्रदेश कृषि विभाग से परामर्श किया जा सकता है; "
            "उनके संबंधित दस्तावेज़ न तो प्राप्त किए गए और न ही सत्यापित किए गए।"
        )
        params = (
            "UNVERIFIED: ज़िला-स्तरीय फसल क्षेत्र-वर्गीकरण, वर्षा के आँकड़े, उपज, मिट्टी प्रोफ़ाइल, "
            "pH, EC, NPK, उर्वरता, जल-प्रवेश, जल-निकास, लवणता, भूजल स्तर, सिंचाई विश्वसनीयता "
            "आँकड़े, कार्यक्रम और संस्थागत मार्गदर्शन। स्पर्श या स्थान से इनका अनुमान नहीं लगाया गया।"
        )
    else:
        context = (
            "Farmer-reported intake is recorded as reported; soil feel is only a tentative qualitative "
            "observation and does not establish a formal soil texture class or other soil properties. "
            "No regional agronomic conclusion was independently verified."
        )
        references = (
            f"{DISCLAIMER} ICAR, the relevant KVK, and the Uttar Pradesh Department of Agriculture "
            "are potential sources to consult; their relevant documents were not accessed or verified."
        )
        params = (
            "UNVERIFIED: district-level crop zoning, rainfall figures, crop yields, soil profiles, "
            "pH, EC, NPK, fertility, infiltration, drainage, salinity, water-table depth, irrigation "
            "reliability statistics, schedules, and institutional guidance. No such values are inferred "
            "from tactile soil observations or location alone."
        )
    return VerifierOutput(
        verified_agronomic_context=context,
        crop_advisory_references=references,
        unverified_parameters=params,
    )


def verify_agricultural_context_clarified(state: WorkflowState) -> VerifierOutput:
    state.execution_history.append("verify_agricultural_context_clarified")
    return _verify(state)


def verify_agricultural_context_direct(state: WorkflowState) -> VerifierOutput:
    state.execution_history.append("verify_agricultural_context_direct")
    return _verify(state)
