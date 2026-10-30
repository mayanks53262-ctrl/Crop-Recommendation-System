from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class Language(str, Enum):
    HINDI = "Hindi"
    ENGLISH = "English"


class SoilFeel(str, Enum):
    STICKY = "Sticky (चिपचिपी / Clayey)"
    SMOOTH = "Smooth (चिकनी / Silty)"
    GRITTY = "Gritty (खुरदरी / Loamy)"
    SANDY = "Sandy (रेतीली)"
    LOOSE = "Loose (ढीली)"
    UNKNOWN = "Unknown (ज्ञात नहीं)"


class IrrigationSource(str, Enum):
    TUBEWELL = "Tubewell / Borewell"
    CANAL = "Canal"
    RAINFED_POND = "Rainfed / Pond"
    RIVER_LIFT = "River / Lift"
    OTHER = "Other"
    NONE = "None / Unirrigated"
    UNKNOWN = "Unknown"


class BackupIrrigationSource(str, Enum):
    NONE = "None"
    TUBEWELL = "Tubewell/Borewell"
    CANAL = "Canal"
    POND_TANK = "Pond/Tank"
    OTHER = "Other"
    UNKNOWN = "Unknown"


class YesNoUnknown(str, Enum):
    YES = "Yes"
    NO = "No"
    UNKNOWN = "Unknown"


class IntakePayload(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class LanguageLocationIntake(IntakePayload):
    language_preference: Language
    district: str = Field(min_length=1, max_length=120)
    field_area_and_unit: str = Field(min_length=1, max_length=120)
    village_or_location: str = Field(min_length=1, max_length=200)
    farmer_name: str | None = Field(default=None, max_length=120)


class SoilWaterIntake(IntakePayload):
    soil_feel: SoilFeel
    irrigation_source: IrrigationSource
    primary_irrigation_source: IrrigationSource | None = None
    backup_irrigation_source: BackupIrrigationSource | None = None
    water_reliability_drainage: str = Field(min_length=1, max_length=1000)
    soil_visual_notes: str | None = Field(default=None, max_length=1000)
    water_availability_reliability: str | None = Field(default=None, max_length=1000)
    canal_release_schedule: str | None = Field(default=None, max_length=1000)


class HistoryPreferencesIntake(IntakePayload):
    soil_test_report_available: YesNoUnknown
    previous_crops: str = Field(min_length=1, max_length=1000)
    intended_sowing_month: str = Field(min_length=1, max_length=120)
    fertilizer_manure_history: str | None = Field(default=None, max_length=1000)
    farming_objectives_budget: str | None = Field(default=None, max_length=1000)


class ClarificationPayload(IntakePayload):
    clarification_response: str = Field(min_length=1, max_length=2000)
    additional_notes: str | None = Field(default=None, max_length=2000)


class RoutingDecision(BaseModel):
    from_node: str
    to_node: str
    reason: str


class CompletenessEvaluation(BaseModel):
    evaluation_status: Literal["complete", "needs_clarification"]
    clarification_prompt: str
    missing_details: str


class VerifierOutput(BaseModel):
    verified_agronomic_context: str
    crop_advisory_references: str
    unverified_parameters: str


class AssessmentOutput(BaseModel):
    field_summary: str
    information_collected_and_source: str
    preliminary_agricultural_assessment: str
    clarifying_questions: str
    missing_information: str
    checks_required_before_recommendation: str
    limitations_and_confidence: str


class WorkflowStatus(BaseModel):
    session_id: str
    status: str
    current_node: str
    execution_history: list[str]
    routing_decisions: list[RoutingDecision]
    evaluation: CompletenessEvaluation | None = None


class WorkflowSessionCreated(BaseModel):
    session_id: str
    status: str
    current_node: str


class SessionResults(BaseModel):
    session_id: str
    original_intake: dict[str, str | None]
    clarification: dict[str, str | None] | None = None
    clarification_history: list[dict[str, str | None]]
    verifier: VerifierOutput
    assessment: AssessmentOutput


class ClarificationResult(BaseModel):
    status: Literal["awaiting_clarification", "completed"]
    evaluation: CompletenessEvaluation
    results: SessionResults | None = None
