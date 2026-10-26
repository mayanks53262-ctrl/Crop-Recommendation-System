import os
from collections.abc import Mapping


WORKFLOW_NAME = "Agricultural Field Assessment"
WORKFLOW_CONFIGURATION = {
    "project_id": "GEMINI_ENTERPRISE_PROJECT_ID",
    "location": "GEMINI_ENTERPRISE_LOCATION",
    "app_id": "GEMINI_ENTERPRISE_APP_ID",
    "agent_id": "GEMINI_ENTERPRISE_AGENT_ID",
}


def get_workflow_status(
    environ: Mapping[str, str] | None = None,
) -> dict[str, object]:
    environment = os.environ if environ is None else environ
    configured = {
        field: bool(environment.get(variable, "").strip())
        for field, variable in WORKFLOW_CONFIGURATION.items()
    }
    missing_configuration = [
        variable
        for field, variable in WORKFLOW_CONFIGURATION.items()
        if not configured[field]
    ]

    blockers = []
    if missing_configuration:
        blockers.append(
            "Set the non-secret Gemini Enterprise resource identifiers before "
            "attempting configuration."
        )
    blockers.extend(
        [
            "The workflow's Manual Launch trigger inputs and intake-node "
            "behavior have not been verified. streamAssist cannot populate "
            "manual-trigger input variables; a required variable prevents the run.",
            "Backend Application Default Credentials, the "
            "discoveryengine.assistants.assist permission, workflow sharing, "
            "and the workflow's published state have not been verified.",
            "The workflow response has not been verified against the required "
            "seven-field assessment schema.",
        ]
    )

    return {
        "workflow_name": WORKFLOW_NAME,
        "status": "not_connected",
        "connected": False,
        "integration": "Gemini Enterprise streamAssist",
        "configuration": configured,
        "missing_configuration": missing_configuration,
        "blockers": blockers,
        "message": (
            "No workflow request has been sent. Configuration presence does "
            "not confirm workflow compatibility, authorization, or connectivity."
        ),
    }
