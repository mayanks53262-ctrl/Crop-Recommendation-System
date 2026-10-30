from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

from app.schemas import CompletenessEvaluation, RoutingDecision


@dataclass
class WorkflowState:
    session_id: str = field(default_factory=lambda: str(uuid4()))
    status: str = "started"
    current_node: str = "manual_trigger"
    execution_history: list[str] = field(default_factory=lambda: ["manual_trigger"])
    routing_decisions: list[RoutingDecision] = field(default_factory=list)
    intake: dict[str, Any] = field(default_factory=dict)
    evaluation: CompletenessEvaluation | None = None
    target_field: str | None = None
    clarified_answers: dict[str, str] = field(default_factory=dict)
    clarification: dict[str, str | None] | None = None
    clarification_history: list[dict[str, str | None]] = field(default_factory=list)
    verifier: dict[str, str] | None = None
    assessment: dict[str, str] | None = None


class SessionStore:
    """Process-local session storage; sessions are lost when the server restarts."""

    def __init__(self) -> None:
        self._sessions: dict[str, WorkflowState] = {}

    def create(self) -> WorkflowState:
        state = WorkflowState()
        self._sessions[state.session_id] = state
        return state

    def get(self, session_id: str) -> WorkflowState | None:
        return self._sessions.get(session_id)


session_store = SessionStore()
