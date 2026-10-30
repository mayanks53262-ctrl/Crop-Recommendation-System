# Agricultural Field Assessment Workflow API

## Run locally

From the `backend` directory:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app.main:app --reload
```

The API is available at `http://127.0.0.1:8000`; interactive documentation is at
`http://127.0.0.1:8000/docs`. Run tests with `pytest -q`.

Sessions use process-local in-memory storage and do not survive server restarts.
This standalone application does not implement or modify `/predict`. External
institutional documents are not fetched or verified by this workflow.

## API

All workflow endpoints are versioned under `/api/workflow/v1`:

- `POST /sessions`
- `POST /sessions/{session_id}/language-location`
- `POST /sessions/{session_id}/soil-water`
- `POST /sessions/{session_id}/history-preferences`
- `GET /sessions/{session_id}/evaluation`
- `POST /sessions/{session_id}/clarification`
- `GET /sessions/{session_id}/assessment`
- `GET /sessions/{session_id}/status`

Pydantic rejects omitted or blank required intake fields with HTTP 422; explicit
answers such as `Unknown`, `No`, `Not decided`, and `Not specified` remain valid
answers. The clarification endpoint can return `status: awaiting_clarification`
with a follow-up prompt if an answer is unrelated, ambiguous, or does not resolve
the irrigation-source discrepancy. An explicit choice permits the workflow to
continue while preserving both original irrigation answers and all clarification
responses. Status responses include executed node IDs and routing decisions.
