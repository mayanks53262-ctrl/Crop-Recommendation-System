# Agricultural Field Assessment Workflow

Standalone, testable implementation of the 11-node workflow. This project is
separate from the Crop Recommendation System and does not implement `/predict`.
The workflow records farmer-reported data, checks intake completeness, follows
the direct or clarification route, and produces a preliminary assessment without
claiming to verify outside agricultural sources.

## Start both services (Windows PowerShell)

Terminal 1 — backend (from the Crop-Recommendation-System repository root):

```powershell
cd .\workflow\backend
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Terminal 2 — frontend (from the Crop-Recommendation-System repository root):

```powershell
cd .\workflow\frontend
npm install
npm run dev
```

Open the frontend address printed by Vite. FastAPI documentation is available
at `http://127.0.0.1:8000/docs`.

## Tests and build

From `backend`, activate the virtual environment and run `pytest -q`.
From `frontend`, run `npm run build`.

Sessions are stored in memory and are lost when the backend restarts. No
institutional source is accessed or independently verified by this app.
