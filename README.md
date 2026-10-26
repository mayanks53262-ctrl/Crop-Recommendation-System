# Crop Recommendation System — Starter Project

## Phase 1: Dataset audit and EDA

### Structure
- `data/raw/`: original datasets; do not edit raw files.
- `data/processed/`: validated/cleaned outputs.
- `data/crop_catalog/`: structured crop knowledge base, separate from the ML dataset.
- `data/research/`: agricultural source reports and research data; no Raebareli report is currently present.
- `notebooks/`: EDA notebooks.
- `src/`: reusable code.
- `models/`: saved models (later phase).
- `reports/`: audit reports and figures.

### Environment
Python 3.12+ is required by the current pinned runtime dependencies. The Vercel
backend runtime is set to Python 3.14 in `.python-version`.

```bash
python -m venv .venv
```

Windows:
```bash
.venv\Scripts\activate
```

macOS/Linux:
```bash
source .venv/bin/activate
```

Install:
```bash
pip install -r requirements-dev.txt
```

`requirements.txt` contains the API and model runtime dependencies.
`requirements-dev.txt` also installs the plotting and notebook tools used by
the research workflow.

Run the first audit from the project root:
```bash
python src/audit_dataset.py
```

The audit reads `data/raw/Crop_recommendation.csv`, checks the schema, missing values,
duplicate rows, numeric ranges and class distribution, and writes a text report to
`reports/dataset_audit.txt`.

Run exploratory data analysis and the baseline evaluation:
```bash
python src/eda.py
python src/train_baseline.py
```

The EDA script writes its summary to `reports/eda_summary.txt` and saves plots in
`reports/figures/`. The baseline script makes a stratified 80/20 train/test split,
compares a most-frequent `DummyClassifier` with a `RandomForestClassifier`, and writes
held-out metrics and classification reports to `reports/baseline_metrics.txt`. It also
saves confusion-matrix plots in `reports/figures/` and the fitted Random Forest with
feature names and class labels to `models/random_forest_baseline.joblib`. The test split
is used only for evaluation, not model selection or tuning. Dataset performance alone
does not establish that recommendations are suitable for real farms.

## Phase 4: Model reliability validation

Run the reproducible 5-fold validation and held-out error inspection:
```bash
python src/validate_model.py
python src/test_predictions.py
```

Cross-validation uses only the 80% training partition. The saved baseline model is
evaluated separately on the matching held-out 20% split. Results go to
`reports/cross_validation_results.txt`, per-sample predictions to
`reports/misclassified_samples.csv`, per-crop scores to
`reports/per_class_metrics.csv`, and the split's row assignments to
`reports/baseline_split_indices.csv`. The confusion matrix is
`reports/figures/test_confusion_matrix.png`.

The prediction checks keep application input validation separate from raw model
prediction. They reject missing or nonnumeric inputs, non-finite values, and negative
N/P/K, humidity, or rainfall; no arbitrary agricultural upper bounds are imposed.
Probabilities are model output scores and are not guarantees of agricultural success.

## Phase 5: Model comparison

From the project root, run:
```powershell
.\.venv\Scripts\python.exe src\compare_models.py
```

The script verifies the existing stratified 80/20 split manifest, compares Logistic
Regression, Random Forest, and an RBF-kernel SVM using the same five stratified folds
from the training partition, and selects by mean CV macro-F1 (ties: lower variability,
then faster mean fit time). Standard scaling for Logistic Regression and SVM is inside
each pipeline, so each fold fits its own scaler using only that fold's training data.
The held-out test set is evaluated only once after selection.

Outputs include `reports/model_comparison_cv.csv`,
`reports/model_comparison_cv.txt`, and
`reports/figures/model_comparison_cv.png`. The selected model's held-out classification
report and confusion matrix are saved in `reports/model_comparison_test.txt` and
`reports/figures/model_comparison_test_confusion_matrix.png`. The fitted pipeline and
metadata (including dataset SHA-256 and split row indices) are saved in
`models/selected_model.joblib`; the original Random Forest baseline artifact is not
overwritten. High benchmark scores do not prove performance on real farms or across
different regions.

## Phase 6: Reusable crop prediction

Run the prediction helper tests from the project root:
```powershell
.\.venv\Scripts\python.exe src\test_predict.py
```

Example use from the project root:
```python
from src.predict import predict_crop

result = predict_crop({
    "N": 90,
    "P": 42,
    "K": 43,
    "temperature": 20.88,
    "humidity": 82.00,
    "ph": 6.50,
    "rainfall": 202.94,
})
print(result["predicted_crop"])
if "top_three" in result:
    for recommendation in result["top_three"]:
        print(recommendation["crop"], recommendation["score"])
```

The helper validates all seven named inputs and passes a one-row pandas DataFrame to
the saved selected model using the artifact's exact feature order. It rejects negative
N/P/K, humidity, pH, or rainfall, as those quantities are non-negative by their
measurement scales; negative temperature is allowed. It does not impose arbitrary
upper bounds. Validation happens in the helper, not inside the estimator.

When available, top-three scores are taken from the model's `predict_proba` output.
They describe the model's output distribution; they are not probabilities of crop
survival, yield, or agricultural success. The benchmark model has not been validated
for real-world farming conditions or different regions. Only load the project-generated
model artifact from this trusted workspace; joblib model files use Python serialization
and must not be loaded from untrusted sources.

## Important data rule
The Kaggle-style crop classifier dataset is the initial baseline only. Do not append rice,
maize, NUE or Dryad records to it unless a later, documented experiment proves that the
features and target definitions are compatible. Fertilizer application is not the same as
measured soil nutrient status. The crop catalog can contain additional crops as
`KNOWLEDGE_BASE_ONLY` without fabricating ML training examples.

## Phase 7: FastAPI backend

Start the API from the project root using the project virtual environment:
```powershell
.\.venv\Scripts\python.exe -m uvicorn src.api:app --host 127.0.0.1 --port 8000
```

Interactive API documentation is available at `http://127.0.0.1:8000/docs`.

Endpoints:
- `GET /`: service status.
- `GET /health`: whether the selected model loaded successfully.
- `GET /features`: required inputs in the exact saved-model order.
- `GET /crop-catalog`: separately maintained crop knowledge base and verification status.
- `GET /assessment/workflow-status`: explicit status of the disconnected legacy workflow;
  this does not perform a field assessment or send data to Gemini Enterprise.
- `POST /predict`: validates inputs and returns the predicted crop and, if supported
  by the saved model, its top-three model output scores. Invalid requests return HTTP
  422 with safe validation details.

Example `POST /predict` request:
```json
{
  "N": 90,
  "P": 42,
  "K": 43,
  "temperature": 20.87974371,
  "humidity": 82.00274423,
  "ph": 6.502985292,
  "rainfall": 202.9355362
}
```

Example response generated by the current saved model:
```json
{
  "predicted_crop": "rice",
  "top_three": [
    {"crop": "rice", "score": 0.9},
    {"crop": "jute", "score": 0.1},
    {"crop": "pomegranate", "score": 0.0}
  ],
  "score_type": "model probability output; not agricultural success probability"
}
```

Run the helper and API tests:
```powershell
.\.venv\Scripts\python.exe src\test_predict.py
.\.venv\Scripts\python.exe src\test_api.py
```

The React frontend is served by Vite at `http://localhost:5173` in local
development. CORS is disabled unless the exact frontend origin is configured.
Enable the documented local origin for the API process with:
```powershell
$env:CROP_RECOMMENDATION_CORS_ORIGINS = "http://localhost:5173"
.\.venv\Scripts\python.exe -m uvicorn src.api:app --host 127.0.0.1 --port 8000
```
If a frontend uses another origin, configure its exact origin instead. Do not use
wildcard CORS in production.

The API loads only the project-generated `models/selected_model.joblib` artifact once
at application startup. Keep that file trusted: joblib uses Python serialization.
The benchmark model has not been validated for real-world farming conditions, and its
scores are not probabilities of crop survival, yield, or agricultural success.

### Phase 8A: Observed-range warnings

`POST /predict` preserves `predicted_crop`, `top_three`, and `score_type`, and also
returns a `warnings` list. The bounds come from the 1,760 training rows recorded in
the selected model metadata (the held-out test rows are excluded):

| Feature | Observed training minimum | Observed training maximum |
| --- | ---: | ---: |
| N | 0 | 140 |
| P | 5 | 145 |
| K | 5 | 205 |
| temperature | 8.825675 | 43.675493 |
| humidity | 14.258040 | 99.981876 |
| ph | 3.504752 | 9.935091 |
| rainfall | 20.211267 | 298.560117 |

An input that passes normal validation but falls outside these observed ranges is
still sent to the model; it is not clipped or rejected solely for being out of range.
For example, `N: 141` produces a warning identifying its value and the observed
minimum/maximum. Negative nutrient, humidity, pH, or rainfall values remain invalid
under the existing input rules.

These are descriptive bounds of this dataset's training partition, not universal
agricultural thresholds, safety limits, or guarantees that an in-range input is
reliable. A warning indicates the model may be less reliable outside its observed
data; absence of a warning does not establish real-world farming accuracy. Model
scores remain model output scores, not probabilities of crop survival or yield.

## Phase 8B: Repeated validation, calibration, and similarity screening

Run the separate Phase 8B analysis from the project root:
```powershell
.\.venv\Scripts\python.exe src\reliability_analysis.py
```

It verifies the selected-model dataset fingerprint and the saved, stratified
80/20 split (1,760 training rows and 440 held-out rows). Repeated stratified
five-fold CV is run only on the 1,760 training rows, ten times with `random_state`
seeds 42 through 51. Logistic Regression, Random Forest, and SVM reuse their
Phase 5 settings and share identical fold assignments. The 440-row held-out
partition is not used by this analysis.

`reports/repeated_cv_results.csv` contains per-repeat/per-fold accuracy, macro-F1,
and fit time, along with mean, sample standard deviation, and corrected
resampled-t 95% intervals. The correction uses
`mean +/- t(df=49) * sqrt((1/50 + 1/4) * sample_variance)`, where `1/4` is the
validation-to-fit size ratio in five-fold CV. Because the folds overlap, this is
an approximate dependence-corrected interval, not a guarantee based on 50
independent observations. `reports/figures/repeated_cv_macro_f1.png` visualizes
the repeated fold scores.

Calibration compares the selected Random Forest with a sigmoid-calibrated
Random Forest. Calibration is fitted only inside each outer training fold using
three-fold stratified inner CV with fixed derived seeds. The outer validation
fold does not fit the model or calibrator. `reports/calibration_metrics.csv`
includes accuracy, macro-F1, multiclass log loss, multiclass Brier score,
top-label expected calibration error (ECE), and one-vs-rest classwise ECE. These
metrics use each training row's probability scores averaged over its ten
out-of-fold repeats. `reports/figures/calibration_reliability.png` plots
top-label and pooled one-vs-rest reliability. With 22 classes, calibration
estimates—especially class-level estimates—can be noisy and depend on binning;
calibration is not assumed to improve predictions. Scores describe model output,
not agricultural success.

Near-duplicate screening reports exact seven-feature matches and distinct rows
within 0.25 standardized Euclidean distance. StandardScaler is fitted on the
training partition only. The report also gives sensitivity counts at 0.10, 0.25,
and 0.50 distance units and notes pairs spanning train/test partitions.
`reports/near_duplicate_pairs.csv` provides row-indexed pairs and labels;
`reports/reliability_analysis.txt` records methods, split/fingerprint, results,
and limitations. The near-similarity cutoff is an exploratory screening
definition, not a universal threshold. Similar rows alone do not prove leakage
or synthetic data; no records are removed or modified.

These analyses are internal to this dataset. Strong cross-validation or
calibration results do not establish reliability across real farms, locations,
seasons, or measurement processes.

### Phase 8B results from the current saved dataset

The run used dataset SHA-256
`54a5a6e5408668e668667efc50de2fc867c1b875e0431b4f54dd331b0a109a4e`.
These are repeated-CV training-partition results only (50 fold scores per row).
Intervals are the approximate corrected resampled-t 95% intervals described
above; they are not based on independent fold samples.

| Candidate | Accuracy mean ± SD (95% corrected CI) | Macro-F1 mean ± SD (95% corrected CI) |
| --- | --- | --- |
| Logistic Regression | 0.969602 ± 0.007174 (0.962111–0.977094) | 0.969460 ± 0.007193 (0.961948–0.976971) |
| Random Forest | 0.994261 ± 0.003785 (0.990309–0.998214) | 0.994238 ± 0.003814 (0.990255–0.998221) |
| Support Vector Machine | 0.979205 ± 0.006922 (0.971977–0.986432) | 0.979138 ± 0.007000 (0.971829–0.986447) |
| Random Forest (sigmoid calibrated) | 0.994034 ± 0.003641 (0.990232–0.997836) | 0.994006 ± 0.003678 (0.990166–0.997846) |

For repeated out-of-fold probabilities averaged per training row, uncalibrated
Random Forest versus sigmoid-calibrated Random Forest yielded:

| Candidate | Accuracy | Macro-F1 | Multiclass log loss | Multiclass Brier | Top-label ECE | Classwise macro ECE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Random Forest | 0.995455 | 0.995452 | 0.059566 | 0.016355 | 0.047502 | 0.004455 |
| Random Forest (sigmoid calibrated) | 0.994318 | 0.994313 | 0.067267 | 0.011202 | 0.054133 | 0.005212 |

The calibrated model had a lower multiclass Brier score but higher log loss,
top-label ECE, and classwise macro ECE, and slightly lower accuracy/macro-F1.
Thus calibration did not uniformly improve the measured properties. ECE values
depend on the ten-bin definition and probability averaging; they should not be
interpreted as definitive 22-class calibration guarantees.

Near-duplicate screening found no exact seven-feature duplicate rows. At the
exploratory 0.25 standardized-distance threshold, it found 182 near pairs
involving 284 rows; 54 pairs span the saved train/test boundary and pair labels
agreed for 99.45%. Sensitivity counts were 0 pairs at 0.10, 182 at 0.25, and
4,987 at 0.50. Similarity and train/test overlap alone do not prove leakage,
synthetic data, or duplicate collection records; no data were removed.

## Phase 9: React website

The frontend is a React + Vite application under `frontend/`. It uses Tailwind
CSS, Framer Motion with reduced-motion support, and `react-i18next` for English
and Hindi. API calls live in a small service module and use the real FastAPI
backend; there are no simulated predictions. The compact page shows the
independent seven-input ML prediction workflow and an explicit disconnected
status for Gemini Enterprise's Agricultural Field Assessment workflow. No
browser-only farm intake form or completeness indicator is presented.

### Prerequisites and local setup

Install Node.js 22.12+ (or 24+) and npm. From PowerShell, copy the example API
configuration and install the pinned JavaScript dependencies:

```powershell
Set-Location .\frontend
Copy-Item .env.example .env.local
npm.cmd install
```

The default `.env.example` points at `http://127.0.0.1:8000`. You can change
`VITE_API_BASE_URL` in the untracked `.env.local` file if the backend runs at a
different address. Do not put secrets in Vite environment variables; values
prefixed with `VITE_` are included in browser code.

### Start backend and website

In the first PowerShell window, from the project root, allow only the actual
local Vite origin and start the API:

```powershell
$env:CROP_RECOMMENDATION_CORS_ORIGINS = "http://localhost:5173"
.\.venv\Scripts\python.exe -m uvicorn src.api:app --host 127.0.0.1 --port 8000
```

In a second PowerShell window:

```powershell
Set-Location .\frontend
npm.cmd run dev
```

Open the Vite URL shown in the terminal (normally `http://localhost:5173`).
Use **Load sample data** to fill seven actual values from one row in the source
dataset, then submit to receive a live prediction from `POST /predict`. Only
that row's feature values are included in the frontend; the raw CSV is not
served or bundled. The API health indicator uses the real `GET /health` route.

### Tests and production build

```powershell
Set-Location .\frontend
npm.cmd test
npm.cmd run build
```

The frontend tests mock responses for deterministic behavior. Existing backend
regression tests can be run from the project root:

```powershell
.\.venv\Scripts\python.exe src\test_predict.py
.\.venv\Scripts\python.exe src\test_api.py
.\.venv\Scripts\python.exe src\test_predictions.py
.\.venv\Scripts\python.exe src\test_reliability_analysis.py
```

To test real integration, start both processes as above and submit the sample
values in the browser. The frontend displays returned values as model output
scores; they are not probabilities of crop survival, yield, or success.
Valid inputs outside the observed training ranges are still submitted and their
API warnings are displayed.

### CORS and connectivity troubleshooting

- If the browser reports a CORS error, confirm the frontend origin in the
  address bar exactly matches the comma-separated
  `CROP_RECOMMENDATION_CORS_ORIGINS` setting. The documented local origin is
  `http://localhost:5173`; `http://127.0.0.1:5173` is a different origin.
- If the status indicator says the API is unavailable, start the backend and
  confirm its address matches `VITE_API_BASE_URL` in `.env.local`.
- After changing `.env.local`, restart Vite so the new environment setting is
  loaded. Do not configure wildcard CORS for production.
- If npm reports dependency installation errors, check the installed Node.js
  version and use `npm.cmd` in PowerShell on systems that block `npm.ps1`.
- `npm.cmd audit` can be used to inspect advisories in the pinned frontend
  dependency tree.

### Visual assets and limits

No third-party photographs are bundled. The streamlined page uses a small
hand-authored brand mark; decorative hero and crop-card illustrations are not
part of the current interface. Image loading is not required for prediction.

Dataset provenance has not been independently verified, the model does not
accept region or season, and the current 22-label classifier omits crops
including wheat and mustard. Its benchmark scores do not establish performance
on real farms. Local climate, soil testing, irrigation, season, and agronomic
guidance remain important; do not use this student project as the sole basis
for farming decisions.

## Phase 10: Field context, crop knowledge status, and economics

The `data/crop_catalog/crops.json` file is a separate knowledge base. Its current
entries are wheat and mustard, marked `knowledge_base_only` because neither is a
class in the saved model. The project currently has no supplied Raebareli research
report under `data/research/`; agronomy fields, scientific names, and source lists
are therefore left blank or explicitly unverified. `GET /crop-catalog` exposes
that status without changing prediction classes or training data. Do not interpret
these entries as locally validated crop guidance.

### Gemini Enterprise workflow connection

Google documents a supported `streamAssist` API that can route a request to a
registered Workflow Builder workflow using
`agentsSpec.agentSpecs[].agentId`. It is therefore a supported invocation
interface for workflows in general, rather than a generic Gemini prompt. The
documented request contains `query.text`, optional `session`, and the numeric
workflow agent ID. Follow-up turns can pass the returned session resource name
to continue that session. The API response is a stream of answer text fragments
and terminal state/session metadata; it does not by itself guarantee a
machine-readable assessment schema or structured source metadata.

The docs impose important Workflow Builder requirements: a workflow must have
exactly one manual trigger; a required manual-trigger input cannot be populated
through `streamAssist` (the run fails even when a default is configured);
query text is provided to workflow LLM nodes but does not fill trigger
variables. Published workflows must be shared with the calling principal unless
that principal is the owner/admin. Google documents the
`discoveryengine.assistants.assist` IAM permission, the
`https://www.googleapis.com/auth/cloud-platform` OAuth scope, and backend
Application Default Credentials (ADC). For ADC, the API docs require the
`X-Goog-User-Project` quota-project header. The Discovery Engine API must be
enabled.

This repository does not have the exact workflow's agent resource, trigger
variable definitions, deployed Google identity, permission grant, or a verified
mapping from streamed answer text to the seven required assessment fields.
Therefore it does **not** call the workflow. The status endpoint reports which
non-secret resource identifiers are configured, but always reports
`not_connected` until workflow compatibility, authorization, and the actual
response contract are verified. No field data is sent and no mock clarification
or assessment is presented.

Configuration scaffolding reads these non-secret environment variables on the
backend only:

- `GEMINI_ENTERPRISE_PROJECT_ID`: Google Cloud project ID.
- `GEMINI_ENTERPRISE_LOCATION`: app location (`global`, `us`, or `eu`).
- `GEMINI_ENTERPRISE_APP_ID`: Gemini Enterprise engine/app ID.
- `GEMINI_ENTERPRISE_AGENT_ID`: numeric registered workflow agent ID.

Google's documented resource conventions are
`projects/{project}/locations/{location}/collections/default_collection/engines/{APP_ID}`
for the app and a registered agent path ending in its numeric agent ID. The
agent list is discovered using the `v1alpha` agents endpoint; invocation uses
`v1` `.../assistants/default_assistant:streamAssist`. These IDs must come from
the user's actual Google Cloud application; they are not inferred from the
workflow's display name. Configure ADC for the backend runtime. Never place
service account credentials or tokens in React/Vite environment variables or
logs.

Before enabling an adapter, the workflow owner must inspect the Manual Launch
trigger and confirm whether it has required input variables. If it does, the
smallest safe Workflow Builder change is to retain the single manual trigger
but remove the requirement for trigger-supplied values and make the existing
intake steps consume the submitted query and session conversation instead.
Keep the completeness checks, clarification branch, research verification, and
assessment compilation intact. Confirm that clarification turns work through
session continuation and that the final output reliably contains exactly
`field_summary`, `information_collected_and_source`,
`preliminary_agricultural_assessment`, `clarifying_questions`,
`missing_information`, `checks_required_before_recommendation`, and
`limitations_and_confidence`; similarly verify the research verifier's
`verified_agronomic_context`, `crop_advisory_references`, and
`unverified_parameters`. Only then can a backend adapter safely validate and
display the response.

To find the required values, ask the Google Cloud/Gemini Enterprise
administrator to identify the project ID, app engine ID/location, registered
agent ID and state, workflow publication/sharing, and the backend principal's
`discoveryengine.assistants.assist` authorization. Enable the Discovery Engine
API and configure ADC plus the quota project. Do not provide secrets in the
frontend.

Official references:
- [Invoke agents and workflows with StreamAssist](https://docs.cloud.google.com/gemini/enterprise/docs/invoke-agent-streamassist)
- [StreamAssist REST reference](https://docs.cloud.google.com/gemini/enterprise/docs/reference/rest/v1/projects.locations.collections.engines.assistants/streamAssist)
- [Registered agents REST reference](https://docs.cloud.google.com/gemini/enterprise/docs/reference/rest/v1alpha/projects.locations.collections.engines.assistants.agents)

The economics calculator is optional and collapsed by default. It compares two
user-named crops using field area (acre/hectare), expected kg per area unit,
selling INR/kg, and an explicit total cost in INR for the field. A missing
assumption is not treated as zero. Revenue is `area × yield per area unit ×
price`; net return is revenue minus total cost; benefit-cost ratio is revenue
divided by total cost (undefined when total cost is zero). No external prices or
yields are prefilled. Economic estimates do not establish agronomic suitability
or guarantee returns.

The seven-feature model, preprocessing, feature order, labels, and `/predict`
contract remain unchanged. Model scores are still classifier output scores, not
probabilities of crop success, yield, or profitability. Soil inputs may be
user-entered observations rather than verified laboratory results; use soil-test
evidence and qualified local advice for real decisions.

## Vercel preparation (not deployed)

The repository is prepared for two Vercel projects so the existing FastAPI
routes remain at their current paths instead of being moved under an `/api`
prefix:

- Backend project root: repository root. `pyproject.toml` points the Python
  runtime to `src.api:app`; `.python-version` selects Python 3.14. The
  `vercel.json` function configuration explicitly includes the selected model,
  its range-reference CSV, and the crop catalog, and excludes frontend and
  research-only assets from the Python bundle.
- Frontend project root: `frontend/`. Use the existing Vite build (`npm run
  build`) and set `VITE_API_BASE_URL` to the backend deployment origin at
  frontend build time. Set `CROP_RECOMMENDATION_CORS_ORIGINS` on the backend to
  the exact frontend origin. Neither setting contains a secret.

The selected model is about 3.6 MB and the range-reference dataset about 150
KB. The API reads that dataset to preserve the current out-of-training-range
warnings, so both it and the model must be available to the backend deployment.
The dataset provenance/license has not been independently verified; confirm
that it may be redistributed before publishing the repository. The model is a
trusted project-generated joblib artifact; do not replace it or load
untrusted serialized models.

The API uses FastAPI, NumPy, pandas, scikit-learn, SciPy, and joblib. The
versions of the model/inference libraries are pinned to the versions installed
when the saved model was successfully loaded in this workspace. The artifact
does not record the exact training-library versions, so this verifies current
loading but cannot independently confirm the original training environment.
Notebook and plotting dependencies have been moved to
`requirements-research.txt` to keep the Python function dependency set focused.
Vercel currently documents Python 3.12–3.14 and a 500 MB uncompressed standard
Python function bundle limit. The local development environment's full
`site-packages` is larger than that limit, but it includes research tools that
are no longer in the deployment requirements. A Vercel build has not been run,
so the actual built function size and deployment compatibility remain to be
confirmed before deployment.

Configure the two project roots and their environment variables in Vercel
after reviewing this repository; no Vercel project, billing, or deployment has
been created here.

### Vercel configuration and rollback checklist

For the backend project, set the project root to the repository root and retain
the Python 3.14 runtime. Configure
`CROP_RECOMMENDATION_CORS_ORIGINS` to the exact deployed frontend origin, with
no wildcard and no trailing path. For the frontend project, set its root to
`frontend/` and configure `VITE_API_BASE_URL` to the backend origin (scheme and
host only) before building. `VITE_` values are public browser configuration;
never put credentials or API keys there. No AI-provider credentials are needed
for the current application.

Before promoting a deployment, build the frontend with `npm ci` and
`npm run build`, verify the backend's `GET /health`, `GET /features`,
`GET /crop-catalog`, and `GET /assessment/workflow-status`, and submit a known
seven-field request to `POST /predict`. Confirm its existing response keys,
frontend connectivity, and exact-origin CORS behavior. These checks are a
release checklist, not evidence that a Vercel deployment has already passed.

After a known-good Vercel deployment exists, use that project's Deployments
dashboard to roll back to the last known-good deployment if a release fails.
Keep the environment variables required by that deployment intact. If source
code must also be reverted, review the change and use `git revert <commit>` to
create a forward rollback commit; do not reset or delete deployment assets.
Re-run the backend and frontend checks before promoting the rollback. No
deployment or rollback has been performed for this project.

### Planned first-party field assessment API

The field assessment should be a separate `POST /assessment` endpoint. It must
not change the seven-input `POST /predict` route or use that model to fill in
missing assessment information. Its response should keep the already
documented assessment fields: `field_summary`,
`information_collected_and_source`, `preliminary_agricultural_assessment`,
`clarifying_questions`, `missing_information`,
`checks_required_before_recommendation`, and
`limitations_and_confidence`.

The existing `/assessment/workflow-status` endpoint remains a disconnected
status report; the planned `/assessment` endpoint should not call Gemini
Enterprise or send field data to an external service.

The endpoint should return missing information and follow-up questions when
the submitted intake is incomplete, and should distinguish user-provided
information from independently verified sources. It should not present a
completed agricultural assessment or crop prediction when the information or
evidence is insufficient. The exact required intake fields, source-verification
policy, and conditions for producing a preliminary assessment still need to be
defined before implementation; no AI provider or external service is part of
this plan.
