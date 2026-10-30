# Smart Crop Recommendation System --- Development Workflow

> **Purpose:** Extend the existing Crop Recommendation System into a
> Smart Crop Recommendation and Farm Profit Optimisation System while
> preserving the current project and its existing ML prediction
> capability.
>
> **Repository:**
> https://github.com/mayanks53262-ctrl/Crop-Recommendation-System\
> **Default branch shown in the supplied screenshot:** `main`

## 1. Current repository baseline

The supplied repository screenshot shows:

``` text
Crop-Recommendation-System/
├── frontend/
├── src/
├── .gitignore
├── .python-version
├── README.md
├── pyproject.toml
├── requirements-dev.txt
├── requirements-research.txt
├── requirements.txt
└── vercel.json
```

This document is a development plan, not a claim that the planned
modules or APIs already exist. Inspect the current code before adding or
moving application files.

**Preservation rules** - Do not replace the existing project with a new
scaffold. - Do not delete, reset, overwrite or reformat existing work
without approval. - Preserve the current ML model, its input schema and
existing `/predict` endpoint. - Keep ML prediction as a separate
feature. - Inspect Git status and protect uncommitted work before
changes. - Keep the separate local `crop workflow` folder separate
unless copying selected files is explicitly approved.

## 2. Product objective

Build a farmer-facing system combining: 1. Farmer and farm input
collection. 2. Crop suitability recommendations. 3. Itemised cultivation
cost and profit estimates. 4. Crop and crop-combination comparisons. 5.
Land allocation and farm-plan optimisation. 6. Location, weather and
market-price context. 7. Explanations, data sources, assumptions and
uncertainty.

Never guarantee profit. Distinguish measured, farmer-reported, estimated
and externally sourced values. Unknown values must remain `null`, not
silently become zero.

## 3. Three-member responsibility split

### Member 1 --- Crop Recommendation and Suitability Engine

**Owns:** crop profiles, agronomic reference data, suitability scoring
and recommendation API.

Responsibilities: - Audit assigned agronomic tables and document fields,
units and limitations. - Maintain crop profiles with stable crop IDs. -
Represent crop categories, seasons, duration, soil, water and climate
requirements. - Calculate explainable suitability scores from available
farm inputs. - Identify constraints, missing information and agronomic
warnings. - Return ranked recommendations with reasons and
limitations. - Add unit tests and API documentation.

Suggested module (adapt after inspecting the repository):

``` text
backend/crop_engine/
├── crop_database.json
├── suitability.py
├── recommendation.py
├── schemas.py
├── routes.py
└── tests/
```

Proposed endpoints:

``` text
GET  /api/crops/catalog
POST /api/crops/recommend
```

### Member 2 --- Economics and Farm Profit Optimisation

**Owns:** cost model, yield and price assumptions, financial
calculations, comparison and land allocation.

Responsibilities: - Define itemised costs: seeds/planting material,
fertiliser/manure, labour, irrigation/energy, crop protection,
machinery, harvesting/post-harvest, packaging/transport and other
relevant costs. - Calculate marketable production, revenue, total cost,
net return and benefit-cost ratio. - Support conservative, expected and
optimistic scenarios. - Compare crops using consistent units and stated
assumptions. - Optimise land allocation subject to land, budget, water,
season, labour and crop-compatibility constraints. - Validate crop IDs
against the shared crop master. - Add tests and API documentation.

Suggested module:

``` text
backend/economics/
├── cost_calculator.py
├── revenue_calculator.py
├── profit_calculator.py
├── scenarios.py
├── optimizer.py
├── schemas.py
├── routes.py
└── tests/
```

Proposed endpoints:

``` text
POST /api/economics/calculate
POST /api/economics/compare
POST /api/economics/optimise
```

### Member 3 --- Integration Lead (project owner)

**Owns:** shared data contract, existing application, live-data
integrations, frontend integration, end-to-end testing and deployment.

Responsibilities: - Audit the repository and 12 reference tables. -
Establish canonical crop IDs, shared schemas, units and provenance
rules. - Preserve the existing ML model and `/predict` endpoint
unchanged. - Integrate farmer input, recommendations and economics into
the existing frontend. - Integrate geocoding and weather after
validating terms and limitations. - Investigate mandi/market data only
after validating access, licensing, coverage and freshness. - Coordinate
API contracts and code reviews. - Run end-to-end tests and maintain the
main branch/deployment.

Suggested integration area (adapt after inspecting the repository):

``` text
backend/integrations/
├── geocoding.py
├── weather.py
└── market_prices.py
```

Do not assume these folders exist yet.

## 4. Reference-table allocation

The mapping below uses the filenames and descriptions previously
supplied. Validate actual columns, contents, units and provenance before
using values.

  --------------------------------------------------------------------------
  Table                Working           Primary owner     Shared use
                       description                         
  -------------------- ----------------- ----------------- -----------------
  `table (1).csv`      Soil reference    Member 1          Member 3
                                                           validates
                                                           source/units

  `table (2).csv`      Water             Member 1          Member 2 uses
                       requirements                        water constraints

  `table (3).csv`      Crop seasons      Member 1          Member 2 uses
                                                           seasonal
                                                           constraints

  `table (6).csv`      Data sources      Member 3          All members

  `table (7).csv`      Crop master       Member 3          Member 2 consumes
                                         coordinates;      stable crop IDs
                                         Member 1          
                                         maintains         
                                         agronomic         
                                         profiles          

  `table (8).csv`      Climate           Member 1          Member 3 checks
                       requirements                        consistency

  `table (8)(1).csv`   Possible          Member 3 audits   Do not
                       climate-table                       choose/delete
                       duplicate                           until compared

  `table (9).csv`      Soil requirements Member 1          Member 3 checks
                                                           consistency

  `table (10).csv`     Regional          Member 3          Members 1 and 2
                       parameters        coordinates       use verified
                                                           relevant fields

  `table (11).csv`     Crop priority     Member 1          Member 2 may use
                       list                                as input, not as
                                                           a profit ranking

  `table (12).csv`     Parameter         Member 3          All members
                       definitions       coordinates       
  --------------------------------------------------------------------------

Table ownership means responsibility for interpretation and proposed
changes. It does not mean maintaining independent conflicting copies.
The project owner coordinates canonical shared versions.

## 5. Common data contract

Agree on this contract before parallel implementation.

### Stable crop identifiers

Every crop must have one stable ID shared by all modules, for example:

``` text
WHEAT_001
RICE_001
MAIZE_001
MUSTARD_001
```

Do not use display names as database keys. Document synonyms and
regional names.

### Example farmer input

``` json
{
  "location": {
    "country": "India",
    "state": null,
    "district": null,
    "village": null,
    "latitude": null,
    "longitude": null
  },
  "land": {"area": null, "unit": "acre"},
  "soil": {
    "type": null,
    "ph": null,
    "nitrogen": null,
    "phosphorus": null,
    "potassium": null,
    "soil_test_available": false
  },
  "irrigation": {"source": null, "availability": null},
  "season": null,
  "budget": {"amount": null, "currency": "INR"},
  "preferences": {"risk": "moderate", "interested_categories": []}
}
```

### Example recommendation response

``` json
{
  "status": "success",
  "recommendations": [
    {
      "crop_id": "WHEAT_001",
      "crop_name": "Wheat",
      "suitability_score": null,
      "reasons": [],
      "warnings": [],
      "missing_inputs": []
    }
  ],
  "data_sources": [],
  "assumptions": []
}
```

The score is `null` deliberately: do not invent a score if required
inputs or a validated scoring method are unavailable.

### Example economics response

``` json
{
  "crop_id": "WHEAT_001",
  "area": null,
  "area_unit": "acre",
  "currency": "INR",
  "cost_breakdown": [],
  "expected_yield": null,
  "yield_unit": null,
  "selling_price": null,
  "gross_revenue": null,
  "total_cost": null,
  "net_return": null,
  "benefit_cost_ratio": null,
  "scenario": "expected",
  "assumptions": [],
  "data_sources": []
}
```

### Data quality and provenance rules

For each important value, record where possible: - value and unit -
source name and source URL/citation - geographic coverage - reference
year/effective date - verification status - whether measured,
farmer-reported, modelled or estimated - method or notes

Never confuse fertiliser applied with measured soil nutrients. Never
infer soil NPK or pH from qualitative soil descriptions, photos or
weather. Unknown values remain `null`.

## 6. Data-use guidance

The collected material includes crop recommendation training data, rice
research and nutrient-use material, maize experiment/economics data,
Dryad research material and 12 reference tables. Use each only for
purposes supported by its actual fields and documentation.

-   The 2,200-row crop recommendation CSV is a baseline for the existing
    ML model; it is not a farm-profit dataset.
-   Rice experiment and nutrient-use files support rice-specific
    analysis; do not automatically generalise them to all crops.
-   Maize experiment files can inform maize-specific cost and yield
    analysis; geography and study period limit generalisation.
-   Dryad research data requires careful interpretation of study,
    treatment, soil and yield fields.
-   Reference tables support agronomic knowledge and suitability rules
    only after checking source, units, region and crop coverage.

Do not combine datasets just because columns have similar names.
Determine whether rows represent farms, plots, experiments, treatment
means, district aggregates or reference ranges.

## 7. API integration sequence

1.  Validate farmer input against the shared schema.
2.  Member 1's crop engine returns ranked crop IDs, suitability
    explanations and warnings.
3.  Member 2's economics engine receives those crop IDs plus farm area,
    budget and relevant assumptions.
4.  The integration layer enriches results with location, weather and
    market context where verified data is available.
5.  The frontend displays recommendations, costs, revenue, profit
    scenarios, comparisons and limitations.

  Endpoint                          Owner
  --------------------------------- ---------------------------------
  `GET /api/crops/catalog`          Member 1
  `POST /api/crops/recommend`       Member 1
  `POST /api/economics/calculate`   Member 2
  `POST /api/economics/compare`     Member 2
  `POST /api/economics/optimise`    Member 2
  Weather/geocoding integration     Member 3
  Market-price integration          Member 3
  Existing `/predict`               Preserve current implementation

Do not change the existing `/predict` request or response contract.

## 8. Git and collaboration workflow

Use separate feature branches:

``` text
main
├── feature/crop-engine
├── feature/economics
└── feature/data-integration
```

Rules: - Start from an agreed current `main` commit. - Inspect
`git status` before switching branches or editing. - Protect local
uncommitted work; do not use destructive reset/clean commands. - Keep
each member's work in their assigned module. - Do not let multiple
members independently edit shared schemas, app entry points or crop
master files. - Open pull requests into `main`. - Require tests and
review before merging. - Merge the shared data contract first, then crop
and economics modules, then frontend integration. - Never commit API
keys, passwords, private farmer data or local environment files.

## 9. Development phases and acceptance criteria

### Phase 1 --- Audit and contract

All members inspect assigned source files and document fields, units,
coverage and limitations. Member 3 publishes crop ID conventions,
schemas and provenance rules.

Acceptance criteria: - Table inventory completed. - Duplicate climate
table compared. - Crop IDs agreed. - Units and unknown-value rules
agreed. - API request/response examples approved.

### Phase 2 --- Independent development

-   **Member 1:** crop catalog, suitability logic, explanations,
    warnings and tests.
-   **Member 2:** cost/revenue/profit calculations, scenarios,
    comparison, optimisation and tests.
-   **Member 3:** frontend and live-data integration, without changing
    the existing ML interface.

Acceptance criteria: - Each module runs independently. - Each module has
documented endpoints and sample requests. - Tests cover valid, invalid
and missing inputs. - No unsupported agricultural or financial values
are fabricated.

### Phase 3 --- Integration

-   Connect recommendation output to economics input using stable crop
    IDs.
-   Connect shared farmer input to both modules.
-   Add verified weather and market context.
-   Display missing data and assumptions explicitly.

Acceptance criteria: - End-to-end sample farm request succeeds. - Units
and area conversions are tested. - API errors and unavailable external
sources are handled. - Existing ML prediction still works.

### Phase 4 --- Validation and release

-   Validate suitability rules with reliable regional agronomy
    references.
-   Validate financial formulas and test cases.
-   Test missing-data cases.
-   Test frontend responsiveness and English/Hindi UI if supported by
    the current app.
-   Update README and deployment instructions.
-   Merge reviewed pull requests and deploy only after approval.

## 10. Product safety and transparency

-   Show that returns are estimates, not guaranteed profits.
-   Show source, geography, date and assumptions for prices and yields.
-   Distinguish wholesale mandi prices from the price a farmer may
    actually receive.
-   Do not label a crop "best" based on profit alone; consider
    suitability, water, season, budget and risk.
-   Do not model long-term orchard investment using annual-crop
    assumptions.
-   Make unavailable or stale data visible.
-   Do not silently substitute a national average for a local value.

## 11. Immediate next actions

1.  The project owner adds this workflow document to the repository.
2.  All three members review and agree on responsibilities and the
    common data contract.
3.  Member 3 completes a read-only audit of the 12 tables and current
    repository.
4.  The team agrees on the final crop master and API schemas.
5.  Members 1 and 2 begin implementation in their own feature branches.
6.  Member 3 integrates after module contracts and tests are available.

------------------------------------------------------------------------

**Status:** Proposed team workflow. Update through reviewed pull
requests as implementation evolves. This document describes planned
responsibilities and interfaces; it does not claim the proposed modules,
endpoints or data have already been implemented or validated.
