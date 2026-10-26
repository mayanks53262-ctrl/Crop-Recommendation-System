export class ApiError extends Error {
  constructor(kind, fields = []) {
    super(kind);
    this.name = "ApiError";
    this.kind = kind;
    this.fields = fields;
  }
}

function apiUrl(path) {
  const configured = import.meta.env.VITE_API_BASE_URL || "";
  return `${configured.replace(/\/+$/, "")}${path}`;
}

async function readJson(response) {
  try {
    return await response.json();
  } catch {
    throw new ApiError("unexpected");
  }
}

function validationFields(body) {
  if (!Array.isArray(body?.detail)) return [];
  return body.detail.flatMap((item) => {
    const field = Array.isArray(item?.loc) ? item.loc.at(-1) : null;
    return typeof field === "string" && field !== "body" ? [field] : [];
  });
}

function validatePredictionResponse(body) {
  if (
    !body ||
    typeof body.predicted_crop !== "string" ||
    !Array.isArray(body.warnings) ||
    !(body.top_three === null || body.top_three === undefined || Array.isArray(body.top_three)) ||
    !(body.score_type === null || body.score_type === undefined || typeof body.score_type === "string")
  ) {
    throw new ApiError("unexpected");
  }

  const topThree = body.top_three ?? [];
  if (
    topThree.some(
      (entry) =>
        typeof entry?.crop !== "string" ||
        typeof entry?.score !== "number" ||
        !Number.isFinite(entry.score) ||
        entry.score < 0 ||
        entry.score > 1,
    )
  ) {
    throw new ApiError("unexpected");
  }

  if (
    body.warnings.some(
      (warning) =>
        typeof warning?.feature !== "string" ||
        typeof warning?.value !== "number" ||
        typeof warning?.observed_min !== "number" ||
        typeof warning?.observed_max !== "number",
    )
  ) {
    throw new ApiError("unexpected");
  }
  return { ...body, top_three: topThree };
}

export async function predictCrop(values, fetchImpl = fetch) {
  let response;
  try {
    response = await fetchImpl(apiUrl("/predict"), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(values),
    });
  } catch {
    throw new ApiError("network");
  }

  const body = await readJson(response);
  if (response.status === 422) {
    throw new ApiError("validation", validationFields(body));
  }
  if (!response.ok) {
    throw new ApiError(response.status >= 500 ? "unavailable" : "unexpected");
  }
  return validatePredictionResponse(body);
}

export async function getCropCatalog(fetchImpl = fetch) {
  let response;
  try {
    response = await fetchImpl(apiUrl("/crop-catalog"), { method: "GET" });
  } catch {
    throw new ApiError("network");
  }
  const body = await readJson(response);
  if (!response.ok) {
    throw new ApiError(response.status >= 500 ? "unavailable" : "unexpected");
  }
  if (
    !body ||
    !Array.isArray(body.crops) ||
    typeof body.research_status?.status !== "string" ||
    typeof body.research_status?.message !== "string" ||
    body.crops.some(
      (crop) =>
        typeof crop?.crop_name !== "string" ||
        typeof crop?.verification_status !== "string" ||
        typeof crop?.model_support !== "string" ||
        !Array.isArray(crop?.sources),
    )
  ) {
    throw new ApiError("unexpected");
  }
  return body;
}

export async function getWorkflowStatus(fetchImpl = fetch) {
  let response;
  try {
    response = await fetchImpl(apiUrl("/assessment/workflow-status"), { method: "GET" });
  } catch {
    throw new ApiError("network");
  }
  const body = await readJson(response);
  if (!response.ok) {
    throw new ApiError(response.status >= 500 ? "unavailable" : "unexpected");
  }
  if (
    body?.workflow_name !== "Agricultural Field Assessment" ||
    body?.status !== "not_connected" ||
    body?.connected !== false ||
    typeof body?.message !== "string"
  ) {
    throw new ApiError("unexpected");
  }
  return body;
}

export async function checkApiHealth(fetchImpl = fetch) {
  let response;
  try {
    response = await fetchImpl(apiUrl("/health"), { method: "GET" });
  } catch {
    return false;
  }
  if (!response.ok) return false;
  try {
    const body = await response.json();
    return body?.model_ready === true;
  } catch {
    return false;
  }
}
