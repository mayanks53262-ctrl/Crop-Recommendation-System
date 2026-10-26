import { afterEach, describe, expect, it, vi } from "vitest";
import { ApiError, getWorkflowStatus, predictCrop } from "./client.js";

const disconnectedStatus = {
  workflow_name: "Agricultural Field Assessment",
  status: "not_connected",
  connected: false,
  message: "Workflow configuration is not available.",
};

function mockResponse(body, status = 200) {
  return {
    ok: status >= 200 && status < 300,
    status,
    json: vi.fn().mockResolvedValue(body),
  };
}

afterEach(() => {
  vi.unstubAllEnvs();
});

describe("predictCrop", () => {
  it("uses the root-relative prediction route when no API base URL is configured", async () => {
    vi.stubEnv("VITE_API_BASE_URL", "");
    const values = {
      N: 90,
      P: 42,
      K: 43,
      temperature: 20.88,
      humidity: 82,
      ph: 6.5,
      rainfall: 202.94,
    };
    const fetchImpl = vi.fn().mockResolvedValue(mockResponse({
      predicted_crop: "rice",
      top_three: null,
      score_type: null,
      warnings: [],
    }));

    await expect(predictCrop(values, fetchImpl)).resolves.toMatchObject({
      predicted_crop: "rice",
      top_three: [],
    });
    expect(fetchImpl).toHaveBeenCalledWith("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(values),
    });
  });
});

describe("getWorkflowStatus", () => {
  it("returns the backend disconnected state without inventing workflow output", async () => {
    const fetchImpl = vi.fn().mockResolvedValue(mockResponse(disconnectedStatus));

    await expect(getWorkflowStatus(fetchImpl)).resolves.toEqual(disconnectedStatus);
    expect(fetchImpl.mock.calls[0][0]).toContain("/assessment/workflow-status");
    expect(fetchImpl.mock.calls[0][1].method).toBe("GET");
  });

  it("rejects an invalid status response instead of displaying it as connected", async () => {
    const fetchImpl = vi.fn().mockResolvedValue(mockResponse({
      ...disconnectedStatus,
      status: "connected",
      connected: true,
    }));

    await expect(getWorkflowStatus(fetchImpl)).rejects.toMatchObject({
      name: "ApiError",
      kind: "unexpected",
    });
  });

  it("surfaces authentication and service failures as unavailable status errors", async () => {
    for (const status of [401, 503]) {
      await expect(
        getWorkflowStatus(vi.fn().mockResolvedValue(mockResponse({}, status))),
      ).rejects.toBeInstanceOf(ApiError);
    }
  });

  it("surfaces network and malformed JSON failures", async () => {
    await expect(getWorkflowStatus(vi.fn().mockRejectedValue(new Error("offline"))))
      .rejects.toMatchObject({ kind: "network" });
    await expect(getWorkflowStatus(vi.fn().mockResolvedValue({
      ok: true,
      status: 200,
      json: vi.fn().mockRejectedValue(new Error("invalid JSON")),
    }))).rejects.toMatchObject({ kind: "unexpected" });
  });
});
