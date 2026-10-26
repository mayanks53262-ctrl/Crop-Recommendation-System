import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import App from "./App.jsx";
import i18n from "./i18n.js";
import { ApiError, checkApiHealth, getWorkflowStatus, predictCrop } from "./api/client.js";

vi.mock("./api/client.js", () => ({
  ApiError: class ApiError extends Error {
    constructor(kind, fields = []) {
      super(kind);
      this.kind = kind;
      this.fields = fields;
    }
  },
  checkApiHealth: vi.fn(),
  getWorkflowStatus: vi.fn(),
  predictCrop: vi.fn(),
}));

const response = {
  predicted_crop: "rice",
  top_three: [
    { crop: "rice", score: 0.73 },
    { crop: "jute", score: 0.14 },
    { crop: "cotton", score: 0.08 },
  ],
  score_type: "model probability output; not agricultural success probability",
  warnings: [],
};

beforeEach(() => {
  checkApiHealth.mockResolvedValue(false);
  getWorkflowStatus.mockResolvedValue({
    workflow_name: "Agricultural Field Assessment",
    status: "not_connected",
    connected: false,
    message: "No verified invocation configuration or response contract is available.",
  });
  predictCrop.mockResolvedValue(response);
});

it("shows local validation feedback and does not call the model API for negative nutrients", () => {
  render(<App />);
  fillValidForm();
  fireEvent.change(screen.getByLabelText("Nitrogen (N)"), { target: { value: "-1" } });
  fireEvent.submit(document.querySelector("form"));

  expect(screen.getByRole("alert")).toHaveTextContent("All values except temperature must be non-negative");
  expect(screen.getByLabelText("Nitrogen (N)")).toHaveAttribute("aria-invalid", "true");
  expect(predictCrop).not.toHaveBeenCalled();
});

afterEach(() => {
  vi.clearAllMocks();
  window.localStorage.clear();
});

function fillValidForm() {
  const fields = {
    "Nitrogen (N)": "90",
    "Phosphorus (P)": "42",
    "Potassium (K)": "43",
    Temperature: "20.5",
    Humidity: "82",
    "Soil pH": "6.5",
    Rainfall: "202",
  };
  for (const [label, value] of Object.entries(fields)) {
    fireEvent.change(screen.getByLabelText(label), { target: { value } });
  }
}

describe("Crop Recommendation website", () => {
  it("renders all seven required model inputs", () => {
    render(<App />);
    for (const label of [
      "Nitrogen (N)",
      "Phosphorus (P)",
      "Potassium (K)",
      "Temperature",
      "Humidity",
      "Soil pH",
      "Rainfall",
    ]) {
      expect(screen.getByLabelText(label)).toBeRequired();
    }
  });

  it("reports the agricultural workflow as disconnected and keeps ML separate", async () => {
    render(<App />);
    expect(await screen.findByText("Not connected")).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "Agricultural Field Assessment" })).toBeInTheDocument();
    expect(screen.getByText(/No workflow connection is configured/)).toBeInTheDocument();
    expect(screen.getByText("Independent ML model")).toBeInTheDocument();
  });

  it("shows an unavailable state if workflow status cannot be reached", async () => {
    getWorkflowStatus.mockRejectedValue(new ApiError("network"));
    render(<App />);
    expect(await screen.findByText("Status unavailable")).toBeInTheDocument();
    expect(screen.getByText("Workflow status could not be retrieved. No assessment has been submitted.")).toBeInTheDocument();
  });

  it("does not collect disconnected workflow details or expose a fake assessment form", () => {
    render(<App />);
    expect(screen.queryByLabelText("Irrigation source")).not.toBeInTheDocument();
    expect(screen.queryByText("Optional field details recorded")).not.toBeInTheDocument();
    expect(screen.getByText("Optional: compare crop economics")).toBeInTheDocument();
  });

  it("keeps profitability optional and calculates only after explicit assumptions", async () => {
    const user = userEvent.setup();
    render(<App />);
    expect(document.querySelector("details")?.open).toBe(false);
    await user.click(screen.getByText("Optional: compare crop economics"));
    fireEvent.change(screen.getByLabelText("Field area"), { target: { value: "2" } });
    fireEvent.change(screen.getAllByLabelText("Crop name")[0], { target: { value: "Wheat" } });
    fireEvent.change(screen.getAllByLabelText("Expected yield (kg per acre)")[0], { target: { value: "1000" } });
    fireEvent.change(screen.getAllByLabelText("Selling price (INR per kg)")[0], { target: { value: "20" } });
    fireEvent.change(screen.getAllByLabelText("Total estimated costs (INR for field)")[0], { target: { value: "10000" } });

    expect(screen.getByText("Estimated revenue")).toBeInTheDocument();
    expect(screen.getByText(/40,000/)).toBeInTheDocument();
    expect(screen.getByText(/30,000/)).toBeInTheDocument();
  });

  it("submits the actual form values and displays the returned prediction and alternatives", async () => {
    const user = userEvent.setup();
    render(<App />);
    fillValidForm();
    await user.click(screen.getByRole("button", { name: "Run crop prediction" }));

    await screen.findByRole("heading", { name: "Predicted crop label" });
    const results = document.querySelector(".results-card");
    expect(results.querySelector(".result-crop")).toHaveTextContent("Rice");
    expect(within(results).getByText("Jute")).toBeInTheDocument();
    expect(within(results).getByText("Cotton")).toBeInTheDocument();
    expect(within(results).getAllByText("Model output score")).toHaveLength(2);
    expect(predictCrop).toHaveBeenCalledWith({
      N: 90,
      P: 42,
      K: 43,
      temperature: 20.5,
      humidity: 82,
      ph: 6.5,
      rainfall: 202,
    });
  });

  it("renders range warnings returned by the API", async () => {
    const user = userEvent.setup();
    predictCrop.mockResolvedValue({
      ...response,
      warnings: [
        {
          feature: "N",
          value: 141,
          observed_min: 0,
          observed_max: 140,
          message: "outside range",
        },
      ],
    });
    render(<App />);
    fillValidForm();
    await user.click(screen.getByRole("button", { name: "Run crop prediction" }));
    expect(await screen.findByText("Outside observed training range")).toBeInTheDocument();
    expect(screen.getByText(/Value 141 is outside the training-data range 0 to 140/)).toBeInTheDocument();
  });

  it("shows localized feedback for API validation errors", async () => {
    const user = userEvent.setup();
    const error = new ApiError("validation", ["N"]);
    predictCrop.mockRejectedValue(error);
    render(<App />);
    fillValidForm();
    await user.click(screen.getByRole("button", { name: "Run crop prediction" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("The API could not accept these values");
    expect(screen.getByRole("alert")).toHaveTextContent("Nitrogen (N)");
  });

  it("handles unavailable backend without crashing", async () => {
    const user = userEvent.setup();
    const error = new ApiError("network");
    predictCrop.mockRejectedValue(error);
    render(<App />);
    fillValidForm();
    await user.click(screen.getByRole("button", { name: "Run crop prediction" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("The API could not be reached");
    expect(screen.getByText("Prediction result")).toBeInTheDocument();
  });

  it("disables duplicate form submissions while a request is pending", async () => {
    const user = userEvent.setup();
    let finishRequest;
    predictCrop.mockReturnValue(new Promise((resolve) => { finishRequest = resolve; }));
    render(<App />);
    fillValidForm();
    const submit = screen.getByRole("button", { name: "Run crop prediction" });
    await user.click(submit);

    expect(await screen.findByRole("button", { name: /Running crop prediction/ })).toBeDisabled();
    expect(predictCrop).toHaveBeenCalledTimes(1);
    finishRequest(response);
    await screen.findByRole("heading", { name: "Predicted crop label" });
  });

  it("loads the genuine project sample feature values", async () => {
    const user = userEvent.setup();
    render(<App />);
    await user.click(screen.getByRole("button", { name: "Load sample data" }));
    expect(screen.getByLabelText("Nitrogen (N)")).toHaveValue(90);
    expect(screen.getByLabelText("Phosphorus (P)")).toHaveValue(42);
    expect(screen.getByText(/A real feature row/)).toBeInTheDocument();
  });

  it("switches to Hindi, persists the preference, and keeps layouts wrap-friendly", async () => {
    const user = userEvent.setup();
    render(<App />);
    await user.selectOptions(screen.getByLabelText("Language"), "hi");

    await waitFor(() => expect(screen.getByRole("heading", { level: 1 })).toHaveTextContent("Agricultural Field Assessment"));
    expect(window.localStorage.getItem("crop-language")).toBe("hi");
    expect(screen.getByLabelText("नाइट्रोजन (N)")).toBeInTheDocument();
    expect(document.documentElement).toHaveAttribute("lang", "hi");
    expect(i18n.resolvedLanguage).toBe("hi");
  });

  it("uses the crop-label fallback without changing API identifiers", async () => {
    const user = userEvent.setup();
    predictCrop.mockResolvedValue({ ...response, predicted_crop: "unknown-crop" });
    render(<App />);
    fillValidForm();
    await user.click(screen.getByRole("button", { name: "Run crop prediction" }));
    expect(await screen.findByText("unknown-crop")).toBeInTheDocument();
  });
});
