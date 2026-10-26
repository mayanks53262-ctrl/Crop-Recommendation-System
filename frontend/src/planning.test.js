import { describe, expect, it } from "vitest";
import { calculateProfitability } from "./planning.js";

function candidate(overrides = {}) {
  return {
    name: "Wheat",
    yieldPerArea: "1000",
    price: "20",
    totalCost: "0",
    ...overrides,
  };
}

describe("calculateProfitability", () => {
  it("calculates revenue, total costs, net return, and revenue-to-cost ratio", () => {
    const result = calculateProfitability("2", candidate({ totalCost: "10000" }));

    expect(result).toEqual({
      missing: [],
      invalid: [],
      revenue: 40000,
      totalCost: 10000,
      netReturn: 30000,
      benefitCostRatio: 4,
    });
  });

  it("does not calculate when any assumption is blank, including crop name or costs", () => {
    const result = calculateProfitability("1", candidate({ name: "", totalCost: "" }));

    expect(result.revenue).toBeNull();
    expect(result.missing).toContain("cost");
    expect(result.missing).toContain("crop");
  });

  it("accepts explicitly entered zero costs and defines a zero-cost ratio as undefined", () => {
    const result = calculateProfitability("1", candidate());

    expect(result.totalCost).toBe(0);
    expect(result.netReturn).toBe(20000);
    expect(result.benefitCostRatio).toBeNull();
  });

  it("rejects negative or non-finite assumptions and zero area", () => {
    const result = calculateProfitability(
      "0",
      candidate({ price: "-2", totalCost: "Infinity" }),
    );

    expect(result.revenue).toBeNull();
    expect(result.invalid).toEqual(expect.arrayContaining(["area", "price", "cost"]));
  });
});
