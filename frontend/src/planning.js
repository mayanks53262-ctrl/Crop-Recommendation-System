export function calculateProfitability(areaValue, crop) {
  const missing = [];
  const invalid = [];
  const readNumber = (key, value, { positive = false } = {}) => {
    if (value === "" || value === null || value === undefined) {
      missing.push(key);
      return null;
    }
    const number = Number(value);
    if (!Number.isFinite(number) || number < 0 || (positive && number === 0)) {
      invalid.push(key);
      return null;
    }
    return number;
  };

  const area = readNumber("area", areaValue, { positive: true });
  if (!crop.name?.trim()) missing.push("crop");
  const yieldPerArea = readNumber("yield", crop.yieldPerArea);
  const price = readNumber("price", crop.price);
  const totalCost = readNumber("cost", crop.totalCost);
  if (missing.length || invalid.length) {
    return { missing, invalid, revenue: null, totalCost: null, netReturn: null, benefitCostRatio: null };
  }

  const revenue = area * yieldPerArea * price;
  return {
    missing,
    invalid,
    revenue,
    totalCost,
    netReturn: revenue - totalCost,
    benefitCostRatio: totalCost === 0 ? null : revenue / totalCost,
  };
}
