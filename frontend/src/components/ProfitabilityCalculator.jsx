import { useState } from "react";
import { useTranslation } from "react-i18next";
import { calculateProfitability } from "../planning.js";

const emptyCrop = (id) => ({ id, name: "", yieldPerArea: "", price: "", totalCost: "" });
const currency = new Intl.NumberFormat("en-IN", {
  style: "currency",
  currency: "INR",
  maximumFractionDigits: 2,
});

export function ProfitabilityCalculator({ predictedCrop }) {
  const { t } = useTranslation();
  const [area, setArea] = useState("");
  const [areaUnit, setAreaUnit] = useState("acre");
  const [crops, setCrops] = useState([emptyCrop(1), emptyCrop(2)]);

  function updateCrop(id, key, value) {
    setCrops((current) =>
      current.map((crop) => (crop.id === id ? { ...crop, [key]: value } : crop)),
    );
  }

  return (
    <section className="compact-profitability" aria-label={t("profitability.title")}>
      <div className="area-input-row">
        <label className="planning-field">
          <span>{t("profitability.fieldArea")}</span>
          <input
            type="number"
            min="0"
            step="any"
            value={area}
            onChange={(event) => setArea(event.target.value)}
          />
        </label>
        <label className="planning-field">
          <span>{t("profitability.areaUnit")}</span>
          <select value={areaUnit} onChange={(event) => setAreaUnit(event.target.value)}>
            <option value="acre">{t("profitability.acre")}</option>
            <option value="hectare">{t("profitability.hectare")}</option>
          </select>
        </label>
      </div>
      <div className="profit-crop-list">
        {crops.map((crop, index) => {
          const estimate = calculateProfitability(area, crop);
          return (
            <article className="profit-crop-card" key={crop.id}>
              <h3>{crop.name.trim() || t("profitability.cropNumber", { number: index + 1 })}</h3>
              <div className="planning-grid">
                <label className="planning-field">
                  <span>{t("profitability.cropName")}</span>
                  <input
                    value={crop.name}
                    onChange={(event) => updateCrop(crop.id, "name", event.target.value)}
                    placeholder={predictedCrop || t("profitability.cropPlaceholder")}
                    autoComplete="off"
                  />
                </label>
                <label className="planning-field">
                  <span>{t("profitability.expectedYield", { unit: t(`profitability.${areaUnit}`) })}</span>
                  <input
                    type="number"
                    min="0"
                    step="any"
                    value={crop.yieldPerArea}
                    onChange={(event) => updateCrop(crop.id, "yieldPerArea", event.target.value)}
                  />
                </label>
                <label className="planning-field">
                  <span>{t("profitability.sellingPrice")}</span>
                  <input
                    type="number"
                    min="0"
                    step="any"
                    value={crop.price}
                    onChange={(event) => updateCrop(crop.id, "price", event.target.value)}
                  />
                </label>
                <label className="planning-field">
                  <span>{t("profitability.totalCostInput")}</span>
                  <input
                    type="number"
                    min="0"
                    step="any"
                    value={crop.totalCost}
                    onChange={(event) => updateCrop(crop.id, "totalCost", event.target.value)}
                  />
                </label>
              </div>
              {estimate.missing.length > 0 && (
                <p className="assumption-note" role="status">
                  {t("profitability.missing", {
                    fields: estimate.missing.map((key) => t(`profitability.fields.${key}`)).join(", "),
                  })}
                </p>
              )}
              {estimate.invalid.length > 0 && (
                <p className="error-banner" role="alert">{t("profitability.invalid")}</p>
              )}
              {estimate.revenue !== null && (
                <dl className="profit-results">
                  <div><dt>{t("profitability.revenue")}</dt><dd>{currency.format(estimate.revenue)}</dd></div>
                  <div><dt>{t("profitability.totalCost")}</dt><dd>{currency.format(estimate.totalCost)}</dd></div>
                  <div><dt>{t("profitability.netReturn")}</dt><dd>{currency.format(estimate.netReturn)}</dd></div>
                  <div>
                    <dt>{t("profitability.benefitCostRatio")}</dt>
                    <dd>{estimate.benefitCostRatio === null ? t("profitability.notDefined") : `${estimate.benefitCostRatio.toFixed(2)} : 1`}</dd>
                  </div>
                </dl>
              )}
              <p className="estimate-source">{t("profitability.userEntered")}</p>
            </article>
          );
        })}
      </div>
      <p className="assessment-footer">{t("profitability.formulaNote")}</p>
    </section>
  );
}
