import { useTranslation } from "react-i18next";
import { cropName } from "../data/crops.js";

export function ResultsPanel({ result, onEdit }) {
  const { t, i18n } = useTranslation();
  const language = i18n.resolvedLanguage === "hi" ? "hi" : "en";
  const alternatives = (result.top_three || [])
    .filter((entry) => entry.crop !== result.predicted_crop)
    .slice(0, 2);

  return (
    <section className="results-card" aria-labelledby="results-title" aria-live="polite">
      <div className="result-success-mark" aria-hidden="true">✓</div>
      <span className="eyebrow">{t("results.eyebrow")}</span>
      <h3 id="results-title">{t("results.title")}</h3>
      <p className="result-crop" lang={language}>{cropName(result.predicted_crop, language)}</p>
      <p className="result-caveat">{t("results.caveat")}</p>

      {alternatives.length > 0 && (
        <div className="alternatives">
          <h4>{t("results.alternatives")}</h4>
          {alternatives.map((entry) => (
            <div className="alternative-row" key={entry.crop}>
              <span lang={language}>{cropName(entry.crop, language)}</span>
              <span className="match-score">
                <small>{t("results.matchScore")}</small>
                {entry.score.toFixed(3)}
              </span>
            </div>
          ))}
          <p className="score-note">{t("results.scoreNote")}</p>
        </div>
      )}

      {result.warnings.length > 0 && (
        <div className="warning-panel" role="status">
          <h4><span aria-hidden="true">!</span>{t("results.warningsTitle")}</h4>
          <p>{t("results.warningsIntro")}</p>
          <ul>
            {result.warnings.map((warning, index) => (
              <li key={`${warning.feature}-${index}`}>
                <strong>{warning.feature}</strong>: {t("results.rangeWarning", {
                  value: warning.value,
                  minimum: warning.observed_min,
                  maximum: warning.observed_max,
                })}
              </li>
            ))}
          </ul>
        </div>
      )}

      <button className="button button-secondary result-edit" type="button" onClick={onEdit}>
        {t("results.edit")}
      </button>
    </section>
  );
}
