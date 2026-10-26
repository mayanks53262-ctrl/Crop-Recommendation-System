import { useTranslation } from "react-i18next";
import { useState } from "react";

export const FEATURE_FIELDS = [
  { key: "N", translationKey: "fields.nitrogen", nonNegative: true },
  { key: "P", translationKey: "fields.phosphorus", nonNegative: true },
  { key: "K", translationKey: "fields.potassium", nonNegative: true },
  { key: "temperature", translationKey: "fields.temperature", nonNegative: false },
  { key: "humidity", translationKey: "fields.humidity", nonNegative: true },
  { key: "ph", translationKey: "fields.ph", nonNegative: true },
  { key: "rainfall", translationKey: "fields.rainfall", nonNegative: true },
];

export const SAMPLE_VALUES = {
  N: "90",
  P: "42",
  K: "43",
  temperature: "20.87974371",
  humidity: "82.00274423",
  ph: "6.502985292",
  rainfall: "202.9355362",
};

export function RecommendationForm({
  values,
  onChange,
  onSubmit,
  onLoadSample,
  isLoading,
  formRef,
  errorMessage,
}) {
  const { t } = useTranslation();
  const [validationError, setValidationError] = useState("");
  const [invalidField, setInvalidField] = useState("");

  function submit(event) {
    event.preventDefault();
    const formData = new FormData(event.currentTarget);
    const numericValues = {};
    for (const { key, nonNegative } of FEATURE_FIELDS) {
      const rawValue = formData.get(key);
      const value = Number(rawValue);
      if (rawValue === "" || !Number.isFinite(value) || (nonNegative && value < 0)) {
        event.currentTarget.elements.namedItem(key)?.focus();
        setValidationError(t("form.invalid"));
        setInvalidField(key);
        return;
      }
      numericValues[key] = value;
    }
    setValidationError("");
    setInvalidField("");
    onSubmit(numericValues);
  }

  return (
    <form ref={formRef} className="form-card" onSubmit={submit} aria-describedby="measurement-note">
      <div className="form-heading">
        <div>
          <span className="eyebrow">{t("form.eyebrow")}</span>
          <h3>{t("form.title")}</h3>
          <p>{t("form.subtitle")}</p>
        </div>
        <button className="text-button" type="button" onClick={onLoadSample} disabled={isLoading}>
          {t("form.loadSample")}
        </button>
      </div>

      <p className="measurement-note" id="measurement-note">{t("form.measurementNote")}</p>

      <div className="field-grid">
        {FEATURE_FIELDS.map(({ key, translationKey, nonNegative }) => (
          <div className="field" key={key}>
            <label htmlFor={`field-${key}`}>{t(translationKey)}</label>
            <input
              id={`field-${key}`}
              name={key}
              type="number"
              inputMode="decimal"
              step="any"
              min={nonNegative ? "0" : undefined}
              required
              value={values[key]}
              onChange={(event) => {
                onChange(key, event.target.value);
                setValidationError("");
                setInvalidField("");
              }}
              aria-invalid={invalidField === key}
              aria-describedby={`hint-${key}`}
              disabled={isLoading}
              autoComplete="off"
            />
            <span className="field-hint" id={`hint-${key}`}>{t(`fieldHints.${key}`)}</span>
          </div>
        ))}
      </div>

      <p className="form-footnote">{t("form.noMax")}</p>
      {validationError && <p className="error-banner" role="alert">{validationError}</p>}
      {errorMessage && (
        <p className="error-banner" role="alert">
          <span aria-hidden="true">!</span>{errorMessage}
        </p>
      )}
      <button className="button button-primary submit-button" type="submit" disabled={isLoading}>
        {isLoading ? (
          <>
            <span className="spinner" aria-hidden="true" />
            {t("form.loading")}
          </>
        ) : (
          <>
            {t("form.submit")}
            <span aria-hidden="true">↗</span>
          </>
        )}
      </button>
    </form>
  );
}
