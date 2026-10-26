import { useEffect, useRef, useState } from "react";
import { useReducedMotion } from "framer-motion";
import { useTranslation } from "react-i18next";
import i18n from "./i18n.js";
import { ApiError, checkApiHealth, getWorkflowStatus, predictCrop } from "./api/client.js";
import { ProfitabilityCalculator } from "./components/ProfitabilityCalculator.jsx";
import {
  FEATURE_FIELDS,
  RecommendationForm,
  SAMPLE_VALUES,
} from "./components/RecommendationForm.jsx";
import { ResultsPanel } from "./components/ResultsPanel.jsx";

const EMPTY_VALUES = Object.fromEntries(FEATURE_FIELDS.map(({ key }) => [key, ""]));

function LeafMark() {
  return (
    <svg viewBox="0 0 36 36" aria-hidden="true" className="brand-mark">
      <path d="M29.5 6.5C18 6.7 8.4 9.8 7.2 18.1c-.9 6.1 4.5 9.2 9.1 7 7-3.3 11.5-12 13.2-18.6Z" fill="currentColor" />
      <path d="M7.5 30c3-7 8.4-11.3 16.4-15" fill="none" stroke="#f7f7ef" strokeWidth="1.8" strokeLinecap="round" />
    </svg>
  );
}

function App() {
  const { t, i18n: translation } = useTranslation();
  const reduceMotion = useReducedMotion();
  const [values, setValues] = useState(EMPTY_VALUES);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [apiReady, setApiReady] = useState(false);
  const [apiError, setApiError] = useState(null);
  const [workflowStatus, setWorkflowStatus] = useState(null);
  const [workflowStatusError, setWorkflowStatusError] = useState(false);
  const [sampleNotice, setSampleNotice] = useState("");
  const [language, setLanguage] = useState(translation.resolvedLanguage || "en");
  const formRef = useRef(null);
  const resultRef = useRef(null);

  useEffect(() => {
    let active = true;
    checkApiHealth().then((ready) => {
      if (active) setApiReady(ready);
    });
    getWorkflowStatus()
      .then((status) => {
        if (active) setWorkflowStatus(status);
      })
      .catch(() => {
        if (active) setWorkflowStatusError(true);
      });
    return () => {
      active = false;
    };
  }, []);

  function changeLanguage(event) {
    const nextLanguage = event.target.value === "hi" ? "hi" : "en";
    setLanguage(nextLanguage);
    window.localStorage.setItem("crop-language", nextLanguage);
    i18n.changeLanguage(nextLanguage);
    document.documentElement.lang = nextLanguage;
  }

  function updateField(key, value) {
    setValues((current) => ({ ...current, [key]: value }));
    setApiError(null);
    setSampleNotice("");
  }

  function loadSample() {
    setValues({ ...SAMPLE_VALUES });
    setApiError(null);
    setResult(null);
    setSampleNotice(t("form.sampleLoaded"));
  }

  function errorText(error) {
    if (!(error instanceof ApiError)) return t("errors.unexpected");
    if (error.kind === "validation") {
      const fieldLabels = error.fields.map((field) => {
        const item = FEATURE_FIELDS.find((candidate) => candidate.key === field);
        return item ? t(item.translationKey) : "";
      }).filter(Boolean);
      return fieldLabels.length
        ? `${t("errors.validation")} ${t("errors.fields", { fields: fieldLabels.join(", ") })}`
        : t("errors.validation");
    }
    return t(`errors.${error.kind}`, { defaultValue: t("errors.unexpected") });
  }

  async function submitPrediction(numericValues) {
    if (loading) return;
    setLoading(true);
    setApiError(null);
    setSampleNotice("");
    setResult(null);
    try {
      const prediction = await predictCrop(numericValues);
      setResult(prediction);
      setApiReady(true);
      window.requestAnimationFrame(() => {
        resultRef.current?.scrollIntoView({
          behavior: reduceMotion ? "auto" : "smooth",
          block: "nearest",
        });
      });
    } catch (error) {
      setApiError(errorText(error));
      if (error instanceof ApiError && error.kind === "network") setApiReady(false);
    } finally {
      setLoading(false);
    }
  }

  function editInputs() {
    formRef.current?.scrollIntoView({
      behavior: reduceMotion ? "auto" : "smooth",
      block: "center",
    });
    formRef.current?.querySelector("input")?.focus({ preventScroll: true });
  }

  return (
    <div className="app-shell">
      <header className="site-header">
        <div className="header-inner page-width">
          <a className="brand" href="#home" aria-label={t("brandSubtitle")}>
            <span className="brand-icon"><LeafMark /></span>
            <span className="brand-copy">
              <strong>{t("brand")}</strong>
              <small>{t("brandSubtitle")}</small>
            </span>
          </a>
          <nav className="main-nav" aria-label={t("nav.menu")}>
            <a href="#recommendation">{t("nav.recommendation")}</a>
            <a href="#profitability">{t("nav.profitability")}</a>
          </nav>
          <label className="language-picker">
            <span className="sr-only">{t("nav.language")}</span>
            <select value={language} onChange={changeLanguage} aria-label={t("nav.language")}>
              <option value="en">EN</option>
              <option value="hi">हिं</option>
            </select>
          </label>
        </div>
      </header>

      <main className="page-width app-main" id="home">
        <section className="workflow-status-card" aria-labelledby="workflow-title">
          <div className="workflow-status-heading">
            <div>
              <span className="eyebrow">{t("workflow.eyebrow")}</span>
              <h1 id="workflow-title">{t("workflow.title")}</h1>
            </div>
            <span className={`workflow-badge ${workflowStatus?.connected ? "workflow-connected" : ""}`}>
              {workflowStatusError
                ? t("workflow.statusUnknown")
                : workflowStatus
                  ? t("workflow.statusNotConnected")
                  : t("workflow.statusChecking")}
            </span>
          </div>
          <p>{workflowStatus ? t("workflow.messageNotConnected") : t("workflow.unavailable")}</p>
          <details>
            <summary>{t("workflow.setupSummary")}</summary>
            <p>{t("workflow.setupDetails")}</p>
            <a href="https://docs.cloud.google.com/gemini/enterprise/docs/invoke-agent-streamassist" target="_blank" rel="noreferrer">
              {t("workflow.docsLink")}
            </a>
          </details>
        </section>

        <section className="prediction-section" id="recommendation">
          <div className={`api-status ${apiReady ? "status-ready" : "status-offline"}`} role="status">
            <span className="status-dot" />
            <span>{t("hero.modelStatus")}: {apiReady ? t("hero.online") : t("hero.offline")}</span>
          </div>
          <div className="recommendation-layout">
            <div>
              <RecommendationForm
                values={values}
                onChange={updateField}
                onSubmit={submitPrediction}
                onLoadSample={loadSample}
                isLoading={loading}
                formRef={formRef}
                errorMessage={apiError}
              />
              <p className="sample-announcement" role="status">{sampleNotice}</p>
            </div>
            <div ref={resultRef} className="result-column" tabIndex={-1}>
              {loading ? (
                <div className="loading-card" role="status" aria-live="polite">
                  <span className="spinner spinner-large" aria-hidden="true" />
                  <h3>{t("form.loading")}</h3>
                </div>
              ) : result ? (
                <ResultsPanel result={result} onEdit={editInputs} />
              ) : (
                <div className="empty-card">
                  <h3>{t("empty.title")}</h3>
                  <p>{t("empty.body")}</p>
                </div>
              )}
            </div>
          </div>
        </section>

        <details className="optional-profitability" id="profitability">
          <summary>{t("profitability.optionalTitle")}</summary>
          <p className="optional-intro">{t("profitability.optionalIntro")}</p>
          <ProfitabilityCalculator predictedCrop={result?.predicted_crop || ""} />
        </details>

        <p className="page-disclaimer">{t("limitations.guidance")}</p>
      </main>

      <footer className="site-footer">
        <div className="page-width footer-inner">
          <div className="footer-brand"><LeafMark /><span>{t("brandSubtitle")}</span></div>
          <p>{t("footer.project")}</p>
        </div>
      </footer>
    </div>
  );
}

export default App;
