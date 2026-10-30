import { ChangeEvent, FormEvent, useState } from 'react'
import { enumLabel, Language, MessageKey, translate } from './i18n'

type Evaluation = {
  evaluation_status: 'complete' | 'needs_clarification'
  clarification_prompt: string
  missing_details: string
}

type Assessment = {
  field_summary: string
  information_collected_and_source: string
  preliminary_agricultural_assessment: string
  clarifying_questions: string
  missing_information: string
  checks_required_before_recommendation: string
  limitations_and_confidence: string
}

type Results = {
  session_id: string
  original_intake: Record<string, string | null>
  clarification: { clarification_response: string; additional_notes?: string | null } | null
  clarification_history: Array<Record<string, string | null>>
  verifier: {
    verified_agronomic_context: string
    crop_advisory_references: string
    unverified_parameters: string
  }
  assessment: Assessment
}

type ClarificationReply = {
  status: 'awaiting_clarification' | 'completed'
  evaluation: Evaluation
  results: Results | null
}
type FormValues = Record<string, string>

const apiRoot = '/api/workflow/v1'
const exactDisclaimer = 'No external institutional source was directly verified during this workflow run.'
const soilValues = [
  'Sticky (चिपचिपी / Clayey)',
  'Smooth (चिकनी / Silty)',
  'Gritty (खुरदरी / Loamy)',
  'Sandy (रेतीली)',
  'Loose (ढीली)',
  'Unknown (ज्ञात नहीं)',
]
const irrigationValues = [
  'Tubewell / Borewell',
  'Canal',
  'Rainfed / Pond',
  'River / Lift',
  'Other',
  'None / Unirrigated',
  'Unknown',
]
const backupValues = ['None', 'Tubewell/Borewell', 'Canal', 'Pond/Tank', 'Other', 'Unknown']
const reportLabels: Record<keyof Assessment, MessageKey> = {
  field_summary: 'fieldSummary',
  information_collected_and_source: 'collectedSource',
  preliminary_agricultural_assessment: 'preliminaryAssessment',
  clarifying_questions: 'clarifyingQuestions',
  missing_information: 'missingInformation',
  checks_required_before_recommendation: 'checksRequired',
  limitations_and_confidence: 'limitationsConfidence',
}

class ApiError extends Error {
  constructor(readonly status: number) {
    super('API request failed')
  }
}

async function api<T>(path: string, language: Language, options?: RequestInit): Promise<T> {
  let response: Response
  try {
    response = await fetch(`${apiRoot}${path}`, {
      ...options,
      headers: { 'Content-Type': 'application/json', ...options?.headers },
    })
  } catch {
    throw new Error(translate(language, 'networkError'))
  }
  if (!response.ok) throw new ApiError(response.status)
  if (response.status === 204) return undefined as T
  return response.json() as Promise<T>
}

function App() {
  const [sessionId, setSessionId] = useState<string | null>(null)
  const [step, setStep] = useState(0)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [statusMessage, setStatusMessage] = useState('')
  const [evaluation, setEvaluation] = useState<Evaluation | null>(null)
  const [results, setResults] = useState<Results | null>(null)
  const [values, setValues] = useState<FormValues>({})
  const [clarification, setClarification] = useState('')
  const [additionalNotes, setAdditionalNotes] = useState('')
  const language: Language = values.language_preference === 'Hindi' ? 'Hindi' : 'English'
  const t = (key: MessageKey) => translate(language, key)

  function update(event: ChangeEvent<HTMLInputElement | HTMLSelectElement | HTMLTextAreaElement>) {
    setValues((current) => ({ ...current, [event.target.name]: event.target.value }))
    setError('')
  }

  function errorMessage(cause: unknown): string {
    if (cause instanceof ApiError) {
      if (cause.status === 404) return t('notFoundError')
      if (cause.status === 409) return t('transitionError')
      if (cause.status === 422) return t('validationError')
      return t('genericError')
    }
    return cause instanceof Error ? cause.message : t('genericError')
  }

  async function run(action: () => Promise<void>) {
    setBusy(true)
    setError('')
    setStatusMessage('')
    try {
      await action()
    } catch (cause) {
      setError(errorMessage(cause))
    } finally {
      setBusy(false)
    }
  }

  function validateForm(event: FormEvent<HTMLFormElement>): boolean {
    const missing = Array.from(event.currentTarget.elements).find(
      (element): element is HTMLInputElement | HTMLTextAreaElement | HTMLSelectElement =>
        (element instanceof HTMLInputElement
          || element instanceof HTMLTextAreaElement
          || element instanceof HTMLSelectElement)
        && element.required
        && !element.value.trim(),
    )
    if (missing) {
      setError(t('requiredError'))
      missing.focus()
      return false
    }
    return true
  }

  function startNewSession() {
    void run(async () => {
      const created = await api<{ session_id: string }>('/sessions', language, { method: 'POST' })
      setSessionId(created.session_id)
      setStep(1)
      setValues({})
      setEvaluation(null)
      setResults(null)
      setClarification('')
      setAdditionalNotes('')
    })
  }

  async function submitLanguage(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    if (!validateForm(event) || !sessionId) return
    await run(async () => {
      await api(`/sessions/${sessionId}/language-location`, language, {
        method: 'POST',
        body: JSON.stringify({
          language_preference: values.language_preference,
          district: values.district,
          field_area_and_unit: values.field_area_and_unit,
          village_or_location: values.village_or_location,
          farmer_name: values.farmer_name || null,
        }),
      })
      setStep(2)
    })
  }

  async function submitSoil(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    if (!validateForm(event) || !sessionId) return
    await run(async () => {
      await api(`/sessions/${sessionId}/soil-water`, language, {
        method: 'POST',
        body: JSON.stringify({
          soil_feel: values.soil_feel,
          irrigation_source: values.irrigation_source,
          primary_irrigation_source: values.primary_irrigation_source || null,
          backup_irrigation_source: values.backup_irrigation_source || null,
          water_reliability_drainage: values.water_reliability_drainage,
          soil_visual_notes: values.soil_visual_notes || null,
          water_availability_reliability: values.water_availability_reliability || null,
          canal_release_schedule: values.canal_release_schedule || null,
        }),
      })
      setStep(3)
    })
  }

  async function submitHistory(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    if (!validateForm(event) || !sessionId) return
    await run(async () => {
      const result = await api<Evaluation>(`/sessions/${sessionId}/history-preferences`, language, {
        method: 'POST',
        body: JSON.stringify({
          soil_test_report_available: values.soil_test_report_available,
          previous_crops: values.previous_crops,
          intended_sowing_month: values.intended_sowing_month,
          fertilizer_manure_history: values.fertilizer_manure_history || null,
          farming_objectives_budget: values.farming_objectives_budget || null,
        }),
      })
      setEvaluation(result)
      if (result.evaluation_status === 'needs_clarification') {
        setStep(4)
      } else {
        setResults(await api<Results>(`/sessions/${sessionId}/assessment`, language))
        setStatusMessage(t('statusComplete'))
        setStep(5)
      }
    })
  }

  async function submitClarification(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    if (!sessionId) return
    if (!clarification.trim()) {
      setError(t('clarificationInvalid'))
      return
    }
    await run(async () => {
      const reply = await api<ClarificationReply>(`/sessions/${sessionId}/clarification`, language, {
        method: 'POST',
        body: JSON.stringify({
          clarification_response: clarification,
          additional_notes: additionalNotes || null,
        }),
      })
      setEvaluation(reply.evaluation)
      if (reply.status === 'awaiting_clarification') {
        setClarification('')
        setStatusMessage(t('statusAwaiting'))
      } else if (reply.results) {
        setResults(reply.results)
        setStatusMessage(t('resolvedClarification'))
        setStep(5)
      }
    })
  }

  const field = (
    name: string,
    label: MessageKey,
    required = false,
    type: 'text' | 'textarea' | 'select' = 'text',
    options?: string[],
  ) => (
    <label className="field" key={name}>
      <span>{t(label)}{required && <span className="required" aria-hidden="true"> *</span>}</span>
      {type === 'textarea' ? (
        <textarea data-testid={name} name={name} value={values[name] ?? ''} onChange={update} required={required} rows={3} />
      ) : type === 'select' ? (
        <select data-testid={name} name={name} value={values[name] ?? ''} onChange={update} required={required}>
          <option value="">{t('selectOption')}</option>
          {options?.map((option) => (
            <option key={option} value={option}>{enumLabel(language, option)}</option>
          ))}
        </select>
      ) : (
        <input data-testid={name} name={name} value={values[name] ?? ''} onChange={update} required={required} />
      )}
    </label>
  )

  const stepTitles: MessageKey[] = [
    'languageLocation',
    'soilWater',
    'historyPreferences',
    'clarification',
    'assessment',
  ]
  const enumStatus = ['Yes', 'No', 'Unknown']

  return (
    <main className="shell" lang={language === 'Hindi' ? 'hi' : 'en'}>
      <header className="hero">
        <div className="brand-mark" aria-hidden="true">🌱</div>
        <div>
          <p className="eyebrow">{t('standalone')}</p>
          <h1>{t('appTitle')}</h1>
          <p className="intro">{t('intro')}</p>
        </div>
      </header>

      {step > 0 && (
        <nav className="progress" aria-label={t('assessment')}>
          {stepTitles.map((key, index) => (
            <div className={`progress-item ${step >= index + 1 ? 'active' : ''}`} key={key}>
              <span>{index + 1}</span><small>{t(key)}</small>
            </div>
          ))}
        </nav>
      )}

      {error && <div className="alert" data-testid="error-message" role="alert">{error}</div>}
      {statusMessage && <div className="success" data-testid="status-message" role="status">{statusMessage}</div>}
      {busy && <div className="loading" data-testid="loading-state" role="status"><span className="spinner" /> {t('loading')}</div>}

      {step === 0 && (
        <section className="card welcome">
          <p className="eyebrow">{t('welcomeEyebrow')}</p>
          <h2>{t('welcomeTitle')}</h2>
          <p>{t('welcomeText')}</p>
          <button className="primary" data-testid="start-session" onClick={startNewSession} disabled={busy}>{t('startSession')}</button>
        </section>
      )}

      {step === 1 && (
        <form className="card" noValidate onSubmit={submitLanguage}>
          <p className="eyebrow">{t('stepOne')}</p><h2>{t('languageLocation')}</h2>
          <div className="form-grid">
            {field('language_preference', 'preferredLanguage', true, 'select', ['Hindi', 'English'])}
            {field('farmer_name', 'farmerName')}
            {field('district', 'district', true)}
            {field('village_or_location', 'villageLocation', true)}
            {field('field_area_and_unit', 'fieldArea', true)}
          </div>
          <button className="primary" data-testid="continue-button" disabled={busy}>{t('continue')}</button>
        </form>
      )}

      {step === 2 && (
        <form className="card" noValidate onSubmit={submitSoil}>
          <p className="eyebrow">{t('stepTwo')}</p><h2>{t('soilWater')}</h2>
          <div className="form-grid">
            {field('soil_feel', 'soilFeel', true, 'select', soilValues)}
            {field('irrigation_source', 'irrigationSource', true, 'select', irrigationValues)}
            {field('primary_irrigation_source', 'primaryIrrigationSource', false, 'select', irrigationValues)}
            {field('backup_irrigation_source', 'backupIrrigationSource', false, 'select', backupValues)}
            {field('water_reliability_drainage', 'waterReliability', true, 'textarea')}
            {field('water_availability_reliability', 'waterAvailability', false, 'textarea')}
            {field('soil_visual_notes', 'soilVisualNotes', false, 'textarea')}
            {field('canal_release_schedule', 'canalSchedule', false, 'textarea')}
          </div>
          <button className="primary" data-testid="continue-button" disabled={busy}>{t('continue')}</button>
        </form>
      )}

      {step === 3 && (
        <form className="card" noValidate onSubmit={submitHistory}>
          <p className="eyebrow">{t('stepThree')}</p><h2>{t('historyPreferences')}</h2>
          <div className="form-grid">
            {field('soil_test_report_available', 'soilTestAvailable', true, 'select', enumStatus)}
            {field('previous_crops', 'previousCrops', true, 'textarea')}
            {field('intended_sowing_month', 'sowingMonth', true)}
            {field('fertilizer_manure_history', 'fertilizerHistory', false, 'textarea')}
            {field('farming_objectives_budget', 'objectivesBudget', false, 'textarea')}
          </div>
          <button className="primary" data-testid="continue-button" disabled={busy}>{t('assessment')}</button>
        </form>
      )}

      {step === 4 && evaluation && (
        <form className="card" noValidate onSubmit={submitClarification}>
          <p className="eyebrow">{t('completenessEyebrow')}</p><h2>{t('clarificationTitle')}</h2>
          <p className="question" data-testid="clarification-question" lang={language === 'Hindi' ? 'hi' : 'en'}>{evaluation.clarification_prompt}</p>
          <p className="muted">{t('answerOnlyQuestion')}</p>
          <label className="field">
            <span>{t('yourClarification')} <span className="required" aria-hidden="true">*</span></span>
            <textarea data-testid="clarification-response" value={clarification} onChange={(event) => setClarification(event.target.value)} required rows={4} />
          </label>
          <label className="field">
            <span>{t('additionalNotes')}</span>
            <textarea data-testid="additional-notes" value={additionalNotes} onChange={(event) => setAdditionalNotes(event.target.value)} rows={3} />
          </label>
          <button className="primary" data-testid="submit-clarification" disabled={busy}>{t('submitClarification')}</button>
        </form>
      )}

      {step === 5 && results && (
        <section className="card report">
          <div className="report-heading">
            <div><p className="eyebrow">{t('reportEyebrow')}</p><h2 data-testid="report-title">{t('fieldAssessment')}</h2></div>
            <button className="secondary" data-testid="new-session" onClick={startNewSession} disabled={busy}>{t('newSession')}</button>
          </div>
          <p className="session-tag">{t('session')} {results.session_id.slice(0, 8)}</p>
          <div className="notice" data-testid="report-disclaimer"><strong>{t('preliminaryOnly')}</strong> {exactDisclaimer}</div>
          {(Object.keys(reportLabels) as (keyof Assessment)[]).map((key) => (
            <article className="report-section" key={key}>
              <h3>{t(reportLabels[key])}</h3>
              <p>{results.assessment[key]}</p>
            </article>
          ))}
          <details className="source-details">
            <summary>{t('verificationDetails')}</summary>
            <p>{results.verifier.verified_agronomic_context}</p>
            <p>{exactDisclaimer} {results.verifier.crop_advisory_references.replace(exactDisclaimer, '').trim()}</p>
            <p>{results.verifier.unverified_parameters}</p>
          </details>
        </section>
      )}

      <footer>{t('footer')}</footer>
    </main>
  )
}

export default App
