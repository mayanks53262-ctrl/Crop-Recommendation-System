import { expect, test } from '@playwright/test'

const disclaimer = 'No external institutional source was directly verified during this workflow run.'

async function startSession(page: any) {
  await page.goto('/')
  await page.getByTestId('start-session').click()
  await expect(page.getByRole('heading', { name: /language and location|भाषा और स्थान/i })).toBeVisible()
}

async function fillLanguageLocation(page: any, language: 'English' | 'Hindi', farmer = 'Farmer A') {
  await page.getByTestId('language_preference').selectOption(language)
  await page.getByTestId('district').fill('Test District')
  await page.getByTestId('village_or_location').fill('Test Village')
  await page.getByTestId('field_area_and_unit').fill('2 acres')
  await page.getByTestId('farmer_name').fill(farmer)
  await page.getByTestId('continue-button').click()
}

async function fillSoilWater(page: any, irrigationSource = 'Canal') {
  await page.getByTestId('soil_feel').selectOption('Unknown (ज्ञात नहीं)')
  await page.getByTestId('irrigation_source').selectOption(irrigationSource)
  await page.getByTestId('water_reliability_drainage').fill('Adequate drainage and reliable supply')
  await page.getByTestId('continue-button').click()
}

async function fillHistory(page: any) {
  await page.getByTestId('soil_test_report_available').selectOption('Unknown')
  await page.getByTestId('previous_crops').fill('Wheat in previous season')
  await page.getByTestId('intended_sowing_month').fill('October')
  await page.getByTestId('continue-button').click()
}

test('workflow completes in English end-to-end', async ({ page }) => {
  await startSession(page)
  await fillLanguageLocation(page, 'English', 'Farmer A')
  await fillSoilWater(page, 'Canal')
  await fillHistory(page)
  await expect(page.getByTestId('report-disclaimer')).toContainText(disclaimer)
  await expect(page.getByTestId('report-title')).toContainText(/field assessment/i)
})

test('workflow completes in Hindi end-to-end', async ({ page }) => {
  await startSession(page)
  await fillLanguageLocation(page, 'Hindi', 'किसान ए')
  await fillSoilWater(page, 'Canal')
  await fillHistory(page)
  await expect(page.getByTestId('report-disclaimer')).toContainText(disclaimer)
  await expect(page.getByTestId('report-title')).toContainText(/खेत.*मूल्यांकन|field assessment/i)
})

test('required-field validation blocks missing inputs', async ({ page }) => {
  await startSession(page)
  await page.getByTestId('language_preference').selectOption('English')
  await page.getByTestId('continue-button').click()
  await expect(page.getByTestId('error-message')).toContainText(/Complete all fields marked as required|Some required information is missing or invalid/i)
})

test('missing vs explicitly unknown values are handled', async ({ page }) => {
  await startSession(page)
  await fillLanguageLocation(page, 'English', 'Farmer A')
  await page.getByTestId('soil_feel').selectOption('Unknown (ज्ञात नहीं)')
  await page.getByTestId('irrigation_source').selectOption('Unknown')
  await page.getByTestId('water_reliability_drainage').fill('Unknown')
  await page.getByTestId('continue-button').click()
  await page.getByTestId('soil_test_report_available').selectOption('No')
  await page.getByTestId('previous_crops').fill('Unknown')
  await page.getByTestId('intended_sowing_month').fill('Not specified')
  await page.getByTestId('continue-button').click()
  await expect(page.getByTestId('report-disclaimer')).toContainText(disclaimer)
})

test('conflicting irrigation sources trigger clarification', async ({ page }) => {
  await startSession(page)
  await fillLanguageLocation(page, 'English', 'Farmer B')
  await page.getByTestId('soil_feel').selectOption('Unknown (ज्ञात नहीं)')
  await page.getByTestId('irrigation_source').selectOption('Canal')
  await page.getByTestId('primary_irrigation_source').selectOption('Tubewell / Borewell')
  await page.getByTestId('water_reliability_drainage').fill('Adequate drainage')
  await page.getByTestId('continue-button').click()
  await page.getByTestId('soil_test_report_available').selectOption('Unknown')
  await page.getByTestId('previous_crops').fill('Wheat')
  await page.getByTestId('intended_sowing_month').fill('October')
  await page.getByTestId('continue-button').click()
  await expect(page.getByTestId('clarification-question')).toContainText(/irrigation|source/i)
})

test('ambiguous clarification keeps the conflict unresolved', async ({ page }) => {
  await startSession(page)
  await fillLanguageLocation(page, 'English', 'Farmer C')
  await page.getByTestId('soil_feel').selectOption('Unknown (ज्ञात नहीं)')
  await page.getByTestId('irrigation_source').selectOption('Canal')
  await page.getByTestId('primary_irrigation_source').selectOption('Tubewell / Borewell')
  await page.getByTestId('water_reliability_drainage').fill('Adequate drainage')
  await page.getByTestId('continue-button').click()
  await page.getByTestId('soil_test_report_available').selectOption('Unknown')
  await page.getByTestId('previous_crops').fill('Wheat')
  await page.getByTestId('intended_sowing_month').fill('October')
  await page.getByTestId('continue-button').click()
  await page.getByTestId('clarification-response').fill('It may be either source.')
  await page.getByTestId('submit-clarification').click()
  await expect(page.getByTestId('clarification-question')).toContainText(/irrigation|source/i)
  await expect(page.getByTestId('status-message')).toContainText(/clarification|awaiting/i)
})

test('clear clarification resolves the intended conflict without overwriting original intake', async ({ page }) => {
  await startSession(page)
  await fillLanguageLocation(page, 'English', 'Farmer D')
  await page.getByTestId('soil_feel').selectOption('Unknown (ज्ञात नहीं)')
  await page.getByTestId('irrigation_source').selectOption('Canal')
  await page.getByTestId('primary_irrigation_source').selectOption('Tubewell / Borewell')
  await page.getByTestId('water_reliability_drainage').fill('Adequate drainage')
  await page.getByTestId('continue-button').click()
  await page.getByTestId('soil_test_report_available').selectOption('No')
  await page.getByTestId('previous_crops').fill('Wheat')
  await page.getByTestId('intended_sowing_month').fill('October')
  await page.getByTestId('continue-button').click()
  await page.getByTestId('clarification-response').fill('The correct source is Canal.')
  await page.getByTestId('submit-clarification').click()
  await expect(page.getByTestId('report-disclaimer')).toContainText(disclaimer)
  await expect(page.getByText(/Original irrigation answers differed|The correct source is Canal/i)).toBeVisible()
})

test('unsupported crop recommendations are blocked when critical information is unresolved', async ({ page }) => {
  await startSession(page)
  await fillLanguageLocation(page, 'English', 'Farmer E')
  await fillSoilWater(page, 'Canal')
  await page.getByTestId('soil_test_report_available').selectOption('Unknown')
  await page.getByTestId('previous_crops').fill('')
  await page.getByTestId('intended_sowing_month').fill('')
  await page.getByTestId('continue-button').click()
  await expect(page.getByTestId('error-message')).toContainText(/required|missing/i)
})

test('separate sessions do not share farmer data', async ({ page }) => {
  await startSession(page)
  await fillLanguageLocation(page, 'English', 'Farmer One')
  await fillSoilWater(page, 'Canal')
  await fillHistory(page)
  await page.getByTestId('new-session').click()
  await expect(page.getByTestId('language_preference')).toBeVisible()
  await fillLanguageLocation(page, 'English', 'Farmer Two')
  await fillSoilWater(page, 'River / Lift')
  await fillHistory(page)
  await expect(page.getByText(/Farmer: Farmer Two;/i).first()).toBeVisible()
  await expect(page.getByText(/Farmer: Farmer One;/i)).toHaveCount(0)
})
