import locales from './locales.json'

export type Language = keyof typeof locales
export type MessageKey = keyof typeof locales.English

export function translate(language: Language, key: MessageKey): string {
  return locales[language][key]
}

export function enumLabel(language: Language, value: string): string {
  const labels: Record<string, MessageKey> = {
    'Sticky (चिपचिपी / Clayey)': 'soilSticky',
    'Smooth (चिकनी / Silty)': 'soilSmooth',
    'Gritty (खुरदरी / Loamy)': 'soilGritty',
    'Sandy (रेतीली)': 'soilSandy',
    'Loose (ढीली)': 'soilLoose',
    'Unknown (ज्ञात नहीं)': 'unknownSoil',
    'Tubewell / Borewell': 'sourceTubewell',
    Canal: 'sourceCanal',
    'Rainfed / Pond': 'sourceRainfed',
    'River / Lift': 'sourceRiver',
    Other: 'sourceOther',
    'None / Unirrigated': 'sourceNone',
    None: 'no',
    'Tubewell/Borewell': 'backupTubewell',
    'Pond/Tank': 'backupPond',
    Unknown: 'unknownSource',
    Yes: 'yes',
    No: 'no',
  }
  const key = labels[value]
  return key ? translate(language, key) : value
}
