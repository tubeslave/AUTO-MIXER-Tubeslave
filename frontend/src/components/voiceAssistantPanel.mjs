// Ф10 (DEC-0010): чистые переходы UI-состояния голосового помощника саундчека.
// Backend (backend/voice_assistant.py, этап 0 — локальный TTS через macOS say,
// НИКАКИХ записей на пульт) уже в ветке, но WS-обвязки ещё нет: команды
// voice_set_language / voice_set_enabled и push spoken_log — за backend
// (claude/codex, класс B по DEC-0006). До контракта панель работает на
// локальном состоянии и честно маркируется «local only».

export const VOICE_ASSISTANT_SPOKEN_LOG_LIMIT = 50;
export const VOICE_ASSISTANT_LANGUAGES = ['ru', 'en'];
export const VOICE_ASSISTANT_BINDING_NOTE = 'local only — backend WS binding pending (DEC-0010, stage 0: local TTS, no console writes)';

export function initialVoiceAssistantState() {
  return { lang: 'ru', enabled: false, spokenLog: [] };
}

export function isVoiceAssistantLanguage(value) {
  return VOICE_ASSISTANT_LANGUAGES.includes(value);
}

function normalizeState(state) {
  if (!state || typeof state !== 'object') return initialVoiceAssistantState();
  const lang = isVoiceAssistantLanguage(state.lang) ? state.lang : 'ru';
  const spokenLog = Array.isArray(state.spokenLog) ? state.spokenLog : [];
  return { lang, enabled: state.enabled === true, spokenLog };
}

// Смена языка ru/en: невалидные значения и повтор текущего языка
// возвращают исходный объект (без лишних ре-рендеров).
export function setVoiceAssistantLanguage(state, lang) {
  if (!isVoiceAssistantLanguage(lang)) return state;
  const current = normalizeState(state);
  if (lang === current.lang) return state;
  return { ...current, lang };
}

// Вкл/выкл голоса. Выключено по умолчанию: пока нет backend-обвязки,
// включённое состояние — только локальное намерение оператора.
export function toggleVoiceAssistantEnabled(state) {
  const current = normalizeState(state);
  return { ...current, enabled: !current.enabled };
}

// Журнал произнесённых фраз — зеркало VoiceAssistant.spoken_log
// ({phrase, text}), новые первыми, capped. Невалидные записи возвращают
// исходный объект состояния без изменений.
export function appendVoiceAssistantSpoken(state, entry, limit = VOICE_ASSISTANT_SPOKEN_LOG_LIMIT) {
  if (!entry || typeof entry !== 'object') return state;
  const phrase = String(entry.phrase ?? '').trim();
  const text = String(entry.text ?? '').trim();
  if (!phrase || !text) return state;
  const current = normalizeState(state);
  return { ...current, spokenLog: [{ phrase, text }, ...current.spokenLog].slice(0, limit) };
}
