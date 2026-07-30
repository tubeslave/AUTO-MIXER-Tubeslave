// Ф10 (DEC-0010): чистые переходы UI-состояния голосового помощника саундчека.
// WS-контракт v1 принят (MSG-CLD-0031). Команды: set_voice_language,
// set_voice_enabled, get_voice_assistant_status. Статус: voice_assistant_status.

export const VOICE_ASSISTANT_SPOKEN_LOG_LIMIT = 50;
export const VOICE_ASSISTANT_LANGUAGES = ['ru', 'en'];
export const VOICE_ASSISTANT_SCHEMA = 'automixer.voice_assistant.v1';

export function initialVoiceAssistantState() {
  return { lang: 'ru', enabled: false, available: false, backend: 'none', routing: 'local_only', spokenLog: [] };
}

export function isVoiceAssistantLanguage(value) {
  return VOICE_ASSISTANT_LANGUAGES.includes(value);
}

function normalizeState(state) {
  if (!state || typeof state !== 'object') return initialVoiceAssistantState();
  const lang = isVoiceAssistantLanguage(state.lang) ? state.lang : 'ru';
  const spokenLog = Array.isArray(state.spokenLog) ? state.spokenLog : [];
  return {
    lang,
    enabled: state.enabled === true,
    available: state.available === true,
    backend: ['say', 'piper'].includes(state.backend) ? state.backend : 'none',
    routing: state.routing === 'monitor_buses' ? 'monitor_buses' : 'local_only',
    spokenLog
  };
}

// Смена языка ru/en: невалидные значения и повтор текущего языка
// возвращают исходный объект (без лишних ре-рендеров).
export function setVoiceAssistantLanguage(state, lang) {
  if (!isVoiceAssistantLanguage(lang)) return state;
  const current = normalizeState(state);
  if (lang === current.lang) return state;
  return { ...current, lang };
}

// Вкл/выкл голоса. Выключено по умолчанию.
export function toggleVoiceAssistantEnabled(state) {
  const current = normalizeState(state);
  return { ...current, enabled: !current.enabled };
}

// Журнал произнесённых фраз — зеркало voice_assistant_status.spoken_log
// ({phrase, text}), новые первыми, capped.
export function appendVoiceAssistantSpoken(state, entry, limit = VOICE_ASSISTANT_SPOKEN_LOG_LIMIT) {
  if (!entry || typeof entry !== 'object') return state;
  const phrase = String(entry.phrase ?? '').trim();
  const text = String(entry.text ?? '').trim();
  if (!phrase || !text) return state;
  const current = normalizeState(state);
  return { ...current, spokenLog: [{ phrase, text }, ...current.spokenLog].slice(0, limit) };
}

// Полное обновление из voice_assistant_status (WS broadcast, schema
// automixer.voice_assistant.v1). Чужая схема — статус игнорируется целиком.
// Backend отдаёт spoken_log хвостом в хронологии (старые первыми) — в UI
// разворачиваем: новые первыми, как в appendVoiceAssistantSpoken.
export function applyVoiceAssistantStatus(state, payload) {
  if (!payload || typeof payload !== 'object') return state;
  if (payload.schema !== undefined && payload.schema !== VOICE_ASSISTANT_SCHEMA) return state;
  const current = normalizeState(state);
  const lang = isVoiceAssistantLanguage(payload.lang) ? payload.lang : current.lang;
  const spokenLog = Array.isArray(payload.spoken_log)
    ? payload.spoken_log.map(e => ({
        phrase: String(e.phrase ?? '').trim(),
        text: String(e.text ?? '').trim()
      })).filter(e => e.phrase && e.text).reverse().slice(0, VOICE_ASSISTANT_SPOKEN_LOG_LIMIT)
    : current.spokenLog;
  return {
    lang,
    enabled: payload.enabled === true,
    available: payload.available === true,
    backend: ['say', 'piper'].includes(payload.backend) ? payload.backend : current.backend,
    routing: payload.routing === 'monitor_buses' ? 'monitor_buses' : 'local_only',
    spokenLog
  };
}
