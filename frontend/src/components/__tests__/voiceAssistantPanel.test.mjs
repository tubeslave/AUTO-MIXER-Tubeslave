import test from 'node:test';
import assert from 'node:assert/strict';

import {
  VOICE_ASSISTANT_LANGUAGES,
  VOICE_ASSISTANT_SCHEMA,
  VOICE_ASSISTANT_SPOKEN_LOG_LIMIT,
  appendVoiceAssistantSpoken,
  applyVoiceAssistantStatus,
  initialVoiceAssistantState,
  isVoiceAssistantLanguage,
  setVoiceAssistantLanguage,
  toggleVoiceAssistantEnabled
} from '../voiceAssistantPanel.mjs';

test('initial voice assistant state mirrors safe stage-0 defaults', () => {
  const state = initialVoiceAssistantState();
  assert.equal(state.lang, 'ru');
  assert.equal(state.enabled, false);
  assert.deepEqual(state.spokenLog, []);
});

test('isVoiceAssistantLanguage accepts only ru/en', () => {
  assert.equal(isVoiceAssistantLanguage('ru'), true);
  assert.equal(isVoiceAssistantLanguage('en'), true);
  ['RU', 'de', '', null, undefined, 7].forEach(value => {
    assert.equal(isVoiceAssistantLanguage(value), false, String(value));
  });
  assert.deepEqual(VOICE_ASSISTANT_LANGUAGES, ['ru', 'en']);
});

test('setVoiceAssistantLanguage switches language and rejects invalid values', () => {
  const initial = initialVoiceAssistantState();
  const english = setVoiceAssistantLanguage(initial, 'en');
  assert.equal(english.lang, 'en');
  assert.equal(initial.lang, 'ru', 'исходное состояние не мутируется');
  assert.equal(setVoiceAssistantLanguage(english, 'en'), english, 'повтор того же языка — тот же объект');
  assert.equal(setVoiceAssistantLanguage(english, 'de'), english, 'невалидный язык отброшен');
  assert.equal(setVoiceAssistantLanguage(english, 'ru').lang, 'ru');
});

test('toggleVoiceAssistantEnabled flips the flag', () => {
  const initial = initialVoiceAssistantState();
  const on = toggleVoiceAssistantEnabled(initial);
  assert.equal(on.enabled, true);
  assert.equal(toggleVoiceAssistantEnabled(on).enabled, false);
  assert.equal(initial.enabled, false, 'исходное состояние не мутируется');
});

test('appendVoiceAssistantSpoken prepends valid entries and caps the log', () => {
  let state = initialVoiceAssistantState();
  state = appendVoiceAssistantSpoken(state, { phrase: 'waiting', text: 'Система ждёт: бас-бочка. Играй.' });
  state = appendVoiceAssistantSpoken(state, { phrase: 'channel_done', text: 'бас-бочка готово. Спасибо.' });
  assert.equal(state.spokenLog.length, 2);
  assert.equal(state.spokenLog[0].phrase, 'channel_done', 'новые фразы первыми');

  [
    null,
    undefined,
    'phrase',
    {},
    { phrase: 'waiting' },
    { text: 'hello' },
    { phrase: '  ', text: 'hello' },
    { phrase: 'waiting', text: '  ' }
  ].forEach(entry => {
    assert.equal(appendVoiceAssistantSpoken(state, entry), state, `невалидная запись: ${JSON.stringify(entry)}`);
  });

  let capped = initialVoiceAssistantState();
  for (let index = 0; index < VOICE_ASSISTANT_SPOKEN_LOG_LIMIT + 5; index += 1) {
    capped = appendVoiceAssistantSpoken(capped, { phrase: 'repeat', text: `фраза ${index}` });
  }
  assert.equal(capped.spokenLog.length, VOICE_ASSISTANT_SPOKEN_LOG_LIMIT);
  assert.equal(capped.spokenLog[0].text, `фраза ${VOICE_ASSISTANT_SPOKEN_LOG_LIMIT + 4}`);
});

test('helpers normalize malformed external state', () => {
  assert.deepEqual(toggleVoiceAssistantEnabled(null).enabled, true);
  assert.equal(setVoiceAssistantLanguage({ lang: 'de', enabled: true, spokenLog: null }, 'en').lang, 'en');
  const restored = appendVoiceAssistantSpoken({ lang: 'en', enabled: true, spokenLog: 'broken' }, { phrase: 'p', text: 't' });
  assert.equal(restored.spokenLog.length, 1);
});

test('applyVoiceAssistantStatus mirrors voice_assistant_status v1 broadcast', () => {
  const payload = {
    type: 'voice_assistant_status',
    schema: VOICE_ASSISTANT_SCHEMA,
    lang: 'en',
    enabled: true,
    available: true,
    backend: 'say',
    routing: 'local_only',
    // backend шлёт хронологический хвост (старые первыми)
    spoken_log: [
      { phrase: 'waiting', text: 'Система ждёт: бас-бочка. Играй.' },
      { phrase: 'channel_done', text: 'бас-бочка готово. Спасибо.' }
    ]
  };
  const next = applyVoiceAssistantStatus(initialVoiceAssistantState(), payload);
  assert.equal(next.lang, 'en');
  assert.equal(next.enabled, true);
  assert.equal(next.available, true);
  assert.equal(next.backend, 'say');
  assert.equal(next.routing, 'local_only');
  assert.equal(next.spokenLog.length, 2);
  assert.equal(next.spokenLog[0].phrase, 'channel_done', 'новые фразы первыми');
});

test('applyVoiceAssistantStatus rejects foreign schema and malformed payloads', () => {
  const current = applyVoiceAssistantStatus(initialVoiceAssistantState(), {
    schema: VOICE_ASSISTANT_SCHEMA,
    lang: 'en',
    enabled: true,
    available: true,
    backend: 'say',
    routing: 'local_only',
    spoken_log: []
  });
  assert.equal(applyVoiceAssistantStatus(current, { ...current, schema: 'automixer.other.v9' }), current);
  assert.equal(applyVoiceAssistantStatus(current, null), current);
  assert.equal(applyVoiceAssistantStatus(current, 'voice_assistant_status'), current);
});

test('applyVoiceAssistantStatus keeps current values for missing/invalid fields', () => {
  const current = applyVoiceAssistantStatus(initialVoiceAssistantState(), {
    schema: VOICE_ASSISTANT_SCHEMA,
    lang: 'en',
    enabled: true,
    available: true,
    backend: 'say',
    routing: 'local_only',
    spoken_log: [{ phrase: 'p', text: 't' }]
  });
  const partial = applyVoiceAssistantStatus(current, {
    schema: VOICE_ASSISTANT_SCHEMA,
    lang: 'de',
    enabled: false,
    available: false,
    backend: 'espeak',
    routing: 'monitor_buses'
  });
  assert.equal(partial.lang, 'en', 'невалидный lang сохраняет текущий');
  assert.equal(partial.enabled, false);
  assert.equal(partial.backend, 'say', 'невалидный backend сохраняет текущий');
  assert.equal(partial.routing, 'monitor_buses');
  assert.equal(partial.spokenLog.length, 1, 'без spoken_log журнал сохраняется');
});
