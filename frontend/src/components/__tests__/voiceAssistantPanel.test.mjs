import test from 'node:test';
import assert from 'node:assert/strict';

import {
  VOICE_ASSISTANT_LANGUAGES,
  VOICE_ASSISTANT_SPOKEN_LOG_LIMIT,
  appendVoiceAssistantSpoken,
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
