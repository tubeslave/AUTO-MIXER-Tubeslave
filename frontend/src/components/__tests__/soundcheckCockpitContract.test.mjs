import assert from 'node:assert/strict';
import fs from 'node:fs';
import test from 'node:test';

const root = new URL('../../', import.meta.url);
const read = path => fs.readFileSync(new URL(path, root), 'utf8');

test('active frontend routes to the real proposal-only soundcheck cockpit', () => {
  const entry = read('main.tsx');
  const nav = read('mock/automixerData.ts');
  const router = read('views/AutomixerViews.tsx');
  const view = read('views/SoundcheckView.tsx');

  assert.match(entry, /AutomixerPrototype/);
  assert.match(nav, /id: 'soundcheck'/);
  assert.match(router, /case 'soundcheck': return SoundcheckView/);
  assert.match(view, /This screen never writes directly to WING/);
  assert.match(view, /Mock channels are intentionally hidden from soundcheck/);
  assert.match(view, /M005 acceptance/);
  assert.match(view, /External review required/);
  assert.match(view, /Standing family authority/);
  assert.match(view, /Console module authority/);
  assert.match(view, /runtime\.transactionOperationCount/);
  assert.match(view, /Mix readiness/);
  assert.match(view, /readiness\.unresolvedRequiredIds\.length/);
  assert.match(view, /readiness\.unresolvedReasons/);
  assert.match(view, /Accept safe next revision/);
  assert.match(view, /Reconcile targets \+ Main \(read only\)/);
  assert.match(view, /safeAlternativeOperationCount/);
  assert.doesNotMatch(view, /state === 'running'\) return 'green'/);
});

test('soundcheck transport is observation-only and uses canonical websocket messages', () => {
  const websocket = read('services/websocket.js');
  const hook = read('services/useOperatorBackend.ts');

  assert.match(websocket, /startAutoSoundcheck\([^)]*observeOnly = true[,)]/s);
  assert.match(websocket, /type: 'start_auto_soundcheck'/);
  assert.match(websocket, /type: 'stop_auto_soundcheck'/);
  assert.match(websocket, /type: 'get_auto_soundcheck_status'/);
  assert.match(websocket, /type: 'accept_soundcheck_replan'/);
  assert.match(websocket, /type: 'finalize_soundcheck_readiness'/);
  assert.match(hook, /activeSoundcheckChannels\(payloads\.channels\)/);
  assert.match(hook, /startAutoSoundcheck\(null, selectedChannels, \{\}, \{\}, \{\}, true\)/);
  assert.doesNotMatch(hook, /startAutoSoundcheck\(null, 48,/);
  assert.match(hook, /soundcheckRuntimeFromPayload/);
  assert.match(hook, /missing-readiness-contract/);
  assert.match(hook, /mixReadiness: 'soundcheck_incomplete'/);
  assert.match(hook, /readiness_contract_missing_from_lifecycle_event/);
  assert.doesNotMatch(hook, /derivationComplete: true/);
});

test('compact soundcheck separates derivation from mix readiness', () => {
  const compact = read('components/prototype/AutomixerComponents.tsx');

  assert.match(compact, /Derived only/);
  assert.match(compact, /ui\.soundcheck\.readiness\.mixReadiness/);
  assert.doesNotMatch(compact, /derivationComplete \? 'green'/);
});
