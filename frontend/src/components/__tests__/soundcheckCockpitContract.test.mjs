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
  assert.match(view, /Physical M005 \/ M006/);
});

test('soundcheck transport is observation-only and uses canonical websocket messages', () => {
  const websocket = read('services/websocket.js');
  const hook = read('services/useOperatorBackend.ts');

  assert.match(websocket, /startAutoSoundcheck\([^)]*observeOnly = true\)/s);
  assert.match(websocket, /type: 'start_auto_soundcheck'/);
  assert.match(websocket, /type: 'stop_auto_soundcheck'/);
  assert.match(websocket, /type: 'get_auto_soundcheck_status'/);
  assert.match(hook, /startAutoSoundcheck\(null, 48, \{\}, \{\}, \{\}, true\)/);
  assert.match(hook, /soundcheckRuntimeFromPayload/);
});

