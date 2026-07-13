import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

// Regression: ISSUE-001 — live cockpit presented fallback values and inert controls as runtime state
// Found by /qa on 2026-07-10
// Report: .gstack/qa-reports/qa-report-localhost-2026-07-10.md

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const srcRoot = path.resolve(__dirname, '../..');

async function source(relativePath) {
  return fs.readFile(path.join(srcRoot, relativePath), 'utf8');
}

test('default operator route autoconnects and subscribes to real readback events', async () => {
  const hook = await source('services/useOperatorBackend.ts');
  const core = await source('services/websocketCore.mjs');

  assert.match(hook, /if \(preference === 'off'\) return false;\s+return true;/);
  assert.match(hook, /websocketService\.on\('mixer_update', handleMixerUpdate\)/);
  assert.match(hook, /websocketService\.on\('audio_capture_status', handleAudioCaptureStatus\)/);
  assert.match(hook, /websocketService\.on\('connection_status', handleConnectionStatus\)/);
  assert.match(hook, /websocketService\.getAudioCaptureStatus\(\)/);
  assert.match(hook, /isReady\?: \(\) => boolean/);
  assert.match(core, /data\.type !== 'authentication_status'/);
  assert.match(core, /this\.notifyListeners\(data\.type, data\)/);
});

test('WING connect and audio meter controls remain read-only by default', async () => {
  const hook = await source('services/useOperatorBackend.ts');
  const websocket = await source('services/websocket.js');
  const shell = await source('components/prototype/AutomixerComponents.tsx');

  assert.match(websocket, /connectWing\(ip, sendPort = 2223, receivePort = 2223, observationOnly = true\)/);
  assert.match(websocket, /observation_only: observationOnly/);
  assert.match(shell, /Start read-only audio meters/);
  assert.match(shell, /Observation only · no console writes/);
  assert.match(shell, /RuntimeControlBar/);
  assert.match(shell, /Stop meters/);
  assert.doesNotMatch(shell, />Rollback UI</);
  assert.match(hook, /_frontend_authoritative_status/);
  assert.match(hook, /mergeDashboardSnapshotIntoPayloads/);
});

test('packaged offline lock hides hardware controls as well as blocking them server-side', async () => {
  const adapter = await source('services/operatorAdapters.ts');
  const hook = await source('services/useOperatorBackend.ts');
  const main = await source('main.tsx');
  const shell = await source('components/prototype/AutomixerComponents.tsx');

  assert.match(adapter, /mixerTransportLocked/);
  assert.match(adapter, /Packaged offline lock is active/);
  assert.match(main, /automixerRuntimeSafety/);
  assert.match(main, /packagedOfflineLock: bootstrap\.packagedOfflineLock/);
  assert.match(hook, /window\.automixerRuntimeSafety\?\.packagedOfflineLock === true/);
  assert.match(hook, /packaged_offline_lock: true/);
  assert.match(hook, /mixer_transport_locked: true/);
  assert.match(shell, /!ui\.backend\.mixerTransportLocked && !ui\.backend\.mixerConnected/);
  assert.match(shell, /!ui\.backend\.mixerTransportLocked && <IconButton/);
  assert.match(shell, /Offline lock/);
});

test('live dashboard renders unavailable and no-signal states instead of mock measurements', async () => {
  const adapter = await source('services/operatorAdapters.ts');
  const dashboard = await source('components/dashboard/AutomixerDashboard.tsx');
  const telemetry = await source('components/dashboard/dashboardTelemetryModel.ts');

  assert.match(adapter, /integratedLufs: metricNum\(master\.integrated_lufs, -100\)/);
  assert.match(adapter, /lraAvailable: masterAvailable && finiteMetric\(master\.lra_lu\)/);
  assert.match(dashboard, /metricDisplay\(runtime\.integratedLufs, runtime\.integratedLufsAvailable, -90\)/);
  assert.match(dashboard, /No measurable signal/);
  assert.doesNotMatch(dashboard, /mixBalanceSegments/);
  assert.match(telemetry, /primary: 'Not measured'/);
  assert.match(telemetry, /sourceLabel: 'Unavailable'/);
});

test('system screen has no inert topology or fake network-health controls', async () => {
  const system = await source('views/SystemViews.tsx');

  assert.match(system, /onClick=\{onRefresh\}/);
  assert.doesNotMatch(system, />Auto Layout</);
  assert.doesNotMatch(system, /1Gbps Up/);
  assert.doesNotMatch(system, /System Clock.*Locked/);
  assert.doesNotMatch(system, /0\.8 ms/);
});
