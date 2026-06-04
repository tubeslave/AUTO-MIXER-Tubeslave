import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

import ts from 'typescript';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const srcRoot = path.resolve(__dirname, '../..');

async function importTypeScript(relativePath) {
  const sourcePath = path.join(srcRoot, relativePath);
  const source = await fs.readFile(sourcePath, 'utf8');
  const { outputText } = ts.transpileModule(source, {
    compilerOptions: {
      jsx: ts.JsxEmit.ReactJSX,
      module: ts.ModuleKind.ES2022,
      target: ts.ScriptTarget.ES2022,
    },
    fileName: sourcePath,
  });
  const encoded = Buffer.from(
    `${outputText}\n//# sourceURL=${pathToFileURL(sourcePath).href}`,
    'utf8',
  ).toString('base64');
  return import(`data:text/javascript;base64,${encoded}`);
}

test('dashboard strip uses read-only channel inventory before mock fallback', async () => {
  const { adaptChannels } = await importTypeScript('services/operatorAdapters.ts');
  const { dashboardTelemetryForChannel } = await importTypeScript('components/dashboard/dashboardTelemetryModel.ts');

  const fallbackChannels = [{
    id: 2,
    name: 'Mock Vocal',
    type: 'Mock',
    source: 'In 02',
    level: -50,
    lufs: -55,
    dynamic: 0,
    status: 'INACTIVE',
    recognition: 'Mock',
    group: 'CHANNELS',
    fader: -40,
    pan: 33,
    gateLabel: 'G mock',
    compLabel: 'C mock',
    confidence: 10,
  }];

  const fallbackTelemetry = {
    waveform: [10, 20, 30, 20],
    meterL: -44,
    meterR: -45,
    gateLabel: 'G fallback',
    compLabel: 'C fallback',
    safeState: 'LOW',
    pan: 33,
    muted: false,
    solo: true,
    accent: 'purple',
  };

  const [channel] = adaptChannels(
    {
      type: 'channel_inventory',
      channels: [{
        id: 2,
        name: 'Lead Vocal',
        role: 'Vocal',
        group: 'Vocals',
        status: 'active',
        level_db: -16.2,
        control_state: {
          fader_db: -8,
          mute: 0,
          pan: 0.25,
          gate_label: 'G -47.5',
          comp_label: 'C 2.4 GR',
        },
        ai_recognition: { label: 'Lead Vocal', confidence: 98 },
      }],
    },
    fallbackChannels,
  );

  const telemetry = dashboardTelemetryForChannel(channel, fallbackTelemetry, { suppressMockSolo: true });

  assert.equal(telemetry.meterL, -16.2);
  assert.equal(telemetry.meterR, -16.2);
  assert.equal(telemetry.pan, -25);
  assert.equal(telemetry.gateLabel, 'G -47.5');
  assert.equal(telemetry.compLabel, 'C 2.4 GR');
  assert.equal(telemetry.safeState, 'OK');
  assert.equal(telemetry.muted, false);
  assert.equal(telemetry.solo, false);
});

test('dashboard spectrum uses read-only analyzer bins before visual fallback', async () => {
  const { adaptDashboardRuntime } = await importTypeScript('services/operatorAdapters.ts');
  const { dashboardSpectrumTrace } = await importTypeScript('components/dashboard/dashboardTelemetryModel.ts');
  const { dashboardMockSafetyContract } = await importTypeScript('mock/dashboardMock.ts');

  assert.equal(dashboardMockSafetyContract.readOnly, true);
  assert.equal(dashboardMockSafetyContract.liveWrite, false);
  assert.equal(dashboardMockSafetyContract.canApplyToConsole, false);
  assert.equal(dashboardMockSafetyContract.runtimeAuthority, false);

  const runtime = adaptDashboardRuntime(
    {
      type: 'dashboard_snapshot',
      master_bus: {
        integrated_lufs: -15.4,
        true_peak_db: -2.2,
      },
      channel_summary: {
        active_channels: 18,
        total_channels: 48,
      },
      analyzer: {
        spectrum_bins: [5, 24, 50, 91, 'bad', 72],
      },
    },
    {
      summary: {
        can_live_execute: false,
      },
      agents: [
        { confidence: 84 },
        { confidence: 92 },
      ],
    },
    {
      summary: {
        pending_count: 4,
        applied_count: 1,
        blocked_count: 2,
      },
    },
  );

  assert.deepEqual(runtime.spectrumBins, [5, 24, 50, 91, 72]);

  const backendTrace = dashboardSpectrumTrace(runtime, [28, 42, 36, 58]);

  assert.deepEqual(backendTrace.values, [8, 24, 50, 88, 72]);
  assert.equal(backendTrace.sourceLabel, 'Backend DTO');
  assert.equal(backendTrace.sourceTone, 'green');
  assert.equal(backendTrace.caption, 'Read-only analyzer trace from dashboard DTO');

  const fallbackTrace = dashboardSpectrumTrace(
    { sourceLabel: 'Mock', spectrumBins: [] },
    [28, 42, 36, 58],
  );

  assert.deepEqual(fallbackTrace.values, [28, 46, 44, 58]);
  assert.equal(fallbackTrace.sourceLabel, dashboardMockSafetyContract.sourceLabel);
  assert.equal(fallbackTrace.sourceTone, dashboardMockSafetyContract.sourceTone);
  assert.equal(fallbackTrace.caption, dashboardMockSafetyContract.caption);
});
