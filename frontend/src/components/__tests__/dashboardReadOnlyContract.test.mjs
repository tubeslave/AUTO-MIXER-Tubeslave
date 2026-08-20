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
  assert.equal(backendTrace.sourceLabel, 'Real audio');
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

test('instrument space preserves unknown axes and rejects write-capable contracts', async () => {
  const { adaptInstrumentSpace } = await importTypeScript('services/operatorAdapters.ts');
  const valid = adaptInstrumentSpace({
    type: 'channel_inventory',
    instrument_space: {
      type: 'instrument_space',
      schema_version: 'automixer.instrument_space.v1',
      generated_at: '2026-08-04T12:00:00Z',
      read_only: true,
      live_write: false,
      queue_mutation: false,
      can_import_to_queue: false,
      queue_import_allowed: false,
      can_apply_to_console: false,
      live_apply_allowed: false,
      live_authority_granted: false,
      readiness_closure_allowed: false,
      mutates_mixer: false,
      physical_localization: false,
      summary: {
        object_count: 1,
        complete_object_count: 0,
        missing_frequency_count: 1,
      },
      decision_support: {
        schema_version: 'automixer.instrument_space_decision_support.v1',
        source_generated_at: '2026-08-04T12:00:00Z',
        status: 'review_required',
        read_only: true,
        live_write: false,
        queue_mutation: false,
        can_import_to_queue: false,
        queue_import_allowed: false,
        can_apply_to_console: false,
        live_apply_allowed: false,
        live_authority_granted: false,
        readiness_closure_allowed: false,
        mutates_mixer: false,
        centroid_is_triage_not_eq_proof: true,
        summary: { review_required_count: 1, watch_count: 0 },
        drift: {
          status: 'review_required',
          baseline_generated_at: '2026-08-04T11:59:58Z',
          current_generated_at: '2026-08-04T12:00:00Z',
          changes: [{
            id: 'instrument-space:drift:1',
            channel_id: 1,
            label: 'KICK',
            changed_axes: ['z_level'],
            deltas: { x_pan: 0, y_frequency: null, z_level: 0.2 },
          }],
        },
        review_candidates: [{
          id: 'instrument-space:pair:1:2',
          status: 'review_required',
          channel_ids: [1, 2],
          labels: ['KICK', 'BASS'],
          score: 0.81,
          measurements: { pan_distance: 0.02, centroid_distance_octaves: 0.1, level_delta_db: 1.5 },
          blockers: [],
          evidence_provenance: {
            channels: [{
              pan_source: 'mixer_state_cache',
              frequency_source: 'audio_capture_spectral_cache',
              level_source: 'audio_capture_meter',
              recognition_source: 'cached_mixer_name',
            }],
          },
          source_generated_at: '2026-08-04T12:00:00Z',
          recommended_next_analyzers: ['frequency_separation_proposal'],
          explanation: 'Run the existing analyzer.',
        }],
      },
      history: {
        schema_version: 'automixer.instrument_space_history.v1',
        status: 'tracking',
        reason: null,
        snapshot_count: 2,
        max_snapshots: 12,
        current_snapshot_id: 'current',
        baseline_snapshot_id: 'baseline',
        source_changed: false,
        read_only: true,
        live_write: false,
        queue_mutation: false,
        can_import_to_queue: false,
        queue_import_allowed: false,
        can_apply_to_console: false,
        live_apply_allowed: false,
        live_authority_granted: false,
        readiness_closure_allowed: false,
        mutates_mixer: false,
      },
      objects: [{
        id: 'channel:1',
        channel_id: 1,
        label: 'KICK',
        role: 'Kick',
        group: 'DRUMS',
        status: 'active',
        coordinates: { x_pan: 0, y_frequency: null, z_level: 0.7 },
        evidence: {
          pan: { available: true, value: 0, unit: 'percent', source: 'mixer_state_cache' },
          frequency: { available: false, value: null, unit: 'Hz', source: 'unavailable', age_sec: null, freshness_status: 'unknown' },
          level: { available: true, value: -18, unit: 'dBFS', source: 'audio_capture_meter' },
          recognition: { confidence_percent: 99 },
          read_coverage_status: 'partial',
        },
      }],
    },
  });

  assert.equal(valid.available, true);
  assert.equal(valid.readOnly, true);
  assert.equal(valid.liveWrite, false);
  assert.equal(valid.canApplyToConsole, false);
  assert.equal(valid.objects[0].yFrequency, null);
  assert.equal(valid.objects[0].frequency.available, false);
  assert.equal(valid.objects[0].frequency.ageSec, null);
  assert.equal(valid.objects[0].frequency.freshnessStatus, 'unknown');
  assert.equal(valid.decisionSupport.available, true);
  assert.equal(valid.decisionSupport.reviewRequiredCount, 1);
  assert.equal(valid.decisionSupport.relations[0].score, 0.81);
  assert.deepEqual(valid.decisionSupport.relations[0].recommendedNextAnalyzers, ['frequency_separation_proposal']);
  assert.equal(valid.decisionSupport.relations[0].measurements.levelDeltaDb, 1.5);
  assert.deepEqual(valid.decisionSupport.relations[0].evidenceSources, [
    'mixer_state_cache',
    'audio_capture_spectral_cache',
    'audio_capture_meter',
    'cached_mixer_name',
  ]);
  assert.equal(valid.decisionSupport.drift.changes[0].channelId, 1);
  assert.equal(valid.history.snapshotCount, 2);

  const rejected = adaptInstrumentSpace({
    instrument_space: {
      read_only: false,
      live_write: true,
      can_apply_to_console: true,
      physical_localization: false,
      objects: [{ id: 'unsafe' }],
    },
  });
  assert.equal(rejected.available, false);
  assert.equal(rejected.contractValid, false);
  assert.deepEqual(rejected.objects, []);

  const unsafeSupport = adaptInstrumentSpace({
    instrument_space: {
      read_only: true,
      live_write: false,
      queue_mutation: false,
      can_import_to_queue: false,
      queue_import_allowed: false,
      can_apply_to_console: false,
      live_apply_allowed: false,
      live_authority_granted: false,
      readiness_closure_allowed: false,
      mutates_mixer: false,
      physical_localization: false,
      objects: [],
      decision_support: {
        read_only: false,
        live_write: true,
        queue_import_allowed: true,
        can_apply_to_console: true,
        live_authority_granted: true,
        review_candidates: [{ id: 'unsafe' }],
      },
    },
  });
  assert.equal(unsafeSupport.available, true);
  assert.equal(unsafeSupport.decisionSupport.available, false);
  assert.equal(unsafeSupport.decisionSupport.contractValid, false);
  assert.deepEqual(unsafeSupport.decisionSupport.relations, []);
});

test('instrument-space surface has no drag or console-apply gesture', async () => {
  const source = await fs.readFile(path.join(srcRoot, 'components/dashboard/AutomixerDashboard.tsx'), 'utf8');
  assert.equal(source.includes('onDrag'), false);
  assert.equal(source.includes('applyInstrument'), false);
  assert.equal(source.includes('Centroid proximity never selects an EQ, pan or fader value.'), true);
  assert.match(source, /mix map, not physical 3D localization/i);
  assert.match(source, /No demo coordinates are substituted/i);
});
