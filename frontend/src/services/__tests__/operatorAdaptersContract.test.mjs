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

test('adaptTasks preserves SafeGain ensemble evidence for read-only cockpit display', async () => {
  const { adaptTasks } = await importTypeScript('services/operatorAdapters.ts');

  const [task] = adaptTasks(
    {
      pending_actions: [
        {
          id: 'proposal-safe-gain-1',
          title: 'SafeGain trim review',
          target: 'Lead Vox',
          severity: 'high',
          status: 'pending',
          source: 'SafeGain',
          reason:
            'SafeGain measured input headroom. Ensemble context inactive (ensemble_time_context_stale); time stale; sources 1/2; lag 10/2 samples; operator review only.',
          requested_change: {
            value_type: 'gain',
            channel: 12,
            value: 3,
            safety_evidence: [
              {
                label: 'Ensemble context',
                value: 'ensemble_time_context_stale',
                reason: 'ensemble_time_context_stale',
              },
              {
                label: 'Ensemble time coherence',
                value: 'stale',
                reason: 'ensemble_time_context_stale',
              },
              {
                label: 'Ensemble time sources',
                value: 1,
                target: 2,
              },
              {
                label: 'Ensemble lag samples',
                value: 10,
                target: 2,
              },
              {
                label: 'Ensemble apply policy',
                value: 'operator_review_only',
              },
              {
                label: 'Distortion risk',
                value: 'low',
              },
            ],
          },
          can_apply: true,
        },
      ],
    },
    [],
  );

  assert.equal(task.reason.includes('Ensemble context inactive'), true);
  assert.deepEqual(
    task.safetyEvidence.map(row => row.label),
    [
      'Ensemble context',
      'Ensemble time coherence',
      'Ensemble time sources',
      'Ensemble lag samples',
      'Ensemble apply policy',
      'Distortion risk',
    ],
  );
  assert.equal(task.safetyEvidence[0].value, 'ensemble time context stale');
  assert.equal(task.safetyEvidence[0].tone, 'yellow');
  assert.equal(task.safetyEvidence[2].value, '1/2');
  assert.equal(task.safetyEvidence[3].value, '10/2');
  assert.equal(task.safetyEvidence[4].tone, 'yellow');
  assert.equal(task.safetyEvidence[5].tone, 'green');
});

test('safetyEvidenceForTask keeps ensemble rows visible before generic evidence', async () => {
  const { ensembleEvidenceForTask, safetyEvidenceForTask } = await importTypeScript('mock/automixerData.ts');

  const task = {
    id: 'proposal-safe-gain-1',
    priority: 'High',
    title: 'SafeGain trim review',
    agent: 'SafeGain',
    status: 'Pending',
    time: '10:31',
    safetyEvidence: [
      { label: 'Distortion risk', value: 'low', tone: 'green' },
      { label: 'Ensemble context', value: 'ensemble time context stale', tone: 'yellow' },
      { label: 'Preamp zone', value: 'clean', tone: 'green' },
      { label: 'Ensemble time coherence', value: 'stale', tone: 'yellow' },
      { label: 'Ensemble time sources', value: '1/2', tone: 'yellow' },
      { label: 'Readback', value: 'required', tone: 'yellow' },
      { label: 'Ensemble lag samples', value: '10/2', tone: 'yellow' },
      { label: 'Rollback', value: 'required', tone: 'yellow' },
      { label: 'Ensemble apply policy', value: 'operator review only', tone: 'yellow' },
    ],
  };

  const evidence = safetyEvidenceForTask(task, 'Supervised');

  assert.deepEqual(
    evidence.map(row => row.label),
    [
      'Ensemble context',
      'Ensemble time coherence',
      'Ensemble time sources',
      'Ensemble lag samples',
      'Ensemble apply policy',
      'Distortion risk',
    ],
  );

  assert.deepEqual(
    ensembleEvidenceForTask(task).map(row => row.label),
    [
      'Ensemble context',
      'Ensemble time coherence',
      'Ensemble time sources',
      'Ensemble lag samples',
      'Ensemble apply policy',
    ],
  );
});

test('adaptChannels treats control_state mute as read-only channel status', async () => {
  const { adaptChannels } = await importTypeScript('services/operatorAdapters.ts');

  const fallback = [{
    id: 12,
    name: 'Lead Vocal',
    type: 'Vocal Lead',
    source: 'In 12',
    level: -18,
    lufs: -22,
    dynamic: 6,
    status: 'ACTIVE',
    recognition: 'Lead Vocal',
    group: 'VOCALS',
    fader: -10,
    confidence: 96,
  }];

  const [muted] = adaptChannels(
    {
      channels: [{
        id: 12,
        name: 'Lead Vocal',
        role: 'Vocal Lead',
        group: 'Vocals',
        status: 'active',
        level_db: -18,
        lufs: -22,
        dynamic_range_db: 6,
        control_state: {
          fader_db: -10,
          mute: 1,
          pan: 0.3,
          gate_label: 'G -48',
          compressor_gain_reduction_db: 2.6,
        },
        ai_recognition: { label: 'Lead Vocal', confidence: 96 },
      }],
    },
    fallback,
  );

  assert.equal(muted.status, 'MUTED');
  assert.equal(muted.fader, -10);
  assert.equal(muted.pan, -20);
  assert.equal(muted.gateLabel, 'G -48');
  assert.equal(muted.compLabel, 'C 2.6');

  const [unmuted] = adaptChannels(
    {
      channels: [{
        id: 12,
        name: 'Lead Vocal',
        role: 'Vocal Lead',
        group: 'Vocals',
        status: 'active',
        level_db: -18,
        lufs: -22,
        dynamic_range_db: 6,
        control_state: { fader_db: -10, mute: '0' },
        ai_recognition: { label: 'Lead Vocal', confidence: 96 },
      }],
    },
    fallback,
  );

  assert.equal(unmuted.status, 'ACTIVE');
});

test('adaptChannels maps cached pan gate and compressor details from read-only inventory', async () => {
  const { adaptChannels } = await importTypeScript('services/operatorAdapters.ts');

  const fallback = [
    {
      id: 2,
      name: 'Fallback Vocal',
      type: 'Fallback',
      source: 'In 02',
      level: -40,
      lufs: -50,
      dynamic: 1,
      status: 'INACTIVE',
      recognition: 'Fallback',
      group: 'CHANNELS',
      fader: -40,
      pan: 0,
      gateLabel: 'G mock',
      compLabel: 'C mock',
      confidence: 10,
    },
    {
      id: 3,
      name: 'Fallback Snare',
      type: 'Fallback',
      source: 'In 03',
      level: -40,
      lufs: -50,
      dynamic: 1,
      status: 'INACTIVE',
      recognition: 'Fallback',
      group: 'CHANNELS',
      fader: -40,
      pan: 0,
      gateLabel: 'G mock',
      compLabel: 'C mock',
      confidence: 10,
    },
  ];

  const [vocal, snare] = adaptChannels(
    {
      type: 'channel_inventory',
      channels: [
        {
          id: 2,
          name: 'Lead Vocal',
          role: 'Vocal',
          group: 'Vocals',
          status: 'active',
          level_db: -16.2,
          lufs: -22.5,
          dynamic_range_db: 5.5,
          control_state: {
            fader_db: -8.0,
            mute: 0,
            pan: 0.25,
            gate_enabled: 1,
            gate_threshold_db: -47.5,
            gate_label: 'G -47.5',
            compressor_enabled: 1,
            compressor_threshold_db: -18.0,
            compressor_gain_reduction_db: 2.4,
            comp_label: 'C 2.4 GR',
          },
          ai_recognition: { label: 'Vocal', confidence: 99 },
        },
        {
          id: 3,
          name: 'Snare',
          role: 'Snare',
          group: 'Drums',
          status: 'active',
          level_db: -14.4,
          control_state: {
            fader_db: -7.0,
            mute: 0,
            pan: -12,
            gate_threshold_db: -42.0,
            compressor_threshold_db: -20.5,
          },
          ai_recognition: { label: 'Snare', confidence: 92 },
        },
      ],
    },
    fallback,
  );

  assert.equal(vocal.pan, -25);
  assert.equal(vocal.gateLabel, 'G -47.5');
  assert.equal(vocal.compLabel, 'C 2.4 GR');
  assert.equal(vocal.status, 'ACTIVE');

  assert.equal(snare.pan, -12);
  assert.equal(snare.gateLabel, 'G -42.0');
  assert.equal(snare.compLabel, 'C -20.5');
  assert.equal(snare.status, 'ACTIVE');
});

test('adaptModuleStatusRecords appends read-only write surface inventory rows', async () => {
  const { adaptModuleStatusRecords } = await importTypeScript('services/operatorAdapters.ts');

  const rows = adaptModuleStatusRecords({
    module_status_map: [
      {
        module_id: 'processing_registry',
        label: 'Processing Registry',
        domain: 'product_layer',
        status: 'canonical',
        entrypoint: 'get_processing_pipeline_snapshot',
        module: 'backend/processing_registry.py',
        ui_surface: 'Settings',
        can_write_live: false,
        live_write_kinds: [],
        pipeline_profiles: ['main_live_assist'],
        next_step: 'Keep the registry read-only.',
      },
    ],
    write_surface_inventory: {
      surfaces: [
        {
          id: 'product_layer_fader_gain_apply',
          label: 'Product Layer supervised fader/gain apply',
          status: 'canonical_supervised',
          paths: ['backend/operator_proposal_queue.py'],
          entrypoints: ['apply_operator_proposal', 'manual_write_supervision'],
          write_kinds: ['fader', 'gain'],
          allowed_live: true,
          gate: 'apply_operator_proposal -> supervised manual write gate',
          current_policy: 'Only operator-approved fader/gain proposals may reach the WING executor.',
          next_action: 'Keep as the only live executor.',
        },
        {
          id: 'legacy_ws_to_osc_router',
          label: 'Legacy websocket to OSC router',
          status: 'blocked',
          paths: ['backend/routing.py'],
          entrypoints: ['command_to_osc'],
          write_kinds: ['fader', 'mute', 'pan'],
          allowed_live: false,
          gate: 'Blocked from canonical runtime.',
          current_policy: 'Do not route frontend commands directly to OSC.',
          next_action: 'Keep only as historical reference or remove after dependency scan.',
        },
      ],
    },
    safety_coverage_matrix: {
      rows: [
        {
          id: 'gain_supervised',
          label: 'Supervised input gain/trim',
          action_type: 'gain',
          status: 'dry_run_ready',
          live_allowed: false,
          canonical_gate: 'apply_operator_proposal -> supervised manual gate',
          current_blocker: 'Physical WING gain proof is not complete.',
          next_required_evidence: 'Run one small supervised gain proof.',
        },
      ],
    },
    autonomy_promotion_readiness: {
      summary: {
        promotion_gap_summary: {
          read_only: true,
          live_write: false,
          full_autonomous_ready: false,
          gap_bucket_counts: {
            autonomy_mode: 1,
            read_surface: 1,
            write_surface: 1,
            evidence_requirement: 3,
            proof: 0,
          },
          blocked_target_ids_by_bucket: {
            autonomy_mode: ['scene_preset_supervised_promotion'],
            read_surface: ['scene_preset_supervised_promotion'],
            write_surface: ['scene_preset_supervised_promotion'],
          },
          evidence_requirement_ids_by_action_family: {
            scene_preset: [
              'current_scene_preset_cache_freshness',
              'read_only_scoped_diff_evidence',
              'read_only_rollback_evidence',
            ],
          },
          next_required_evidence_by_action_family: {
            scene_preset: ['Keep scene apply blocked.'],
          },
        },
      },
      rows: [
        {
          id: 'scene_preset_supervised_promotion',
          label: 'Scene/preset supervised promotion',
          action_family: 'scene_preset',
          target_mode: 'supervised_live_candidate',
          status: 'manual_only_blocked',
          supervised_live_allowed: false,
          autonomous_ready: false,
          safety_row_id: 'routing_snapshots_scenes',
          safety_status: 'blocked',
          read_surface_statuses: ['partial', 'partial'],
          write_surface_statuses: ['quarantined'],
          blockers: [
            'safety_status_blocked',
            'write_surface_routing_snapshot_scene_writes_quarantined',
          ],
          next_required_evidence: ['Keep scene apply blocked.'],
          evidence_requirement_ids: [
            'current_scene_preset_cache_freshness',
            'read_only_scoped_diff_evidence',
            'read_only_rollback_evidence',
          ],
        },
      ],
    },
    autonomy_mode_transition_gate: {
      rows: [
        {
          mode: 'analyze',
          project_mode_name: 'Analysis Mode',
          status: 'available',
          runtime_entry_allowed: true,
          proposal_creation_allowed: false,
          live_apply_allowed: false,
          current_live_write_kinds: [],
          blockers: ['mode_has_no_live_apply_authority'],
          next_required_actions: [],
          readiness_gap_bucket_counts: {},
        },
        {
          mode: 'supervised',
          project_mode_name: 'Supervised AutoMix Mode',
          status: 'limited_available',
          runtime_entry_allowed: true,
          proposal_creation_allowed: true,
          live_apply_allowed: true,
          current_live_write_kinds: ['fader', 'gain'],
          blockers: ['supervised_mode_is_narrow_fader_gain_only'],
          next_required_actions: ['Keep physical proof review manual.'],
          readiness_gap_bucket_counts: {
            read_surface: 1,
          },
        },
        {
          mode: 'auto',
          project_mode_name: 'Full Autonomous Mixing Mode',
          status: 'blocked',
          runtime_entry_allowed: false,
          proposal_creation_allowed: true,
          live_apply_allowed: false,
          current_live_write_kinds: [],
          blockers: ['readiness_gap_read_surface', 'full_autonomous_mode_remains_blocked'],
          next_required_actions: ['Complete all supervised proof gates before Full Autonomous mode can be reviewed.'],
          readiness_gap_bucket_counts: {
            read_surface: 14,
            write_surface: 19,
          },
        },
      ],
    },
  });

  assert.deepEqual(
    rows.map(row => row.moduleId),
    [
      'processing_registry',
      'write_surface:product_layer_fader_gain_apply',
      'write_surface:legacy_ws_to_osc_router',
      'safety_coverage:gain_supervised',
      'mode_gate:analyze',
      'mode_gate:supervised',
      'mode_gate:auto',
      'promotion_readiness:scene_preset_supervised_promotion',
      'promotion_gap:autonomy_mode',
      'promotion_gap:read_surface',
      'promotion_gap:write_surface',
      'promotion_gap:evidence_requirement',
    ],
  );

  const liveSurface = rows[1];
  assert.equal(liveSurface.domain, 'write_surface');
  assert.equal(liveSurface.canWriteLive, true);
  assert.deepEqual(liveSurface.liveWriteKinds, ['fader', 'gain']);
  assert.equal(liveSurface.entrypoint, 'apply_operator_proposal / manual_write_supervision');
  assert.equal(liveSurface.tone, 'yellow');

  const blockedSurface = rows[2];
  assert.equal(blockedSurface.canWriteLive, false);
  assert.equal(blockedSurface.status, 'blocked');
  assert.equal(blockedSurface.blockedReason, 'Do not route frontend commands directly to OSC.');
  assert.equal(blockedSurface.tone, 'red');

  const gainCoverage = rows[3];
  assert.equal(gainCoverage.domain, 'safety_coverage');
  assert.equal(gainCoverage.status, 'dry_run_ready');
  assert.equal(gainCoverage.canWriteLive, false);
  assert.equal(gainCoverage.blockedReason, 'Physical WING gain proof is not complete.');
  assert.equal(gainCoverage.tone, 'blue');

  const analysisMode = rows.find(row => row.moduleId === 'mode_gate:analyze');
  assert.equal(analysisMode.domain, 'autonomy_mode_gate');
  assert.equal(analysisMode.label, 'Analysis Mode');
  assert.equal(analysisMode.canWriteLive, false);
  assert.deepEqual(analysisMode.liveWriteKinds, []);
  assert.equal(analysisMode.tone, 'green');
  assert.match(analysisMode.blockedReason, /mode_has_no_live_apply_authority/);

  const supervisedMode = rows.find(row => row.moduleId === 'mode_gate:supervised');
  assert.equal(supervisedMode.label, 'Supervised AutoMix Mode');
  assert.equal(supervisedMode.status, 'limited_available');
  assert.equal(supervisedMode.canWriteLive, false);
  assert.deepEqual(supervisedMode.liveWriteKinds, []);
  assert.equal(supervisedMode.tone, 'yellow');
  assert.match(supervisedMode.module, /policy:fader,gain/);
  assert.match(supervisedMode.nextStep, /Current policy live-write kinds: fader, gain/);

  const autoMode = rows.find(row => row.moduleId === 'mode_gate:auto');
  assert.equal(autoMode.label, 'Full Autonomous Mixing Mode');
  assert.equal(autoMode.status, 'blocked');
  assert.equal(autoMode.canWriteLive, false);
  assert.deepEqual(autoMode.liveWriteKinds, []);
  assert.equal(autoMode.tone, 'red');
  assert.match(autoMode.blockedReason, /full_autonomous_mode_remains_blocked/);
  assert.match(autoMode.nextStep, /Full Auto gaps: read_surface:14, write_surface:19/);

  const scenePresetReadiness = rows.find(row => row.moduleId === 'promotion_readiness:scene_preset_supervised_promotion');
  assert.equal(scenePresetReadiness.domain, 'promotion_readiness');
  assert.equal(scenePresetReadiness.canWriteLive, false);
  assert.deepEqual(scenePresetReadiness.liveWriteKinds, []);
  assert.equal(scenePresetReadiness.status, 'manual_only_blocked');
  assert.equal(scenePresetReadiness.entrypoint, 'scene_preset');
  assert.equal(scenePresetReadiness.module, 'routing_snapshots_scenes');
  assert.equal(scenePresetReadiness.tone, 'red');
  assert.match(scenePresetReadiness.blockedReason, /safety_status_blocked/);
  assert.match(scenePresetReadiness.nextStep, /read_only_scoped_diff_evidence/);

  const readSurfaceGap = rows.find(row => row.moduleId === 'promotion_gap:read_surface');
  assert.equal(readSurfaceGap.domain, 'promotion_gap');
  assert.equal(readSurfaceGap.canWriteLive, false);
  assert.deepEqual(readSurfaceGap.liveWriteKinds, []);
  assert.equal(readSurfaceGap.status, 'blocked');
  assert.equal(readSurfaceGap.entrypoint, 'read_surface');
  assert.equal(readSurfaceGap.tone, 'red');
  assert.match(readSurfaceGap.blockedReason, /scene_preset_supervised_promotion/);

  const evidenceGap = rows.find(row => row.moduleId === 'promotion_gap:evidence_requirement');
  assert.equal(evidenceGap.canWriteLive, false);
  assert.equal(evidenceGap.module, 'all_promotion_targets');
  assert.match(evidenceGap.nextStep, /read_only_scoped_diff_evidence/);
  assert.match(evidenceGap.nextStep, /Keep scene apply blocked/);
});

test('adaptTelemetryAlertsSnapshot maps generated alert artifacts as read-only evidence', async () => {
  const { adaptTelemetryAlertsSnapshot } = await importTypeScript('services/operatorAdapters.ts');

  const snapshot = adaptTelemetryAlertsSnapshot({
    type: 'telemetry_alerts_snapshot',
    generated_at: '2026-06-01T18:30:00Z',
    source: 'telemetry_alert_artifacts',
    available: true,
    storage_dir: 'data/telemetry/sessions',
    safety: {
      read_only: true,
      live_write: false,
      generates_artifacts: false,
    },
    summary: {
      session_count: 3,
      artifact_session_count: 1,
      alert_count: 2,
      critical_count: 1,
      warning_count: 1,
      latest_session_id: 'session-1',
    },
    sessions: [
      {
        session_id: 'session-1',
        generated_at: '2026-06-01T18:29:59Z',
        event_count: 12,
        alert_count: 2,
        dashboard_row_count: 12,
        safety: {
          read_only: true,
          live_write: false,
        },
      },
    ],
    alerts: [
      {
        timestamp_ms: 2000,
        session_id: 'session-1',
        rule: 'clipping_risk',
        severity: 'critical',
        target_type: 'channel',
        target_id: '12',
        message: 'Peak -0.05 dBFS is at or above clipping guard.',
        value: -0.05,
        threshold: -0.1,
      },
      {
        timestamp_ms: 1000,
        session_id: 'session-1',
        rule: 'headroom_low',
        severity: 'warning',
        target_type: 'bus',
        target_id: 'main',
        message: 'Peak leaves limited headroom.',
        value: -2.5,
        threshold: -3,
      },
    ],
  });

  assert.equal(snapshot.available, true);
  assert.equal(snapshot.alertCount, 2);
  assert.equal(snapshot.criticalCount, 1);
  assert.equal(snapshot.warningCount, 1);
  assert.equal(snapshot.readOnly, true);
  assert.equal(snapshot.liveWrite, false);
  assert.equal(snapshot.generatesArtifacts, false);
  assert.equal(snapshot.sessions[0].readOnly, true);
  assert.equal(snapshot.sessions[0].liveWrite, false);
  assert.deepEqual(
    snapshot.alerts.map(row => [row.rule, row.target, row.tone]),
    [
      ['clipping risk', 'channel:12', 'red'],
      ['headroom low', 'bus:main', 'yellow'],
    ],
  );
  assert.equal(snapshot.alerts[0].time, '2000 ms');
  assert.equal(snapshot.alerts[0].value, '-0.05');
  assert.equal(snapshot.alerts[0].threshold, '-0.10');
});

test('Settings bottom section grid has one column per read-only section card', async () => {
  const source = await fs.readFile(path.join(srcRoot, 'views/SettingsView.tsx'), 'utf8');
  const sectionTitles = [
    'Mix Defaults',
    'Mode Policy',
    'Write Policy',
    'Module Status',
    'Audit',
    'Transport',
  ];

  assert.match(source, /className="grid grid-cols-6 gap-2"/);
  for (const title of sectionTitles) {
    assert.match(source, new RegExp(`<Panel title="${title}"`));
  }
});

test('Settings Mode Policy panel summarizes read-only mode gate rows', async () => {
  const source = await fs.readFile(path.join(srcRoot, 'views/SettingsView.tsx'), 'utf8');

  assert.match(source, /function modePolicyRows/);
  assert.match(source, /modeGateRow\(modules, 'supervised'\)/);
  assert.match(source, /modeGateRow\(modules, 'auto'\)/);
  assert.match(source, /current_write:/);
  assert.match(source, /Full Auto/);
  assert.match(source, /<Panel title="Mode Policy">/);
  assert.match(source, /rows=\{modeRows\.slice\(0, 2\)\}/);
});

test('directorDecisionRows preserves backend task status instead of forcing applied', async () => {
  const { directorDecisionRows } = await importTypeScript('components/dashboard/directorPanelModel.ts');

  const rows = directorDecisionRows(
    [
      {
        id: 'proposal-safe-gain-1',
        priority: 'High',
        title: 'SafeGain trim review',
        agent: 'SafeGain',
        status: 'Pending',
        time: '10:31',
        target: 'CH 12',
      },
      {
        id: 'proposal-eq-1',
        priority: 'Medium',
        title: 'EQ proposal remains review-only',
        agent: 'AutoEQ',
        status: 'Review',
        time: '10:32',
      },
      {
        id: 'proposal-applied-1',
        priority: 'Low',
        title: 'Operator-approved fader down',
        agent: 'AutoFader',
        status: 'Applied',
        time: '10:33',
      },
    ],
    [
      {
        time: '10:00',
        title: 'Fallback decision',
        target: 'CH 01',
        status: 'Applied',
        tone: 'green',
      },
    ],
  );

  assert.deepEqual(
    rows.map(row => row.status),
    ['Pending', 'Review', 'Applied'],
  );
  assert.equal(rows[0].tone, 'yellow');
  assert.equal(rows[1].tone, 'yellow');
  assert.equal(rows[2].tone, 'green');
  assert.equal(rows[1].target, 'proposal-eq-1');
});

test('directorDecisionRows keeps mock fallback only when no backend tasks exist', async () => {
  const { directorDecisionRows } = await importTypeScript('components/dashboard/directorPanelModel.ts');

  const fallback = [
    {
      time: '10:00',
      title: 'Fallback decision',
      target: 'CH 01',
      status: 'Review',
      tone: 'purple',
    },
  ];

  assert.deepEqual(directorDecisionRows([], fallback), fallback);
});

test('toneForDirectorStatus keeps non-applied director states visually distinct', async () => {
  const { toneForDirectorStatus } = await importTypeScript('components/dashboard/directorPanelModel.ts');

  assert.equal(toneForDirectorStatus('Pending', 'Low'), 'yellow');
  assert.equal(toneForDirectorStatus('Queued'), 'yellow');
  assert.equal(toneForDirectorStatus('Blocked'), 'red');
  assert.equal(toneForDirectorStatus('Failed'), 'red');
  assert.equal(toneForDirectorStatus('Safe'), 'green');
  assert.equal(toneForDirectorStatus('Approved'), 'green');
});

test('directorProposalRows derives Agent Suggestions from backend task rows', async () => {
  const { directorProposalRows } = await importTypeScript('components/dashboard/directorPanelModel.ts');

  const rows = directorProposalRows(
    [
      {
        id: 'proposal-safe-gain-1',
        priority: 'High',
        title: 'SafeGain trim review',
        agent: 'SafeGain',
        status: 'Pending',
        time: '10:31',
        target: 'CH 12',
        reason: 'Operator review required before any supervised gate apply.',
        requestedValueType: 'gain',
      },
      {
        id: 'proposal-eq-1',
        priority: 'Medium',
        title: 'EQ remains proposal-only',
        agent: 'AutoEQ',
        status: 'Review',
        time: '10:32',
        reviewOnly: true,
        canApply: false,
        backendStatus: 'proposal_only',
      },
    ],
    [
      {
        id: 'fallback',
        agent: 'Mock',
        title: 'Fallback suggestion',
        detail: 'UI fallback',
        status: 'Review',
        tone: 'purple',
        reviewMessage: 'fallback reviewed',
        rejectMessage: 'fallback rejected',
      },
    ],
  );

  assert.deepEqual(
    rows.map(row => row.id),
    ['proposal-safe-gain-1', 'proposal-eq-1'],
  );
  assert.equal(rows[0].agent, 'SafeGain');
  assert.equal(rows[0].status, 'Pending');
  assert.equal(rows[0].tone, 'yellow');
  assert.equal(rows[0].detail, 'Operator review required before any supervised gate apply.');
  assert.equal(rows[0].reviewMessage.includes('No OSC write sent'), true);
  assert.equal(rows[1].detail, 'proposal_only - review only');
  assert.equal(rows[1].status, 'Review only');
  assert.equal(rows[1].tone, 'red');
});

test('directorProposalRows keeps hardcoded suggestions only without backend tasks', async () => {
  const { directorProposalRows } = await importTypeScript('components/dashboard/directorPanelModel.ts');

  const fallback = [
    {
      id: 'fallback',
      agent: 'Mock',
      title: 'Fallback suggestion',
      detail: 'UI fallback',
      status: 'Review',
      tone: 'purple',
      reviewMessage: 'fallback reviewed',
      rejectMessage: 'fallback rejected',
    },
  ];

  assert.deepEqual(directorProposalRows([], fallback), fallback);
});

test('directorProposalRows suppresses fallback suggestions for backend source', async () => {
  const { directorFallbackRowsForSource, directorDecisionRows, directorProposalRows } = await importTypeScript('components/dashboard/directorPanelModel.ts');

  const fallback = [
    {
      id: 'fallback',
      agent: 'Mock',
      title: 'Fallback suggestion',
      detail: 'UI fallback',
      status: 'Review',
      tone: 'purple',
      reviewMessage: 'fallback reviewed',
      rejectMessage: 'fallback rejected',
    },
  ];
  const fallbackDecision = [
    {
      time: '10:00',
      title: 'Fallback decision',
      target: 'CH 01',
      status: 'Review',
      tone: 'purple',
    },
  ];

  assert.deepEqual(directorFallbackRowsForSource('Backend', fallback), []);
  assert.deepEqual(directorFallbackRowsForSource('Mock', fallback), fallback);
  assert.deepEqual(directorProposalRows([], fallback, 'Backend'), []);
  assert.deepEqual(directorDecisionRows([], fallbackDecision, 'Backend'), []);
  assert.deepEqual(directorProposalRows([], fallback, 'Mock'), fallback);
});

test('directorEventRows derives backend events from runtime state instead of mock rows', async () => {
  const { directorEventRows } = await importTypeScript('components/dashboard/directorPanelModel.ts');

  const fallbackEvents = [
    {
      time: '10:31:08',
      source: 'Telemetry',
      message: 'Mock meter frame accepted for dashboard render.',
      tone: 'blue',
    },
  ];
  const runtime = {
    currentGoal: 'Keep writes supervised.',
    readyLabel: 'Not live ready',
    readyTone: 'red',
    audioLabel: 'Read-only',
    audioTone: 'yellow',
    proposalQueueLabel: '2 pending',
    proposalQueueTone: 'yellow',
    liveWritePolicy: 'supervised_fader_gain_only',
    manualGateLabel: 'Manual gate armed',
    manualGateTone: 'green',
    resetRequired: false,
    hardwareProofItems: [
      { label: 'Readback', value: 'cache only', tone: 'yellow' },
      { label: 'Rollback', value: 'required', tone: 'yellow' },
    ],
    blockedReasons: ['physical_pre_read_required'],
    statusItems: [],
    sourceLabel: 'Backend',
  };

  const rows = directorEventRows(runtime, fallbackEvents);

  assert.deepEqual(
    rows.map(row => row.source),
    ['Safety', 'Readiness', 'Blockers'],
  );
  assert.equal(rows.some(row => row.message.includes('Mock meter frame')), false);
  assert.equal(rows[0].message.includes('supervised_fader_gain_only'), true);
  assert.equal(rows[2].message, 'physical_pre_read_required');
});

test('directorEventRows keeps static events only for mock source', async () => {
  const { directorEventRows } = await importTypeScript('components/dashboard/directorPanelModel.ts');

  const fallbackEvents = [
    {
      time: '10:31:08',
      source: 'Telemetry',
      message: 'Mock meter frame accepted for dashboard render.',
      tone: 'blue',
    },
  ];
  const runtime = {
    currentGoal: 'Mock goal',
    readyLabel: 'Mock only',
    readyTone: 'red',
    audioLabel: 'Mock meters',
    audioTone: 'yellow',
    proposalQueueLabel: 'Mock queue',
    proposalQueueTone: 'blue',
    liveWritePolicy: 'mock_frontend_only',
    manualGateLabel: 'UI only',
    manualGateTone: 'yellow',
    resetRequired: false,
    hardwareProofItems: [],
    blockedReasons: [],
    statusItems: [],
    sourceLabel: 'Mock',
  };

  assert.deepEqual(directorEventRows(runtime, fallbackEvents), fallbackEvents);
});

test('directorProposalRows excludes completed backend tasks from active suggestions', async () => {
  const { directorProposalRows } = await importTypeScript('components/dashboard/directorPanelModel.ts');

  const rows = directorProposalRows(
    [
      {
        id: 'proposal-applied-1',
        priority: 'Low',
        title: 'Operator-approved fader down',
        agent: 'AutoFader',
        status: 'Applied',
        time: '10:33',
      },
      {
        id: 'proposal-rejected-1',
        priority: 'Medium',
        title: 'Rejected EQ change',
        agent: 'AutoEQ',
        status: 'Rejected',
        time: '10:34',
      },
      {
        id: 'proposal-review-1',
        priority: 'High',
        title: 'Review-only SafeGain evidence',
        agent: 'SafeGain',
        status: 'Review',
        time: '10:35',
        reviewOnly: true,
        canApply: false,
        backendStatus: 'proposal_only',
      },
    ],
    [
      {
        id: 'fallback',
        agent: 'Mock',
        title: 'Fallback suggestion',
        detail: 'UI fallback',
        status: 'Review',
        tone: 'purple',
        reviewMessage: 'fallback reviewed',
        rejectMessage: 'fallback rejected',
      },
    ],
  );

  assert.deepEqual(
    rows.map(row => row.id),
    ['proposal-review-1'],
  );
  assert.equal(rows[0].detail, 'proposal_only - review only');
  assert.equal(rows[0].status, 'Review only');
  assert.equal(rows[0].tone, 'red');
});

test('directorProposalRows treats apply blockers as review-only safety evidence', async () => {
  const { directorProposalRows, taskRequiresReviewOnly } = await importTypeScript('components/dashboard/directorPanelModel.ts');

  const blockedTask = {
    id: 'proposal-gain-proof-1',
    priority: 'High',
    title: 'Gain proof requires physical pre-read',
    agent: 'Product Layer',
    status: 'Review',
    time: '10:36',
    target: 'CH 09',
    canApply: true,
    backendStatus: 'blocked_safety_check',
    applyBlockers: ['physical_pre_read_required', 'rollback_readback_missing'],
    requestedValueType: 'gain',
  };

  const rows = directorProposalRows(
    [blockedTask],
    [
      {
        id: 'fallback',
        agent: 'Mock',
        title: 'Fallback suggestion',
        detail: 'UI fallback',
        status: 'Review',
        tone: 'purple',
        reviewMessage: 'fallback reviewed',
        rejectMessage: 'fallback rejected',
      },
    ],
  );

  assert.equal(taskRequiresReviewOnly(blockedTask), true);
  assert.equal(rows[0].status, 'Review only');
  assert.equal(rows[0].tone, 'red');
  assert.equal(rows[0].detail, 'physical_pre_read_required, rollback_readback_missing - review only');
  assert.equal(rows[0].reviewMessage.includes('No OSC write sent'), true);
});

test('dashboard director panel renders agent suggestions as review-only controls', async () => {
  const source = await fs.readFile(
    path.join(srcRoot, 'components/dashboard/DirectorPanel.tsx'),
    'utf8',
  );

  assert.match(source, /directorProposalRows/);
  assert.match(source, /Просмотр/);
  assert.match(source, /No active backend proposals require review/);
  assert.match(source, /No OSC write sent/);
  assert.equal(source.includes('Принять'), false);
});

test('dashboard mixer header uses the selected channel dynamically', async () => {
  const source = await fs.readFile(
    path.join(srcRoot, 'components/dashboard/AutomixerDashboard.tsx'),
    'utf8',
  );

  assert.match(source, /selectedChannel/);
  assert.match(source, /Selected: \{selectedChannel \? `CH/);
  assert.match(source, /suppressMockSolo/);
  assert.equal(source.includes('Selected: Lead Vocal'), false);
});

test('dashboardTelemetryForChannel derives read-only strip state from channel snapshots', async () => {
  const { dashboardTelemetryForChannel } = await importTypeScript('components/dashboard/dashboardTelemetryModel.ts');

  const fallback = {
    waveform: [20, 40, 60],
    meterL: -18,
    meterR: -19,
    gateLabel: 'G -40',
    compLabel: 'C 2.0',
    safeState: 'FOCUS',
    pan: 0,
    muted: false,
    solo: true,
    accent: 'purple',
  };

  const muted = dashboardTelemetryForChannel(
    {
      id: 14,
      name: 'Backing Vocal',
      type: 'Vocal Back',
      source: 'In 14',
      level: -48,
      lufs: -55,
      dynamic: 2,
      status: 'MUTED',
      recognition: 'Backing Vocal',
      group: 'VOCALS',
      fader: -50,
      confidence: 95,
    },
    fallback,
  );

  assert.equal(muted.meterL, -48);
  assert.equal(muted.meterR, -48);
  assert.equal(muted.muted, true);
  assert.equal(muted.safeState, 'MUTE');
  assert.equal(muted.accent, 'red');
  assert.deepEqual(muted.waveform, fallback.waveform);
  assert.equal(muted.solo, true);

  const backendMuted = dashboardTelemetryForChannel(
    {
      id: 14,
      name: 'Backing Vocal',
      type: 'Vocal Back',
      source: 'In 14',
      level: -48,
      lufs: -55,
      dynamic: 2,
      status: 'MUTED',
      recognition: 'Backing Vocal',
      group: 'VOCALS',
      fader: -50,
      confidence: 95,
    },
    fallback,
    { suppressMockSolo: true },
  );

  assert.equal(backendMuted.solo, false);

  const active = dashboardTelemetryForChannel(
    {
      id: 12,
      name: 'Lead Vocal',
      type: 'Vocal Lead',
      source: 'In 12',
      level: -18,
      lufs: -18,
      dynamic: 6,
      status: 'ACTIVE',
      recognition: 'Lead Vocal',
      group: 'VOCALS',
      fader: -10,
      pan: -18,
      gateLabel: 'G -46',
      compLabel: 'C 3.2',
      confidence: 98,
    },
    fallback,
  );

  assert.equal(active.safeState, 'OK');
  assert.equal(active.muted, false);
  assert.equal(active.accent, 'purple');
  assert.equal(active.pan, -18);
  assert.equal(active.gateLabel, 'G -46');
  assert.equal(active.compLabel, 'C 3.2');

  const hot = dashboardTelemetryForChannel(
    {
      id: 12,
      name: 'Lead Vocal',
      type: 'Vocal Lead',
      source: 'In 12',
      level: -8.5,
      lufs: -18,
      dynamic: 6,
      status: 'ACTIVE',
      recognition: 'Lead Vocal',
      group: 'VOCALS',
      fader: -10,
      confidence: 98,
    },
    fallback,
  );

  assert.equal(hot.safeState, 'HOT');
  assert.equal(hot.accent, 'yellow');

  const clipping = dashboardTelemetryForChannel(
    {
      id: 13,
      name: 'Guest Mic',
      type: 'Vocal Lead',
      source: 'In 13',
      level: -2.5,
      lufs: -10,
      dynamic: 4,
      status: 'ACTIVE',
      recognition: 'Vocal',
      group: 'VOCALS',
      fader: -6,
      confidence: 90,
    },
    fallback,
  );

  assert.equal(clipping.safeState, 'CLIP');
  assert.equal(clipping.accent, 'red');
});

test('dashboard analyzer metrics populate read-only phase dynamics and LUFS cards', async () => {
  const { adaptDashboardRuntime } = await importTypeScript('services/operatorAdapters.ts');
  const { dashboardAnalysisCards } = await importTypeScript('components/dashboard/dashboardTelemetryModel.ts');

  const runtime = adaptDashboardRuntime(
    {
      master_bus: {
        integrated_lufs: -18.2,
        true_peak_db: -2.4,
      },
      analyzer: {
        spectrum_bins: [12, 24, 36, 48],
        phase_history: [-0.5, 0, 0.5],
        phase_label: 'Phase locked',
        phase_summary: 'OH pair aligned',
        correlation_history: [-0.25, 0.25],
        dynamics_history: [44, 55, 66],
        dynamics_label: 'GR avg 2.4',
        dynamics_summary: 'Transient window stable',
        lufs_history: [-30, -24, -18],
        lufs_label: '-18.2 LUFS',
        loudness_summary: 'Inside Ayaic corridor',
      },
      channel_summary: {
        active_channels: 14,
        total_channels: 48,
      },
    },
    {
      agents: [
        { confidence: 88 },
      ],
    },
    {
      summary: {
        pending_count: 2,
        blocked_count: 1,
      },
    },
  );

  assert.deepEqual(runtime.spectrumBins, [12, 24, 36, 48]);
  assert.deepEqual(runtime.analysisTraces.phase.values, [31, 50, 69]);
  assert.deepEqual(runtime.analysisTraces.correlation.values, [40.5, 59.5]);
  assert.deepEqual(runtime.analysisTraces.dynamics.values, [44, 55, 66]);
  assert.deepEqual(runtime.analysisTraces.loudness.values, [50, 60, 70]);
  assert.equal(runtime.analysisTraces.phase.sourceLabel, 'Backend analyzer DTO');
  assert.equal(runtime.analysisTraces.phase.sourceTone, 'green');

  const cards = dashboardAnalysisCards(runtime, [
    { id: 'phase', title: 'Phase Scope', accent: 'green', values: [10], primary: 'fallback phase', secondary: 'fallback secondary' },
    { id: 'dynamics', title: 'Dynamics Overview', accent: 'yellow', values: [20], primary: 'fallback dynamics', secondary: 'fallback secondary' },
    { id: 'loudness', title: 'Loudness History', accent: 'green', values: [30], primary: 'fallback loudness', secondary: 'fallback secondary' },
    { id: 'spectral', title: 'Spectral Balance', accent: 'purple', values: [40], primary: 'fallback spectral', secondary: 'fallback secondary' },
  ]);

  assert.equal(cards[0].primary, 'Phase locked');
  assert.equal(cards[0].secondary, 'OH pair aligned');
  assert.equal(cards[0].sourceLabel, 'Backend DTO');
  assert.equal(cards[0].sourceTone, 'green');
  assert.deepEqual(cards[1].values, [44, 55, 66]);
  assert.equal(cards[2].primary, '-18.2 LUFS');
  assert.equal(cards[3].sourceLabel, undefined);
  assert.deepEqual(cards[3].values, [40]);

  const dashboardMockSource = await fs.readFile(path.join(srcRoot, 'mock/dashboardMock.ts'), 'utf8');
  assert.equal(dashboardMockSource.includes('TODO(analyzer metrics)'), false);
});

test('read-only DTO adapters preserve explicit empty backend snapshots', async () => {
  const {
    adaptBusesSnapshot,
    adaptCommunicationsSnapshot,
    adaptPerformanceRecords,
    adaptMemorySnapshot,
    adaptTrainingSnapshot,
    adaptRulesSnapshot,
    adaptDevices,
  } = await importTypeScript('services/operatorAdapters.ts');

  const fallbackBus = [{
    name: 'Mock Bus',
    role: 'Group',
    level: -12,
    lufs: -14,
    channels: 2,
    color: 'green',
  }];
  const fallbackCommunication = [{
    time: '10:00',
    from: 'Mock',
    to: 'Director',
    topic: 'Fallback message',
    status: 'Info',
    tone: 'blue',
  }];
  const fallbackMemory = [{
    name: 'Mock memory',
    type: 'Mock',
    size: '1 KB',
    confidence: 80,
    updated: 'mock',
  }];
  const fallbackTraining = [{
    name: 'Mock training',
    status: 'Queued',
    examples: 1,
    score: '80%',
    updated: 'mock',
  }];
  const fallbackRule = [{
    name: 'Mock rule',
    category: 'Mock',
    confidence: '80%',
    state: 'Review',
  }];
  const fallbackDevice = [{
    name: 'Mock Device',
    ip: '127.0.0.1',
    type: 'Console',
    sampleRate: '48 kHz',
    io: 'mock',
    status: 'Online',
  }];

  assert.deepEqual(adaptBusesSnapshot({ buses: [] }, fallbackBus), []);
  assert.deepEqual(adaptCommunicationsSnapshot({ communications: [] }, fallbackCommunication), []);
  assert.deepEqual(adaptPerformanceRecords({ agents: [] }, {}, [{
    agent: 'Mock Agent',
    cpu: '1%',
    memory: '1 MB',
    decisions: 1,
    success: '80%',
    latency: '1 ms',
    status: 'Good',
  }]), []);
  assert.deepEqual(adaptMemorySnapshot({ memories: [] }, fallbackMemory), []);
  assert.deepEqual(adaptTrainingSnapshot({ sessions: [] }, fallbackTraining), []);
  assert.deepEqual(adaptRulesSnapshot({ rules: [] }, fallbackRule), []);
  assert.deepEqual(adaptDevices({ connected_devices: [], nodes: [] }, fallbackDevice), []);

  assert.deepEqual(adaptBusesSnapshot(undefined, fallbackBus), fallbackBus);
  assert.deepEqual(adaptCommunicationsSnapshot(undefined, fallbackCommunication), fallbackCommunication);
  assert.deepEqual(adaptPerformanceRecords(undefined, {}, [{
    agent: 'Mock Agent',
    cpu: '1%',
    memory: '1 MB',
    decisions: 1,
    success: '80%',
    latency: '1 ms',
    status: 'Good',
  }]).length, 1);
  assert.deepEqual(adaptMemorySnapshot(undefined, fallbackMemory), fallbackMemory);
  assert.deepEqual(adaptTrainingSnapshot(undefined, fallbackTraining), fallbackTraining);
  assert.deepEqual(adaptRulesSnapshot(undefined, fallbackRule), fallbackRule);
  assert.deepEqual(adaptDevices(undefined, fallbackDevice), fallbackDevice);
});

test('cockpit marks visual placeholder metrics instead of presenting them as live telemetry', async () => {
  const dashboardSource = await fs.readFile(path.join(srcRoot, 'components/dashboard/AutomixerDashboard.tsx'), 'utf8');
  const performanceSource = await fs.readFile(path.join(srcRoot, 'views/PerformanceView.tsx'), 'utf8');
  const systemSource = await fs.readFile(path.join(srcRoot, 'views/SystemViews.tsx'), 'utf8');
  const backendHookSource = await fs.readFile(path.join(srcRoot, 'services/useOperatorBackend.ts'), 'utf8');
  const shellSource = await fs.readFile(path.join(srcRoot, 'components/prototype/AutomixerComponents.tsx'), 'utf8');
  const agentsSource = await fs.readFile(path.join(srcRoot, 'components/prototype/AgentsConsolePanels.tsx'), 'utf8');

  assert.match(dashboardSource, /Backend DTO/);
  assert.match(dashboardSource, /Display-only/);
  assert.equal(dashboardSource.includes('BS.1770 mock readout'), false);
  assert.match(backendHookSource, /performance: payloadSource\(payloads\.agents/);
  assert.match(performanceSource, /data\.readModelSources\.performance/);
  assert.match(performanceSource, /rowsForReadModelSource/);
  assert.equal(performanceSource.includes("data.performanceRecords.length ? 'Backend-derived' : 'Mock'"), false);
  assert.match(performanceSource, /Display Trace/);
  assert.match(performanceSource, /Trend traces/);
  assert.match(systemSource, /System Latency · Display-only/);
  assert.match(systemSource, /Activity trace is display-only/);
  assert.match(shellSource, /Display-only health/);
  assert.match(agentsSource, /Traffic by Agent · Display Trace/);
});
