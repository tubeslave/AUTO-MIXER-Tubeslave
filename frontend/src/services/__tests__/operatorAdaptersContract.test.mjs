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

test('toUiMode normalizes known aliases to canonical operator modes', async () => {
  const { toUiMode } = await importTypeScript('services/operatorAdapters.ts');

  assert.equal(toUiMode('analysis mode'), 'Analyze');
  assert.equal(toUiMode('analysis-mode'), 'Analyze');
  assert.equal(toUiMode('analysis_mode'), 'Analyze');
  assert.equal(toUiMode('full autonomous mixing mode'), 'Auto');
  assert.equal(toUiMode('full-autonomous-mixing-mode'), 'Auto');
  assert.equal(toUiMode('full autonomous mode'), 'Auto');
  assert.equal(toUiMode('full_autonomous_mode'), 'Auto');
  assert.equal(toUiMode('full_autonomous_mixing'), 'Auto');
  assert.equal(toUiMode('full autonomous mix'), 'Auto');
  assert.equal(toUiMode('full autonomous mix mode'), 'Auto');
  assert.equal(toUiMode('full_autonomous_mix_mode'), 'Auto');
  assert.equal(toUiMode('full-autonomous-mix'), 'Auto');
  assert.equal(toUiMode('full_autonomous--mix'), 'Auto');
  assert.equal(toUiMode('full  autonomous   mix'), 'Auto');
  assert.equal(toUiMode('supervised automix'), 'Supervised');
  assert.equal(toUiMode('supervised automix mode'), 'Supervised');
  assert.equal(toUiMode('supervised auto mix'), 'Supervised');
  assert.equal(toUiMode('supervised_auto_mode'), 'Supervised');
  assert.equal(toUiMode('assisted mode'), 'Assist');
  assert.equal(toUiMode('unknown-mode'), 'Assist');
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

test('mode display labels map objective names to each operational mode', async () => {
  const { getModeDisplayLabel, modeDisplayLabel, formatOperatorModeLabel, formatAIAgentModeLabel } = await importTypeScript('utils/prototypeTone.ts');

  assert.equal(getModeDisplayLabel('Analyze'), 'Analysis Mode');
  assert.equal(getModeDisplayLabel('Assist'), 'Assisted Mixing');
  assert.equal(getModeDisplayLabel('Supervised'), 'Supervised AutoMix');
  assert.equal(getModeDisplayLabel('Auto'), 'Full Autonomous Mixing');
  assert.deepEqual(Object.keys(modeDisplayLabel).sort(), ['Analyze', 'Assist', 'Auto', 'Supervised']);
  assert.equal(formatOperatorModeLabel('analysis_mode'), 'Analysis Mode');
  assert.equal(formatOperatorModeLabel('supervised auto mix'), 'Supervised AutoMix');
  assert.equal(formatOperatorModeLabel('full autonomous mode'), 'Full Autonomous Mixing');
  assert.equal(formatOperatorModeLabel('analysis-mode'), 'Analysis Mode');
  assert.equal(formatOperatorModeLabel('full-autonomous-mode'), 'Full Autonomous Mixing');
  assert.equal(formatOperatorModeLabel('full autonomous mix'), 'Full Autonomous Mixing');
  assert.equal(formatOperatorModeLabel('full_autonomous_mixing'), 'Full Autonomous Mixing');
  assert.equal(formatOperatorModeLabel('full autonomous mix mode'), 'Full Autonomous Mixing');
  assert.equal(formatOperatorModeLabel('full_autonomous_mix'), 'Full Autonomous Mixing');
  assert.equal(formatOperatorModeLabel('full_autonomous_mix_mode'), 'Full Autonomous Mixing');
  assert.equal(formatOperatorModeLabel('full-autonomous-mix'), 'Full Autonomous Mixing');
  assert.equal(formatOperatorModeLabel('full  autonomous   mix'), 'Full Autonomous Mixing');
  assert.equal(formatOperatorModeLabel('full--autonomous--mix'), 'Full Autonomous Mixing');
  assert.equal(formatOperatorModeLabel('supervised automix mode'), 'Supervised AutoMix');
  assert.equal(formatOperatorModeLabel('full autonomous mixing'), 'Full Autonomous Mixing');
  assert.equal(formatOperatorModeLabel('auto mode'), 'Full Autonomous Mixing');
  assert.equal(formatOperatorModeLabel('unknown'), 'Assisted Mixing');
  assert.equal(formatAIAgentModeLabel('suggest'), 'Suggest');
  assert.equal(formatAIAgentModeLabel('manual_mode'), 'Manual');
  assert.equal(formatAIAgentModeLabel('full autonomous'), 'Auto');
  assert.equal(formatAIAgentModeLabel('full autonomous mix'), 'Auto');
  assert.equal(formatAIAgentModeLabel('full autonomous mix mode'), 'Auto');
  assert.equal(formatAIAgentModeLabel('full_autonomous_mix'), 'Auto');
  assert.equal(formatAIAgentModeLabel('full_autonomous_mix_mode'), 'Auto');
  assert.equal(formatAIAgentModeLabel(undefined), 'Suggest');
  assert.equal(formatAIAgentModeLabel(null), 'Suggest');
  assert.equal(formatAIAgentModeLabel(0), 'Suggest');
  assert.equal(formatAIAgentModeLabel('unknown-mode'), 'Unknown Mode');
  assert.equal(formatAIAgentModeLabel(''), 'Suggest');
});

test('mode formatting falls back safely on malformed inputs', async () => {
  const { formatOperatorModeLabel } = await importTypeScript('utils/prototypeTone.ts');
  const { toUiMode } = await importTypeScript('services/operatorAdapters.ts');

  assert.equal(toUiMode(undefined), 'Assist');
  assert.equal(toUiMode(null), 'Assist');
  assert.equal(toUiMode(0), 'Assist');
  assert.equal(formatOperatorModeLabel(undefined), 'Assisted Mixing');
  assert.equal(formatOperatorModeLabel(null), 'Assisted Mixing');
  assert.equal(formatOperatorModeLabel(0), 'Assisted Mixing');
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
        autonomy_promotion_full_autonomous_ready: false,
        autonomy_mode_transition_full_autonomous_ready: true,
        autonomy_promotion_supervised_live_allowed_target_count: 2,
        autonomy_promotion_blocked_target_count: 4,
        autonomy_mode_transition_gate_mode_count: 3,
        autonomy_mode_transition_gate_blocked_mode_count: 1,
        autonomy_mode_transition_gate_blocked_modes: ['analyze'],
        autonomy_mode_transition_gate_live_apply_allowed_modes: ['supervised'],
        autonomy_mode_transition_gate_supervised_current_live_write_kinds: ['fader'],
        autonomy_mode_transition_gate_supervised_current_live_write_surfaces: ['preamp', 'trim', 'fader'],
        offline_integration_bundle: {
          ready_for_integration_promotion: false,
          status: 'blocked',
          source_authority_denied: true,
          blockers: ['offline_integration_bundle_contract_not_ready'],
          missing_integration_checks: ['operator_report_bound'],
          child_contract_statuses: {
            replay_evaluation: 'ready',
            fault_recovery: 'blocked',
          },
        },
        promotion_gap_summary: {
          read_only: true,
          live_write: false,
          full_autonomous_ready: false,
          gap_bucket_counts: {
            autonomy_mode: 1,
            read_surface: 1,
            write_surface: 1,
            evidence_requirement: 15,
            proof: 0,
          },
          blocked_target_ids_by_bucket: {
            autonomy_mode: ['scene_preset_supervised_promotion'],
            read_surface: ['scene_preset_supervised_promotion'],
            write_surface: ['scene_preset_supervised_promotion'],
            evidence_requirement: [
              'bus_group_matrix_supervised_promotion',
              'mute_supervised_promotion',
            ],
          },
          evidence_requirement_ids_by_action_family: {
            bus_group_matrix: [
              'bus_group_assignment_cache_freshness',
              'bus_group_fader_readback',
              'matrix_send_readback',
              'main_bus_true_peak_headroom',
              'bus_group_physical_fixture',
              'hardware_readback_and_rollback_proof',
            ],
            mute: [
              'current_mute_cache_freshness',
              'mute_owner_intent_evidence',
              'signal_return_or_idle_window_evidence',
              'manual_mute_authority_review',
              'mute_physical_fixture',
              'hardware_readback_rollback_and_emergency_proof',
            ],
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
          current_live_write_surfaces: ['preamp', 'trim', 'fader'],
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
    supervised_proof_evidence: {
      type: 'supervised_proof_evidence_snapshot',
      available: true,
      report_path: 'artifacts/no_wing_release_check_current_cycle/latest/supervised_proof_artifact_report.json',
      safety: {
        read_only: true,
        live_write: false,
        generates_artifacts: false,
        live_widening_allowed: false,
        automatic_safety_update_allowed: false,
        full_autonomous_ready: false,
      },
      summary: {
        latest_report_status: 'ok',
        latest_report_age_sec: 120,
        latest_report_stale: false,
        stale_after_sec: 86400,
        artifact_count: 1,
        invalid_summary_count: 0,
        overlay_status: 'dry_run_only',
        overlay_row_count: 2,
        read_surface_proof_quality: {
          available: true,
          status: 'passed',
          snapshot_status: 'fixture_evidence_incomplete',
          surface_count: 10,
          surface_with_fixture_count: 10,
          declared_fixture_path_count: 16,
          declared_test_name_count: 43,
          physical_gap_surface_count: 10,
          summary_recomputed_from_rows: true,
          fixture_declaration_validation: true,
          contract_gap_consistency_enforced: true,
          read_only: true,
          live_write: false,
          full_autonomous_ready: false,
          physical_fixture_missing_surface_ids: [
            'raw_mixer_state_cache',
            'channel_identity_and_basic_controls',
            'channel_eq_dynamics_gate',
            'bus_group_main_dca_state',
            'matrix_state',
            'fx_slots_and_inserts',
            'routing_and_sends',
            'snapshots_scenes',
            'user_presets',
            'dlive_state',
          ],
          next_physical_readback_surface_id: 'raw_mixer_state_cache',
          source_next_physical_readback_surface_id: 'raw_mixer_state_cache',
          next_physical_readback_target_consistent: true,
          read_surface_physical_readback_capture_plan: {
            available: true,
            status: 'passed',
            plan_status: 'operator_capture_required',
            read_only: true,
            live_write: false,
            mutates_console_state: false,
            queries_console: false,
            executes_tests: false,
            operator_required: true,
            execution_allowed_by_plan: false,
            live_authority_granted: false,
            full_autonomous_ready: false,
            capture_step_count: 1,
            physical_gap_surface_count: 10,
            physical_fixture_missing_surface_ids: [
              'raw_mixer_state_cache',
              'channel_identity_and_basic_controls',
            ],
            next_physical_readback_surface_id: 'raw_mixer_state_cache',
            source_next_physical_readback_surface_id: 'raw_mixer_state_cache',
            next_physical_readback_action_id: 'remediate_raw_mixer_state_cache',
            source_next_physical_readback_action_id: 'remediate_raw_mixer_state_cache',
            target_consistency: true,
            required_artifact_contracts: [
              'physical_readback_fixture',
              'raw_mixer_state_physical_readback_contract',
            ],
            blocked_by: [
              'physical_readback_fixtures_incomplete',
              'operator_must_capture_and_review_physical_readback_artifact',
            ],
            step_surface_id: 'raw_mixer_state_cache',
            step_remediation_action_id: 'remediate_raw_mixer_state_cache',
            step_forbidden_actions: [
              'do not mark cached state as physical proof',
            ],
          },
          raw_mixer_state_physical_readback_contract: {
            available: true,
            status: 'passed',
            schema_id: 'raw_mixer_state_physical_readback.v1',
            contract_status: 'contract_ready_physical_fixture_missing',
            surface_id: 'raw_mixer_state_cache',
            remediation_action_id: 'remediate_raw_mixer_state_cache',
            artifact_required: true,
            physical_fixture_gap_remains: true,
            accepted_as_physical_fixture_candidate: false,
            read_only: true,
            live_write: false,
            live_authority_granted: false,
            full_autonomous_ready: false,
            target_consistency: true,
            next_physical_readback_surface_id: 'raw_mixer_state_cache',
            source_next_physical_readback_surface_id: 'raw_mixer_state_cache',
            next_physical_readback_action_id: 'remediate_raw_mixer_state_cache',
            source_next_physical_readback_action_id: 'remediate_raw_mixer_state_cache',
            allowed_sources: ['physical_readback', 'wing_physical_readback'],
            accepted_row_statuses: ['verified_from_physical_readback'],
            forbidden_mutation_evidence_fields: ['manual_write_events'],
            violations: [],
          },
          raw_mixer_state_physical_readback_artifact_intake: {
            available: true,
            status: 'passed',
            schema_id: 'raw_mixer_state_physical_readback.v1',
            intake_status: 'operator_artifact_missing',
            surface_id: 'raw_mixer_state_cache',
            remediation_action_id: 'remediate_raw_mixer_state_cache',
            required_contract_method_id: 'raw_mixer_state_physical_readback_contract',
            artifact_count: 0,
            valid_candidate_count: 0,
            invalid_candidate_count: 0,
            accepted_as_physical_fixture_candidate: false,
            operator_review_required: true,
            physical_fixture_gap_remains: true,
            fixture_closure_allowed_by_intake: false,
            live_authority_granted: false,
            full_autonomous_ready: false,
            target_consistency: true,
            artifact_search_roots: ['artifacts/read_surface_physical_readback/raw_mixer_state'],
            artifact_source_ids: [],
            blocked_by: ['operator_must_capture_raw_mixer_state_physical_readback_artifact'],
            intake_violations: [],
            violations: [],
          },
          channel_basic_controls_physical_readback_contract: {
            available: true,
            status: 'passed',
            schema_id: 'channel_basic_controls_physical_readback.v1',
            contract_status: 'contract_ready_physical_fixture_missing',
            surface_id: 'channel_identity_and_basic_controls',
            remediation_action_id: 'remediate_channel_identity_and_basic_controls',
            artifact_required: true,
            physical_fixture_gap_remains: true,
            accepted_as_physical_fixture_candidate: false,
            read_only: true,
            live_write: false,
            live_authority_granted: false,
            full_autonomous_ready: false,
            target_consistency: true,
            physical_readback_surface_id: 'channel_identity_and_basic_controls',
            physical_readback_action_id: 'remediate_channel_identity_and_basic_controls',
            fixture_physical_gap_declared: true,
            remediation_physical_gap_declared: true,
            promotion_families_requiring_surface: ['fader', 'gain', 'mute', 'pan_phase_delay'],
            required_control_fields: ['fader_db', 'trim_db', 'mute', 'pan'],
            required_coverage_flags: ['identity', 'basic_controls', 'cache_matches_physical_readback'],
            allowed_sources: ['physical_readback', 'wing_physical_readback'],
            accepted_channel_statuses: ['verified_from_physical_readback'],
            forbidden_mutation_evidence_fields: ['manual_write_events', 'mute_write_events'],
            violations: [],
          },
          channel_basic_controls_physical_readback_artifact_intake: {
            available: true,
            status: 'passed',
            schema_id: 'channel_basic_controls_physical_readback.v1',
            intake_status: 'operator_artifact_missing',
            surface_id: 'channel_identity_and_basic_controls',
            remediation_action_id: 'remediate_channel_identity_and_basic_controls',
            required_contract_method_id: 'channel_basic_controls_physical_readback_contract',
            artifact_count: 0,
            valid_candidate_count: 0,
            invalid_candidate_count: 0,
            valid_channel_count: 0,
            accepted_as_physical_fixture_candidate: false,
            operator_review_required: true,
            physical_fixture_gap_remains: true,
            fixture_closure_allowed_by_intake: false,
            live_authority_granted: false,
            full_autonomous_ready: false,
            target_consistency: true,
            promotion_families_requiring_surface: ['fader', 'gain', 'mute', 'pan_phase_delay'],
            artifact_search_roots: ['artifacts/read_surface_physical_readback/channel_basic_controls'],
            artifact_source_ids: [],
            blocked_by: ['operator_must_capture_channel_basic_controls_physical_readback_artifact'],
            intake_violations: [],
            violations: [],
          },
          channel_processing_physical_readback_contract: {
            available: true,
            status: 'passed',
            schema_id: 'channel_processing_physical_readback.v1',
            contract_status: 'contract_ready_physical_fixture_missing',
            surface_id: 'channel_eq_dynamics_gate',
            remediation_action_id: 'remediate_channel_eq_dynamics_gate',
            artifact_required: true,
            physical_fixture_gap_remains: true,
            accepted_as_physical_fixture_candidate: false,
            read_only: true,
            live_write: false,
            live_authority_granted: false,
            full_autonomous_ready: false,
            target_consistency: true,
            physical_readback_surface_id: 'channel_eq_dynamics_gate',
            physical_readback_action_id: 'remediate_channel_eq_dynamics_gate',
            fixture_physical_gap_declared: true,
            remediation_physical_gap_declared: true,
            promotion_families_requiring_surface: ['eq', 'compressor', 'gate', 'deessing'],
            required_eq_band_fields: ['band', 'frequency_hz', 'gain_db', 'q'],
            required_compressor_fields: ['enabled', 'model', 'gain_reduction_db'],
            required_gate_fields: ['enabled', 'model', 'range_db'],
            required_coverage_flags: ['hpf', 'eq', 'compressor', 'gate', 'cache_matches_physical_readback'],
            allowed_sources: ['physical_readback', 'wing_physical_readback'],
            accepted_channel_statuses: ['verified_from_physical_readback'],
            forbidden_mutation_evidence_fields: ['manual_write_events', 'eq_write_events'],
            minimum_eq_band_count: 4,
            violations: [],
          },
          channel_processing_physical_readback_artifact_intake: {
            available: true,
            status: 'passed',
            schema_id: 'channel_processing_physical_readback.v1',
            intake_status: 'operator_artifact_missing',
            surface_id: 'channel_eq_dynamics_gate',
            remediation_action_id: 'remediate_channel_eq_dynamics_gate',
            required_contract_method_id: 'channel_processing_physical_readback_contract',
            artifact_count: 0,
            valid_candidate_count: 0,
            invalid_candidate_count: 0,
            valid_channel_count: 0,
            accepted_as_physical_fixture_candidate: false,
            operator_review_required: true,
            physical_fixture_gap_remains: true,
            fixture_closure_allowed_by_intake: false,
            live_authority_granted: false,
            full_autonomous_ready: false,
            target_consistency: true,
            promotion_families_requiring_surface: [
              'eq',
              'compressor',
              'gate',
              'deessing',
              'frequency_separation',
              'cross_channel_unmasking',
            ],
            artifact_search_roots: ['artifacts/read_surface_physical_readback/channel_processing'],
            artifact_source_ids: [],
            blocked_by: ['operator_must_capture_channel_processing_physical_readback_artifact'],
            intake_violations: [],
            violations: [],
          },
          bus_group_main_dca_physical_readback_contract: {
            available: true,
            status: 'passed',
            schema_id: 'bus_group_main_dca_physical_readback.v1',
            contract_status: 'contract_ready_physical_fixture_missing',
            surface_id: 'bus_group_main_dca_state',
            remediation_action_id: 'remediate_bus_group_main_dca_state',
            artifact_required: true,
            physical_fixture_gap_remains: true,
            accepted_as_physical_fixture_candidate: false,
            read_only: true,
            live_write: false,
            live_authority_granted: false,
            full_autonomous_ready: false,
            target_consistency: true,
            physical_readback_surface_id: 'bus_group_main_dca_state',
            physical_readback_action_id: 'remediate_bus_group_main_dca_state',
            fixture_physical_gap_declared: true,
            remediation_physical_gap_declared: true,
            promotion_families_requiring_surface: ['bus_group_matrix', 'dca', 'master_bus'],
            required_bus_control_fields: ['fader_db', 'mute', 'pan', 'width', 'mono'],
            required_main_control_fields: ['fader_db', 'mute', 'pan'],
            required_dca_control_fields: ['fader_db', 'mute', 'monitor'],
            required_bus_assignment_fields: ['dca_assignments', 'main_send_on', 'main_send_level_db', 'main_send_pre'],
            required_dca_assignment_fields: ['member_channels', 'member_buses'],
            required_metering_fields: ['lufs', 'true_peak_dbtp', 'headroom_db'],
            allowed_sources: ['physical_readback', 'wing_physical_readback'],
            accepted_surface_statuses: ['verified_from_physical_readback'],
            forbidden_mutation_evidence_fields: ['manual_write_events', 'dca_write_events'],
            violations: [],
          },
          bus_group_main_dca_physical_readback_artifact_intake: {
            available: true,
            status: 'passed',
            schema_id: 'bus_group_main_dca_physical_readback.v1',
            intake_status: 'operator_artifact_missing',
            surface_id: 'bus_group_main_dca_state',
            remediation_action_id: 'remediate_bus_group_main_dca_state',
            required_contract_method_id: 'bus_group_main_dca_physical_readback_contract',
            artifact_count: 0,
            valid_candidate_count: 0,
            invalid_candidate_count: 0,
            valid_bus_count: 0,
            valid_main_count: 0,
            valid_dca_count: 0,
            accepted_as_physical_fixture_candidate: false,
            operator_review_required: true,
            physical_fixture_gap_remains: true,
            fixture_closure_allowed_by_intake: false,
            live_authority_granted: false,
            full_autonomous_ready: false,
            target_consistency: true,
            promotion_families_requiring_surface: [
              'bus_group_matrix',
              'dca',
              'master_bus',
              'mix_balance',
              'performance_adaptation',
            ],
            artifact_search_roots: ['artifacts/read_surface_physical_readback/bus_group_main_dca'],
            artifact_source_ids: [],
            blocked_by: ['operator_must_capture_bus_group_main_dca_physical_readback_artifact'],
            intake_violations: [],
            violations: [],
          },
          matrix_state_physical_readback_contract: {
            available: true,
            status: 'passed',
            schema_id: 'matrix_state_physical_readback.v1',
            contract_status: 'contract_ready_physical_fixture_missing',
            surface_id: 'matrix_state',
            remediation_action_id: 'remediate_matrix_state',
            artifact_required: true,
            physical_fixture_gap_remains: true,
            accepted_as_physical_fixture_candidate: false,
            read_only: true,
            live_write: false,
            live_authority_granted: false,
            full_autonomous_ready: false,
            target_consistency: true,
            physical_readback_surface_id: 'matrix_state',
            physical_readback_action_id: 'remediate_matrix_state',
            fixture_physical_gap_declared: true,
            remediation_physical_gap_declared: true,
            promotion_families_requiring_surface: ['bus_group_matrix', 'fx_send_insert', 'routing'],
            required_control_fields: ['fader_db', 'mute', 'pan'],
            required_eq_band_fields: ['band', 'frequency_hz', 'gain_db', 'q'],
            required_dynamics_fields: ['enabled', 'model', 'gain_reduction_db'],
            required_insert_fields: ['pre_insert_on', 'post_insert_weight'],
            required_metering_fields: ['lufs', 'true_peak_dbtp', 'headroom_db'],
            allowed_sources: ['physical_readback', 'wing_physical_readback'],
            accepted_matrix_statuses: ['verified_from_physical_readback'],
            forbidden_mutation_evidence_fields: ['manual_write_events', 'matrix_write_events'],
            minimum_matrix_count: 1,
            minimum_eq_band_count: 8,
            violations: [],
          },
          matrix_state_physical_readback_artifact_intake: {
            available: true,
            status: 'passed',
            schema_id: 'matrix_state_physical_readback.v1',
            intake_status: 'operator_artifact_missing',
            surface_id: 'matrix_state',
            remediation_action_id: 'remediate_matrix_state',
            required_contract_method_id: 'matrix_state_physical_readback_contract',
            artifact_count: 0,
            valid_candidate_count: 0,
            invalid_candidate_count: 0,
            valid_matrix_count: 0,
            accepted_as_physical_fixture_candidate: false,
            operator_review_required: true,
            physical_fixture_gap_remains: true,
            fixture_closure_allowed_by_intake: false,
            live_authority_granted: false,
            full_autonomous_ready: false,
            target_consistency: true,
            promotion_families_requiring_surface: [
              'bus_group_matrix',
              'fx_send_insert',
              'master_bus',
              'mix_balance',
              'routing',
              'performance_adaptation',
            ],
            artifact_search_roots: ['artifacts/read_surface_physical_readback/matrix_state'],
            artifact_source_ids: [],
            blocked_by: ['operator_must_capture_matrix_state_physical_readback_artifact'],
            intake_violations: [],
            violations: [],
          },
          fx_slots_inserts_physical_readback_contract: {
            available: true,
            status: 'passed',
            schema_id: 'fx_slots_inserts_physical_readback.v1',
            contract_status: 'contract_ready_physical_fixture_missing',
            surface_id: 'fx_slots_and_inserts',
            remediation_action_id: 'remediate_fx_slots_and_inserts',
            artifact_required: true,
            physical_fixture_gap_remains: true,
            accepted_as_physical_fixture_candidate: false,
            read_only: true,
            live_write: false,
            live_authority_granted: false,
            full_autonomous_ready: false,
            target_consistency: true,
            physical_readback_surface_id: 'fx_slots_and_inserts',
            physical_readback_action_id: 'remediate_fx_slots_and_inserts',
            fixture_physical_gap_declared: true,
            remediation_physical_gap_declared: true,
            promotion_families_requiring_surface: ['fx_send_insert', 'routing', 'mix_balance'],
            required_fx_slot_fields: ['slot', 'slot_number', 'parameter_schema'],
            required_parameter_fields: ['parameter_index', 'value', 'safe_range'],
            required_parameter_schema_fields: ['status', 'physical_model_schema_verified'],
            required_insert_fields: ['target_type', 'position', 'linked_fx_slot'],
            required_insert_coverage_flags: ['assignment', 'status', 'scoped_diff_marker'],
            required_false_safety_flags: [
              'live_write',
              'mutates_console_state',
              'live_widening_allowed',
              'full_autonomous_ready',
            ],
            allowed_sources: ['physical_readback', 'wing_physical_readback'],
            accepted_row_statuses: ['verified_from_physical_readback'],
            forbidden_mutation_evidence_fields: ['manual_write_events', 'fx_write_events'],
            minimum_fx_slot_count: 1,
            minimum_fx_parameter_count: 4,
            minimum_insert_count: 1,
            violations: [],
          },
          fx_slots_inserts_physical_readback_artifact_intake: {
            available: true,
            status: 'passed',
            schema_id: 'fx_slots_inserts_physical_readback.v1',
            intake_status: 'operator_artifact_missing',
            surface_id: 'fx_slots_and_inserts',
            remediation_action_id: 'remediate_fx_slots_and_inserts',
            required_contract_method_id: 'fx_slots_inserts_physical_readback_contract',
            artifact_count: 0,
            valid_candidate_count: 0,
            invalid_candidate_count: 0,
            valid_fx_slot_count: 0,
            valid_insert_count: 0,
            accepted_as_physical_fixture_candidate: false,
            operator_review_required: true,
            physical_fixture_gap_remains: true,
            fixture_closure_allowed_by_intake: false,
            live_authority_granted: false,
            full_autonomous_ready: false,
            target_consistency: true,
            promotion_families_requiring_surface: [
              'fx_send_insert',
              'routing',
              'mix_balance',
              'performance_adaptation',
            ],
            artifact_search_roots: ['artifacts/read_surface_physical_readback/fx_slots_and_inserts'],
            artifact_source_ids: [],
            blocked_by: ['operator_must_capture_fx_slots_inserts_physical_readback_artifact'],
            intake_violations: [],
            violations: [],
          },
          routing_sends_physical_readback_contract: {
            available: true,
            status: 'passed',
            schema_id: 'routing_sends_physical_readback.v1',
            contract_status: 'contract_ready_physical_fixture_missing',
            surface_id: 'routing_and_sends',
            remediation_action_id: 'remediate_routing_and_sends',
            artifact_required: true,
            physical_fixture_gap_remains: true,
            accepted_as_physical_fixture_candidate: false,
            read_only: true,
            live_write: false,
            live_authority_granted: false,
            full_autonomous_ready: false,
            target_consistency: true,
            physical_readback_surface_id: 'routing_and_sends',
            physical_readback_action_id: 'remediate_routing_and_sends',
            fixture_physical_gap_declared: true,
            remediation_physical_gap_declared: true,
            promotion_families_requiring_surface: [
              'routing',
              'fx_send_insert',
              'bus_group_matrix',
              'scene_preset',
              'mix_balance',
              'performance_adaptation',
            ],
            required_static_route_fields: ['start', 'end', 'role', 'wing_routing_hint'],
            required_bus_send_fields: ['channel', 'send', 'level_db', 'pre_on'],
            required_main_send_fields: ['channel', 'main', 'level_db', 'pre'],
            required_matrix_send_fields: ['channel', 'send', 'target_matrix'],
            required_output_route_fields: ['output_group', 'output_number', 'source_channel'],
            required_send_coverage_flags: ['assignment', 'level', 'scoped_diff_marker'],
            required_output_route_coverage_flags: ['assignment', 'source', 'scoped_diff_marker'],
            required_false_safety_flags: [
              'live_write',
              'mutates_console_state',
              'live_widening_allowed',
              'full_autonomous_ready',
            ],
            allowed_sources: ['physical_readback', 'wing_physical_readback', 'dlive_physical_readback'],
            accepted_row_statuses: ['verified_from_physical_readback'],
            forbidden_mutation_evidence_fields: ['manual_write_events', 'routing_write_events'],
            minimum_static_dante_route_count: 1,
            minimum_channel_bus_send_count: 1,
            minimum_channel_main_send_count: 1,
            minimum_channel_matrix_send_count: 1,
            minimum_output_route_count: 1,
            violations: [],
          },
          routing_sends_physical_readback_artifact_intake: {
            available: true,
            status: 'passed',
            schema_id: 'routing_sends_physical_readback.v1',
            intake_status: 'operator_artifact_missing',
            surface_id: 'routing_and_sends',
            remediation_action_id: 'remediate_routing_and_sends',
            required_contract_method_id: 'routing_sends_physical_readback_contract',
            artifact_count: 0,
            valid_candidate_count: 0,
            invalid_candidate_count: 0,
            valid_static_dante_route_count: 0,
            valid_channel_bus_send_count: 0,
            valid_channel_main_send_count: 0,
            valid_channel_matrix_send_count: 0,
            valid_output_route_count: 0,
            accepted_as_physical_fixture_candidate: false,
            operator_review_required: true,
            physical_fixture_gap_remains: true,
            fixture_closure_allowed_by_intake: false,
            live_authority_granted: false,
            full_autonomous_ready: false,
            target_consistency: true,
            promotion_families_requiring_surface: [
              'routing',
              'fx_send_insert',
              'bus_group_matrix',
              'scene_preset',
              'mix_balance',
              'performance_adaptation',
            ],
            artifact_search_roots: ['artifacts/read_surface_physical_readback/routing_and_sends'],
            artifact_source_ids: [],
            blocked_by: ['operator_must_capture_routing_sends_physical_readback_artifact'],
            intake_violations: [],
            violations: [],
          },
          snapshots_scenes_physical_readback_contract: {
            available: true,
            status: 'passed',
            schema_id: 'snapshots_scenes_physical_readback.v1',
            contract_status: 'contract_ready_physical_fixture_missing',
            surface_id: 'snapshots_scenes',
            remediation_action_id: 'remediate_snapshots_scenes',
            physical_readback_surface_id: 'snapshots_scenes',
            physical_readback_action_id: 'remediate_snapshots_scenes',
            artifact_required: true,
            physical_fixture_gap_remains: true,
            accepted_as_physical_fixture_candidate: false,
            read_only: true,
            live_write: false,
            live_authority_granted: false,
            full_autonomous_ready: false,
            scene_load_allowed: false,
            scene_save_allowed: false,
            target_consistency: true,
            fixture_physical_gap_declared: true,
            remediation_physical_gap_declared: true,
            promotion_families_requiring_surface: ['scene_preset', 'routing', 'mix_balance'],
            required_active_scene_fields: ['active_snapshot_index', 'active_snapshot_name', 'current_show'],
            required_snapshot_fields: [
              'snapshot_index',
              'snapshot_name',
              'scene_content_fingerprint',
              'rollback_reference',
            ],
            required_coverage_flags: ['catalog_identity', 'active_scene', 'rollback_reference'],
            required_false_safety_flags: [
              'live_write',
              'mutates_console_state',
              'scene_load_allowed',
              'scene_save_allowed',
            ],
            allowed_sources: ['physical_readback', 'wing_physical_readback'],
            accepted_row_statuses: ['verified_from_physical_readback'],
            forbidden_mutation_evidence_fields: ['scene_recall_events'],
            minimum_snapshot_count: 1,
            violations: [],
          },
          snapshots_scenes_physical_readback_artifact_intake: {
            available: true,
            status: 'passed',
            intake_status: 'operator_artifact_missing',
            schema_id: 'snapshots_scenes_physical_readback.v1',
            surface_id: 'snapshots_scenes',
            remediation_action_id: 'remediate_snapshots_scenes',
            required_contract_method_id: 'snapshots_scenes_physical_readback_contract',
            artifact_count: 0,
            valid_candidate_count: 0,
            invalid_candidate_count: 0,
            valid_snapshot_count: 0,
            accepted_as_physical_fixture_candidate: false,
            operator_review_required: true,
            physical_fixture_gap_remains: true,
            fixture_closure_allowed_by_intake: false,
            scene_load_allowed: false,
            scene_save_allowed: false,
            live_authority_granted: false,
            full_autonomous_ready: false,
            target_consistency: true,
            promotion_families_requiring_surface: [
              'scene_preset',
              'routing',
              'mix_balance',
              'performance_adaptation',
            ],
            artifact_search_roots: ['artifacts/read_surface_physical_readback/snapshots_scenes'],
            artifact_source_ids: [],
            blocked_by: ['operator_must_capture_snapshots_scenes_physical_readback_artifact'],
            intake_violations: [],
            violations: [],
          },
          user_presets_physical_readback_contract: {
            available: true,
            status: 'passed',
            schema_id: 'user_presets_physical_readback.v1',
            contract_status: 'contract_ready_physical_fixture_missing',
            surface_id: 'user_presets',
            remediation_action_id: 'remediate_user_presets',
            physical_readback_surface_id: 'user_presets',
            physical_readback_action_id: 'remediate_user_presets',
            artifact_required: true,
            physical_fixture_gap_remains: true,
            accepted_as_physical_fixture_candidate: false,
            read_only: true,
            live_write: false,
            live_authority_granted: false,
            full_autonomous_ready: false,
            preset_recall_allowed: false,
            preset_store_allowed: false,
            target_consistency: true,
            fixture_physical_gap_declared: true,
            remediation_physical_gap_declared: true,
            promotion_families_requiring_surface: ['scene_preset', 'routing', 'mix_balance'],
            required_preset_fields: [
              'preset_index',
              'preset_name',
              'preset_scope',
              'target_type',
              'target_label',
              'preset_content_fingerprint',
              'rollback_reference',
            ],
            required_coverage_flags: ['catalog_identity', 'scope', 'target', 'rollback_reference'],
            required_false_safety_flags: [
              'live_write',
              'mutates_console_state',
              'preset_recall_allowed',
              'preset_store_allowed',
            ],
            allowed_sources: ['physical_readback', 'wing_physical_readback'],
            accepted_row_statuses: ['verified_from_physical_readback'],
            forbidden_mutation_evidence_fields: ['preset_recall_events', 'preset_store_events'],
            minimum_preset_count: 1,
            violations: [],
          },
          user_presets_physical_readback_artifact_intake: {
            available: true,
            status: 'passed',
            intake_status: 'operator_artifact_missing',
            schema_id: 'user_presets_physical_readback.v1',
            surface_id: 'user_presets',
            remediation_action_id: 'remediate_user_presets',
            required_contract_method_id: 'user_presets_physical_readback_contract',
            artifact_count: 0,
            valid_candidate_count: 0,
            invalid_candidate_count: 0,
            valid_preset_count: 0,
            accepted_as_physical_fixture_candidate: false,
            operator_review_required: true,
            physical_fixture_gap_remains: true,
            fixture_closure_allowed_by_intake: false,
            preset_recall_allowed: false,
            preset_store_allowed: false,
            live_authority_granted: false,
            full_autonomous_ready: false,
            target_consistency: true,
            promotion_families_requiring_surface: [
              'scene_preset',
              'routing',
              'mix_balance',
              'performance_adaptation',
            ],
            artifact_search_roots: ['artifacts/read_surface_physical_readback/user_presets'],
            artifact_source_ids: [],
            blocked_by: ['operator_must_capture_user_presets_physical_readback_artifact'],
            intake_violations: [],
            violations: [],
          },
          dlive_state_physical_readback_contract: {
            available: true,
            status: 'passed',
            schema_id: 'dlive_state_physical_readback.v1',
            contract_status: 'contract_ready_physical_fixture_missing',
            surface_id: 'dlive_state',
            remediation_action_id: 'remediate_dlive_state',
            physical_readback_surface_id: 'dlive_state',
            physical_readback_action_id: 'remediate_dlive_state',
            artifact_required: true,
            physical_fixture_gap_remains: true,
            accepted_as_physical_fixture_candidate: false,
            read_only: true,
            live_write: false,
            live_authority_granted: false,
            full_autonomous_ready: false,
            midi_send_allowed: false,
            tcp_send_allowed: false,
            dlive_write_allowed: false,
            dlive_apply_allowed: false,
            target_consistency: true,
            fixture_physical_gap_declared: true,
            remediation_physical_gap_declared: true,
            promotion_families_requiring_surface: [
              'fader',
              'gain',
              'mute',
              'routing',
              'performance_adaptation',
            ],
            required_channel_fields: [
              'channel',
              'cached_channel_name',
              'fader_raw_14bit',
            ],
            required_midi_state_fields: [
              'midi_base_channel',
              'raw_state_key_count',
            ],
            required_parity_fields: [
              'wing_surface_equivalent',
              'mapping_status',
            ],
            required_coverage_flags: ['channel_identity', 'wing_parity_mapping'],
            required_false_safety_flags: [
              'live_write',
              'midi_send_allowed',
              'tcp_send_allowed',
              'dlive_write_allowed',
            ],
            allowed_sources: ['physical_readback', 'dlive_physical_readback'],
            accepted_row_statuses: ['verified_from_physical_readback'],
            forbidden_mutation_evidence_fields: ['midi_send_events'],
            minimum_channel_count: 1,
            violations: [],
          },
          dlive_state_physical_readback_artifact_intake: {
            available: true,
            status: 'passed',
            intake_status: 'operator_artifact_missing',
            schema_id: 'dlive_state_physical_readback.v1',
            surface_id: 'dlive_state',
            remediation_action_id: 'remediate_dlive_state',
            required_contract_method_id: 'dlive_state_physical_readback_contract',
            artifact_count: 0,
            valid_candidate_count: 0,
            invalid_candidate_count: 0,
            valid_channel_count: 0,
            accepted_as_physical_fixture_candidate: false,
            operator_review_required: true,
            physical_fixture_gap_remains: true,
            fixture_closure_allowed_by_intake: false,
            midi_send_allowed: false,
            tcp_send_allowed: false,
            dlive_write_allowed: false,
            dlive_apply_allowed: false,
            live_authority_granted: false,
            full_autonomous_ready: false,
            target_consistency: true,
            promotion_families_requiring_surface: [
              'fader',
              'gain',
              'mute',
              'bus_group_matrix',
              'routing',
              'scene_preset',
              'mix_balance',
              'performance_adaptation',
            ],
            artifact_search_roots: ['artifacts/read_surface_physical_readback/dlive_state'],
            artifact_source_ids: [],
            blocked_by: ['operator_must_capture_dlive_state_physical_readback_artifact'],
            intake_violations: [],
            violations: [],
          },
          read_surface_physical_readback_artifact_review_queue: {
            available: true,
            status: 'passed',
            queue_status: 'operator_artifact_capture_required',
            read_only: true,
            live_write: false,
            mutates_console_state: false,
            queries_console: false,
            executes_tests: false,
            operator_required: true,
            execution_allowed_by_queue: false,
            fixture_closure_allowed_by_queue: false,
            live_widening_allowed: false,
            live_authority_granted: false,
            full_autonomous_ready: false,
            intake_surface_count: 10,
            available_intake_count: 10,
            artifact_count: 0,
            valid_candidate_count: 0,
            invalid_candidate_count: 0,
            review_candidate_count: 0,
            missing_artifact_surface_count: 10,
            unsafe_authority_claim_count: 0,
            target_inconsistent_surface_count: 0,
            review_candidates_available: false,
            operator_review_required: true,
            physical_fixture_gap_remains: true,
            review_candidate_surface_ids: [],
            invalid_surface_ids: [],
            missing_artifact_surface_ids: [
              'raw_mixer_state_cache',
              'channel_identity_and_basic_controls',
              'dlive_state',
            ],
            unsafe_authority_surface_ids: [],
            target_inconsistent_surface_ids: [],
            queue_item_method_ids: [
              'raw_mixer_state_physical_readback_artifact_intake',
              'channel_basic_controls_physical_readback_artifact_intake',
              'dlive_state_physical_readback_artifact_intake',
            ],
            blocked_by: ['operator_must_capture_missing_physical_readback_artifacts'],
            violations: [],
          },
          read_surface_physical_readback_fixture_closure_plan: {
            available: true,
            status: 'passed',
            plan_status: 'fixture_closure_not_ready',
            read_only: true,
            live_write: false,
            mutates_console_state: false,
            queries_console: false,
            executes_tests: false,
            operator_required: true,
            execution_allowed_by_plan: false,
            fixture_update_allowed_by_plan: false,
            fixture_closure_allowed_by_plan: false,
            live_widening_allowed: false,
            live_authority_granted: false,
            full_autonomous_ready: false,
            review_candidate_count: 0,
            acceptance_record_count: 0,
            accepted_record_count: 0,
            rejected_record_count: 0,
            invalid_record_count: 0,
            unreviewed_candidate_count: 0,
            closure_candidate_count: 0,
            manual_fixture_closure_candidate_available: false,
            manual_fixture_closure_required: true,
            operator_review_required: true,
            physical_fixture_gap_remains: true,
            closure_candidate_surface_ids: [],
            rejected_surface_ids: [],
            invalid_record_ids: [],
            unreviewed_candidate_surface_ids: [],
            blocked_by: ['accepted_physical_readback_candidate_required_before_fixture_closure'],
            violations: [],
          },
        },
      },
    },
    read_surface_remediation_plan: {
      type: 'read_surface_remediation_plan',
      status: 'read_remediation_required',
      safety: {
        read_only: true,
        live_write: false,
        mutates_console_state: false,
        live_widening_allowed: false,
        full_autonomous_ready: false,
      },
      summary: {
        surface_count: 10,
        incomplete_surface_count: 10,
        action_count: 10,
        physical_fixture_missing_surface_ids: [
          'raw_mixer_state_cache',
          'channel_identity_and_basic_controls',
        ],
        next_physical_readback_surface_id: 'raw_mixer_state_cache',
        next_physical_readback_action_id: 'remediate_raw_mixer_state_cache',
        global_blockers: [
          'full_console_read_coverage_incomplete',
          'physical_readback_fixtures_incomplete',
          'read_surface_remediation_does_not_contact_mixer',
        ],
        next_global_step: 'Complete read DTOs and physical fixtures before broad autonomous console control.',
      },
    },
    autonomy_evidence_overlay: {
      rows: [
        {
          id: 'fader_autonomy_evidence_overlay',
          target_id: 'fader_supervised_promotion',
          value_type: 'fader',
          status: 'no_current_evidence',
          physical_candidate_count: 0,
          dry_run_ready_count: 0,
          blocked_artifact_count: 0,
          remediation_action_count: 0,
          physical_candidate_proof_ids: [],
          dry_run_ready_proof_ids: [],
          blocked_proof_ids: [],
          remediation_action_ids: [],
          blockers: ['proof_artifact_missing'],
          next_required_actions: ['Run one physical supervised fader proof.'],
        },
        {
          id: 'gain_autonomy_evidence_overlay',
          target_id: 'gain_trim_supervised_promotion',
          value_type: 'gain',
          status: 'dry_run_only',
          physical_candidate_count: 0,
          dry_run_ready_count: 1,
          blocked_artifact_count: 0,
          remediation_action_count: 0,
          physical_candidate_proof_ids: [],
          dry_run_ready_proof_ids: ['gain-proof-dry-run'],
          blocked_proof_ids: [],
          remediation_action_ids: [],
          blockers: ['physical_proof_required_for_promotion'],
          next_required_actions: ['Run one small supervised gain proof on WING.'],
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
      'supervised_proof_evidence',
      'read_surface_physical_readback_capture_plan',
      'raw_mixer_state_physical_readback_contract',
      'raw_mixer_state_physical_readback_artifact_intake',
      'channel_basic_controls_physical_readback_contract',
      'channel_basic_controls_physical_readback_artifact_intake',
      'channel_processing_physical_readback_contract',
      'channel_processing_physical_readback_artifact_intake',
      'bus_group_main_dca_physical_readback_contract',
      'bus_group_main_dca_physical_readback_artifact_intake',
      'matrix_state_physical_readback_contract',
      'matrix_state_physical_readback_artifact_intake',
      'fx_slots_inserts_physical_readback_contract',
      'fx_slots_inserts_physical_readback_artifact_intake',
      'routing_sends_physical_readback_contract',
      'routing_sends_physical_readback_artifact_intake',
      'snapshots_scenes_physical_readback_contract',
      'snapshots_scenes_physical_readback_artifact_intake',
      'user_presets_physical_readback_contract',
      'user_presets_physical_readback_artifact_intake',
      'dlive_state_physical_readback_contract',
      'dlive_state_physical_readback_artifact_intake',
      'read_surface_physical_readback_artifact_review_queue',
      'read_surface_physical_readback_fixture_closure_plan',
      'read_surface_remediation_plan',
      'autonomy_evidence:fader_supervised_promotion',
      'autonomy_evidence:gain_trim_supervised_promotion',
      'promotion_gap:autonomy_mode',
      'promotion_gap:read_surface',
      'promotion_gap:write_surface',
      'promotion_gap:evidence_requirement',
      'autonomy_promotion_summary',
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
  assert.equal(analysisMode.module, 'no_current_write_authority');
  assert.equal(analysisMode.tone, 'green');
  assert.match(analysisMode.blockedReason, /mode_has_no_live_apply_authority/);
  assert.match(analysisMode.nextStep, /No live-write authority for this mode/);

  const supervisedMode = rows.find(row => row.moduleId === 'mode_gate:supervised');
  assert.equal(supervisedMode.label, 'Supervised AutoMix Mode');
  assert.equal(supervisedMode.status, 'limited_available');
  assert.equal(supervisedMode.canWriteLive, false);
  assert.deepEqual(supervisedMode.liveWriteKinds, []);
  assert.equal(supervisedMode.tone, 'yellow');
  assert.match(supervisedMode.module, /policy:fader,gain/);
  assert.match(supervisedMode.module, /surfaces:preamp,trim,fader/);
  assert.match(supervisedMode.nextStep, /Current policy live-write kinds: fader, gain/);
  assert.equal(supervisedMode.pipelineProfiles.includes('current_write_surface:preamp'), true);
  assert.equal(supervisedMode.pipelineProfiles.includes('current_write_surface:trim'), true);
  assert.equal(supervisedMode.pipelineProfiles.includes('current_write_surface:fader'), true);

  const autoMode = rows.find(row => row.moduleId === 'mode_gate:auto');
  assert.equal(autoMode.label, 'Full Autonomous Mixing Mode');
  assert.equal(autoMode.status, 'blocked');
  assert.equal(autoMode.canWriteLive, false);
  assert.deepEqual(autoMode.liveWriteKinds, []);
  assert.equal(autoMode.module, 'no_current_write_authority');
  assert.equal(autoMode.tone, 'red');
  assert.match(autoMode.blockedReason, /full_autonomous_mode_remains_blocked/);
  assert.match(autoMode.nextStep, /No live-write authority for this mode/);
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

  const autonomySummary = rows.find(row => row.moduleId === 'autonomy_promotion_summary');
  assert.equal(autonomySummary.domain, 'autonomy_promotion_summary');
  assert.equal(autonomySummary.canWriteLive, false);
  assert.deepEqual(autonomySummary.liveWriteKinds, []);
  assert.equal(autonomySummary.status, 'blocked');
  assert.equal(autonomySummary.entrypoint, 'main_pipeline_preview.summary');
  assert.equal(autonomySummary.module, 'main_pipeline_preview.summary');
  assert.equal(autonomySummary.tone, 'red');
  assert.deepEqual(autonomySummary.pipelineProfiles, [
    'promotion_full_autonomous_ready:false',
    'mode_transition_full_autonomous_ready:true',
    'promotion_supervised_live_allowed_target_count:2',
    'promotion_blocked_target_count:4',
    'mode_transition_gate_mode_count:3',
    'mode_transition_gate_blocked_mode_count:1',
    'mode_transition_gate_blocked_mode:analyze',
    'mode_transition_gate_live_apply_mode:supervised',
    'supervised_current_live_write_kind:fader',
    'supervised_current_live_write_surface:preamp',
    'supervised_current_live_write_surface:trim',
    'supervised_current_live_write_surface:fader',
    'integration_bundle_ready:false',
    'integration_bundle_status:blocked',
    'integration_bundle_source_authority_denied:true',
    'integration_bundle_blocker:offline_integration_bundle_contract_not_ready',
    'integration_bundle_missing_check:operator_report_bound',
    'integration_bundle_child_status:replay_evaluation:ready',
    'integration_bundle_child_status:fault_recovery:blocked',
  ]);
  assert.equal(
    autonomySummary.blockedReason,
    'Mode transition full autonomous ready: true; Promotion full autonomous ready: false; Offline integration bundle ready: false; Integration bundle blockers: offline_integration_bundle_contract_not_ready',
  );
  assert.equal(
    autonomySummary.nextStep,
    'supervised_live_allowed_target_count:2 / blocked_target_count:4 / blocked_transition_mode:analyze / integration_bundle_status:blocked / integration_bundle_blocker:offline_integration_bundle_contract_not_ready',
  );

  const proofEvidence = rows.find(row => row.moduleId === 'supervised_proof_evidence');
  assert.equal(proofEvidence.domain, 'supervised_proof_evidence');
  assert.equal(proofEvidence.canWriteLive, false);
  assert.deepEqual(proofEvidence.liveWriteKinds, []);
  assert.equal(proofEvidence.status, 'artifact_available');
  assert.equal(proofEvidence.entrypoint, 'supervised_proof_artifact_report');
  assert.match(proofEvidence.module, /supervised_proof_artifact_report\.json/);
  assert.equal(proofEvidence.tone, 'blue');
  assert.deepEqual(proofEvidence.pipelineProfiles, [
    'available:true',
    'overlay:dry_run_only',
    'artifacts:1',
    'invalid:0',
    'stale:false',
    'read_surface_quality:true',
    'read_surface_gaps:10',
    'read_surface_gap_ids:raw_mixer_state_cache|channel_identity_and_basic_controls|channel_eq_dynamics_gate|bus_group_main_dca_state|matrix_state|fx_slots_and_inserts|routing_and_sends|snapshots_scenes|user_presets|dlive_state',
    'next_readback_target:raw_mixer_state_cache',
    'source_next_readback_target:raw_mixer_state_cache',
    'next_readback_gate:true',
    'capture_plan_available:true',
    'capture_plan_status:operator_capture_required',
    'capture_plan_steps:1',
    'capture_plan_next_surface:raw_mixer_state_cache',
    'capture_plan_execution_allowed:false',
    'capture_plan_target_gate:true',
    'fixture_tests:43',
    'contract_gap:true',
    'raw_contract_available:true',
    'raw_contract_status:contract_ready_physical_fixture_missing',
    'raw_contract_schema:raw_mixer_state_physical_readback.v1',
    'raw_contract_gap:true',
    'raw_contract_live_authority:false',
    'raw_contract_target_gate:true',
    'raw_intake_available:true',
    'raw_intake_status:operator_artifact_missing',
    'raw_intake_artifacts:0',
    'raw_intake_valid:0',
    'raw_intake_invalid:0',
    'raw_intake_candidate:false',
    'raw_intake_gap:true',
    'raw_intake_closure_allowed:false',
    'raw_intake_live_authority:false',
    'raw_intake_target_gate:true',
    'channel_contract_available:true',
    'channel_contract_status:contract_ready_physical_fixture_missing',
    'channel_contract_schema:channel_basic_controls_physical_readback.v1',
    'channel_contract_gap:true',
    'channel_contract_live_authority:false',
    'channel_contract_target_gate:true',
    'channel_intake_available:true',
    'channel_intake_status:operator_artifact_missing',
    'channel_intake_artifacts:0',
    'channel_intake_valid:0',
    'channel_intake_invalid:0',
    'channel_intake_valid_channels:0',
    'channel_intake_candidate:false',
    'channel_intake_gap:true',
    'channel_intake_closure_allowed:false',
    'channel_intake_live_authority:false',
    'channel_intake_target_gate:true',
    'processing_contract_available:true',
    'processing_contract_status:contract_ready_physical_fixture_missing',
    'processing_contract_schema:channel_processing_physical_readback.v1',
    'processing_contract_gap:true',
    'processing_contract_live_authority:false',
    'processing_contract_target_gate:true',
    'processing_intake_available:true',
    'processing_intake_status:operator_artifact_missing',
    'processing_intake_artifacts:0',
    'processing_intake_valid:0',
    'processing_intake_invalid:0',
    'processing_intake_valid_channels:0',
    'processing_intake_candidate:false',
    'processing_intake_gap:true',
    'processing_intake_closure_allowed:false',
    'processing_intake_live_authority:false',
    'processing_intake_target_gate:true',
    'bus_contract_available:true',
    'bus_contract_status:contract_ready_physical_fixture_missing',
    'bus_contract_schema:bus_group_main_dca_physical_readback.v1',
    'bus_contract_gap:true',
    'bus_contract_live_authority:false',
    'bus_contract_target_gate:true',
    'bus_intake_available:true',
    'bus_intake_status:operator_artifact_missing',
    'bus_intake_artifacts:0',
    'bus_intake_valid:0',
    'bus_intake_invalid:0',
    'bus_intake_valid_buses:0',
    'bus_intake_valid_mains:0',
    'bus_intake_valid_dcas:0',
    'bus_intake_candidate:false',
    'bus_intake_gap:true',
    'bus_intake_closure_allowed:false',
    'bus_intake_live_authority:false',
    'bus_intake_target_gate:true',
    'matrix_contract_available:true',
    'matrix_contract_status:contract_ready_physical_fixture_missing',
    'matrix_contract_schema:matrix_state_physical_readback.v1',
    'matrix_contract_gap:true',
    'matrix_contract_live_authority:false',
    'matrix_contract_target_gate:true',
    'matrix_intake_available:true',
    'matrix_intake_status:operator_artifact_missing',
    'matrix_intake_artifacts:0',
    'matrix_intake_valid:0',
    'matrix_intake_invalid:0',
    'matrix_intake_valid_matrices:0',
    'matrix_intake_candidate:false',
    'matrix_intake_gap:true',
    'matrix_intake_closure_allowed:false',
    'matrix_intake_live_authority:false',
    'matrix_intake_target_gate:true',
    'fx_contract_available:true',
    'fx_contract_status:contract_ready_physical_fixture_missing',
    'fx_contract_schema:fx_slots_inserts_physical_readback.v1',
    'fx_contract_gap:true',
    'fx_contract_live_authority:false',
    'fx_contract_target_gate:true',
    'fx_intake_available:true',
    'fx_intake_status:operator_artifact_missing',
    'fx_intake_artifacts:0',
    'fx_intake_valid:0',
    'fx_intake_invalid:0',
    'fx_intake_valid_slots:0',
    'fx_intake_valid_inserts:0',
    'fx_intake_candidate:false',
    'fx_intake_gap:true',
    'fx_intake_closure_allowed:false',
    'fx_intake_live_authority:false',
    'fx_intake_target_gate:true',
    'routing_contract_available:true',
    'routing_contract_status:contract_ready_physical_fixture_missing',
    'routing_contract_schema:routing_sends_physical_readback.v1',
    'routing_contract_gap:true',
    'routing_contract_live_authority:false',
    'routing_contract_target_gate:true',
    'routing_intake_available:true',
    'routing_intake_status:operator_artifact_missing',
    'routing_intake_artifacts:0',
    'routing_intake_valid:0',
    'routing_intake_invalid:0',
    'routing_intake_valid_static_routes:0',
    'routing_intake_valid_bus_sends:0',
    'routing_intake_valid_main_sends:0',
    'routing_intake_valid_matrix_sends:0',
    'routing_intake_valid_output_routes:0',
    'routing_intake_candidate:false',
    'routing_intake_gap:true',
    'routing_intake_closure_allowed:false',
    'routing_intake_live_authority:false',
    'routing_intake_target_gate:true',
    'scenes_contract_available:true',
    'scenes_contract_status:contract_ready_physical_fixture_missing',
    'scenes_contract_schema:snapshots_scenes_physical_readback.v1',
    'scenes_contract_gap:true',
    'scenes_contract_live_authority:false',
    'scenes_contract_target_gate:true',
    'scenes_intake_available:true',
    'scenes_intake_status:operator_artifact_missing',
    'scenes_intake_artifacts:0',
    'scenes_intake_valid:0',
    'scenes_intake_invalid:0',
    'scenes_intake_valid_snapshots:0',
    'scenes_intake_candidate:false',
    'scenes_intake_gap:true',
    'scenes_intake_closure_allowed:false',
    'scenes_intake_scene_load_allowed:false',
    'scenes_intake_scene_save_allowed:false',
    'scenes_intake_live_authority:false',
    'scenes_intake_target_gate:true',
    'presets_contract_available:true',
    'presets_contract_status:contract_ready_physical_fixture_missing',
    'presets_contract_schema:user_presets_physical_readback.v1',
    'presets_contract_gap:true',
    'presets_contract_live_authority:false',
    'presets_contract_target_gate:true',
    'presets_intake_available:true',
    'presets_intake_status:operator_artifact_missing',
    'presets_intake_artifacts:0',
    'presets_intake_valid:0',
    'presets_intake_invalid:0',
    'presets_intake_valid_presets:0',
    'presets_intake_candidate:false',
    'presets_intake_gap:true',
    'presets_intake_closure_allowed:false',
    'presets_intake_preset_recall_allowed:false',
    'presets_intake_preset_store_allowed:false',
    'presets_intake_live_authority:false',
    'presets_intake_target_gate:true',
    'dlive_contract_available:true',
    'dlive_contract_status:contract_ready_physical_fixture_missing',
    'dlive_contract_schema:dlive_state_physical_readback.v1',
    'dlive_contract_gap:true',
    'dlive_contract_live_authority:false',
    'dlive_contract_target_gate:true',
    'dlive_intake_available:true',
    'dlive_intake_status:operator_artifact_missing',
    'dlive_intake_artifacts:0',
    'dlive_intake_valid:0',
    'dlive_intake_invalid:0',
    'dlive_intake_valid_channels:0',
    'dlive_intake_candidate:false',
    'dlive_intake_gap:true',
    'dlive_intake_closure_allowed:false',
    'dlive_intake_midi_send_allowed:false',
    'dlive_intake_tcp_send_allowed:false',
    'dlive_intake_dlive_write_allowed:false',
    'dlive_intake_dlive_apply_allowed:false',
    'dlive_intake_live_authority:false',
    'dlive_intake_target_gate:true',
    'artifact_queue_available:true',
    'artifact_queue_status:operator_artifact_capture_required',
    'artifact_queue_surfaces:10',
    'artifact_queue_available_intakes:10',
    'artifact_queue_artifacts:0',
    'artifact_queue_valid:0',
    'artifact_queue_invalid:0',
    'artifact_queue_review_candidates:0',
    'artifact_queue_missing:10',
    'artifact_queue_unsafe:0',
    'artifact_queue_target_inconsistent:0',
    'artifact_queue_review_available:false',
    'artifact_queue_closure_allowed:false',
    'artifact_queue_live_authority:false',
    'artifact_queue_full_auto:false',
    'fixture_closure_plan_available:true',
    'fixture_closure_plan_status:fixture_closure_not_ready',
    'fixture_closure_plan_review_candidates:0',
    'fixture_closure_plan_acceptance_records:0',
    'fixture_closure_plan_accepted:0',
    'fixture_closure_plan_rejected:0',
    'fixture_closure_plan_invalid:0',
    'fixture_closure_plan_unreviewed:0',
    'fixture_closure_plan_candidates:0',
    'fixture_closure_plan_candidate_available:false',
    'fixture_update_allowed:false',
    'fixture_closure_allowed:false',
    'fixture_closure_plan_gap:true',
    'fixture_closure_plan_live_authority:false',
    'fixture_closure_plan_full_auto:false',
  ]);
  assert.match(proofEvidence.blockedReason, /Read-only artifact source/);
  assert.match(proofEvidence.nextStep, /overlay: dry_run_only/);
  assert.match(proofEvidence.nextStep, /age:120s stale_after:86400s stale:false/);
  assert.match(proofEvidence.nextStep, /read_surface_gaps:10/);
  assert.match(proofEvidence.nextStep, /fixture_paths:16/);
  assert.match(proofEvidence.nextStep, /test_names:43/);
  assert.match(proofEvidence.nextStep, /contract_gap:true/);
  assert.match(proofEvidence.nextStep, /gap_surfaces:raw_mixer_state_cache/);
  assert.match(proofEvidence.nextStep, /next_readback:raw_mixer_state_cache/);
  assert.match(proofEvidence.nextStep, /source_next_readback:raw_mixer_state_cache/);
  assert.match(proofEvidence.nextStep, /next_readback_gate:true/);
  assert.match(proofEvidence.nextStep, /capture_plan:operator_capture_required/);
  assert.match(proofEvidence.nextStep, /capture_steps:1/);
  assert.match(proofEvidence.nextStep, /capture_execution:false/);
  assert.match(proofEvidence.nextStep, /capture_target_gate:true/);
  assert.match(proofEvidence.nextStep, /raw_contract:contract_ready_physical_fixture_missing/);
  assert.match(proofEvidence.nextStep, /raw_contract_gap:true/);
  assert.match(proofEvidence.nextStep, /raw_contract_live_authority:false/);
  assert.match(proofEvidence.nextStep, /raw_contract_target_gate:true/);
  assert.match(proofEvidence.nextStep, /channel_contract:contract_ready_physical_fixture_missing/);
  assert.match(proofEvidence.nextStep, /channel_contract_gap:true/);
  assert.match(proofEvidence.nextStep, /channel_contract_live_authority:false/);
  assert.match(proofEvidence.nextStep, /channel_contract_target_gate:true/);
  assert.match(proofEvidence.nextStep, /channel_intake:operator_artifact_missing/);
  assert.match(proofEvidence.nextStep, /channel_intake_valid_channels:0/);
  assert.match(proofEvidence.nextStep, /channel_intake_gap:true/);
  assert.match(proofEvidence.nextStep, /channel_intake_live_authority:false/);
  assert.match(proofEvidence.nextStep, /channel_intake_target_gate:true/);
  assert.match(proofEvidence.nextStep, /processing_contract:contract_ready_physical_fixture_missing/);
  assert.match(proofEvidence.nextStep, /processing_contract_gap:true/);
  assert.match(proofEvidence.nextStep, /processing_contract_live_authority:false/);
  assert.match(proofEvidence.nextStep, /processing_contract_target_gate:true/);
  assert.match(proofEvidence.nextStep, /processing_intake:operator_artifact_missing/);
  assert.match(proofEvidence.nextStep, /processing_intake_valid_channels:0/);
  assert.match(proofEvidence.nextStep, /processing_intake_gap:true/);
  assert.match(proofEvidence.nextStep, /processing_intake_live_authority:false/);
  assert.match(proofEvidence.nextStep, /processing_intake_target_gate:true/);
  assert.match(proofEvidence.nextStep, /bus_contract:contract_ready_physical_fixture_missing/);
  assert.match(proofEvidence.nextStep, /bus_contract_gap:true/);
  assert.match(proofEvidence.nextStep, /bus_contract_live_authority:false/);
  assert.match(proofEvidence.nextStep, /bus_contract_target_gate:true/);
  assert.match(proofEvidence.nextStep, /bus_intake:operator_artifact_missing/);
  assert.match(proofEvidence.nextStep, /bus_intake_valid_buses:0/);
  assert.match(proofEvidence.nextStep, /bus_intake_valid_mains:0/);
  assert.match(proofEvidence.nextStep, /bus_intake_valid_dcas:0/);
  assert.match(proofEvidence.nextStep, /bus_intake_gap:true/);
  assert.match(proofEvidence.nextStep, /bus_intake_live_authority:false/);
  assert.match(proofEvidence.nextStep, /bus_intake_target_gate:true/);
  assert.match(proofEvidence.nextStep, /matrix_contract:contract_ready_physical_fixture_missing/);
  assert.match(proofEvidence.nextStep, /matrix_contract_gap:true/);
  assert.match(proofEvidence.nextStep, /matrix_contract_live_authority:false/);
  assert.match(proofEvidence.nextStep, /matrix_contract_target_gate:true/);
  assert.match(proofEvidence.nextStep, /matrix_intake:operator_artifact_missing/);
  assert.match(proofEvidence.nextStep, /matrix_intake_valid_matrices:0/);
  assert.match(proofEvidence.nextStep, /matrix_intake_gap:true/);
  assert.match(proofEvidence.nextStep, /matrix_intake_live_authority:false/);
  assert.match(proofEvidence.nextStep, /matrix_intake_target_gate:true/);
  assert.match(proofEvidence.nextStep, /fx_contract:contract_ready_physical_fixture_missing/);
  assert.match(proofEvidence.nextStep, /fx_contract_gap:true/);
  assert.match(proofEvidence.nextStep, /fx_contract_live_authority:false/);
  assert.match(proofEvidence.nextStep, /fx_contract_target_gate:true/);
  assert.match(proofEvidence.nextStep, /fx_intake:operator_artifact_missing/);
  assert.match(proofEvidence.nextStep, /fx_intake_valid_slots:0/);
  assert.match(proofEvidence.nextStep, /fx_intake_valid_inserts:0/);
  assert.match(proofEvidence.nextStep, /fx_intake_gap:true/);
  assert.match(proofEvidence.nextStep, /fx_intake_live_authority:false/);
  assert.match(proofEvidence.nextStep, /fx_intake_target_gate:true/);
  assert.match(proofEvidence.nextStep, /routing_contract:contract_ready_physical_fixture_missing/);
  assert.match(proofEvidence.nextStep, /routing_contract_gap:true/);
  assert.match(proofEvidence.nextStep, /routing_contract_live_authority:false/);
  assert.match(proofEvidence.nextStep, /routing_contract_target_gate:true/);
  assert.match(proofEvidence.nextStep, /scenes_contract:contract_ready_physical_fixture_missing/);
  assert.match(proofEvidence.nextStep, /scenes_contract_gap:true/);
  assert.match(proofEvidence.nextStep, /scenes_contract_live_authority:false/);
  assert.match(proofEvidence.nextStep, /scenes_contract_target_gate:true/);
  assert.match(proofEvidence.nextStep, /presets_contract:contract_ready_physical_fixture_missing/);
  assert.match(proofEvidence.nextStep, /presets_contract_gap:true/);
  assert.match(proofEvidence.nextStep, /presets_contract_live_authority:false/);
  assert.match(proofEvidence.nextStep, /presets_contract_target_gate:true/);
  assert.match(proofEvidence.nextStep, /presets_intake:operator_artifact_missing/);
  assert.match(proofEvidence.nextStep, /presets_intake_valid_presets:0/);
  assert.match(proofEvidence.nextStep, /presets_intake_closure:false/);
  assert.match(proofEvidence.nextStep, /presets_intake_preset_recall:false/);
  assert.match(proofEvidence.nextStep, /presets_intake_preset_store:false/);
  assert.match(proofEvidence.nextStep, /presets_intake_live_authority:false/);
  assert.match(proofEvidence.nextStep, /presets_intake_target_gate:true/);
  assert.match(proofEvidence.nextStep, /dlive_contract:contract_ready_physical_fixture_missing/);
  assert.match(proofEvidence.nextStep, /dlive_contract_gap:true/);
  assert.match(proofEvidence.nextStep, /dlive_contract_live_authority:false/);
  assert.match(proofEvidence.nextStep, /dlive_contract_target_gate:true/);
  assert.match(proofEvidence.nextStep, /dlive_intake:operator_artifact_missing/);
  assert.match(proofEvidence.nextStep, /dlive_intake_valid_channels:0/);
  assert.match(proofEvidence.nextStep, /dlive_intake_closure:false/);
  assert.match(proofEvidence.nextStep, /dlive_intake_midi_send:false/);
  assert.match(proofEvidence.nextStep, /dlive_intake_tcp_send:false/);
  assert.match(proofEvidence.nextStep, /dlive_intake_dlive_write:false/);
  assert.match(proofEvidence.nextStep, /dlive_intake_dlive_apply:false/);
  assert.match(proofEvidence.nextStep, /dlive_intake_live_authority:false/);
  assert.match(proofEvidence.nextStep, /dlive_intake_target_gate:true/);
  assert.match(proofEvidence.nextStep, /artifact_queue:operator_artifact_capture_required/);
  assert.match(proofEvidence.nextStep, /artifact_queue_surfaces:10/);
  assert.match(proofEvidence.nextStep, /artifact_queue_available_intakes:10/);
  assert.match(proofEvidence.nextStep, /artifact_queue_closure:false/);
  assert.match(proofEvidence.nextStep, /artifact_queue_live_authority:false/);
  assert.match(proofEvidence.nextStep, /artifact_queue_full_auto:false/);
  assert.match(proofEvidence.nextStep, /fixture_closure_plan:fixture_closure_not_ready/);
  assert.match(proofEvidence.nextStep, /fixture_closure_candidates:0/);
  assert.match(proofEvidence.nextStep, /fixture_update_allowed:false/);
  assert.match(proofEvidence.nextStep, /fixture_closure_allowed:false/);
  assert.match(proofEvidence.nextStep, /fixture_closure_gap:true/);
  assert.match(proofEvidence.nextStep, /fixture_closure_live_authority:false/);
  assert.match(proofEvidence.nextStep, /fixture_closure_full_auto:false/);
  assert.match(proofEvidence.nextStep, /dlive_state/);

  const capturePlan = rows.find(row => row.moduleId === 'read_surface_physical_readback_capture_plan');
  assert.equal(capturePlan.domain, 'read_surface_capture_plan');
  assert.equal(capturePlan.canWriteLive, false);
  assert.deepEqual(capturePlan.liveWriteKinds, []);
  assert.equal(capturePlan.status, 'operator_capture_required');
  assert.equal(capturePlan.entrypoint, 'build_read_surface_physical_readback_capture_plan');
  assert.equal(capturePlan.module, 'remediate_raw_mixer_state_cache');
  assert.equal(capturePlan.tone, 'yellow');
  assert.deepEqual(capturePlan.pipelineProfiles, [
    'available:true',
    'steps:1',
    'next_surface:raw_mixer_state_cache',
    'source_next_surface:raw_mixer_state_cache',
    'next_action:remediate_raw_mixer_state_cache',
    'source_next_action:remediate_raw_mixer_state_cache',
    'operator_required:true',
    'execution_allowed_by_plan:false',
    'live_authority:false',
    'target_consistency:true',
    'contracts:physical_readback_fixture,raw_mixer_state_physical_readback_contract',
    'forbidden:do not mark cached state as physical proof',
  ]);
  assert.match(capturePlan.blockedReason, /operator_must_capture_and_review_physical_readback_artifact/);
  assert.match(capturePlan.nextStep, /next_physical_readback:raw_mixer_state_cache/);
  assert.match(capturePlan.nextStep, /execution_allowed_by_plan:false/);
  assert.match(capturePlan.nextStep, /target_consistency:true/);

  const rawMixerContract = rows.find(row => row.moduleId === 'raw_mixer_state_physical_readback_contract');
  assert.equal(rawMixerContract.domain, 'read_surface_contract');
  assert.equal(rawMixerContract.canWriteLive, false);
  assert.deepEqual(rawMixerContract.liveWriteKinds, []);
  assert.equal(rawMixerContract.status, 'contract_ready_physical_fixture_missing');
  assert.equal(rawMixerContract.entrypoint, 'raw_mixer_state_physical_readback_contract');
  assert.equal(rawMixerContract.module, 'remediate_raw_mixer_state_cache');
  assert.equal(rawMixerContract.tone, 'yellow');
  assert.deepEqual(rawMixerContract.pipelineProfiles, [
    'available:true',
    'schema:raw_mixer_state_physical_readback.v1',
    'surface:raw_mixer_state_cache',
    'next_readback:raw_mixer_state_cache',
    'next_action:remediate_raw_mixer_state_cache',
    'physical_gap_remains:true',
    'live_authority:false',
    'target_consistency:true',
  ]);
  assert.match(rawMixerContract.blockedReason, /physical readback fixture evidence is still missing/);
  assert.match(rawMixerContract.nextStep, /next_readback:raw_mixer_state_cache/);
  assert.match(rawMixerContract.nextStep, /next_action:remediate_raw_mixer_state_cache/);
  assert.match(rawMixerContract.nextStep, /live_authority:false/);
  assert.match(rawMixerContract.nextStep, /target_consistency:true/);

  const rawMixerIntake = rows.find(row => row.moduleId === 'raw_mixer_state_physical_readback_artifact_intake');
  assert.equal(rawMixerIntake.domain, 'read_surface_artifact_intake');
  assert.equal(rawMixerIntake.canWriteLive, false);
  assert.deepEqual(rawMixerIntake.liveWriteKinds, []);
  assert.equal(rawMixerIntake.status, 'operator_artifact_missing');
  assert.equal(rawMixerIntake.entrypoint, 'build_raw_mixer_state_physical_readback_artifact_intake');
  assert.equal(rawMixerIntake.module, 'remediate_raw_mixer_state_cache');
  assert.equal(rawMixerIntake.tone, 'yellow');
  assert.deepEqual(rawMixerIntake.pipelineProfiles, [
    'available:true',
    'schema:raw_mixer_state_physical_readback.v1',
    'surface:raw_mixer_state_cache',
    'action:remediate_raw_mixer_state_cache',
    'artifacts:0',
    'valid:0',
    'invalid:0',
    'candidate:false',
    'operator_review:true',
    'physical_gap_remains:true',
    'fixture_closure_allowed:false',
    'live_authority:false',
    'target_consistency:true',
  ]);
  assert.match(rawMixerIntake.blockedReason, /operator_must_capture_raw_mixer_state_physical_readback_artifact/);
  assert.match(rawMixerIntake.nextStep, /surface:raw_mixer_state_cache/);
  assert.match(rawMixerIntake.nextStep, /artifacts:0/);
  assert.match(rawMixerIntake.nextStep, /operator_review:true/);
  assert.match(rawMixerIntake.nextStep, /fixture_closure_allowed:false/);
  assert.match(rawMixerIntake.nextStep, /live_authority:false/);

  const channelBasicContract = rows.find(row => row.moduleId === 'channel_basic_controls_physical_readback_contract');
  assert.equal(channelBasicContract.domain, 'read_surface_contract');
  assert.equal(channelBasicContract.canWriteLive, false);
  assert.deepEqual(channelBasicContract.liveWriteKinds, []);
  assert.equal(channelBasicContract.status, 'contract_ready_physical_fixture_missing');
  assert.equal(channelBasicContract.entrypoint, 'channel_basic_controls_physical_readback_contract');
  assert.equal(channelBasicContract.module, 'remediate_channel_identity_and_basic_controls');
  assert.equal(channelBasicContract.tone, 'yellow');
  assert.deepEqual(channelBasicContract.pipelineProfiles, [
    'available:true',
    'schema:channel_basic_controls_physical_readback.v1',
    'surface:channel_identity_and_basic_controls',
    'physical_readback:channel_identity_and_basic_controls',
    'physical_action:remediate_channel_identity_and_basic_controls',
    'physical_gap_remains:true',
    'live_authority:false',
    'target_consistency:true',
    'promotion_families:fader|gain|mute|pan_phase_delay',
    'controls:fader_db|trim_db|mute|pan',
  ]);
  assert.match(channelBasicContract.blockedReason, /physical readback fixture evidence is still missing/);
  assert.match(channelBasicContract.nextStep, /physical_readback:channel_identity_and_basic_controls/);
  assert.match(channelBasicContract.nextStep, /physical_action:remediate_channel_identity_and_basic_controls/);
  assert.match(channelBasicContract.nextStep, /families:fader,gain,mute,pan_phase_delay/);
  assert.match(channelBasicContract.nextStep, /controls:fader_db,trim_db,mute,pan/);
  assert.match(channelBasicContract.nextStep, /live_authority:false/);
  assert.match(channelBasicContract.nextStep, /target_consistency:true/);

  const channelBasicIntake = rows.find(row => row.moduleId === 'channel_basic_controls_physical_readback_artifact_intake');
  assert.equal(channelBasicIntake.domain, 'read_surface_artifact_intake');
  assert.equal(channelBasicIntake.canWriteLive, false);
  assert.deepEqual(channelBasicIntake.liveWriteKinds, []);
  assert.equal(channelBasicIntake.status, 'operator_artifact_missing');
  assert.equal(
    channelBasicIntake.entrypoint,
    'build_channel_basic_controls_physical_readback_artifact_intake',
  );
  assert.equal(channelBasicIntake.module, 'remediate_channel_identity_and_basic_controls');
  assert.equal(channelBasicIntake.tone, 'yellow');
  assert.deepEqual(channelBasicIntake.pipelineProfiles, [
    'available:true',
    'schema:channel_basic_controls_physical_readback.v1',
    'surface:channel_identity_and_basic_controls',
    'action:remediate_channel_identity_and_basic_controls',
    'artifacts:0',
    'valid:0',
    'invalid:0',
    'valid_channels:0',
    'candidate:false',
    'operator_review:true',
    'physical_gap_remains:true',
    'fixture_closure_allowed:false',
    'live_authority:false',
    'target_consistency:true',
    'promotion_families:fader|gain|mute|pan_phase_delay',
  ]);
  assert.match(
    channelBasicIntake.blockedReason,
    /operator_must_capture_channel_basic_controls_physical_readback_artifact/,
  );
  assert.match(channelBasicIntake.nextStep, /surface:channel_identity_and_basic_controls/);
  assert.match(channelBasicIntake.nextStep, /artifacts:0/);
  assert.match(channelBasicIntake.nextStep, /valid_channels:0/);
  assert.match(channelBasicIntake.nextStep, /operator_review:true/);
  assert.match(channelBasicIntake.nextStep, /fixture_closure_allowed:false/);
  assert.match(channelBasicIntake.nextStep, /live_authority:false/);

  const channelProcessingContract = rows.find(row => row.moduleId === 'channel_processing_physical_readback_contract');
  assert.equal(channelProcessingContract.domain, 'read_surface_contract');
  assert.equal(channelProcessingContract.canWriteLive, false);
  assert.deepEqual(channelProcessingContract.liveWriteKinds, []);
  assert.equal(channelProcessingContract.status, 'contract_ready_physical_fixture_missing');
  assert.equal(channelProcessingContract.entrypoint, 'channel_processing_physical_readback_contract');
  assert.equal(channelProcessingContract.module, 'remediate_channel_eq_dynamics_gate');
  assert.equal(channelProcessingContract.tone, 'yellow');
  assert.deepEqual(channelProcessingContract.pipelineProfiles, [
    'available:true',
    'schema:channel_processing_physical_readback.v1',
    'surface:channel_eq_dynamics_gate',
    'physical_readback:channel_eq_dynamics_gate',
    'physical_action:remediate_channel_eq_dynamics_gate',
    'physical_gap_remains:true',
    'live_authority:false',
    'target_consistency:true',
    'promotion_families:eq|compressor|gate|deessing',
    'eq_band_fields:band|frequency_hz|gain_db|q',
    'compressor_fields:enabled|model|gain_reduction_db',
    'gate_fields:enabled|model|range_db',
  ]);
  assert.match(channelProcessingContract.blockedReason, /physical readback fixture evidence is still missing/);
  assert.match(channelProcessingContract.nextStep, /physical_readback:channel_eq_dynamics_gate/);
  assert.match(channelProcessingContract.nextStep, /physical_action:remediate_channel_eq_dynamics_gate/);
  assert.match(channelProcessingContract.nextStep, /families:eq,compressor,gate,deessing/);
  assert.match(channelProcessingContract.nextStep, /eq_bands:band,frequency_hz,gain_db,q/);
  assert.match(channelProcessingContract.nextStep, /compressor:enabled,model,gain_reduction_db/);
  assert.match(channelProcessingContract.nextStep, /gate:enabled,model,range_db/);
  assert.match(channelProcessingContract.nextStep, /live_authority:false/);
  assert.match(channelProcessingContract.nextStep, /target_consistency:true/);

  const channelProcessingIntake = rows.find(row => row.moduleId === 'channel_processing_physical_readback_artifact_intake');
  assert.equal(channelProcessingIntake.domain, 'read_surface_artifact_intake');
  assert.equal(channelProcessingIntake.canWriteLive, false);
  assert.deepEqual(channelProcessingIntake.liveWriteKinds, []);
  assert.equal(channelProcessingIntake.status, 'operator_artifact_missing');
  assert.equal(
    channelProcessingIntake.entrypoint,
    'build_channel_processing_physical_readback_artifact_intake',
  );
  assert.equal(channelProcessingIntake.module, 'remediate_channel_eq_dynamics_gate');
  assert.equal(channelProcessingIntake.tone, 'yellow');
  assert.deepEqual(channelProcessingIntake.pipelineProfiles, [
    'available:true',
    'schema:channel_processing_physical_readback.v1',
    'surface:channel_eq_dynamics_gate',
    'action:remediate_channel_eq_dynamics_gate',
    'artifacts:0',
    'valid:0',
    'invalid:0',
    'valid_channels:0',
    'candidate:false',
    'operator_review:true',
    'physical_gap_remains:true',
    'fixture_closure_allowed:false',
    'live_authority:false',
    'target_consistency:true',
    'promotion_families:eq|compressor|gate|deessing|frequency_separation|cross_channel_unmasking',
  ]);
  assert.match(
    channelProcessingIntake.blockedReason,
    /operator_must_capture_channel_processing_physical_readback_artifact/,
  );
  assert.match(channelProcessingIntake.nextStep, /surface:channel_eq_dynamics_gate/);
  assert.match(channelProcessingIntake.nextStep, /artifacts:0/);
  assert.match(channelProcessingIntake.nextStep, /valid_channels:0/);
  assert.match(channelProcessingIntake.nextStep, /operator_review:true/);
  assert.match(channelProcessingIntake.nextStep, /fixture_closure_allowed:false/);
  assert.match(channelProcessingIntake.nextStep, /live_authority:false/);

  const busMainDcaContract = rows.find(row => row.moduleId === 'bus_group_main_dca_physical_readback_contract');
  assert.equal(busMainDcaContract.domain, 'read_surface_contract');
  assert.equal(busMainDcaContract.canWriteLive, false);
  assert.deepEqual(busMainDcaContract.liveWriteKinds, []);
  assert.equal(busMainDcaContract.status, 'contract_ready_physical_fixture_missing');
  assert.equal(busMainDcaContract.entrypoint, 'bus_group_main_dca_physical_readback_contract');
  assert.equal(busMainDcaContract.module, 'remediate_bus_group_main_dca_state');
  assert.equal(busMainDcaContract.tone, 'yellow');
  assert.deepEqual(busMainDcaContract.pipelineProfiles, [
    'available:true',
    'schema:bus_group_main_dca_physical_readback.v1',
    'surface:bus_group_main_dca_state',
    'physical_readback:bus_group_main_dca_state',
    'physical_action:remediate_bus_group_main_dca_state',
    'physical_gap_remains:true',
    'live_authority:false',
    'target_consistency:true',
    'promotion_families:bus_group_matrix|dca|master_bus',
    'bus_controls:fader_db|mute|pan|width|mono',
    'main_controls:fader_db|mute|pan',
    'dca_controls:fader_db|mute|monitor',
    'bus_assignments:dca_assignments|main_send_on|main_send_level_db|main_send_pre',
    'dca_assignments:member_channels|member_buses',
    'metering:lufs|true_peak_dbtp|headroom_db',
  ]);
  assert.match(busMainDcaContract.blockedReason, /physical readback fixture evidence is still missing/);
  assert.match(busMainDcaContract.nextStep, /physical_readback:bus_group_main_dca_state/);
  assert.match(busMainDcaContract.nextStep, /physical_action:remediate_bus_group_main_dca_state/);
  assert.match(busMainDcaContract.nextStep, /families:bus_group_matrix,dca,master_bus/);
  assert.match(busMainDcaContract.nextStep, /bus_controls:fader_db,mute,pan,width,mono/);
  assert.match(busMainDcaContract.nextStep, /main_controls:fader_db,mute,pan/);
  assert.match(busMainDcaContract.nextStep, /dca_controls:fader_db,mute,monitor/);
  assert.match(busMainDcaContract.nextStep, /metering:lufs,true_peak_dbtp,headroom_db/);
  assert.match(busMainDcaContract.nextStep, /live_authority:false/);
  assert.match(busMainDcaContract.nextStep, /target_consistency:true/);

  const busMainDcaIntake = rows.find(row => row.moduleId === 'bus_group_main_dca_physical_readback_artifact_intake');
  assert.equal(busMainDcaIntake.domain, 'read_surface_artifact_intake');
  assert.equal(busMainDcaIntake.canWriteLive, false);
  assert.deepEqual(busMainDcaIntake.liveWriteKinds, []);
  assert.equal(busMainDcaIntake.status, 'operator_artifact_missing');
  assert.equal(
    busMainDcaIntake.entrypoint,
    'build_bus_group_main_dca_physical_readback_artifact_intake',
  );
  assert.equal(busMainDcaIntake.module, 'remediate_bus_group_main_dca_state');
  assert.equal(busMainDcaIntake.tone, 'yellow');
  assert.deepEqual(busMainDcaIntake.pipelineProfiles, [
    'available:true',
    'schema:bus_group_main_dca_physical_readback.v1',
    'surface:bus_group_main_dca_state',
    'action:remediate_bus_group_main_dca_state',
    'artifacts:0',
    'valid:0',
    'invalid:0',
    'valid_buses:0',
    'valid_mains:0',
    'valid_dcas:0',
    'candidate:false',
    'operator_review:true',
    'physical_gap_remains:true',
    'fixture_closure_allowed:false',
    'live_authority:false',
    'target_consistency:true',
    'promotion_families:bus_group_matrix|dca|master_bus|mix_balance|performance_adaptation',
  ]);
  assert.match(
    busMainDcaIntake.blockedReason,
    /operator_must_capture_bus_group_main_dca_physical_readback_artifact/,
  );
  assert.match(busMainDcaIntake.nextStep, /surface:bus_group_main_dca_state/);
  assert.match(busMainDcaIntake.nextStep, /artifacts:0/);
  assert.match(busMainDcaIntake.nextStep, /valid_buses:0/);
  assert.match(busMainDcaIntake.nextStep, /valid_mains:0/);
  assert.match(busMainDcaIntake.nextStep, /valid_dcas:0/);
  assert.match(busMainDcaIntake.nextStep, /operator_review:true/);
  assert.match(busMainDcaIntake.nextStep, /fixture_closure_allowed:false/);
  assert.match(busMainDcaIntake.nextStep, /live_authority:false/);

  const matrixStateContract = rows.find(row => row.moduleId === 'matrix_state_physical_readback_contract');
  assert.equal(matrixStateContract.domain, 'read_surface_contract');
  assert.equal(matrixStateContract.canWriteLive, false);
  assert.deepEqual(matrixStateContract.liveWriteKinds, []);
  assert.equal(matrixStateContract.status, 'contract_ready_physical_fixture_missing');
  assert.equal(matrixStateContract.entrypoint, 'matrix_state_physical_readback_contract');
  assert.equal(matrixStateContract.module, 'remediate_matrix_state');
  assert.equal(matrixStateContract.tone, 'yellow');
  assert.deepEqual(matrixStateContract.pipelineProfiles, [
    'available:true',
    'schema:matrix_state_physical_readback.v1',
    'surface:matrix_state',
    'physical_readback:matrix_state',
    'physical_action:remediate_matrix_state',
    'physical_gap_remains:true',
    'live_authority:false',
    'target_consistency:true',
    'promotion_families:bus_group_matrix|fx_send_insert|routing',
    'controls:fader_db|mute|pan',
    'eq_band_fields:band|frequency_hz|gain_db|q',
    'dynamics:enabled|model|gain_reduction_db',
    'inserts:pre_insert_on|post_insert_weight',
    'metering:lufs|true_peak_dbtp|headroom_db',
  ]);
  assert.match(matrixStateContract.blockedReason, /physical readback fixture evidence is still missing/);
  assert.match(matrixStateContract.nextStep, /physical_readback:matrix_state/);
  assert.match(matrixStateContract.nextStep, /physical_action:remediate_matrix_state/);
  assert.match(matrixStateContract.nextStep, /families:bus_group_matrix,fx_send_insert,routing/);
  assert.match(matrixStateContract.nextStep, /controls:fader_db,mute,pan/);
  assert.match(matrixStateContract.nextStep, /eq_bands:band,frequency_hz,gain_db,q/);
  assert.match(matrixStateContract.nextStep, /dynamics:enabled,model,gain_reduction_db/);
  assert.match(matrixStateContract.nextStep, /inserts:pre_insert_on,post_insert_weight/);
  assert.match(matrixStateContract.nextStep, /metering:lufs,true_peak_dbtp,headroom_db/);
  assert.match(matrixStateContract.nextStep, /live_authority:false/);
  assert.match(matrixStateContract.nextStep, /target_consistency:true/);

  const matrixStateIntake = rows.find(row => row.moduleId === 'matrix_state_physical_readback_artifact_intake');
  assert.equal(matrixStateIntake.domain, 'read_surface_artifact_intake');
  assert.equal(matrixStateIntake.canWriteLive, false);
  assert.deepEqual(matrixStateIntake.liveWriteKinds, []);
  assert.equal(matrixStateIntake.status, 'operator_artifact_missing');
  assert.equal(matrixStateIntake.entrypoint, 'build_matrix_state_physical_readback_artifact_intake');
  assert.equal(matrixStateIntake.module, 'remediate_matrix_state');
  assert.equal(matrixStateIntake.tone, 'yellow');
  assert.deepEqual(matrixStateIntake.pipelineProfiles, [
    'available:true',
    'schema:matrix_state_physical_readback.v1',
    'surface:matrix_state',
    'action:remediate_matrix_state',
    'artifacts:0',
    'valid:0',
    'invalid:0',
    'valid_matrices:0',
    'candidate:false',
    'operator_review:true',
    'physical_gap_remains:true',
    'fixture_closure_allowed:false',
    'live_authority:false',
    'target_consistency:true',
    'promotion_families:bus_group_matrix|fx_send_insert|master_bus|mix_balance|routing|performance_adaptation',
  ]);
  assert.match(matrixStateIntake.blockedReason, /operator_must_capture_matrix_state_physical_readback_artifact/);
  assert.match(matrixStateIntake.nextStep, /surface:matrix_state/);
  assert.match(matrixStateIntake.nextStep, /action:remediate_matrix_state/);
  assert.match(matrixStateIntake.nextStep, /artifacts:0/);
  assert.match(matrixStateIntake.nextStep, /valid_matrices:0/);
  assert.match(matrixStateIntake.nextStep, /operator_review:true/);
  assert.match(matrixStateIntake.nextStep, /fixture_closure_allowed:false/);
  assert.match(matrixStateIntake.nextStep, /live_authority:false/);

  const fxSlotsInsertsContract = rows.find(row => row.moduleId === 'fx_slots_inserts_physical_readback_contract');
  assert.equal(fxSlotsInsertsContract.domain, 'read_surface_contract');
  assert.equal(fxSlotsInsertsContract.canWriteLive, false);
  assert.deepEqual(fxSlotsInsertsContract.liveWriteKinds, []);
  assert.equal(fxSlotsInsertsContract.status, 'contract_ready_physical_fixture_missing');
  assert.equal(fxSlotsInsertsContract.entrypoint, 'fx_slots_inserts_physical_readback_contract');
  assert.equal(fxSlotsInsertsContract.module, 'remediate_fx_slots_and_inserts');
  assert.equal(fxSlotsInsertsContract.tone, 'yellow');
  assert.deepEqual(fxSlotsInsertsContract.pipelineProfiles, [
    'available:true',
    'schema:fx_slots_inserts_physical_readback.v1',
    'surface:fx_slots_and_inserts',
    'physical_readback:fx_slots_and_inserts',
    'physical_action:remediate_fx_slots_and_inserts',
    'physical_gap_remains:true',
    'live_authority:false',
    'target_consistency:true',
    'promotion_families:fx_send_insert|routing|mix_balance',
    'slot_fields:slot|slot_number|parameter_schema',
    'parameter_fields:parameter_index|value|safe_range',
    'parameter_schema:status|physical_model_schema_verified',
    'insert_fields:target_type|position|linked_fx_slot',
    'insert_coverage:assignment|status|scoped_diff_marker',
    'false_safety:live_write|mutates_console_state|live_widening_allowed|full_autonomous_ready',
  ]);
  assert.match(fxSlotsInsertsContract.blockedReason, /physical readback fixture evidence is still missing/);
  assert.match(fxSlotsInsertsContract.nextStep, /physical_readback:fx_slots_and_inserts/);
  assert.match(fxSlotsInsertsContract.nextStep, /physical_action:remediate_fx_slots_and_inserts/);
  assert.match(fxSlotsInsertsContract.nextStep, /families:fx_send_insert,routing,mix_balance/);
  assert.match(fxSlotsInsertsContract.nextStep, /slot_fields:slot,slot_number,parameter_schema/);
  assert.match(fxSlotsInsertsContract.nextStep, /parameters:parameter_index,value,safe_range/);
  assert.match(fxSlotsInsertsContract.nextStep, /inserts:target_type,position,linked_fx_slot/);
  assert.match(
    fxSlotsInsertsContract.nextStep,
    /false_safety:live_write,mutates_console_state,live_widening_allowed,full_autonomous_ready/,
  );
  assert.match(fxSlotsInsertsContract.nextStep, /live_authority:false/);
  assert.match(fxSlotsInsertsContract.nextStep, /target_consistency:true/);

  const fxSlotsInsertsIntake = rows.find(row => row.moduleId === 'fx_slots_inserts_physical_readback_artifact_intake');
  assert.equal(fxSlotsInsertsIntake.domain, 'read_surface_artifact_intake');
  assert.equal(fxSlotsInsertsIntake.canWriteLive, false);
  assert.deepEqual(fxSlotsInsertsIntake.liveWriteKinds, []);
  assert.equal(fxSlotsInsertsIntake.status, 'operator_artifact_missing');
  assert.equal(fxSlotsInsertsIntake.entrypoint, 'build_fx_slots_inserts_physical_readback_artifact_intake');
  assert.equal(fxSlotsInsertsIntake.module, 'remediate_fx_slots_and_inserts');
  assert.equal(fxSlotsInsertsIntake.tone, 'yellow');
  assert.deepEqual(fxSlotsInsertsIntake.pipelineProfiles, [
    'available:true',
    'schema:fx_slots_inserts_physical_readback.v1',
    'surface:fx_slots_and_inserts',
    'action:remediate_fx_slots_and_inserts',
    'artifacts:0',
    'valid:0',
    'invalid:0',
    'valid_slots:0',
    'valid_inserts:0',
    'candidate:false',
    'operator_review:true',
    'physical_gap_remains:true',
    'fixture_closure_allowed:false',
    'live_authority:false',
    'target_consistency:true',
    'promotion_families:fx_send_insert|routing|mix_balance|performance_adaptation',
  ]);
  assert.match(fxSlotsInsertsIntake.blockedReason, /operator_must_capture_fx_slots_inserts_physical_readback_artifact/);
  assert.match(fxSlotsInsertsIntake.nextStep, /surface:fx_slots_and_inserts/);
  assert.match(fxSlotsInsertsIntake.nextStep, /action:remediate_fx_slots_and_inserts/);
  assert.match(fxSlotsInsertsIntake.nextStep, /artifacts:0/);
  assert.match(fxSlotsInsertsIntake.nextStep, /valid_slots:0/);
  assert.match(fxSlotsInsertsIntake.nextStep, /valid_inserts:0/);
  assert.match(fxSlotsInsertsIntake.nextStep, /operator_review:true/);
  assert.match(fxSlotsInsertsIntake.nextStep, /fixture_closure_allowed:false/);
  assert.match(fxSlotsInsertsIntake.nextStep, /live_authority:false/);

  const routingSendsContract = rows.find(row => row.moduleId === 'routing_sends_physical_readback_contract');
  assert.equal(routingSendsContract.domain, 'read_surface_contract');
  assert.equal(routingSendsContract.canWriteLive, false);
  assert.deepEqual(routingSendsContract.liveWriteKinds, []);
  assert.equal(routingSendsContract.status, 'contract_ready_physical_fixture_missing');
  assert.equal(routingSendsContract.entrypoint, 'routing_sends_physical_readback_contract');
  assert.equal(routingSendsContract.module, 'remediate_routing_and_sends');
  assert.equal(routingSendsContract.tone, 'yellow');
  assert.deepEqual(routingSendsContract.pipelineProfiles, [
    'available:true',
    'schema:routing_sends_physical_readback.v1',
    'surface:routing_and_sends',
    'physical_readback:routing_and_sends',
    'physical_action:remediate_routing_and_sends',
    'physical_gap_remains:true',
    'live_authority:false',
    'target_consistency:true',
    'promotion_families:routing|fx_send_insert|bus_group_matrix|scene_preset|mix_balance|performance_adaptation',
    'static_routes:start|end|role|wing_routing_hint',
    'bus_sends:channel|send|level_db|pre_on',
    'main_sends:channel|main|level_db|pre',
    'matrix_sends:channel|send|target_matrix',
    'output_routes:output_group|output_number|source_channel',
    'send_coverage:assignment|level|scoped_diff_marker',
    'output_coverage:assignment|source|scoped_diff_marker',
    'false_safety:live_write|mutates_console_state|live_widening_allowed|full_autonomous_ready',
  ]);
  assert.match(routingSendsContract.blockedReason, /physical readback fixture evidence is still missing/);
  assert.match(routingSendsContract.nextStep, /physical_readback:routing_and_sends/);
  assert.match(routingSendsContract.nextStep, /physical_action:remediate_routing_and_sends/);
  assert.match(
    routingSendsContract.nextStep,
    /families:routing,fx_send_insert,bus_group_matrix,scene_preset,mix_balance,performance_adaptation/,
  );
  assert.match(routingSendsContract.nextStep, /static_routes:start,end,role,wing_routing_hint/);
  assert.match(routingSendsContract.nextStep, /bus_sends:channel,send,level_db,pre_on/);
  assert.match(routingSendsContract.nextStep, /main_sends:channel,main,level_db,pre/);
  assert.match(routingSendsContract.nextStep, /matrix_sends:channel,send,target_matrix/);
  assert.match(routingSendsContract.nextStep, /output_routes:output_group,output_number,source_channel/);
  assert.match(
    routingSendsContract.nextStep,
    /false_safety:live_write,mutates_console_state,live_widening_allowed,full_autonomous_ready/,
  );
  assert.match(routingSendsContract.nextStep, /live_authority:false/);
  assert.match(routingSendsContract.nextStep, /target_consistency:true/);

  const routingSendsIntake = rows.find(row => row.moduleId === 'routing_sends_physical_readback_artifact_intake');
  assert.equal(routingSendsIntake.domain, 'read_surface_artifact_intake');
  assert.equal(routingSendsIntake.canWriteLive, false);
  assert.deepEqual(routingSendsIntake.liveWriteKinds, []);
  assert.equal(routingSendsIntake.status, 'operator_artifact_missing');
  assert.equal(routingSendsIntake.entrypoint, 'build_routing_sends_physical_readback_artifact_intake');
  assert.equal(routingSendsIntake.module, 'remediate_routing_and_sends');
  assert.equal(routingSendsIntake.tone, 'yellow');
  assert.deepEqual(routingSendsIntake.pipelineProfiles, [
    'available:true',
    'schema:routing_sends_physical_readback.v1',
    'surface:routing_and_sends',
    'action:remediate_routing_and_sends',
    'artifacts:0',
    'valid:0',
    'invalid:0',
    'valid_static_routes:0',
    'valid_bus_sends:0',
    'valid_main_sends:0',
    'valid_matrix_sends:0',
    'valid_output_routes:0',
    'candidate:false',
    'operator_review:true',
    'physical_gap_remains:true',
    'fixture_closure_allowed:false',
    'live_authority:false',
    'target_consistency:true',
    'promotion_families:routing|fx_send_insert|bus_group_matrix|scene_preset|mix_balance|performance_adaptation',
  ]);
  assert.match(routingSendsIntake.blockedReason, /operator_must_capture_routing_sends_physical_readback_artifact/);
  assert.match(routingSendsIntake.nextStep, /surface:routing_and_sends/);
  assert.match(routingSendsIntake.nextStep, /action:remediate_routing_and_sends/);
  assert.match(routingSendsIntake.nextStep, /artifacts:0/);
  assert.match(routingSendsIntake.nextStep, /valid_static_routes:0/);
  assert.match(routingSendsIntake.nextStep, /valid_bus_sends:0/);
  assert.match(routingSendsIntake.nextStep, /valid_output_routes:0/);
  assert.match(routingSendsIntake.nextStep, /operator_review:true/);
  assert.match(routingSendsIntake.nextStep, /fixture_closure_allowed:false/);
  assert.match(routingSendsIntake.nextStep, /live_authority:false/);

  const snapshotsScenesContract = rows.find(row => row.moduleId === 'snapshots_scenes_physical_readback_contract');
  assert.equal(snapshotsScenesContract.domain, 'read_surface_contract');
  assert.equal(snapshotsScenesContract.canWriteLive, false);
  assert.deepEqual(snapshotsScenesContract.liveWriteKinds, []);
  assert.equal(snapshotsScenesContract.status, 'contract_ready_physical_fixture_missing');
  assert.equal(snapshotsScenesContract.entrypoint, 'snapshots_scenes_physical_readback_contract');
  assert.equal(snapshotsScenesContract.module, 'remediate_snapshots_scenes');
  assert.equal(snapshotsScenesContract.tone, 'yellow');
  assert.deepEqual(snapshotsScenesContract.pipelineProfiles, [
    'available:true',
    'schema:snapshots_scenes_physical_readback.v1',
    'surface:snapshots_scenes',
    'physical_readback:snapshots_scenes',
    'physical_action:remediate_snapshots_scenes',
    'physical_gap_remains:true',
    'live_authority:false',
    'target_consistency:true',
    'scene_load_allowed:false',
    'scene_save_allowed:false',
    'promotion_families:scene_preset|routing|mix_balance',
    'active_scene:active_snapshot_index|active_snapshot_name|current_show',
    'snapshots:snapshot_index|snapshot_name|scene_content_fingerprint|rollback_reference',
    'coverage:catalog_identity|active_scene|rollback_reference',
    'false_safety:live_write|mutates_console_state|scene_load_allowed|scene_save_allowed',
  ]);
  assert.match(snapshotsScenesContract.blockedReason, /physical readback fixture evidence is still missing/);
  assert.match(snapshotsScenesContract.nextStep, /physical_readback:snapshots_scenes/);
  assert.match(snapshotsScenesContract.nextStep, /physical_action:remediate_snapshots_scenes/);
  assert.match(snapshotsScenesContract.nextStep, /families:scene_preset,routing,mix_balance/);
  assert.match(snapshotsScenesContract.nextStep, /active_scene:active_snapshot_index,active_snapshot_name,current_show/);
  assert.match(snapshotsScenesContract.nextStep, /snapshots:snapshot_index,snapshot_name,scene_content_fingerprint,rollback_reference/);
  assert.match(snapshotsScenesContract.nextStep, /coverage:catalog_identity,active_scene,rollback_reference/);
  assert.match(snapshotsScenesContract.nextStep, /scene_load_allowed:false/);
  assert.match(snapshotsScenesContract.nextStep, /scene_save_allowed:false/);
  assert.match(snapshotsScenesContract.nextStep, /live_authority:false/);
  assert.match(snapshotsScenesContract.nextStep, /target_consistency:true/);

  const snapshotsScenesIntake = rows.find(row => row.moduleId === 'snapshots_scenes_physical_readback_artifact_intake');
  assert.equal(snapshotsScenesIntake.domain, 'read_surface_artifact_intake');
  assert.equal(snapshotsScenesIntake.canWriteLive, false);
  assert.deepEqual(snapshotsScenesIntake.liveWriteKinds, []);
  assert.equal(snapshotsScenesIntake.status, 'operator_artifact_missing');
  assert.equal(snapshotsScenesIntake.entrypoint, 'build_snapshots_scenes_physical_readback_artifact_intake');
  assert.equal(snapshotsScenesIntake.module, 'remediate_snapshots_scenes');
  assert.equal(snapshotsScenesIntake.tone, 'yellow');
  assert.deepEqual(snapshotsScenesIntake.pipelineProfiles, [
    'available:true',
    'schema:snapshots_scenes_physical_readback.v1',
    'surface:snapshots_scenes',
    'action:remediate_snapshots_scenes',
    'artifacts:0',
    'valid:0',
    'invalid:0',
    'valid_snapshots:0',
    'candidate:false',
    'operator_review:true',
    'physical_gap_remains:true',
    'fixture_closure_allowed:false',
    'scene_load_allowed:false',
    'scene_save_allowed:false',
    'live_authority:false',
    'target_consistency:true',
    'promotion_families:scene_preset|routing|mix_balance|performance_adaptation',
  ]);
  assert.match(snapshotsScenesIntake.blockedReason, /operator_must_capture_snapshots_scenes_physical_readback_artifact/);
  assert.match(snapshotsScenesIntake.nextStep, /surface:snapshots_scenes/);
  assert.match(snapshotsScenesIntake.nextStep, /action:remediate_snapshots_scenes/);
  assert.match(snapshotsScenesIntake.nextStep, /artifacts:0/);
  assert.match(snapshotsScenesIntake.nextStep, /valid_snapshots:0/);
  assert.match(snapshotsScenesIntake.nextStep, /operator_review:true/);
  assert.match(snapshotsScenesIntake.nextStep, /fixture_closure_allowed:false/);
  assert.match(snapshotsScenesIntake.nextStep, /scene_load_allowed:false/);
  assert.match(snapshotsScenesIntake.nextStep, /scene_save_allowed:false/);
  assert.match(snapshotsScenesIntake.nextStep, /live_authority:false/);

  const userPresetsContract = rows.find(row => row.moduleId === 'user_presets_physical_readback_contract');
  assert.equal(userPresetsContract.domain, 'read_surface_contract');
  assert.equal(userPresetsContract.canWriteLive, false);
  assert.deepEqual(userPresetsContract.liveWriteKinds, []);
  assert.equal(userPresetsContract.status, 'contract_ready_physical_fixture_missing');
  assert.equal(userPresetsContract.entrypoint, 'user_presets_physical_readback_contract');
  assert.equal(userPresetsContract.module, 'remediate_user_presets');
  assert.equal(userPresetsContract.tone, 'yellow');
  assert.deepEqual(userPresetsContract.pipelineProfiles, [
    'available:true',
    'schema:user_presets_physical_readback.v1',
    'surface:user_presets',
    'physical_readback:user_presets',
    'physical_action:remediate_user_presets',
    'physical_gap_remains:true',
    'live_authority:false',
    'target_consistency:true',
    'preset_recall_allowed:false',
    'preset_store_allowed:false',
    'promotion_families:scene_preset|routing|mix_balance',
    'presets:preset_index|preset_name|preset_scope|target_type|target_label|preset_content_fingerprint|rollback_reference',
    'coverage:catalog_identity|scope|target|rollback_reference',
    'false_safety:live_write|mutates_console_state|preset_recall_allowed|preset_store_allowed',
  ]);
  assert.match(userPresetsContract.blockedReason, /physical readback fixture evidence is still missing/);
  assert.match(userPresetsContract.nextStep, /physical_readback:user_presets/);
  assert.match(userPresetsContract.nextStep, /physical_action:remediate_user_presets/);
  assert.match(userPresetsContract.nextStep, /families:scene_preset,routing,mix_balance/);
  assert.match(userPresetsContract.nextStep, /presets:preset_index,preset_name,preset_scope,target_type,target_label,preset_content_fingerprint,rollback_reference/);
  assert.match(userPresetsContract.nextStep, /coverage:catalog_identity,scope,target,rollback_reference/);
  assert.match(userPresetsContract.nextStep, /preset_recall_allowed:false/);
  assert.match(userPresetsContract.nextStep, /preset_store_allowed:false/);
  assert.match(userPresetsContract.nextStep, /live_authority:false/);
  assert.match(userPresetsContract.nextStep, /target_consistency:true/);

  const userPresetsIntake = rows.find(row => row.moduleId === 'user_presets_physical_readback_artifact_intake');
  assert.equal(userPresetsIntake.domain, 'read_surface_artifact_intake');
  assert.equal(userPresetsIntake.canWriteLive, false);
  assert.deepEqual(userPresetsIntake.liveWriteKinds, []);
  assert.equal(userPresetsIntake.status, 'operator_artifact_missing');
  assert.equal(userPresetsIntake.entrypoint, 'build_user_presets_physical_readback_artifact_intake');
  assert.equal(userPresetsIntake.module, 'remediate_user_presets');
  assert.equal(userPresetsIntake.tone, 'yellow');
  assert.deepEqual(userPresetsIntake.pipelineProfiles, [
    'available:true',
    'schema:user_presets_physical_readback.v1',
    'surface:user_presets',
    'action:remediate_user_presets',
    'artifacts:0',
    'valid:0',
    'invalid:0',
    'valid_presets:0',
    'candidate:false',
    'operator_review:true',
    'physical_gap_remains:true',
    'fixture_closure_allowed:false',
    'preset_recall_allowed:false',
    'preset_store_allowed:false',
    'live_authority:false',
    'target_consistency:true',
    'promotion_families:scene_preset|routing|mix_balance|performance_adaptation',
  ]);
  assert.match(userPresetsIntake.blockedReason, /operator_must_capture_user_presets_physical_readback_artifact/);
  assert.match(userPresetsIntake.nextStep, /surface:user_presets/);
  assert.match(userPresetsIntake.nextStep, /action:remediate_user_presets/);
  assert.match(userPresetsIntake.nextStep, /artifacts:0/);
  assert.match(userPresetsIntake.nextStep, /valid_presets:0/);
  assert.match(userPresetsIntake.nextStep, /operator_review:true/);
  assert.match(userPresetsIntake.nextStep, /fixture_closure_allowed:false/);
  assert.match(userPresetsIntake.nextStep, /preset_recall_allowed:false/);
  assert.match(userPresetsIntake.nextStep, /preset_store_allowed:false/);
  assert.match(userPresetsIntake.nextStep, /live_authority:false/);

  const dliveStateContract = rows.find(row => row.moduleId === 'dlive_state_physical_readback_contract');
  assert.equal(dliveStateContract.domain, 'read_surface_contract');
  assert.equal(dliveStateContract.canWriteLive, false);
  assert.deepEqual(dliveStateContract.liveWriteKinds, []);
  assert.equal(dliveStateContract.status, 'contract_ready_physical_fixture_missing');
  assert.equal(dliveStateContract.entrypoint, 'dlive_state_physical_readback_contract');
  assert.equal(dliveStateContract.module, 'remediate_dlive_state');
  assert.equal(dliveStateContract.tone, 'yellow');
  assert.deepEqual(dliveStateContract.pipelineProfiles, [
    'available:true',
    'schema:dlive_state_physical_readback.v1',
    'surface:dlive_state',
    'physical_readback:dlive_state',
    'physical_action:remediate_dlive_state',
    'physical_gap_remains:true',
    'live_authority:false',
    'target_consistency:true',
    'midi_send_allowed:false',
    'tcp_send_allowed:false',
    'dlive_write_allowed:false',
    'dlive_apply_allowed:false',
    'promotion_families:fader|gain|mute|routing|performance_adaptation',
    'channels:channel|cached_channel_name|fader_raw_14bit',
    'midi_state:midi_base_channel|raw_state_key_count',
    'parity:wing_surface_equivalent|mapping_status',
    'coverage:channel_identity|wing_parity_mapping',
    'false_safety:live_write|midi_send_allowed|tcp_send_allowed|dlive_write_allowed',
  ]);
  assert.match(dliveStateContract.blockedReason, /physical readback and WING parity fixture evidence is still missing/);
  assert.match(dliveStateContract.nextStep, /physical_readback:dlive_state/);
  assert.match(dliveStateContract.nextStep, /physical_action:remediate_dlive_state/);
  assert.match(dliveStateContract.nextStep, /families:fader,gain,mute,routing,performance_adaptation/);
  assert.match(dliveStateContract.nextStep, /channels:channel,cached_channel_name,fader_raw_14bit/);
  assert.match(dliveStateContract.nextStep, /midi_state:midi_base_channel,raw_state_key_count/);
  assert.match(dliveStateContract.nextStep, /parity:wing_surface_equivalent,mapping_status/);
  assert.match(dliveStateContract.nextStep, /coverage:channel_identity,wing_parity_mapping/);
  assert.match(dliveStateContract.nextStep, /midi_send_allowed:false/);
  assert.match(dliveStateContract.nextStep, /tcp_send_allowed:false/);
  assert.match(dliveStateContract.nextStep, /dlive_write_allowed:false/);
  assert.match(dliveStateContract.nextStep, /dlive_apply_allowed:false/);
  assert.match(dliveStateContract.nextStep, /live_authority:false/);
  assert.match(dliveStateContract.nextStep, /target_consistency:true/);

  const dliveStateIntake = rows.find(row => row.moduleId === 'dlive_state_physical_readback_artifact_intake');
  assert.equal(dliveStateIntake.domain, 'read_surface_artifact_intake');
  assert.equal(dliveStateIntake.canWriteLive, false);
  assert.deepEqual(dliveStateIntake.liveWriteKinds, []);
  assert.equal(dliveStateIntake.status, 'operator_artifact_missing');
  assert.equal(dliveStateIntake.entrypoint, 'build_dlive_state_physical_readback_artifact_intake');
  assert.equal(dliveStateIntake.module, 'remediate_dlive_state');
  assert.equal(dliveStateIntake.tone, 'yellow');
  assert.deepEqual(dliveStateIntake.pipelineProfiles, [
    'available:true',
    'schema:dlive_state_physical_readback.v1',
    'surface:dlive_state',
    'action:remediate_dlive_state',
    'artifacts:0',
    'valid:0',
    'invalid:0',
    'valid_channels:0',
    'candidate:false',
    'operator_review:true',
    'physical_gap_remains:true',
    'fixture_closure_allowed:false',
    'midi_send_allowed:false',
    'tcp_send_allowed:false',
    'dlive_write_allowed:false',
    'dlive_apply_allowed:false',
    'live_authority:false',
    'target_consistency:true',
    'promotion_families:fader|gain|mute|bus_group_matrix|routing|scene_preset|mix_balance|performance_adaptation',
  ]);
  assert.match(dliveStateIntake.blockedReason, /operator_must_capture_dlive_state_physical_readback_artifact/);
  assert.match(dliveStateIntake.nextStep, /surface:dlive_state/);
  assert.match(dliveStateIntake.nextStep, /action:remediate_dlive_state/);
  assert.match(dliveStateIntake.nextStep, /artifacts:0/);
  assert.match(dliveStateIntake.nextStep, /valid_channels:0/);
  assert.match(dliveStateIntake.nextStep, /operator_review:true/);
  assert.match(dliveStateIntake.nextStep, /fixture_closure_allowed:false/);
  assert.match(dliveStateIntake.nextStep, /midi_send_allowed:false/);
  assert.match(dliveStateIntake.nextStep, /tcp_send_allowed:false/);
  assert.match(dliveStateIntake.nextStep, /dlive_write_allowed:false/);
  assert.match(dliveStateIntake.nextStep, /dlive_apply_allowed:false/);
  assert.match(dliveStateIntake.nextStep, /live_authority:false/);

  const artifactReviewQueue = rows.find(row => row.moduleId === 'read_surface_physical_readback_artifact_review_queue');
  assert.equal(artifactReviewQueue.domain, 'read_surface_artifact_review_queue');
  assert.equal(artifactReviewQueue.canWriteLive, false);
  assert.deepEqual(artifactReviewQueue.liveWriteKinds, []);
  assert.equal(artifactReviewQueue.status, 'operator_artifact_capture_required');
  assert.equal(artifactReviewQueue.entrypoint, 'build_read_surface_physical_readback_artifact_review_queue');
  assert.equal(artifactReviewQueue.module, 'operator_must_capture_missing_physical_readback_artifacts');
  assert.equal(artifactReviewQueue.tone, 'yellow');
  assert.deepEqual(artifactReviewQueue.pipelineProfiles, [
    'available:true',
    'status:operator_artifact_capture_required',
    'surfaces:10',
    'available_intakes:10',
    'artifacts:0',
    'valid:0',
    'invalid:0',
    'review_candidates:0',
    'missing:10',
    'unsafe_authority:0',
    'target_inconsistent:0',
    'review_available:false',
    'operator_review:true',
    'physical_gap_remains:true',
    'fixture_closure_allowed:false',
    'live_authority:false',
    'full_autonomous_ready:false',
    'missing_surfaces:raw_mixer_state_cache|channel_identity_and_basic_controls|dlive_state',
    'methods:raw_mixer_state_physical_readback_artifact_intake|channel_basic_controls_physical_readback_artifact_intake|dlive_state_physical_readback_artifact_intake',
  ]);
  assert.match(artifactReviewQueue.blockedReason, /operator_must_capture_missing_physical_readback_artifacts/);
  assert.match(artifactReviewQueue.nextStep, /queue:operator_artifact_capture_required/);
  assert.match(artifactReviewQueue.nextStep, /surfaces:10/);
  assert.match(artifactReviewQueue.nextStep, /available_intakes:10/);
  assert.match(artifactReviewQueue.nextStep, /fixture_closure_allowed:false/);
  assert.match(artifactReviewQueue.nextStep, /live_authority:false/);
  assert.match(artifactReviewQueue.nextStep, /full_autonomous_ready:false/);

  const fixtureClosurePlan = rows.find(row => row.moduleId === 'read_surface_physical_readback_fixture_closure_plan');
  assert.equal(fixtureClosurePlan.domain, 'read_surface_fixture_closure_plan');
  assert.equal(fixtureClosurePlan.canWriteLive, false);
  assert.deepEqual(fixtureClosurePlan.liveWriteKinds, []);
  assert.equal(fixtureClosurePlan.status, 'fixture_closure_not_ready');
  assert.equal(fixtureClosurePlan.entrypoint, 'build_read_surface_physical_readback_fixture_closure_plan');
  assert.equal(fixtureClosurePlan.module, 'accepted_physical_readback_candidate_required_before_fixture_closure');
  assert.equal(fixtureClosurePlan.tone, 'yellow');
  assert.deepEqual(fixtureClosurePlan.pipelineProfiles, [
    'available:true',
    'status:fixture_closure_not_ready',
    'review_candidates:0',
    'acceptance_records:0',
    'accepted:0',
    'rejected:0',
    'invalid:0',
    'unreviewed:0',
    'closure_candidates:0',
    'manual_candidate_available:false',
    'manual_closure_required:true',
    'operator_review:true',
    'physical_gap_remains:true',
    'fixture_update_allowed:false',
    'fixture_closure_allowed:false',
    'live_authority:false',
    'full_autonomous_ready:false',
  ]);
  assert.match(fixtureClosurePlan.blockedReason, /accepted_physical_readback_candidate_required_before_fixture_closure/);
  assert.match(fixtureClosurePlan.nextStep, /plan:fixture_closure_not_ready/);
  assert.match(fixtureClosurePlan.nextStep, /closure_candidates:0/);
  assert.match(fixtureClosurePlan.nextStep, /manual_closure_required:true/);
  assert.match(fixtureClosurePlan.nextStep, /physical_gap_remains:true/);
  assert.match(fixtureClosurePlan.nextStep, /fixture_update_allowed:false/);
  assert.match(fixtureClosurePlan.nextStep, /fixture_closure_allowed:false/);
  assert.match(fixtureClosurePlan.nextStep, /live_authority:false/);
  assert.match(fixtureClosurePlan.nextStep, /full_autonomous_ready:false/);

  const readRemediation = rows.find(row => row.moduleId === 'read_surface_remediation_plan');
  assert.equal(readRemediation.domain, 'read_surface_remediation');
  assert.equal(readRemediation.canWriteLive, false);
  assert.deepEqual(readRemediation.liveWriteKinds, []);
  assert.equal(readRemediation.status, 'read_remediation_required');
  assert.equal(readRemediation.module, 'remediate_raw_mixer_state_cache');
  assert.deepEqual(readRemediation.pipelineProfiles, [
    'available:true',
    'actions:10',
    'incomplete:10',
    'physical_gaps:2',
    'next_physical_readback:raw_mixer_state_cache',
    'next_action:remediate_raw_mixer_state_cache',
  ]);
  assert.match(readRemediation.blockedReason, /physical_readback_fixtures_incomplete/);
  assert.match(readRemediation.nextStep, /next_physical_readback:raw_mixer_state_cache/);
  assert.match(readRemediation.nextStep, /next_action:remediate_raw_mixer_state_cache/);

  const faderOverlay = rows.find(row => row.moduleId === 'autonomy_evidence:fader_supervised_promotion');
  assert.equal(faderOverlay.domain, 'autonomy_evidence_overlay');
  assert.equal(faderOverlay.canWriteLive, false);
  assert.deepEqual(faderOverlay.liveWriteKinds, []);
  assert.equal(faderOverlay.status, 'no_current_evidence');
  assert.equal(faderOverlay.entrypoint, 'fader');
  assert.equal(faderOverlay.module, 'fader_supervised_promotion');
  assert.equal(faderOverlay.tone, 'yellow');
  assert.match(faderOverlay.blockedReason, /proof_artifact_missing/);
  assert.match(faderOverlay.nextStep, /physical supervised fader proof/);

  const gainOverlay = rows.find(row => row.moduleId === 'autonomy_evidence:gain_trim_supervised_promotion');
  assert.equal(gainOverlay.domain, 'autonomy_evidence_overlay');
  assert.equal(gainOverlay.canWriteLive, false);
  assert.deepEqual(gainOverlay.liveWriteKinds, []);
  assert.equal(gainOverlay.status, 'dry_run_only');
  assert.equal(gainOverlay.entrypoint, 'gain');
  assert.equal(gainOverlay.module, 'gain_trim_supervised_promotion');
  assert.equal(gainOverlay.tone, 'blue');
  assert.match(gainOverlay.nextStep, /gain-proof-dry-run/);

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
  assert.equal(
    evidenceGap.module,
    'bus_group_matrix_supervised_promotion, mute_supervised_promotion',
  );
  assert.match(evidenceGap.nextStep, /bus_group_assignment_cache_freshness/);
  assert.match(evidenceGap.nextStep, /main_bus_true_peak_headroom/);
  assert.match(evidenceGap.nextStep, /mute_owner_intent_evidence/);
  assert.match(evidenceGap.nextStep, /hardware_readback_rollback_and_emergency_proof/);
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

test('adaptModuleStatusRecords marks stale supervised proof evidence separately from available evidence', async () => {
  const { adaptModuleStatusRecords } = await importTypeScript('services/operatorAdapters.ts');

  const rows = adaptModuleStatusRecords({
    supervised_proof_evidence: {
      available: true,
      report_path: 'artifacts/no_wing_release_check_current_cycle/stale/supervised_proof_artifact_report.json',
      summary: {
        latest_report_status: 'ok',
        latest_report_age_sec: 90000,
        latest_report_stale: true,
        stale_after_sec: 86400,
        artifact_count: 1,
        invalid_summary_count: 0,
        overlay_status: 'dry_run_only',
      },
    },
  });

  assert.equal(rows.length, 24);
  assert.equal(rows[0].moduleId, 'supervised_proof_evidence');
  assert.equal(rows[0].status, 'artifact_stale');
  assert.equal(rows[0].tone, 'yellow');
  assert.equal(rows[0].canWriteLive, false);
  assert.deepEqual(rows[0].liveWriteKinds, []);
  assert.match(rows[0].blockedReason, /stale/);
  assert.match(rows[0].nextStep, /stale:true/);
  assert.deepEqual(rows[0].pipelineProfiles, [
    'available:true',
    'overlay:dry_run_only',
    'artifacts:1',
    'invalid:0',
    'stale:true',
    'read_surface_quality:false',
    'read_surface_gaps:0',
    'read_surface_gap_ids:',
    'next_readback_target:',
    'source_next_readback_target:',
    'next_readback_gate:false',
    'capture_plan_available:false',
    'capture_plan_status:missing',
    'capture_plan_steps:0',
    'capture_plan_next_surface:',
    'capture_plan_execution_allowed:false',
    'capture_plan_target_gate:false',
    'fixture_tests:0',
    'contract_gap:false',
    'raw_contract_available:false',
    'raw_contract_status:missing',
    'raw_contract_schema:',
    'raw_contract_gap:false',
    'raw_contract_live_authority:false',
    'raw_contract_target_gate:false',
    'raw_intake_available:false',
    'raw_intake_status:missing',
    'raw_intake_artifacts:0',
    'raw_intake_valid:0',
    'raw_intake_invalid:0',
    'raw_intake_candidate:false',
    'raw_intake_gap:false',
    'raw_intake_closure_allowed:false',
    'raw_intake_live_authority:false',
    'raw_intake_target_gate:false',
    'channel_contract_available:false',
    'channel_contract_status:missing',
    'channel_contract_schema:',
    'channel_contract_gap:false',
    'channel_contract_live_authority:false',
    'channel_contract_target_gate:false',
    'channel_intake_available:false',
    'channel_intake_status:missing',
    'channel_intake_artifacts:0',
    'channel_intake_valid:0',
    'channel_intake_invalid:0',
    'channel_intake_valid_channels:0',
    'channel_intake_candidate:false',
    'channel_intake_gap:false',
    'channel_intake_closure_allowed:false',
    'channel_intake_live_authority:false',
    'channel_intake_target_gate:false',
    'processing_contract_available:false',
    'processing_contract_status:missing',
    'processing_contract_schema:',
    'processing_contract_gap:false',
    'processing_contract_live_authority:false',
    'processing_contract_target_gate:false',
    'processing_intake_available:false',
    'processing_intake_status:missing',
    'processing_intake_artifacts:0',
    'processing_intake_valid:0',
    'processing_intake_invalid:0',
    'processing_intake_valid_channels:0',
    'processing_intake_candidate:false',
    'processing_intake_gap:false',
    'processing_intake_closure_allowed:false',
    'processing_intake_live_authority:false',
    'processing_intake_target_gate:false',
    'bus_contract_available:false',
    'bus_contract_status:missing',
    'bus_contract_schema:',
    'bus_contract_gap:false',
    'bus_contract_live_authority:false',
    'bus_contract_target_gate:false',
    'bus_intake_available:false',
    'bus_intake_status:missing',
    'bus_intake_artifacts:0',
    'bus_intake_valid:0',
    'bus_intake_invalid:0',
    'bus_intake_valid_buses:0',
    'bus_intake_valid_mains:0',
    'bus_intake_valid_dcas:0',
    'bus_intake_candidate:false',
    'bus_intake_gap:false',
    'bus_intake_closure_allowed:false',
    'bus_intake_live_authority:false',
    'bus_intake_target_gate:false',
    'matrix_contract_available:false',
    'matrix_contract_status:missing',
    'matrix_contract_schema:',
    'matrix_contract_gap:false',
    'matrix_contract_live_authority:false',
    'matrix_contract_target_gate:false',
    'matrix_intake_available:false',
    'matrix_intake_status:missing',
    'matrix_intake_artifacts:0',
    'matrix_intake_valid:0',
    'matrix_intake_invalid:0',
    'matrix_intake_valid_matrices:0',
    'matrix_intake_candidate:false',
    'matrix_intake_gap:false',
    'matrix_intake_closure_allowed:false',
    'matrix_intake_live_authority:false',
    'matrix_intake_target_gate:false',
    'fx_contract_available:false',
    'fx_contract_status:missing',
    'fx_contract_schema:',
    'fx_contract_gap:false',
    'fx_contract_live_authority:false',
    'fx_contract_target_gate:false',
    'fx_intake_available:false',
    'fx_intake_status:missing',
    'fx_intake_artifacts:0',
    'fx_intake_valid:0',
    'fx_intake_invalid:0',
    'fx_intake_valid_slots:0',
    'fx_intake_valid_inserts:0',
    'fx_intake_candidate:false',
    'fx_intake_gap:false',
    'fx_intake_closure_allowed:false',
    'fx_intake_live_authority:false',
    'fx_intake_target_gate:false',
    'routing_contract_available:false',
    'routing_contract_status:missing',
    'routing_contract_schema:',
    'routing_contract_gap:false',
    'routing_contract_live_authority:false',
    'routing_contract_target_gate:false',
    'routing_intake_available:false',
    'routing_intake_status:missing',
    'routing_intake_artifacts:0',
    'routing_intake_valid:0',
    'routing_intake_invalid:0',
    'routing_intake_valid_static_routes:0',
    'routing_intake_valid_bus_sends:0',
    'routing_intake_valid_main_sends:0',
    'routing_intake_valid_matrix_sends:0',
    'routing_intake_valid_output_routes:0',
    'routing_intake_candidate:false',
    'routing_intake_gap:false',
    'routing_intake_closure_allowed:false',
    'routing_intake_live_authority:false',
    'routing_intake_target_gate:false',
    'scenes_contract_available:false',
    'scenes_contract_status:missing',
    'scenes_contract_schema:',
    'scenes_contract_gap:false',
    'scenes_contract_live_authority:false',
    'scenes_contract_target_gate:false',
    'scenes_intake_available:false',
    'scenes_intake_status:missing',
    'scenes_intake_artifacts:0',
    'scenes_intake_valid:0',
    'scenes_intake_invalid:0',
    'scenes_intake_valid_snapshots:0',
    'scenes_intake_candidate:false',
    'scenes_intake_gap:false',
    'scenes_intake_closure_allowed:false',
    'scenes_intake_scene_load_allowed:false',
    'scenes_intake_scene_save_allowed:false',
    'scenes_intake_live_authority:false',
    'scenes_intake_target_gate:false',
    'presets_contract_available:false',
    'presets_contract_status:missing',
    'presets_contract_schema:',
    'presets_contract_gap:false',
    'presets_contract_live_authority:false',
    'presets_contract_target_gate:false',
    'presets_intake_available:false',
    'presets_intake_status:missing',
    'presets_intake_artifacts:0',
    'presets_intake_valid:0',
    'presets_intake_invalid:0',
    'presets_intake_valid_presets:0',
    'presets_intake_candidate:false',
    'presets_intake_gap:false',
    'presets_intake_closure_allowed:false',
    'presets_intake_preset_recall_allowed:false',
    'presets_intake_preset_store_allowed:false',
    'presets_intake_live_authority:false',
    'presets_intake_target_gate:false',
    'dlive_contract_available:false',
    'dlive_contract_status:missing',
    'dlive_contract_schema:',
    'dlive_contract_gap:false',
    'dlive_contract_live_authority:false',
    'dlive_contract_target_gate:false',
    'dlive_intake_available:false',
    'dlive_intake_status:missing',
    'dlive_intake_artifacts:0',
    'dlive_intake_valid:0',
    'dlive_intake_invalid:0',
    'dlive_intake_valid_channels:0',
    'dlive_intake_candidate:false',
    'dlive_intake_gap:false',
    'dlive_intake_closure_allowed:false',
    'dlive_intake_midi_send_allowed:false',
    'dlive_intake_tcp_send_allowed:false',
    'dlive_intake_dlive_write_allowed:false',
    'dlive_intake_dlive_apply_allowed:false',
    'dlive_intake_live_authority:false',
    'dlive_intake_target_gate:false',
    'artifact_queue_available:false',
    'artifact_queue_status:missing',
    'artifact_queue_surfaces:0',
    'artifact_queue_available_intakes:0',
    'artifact_queue_artifacts:0',
    'artifact_queue_valid:0',
    'artifact_queue_invalid:0',
    'artifact_queue_review_candidates:0',
    'artifact_queue_missing:0',
    'artifact_queue_unsafe:0',
    'artifact_queue_target_inconsistent:0',
    'artifact_queue_review_available:false',
    'artifact_queue_closure_allowed:false',
    'artifact_queue_live_authority:false',
    'artifact_queue_full_auto:false',
    'fixture_closure_plan_available:false',
    'fixture_closure_plan_status:missing',
    'fixture_closure_plan_review_candidates:0',
    'fixture_closure_plan_acceptance_records:0',
    'fixture_closure_plan_accepted:0',
    'fixture_closure_plan_rejected:0',
    'fixture_closure_plan_invalid:0',
    'fixture_closure_plan_unreviewed:0',
    'fixture_closure_plan_candidates:0',
    'fixture_closure_plan_candidate_available:false',
    'fixture_update_allowed:false',
    'fixture_closure_allowed:false',
    'fixture_closure_plan_gap:false',
    'fixture_closure_plan_live_authority:false',
    'fixture_closure_plan_full_auto:false',
  ]);

  assert.equal(rows[1].moduleId, 'read_surface_physical_readback_capture_plan');
  assert.equal(rows[1].domain, 'read_surface_capture_plan');
  assert.equal(rows[1].status, 'plan_missing');
  assert.equal(rows[1].canWriteLive, false);
  assert.deepEqual(rows[1].liveWriteKinds, []);
  assert.match(rows[1].blockedReason, /no current blockers/);
  assert.match(rows[1].nextStep, /capture-plan validation/);

  assert.equal(rows[2].moduleId, 'raw_mixer_state_physical_readback_contract');
  assert.equal(rows[2].domain, 'read_surface_contract');
  assert.equal(rows[2].status, 'contract_missing');
  assert.equal(rows[2].canWriteLive, false);
  assert.deepEqual(rows[2].liveWriteKinds, []);
  assert.match(rows[2].blockedReason, /contract is missing/);

  assert.equal(rows[3].moduleId, 'raw_mixer_state_physical_readback_artifact_intake');
  assert.equal(rows[3].domain, 'read_surface_artifact_intake');
  assert.equal(rows[3].status, 'intake_missing');
  assert.equal(rows[3].canWriteLive, false);
  assert.deepEqual(rows[3].liveWriteKinds, []);
  assert.match(rows[3].blockedReason, /artifact intake is missing/);

  assert.equal(rows[4].moduleId, 'channel_basic_controls_physical_readback_contract');
  assert.equal(rows[4].domain, 'read_surface_contract');
  assert.equal(rows[4].status, 'contract_missing');
  assert.equal(rows[4].canWriteLive, false);
  assert.deepEqual(rows[4].liveWriteKinds, []);
  assert.match(rows[4].blockedReason, /contract is missing/);

  assert.equal(rows[5].moduleId, 'channel_basic_controls_physical_readback_artifact_intake');
  assert.equal(rows[5].domain, 'read_surface_artifact_intake');
  assert.equal(rows[5].status, 'intake_missing');
  assert.equal(rows[5].canWriteLive, false);
  assert.deepEqual(rows[5].liveWriteKinds, []);
  assert.match(rows[5].blockedReason, /artifact intake is missing/);

  assert.equal(rows[6].moduleId, 'channel_processing_physical_readback_contract');
  assert.equal(rows[6].domain, 'read_surface_contract');
  assert.equal(rows[6].status, 'contract_missing');
  assert.equal(rows[6].canWriteLive, false);
  assert.deepEqual(rows[6].liveWriteKinds, []);
  assert.match(rows[6].blockedReason, /contract is missing/);

  assert.equal(rows[7].moduleId, 'channel_processing_physical_readback_artifact_intake');
  assert.equal(rows[7].domain, 'read_surface_artifact_intake');
  assert.equal(rows[7].status, 'intake_missing');
  assert.equal(rows[7].canWriteLive, false);
  assert.deepEqual(rows[7].liveWriteKinds, []);
  assert.match(rows[7].blockedReason, /artifact intake is missing/);

  assert.equal(rows[8].moduleId, 'bus_group_main_dca_physical_readback_contract');
  assert.equal(rows[8].domain, 'read_surface_contract');
  assert.equal(rows[8].status, 'contract_missing');
  assert.equal(rows[8].canWriteLive, false);
  assert.deepEqual(rows[8].liveWriteKinds, []);
  assert.match(rows[8].blockedReason, /contract is missing/);

  assert.equal(rows[9].moduleId, 'bus_group_main_dca_physical_readback_artifact_intake');
  assert.equal(rows[9].domain, 'read_surface_artifact_intake');
  assert.equal(rows[9].status, 'intake_missing');
  assert.equal(rows[9].canWriteLive, false);
  assert.deepEqual(rows[9].liveWriteKinds, []);
  assert.match(rows[9].blockedReason, /artifact intake is missing/);

  assert.equal(rows[10].moduleId, 'matrix_state_physical_readback_contract');
  assert.equal(rows[10].domain, 'read_surface_contract');
  assert.equal(rows[10].status, 'contract_missing');
  assert.equal(rows[10].canWriteLive, false);
  assert.deepEqual(rows[10].liveWriteKinds, []);
  assert.match(rows[10].blockedReason, /contract is missing/);

  assert.equal(rows[11].moduleId, 'matrix_state_physical_readback_artifact_intake');
  assert.equal(rows[11].domain, 'read_surface_artifact_intake');
  assert.equal(rows[11].status, 'intake_missing');
  assert.equal(rows[11].canWriteLive, false);
  assert.deepEqual(rows[11].liveWriteKinds, []);
  assert.match(rows[11].blockedReason, /artifact intake is missing/);

  assert.equal(rows[12].moduleId, 'fx_slots_inserts_physical_readback_contract');
  assert.equal(rows[12].domain, 'read_surface_contract');
  assert.equal(rows[12].status, 'contract_missing');
  assert.equal(rows[12].canWriteLive, false);
  assert.deepEqual(rows[12].liveWriteKinds, []);
  assert.match(rows[12].blockedReason, /contract is missing/);

  assert.equal(rows[13].moduleId, 'fx_slots_inserts_physical_readback_artifact_intake');
  assert.equal(rows[13].domain, 'read_surface_artifact_intake');
  assert.equal(rows[13].status, 'intake_missing');
  assert.equal(rows[13].canWriteLive, false);
  assert.deepEqual(rows[13].liveWriteKinds, []);
  assert.match(rows[13].blockedReason, /artifact intake is missing/);

  assert.equal(rows[14].moduleId, 'routing_sends_physical_readback_contract');
  assert.equal(rows[14].domain, 'read_surface_contract');
  assert.equal(rows[14].status, 'contract_missing');
  assert.equal(rows[14].canWriteLive, false);
  assert.deepEqual(rows[14].liveWriteKinds, []);
  assert.match(rows[14].blockedReason, /contract is missing/);

  assert.equal(rows[15].moduleId, 'routing_sends_physical_readback_artifact_intake');
  assert.equal(rows[15].domain, 'read_surface_artifact_intake');
  assert.equal(rows[15].status, 'intake_missing');
  assert.equal(rows[15].canWriteLive, false);
  assert.deepEqual(rows[15].liveWriteKinds, []);
  assert.match(rows[15].blockedReason, /artifact intake is missing/);

  assert.equal(rows[16].moduleId, 'snapshots_scenes_physical_readback_contract');
  assert.equal(rows[16].domain, 'read_surface_contract');
  assert.equal(rows[16].status, 'contract_missing');
  assert.equal(rows[16].canWriteLive, false);
  assert.deepEqual(rows[16].liveWriteKinds, []);
  assert.match(rows[16].blockedReason, /contract is missing/);

  assert.equal(rows[17].moduleId, 'snapshots_scenes_physical_readback_artifact_intake');
  assert.equal(rows[17].domain, 'read_surface_artifact_intake');
  assert.equal(rows[17].status, 'intake_missing');
  assert.equal(rows[17].canWriteLive, false);
  assert.deepEqual(rows[17].liveWriteKinds, []);
  assert.match(rows[17].blockedReason, /artifact intake is missing/);

  assert.equal(rows[18].moduleId, 'user_presets_physical_readback_contract');
  assert.equal(rows[18].domain, 'read_surface_contract');
  assert.equal(rows[18].status, 'contract_missing');
  assert.equal(rows[18].canWriteLive, false);
  assert.deepEqual(rows[18].liveWriteKinds, []);
  assert.match(rows[18].blockedReason, /contract is missing/);

  assert.equal(rows[19].moduleId, 'user_presets_physical_readback_artifact_intake');
  assert.equal(rows[19].domain, 'read_surface_artifact_intake');
  assert.equal(rows[19].status, 'intake_missing');
  assert.equal(rows[19].canWriteLive, false);
  assert.deepEqual(rows[19].liveWriteKinds, []);
  assert.match(rows[19].blockedReason, /artifact intake is missing/);

  assert.equal(rows[20].moduleId, 'dlive_state_physical_readback_contract');
  assert.equal(rows[20].domain, 'read_surface_contract');
  assert.equal(rows[20].status, 'contract_missing');
  assert.equal(rows[20].canWriteLive, false);
  assert.deepEqual(rows[20].liveWriteKinds, []);
  assert.match(rows[20].blockedReason, /contract is missing/);

  assert.equal(rows[21].moduleId, 'dlive_state_physical_readback_artifact_intake');
  assert.equal(rows[21].domain, 'read_surface_artifact_intake');
  assert.equal(rows[21].status, 'intake_missing');
  assert.equal(rows[21].canWriteLive, false);
  assert.deepEqual(rows[21].liveWriteKinds, []);
  assert.match(rows[21].blockedReason, /artifact intake is missing/);

  assert.equal(rows[22].moduleId, 'read_surface_physical_readback_artifact_review_queue');
  assert.equal(rows[22].domain, 'read_surface_artifact_review_queue');
  assert.equal(rows[22].status, 'queue_missing');
  assert.equal(rows[22].canWriteLive, false);
  assert.deepEqual(rows[22].liveWriteKinds, []);
  assert.match(rows[22].blockedReason, /review queue is missing/);

  assert.equal(rows[23].moduleId, 'read_surface_physical_readback_fixture_closure_plan');
  assert.equal(rows[23].domain, 'read_surface_fixture_closure_plan');
  assert.equal(rows[23].status, 'closure_plan_missing');
  assert.equal(rows[23].canWriteLive, false);
  assert.deepEqual(rows[23].liveWriteKinds, []);
  assert.match(rows[23].blockedReason, /closure plan is missing/);
});

test('adaptModuleStatusRecords surfaces project goal capability matrix and progress summary', async () => {
  const { adaptModuleStatusRecords } = await importTypeScript('services/operatorAdapters.ts');

  const rows = adaptModuleStatusRecords({
    project_goal_capability_matrix: {
      status: 'available',
      rows: [
        {
          id: 'cg_vocal_trim',
          label: 'Safe vocal trim',
          goal_area: 'input_gain',
          status: 'limited_available',
          autonomy_mode_ceiling: 'supervised_live_candidate',
          currently_live_writable: true,
          blocker: '',
          method_ids: ['input_trim', 'noise_gate'],
          support_paths: ['backend/safegain.py'],
          current_live_write_kinds: ['gain'],
          target_console_surfaces: ['channel_strip'],
          next_step: 'Capture supervised proof for current vocal source.',
        },
        {
          id: 'cg_room_mic_eq',
          label: 'Room mic equalization',
          goal_area: 'eq',
          status: 'blocked',
          autonomy_mode_ceiling: 'full_autonomous',
          currently_live_writable: false,
          blocker: 'No validated eq method available yet.',
          method_ids: ['eq'],
          support_paths: ['backend/smart_eq.py', 'backend/eq_router.py'],
          current_live_write_kinds: ['eq'],
          target_console_surfaces: ['channel_eq'],
          next_step: 'Add physical readback-backed method for room mic EQ.',
        },
      ],
      summary: {
        row_count: 2,
        missing_capability_ids: ['cg_room_mics'],
        blocked_live_capability_ids: ['cg_room_mic_eq'],
        currently_live_writable_capability_ids: ['cg_vocal_trim'],
        status_counts: {
          limited_available: 1,
          blocked: 1,
        },
        next_global_step: 'Finish supervised proof work before live rollout.',
      },
    },
    project_goal_capability_progress_snapshot: {
      status: 'limited_available',
      next_steps: [
        {
          capability_id: 'cg_room_mic_eq',
          goal_area: 'eq',
          autonomy_mode_ceiling: 'full_autonomous',
          next_step: 'Resolve EQ method and proof gap.',
        },
      ],
      summary: {
        row_count: 1,
        missing_capability_ids: ['cg_8'],
        blocked_live_capability_ids: ['cg_room_mic_eq'],
        proposal_only_capability_ids: ['cg_gate'],
        supervised_narrow_live_candidates: ['cg_vocal_pan'],
        goal_areas_with_blockers: ['eq', 'input_gain'],
        next_global_step: 'Collect proofs in progress order.',
      },
    },
    project_goal_capability_objective_alignment_snapshot: {
      status: 'ok',
      summary: {
        required_row_count: 2,
        present_row_count: 1,
        autonomy_readiness_score: 50,
        autonomy_readiness_ready_row_count: 1,
        autonomy_readiness_gap_row_count: 1,
        missing_row_count: 1,
        proposal_only_row_count: 1,
        blocked_live_row_count: 1,
        currently_live_writable_row_count: 0,
        proposal_only_row_ids: ['cg_gate'],
        supervised_candidate_row_ids: ['cg_vocal_pan'],
        supervised_candidate_row_count: 1,
        blocked_live_rows: ['cg_room_mic_eq'],
        currently_live_writable_rows: [],
        blocked_live_row_actions: ['cg_room_mic_eq: Resolve EQ method and proof gap.'],
        missing_rows: ['cg_room_mics'],
        full_autonomous_status: 'blocked_live',
      },
    },
  });

  const capabilityRows = rows.filter((row) => row.domain === 'project_goal_capability' && row.moduleId.startsWith('project_goal_capability:'));
  const matrixSummary = rows.find((row) => row.moduleId === 'project_goal_capability_summary');
  const progressSummary = rows.find((row) => row.moduleId === 'project_goal_capability_progress_summary');

  assert.equal(capabilityRows.length, 2);
  assert.equal(capabilityRows[0].moduleId, 'project_goal_capability:cg_vocal_trim');
  assert.equal(capabilityRows[0].status, 'limited_available');
  assert.equal(capabilityRows[0].uiSurface, 'Project goal capability matrix');
  assert.equal(capabilityRows[0].canWriteLive, true);
  assert.equal(capabilityRows[0].liveWriteKinds.includes('gain'), true);
  assert.equal(capabilityRows[0].pipelineProfiles.includes('area:input_gain'), true);
  assert.equal(capabilityRows[0].pipelineProfiles.includes('blockers:0'), true);
  assert.equal(capabilityRows[0].pipelineProfiles.includes('target_surfaces:1'), true);
  assert.equal(capabilityRows[0].pipelineProfiles.includes('methods:2'), true);
  assert.equal(capabilityRows[0].pipelineProfiles.includes('status:limited_available=1'), true);

  const eqRow = rows.find((row) => row.moduleId === 'project_goal_capability:cg_room_mic_eq');
  assert.equal(eqRow?.status, 'blocked');
  assert.equal(eqRow?.tone, 'red');
  assert.equal(eqRow?.canWriteLive, false);
  assert.equal(eqRow?.pipelineProfiles.includes('blockers:1'), true);

  assert.equal(matrixSummary?.domain, 'project_goal_capability');
  assert.equal(matrixSummary?.module, 'project_goal_capability_matrix.row_count:2');
  assert.equal(matrixSummary?.pipelineProfiles.includes('row_count:2'), true);
  assert.equal(matrixSummary?.pipelineProfiles.includes('blocked_live_count:1'), true);
  assert.equal(matrixSummary?.pipelineProfiles.includes('missing_capability_count:1'), true);
  assert.equal(matrixSummary?.tone, 'red');
  assert.equal(matrixSummary?.status, 'available');

  assert.equal(progressSummary?.domain, 'project_goal_capability');
  assert.equal(progressSummary?.status, 'limited_available');
  assert.equal(progressSummary?.module, 'project_goal_progress_snapshot.row_count:1');
  assert.equal(progressSummary?.pipelineProfiles.includes('blocked_live:1'), true);
  assert.equal(progressSummary?.pipelineProfiles.includes('proposal_only:1'), true);
  assert.equal(progressSummary?.pipelineProfiles.includes('current_supervised_candidates:1'), true);
  assert.equal(progressSummary?.pipelineProfiles.includes('goal_areas_with_blockers:eq,input_gain'), true);
  assert.equal(progressSummary?.tone, 'red');
  assert.equal(progressSummary?.nextStep.includes('cg_room_mic_eq (eq/full_autonomous): Resolve EQ method and proof gap.'), true);

  const objectiveAlignmentSummary = rows.find(
    (row) => row.moduleId === 'project_goal_capability_objective_alignment_summary',
  );
  assert.equal(objectiveAlignmentSummary?.module, 'required:2');
  assert.equal(objectiveAlignmentSummary?.pipelineProfiles.includes('readiness_score:50'), true);
  assert.equal(objectiveAlignmentSummary?.pipelineProfiles.includes('readiness_ready_row_count:1'), true);
  assert.equal(objectiveAlignmentSummary?.pipelineProfiles.includes('readiness_gap_row_count:1'), true);
  assert.equal(objectiveAlignmentSummary?.pipelineProfiles.includes('missing:1'), true);
  assert.equal(objectiveAlignmentSummary?.pipelineProfiles.includes('proposal_only:1'), true);
  assert.equal(objectiveAlignmentSummary?.pipelineProfiles.includes('blocked_live_rows:1'), true);
  assert.equal(objectiveAlignmentSummary?.pipelineProfiles.includes('currently_live_writable:0'), true);
  assert.equal(objectiveAlignmentSummary?.pipelineProfiles.includes('supervised_candidates:1'), true);
  assert.equal(objectiveAlignmentSummary?.pipelineProfiles.includes('missing_row_ids:cg_room_mics'), true);
  assert.equal(objectiveAlignmentSummary?.pipelineProfiles.includes('proposal_only_row_ids:cg_gate'), true);
  assert.equal(objectiveAlignmentSummary?.pipelineProfiles.includes('supervised_candidate_row_ids:cg_vocal_pan'), true);
  assert.equal(objectiveAlignmentSummary?.pipelineProfiles.includes('blocked_live_row_ids:cg_room_mic_eq'), true);
  assert.equal(objectiveAlignmentSummary?.pipelineProfiles.includes('blocked_live_actions:cg_room_mic_eq: Resolve EQ method and proof gap.'), true);
  assert.equal(objectiveAlignmentSummary?.pipelineProfiles.includes('currently_live_writable_ids:'), true);
  assert.equal(objectiveAlignmentSummary?.pipelineProfiles.includes('full_autonomous_status:blocked_live'), true);
  assert.equal(objectiveAlignmentSummary?.blockedReason?.includes('Missing objective capabilities: cg_room_mics'), true);
  assert.equal(objectiveAlignmentSummary?.tone, 'red');
  assert.equal(objectiveAlignmentSummary?.nextStep.includes('full_autonomous_status:blocked_live'), true);
  assert.equal(objectiveAlignmentSummary?.nextStep.includes('missing_rows:cg_room_mics'), true);
  assert.equal(objectiveAlignmentSummary?.nextStep.includes('blocked_live_rows:cg_room_mic_eq'), true);
  assert.equal(objectiveAlignmentSummary?.nextStep.includes('blocked_live_actions:cg_room_mic_eq: Resolve EQ method and proof gap.'), true);
});

test('adaptSafetyMonitor maps new goal-capability objective readiness hotspots', async () => {
  const { adaptSafetyMonitor, mockBackendState } = await importTypeScript('services/operatorAdapters.ts');

  const safetyMonitor = adaptSafetyMonitor({
    operator_mode: { mode: 'supervised' },
    live_write_policy: { apply_path: '/console', policy: 'fader_gain_supervised_review' },
    manual_write_supervision: {
      emergency_stopped: false,
      reset_required: true,
      approved_live_write_kinds: ['gain'],
    },
    proof: {
      last_readback: { status: 'verified' },
      goal_capability_objective_readiness_goal_area_priorities: [
        { goal_area: 'eq', gap_row_count: 1, readiness_score: 66 },
      ],
      goal_capability_objective_readiness_by_goal_area: {
        eq: {
          goal_area: 'eq',
          ready_row_count: 2,
          gap_row_count: 1,
          readiness_score: 66,
        },
      },
      goal_capability_objective_readiness_hotspots: [
        {
          goal_area: 'eq',
          gap_row_count: 1,
          blocker_row_ids: ['cg_room_mic_eq'],
        },
      ],
      last_proof_status: 'applied',
    },
    summary: {
      ready: true,
      mixer_connected: true,
      trace_count: 14,
      blocked_trace_count: 3,
      project_goal_capability_objective_readiness_score: 78,
      project_goal_capability_objective_readiness_goal_area_priorities: [
        { goal_area: 'input_gain', gap_row_count: 0, readiness_score: 100 },
      ],
      project_goal_capability_objective_readiness_by_goal_area: {
        input_gain: {
          goal_area: 'input_gain',
          ready_row_count: 4,
          gap_row_count: 0,
          readiness_score: 100,
        },
      },
      project_goal_capability_objective_readiness_hotspots: [
        {
          goal_area: 'input_gain',
          gap_row_count: 0,
          blocker_row_ids: [],
        },
      ],
    },
    blocked_surfaces: ['eq', 'compressor'],
    type: 'safety_monitor',
    source: 'runtime_policy',
  }, {
    ...mockBackendState,
    liveWritePolicy: 'fader_gain_supervised_review',
    applyPath: '/console',
  });

  assert.equal(safetyMonitor.goalCapabilityObjectiveReadinessScore, 78);
  assert.deepEqual(safetyMonitor.goalCapabilityObjectiveReadinessGoalAreaPriorities, [
    { goal_area: 'input_gain', gap_row_count: 0, readiness_score: 100 },
  ]);
  assert.deepEqual(safetyMonitor.goalCapabilityObjectiveReadinessByGoalArea, {
    input_gain: {
      goal_area: 'input_gain',
      ready_row_count: 4,
      gap_row_count: 0,
      readiness_score: 100,
    },
  });
  assert.deepEqual(safetyMonitor.goalCapabilityObjectiveReadinessHotspots, [
    {
      goal_area: 'input_gain',
      gap_row_count: 0,
      blocker_row_ids: [],
    },
  ]);
  assert.deepEqual(safetyMonitor.proofGoalCapabilityObjectiveReadinessGoalAreaPriorities, [
    { goal_area: 'eq', gap_row_count: 1, readiness_score: 66 },
  ]);
  assert.deepEqual(safetyMonitor.proofGoalCapabilityObjectiveReadinessByGoalArea, {
    eq: {
      goal_area: 'eq',
      ready_row_count: 2,
      gap_row_count: 1,
      readiness_score: 66,
    },
  });
  assert.deepEqual(safetyMonitor.proofGoalCapabilityObjectiveReadinessHotspots, [
    {
      goal_area: 'eq',
      gap_row_count: 1,
      blocker_row_ids: ['cg_room_mic_eq'],
    },
  ]);
  assert.equal(safetyMonitor.traceCount, 14);
  assert.equal(safetyMonitor.blockedTraceCount, 3);
  assert.equal(safetyMonitor.emergencyStopped, false);
  assert.equal(safetyMonitor.resetRequired, true);
  assert.equal(safetyMonitor.lastReadbackStatus, 'verified');
});

test('adaptSafetyMonitor normalizes malformed goal-capability readiness payloads', async () => {
  const { adaptSafetyMonitor, mockBackendState } = await importTypeScript('services/operatorAdapters.ts');

  const safetyMonitor = adaptSafetyMonitor({
    operator_mode: { mode: 'auto' },
    live_write_policy: { apply_path: '/console', policy: 'full_autonomous_review' },
    manual_write_supervision: {
      emergency_stopped: 0,
      reset_required: '',
      approved_live_write_kinds: ['gain', null, 12],
    },
    proof: {
      last_readback: { status: '' },
      goal_capability_objective_readiness_goal_area_priorities: [
        { goal_area: 'eq', readiness_score: 66, gap_row_count: '4' },
        'bad-entry',
        { bad: true },
      ],
      goal_capability_objective_readiness_by_goal_area: {
        eq: {
          goal_area: 'eq',
          gap_row_count: '1',
          blocker_row_count: 2,
          blocker_row_ids: ['cg_eq_1', 'cg_eq_2'],
          full_autonomous_status: 'blocked_live',
          target_live_writable_row_count: 0,
          target_proof_gap_count: 1,
        },
        input_gain: {
          goal_area: 'input_gain',
          ready_row_count: '2',
          gap_row_count: 0,
          blocker_rows: ['cg_gain_1'],
        },
        malformed: 'value',
      },
      goal_capability_objective_readiness_hotspots: [
        { goal_area: 'eq', gap_row_count: 2, blocker_row_ids: ['cg_eq_1'] },
        null,
        { goal_area: 'eq', gap_row_count: '3', blocker_row_count: 2 },
      ],
      last_proof_status: 'applied',
    },
    summary: {
      ready: 'true',
      mixer_connected: 1,
      trace_count: '12',
      blocked_trace_count: null,
      project_goal_capability_objective_readiness_score: '88',
      project_goal_capability_objective_readiness_goal_area_priorities: [
        { goal_area: 'input_gain', readiness_score: 50, gap_row_count: 1 },
      ],
      project_goal_capability_objective_readiness_by_goal_area: {
        eq: {
          goal_area: 'eq',
          ready_row_count: '4',
          gap_row_count: 5,
          blocker_row_count: 2,
          blocker_row_ids: 'not-an-array',
        },
      },
      project_goal_capability_objective_readiness_hotspots: [
        { goal_area: 'eq', gap_row_count: 2, blocker_row_ids: [] },
      ],
    },
    blocked_surfaces: [],
    type: 'safety_monitor',
    source: 'runtime_policy',
  }, {
    ...mockBackendState,
    liveWritePolicy: 'full_autonomous_review',
    applyPath: '/console',
  });

  assert.equal(safetyMonitor.operatorMode, 'auto');
  assert.equal(safetyMonitor.traceCount, 12);
  assert.equal(safetyMonitor.blockedTraceCount, 0);
  assert.equal(safetyMonitor.goalCapabilityObjectiveReadinessScore, 88);
  assert.deepEqual(safetyMonitor.goalCapabilityObjectiveReadinessGoalAreaPriorities, [
    { goal_area: 'input_gain', gap_row_count: 1, readiness_score: 50 },
  ]);
  assert.deepEqual(safetyMonitor.goalCapabilityObjectiveReadinessByGoalArea, {
    eq: {
      goal_area: 'eq',
      ready_row_count: 4,
      gap_row_count: 5,
      blocker_row_count: 2,
      blocker_row_ids: [],
    },
  });
  assert.deepEqual(safetyMonitor.goalCapabilityObjectiveReadinessHotspots, [
    {
      goal_area: 'eq',
      gap_row_count: 2,
      blocker_row_ids: [],
    },
  ]);
  assert.deepEqual(safetyMonitor.proofGoalCapabilityObjectiveReadinessGoalAreaPriorities, [
    { goal_area: 'eq', gap_row_count: 4, readiness_score: 66 },
  ]);
  assert.deepEqual(safetyMonitor.proofGoalCapabilityObjectiveReadinessByGoalArea, {
    eq: {
      goal_area: 'eq',
      gap_row_count: 1,
      blocker_row_count: 2,
      blocker_row_ids: ['cg_eq_1', 'cg_eq_2'],
      full_autonomous_status: 'blocked_live',
      target_live_writable_row_count: 0,
      target_proof_gap_count: 1,
    },
    input_gain: {
      goal_area: 'input_gain',
      ready_row_count: 2,
      gap_row_count: 0,
      blocker_rows: ['cg_gain_1'],
      blocker_row_ids: ['cg_gain_1'],
    },
  });
  assert.deepEqual(safetyMonitor.proofGoalCapabilityObjectiveReadinessHotspots, [
    {
      goal_area: 'eq',
      gap_row_count: 2,
      blocker_row_ids: ['cg_eq_1'],
    },
    {
      goal_area: 'eq',
      gap_row_count: 3,
      blocker_row_count: 2,
    },
  ]);
  assert.equal(safetyMonitor.lastReadbackStatus, 'none');
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
  assert.match(source, /current_write_surface:/);
  assert.match(source, /Full Auto/);
  assert.match(source, /<Panel title="Mode Policy">/);
  assert.match(source, /rows=\{modeRows\.slice\(0, 3\)\}/);
  assert.match(source, /Supervised surfaces/);
});

test('Settings Module Status inspector includes project goal capability summaries', async () => {
  const source = await fs.readFile(path.join(srcRoot, 'views/SettingsView.tsx'), 'utf8');

  assert.match(source, /row\.domain === 'project_goal_capability'/);
  assert.match(source, /label: 'Goal', predicate: row => row\.domain === 'project_goal_capability'/);
  assert.match(source, /label: 'Goal capabilities'/);
  assert.match(source, /label: 'Goal blockers'/);
  assert.match(source, /writeScope: string;/);
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

test('directorEventRows renders integration bundle blockers as operator-readable events', async () => {
  const { directorEventRows } = await importTypeScript('components/dashboard/directorPanelModel.ts');

  const runtime = {
    currentGoal: 'Keep writes supervised.',
    readyLabel: 'Not live ready',
    readyTone: 'red',
    audioLabel: 'Read-only',
    audioTone: 'yellow',
    proposalQueueLabel: '2 pending',
    proposalQueueTone: 'yellow',
    liveWritePolicy: 'supervised_fader_gain_only',
    manualGateLabel: 'Apply blocked',
    manualGateTone: 'yellow',
    resetRequired: false,
    hardwareProofItems: [
      { label: 'Readback', value: 'cache only', tone: 'yellow' },
    ],
    blockedReasons: [
      'integration_bundle_status:blocked',
      'integration_bundle_blocker:offline_integration_bundle_contract_not_ready',
      'integration_bundle_missing_check:operator_report_bound',
      'integration_bundle_source_authority_denied:true',
      'integration_bundle_live_authority_granted:false',
      'integration_bundle_readiness_closure_allowed:false',
    ],
    statusItems: [],
    sourceLabel: 'Backend',
  };

  const rows = directorEventRows(runtime, []);
  const integrationBundleRow = rows.find(row => row.source === 'Integration bundle');

  assert.equal(integrationBundleRow?.tone, 'red');
  assert.equal(
    integrationBundleRow?.message,
    'Integration bundle blocked; 1 blocker(s); missing: operator_report_bound; source authority denied, live authority denied, readiness closure denied.',
  );
  assert.equal(rows.some(row => row.source === 'Blockers'), false);
  assert.equal(
    rows.some(row => row.message.includes('integration_bundle_blocker:')),
    false,
  );
});

test('director runtime guard banner uses readable integration bundle blocker messages', async () => {
  const { directorBlockedReasonMessages } = await importTypeScript('components/dashboard/directorPanelModel.ts');
  const directorPanelSource = await fs.readFile(path.join(srcRoot, 'views/DirectorPanel.tsx'), 'utf8');

  const messages = directorBlockedReasonMessages([
    'integration_bundle_status:blocked',
    'integration_bundle_blocker:offline_integration_bundle_contract_not_ready',
    'integration_bundle_missing_check:operator_report_bound',
    'integration_bundle_source_authority_denied:true',
    'integration_bundle_live_authority_granted:false',
    'integration_bundle_readiness_closure_allowed:false',
    'physical_pre_read_required',
  ]);

  assert.deepEqual(messages, [
    'Integration bundle blocked; 1 blocker(s); missing: operator_report_bound; source authority denied, live authority denied, readiness closure denied.',
    'physical_pre_read_required',
  ]);
  assert.match(directorPanelSource, /directorBlockedReasonMessages\(runtime\.blockedReasons\)/);
  assert.equal(
    directorPanelSource.includes("runtime.blockedReasons.slice(0, 2).join(' · ')"),
    false,
  );
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
  const rightDirectorSource = await fs.readFile(
    path.join(srcRoot, 'views/DirectorPanel.tsx'),
    'utf8',
  );

  assert.match(source, /directorProposalRows/);
  assert.match(rightDirectorSource, /suggestionBlockerNotice/);
  assert.match(rightDirectorSource, /Promotion planning blocked\. Suggestions remain review-only/);
  assert.match(rightDirectorSource, /blockedReasonMessages\[0\]/);
  assert.match(source, /Просмотр/);
  assert.match(source, /No active backend proposals require review/);
  assert.match(source, /No OSC write sent/);
  assert.equal(source.includes('Принять'), false);
});

test('task queue renders integration bundle blocker as review-only notice', async () => {
  const source = await fs.readFile(
    path.join(srcRoot, 'components/prototype/TaskQueuePanels.tsx'),
    'utf8',
  );
  const viewSource = await fs.readFile(
    path.join(srcRoot, 'views/TaskQueueView.tsx'),
    'utf8',
  );

  assert.match(source, /taskQueueBlockerNotice/);
  assert.match(source, /directorBlockedReasonMessages\(runtime\.blockedReasons\)/);
  assert.match(source, /Promotion planning blocked\. Task Queue evidence remains review-only/);
  assert.match(viewSource, /runtime=\{data\.directorRuntime\}/);
  assert.equal(source.includes('queueMutation'), false);
});

test('supervised apply modal renders integration bundle blocker before apply review', async () => {
  const source = await fs.readFile(
    path.join(srcRoot, 'components/prototype/TaskQueuePanels.tsx'),
    'utf8',
  );

  assert.match(source, /supervisedApplyBlockerNotice/);
  assert.match(source, /integrationBundleBlockerNotice/);
  assert.match(source, /Promotion planning blocked\. Supervised apply remains blocked/);
  assert.match(source, /directorBlockedReasonMessages\(runtime\.blockedReasons\)/);
  assert.equal(source.includes('queueMutation'), false);
});

test('supervised apply modal blocks confirm when integration bundle is blocked', async () => {
  const source = await fs.readFile(
    path.join(srcRoot, 'components/prototype/TaskQueuePanels.tsx'),
    'utf8',
  );

  assert.match(source, /integrationBundleApplyBlocked/);
  assert.match(source, /&& !integrationBundleApplyBlocked/);
  assert.match(source, /Blocked by integration bundle/);
  assert.match(source, /disabled=\{!canRequestBackendGate\}/);
  assert.match(source, /integrationBundleApplyBlocked \|\| task\.canApply === false \|\| task\.reviewOnly/);
  assert.match(source, /canRequestBackendGate \? 'Request Supervised Apply' : 'Apply Blocked'/);
  assert.equal(source.includes('queueMutation'), false);
});

test('supervised apply confirm callback is guarded by integration bundle blocker', async () => {
  const source = await fs.readFile(
    path.join(srcRoot, 'components/prototype/TaskQueuePanels.tsx'),
    'utf8',
  );

  assert.match(source, /guardedConfirmSupervisedApply/);
  assert.match(source, /if \(integrationBundleApplyBlocked\) return/);
  assert.match(source, /actions\.confirmSupervisedApply\(taskId\)/);
  assert.match(source, /onConfirm=\{\(\) => guardedConfirmSupervisedApply\(modalTask\.id\)\}/);
  assert.equal(source.includes('queueMutation'), false);
});

test('task apply request helper is guarded by integration bundle blocker', async () => {
  const source = await fs.readFile(
    path.join(srcRoot, 'components/prototype/TaskQueuePanels.tsx'),
    'utf8',
  );

  assert.match(source, /integrationBundleApplyBlocked = false/);
  assert.match(source, /if \(integrationBundleApplyBlocked\) \{/);
  assert.match(source, /action: 'Apply blocked'/);
  assert.match(source, /offline integration bundle blockers are visible/);
  assert.match(source, /requestTaskApply\(task, ui, actions, integrationBundleApplyBlocked\)/);
  assert.match(source, /actions\.openSupervisedProposal\(task\.id\)/);
  assert.equal(source.includes('queueMutation'), false);
});

test('task details apply entrypoint is guarded by integration bundle blocker', async () => {
  const source = await fs.readFile(
    path.join(srcRoot, 'components/prototype/TaskQueuePanels.tsx'),
    'utf8',
  );
  const viewSource = await fs.readFile(path.join(srcRoot, 'views/TaskQueueView.tsx'), 'utf8');

  assert.match(source, /integrationBundleApplyBlocked = false/);
  assert.match(source, /integrationBundleApplyBlocked\?: boolean/);
  assert.match(source, /const applyBlocked = integrationBundleApplyBlocked \|\| task\.canApply === false \|\| task\.reviewOnly/);
  assert.match(source, /requestTaskApply\(task, ui, actions, integrationBundleApplyBlocked\)/);
  assert.match(viewSource, /directorBlockedReasonMessages\(data\.directorRuntime\.blockedReasons\)\.length > 0/);
  assert.match(viewSource, /integrationBundleApplyBlocked=\{integrationBundleApplyBlocked\}/);
  assert.equal(source.includes('queueMutation'), false);
});

test('task queue metrics show apply blocked when integration bundle is blocked', async () => {
  const source = await fs.readFile(
    path.join(srcRoot, 'components/prototype/TaskQueuePanels.tsx'),
    'utf8',
  );

  assert.match(source, /function taskQueueMetrics\(/);
  assert.match(source, /integrationBundleApplyBlocked = false/);
  assert.match(source, /const applyableCount = integrationBundleApplyBlocked/);
  assert.match(source, /\? 0/);
  assert.match(source, /label: integrationBundleApplyBlocked \? 'Apply Blocked' : 'Applyable'/);
  assert.match(source, /sub: integrationBundleApplyBlocked \? 'integration bundle' : 'fader \/ gain only'/);
  assert.match(source, /tone: integrationBundleApplyBlocked \? 'red' : 'green'/);
  assert.match(source, /taskQueueMetrics\(taskRows, integrationBundleApplyBlocked\)/);
  assert.equal(source.includes('queueMutation'), false);
});

test('task queue review-only metric counts all rows when integration bundle is blocked', async () => {
  const source = await fs.readFile(
    path.join(srcRoot, 'components/prototype/TaskQueuePanels.tsx'),
    'utf8',
  );

  assert.match(source, /const reviewOnlyCount = integrationBundleApplyBlocked/);
  assert.match(source, /\? taskRows\.length/);
  assert.match(source, /: taskRows\.filter\(task => task\.reviewOnly \|\| task\.canApply === false\)\.length/);
  assert.match(source, /label: 'Review Only', value: String\(reviewOnlyCount\)/);
  assert.match(source, /taskQueueMetrics\(taskRows, integrationBundleApplyBlocked\)/);
  assert.equal(source.includes('queueMutation'), false);
});

test('task details blocker list includes integration bundle blocker when blocked', async () => {
  const source = await fs.readFile(
    path.join(srcRoot, 'components/prototype/TaskQueuePanels.tsx'),
    'utf8',
  );

  assert.match(source, /const detailApplyBlockers = integrationBundleApplyBlocked/);
  assert.match(source, /Array\.from\(new Set\(\['offline_integration_bundle_blockers_visible', \.\.\.blockers\]\)\)/);
  assert.match(source, /\? Array\.from\(new Set\(\['offline_integration_bundle_blockers_visible', \.\.\.blockers\]\)\)/);
  assert.match(source, /: blockers/);
  assert.match(source, /detailApplyBlockers\.length > 0/);
  assert.match(source, /detailApplyBlockers\.slice\(0, 4\)\.map\(blocker =>/);
  assert.equal(source.includes('queueMutation'), false);
});

test('apply plan sequence is guarded when integration bundle is blocked', async () => {
  const source = await fs.readFile(
    path.join(srcRoot, 'components/prototype/TaskQueuePanels.tsx'),
    'utf8',
  );

  assert.match(source, /function applyPlanForIntegrationBundleBlocker\(/);
  assert.match(source, /if \(!integrationBundleApplyBlocked\) return plan/);
  assert.match(source, /value: 'Bundle blocked', tone: 'red'/);
  assert.match(source, /value: 'No contact', tone: 'red'/);
  assert.match(source, /const guardedApplyPlan = applyPlanForIntegrationBundleBlocker\(applyPlan, integrationBundleApplyBlocked\)/);
  assert.match(source, /const modalApplyPlan = applyPlanForIntegrationBundleBlocker\(/);
  assert.match(source, /<ApplyPlanPanel plan=\{guardedApplyPlan\} \/>/);
  assert.match(source, /applyPlan=\{modalApplyPlan\}/);
  assert.equal(source.includes('queueMutation'), false);
});

test('operator decision status is guarded when integration bundle is blocked', async () => {
  const source = await fs.readFile(
    path.join(srcRoot, 'components/prototype/TaskQueuePanels.tsx'),
    'utf8',
  );

  assert.match(source, /function OperatorDecisionPanel\(\{/);
  assert.match(source, /integrationBundleApplyBlocked\?: boolean/);
  assert.match(source, /const writePathStatus = integrationBundleApplyBlocked/);
  assert.match(source, /value: 'Integration bundle blocked', tone: 'red' as Tone/);
  assert.match(source, /const backendStatus = integrationBundleApplyBlocked/);
  assert.match(source, /value: 'No contact', tone: 'red' as Tone/);
  assert.match(source, /value: writePathStatus\.value, tone: writePathStatus\.tone/);
  assert.match(source, /value: backendStatus\.value, tone: backendStatus\.tone/);
  assert.match(source, /<OperatorDecisionPanel ui=\{ui\} policy=\{policy\} integrationBundleApplyBlocked=\{integrationBundleApplyBlocked\} \/>/);
  assert.equal(source.includes('queueMutation'), false);
});

test('operator decision apply result is guarded when integration bundle is blocked', async () => {
  const source = await fs.readFile(
    path.join(srcRoot, 'components/prototype/TaskQueuePanels.tsx'),
    'utf8',
  );

  assert.match(source, /const applyResultStatus = integrationBundleApplyBlocked/);
  assert.match(source, /value: 'Apply blocked', tone: 'red' as Tone/);
  assert.match(source, /value: policy\.apply, tone: toneForMode\(ui\.mode\)/);
  assert.match(source, /value: applyResultStatus\.value, tone: applyResultStatus\.tone/);
  assert.match(source, /value: writePathStatus\.value, tone: writePathStatus\.tone/);
  assert.match(source, /value: backendStatus\.value, tone: backendStatus\.tone/);
  assert.equal(source.includes('queueMutation'), false);
});

test('safety monitor policy is guarded when integration bundle is blocked', async () => {
  const source = await fs.readFile(
    path.join(srcRoot, 'components/prototype/TaskQueuePanels.tsx'),
    'utf8',
  );

  assert.match(source, /function SafetyMonitorPanel\(\{/);
  assert.match(source, /integrationBundleApplyBlocked\?: boolean/);
  assert.match(source, /const policyStatus = integrationBundleApplyBlocked/);
  assert.match(source, /value: 'Integration bundle blocked', tone: 'red' as const/);
  assert.match(source, /value: safety\.liveWritePolicy, tone: safety\.liveWritePolicy\.includes\('fader_gain'\) \? 'green' as const : 'yellow' as const/);
  assert.match(source, /label: 'Policy', value: policyStatus\.value, tone: policyStatus\.tone/);
  assert.match(source, /<SafetyMonitorPanel safety=\{safetyMonitor\} integrationBundleApplyBlocked=\{integrationBundleApplyBlocked\} \/>/);
  assert.equal(source.includes('queueMutation'), false);
});

test('safety monitor goal readiness is guarded when integration bundle is blocked', async () => {
  const source = await fs.readFile(
    path.join(srcRoot, 'components/prototype/TaskQueuePanels.tsx'),
    'utf8',
  );

  assert.match(source, /const goalReadinessStatus = integrationBundleApplyBlocked/);
  assert.match(source, /value: 'Blocked by integration bundle', tone: 'red' as const/);
  assert.match(source, /value: `\$\{safety\.goalCapabilityObjectiveReadinessScore\}%`/);
  assert.match(source, /tone: toneForReadinessScore\(safety\.goalCapabilityObjectiveReadinessScore\)/);
  assert.match(source, /label: 'Goal readiness', value: goalReadinessStatus\.value, tone: goalReadinessStatus\.tone/);
  assert.match(source, /label: 'Policy', value: policyStatus\.value, tone: policyStatus\.tone/);
  assert.match(source, /<SafetyMonitorPanel safety=\{safetyMonitor\} integrationBundleApplyBlocked=\{integrationBundleApplyBlocked\} \/>/);
  assert.equal(source.includes('queueMutation'), false);
});

test('safety monitor blocked surfaces include integration bundle blocker when blocked', async () => {
  const source = await fs.readFile(
    path.join(srcRoot, 'components/prototype/TaskQueuePanels.tsx'),
    'utf8',
  );

  assert.match(source, /const blockedSurfaces = integrationBundleApplyBlocked/);
  assert.match(source, /Array\.from\(new Set\(\['offline_integration_bundle_blockers_visible', \.\.\.safety\.blockedSurfaces\]\)\)/);
  assert.match(source, /: safety\.blockedSurfaces/);
  assert.match(source, /const blockedSurfaceText = blockedSurfaces\.slice\(0, 4\)\.join\(', '\) \|\| 'none'/);
  assert.match(source, /Blocked: \{blockedSurfaceText\}/);
  assert.doesNotMatch(source, /Blocked: \{safety\.blockedSurfaces\.slice\(0, 4\)\.join\(', '\) \|\| 'none'\}/);
  assert.match(source, /<SafetyMonitorPanel safety=\{safetyMonitor\} integrationBundleApplyBlocked=\{integrationBundleApplyBlocked\} \/>/);
  assert.equal(source.includes('queueMutation'), false);
});

test('replay evidence decision trace status is guarded when integration bundle is blocked', async () => {
  const source = await fs.readFile(
    path.join(srcRoot, 'components/prototype/TaskQueuePanels.tsx'),
    'utf8',
  );

  assert.match(source, /function ReplayEvidencePanel\(\{/);
  assert.match(source, /integrationBundleApplyBlocked = false/);
  assert.match(source, /integrationBundleApplyBlocked\?: boolean/);
  assert.match(source, /const traceStatusLabel = integrationBundleApplyBlocked/);
  assert.match(source, /\? 'bundle blocked'/);
  assert.match(source, /\? latestTrace\.blockedReason \? 'blocked' : latestTrace\.proofStatus/);
  assert.match(source, /: 'no trace'/);
  assert.match(source, /const traceStatusTone = integrationBundleApplyBlocked/);
  assert.match(source, /\? 'red' as const/);
  assert.match(source, /\? latestTrace\.blockedReason \? 'red' as const : latestTrace\.tone/);
  assert.match(source, /: 'muted' as const/);
  assert.match(source, /<StatusBadge tone=\{traceStatusTone\}>\{traceStatusLabel\}<\/StatusBadge>/);
  assert.match(source, /<ReplayEvidencePanel[\s\S]*integrationBundleApplyBlocked=\{integrationBundleApplyBlocked\}/);
  assert.equal(source.includes('queueMutation'), false);
});

test('replay evidence latest run status is guarded when integration bundle is blocked', async () => {
  const source = await fs.readFile(
    path.join(srcRoot, 'components/prototype/TaskQueuePanels.tsx'),
    'utf8',
  );

  assert.match(source, /function ReplayEvidencePanel\(\{/);
  assert.match(source, /const latestRunStatusLabel = integrationBundleApplyBlocked/);
  assert.match(source, /\? 'bundle blocked'/);
  assert.match(source, /: latestRun\?\.status \?\? 'idle'/);
  assert.match(source, /const latestRunStatusTone = integrationBundleApplyBlocked/);
  assert.match(source, /\? 'red' as const/);
  assert.match(source, /: toneForReplayStatus\(latestRun\?\.status \?\? 'idle'\)/);
  assert.match(source, /<StatusBadge tone=\{latestRunStatusTone\}>\{latestRunStatusLabel\}<\/StatusBadge>/);
  assert.doesNotMatch(source, /<StatusBadge tone=\{toneForReplayStatus\(latestRun\?\.status \?\? 'idle'\)\}>\{latestRun\?\.status \?\? 'idle'\}<\/StatusBadge>/);
  assert.match(source, /<ReplayEvidencePanel[\s\S]*integrationBundleApplyBlocked=\{integrationBundleApplyBlocked\}/);
  assert.equal(source.includes('queueMutation'), false);
});

test('replay evidence policy row is guarded when integration bundle is blocked', async () => {
  const source = await fs.readFile(
    path.join(srcRoot, 'components/prototype/TaskQueuePanels.tsx'),
    'utf8',
  );

  assert.match(source, /function ReplayEvidencePanel\(\{/);
  assert.match(source, /const replayPolicyLabel = integrationBundleApplyBlocked/);
  assert.match(source, /\? 'bundle blocked'/);
  assert.match(source, /: latestRun\?\.offlineOnly === false \? 'review' : 'offline only'/);
  assert.match(source, /const replayPolicyDetail = integrationBundleApplyBlocked/);
  assert.match(source, /\? 'no live apply'/);
  assert.match(source, /: latestRun\?\.liveWrite \? 'live write' : 'no live write'/);
  assert.match(source, /const replayPolicyDetailClassName = integrationBundleApplyBlocked \|\| latestRun\?\.liveWrite/);
  assert.match(source, /\? 'text-danger'/);
  assert.match(source, /: 'text-success'/);
  assert.match(source, /\{replayPolicyLabel\} ·\{' '\}/);
  assert.match(source, /<span className=\{replayPolicyDetailClassName\}>/);
  assert.match(source, /\{replayPolicyDetail\}/);
  assert.doesNotMatch(source, /\{latestRun\?\.offlineOnly === false \? 'review' : 'offline only'\} ·/);
  assert.match(source, /<ReplayEvidencePanel[\s\S]*integrationBundleApplyBlocked=\{integrationBundleApplyBlocked\}/);
  assert.equal(source.includes('queueMutation'), false);
});

test('replay evidence channel status is guarded when integration bundle is blocked', async () => {
  const source = await fs.readFile(
    path.join(srcRoot, 'components/prototype/TaskQueuePanels.tsx'),
    'utf8',
  );

  assert.match(source, /function ReplayEvidencePanel\(\{/);
  assert.match(source, /const channelStatusLabel = integrationBundleApplyBlocked/);
  assert.match(source, /\? 'bundle blocked'/);
  assert.match(source, /: channel\?\.gainRideState \?\? 'shadow'/);
  assert.match(source, /const channelStatusTone = integrationBundleApplyBlocked/);
  assert.match(source, /\? 'red' as const/);
  assert.match(source, /: channel\?\.liveApplyAllowed \? 'yellow' as const : 'blue' as const/);
  assert.match(source, /<StatusBadge tone=\{channelStatusTone\}>\{channelStatusLabel\}<\/StatusBadge>/);
  assert.doesNotMatch(source, /<StatusBadge tone=\{channel\?\.liveApplyAllowed \? 'yellow' : 'blue'\}>\{channel\?\.gainRideState \?\? 'shadow'\}<\/StatusBadge>/);
  assert.match(source, /<ReplayEvidencePanel[\s\S]*integrationBundleApplyBlocked=\{integrationBundleApplyBlocked\}/);
  assert.equal(source.includes('queueMutation'), false);
});

test('replay evidence report path is guarded when integration bundle is blocked', async () => {
  const source = await fs.readFile(
    path.join(srcRoot, 'components/prototype/TaskQueuePanels.tsx'),
    'utf8',
  );

  assert.match(source, /function ReplayEvidencePanel\(\{/);
  assert.match(source, /const replayReportLabel = integrationBundleApplyBlocked/);
  assert.match(source, /\? 'bundle blocked'/);
  assert.match(source, /: latestRun\?\.reportPath \|\| 'pending'/);
  assert.match(source, /const replayReportClassName = integrationBundleApplyBlocked/);
  assert.match(source, /\? 'mt-0\.5 truncate text-\[8px\] text-danger'/);
  assert.match(source, /: 'mt-0\.5 truncate text-\[8px\] text-muted'/);
  assert.match(source, /<div className=\{replayReportClassName\}>/);
  assert.match(source, /Report: \{replayReportLabel\}/);
  assert.doesNotMatch(source, /Report: \{latestRun\.reportPath \|\| 'pending'\}/);
  assert.match(source, /<ReplayEvidencePanel[\s\S]*integrationBundleApplyBlocked=\{integrationBundleApplyBlocked\}/);
  assert.equal(source.includes('queueMutation'), false);
});

test('supervised apply final readiness is guarded when integration bundle is blocked', async () => {
  const source = await fs.readFile(
    path.join(srcRoot, 'components/prototype/TaskQueuePanels.tsx'),
    'utf8',
  );

  assert.match(source, /function SupervisedApplyModal\(\{/);
  assert.match(source, /const finalReadinessStatus = integrationBundleApplyBlocked/);
  assert.match(source, /value: 'Blocked by integration bundle', tone: 'red' as Tone/);
  assert.match(source, /value: ui\.backend\.statusLabel, tone: ui\.backend\.statusTone/);
  assert.match(source, /label: 'Final readiness', value: finalReadinessStatus\.value, tone: finalReadinessStatus\.tone/);
  assert.match(source, /disabled=\{!canRequestBackendGate\}/);
  assert.equal(source.includes('queueMutation'), false);
});

test('supervised apply backend result is hidden when integration bundle is blocked', async () => {
  const source = await fs.readFile(
    path.join(srcRoot, 'components/prototype/TaskQueuePanels.tsx'),
    'utf8',
  );

  assert.match(source, /const visibleBackendResultMatches = resultMatches && !integrationBundleApplyBlocked/);
  assert.match(source, /: visibleBackendResultMatches \? backendResult\?\.applyRequest \?\? applyRequest : applyRequest/);
  assert.match(source, /: visibleBackendResultMatches \? backendResult\?\.readbackStatus \?\? 'Required after backend apply' : 'Required after backend apply'/);
  assert.match(source, /: visibleBackendResultMatches \? backendResult\?\.rollbackStatus \?\? 'Snapshot required' : 'Snapshot required'/);
  assert.match(source, /const backendResultTone = visibleBackendResultMatches \? backendResult\?\.tone \?\? approvalTone : approvalTone/);
  assert.match(source, /visibleBackendResultMatches && readbackLabel\.includes\('verified'\)/);
  assert.match(source, /visibleBackendResultMatches && rollbackLabel\.includes\('available'\)/);
  assert.match(source, /\{visibleBackendResultMatches && \(/);
  assert.equal(source.includes('queueMutation'), false);
});

test('supervised apply gate rows are blocked when integration bundle is blocked', async () => {
  const source = await fs.readFile(
    path.join(srcRoot, 'components/prototype/TaskQueuePanels.tsx'),
    'utf8',
  );

  assert.match(source, /const modalGateBlockerStatus = integrationBundleApplyBlocked/);
  assert.match(source, /applyRequest: 'Blocked'/);
  assert.match(source, /readback: 'No backend contact'/);
  assert.match(source, /rollback: 'No live apply'/);
  assert.match(source, /readbackTone: 'red' as Tone/);
  assert.match(source, /rollbackTone: 'red' as Tone/);
  assert.match(source, /const applyRequestLabel = modalGateBlockerStatus/);
  assert.match(source, /\? modalGateBlockerStatus\.applyRequest/);
  assert.match(source, /const readbackLabel = modalGateBlockerStatus/);
  assert.match(source, /\? modalGateBlockerStatus\.readback/);
  assert.match(source, /const rollbackLabel = modalGateBlockerStatus/);
  assert.match(source, /\? modalGateBlockerStatus\.rollback/);
  assert.match(source, /const readbackTone = modalGateBlockerStatus/);
  assert.match(source, /const rollbackTone = modalGateBlockerStatus/);
  assert.match(source, /label: 'Readback status', value: readbackLabel, tone: readbackTone/);
  assert.match(source, /label: 'Rollback status', value: rollbackLabel, tone: rollbackTone/);
  assert.equal(source.includes('queueMutation'), false);
});

test('task queue review-only filter includes all rows when integration bundle is blocked', async () => {
  const source = await fs.readFile(
    path.join(srcRoot, 'components/prototype/TaskQueuePanels.tsx'),
    'utf8',
  );

  assert.match(source, /label: 'Review only'/);
  assert.match(source, /predicate: task => integrationBundleApplyBlocked \|\| Boolean\(task\.reviewOnly \|\| task\.canApply === false\)/);
  assert.match(source, /actions=\{taskQueueRowActions\(task, integrationBundleApplyBlocked\)\}/);
  assert.match(source, /taskQueueMetrics\(taskRows, integrationBundleApplyBlocked\)/);
  assert.equal(source.includes('queueMutation'), false);
});

test('task queue row badge shows blocked state when integration bundle is blocked', async () => {
  const source = await fs.readFile(
    path.join(srcRoot, 'components/prototype/TaskQueuePanels.tsx'),
    'utf8',
  );

  assert.match(source, /\(integrationBundleApplyBlocked \|\| task\.reviewOnly\)/);
  assert.match(source, /integrationBundleApplyBlocked \? 'Blocked' : 'Review'/);
  assert.match(source, /predicate: task => integrationBundleApplyBlocked \|\| Boolean\(task\.reviewOnly \|\| task\.canApply === false\)/);
  assert.match(source, /actions=\{taskQueueRowActions\(task, integrationBundleApplyBlocked\)\}/);
  assert.equal(source.includes('queueMutation'), false);
});

test('task queue status column shows blocked state when integration bundle is blocked', async () => {
  const source = await fs.readFile(
    path.join(srcRoot, 'components/prototype/TaskQueuePanels.tsx'),
    'utf8',
  );

  assert.match(source, /function taskStatusDisplay\(/);
  assert.match(source, /integrationBundleApplyBlocked = false/);
  assert.match(source, /const label = integrationBundleApplyBlocked \? 'Blocked' : task\.status/);
  assert.match(source, /const tone = integrationBundleApplyBlocked \? 'red' as const : toneFromState\(task\.status\)/);
  assert.match(source, /return \{ label, tone \}/);
  assert.match(source, /const status = taskStatusDisplay\(task, integrationBundleApplyBlocked\)/);
  assert.match(source, /<StatusBadge tone=\{status\.tone\}>\{status\.label\}<\/StatusBadge>/);
  assert.doesNotMatch(source, /key: 'status'[\s\S]{0,120}toneFromState\(task\.status\)[\s\S]{0,80}task\.status/);
  assert.equal(source.includes('queueMutation'), false);
});

test('task queue move column shows blocked state when integration bundle is blocked', async () => {
  const source = await fs.readFile(
    path.join(srcRoot, 'components/prototype/TaskQueuePanels.tsx'),
    'utf8',
  );

  assert.match(source, /function taskMoveDisplay\(/);
  assert.match(source, /const label = integrationBundleApplyBlocked \? 'Blocked' : task\.proposal\.delta/);
  assert.match(source, /const className = integrationBundleApplyBlocked \? 'text-danger' : 'text-audio'/);
  assert.match(source, /const move = taskMoveDisplay\(task, integrationBundleApplyBlocked\)/);
  assert.match(source, /className=\{`ml-2 font-mono \$\{move\.className\}`\}/);
  assert.match(source, /\{move\.label\}/);
  assert.doesNotMatch(source, /<span className="ml-2 font-mono text-audio">\{task\.proposal\.delta\}<\/span>/);
  assert.equal(source.includes('queueMutation'), false);
});

test('task queue row reject action is hidden when integration bundle is blocked', async () => {
  const source = await fs.readFile(
    path.join(srcRoot, 'components/prototype/TaskQueuePanels.tsx'),
    'utf8',
  );

  assert.match(source, /function taskQueueRowActions\(/);
  assert.match(source, /if \(integrationBundleApplyBlocked\) return \['Detail'\]/);
  assert.match(source, /if \(task\.canApply === false \|\| task\.reviewOnly\) return \['Detail', 'Reject'\]/);
  assert.match(source, /return \['Detail', 'Apply', 'Reject'\]/);
  assert.match(source, /actions=\{taskQueueRowActions\(task, integrationBundleApplyBlocked\)\}/);
  assert.match(source, /else if \(action === 'Reject'\) actions\.setTaskStatus\(task\.id, 'Rejected'\)/);
  assert.equal(source.includes('queueMutation'), false);
});

test('task details badge shows blocked state when integration bundle is blocked', async () => {
  const source = await fs.readFile(
    path.join(srcRoot, 'components/prototype/TaskQueuePanels.tsx'),
    'utf8',
  );

  assert.match(source, /integrationBundleApplyBlocked\?: boolean/);
  assert.match(source, /const applyBlocked = integrationBundleApplyBlocked \|\| task\.canApply === false \|\| task\.reviewOnly/);
  assert.match(source, /\(integrationBundleApplyBlocked \|\| task\.reviewOnly\)/);
  assert.match(source, /integrationBundleApplyBlocked \? 'Blocked' : 'Review only'/);
  assert.match(source, /requestTaskApply\(task, ui, actions, integrationBundleApplyBlocked\)/);
  assert.equal(source.includes('queueMutation'), false);
});

test('task details status value shows blocked state when integration bundle is blocked', async () => {
  const source = await fs.readFile(
    path.join(srcRoot, 'components/prototype/TaskQueuePanels.tsx'),
    'utf8',
  );

  assert.match(source, /function taskStatusDisplay\(/);
  assert.match(source, /const detailStatus = taskStatusDisplay\(task, integrationBundleApplyBlocked\)/);
  assert.match(source, /<StatusBadge tone=\{detailStatus\.tone\}>\{detailStatus\.label\}<\/StatusBadge>/);
  assert.match(source, /const label = integrationBundleApplyBlocked \? 'Blocked' : task\.status/);
  assert.match(source, /const tone = integrationBundleApplyBlocked \? 'red' as const : toneFromState\(task\.status\)/);
  assert.doesNotMatch(source, /<StatusBadge tone=\{toneFromState\(task\.status\)\}>\{task\.status\}<\/StatusBadge>/);
  assert.equal(source.includes('queueMutation'), false);
});

test('task details proposal risk badge shows blocked state when integration bundle is blocked', async () => {
  const source = await fs.readFile(
    path.join(srcRoot, 'components/prototype/TaskQueuePanels.tsx'),
    'utf8',
  );

  assert.match(source, /function taskProposalRiskDisplay\(/);
  assert.match(source, /proposal: ProposalPreview/);
  assert.match(source, /const label = integrationBundleApplyBlocked \? 'Blocked' : proposal\.risk/);
  assert.match(source, /const tone = integrationBundleApplyBlocked \? 'red' as const : proposal\.tone/);
  assert.match(source, /const detailProposalRisk = taskProposalRiskDisplay\(proposal, integrationBundleApplyBlocked\)/);
  assert.match(source, /<StatusBadge tone=\{detailProposalRisk\.tone\}>\{detailProposalRisk\.label\}<\/StatusBadge>/);
  assert.doesNotMatch(source, /<StatusBadge tone=\{proposal\.tone\}>\{proposal\.risk\}<\/StatusBadge>/);
  assert.equal(source.includes('queueMutation'), false);
});

test('task details apply button is visibly blocked when integration bundle is blocked', async () => {
  const source = await fs.readFile(
    path.join(srcRoot, 'components/prototype/TaskQueuePanels.tsx'),
    'utf8',
  );

  assert.match(source, /const detailApplyButtonLabel = integrationBundleApplyBlocked/);
  assert.match(source, /\? 'Apply Blocked'/);
  assert.match(source, /const detailApplyButtonClassName = integrationBundleApplyBlocked/);
  assert.match(source, /applyBlocked && !integrationBundleApplyBlocked/);
  assert.match(source, /disabled=\{integrationBundleApplyBlocked\}/);
  assert.match(source, /\{detailApplyButtonLabel\}/);
  assert.match(source, /requestTaskApply\(task, ui, actions, integrationBundleApplyBlocked\)/);
  assert.equal(source.includes('queueMutation'), false);
});

test('task details reject button is visibly blocked when integration bundle is blocked', async () => {
  const source = await fs.readFile(
    path.join(srcRoot, 'components/prototype/TaskQueuePanels.tsx'),
    'utf8',
  );

  assert.match(source, /const detailRejectButtonLabel = integrationBundleApplyBlocked/);
  assert.match(source, /\? 'Reject Blocked'/);
  assert.match(source, /const detailRejectButtonClassName = integrationBundleApplyBlocked/);
  assert.match(source, /className=\{detailRejectButtonClassName\}/);
  assert.match(source, /disabled=\{integrationBundleApplyBlocked\}/);
  assert.match(source, /\{detailRejectButtonLabel\}/);
  assert.match(source, /actions\.setTaskStatus\(task\.id, 'Rejected'\)/);
  assert.equal(source.includes('queueMutation'), false);
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

test('director runtime surfaces preview integration bundle readiness without live authority', async () => {
  const { adaptDirectorRuntime } = await importTypeScript('services/operatorAdapters.ts');

  const runtime = adaptDirectorRuntime(
    {
      safety: {
        live_write_policy: 'supervised_fader_gain_only',
        ready_for_live: {
          ready: false,
          checks: {
            mixer_connected: true,
            gain_staging_available: true,
            auto_fader_available: true,
            no_clipping: true,
            automation_unfrozen: true,
          },
        },
        manual_write_supervision: {
          emergency_stopped: false,
          reset_required: false,
        },
      },
      audio_capture: { running: true },
      runtime: {
        proposal_queue_available: true,
        processing: {
          main_pipeline_id: 'main_live_assist',
          main_pipeline_status: 'canonical',
          live_apply_policy: 'supervised_fader_gain_only',
        },
      },
    },
    {
      summary: {
        can_live_execute: false,
        proposal_queue_available: true,
      },
    },
    {
      status: 'ok',
      pipeline_id: 'main_live_assist',
      pipeline_status: 'canonical',
      summary: {
        proposal_count: 2,
        blocker_count: 1,
        queue_mutation: false,
        autonomy_promotion_offline_integration_bundle_ready: false,
        autonomy_promotion_offline_integration_bundle_status: 'blocked',
        autonomy_promotion_offline_integration_bundle_blockers: [
          'offline_integration_bundle_contract_not_ready',
        ],
        autonomy_promotion_offline_integration_bundle_missing_checks: [
          'operator_report_bound',
        ],
        autonomy_promotion_offline_integration_bundle_source_authority_denied: true,
        autonomy_promotion_offline_integration_bundle_live_authority_granted: false,
        autonomy_promotion_offline_integration_bundle_readiness_closure_allowed: false,
      },
      executor_policy: {
        live_apply_policy: 'supervised_fader_gain_only',
        queue_mutation: false,
      },
    },
  );

  const integrationBundle = runtime.statusItems.find(item => item.label === 'Integration bundle');
  assert.deepEqual(integrationBundle, {
    label: 'Integration bundle',
    value: 'blocked / ready:false',
    tone: 'red',
  });
  assert.equal(runtime.blockedReasons.includes('integration_bundle_status:blocked'), true);
  assert.equal(
    runtime.blockedReasons.includes('integration_bundle_blocker:offline_integration_bundle_contract_not_ready'),
    true,
  );
  assert.equal(runtime.blockedReasons.includes('integration_bundle_missing_check:operator_report_bound'), true);
  assert.equal(runtime.blockedReasons.includes('integration_bundle_source_authority_denied:true'), true);
  assert.equal(runtime.blockedReasons.includes('integration_bundle_live_authority_granted:false'), true);
  assert.equal(runtime.blockedReasons.includes('integration_bundle_readiness_closure_allowed:false'), true);
  assert.equal(
    runtime.currentGoal,
    'Resolve the offline integration bundle blocker before promotion planning. Keep proposals review-only; no live authority, readiness closure or console apply is granted.',
  );
  assert.equal(runtime.proposalQueueLabel, 'Promotion planning blocked');
  assert.equal(runtime.proposalQueueTone, 'red');
  assert.equal(runtime.manualGateLabel, 'Apply blocked');
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
