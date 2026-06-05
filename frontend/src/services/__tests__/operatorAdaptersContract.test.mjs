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
          declared_fixture_path_count: 11,
          declared_test_name_count: 28,
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
      'raw_mixer_state_physical_readback_contract',
      'channel_basic_controls_physical_readback_contract',
      'channel_processing_physical_readback_contract',
      'bus_group_main_dca_physical_readback_contract',
      'matrix_state_physical_readback_contract',
      'read_surface_remediation_plan',
      'autonomy_evidence:fader_supervised_promotion',
      'autonomy_evidence:gain_trim_supervised_promotion',
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
    'fixture_tests:28',
    'contract_gap:true',
    'raw_contract_available:true',
    'raw_contract_status:contract_ready_physical_fixture_missing',
    'raw_contract_schema:raw_mixer_state_physical_readback.v1',
    'raw_contract_gap:true',
    'raw_contract_live_authority:false',
    'raw_contract_target_gate:true',
    'channel_contract_available:true',
    'channel_contract_status:contract_ready_physical_fixture_missing',
    'channel_contract_schema:channel_basic_controls_physical_readback.v1',
    'channel_contract_gap:true',
    'channel_contract_live_authority:false',
    'channel_contract_target_gate:true',
    'processing_contract_available:true',
    'processing_contract_status:contract_ready_physical_fixture_missing',
    'processing_contract_schema:channel_processing_physical_readback.v1',
    'processing_contract_gap:true',
    'processing_contract_live_authority:false',
    'processing_contract_target_gate:true',
    'bus_contract_available:true',
    'bus_contract_status:contract_ready_physical_fixture_missing',
    'bus_contract_schema:bus_group_main_dca_physical_readback.v1',
    'bus_contract_gap:true',
    'bus_contract_live_authority:false',
    'bus_contract_target_gate:true',
    'matrix_contract_available:true',
    'matrix_contract_status:contract_ready_physical_fixture_missing',
    'matrix_contract_schema:matrix_state_physical_readback.v1',
    'matrix_contract_gap:true',
    'matrix_contract_live_authority:false',
    'matrix_contract_target_gate:true',
  ]);
  assert.match(proofEvidence.blockedReason, /Read-only artifact source/);
  assert.match(proofEvidence.nextStep, /overlay: dry_run_only/);
  assert.match(proofEvidence.nextStep, /age:120s stale_after:86400s stale:false/);
  assert.match(proofEvidence.nextStep, /read_surface_gaps:10/);
  assert.match(proofEvidence.nextStep, /fixture_paths:11/);
  assert.match(proofEvidence.nextStep, /test_names:28/);
  assert.match(proofEvidence.nextStep, /contract_gap:true/);
  assert.match(proofEvidence.nextStep, /gap_surfaces:raw_mixer_state_cache/);
  assert.match(proofEvidence.nextStep, /next_readback:raw_mixer_state_cache/);
  assert.match(proofEvidence.nextStep, /source_next_readback:raw_mixer_state_cache/);
  assert.match(proofEvidence.nextStep, /next_readback_gate:true/);
  assert.match(proofEvidence.nextStep, /raw_contract:contract_ready_physical_fixture_missing/);
  assert.match(proofEvidence.nextStep, /raw_contract_gap:true/);
  assert.match(proofEvidence.nextStep, /raw_contract_live_authority:false/);
  assert.match(proofEvidence.nextStep, /raw_contract_target_gate:true/);
  assert.match(proofEvidence.nextStep, /channel_contract:contract_ready_physical_fixture_missing/);
  assert.match(proofEvidence.nextStep, /channel_contract_gap:true/);
  assert.match(proofEvidence.nextStep, /channel_contract_live_authority:false/);
  assert.match(proofEvidence.nextStep, /channel_contract_target_gate:true/);
  assert.match(proofEvidence.nextStep, /processing_contract:contract_ready_physical_fixture_missing/);
  assert.match(proofEvidence.nextStep, /processing_contract_gap:true/);
  assert.match(proofEvidence.nextStep, /processing_contract_live_authority:false/);
  assert.match(proofEvidence.nextStep, /processing_contract_target_gate:true/);
  assert.match(proofEvidence.nextStep, /bus_contract:contract_ready_physical_fixture_missing/);
  assert.match(proofEvidence.nextStep, /bus_contract_gap:true/);
  assert.match(proofEvidence.nextStep, /bus_contract_live_authority:false/);
  assert.match(proofEvidence.nextStep, /bus_contract_target_gate:true/);
  assert.match(proofEvidence.nextStep, /matrix_contract:contract_ready_physical_fixture_missing/);
  assert.match(proofEvidence.nextStep, /matrix_contract_gap:true/);
  assert.match(proofEvidence.nextStep, /matrix_contract_live_authority:false/);
  assert.match(proofEvidence.nextStep, /matrix_contract_target_gate:true/);
  assert.match(proofEvidence.nextStep, /dlive_state/);

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

  assert.equal(rows.length, 6);
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
    'fixture_tests:0',
    'contract_gap:false',
    'raw_contract_available:false',
    'raw_contract_status:missing',
    'raw_contract_schema:',
    'raw_contract_gap:false',
    'raw_contract_live_authority:false',
    'raw_contract_target_gate:false',
    'channel_contract_available:false',
    'channel_contract_status:missing',
    'channel_contract_schema:',
    'channel_contract_gap:false',
    'channel_contract_live_authority:false',
    'channel_contract_target_gate:false',
    'processing_contract_available:false',
    'processing_contract_status:missing',
    'processing_contract_schema:',
    'processing_contract_gap:false',
    'processing_contract_live_authority:false',
    'processing_contract_target_gate:false',
    'bus_contract_available:false',
    'bus_contract_status:missing',
    'bus_contract_schema:',
    'bus_contract_gap:false',
    'bus_contract_live_authority:false',
    'bus_contract_target_gate:false',
    'matrix_contract_available:false',
    'matrix_contract_status:missing',
    'matrix_contract_schema:',
    'matrix_contract_gap:false',
    'matrix_contract_live_authority:false',
    'matrix_contract_target_gate:false',
  ]);

  assert.equal(rows[1].moduleId, 'raw_mixer_state_physical_readback_contract');
  assert.equal(rows[1].domain, 'read_surface_contract');
  assert.equal(rows[1].status, 'contract_missing');
  assert.equal(rows[1].canWriteLive, false);
  assert.deepEqual(rows[1].liveWriteKinds, []);
  assert.match(rows[1].blockedReason, /contract is missing/);

  assert.equal(rows[2].moduleId, 'channel_basic_controls_physical_readback_contract');
  assert.equal(rows[2].domain, 'read_surface_contract');
  assert.equal(rows[2].status, 'contract_missing');
  assert.equal(rows[2].canWriteLive, false);
  assert.deepEqual(rows[2].liveWriteKinds, []);
  assert.match(rows[2].blockedReason, /contract is missing/);

  assert.equal(rows[3].moduleId, 'channel_processing_physical_readback_contract');
  assert.equal(rows[3].domain, 'read_surface_contract');
  assert.equal(rows[3].status, 'contract_missing');
  assert.equal(rows[3].canWriteLive, false);
  assert.deepEqual(rows[3].liveWriteKinds, []);
  assert.match(rows[3].blockedReason, /contract is missing/);

  assert.equal(rows[4].moduleId, 'bus_group_main_dca_physical_readback_contract');
  assert.equal(rows[4].domain, 'read_surface_contract');
  assert.equal(rows[4].status, 'contract_missing');
  assert.equal(rows[4].canWriteLive, false);
  assert.deepEqual(rows[4].liveWriteKinds, []);
  assert.match(rows[4].blockedReason, /contract is missing/);

  assert.equal(rows[5].moduleId, 'matrix_state_physical_readback_contract');
  assert.equal(rows[5].domain, 'read_surface_contract');
  assert.equal(rows[5].status, 'contract_missing');
  assert.equal(rows[5].canWriteLive, false);
  assert.deepEqual(rows[5].liveWriteKinds, []);
  assert.match(rows[5].blockedReason, /contract is missing/);
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
