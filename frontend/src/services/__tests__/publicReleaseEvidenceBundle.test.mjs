import test from 'node:test';
import assert from 'node:assert/strict';
import crypto from 'node:crypto';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { spawnSync } from 'node:child_process';
import { createRequire } from 'node:module';
import { fileURLToPath, pathToFileURL } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const workspaceRoot = path.resolve(__dirname, '../../../..');
const require = createRequire(import.meta.url);
const {
  BACKEND_NETWORK_SANDBOX_PROBES,
  BACKEND_SANDBOX_PROFILE_SHA256
} = require('../../../public/electronSecurity.js');
const passingSandboxProbeResults = Object.fromEntries(
  Object.keys(BACKEND_NETWORK_SANDBOX_PROBES).map(name => [name, true])
);
const bundlePath = path.resolve(
  __dirname,
  '../../../scripts/public-release-evidence-bundle.mjs'
);
const releaseEvidencePath = path.resolve(
  __dirname,
  '../../../scripts/release-evidence.mjs'
);
const launcherPath = path.resolve(
  __dirname,
  '../../../scripts/run-public-release-evidence-bundle'
);
const m006ContractPath = path.resolve(
  __dirname,
  '../../../../backend/contracts/m110_m006_restore_20260711.json'
);
const {
  buildPublicReleaseEvidenceBundle,
  computeCp05SourceBinding,
  computeM110FixedOperationDigests,
  createPublicReleaseEvidenceManifest,
  publicReleaseEvidenceTestHooks,
  validateCp05Markers,
  validateM006Contract,
  validateM110AttendedEvidence,
  verifyPublicReleaseEvidenceBundle
} = await import(pathToFileURL(bundlePath));
const verifyReleasePath = path.resolve(
  __dirname,
  '../../../scripts/verify-macos-release.mjs'
);
const {
  createPortableReleaseManifest,
  EXPECTED_RUNTIME_CRITICAL_SOURCE_PATHS,
  packagedSourceManifestSha256,
  runtimeIdentityManifestSha256,
  verifyRetainedReleaseManifestEvidence
} = await import(pathToFileURL(verifyReleasePath));
const {
  artifactFileIdentity,
  fileIdentity,
  hashTree,
  pinReleaseEvidenceHelperRuntime,
  publishNewArtifactFileNoFollow,
  runReleaseEvidenceGit,
  stableFlatDirectoryFileIdentitiesNoFollow,
  stableTreeIdentityNoFollow
} = await import(pathToFileURL(releaseEvidencePath));

function canonicalJson(value) {
  if (Array.isArray(value)) return `[${value.map(canonicalJson).join(',')}]`;
  if (value && typeof value === 'object') {
    return `{${Object.keys(value).sort().map(key => (
      `${JSON.stringify(key)}:${canonicalJson(value[key])}`
    )).join(',')}}`;
  }
  return JSON.stringify(value);
}

const PYTHON_ROUTE_FLOAT_PATHS = new Set([
  'evidence.capture.conservative_true_peak_dbtp',
  'evidence.capture.headroom_ceiling_dbtp',
  'evidence.capture.sample_peaks_dbfs.47',
  'evidence.capture.sample_peaks_dbfs.48',
  'evidence.capture.true_peak_allowance_db',
  'expires_at_epoch',
  'issued_at_epoch'
]);

function pythonRouteCanonicalJson(value, pathParts = []) {
  if (Array.isArray(value)) {
    return `[${value.map((item, index) => (
      pythonRouteCanonicalJson(item, [...pathParts, String(index)])
    )).join(',')}]`;
  }
  if (value && typeof value === 'object') {
    return `{${Object.keys(value).sort().map(key => (
      `${JSON.stringify(key)}:${pythonRouteCanonicalJson(value[key], [...pathParts, key])}`
    )).join(',')}}`;
  }
  if (typeof value === 'number' && PYTHON_ROUTE_FLOAT_PATHS.has(pathParts.join('.'))) {
    return publicReleaseEvidenceTestHooks.pythonCanonicalFloat(value);
  }
  return JSON.stringify(value);
}

function canonicalBytes(value) {
  return Buffer.from(`${canonicalJson(value)}\n`, 'utf8');
}

function sha256(bytes) {
  return crypto.createHash('sha256').update(bytes).digest('hex');
}

function rebindRouteBytes(fixture) {
  fixture.routeProof.evidence_sha256 = `sha256:${sha256(Buffer.from(
    pythonRouteCanonicalJson(fixture.routeProof.evidence, ['evidence']),
    'utf8'
  ))}`;
  fixture.routeBytes = Buffer.from(pythonRouteCanonicalJson(fixture.routeProof), 'utf8');
  fixture.routeIdentity = {
    size: fixture.routeBytes.length,
    sha256: sha256(fixture.routeBytes)
  };
  return fixture;
}

function productionRuntimeEnvironment(extra = {}) {
  return {
    HOME: '/var/empty',
    TMPDIR: '/tmp',
    LANG: 'C',
    LC_ALL: 'C',
    PATH: `${path.dirname(process.execPath)}:/usr/bin:/bin:/usr/sbin:/sbin`,
    AUTOMIXER_PUBLIC_RELEASE_EVIDENCE_CLEAN_ENV: '1',
    ...extra
  };
}

function withProductionRuntime(callback) {
  const previousEnvironment = { ...process.env };
  const previousCwd = process.cwd();
  const previousExecArgv = [...process.execArgv];
  try {
    for (const name of Object.keys(process.env)) delete process.env[name];
    Object.assign(process.env, productionRuntimeEnvironment());
    if (previousEnvironment.__CF_USER_TEXT_ENCODING !== undefined) {
      process.env.__CF_USER_TEXT_ENCODING = previousEnvironment.__CF_USER_TEXT_ENCODING;
    }
    process.chdir(path.join(workspaceRoot, 'frontend'));
    process.execArgv.splice(0, process.execArgv.length);
    return callback();
  } finally {
    process.execArgv.splice(0, process.execArgv.length, ...previousExecArgv);
    process.chdir(previousCwd);
    for (const name of Object.keys(process.env)) delete process.env[name];
    Object.assign(process.env, previousEnvironment);
  }
}

function cp05Fixture() {
  const sourceBinding = {
    schema: 'automixer.m110_cp05_git_tree_binding.v1',
    sha256: `sha256:${'1'.repeat(64)}`,
    tracked_entry_count: 42
  };
  const evidence = {
    checks: {
      focused_tests: true,
      frontend_build_and_tests: true,
      full_backend_suite: true,
      independent_review: true,
      no_wing_release_check: true,
      released_development_lease: true,
      static_scan: true
    },
    schema: 'automixer.m110_cp05_software_validation.v1',
    source_binding_sha256: sourceBinding.sha256,
    status: 'passed'
  };
  const evidenceBytes = canonicalBytes(evidence);
  const gate = {
    checkpoint: 'M110-CP05',
    hardware_enabled: true,
    schema: 'automixer.m110_cp05_hardware_gate.v1',
    software_evidence_sha256: `sha256:${sha256(evidenceBytes)}`
  };
  const gateBytes = canonicalBytes(gate);
  const verified = validateCp05Markers({
    gate,
    gateBytes,
    evidence,
    evidenceBytes,
    sourceBinding
  });
  return { gate, gateBytes, evidence, evidenceBytes, sourceBinding, verified };
}

function routePass(serial, firmware) {
  const values = {
    '/$syscfg/$serial': serial,
    '/$syscfg/$firmware': firmware,
    '/io/out/USB/47/grp': 'MAIN',
    '/io/out/USB/47/in': 0,
    '/io/out/USB/48/grp': 'MAIN',
    '/io/out/USB/48/in': 1
  };
  return {
    addresses: Object.keys(values),
    confirmed_round_trips_required: 2,
    errors: [],
    readbacks: Object.fromEntries(Object.entries(values).map(([address, value]) => [
      address,
      {
        address,
        args: [value],
        confirmed_round_trips: 2,
        status: 'received',
        value
      }
    ])),
    reason: 'fixed_master_route_pass_verified',
    values,
    verified: true
  };
}

function checkpoint(name, reason) {
  return {
    verified: true,
    checkpoint: name,
    reason,
    console_mutation_write_attempt_count: 0,
    terminal: {
      verified: true,
      checkpoint: name,
      reason,
      audio_status: 'passed',
      console_mutation_write_attempt_count: 0,
      live_mixer_writes: false,
      live_authority_granted: false,
      readiness_closure_allowed: false
    }
  };
}

function immutableM006ContractFixture() {
  const bytes = fs.readFileSync(m006ContractPath);
  return {
    contract: JSON.parse(bytes.toString('utf8')),
    contractIdentity: { size: bytes.length, sha256: sha256(bytes) }
  };
}

function resetOperations() {
  const parameters = [
    ['eq/on', 'eq_enabled'],
    ['flt/lc', 'hpf_enabled'],
    ['gate/on', 'gate_enabled'],
    ['dyn/on', 'compressor_enabled']
  ];
  return Array.from({ length: 16 }, (_, index) => index + 1).flatMap(channel => (
    parameters.map(([suffix, parameter]) => ({
      operationId: `m110-reset-ch${String(channel).padStart(2, '0')}-${suffix.replace('/', '-')}`,
      address: `/ch/${channel}/${suffix}`,
      parameter
    }))
  ));
}

function durableReceipt({
  role,
  proposalId,
  operationDigest,
  bundleKind,
  operationIds,
  scopeAddresses,
  summaryEpoch,
  sourceOperationIds = [],
  reviewPreimageSha256 = '',
  reset = false,
  commitCapabilityKind = 'maintenance_dsp_reset_commit_once',
  proofReceiptCharacter = 'a',
  digestCharacter = '8'
}) {
  const proofRun = role === 'proof';
  const capabilityKind = proofRun
    ? 'restore_only_proof'
    : commitCapabilityKind;
  const status = proofRun ? 'proof_passed' : 'applied';
  const reviewContentDigest = `sha256:${digestCharacter.repeat(64)}`;
  const resultDigest = `sha256:${(digestCharacter === 'f' ? 'e' : 'f').repeat(64)}`;
  const terminal = {
    type: 'operator_proposal_apply_result',
    contract_version: 2,
    status,
    success: true,
    proposal_id: proposalId,
    proposal_status: status,
    apply_status: status,
    accepted: reset ? false : !proofRun,
    owner_review_required: reset ? true : proofRun,
    live_authority_granted: false,
    readiness_closure_allowed: false,
    proof_run: proofRun,
    restore_after_verify: proofRun,
    capability_kind: capabilityKind,
    operation_digest: operationDigest,
    bundle_kind: bundleKind,
    review_preimage_sha256: reset ? reviewPreimageSha256 : null,
    proof_target_verified: proofRun,
    no_console_send_proven: false,
    physical_verified: true,
    physical_state_restored: proofRun,
    terminal_after_console_send: true,
    write_state_unknown: false,
    rollback_attempted: proofRun,
    rollback_succeeded: proofRun,
    rollback_state_unknown: false,
    rollback_status: proofRun ? 'verified' : 'not_attempted',
    manual_review_required: false,
    emergency_stop_activated: false,
    emergency_stop_required: false,
    wal_terminal: proofRun ? 'proof_restored' : 'committed',
    audio_recheck_passed: true,
    result_detail_available: true,
    result_evidence_pinned: false,
    applied_operation_ids: [...operationIds],
    rollback_operation_ids: proofRun ? [...operationIds] : [],
    already_satisfied_operation_ids: [],
    result_digest: resultDigest,
    proposal: {
      id: proposalId,
      status,
      review_contract_version: 2,
      review_content_digest: reviewContentDigest,
      result_digest: resultDigest,
      result_detail_available: true,
      result_evidence_pinned: false
    }
  };
  if (proofRun) terminal.proof_receipt_id = null;
  else {
    terminal.parent_commit_recorded = true;
    if (!reset) {
      terminal.proof_receipt_id = `m110proof_v1_${proofReceiptCharacter.repeat(64)}`;
    }
  }
  if (reset) {
    Object.assign(terminal, {
      before_manifest_sha256: `sha256:${'1'.repeat(64)}`,
      after_manifest_sha256: `sha256:${'2'.repeat(64)}`,
      verification_cut_reset_surface_sha256: `sha256:${'3'.repeat(64)}`,
      verification_cut_preservation_manifest_sha256: `sha256:${'2'.repeat(64)}`,
      no_outside_mutation_attempt_proven: true,
      current_state_guaranteed_at_response: false,
      fresh_readback_required_before_next_stage: true,
      attended_no_touch_required_during_execution: true,
      later_console_event_invalidates_readiness_freshness: true
    });
  }
  const receipt = {
    verified: true,
    proposal_id: proposalId,
    capability_kind: capabilityKind,
    review_contract_version: 2,
    review_content_digest: reviewContentDigest,
    operation_digest: operationDigest,
    review_preimage_sha256: reset ? reviewPreimageSha256 : '',
    bundle_kind: bundleKind,
    operation_count: operationIds.length,
    scope_addresses: [...scopeAddresses],
    review_operation_ids: [...operationIds],
    terminal,
    operator_checkpoint: {
      verified: true,
      expected_literal: operationDigest,
      operator_response_matched: true,
      captured_at_epoch: summaryEpoch - 20,
      deadline_epoch: null,
      proposal_id: proposalId,
      capability_kind: capabilityKind,
      operation_digest: operationDigest
    },
    token_retained: false
  };
  if (!reset) receipt.source_operation_ids = [...sourceOperationIds];
  return receipt;
}

function laneCommitFixture({ lane, summaryEpoch, digestCharacter }) {
  const processor = lane === 'processor';
  const operationIds = [
    processor
      ? 'm110:processor:eq:ch1:0123456789abcdef'
      : 'soundcheck-operation-1'
  ];
  const sourceOperationIds = [
    processor ? 'source-processor-operation-1' : operationIds[0]
  ];
  const scopeAddresses = [processor ? '/ch/1/eq/on' : '/ch/1/fdr'];
  const bundleKind = processor
    ? 'processor_enable_bundle'
    : 'single_soundcheck_operation';
  const operationDigest = `sha256:${digestCharacter.repeat(64)}`;
  const proposalPrefix = processor
    ? 'm110-processor'
    : 'm110-soundcheck-child';
  const commitCapabilityKind = processor
    ? 'processor_enable_commit_once'
    : 'soundcheck_child_commit_once';
  const proof = durableReceipt({
    role: 'proof',
    proposalId: `${proposalPrefix}-proof-${(processor ? '1' : '3').repeat(32)}`,
    operationDigest,
    bundleKind,
    operationIds,
    scopeAddresses,
    sourceOperationIds,
    summaryEpoch,
    commitCapabilityKind,
    digestCharacter: processor ? '4' : '6'
  });
  const commit = durableReceipt({
    role: 'commit',
    proposalId: `${proposalPrefix}-commit-${(processor ? '2' : '5').repeat(32)}`,
    operationDigest,
    bundleKind,
    operationIds,
    scopeAddresses,
    sourceOperationIds,
    summaryEpoch,
    commitCapabilityKind,
    proofReceiptCharacter: processor ? 'b' : 'c',
    digestCharacter: processor ? '5' : '7'
  });
  return {
    state: 'committed',
    success: true,
    lane,
    operation_digest: operationDigest,
    source_operation_ids: sourceOperationIds,
    proof,
    commit
  };
}

function acceptedM110Fixture(overrides = {}) {
  const source = overrides.source || {
    head: 'a'.repeat(40),
    tree: 'b'.repeat(40),
    cleanCheckout: true
  };
  const cp05 = overrides.cp05 || cp05Fixture().verified;
  const m006Contract = immutableM006ContractFixture();
  const expectedM006 = validateM006Contract(m006Contract);
  const proofId = 'c'.repeat(32);
  const serial = '0100Q1V0604AAE';
  const firmware = '3.0.5-4-ga4ea833d:release';
  // The Python writer retains fractional epoch precision but truncates the UTC
  // rendering to whole seconds.
  const issued = 1_700_000_000.789;
  const summaryEpoch = issued + 60;
  const expires = issued + 1800;
  const runtime = {
    head: source.head,
    tree: source.tree,
    tracked_tree_clean: true,
    identity_schema: 'automixer.backend_runtime_identity.v2',
    manifest_sha256: 'e'.repeat(64),
    launch_manifest_sha256: 'f'.repeat(64),
    instance_nonce: '0123456789abcdef0123456789abcdef'
  };
  const usbSerial = `${serial}214`;
  const usbLocationId = 0x00100000;
  const coreaudioIdentity = {
    input_channels: 48,
    manufacturer: 'BEHRINGER',
    output_channels: 48,
    sample_rate_hz: 48000,
    transport: 'usb'
  };
  const usbIdentity = {
    location_id: usbLocationId,
    product_id: 1291,
    serial: usbSerial,
    vendor_id: 5015
  };
  const hardwareBinding = {
    schema: 'automixer.wing_capture_hardware_identity.v1',
    device_name: 'WING',
    device_index: 1,
    console_serial: serial,
    coreaudio: coreaudioIdentity,
    usb: usbIdentity,
    serial_binding: 'chassis_serial_plus_three_digit_usb_interface'
  };
  const hardwareIdentity = {
    ...hardwareBinding,
    verified: true,
    reason: 'physical_wing_usb_capture_bound_to_console',
    hardware_uid: `usb:1397:050b:${usbSerial}:${usbLocationId.toString(16).padStart(8, '0')}`,
    binding_sha256: sha256(Buffer.from(canonicalJson(hardwareBinding), 'utf8'))
  };
  const routeProof = {
    accepted: true,
    evidence: {
      request: {
        approval_id: 'unit',
        request_id: 'route-unit',
        request_nonce: '0'.repeat(32)
      },
      runtime,
      console: {
        client_type: 'EnhancedOSCClient',
        connection_generation: 1,
        firmware,
        serial,
        transport_generation: 1,
        transport_instance_id: 1
      },
      route: {
        addresses: [
          '/$syscfg/$serial',
          '/$syscfg/$firmware',
          '/io/out/USB/47/grp',
          '/io/out/USB/47/in',
          '/io/out/USB/48/grp',
          '/io/out/USB/48/in'
        ],
        complete_passes_required: 2,
        confirmed_round_trips_required: 2,
        consistent: true,
        expected: {
          '/io/out/USB/47/grp': 'MAIN',
          '/io/out/USB/47/in': 0,
          '/io/out/USB/48/grp': 'MAIN',
          '/io/out/USB/48/in': 1
        },
        logical_query_count: 12,
        pass_1: routePass(serial, firmware),
        pass_2: routePass(serial, firmware)
      },
      capture: {
        capture_instance_id: 1,
        capture_stream_instance_id: 2,
        captured_sample_count_before: 1000,
        captured_sample_count_after: 14000,
        captured_sample_count_snapshot: 14000,
        channel_count: 48,
        conservative_true_peak_dbtp: -3,
        device_index: 1,
        device_name: 'WING',
        hardware_identity: hardwareIdentity,
        headroom_ceiling_dbtp: -1,
        master_channels: [47, 48],
        reason: 'fresh_real_master_capture_verified',
        required_fresh_sample_count: 12000,
        sample_counts: { '47': 12000, '48': 12000 },
        sample_peaks_dbfs: { '47': -6, '48': -6 },
        sample_rate_hz: 48000,
        source_type: 'sounddevice',
        true_peak_allowance_db: 3,
        verified: true
      },
      safety: {
        authenticated_control_session: true,
        console_mutation_write_attempt_count: 0,
        console_writer_scope_acquired: true,
        emergency_generation: 0,
        emergency_generation_stable: true,
        live_authority_granted: false,
        normal_live_allowlist_changed: false,
        observation_only: true,
        operator_mode: 'analyze',
        readiness_closure_allowed: false,
        recovery_transaction_count: 0,
        wal_integrity_ok: true
      }
    },
    evidence_sha256: null,
    expires_at_epoch: expires,
    issued_at_epoch: issued,
    issued_at_utc: new Date(Math.floor(issued) * 1000).toISOString().replace('.000Z', 'Z'),
    proof_id: proofId,
    schema: 'automixer.wing_master_capture_route_proof.v1'
  };
  const evidenceSha = `sha256:${sha256(
    Buffer.from(pythonRouteCanonicalJson(routeProof.evidence, ['evidence']), 'utf8')
  )}`;
  routeProof.evidence_sha256 = evidenceSha;
  const routeBytes = Buffer.from(pythonRouteCanonicalJson(routeProof), 'utf8');
  const routeIdentity = { size: routeBytes.length, sha256: sha256(routeBytes) };
  const fixedOperationDigests = computeM110FixedOperationDigests({
    routeProof,
    routeIdentity,
    source,
    m006Contract
  });
  const routeResult = {
    accepted: true,
    artifact: {
      evidence_sha256: evidenceSha,
      path: `/private/audit/${proofId}.json`,
      schema: routeProof.schema,
      sha256: routeIdentity.sha256,
      size: routeIdentity.size
    },
    capture_device: 'WING',
    capture_device_index: 1,
    capture_hardware_uid: routeProof.evidence.capture.hardware_identity.hardware_uid,
    capture_usb_serial: usbSerial,
    conservative_true_peak_dbtp: -3,
    console_mutation_write_attempt_count: 0,
    console_serial: serial,
    expires_at_epoch: expires,
    firmware_version: firmware,
    issued_at_epoch: issued,
    live_authority_granted: false,
    master_channels: [47, 48],
    proof_id: proofId,
    readiness_closure_allowed: false,
    reason: 'process_private_master_capture_route_authority_issued',
    status: 'issued',
    success: true,
    type: 'wing_master_route_proof_result'
  };
  const route = {
    verified: true,
    reason: 'persistent_proof_owned_master_route_session_established',
    route_result: routeResult,
    artifact_verification: {
      evidence_sha256: evidenceSha,
      hardware_binding_verified: true,
      live_authority_granted: false,
      path: `/private/audit/${proofId}.json`,
      proof_id: proofId,
      reason: 'runner_independently_verified_master_route_artifact',
      request_binding_verified: true,
      sha256: routeIdentity.sha256,
      verified: true
    },
    cp05_hardware_gate: cp05,
    same_websocket_retained: true,
    backend_process_retained: true,
    live_authority_granted: false,
    readiness_closure_allowed: false
  };
  const stateSha = `sha256:${'2'.repeat(64)}`;
  const ledgerSha = `sha256:${'3'.repeat(64)}`;
  const runId = 'm110-unit-run';
  const audit = {
    type: 'm110_soundcheck_final_audit_result',
    contract_version: 1,
    status: 'verified',
    success: true,
    verified: true,
    reason: 'm110_show_mix_full_reconciliation_verified',
    run_id: runId,
    mix_readiness: 'show_mix_ready',
    audited_address_count: 51,
    audit_pass_count: 2,
    operator_listening_required: true,
    console_mutation_write_attempt_count: 0,
    live_mixer_writes: false,
    live_authority_granted: false,
    readiness_closure_allowed: false,
    verified_state_sha256: stateSha,
    committed_surface_ledger_sha256: ledgerSha
  };
  const readiness = {
    run_id: runId,
    plan_revision: 1,
    plan_fingerprint: '4'.repeat(64),
    status: 'show_mix_ready',
    mix_readiness: 'show_mix_ready',
    safety_state: 'clear',
    execution_status: 'complete',
    proposal_only: false,
    global_blockers: [],
    unresolved_required_operation_ids: [],
    unresolved_reasons: {},
    readiness_evidence_complete: true,
    source_coverage_fresh: true,
    main_47_48_evidence_fresh: true,
    safe_core_stage_coverage_complete: true,
    show_mix_stage_coverage_complete: true,
    live_authority_changed: false
  };
  const processorState = {
    state: 'complete',
    success: true,
    reason: 'm110_no_unresolved_processor_candidate',
    run_id: runId,
    residuals: [],
    soundcheck_readiness: null,
    replan_available: false,
    replan_from_plan_revision: null,
    replan_to_plan_revision: null,
    replan_split_operation_count: 0,
    replan_safe_alternative_operation_count: 0
  };
  const soundcheckState = {
    state: 'complete',
    success: true,
    reason: 'm110_no_unresolved_soundcheck_child',
    run_id: runId,
    residuals: [],
    soundcheck_readiness: readiness,
    replan_available: false,
    replan_from_plan_revision: null,
    replan_to_plan_revision: null,
    replan_split_operation_count: 0,
    replan_safe_alternative_operation_count: 0
  };
  const processorCommit = laneCommitFixture({
    lane: 'processor',
    summaryEpoch,
    digestCharacter: '8'
  });
  const soundcheckCommit = laneCommitFixture({
    lane: 'soundcheck',
    summaryEpoch,
    digestCharacter: '9'
  });
  const lanes = {
    verified: true,
    status: 'complete',
    reason: 'm110_interleaved_show_mix_ready',
    lane_states: {
      processor: processorState,
      soundcheck: soundcheckState
    },
    processor: processorState,
    soundcheck: soundcheckState,
    soundcheck_readiness: readiness,
    commit_count: 2,
    commits: {
      processor: [processorCommit],
      soundcheck: [soundcheckCommit]
    },
    reanalyses: [],
    replans: []
  };
  const execution = {
    verified: true,
    status: 'show_mix_ready',
    reason: 'm110_attended_show_mix_hw05_verified_and_torn_down',
    identity: {
      manifest_sha256: runtime.manifest_sha256,
      instance_nonce: runtime.instance_nonce
    },
    cp05_hardware_gate: cp05,
    route,
    arm: {
      verified: true,
      reason: 'm110_supervised_capability_only_armed',
      route_proof_id: proofId,
      normal_operator_mode_unchanged: true,
      normal_live_allowlist_changed: false,
      live_authority_granted: false,
      readiness_closure_allowed: false
    },
    reset_containment_operator_ack: {
      verified: true,
      expected_literal: 'CONTAIN_ALL_PA_MONITOR_OUTPUTS_AND_MUTE_MAIN_1_FOR_M110_RESET',
      operator_response_matched: true,
      operator_attested_external_output_containment: true
    },
    reset_muted_checkpoint: checkpoint(
      'reset_muted',
      'm110_reset_muted_containment_verified'
    ),
    reset: null,
    m006_restore: null,
    soundcheck_unmute_operator_ack: {
      verified: true,
      expected_literal: 'UNMUTE_MAIN_1_FOR_M110_SOUNDCHECK',
      operator_response_matched: true
    },
    soundcheck_unmuted_checkpoint: checkpoint(
      'soundcheck_unmuted',
      'm110_soundcheck_unmuted_audio_verified'
    ),
    final_soundcheck_unmuted_checkpoint: checkpoint(
      'soundcheck_unmuted',
      'm110_soundcheck_unmuted_audio_verified'
    ),
    analysis: {
      verified: true,
      reason: 'm110_soundcheck_manifest_registered',
      run_id: runId,
      plan_revision: 1,
      imported_count: 2,
      soundcheck_readiness: readiness,
      shared_capture_retained: true
    },
    lanes,
    processors: lanes.processor,
    soundcheck: lanes.soundcheck,
    soundcheck_readiness: readiness,
    final_audit_before_listening: { ...audit },
    hw05_operator_listening_ack: {
      verified: true,
      expected_literal: `CONFIRM_M110_HW05_LISTENING_${stateSha}`,
      operator_response_matched: true,
      captured_at_epoch: summaryEpoch - 10,
      deadline_epoch: summaryEpoch + 10,
      verified_state_sha256: stateSha
    },
    final_audit_after_listening: { ...audit },
    teardown: {
      verified: true,
      capture_stopped: true,
      mixer_disconnected: true,
      websocket_closed: true,
      backend_process_terminated: true,
      route_authority_retained: false,
      errors: [],
      backend_shutdown: {
        terminated: true,
        clean_shutdown: true,
        forced: false,
        errors: []
      }
    },
    live_authority_granted: false,
    readiness_closure_allowed: true
  };
  const resetRows = resetOperations();
  const resetOperationIds = resetRows.map(row => row.operationId);
  const resetScope = resetRows.map(row => row.address);
  const resetDigest = fixedOperationDigests.reset;
  const resetPreimage = `sha256:${'5'.repeat(64)}`;
  execution.reset = {
    verified: true,
    reason: 'm110_fixed_maintenance_reset_committed',
    operation_digest: resetDigest,
    review_preimage_sha256: resetPreimage,
    reset_applied: true,
    same_websocket_retained: true,
    backend_process_retained: true,
    token_retained: false,
    live_authority_granted: false,
    readiness_closure_allowed: false,
    scope_addresses: resetScope,
    proof: durableReceipt({
      role: 'proof',
      proposalId: `m110-maintenance-reset-proof-${'1'.repeat(32)}`,
      operationDigest: resetDigest,
      bundleKind: 'maintenance_dsp_reset_bundle',
      operationIds: resetOperationIds,
      scopeAddresses: resetScope,
      reviewPreimageSha256: resetPreimage,
      summaryEpoch,
      reset: true,
      digestCharacter: '6'
    }),
    commit: durableReceipt({
      role: 'commit',
      proposalId: `m110-maintenance-reset-commit-${'2'.repeat(32)}`,
      operationDigest: resetDigest,
      bundleKind: 'maintenance_dsp_reset_bundle',
      operationIds: resetOperationIds,
      scopeAddresses: resetScope,
      reviewPreimageSha256: resetPreimage,
      summaryEpoch,
      reset: true,
      digestCharacter: '7'
    })
  };
  execution.m006_restore = {
    verified: true,
    reason: 'm110_m006_exact_preimage_restored',
    source_transaction_id: 'soundcheck:normal:13c43cc98a3c487c',
    source_event_sha256: '0ce7efb80f19e38f0a5e926467f77845b05ebee9820c78e7ec362b2443128d51',
    stage_count: 3,
    stages: ['foundation_level', 'tonal_dynamics', 'spatial_balance'].map((stage, index) => {
      const expected = expectedM006[stage];
      const operationDigest = fixedOperationDigests.m006[stage];
      return {
        stage,
        operation_digest: operationDigest,
        scope_addresses: expected.scopeAddresses,
        proof: durableReceipt({
          role: 'proof',
          proposalId: `m110-m006-restore-${stage}-proof-${String(index + 3).repeat(32)}`,
          operationDigest,
          bundleKind: 'm006_exact_preimage_restore_bundle',
          operationIds: expected.operationIds,
          scopeAddresses: expected.scopeAddresses,
          sourceOperationIds: expected.sourceOperationIds,
          summaryEpoch,
          digestCharacter: ['a', 'b', 'c'][index]
        }),
        commit: durableReceipt({
          role: 'commit',
          proposalId: `m110-m006-restore-${stage}-commit-${String(index + 6).repeat(32)}`,
          operationDigest,
          bundleKind: 'm006_exact_preimage_restore_bundle',
          operationIds: expected.operationIds,
          scopeAddresses: expected.scopeAddresses,
          sourceOperationIds: expected.sourceOperationIds,
          summaryEpoch,
          digestCharacter: ['d', 'e', 'f'][index]
        })
      };
    }),
    m006_restore_complete: true,
    same_websocket_retained: true,
    backend_process_retained: true,
    token_retained: false,
    live_authority_granted: false,
    readiness_closure_allowed: false
  };
  const summary = {
    schema: 'automixer.m110_attended_runner_summary.v1',
    timestamp_utc: new Date(summaryEpoch * 1000).toISOString().replace('.000Z', 'Z'),
    status: 'show_mix_ready',
    reason: 'm110_attended_show_mix_committed',
    authorization_failures: [],
    required_confirmation: 'EXECUTE_M110_ATTENDED_WING_SOUNDCHECK',
    route_confirmation: 'CAPTURE_WING_MAIN_USB_47_48_ROUTE_PROOF',
    reset_containment_confirmation: 'CONTAIN_ALL_PA_MONITOR_OUTPUTS_AND_MUTE_MAIN_1_FOR_M110_RESET',
    soundcheck_unmute_confirmation: 'UNMUTE_MAIN_1_FOR_M110_SOUNDCHECK',
    interactive_tty_required: true,
    contacts_backend: true,
    contacts_wing: true,
    opens_udp_or_osc: true,
    runner_opens_udp_or_osc: false,
    proof_owned_backend_opens_udp_or_osc: true,
    live_write: true,
    accepted: true,
    readiness_closure_allowed: true,
    launch_probe: execution.identity,
    route_probe: execution.route,
    execution
  };
  return {
    summary,
    routeProof,
    routeBytes,
    routeIdentity,
    source,
    cp05,
    m006Contract
  };
}

function runGit(cwd, args) {
  const completed = spawnSync('/usr/bin/git', args, { cwd, encoding: 'utf8' });
  assert.equal(completed.status, 0, completed.stderr || completed.stdout);
  return completed.stdout.trim();
}

function writeCanonicalJson(filePath, value) {
  fs.mkdirSync(path.dirname(filePath), { recursive: true });
  fs.writeFileSync(filePath, canonicalBytes(value));
}

function createRealRetainedPackageRelease(release, source) {
  const appName = 'AUTO MIXER Tubeslave.app';
  const appPath = path.join(release, 'mac-arm64', appName);
  const mainExecutablePath = path.join(
    appPath,
    'Contents',
    'MacOS',
    'AUTO MIXER Tubeslave'
  );
  const backendServerPath = path.join(
    appPath,
    'Contents',
    'Resources',
    'backend',
    'server'
  );
  const runtimeManifestPath = path.join(
    appPath,
    'Contents',
    'Resources',
    'backend',
    '_internal',
    'automixer-runtime-source-manifest.json'
  );
  const asarPath = path.join(appPath, 'Contents', 'Resources', 'app.asar');
  const infoPlistPath = path.join(appPath, 'Contents', 'Info.plist');
  const attestationPath = path.join(
    appPath,
    'Contents',
    'Resources',
    'release-build-attestation.json'
  );
  const appRelativePath = 'mac-arm64/AUTO MIXER Tubeslave.app';
  const frontendTreeSha256 = '9'.repeat(64);
  const runtimeCodeFiles = EXPECTED_RUNTIME_CRITICAL_SOURCE_PATHS.map(
    (runtimePath, index) => ({
      path: runtimePath,
      exists: true,
      mode: 0o644,
      size: 100 + index,
      sha256: index.toString(16).padStart(64, '0')
    })
  );
  const runtimeSource = {
    schema: 'automixer.backend_runtime_source.v1',
    observation: 'packaged_build_manifest',
    head: source.head,
    tree: source.tree,
    worktree_dirty: false,
    status_sha256: '1'.repeat(64),
    tracked_diff_sha256: '2'.repeat(64),
    source_content_sha256: '3'.repeat(64),
    content_scope: 'automixer.production_source_inputs.v2'
  };
  const runtimeCode = {
    scope: 'automixer.live_safety_critical.v1',
    files: runtimeCodeFiles,
    files_sha256: runtimeSource.source_content_sha256
  };
  const packagedManifest = {
    schema: 'automixer.packaged_runtime_source_manifest.v1',
    source: runtimeSource,
    code: runtimeCode
  };
  packagedManifest.manifest_sha256 = packagedSourceManifestSha256(packagedManifest);

  fs.mkdirSync(path.dirname(mainExecutablePath), { recursive: true });
  fs.mkdirSync(path.dirname(runtimeManifestPath), { recursive: true });
  fs.writeFileSync(mainExecutablePath, 'retained signed executable bytes\n');
  fs.chmodSync(mainExecutablePath, 0o755);
  fs.writeFileSync(backendServerPath, 'retained packaged backend bytes\n');
  fs.chmodSync(backendServerPath, 0o755);
  fs.writeFileSync(asarPath, 'retained app asar bytes\n');
  fs.writeFileSync(infoPlistPath, 'retained Info.plist bytes\n');
  writeCanonicalJson(runtimeManifestPath, packagedManifest);

  const backendIdentity = artifactFileIdentity(
    release,
    backendServerPath,
    'Synthetic retained backend'
  );
  const backendBundleTreeSha256 = hashTree(path.dirname(backendServerPath));
  const buildAttestation = {
    schema: 'automixer.macos_release_build_attestation.v1',
    source: { head: source.head, tree: source.tree },
    buildOutputs: {
      backendSha256: backendIdentity.sha256,
      backendBundleTreeSha256,
      frontendTreeSha256
    }
  };
  writeCanonicalJson(attestationPath, buildAttestation);
  const appTreeSha256 = hashTree(appPath);

  const lockIdentity = (logicalPath, character) => ({
    path: logicalPath,
    pathBase: 'source_checkout',
    size: 1,
    sha256: character.repeat(64)
  });
  const buildMetadata = {
    schema: 'automixer.macos_release_build_metadata.v1',
    pathPolicy: 'artifact_root_relative_and_logical_v1',
    startedAt: '2026-07-15T10:00:00.000Z',
    generatedAt: '2026-07-15T10:00:01.000Z',
    source: { head: source.head, tree: source.tree, cleanCheckout: true },
    locks: {
      frontendPackageLock: lockIdentity('frontend/package-lock.json', 'a'),
      backendRequirementsLock: lockIdentity(
        'backend/requirements-macos-arm64.lock.txt',
        'b'
      ),
      backendBuildRequirementsLock: lockIdentity(
        'backend/requirements-macos-arm64-build.lock.txt',
        'c'
      )
    },
    toolchainInputs: {
      pythonExecutable: {
        path: 'cpython/3.14.3/python',
        pathBase: 'toolchain_runtime',
        size: 1,
        sha256: 'd'.repeat(64)
      }
    },
    tools: Object.fromEntries([
      'node',
      'npm',
      'python',
      'pip',
      'pyinstaller',
      'electronBuilder',
      'electron',
      'osxSign',
      'macOS',
      'xcode'
    ].map(name => [name, name === 'pip' ? '25.3' : '1.0'])),
    buildOutputs: {
      backendServer: {
        path: '.build/backend-dist/server/server',
        size: backendIdentity.size,
        sha256: backendIdentity.sha256
      },
      backendBundleTreeSha256,
      frontendTreeSha256
    }
  };
  const buildMetadataPath = path.join(release, 'build-metadata.json');
  writeCanonicalJson(buildMetadataPath, buildMetadata);

  const attestationIdentity = artifactFileIdentity(
    release,
    attestationPath,
    'Synthetic retained attestation'
  );
  const buildProfile = {
    schema: 'automixer.macos_build_profile.v1',
    pathPolicy: 'artifact_root_relative_v1',
    generatedAt: '2026-07-15T10:00:02.000Z',
    profile: 'release',
    releaseEligible: false,
    reason: 'Synthetic retained-v2 verifier fixture',
    source: {
      head: source.head,
      tree: source.tree,
      worktreeDirty: false,
      statusSha256: '0'.repeat(64),
      sourceContentSha256: '3'.repeat(64)
    },
    app: { path: appRelativePath, treeSha256: appTreeSha256 },
    embeddedAttestation: attestationIdentity
  };
  const buildProfilePath = path.join(release, 'build-profile.json');
  writeCanonicalJson(buildProfilePath, buildProfile);
  const buildProfileIdentity = artifactFileIdentity(
    release,
    buildProfilePath,
    'Synthetic retained build profile'
  );

  const packageFiles = {
    electron: artifactFileIdentity(release, mainExecutablePath, 'Synthetic packaged executable'),
    backend: backendIdentity,
    asar: artifactFileIdentity(release, asarPath, 'Synthetic packaged ASAR'),
    infoPlist: artifactFileIdentity(release, infoPlistPath, 'Synthetic packaged Info.plist'),
    runtimeSourceManifest: artifactFileIdentity(
      release,
      runtimeManifestPath,
      'Synthetic packaged runtime source manifest'
    )
  };
  const runtimeChallenge = '4'.repeat(32);
  const runtimeIdentity = {
    type: 'runtime_identity',
    schema: 'automixer.backend_runtime_identity.v2',
    runtime_kind: 'packaged',
    instance_nonce: '5'.repeat(32),
    started_at_utc: '2026-07-15T10:00:03Z',
    process: {
      pid: 1234,
      instance_nonce: '5'.repeat(32),
      started_at_utc: '2026-07-15T10:00:03Z'
    },
    source: runtimeSource,
    code: runtimeCode,
    backend: {
      path: 'backend/server',
      exists: true,
      size: backendIdentity.size,
      sha256: backendIdentity.sha256
    },
    launch_binding: {
      mode: 'packaged_build_manifest',
      packaged_manifest_schema: 'automixer.packaged_runtime_source_manifest.v1',
      packaged_manifest_name: 'automixer-runtime-source-manifest.json',
      packaged_manifest_size: packageFiles.runtimeSourceManifest.size,
      packaged_manifest_file_sha256: packageFiles.runtimeSourceManifest.sha256,
      packaged_manifest_sha256: packagedSourceManifestSha256(packagedManifest),
      bound_before_process_start: true,
      proof_eligible: false,
      authenticated_launcher_bound: true,
      launch_token_sha256: '8'.repeat(64)
    },
    websocket: { host: '127.0.0.1', port: 8765, loopback_only: true },
    read_only_identity: true,
    live_authority_granted: false,
    request_challenge_nonce: runtimeChallenge
  };
  runtimeIdentity.manifest_sha256 = runtimeIdentityManifestSha256(runtimeIdentity);
  const backendBundle = {
    path: path.relative(release, path.dirname(backendServerPath)),
    sha256: backendBundleTreeSha256
  };
  const smokeBuildProfile = {
    schema: 'automixer.macos_build_profile.v1',
    profile: 'release',
    releaseEligible: false,
    bindingValid: true,
    appTreeMatches: true,
    path: 'build-profile.json',
    size: buildProfileIdentity.size,
    sha256: buildProfileIdentity.sha256,
    source: { head: source.head, tree: source.tree, worktreeDirty: false },
    declaredApp: { path: appRelativePath, treeSha256: appTreeSha256 }
  };
  const smokeResult = {
    schema: 'automixer.packaged_runtime_smoke.v1',
    pathPolicy: 'packaged_build_artifact_root_relative_v1',
    pass: true,
    readOnly: true,
    contactsWing: false,
    liveWrites: false,
    liveAuthorityGranted: false,
    appPath: appRelativePath,
    provenance: {
      head: source.head,
      tree: source.tree,
      worktreeDirty: false,
      sourceBindingStrength: 'clean_checkout'
    },
    before: {
      appTree: { sha256: appTreeSha256 },
      packageFiles,
      backendBundle
    },
    after: {
      appTree: { sha256: appTreeSha256 },
      packageFiles: structuredClone(packageFiles),
      backendBundle: { ...backendBundle }
    },
    runtime: {
      cdpReady: true,
      backendReadyMs: 100,
      backendReadyBudgetMs: 20_000,
      backendReadyWithinBudget: true
    },
    bootstrap: {
      validShape: true,
      tokenSha256: '8'.repeat(64),
      packagedOfflineLock: true,
      backendNetworkSandboxed: true,
      backendNetworkSandboxProfileSha256: BACKEND_SANDBOX_PROFILE_SHA256,
      backendNetworkSandboxProbeResults: passingSandboxProbeResults
    },
    invalidToken: { rejected: true },
    runtimeProbe: {
      authenticated: true,
      runtimeIdentityChallengeNonce: runtimeChallenge,
      runtimeIdentity,
      mode: {
        mode: 'analyze',
        packagedOfflineLock: true,
        mixerTransportLocked: true
      },
      blockedOperations: ['auto_connect', 'connect_wing', 'scan_mixers']
    },
    runtimeIdentityValidation: {
      valid: true,
      checks: { allRequiredChecksPassed: true },
      identity: runtimeIdentity,
      embeddedManifest: {
        size: packageFiles.runtimeSourceManifest.size,
        sha256: packageFiles.runtimeSourceManifest.sha256
      }
    },
    processGroup: [{ pid: 1234, role: 'backend' }],
    blockedOperationsPass: true,
    reloadRecovery: {
      staleSessionTokenRemoved: true,
      secureBootstrapFailureShown: false,
      offlineLockShown: true,
      hardwareControlsHidden: true
    },
    networkEnforcement: {
      schema: 'automixer.packaged_runtime_network_enforcement.v1',
      mechanism: 'macos_sandbox_exec',
      scope: 'packaged_backend_process_tree',
      profileSha256: BACKEND_SANDBOX_PROFILE_SHA256,
      probeResults: passingSandboxProbeResults,
      nonLoopbackOutboundDenied: true,
      loopbackOutboundAllowed: true,
      preflightPassed: true,
      backendBoundaryReported: true,
      profileMatchesBackend: true,
      probeMatrixMatchesBackend: true
    },
    launchCopy: {
      signatureVerifiedBefore: true,
      signatureVerifiedAfter: true,
      treeSha256Before: appTreeSha256,
      treeSha256After: appTreeSha256,
      matchesSourceApp: true,
      unchanged: true,
      stillMatchesSourceApp: true
    },
    appTreeUnchanged: true,
    canonicalLogUnchanged: true,
    canonicalDecisionTraceUnchanged: true,
    canonicalDecisionMemoryUnchanged: true,
    consoleWalUnchanged: true,
    backendBundleUnchanged: true,
    packageFilesUnchanged: true,
    buildProfile: smokeBuildProfile,
    tempWriteArtifactsAbsent: true,
    network: {
      externalConnections: [],
      wingPortSockets: [],
      monitorErrors: [],
      sampleCount: 3
    }
  };
  const smokeDirectory = path.join(release, 'runtime-smoke', 'run');
  const smokeResultPath = path.join(smokeDirectory, 'result.json');
  const smokeMarkdownPath = path.join(smokeDirectory, 'result.md');
  const smokeManifestPath = path.join(smokeDirectory, 'artifact-manifest.json');
  writeCanonicalJson(smokeResultPath, smokeResult);
  fs.writeFileSync(smokeMarkdownPath, '# Synthetic retained smoke evidence\n');
  writeCanonicalJson(smokeManifestPath, {
    schema: 'automixer.packaged_runtime_smoke_manifest.v1',
    pathPolicy: 'manifest_directory_relative_v1',
    readOnly: true,
    contactsWing: false,
    liveAuthorityGranted: false,
    buildProfile: {
      profile: smokeBuildProfile.profile,
      releaseEligible: smokeBuildProfile.releaseEligible,
      path: smokeBuildProfile.path,
      pathBase: 'packaged_build_artifact_root',
      size: smokeBuildProfile.size,
      sha256: smokeBuildProfile.sha256,
      bindingValid: smokeBuildProfile.bindingValid,
      appTreeMatches: smokeBuildProfile.appTreeMatches
    },
    files: {
      'result.json': artifactFileIdentity(
        smokeDirectory,
        smokeResultPath,
        'Synthetic smoke result'
      ),
      'result.md': artifactFileIdentity(
        smokeDirectory,
        smokeMarkdownPath,
        'Synthetic smoke Markdown'
      )
    }
  });

  const zipPath = path.join(release, 'AUTO MIXER Tubeslave.zip');
  const dmgPath = path.join(release, 'AUTO MIXER Tubeslave.dmg');
  fs.writeFileSync(zipPath, 'retained ZIP bytes\n');
  fs.writeFileSync(dmgPath, 'retained DMG bytes\n');
  const dmgIdentity = artifactFileIdentity(release, dmgPath, 'Synthetic retained DMG');
  const appCdhash = '1'.repeat(40);
  const dmgCdhash = '2'.repeat(40);
  const appSubmissionId = '12345678-1234-1234-1234-123456789abc';
  const dmgSubmissionId = '22345678-1234-1234-1234-123456789abc';
  const appArchiveFilename = 'app-submission.zip';
  const appArchiveSha256 = '6'.repeat(64);
  const startedAt = '2026-07-15T10:00:04.000Z';
  const completedAt = '2026-07-15T10:00:05.000Z';
  const appTicket = {
    path: `${appArchiveFilename}/${appName}`,
    pathSuffix: appName,
    architecture: 'arm64',
    digestAlgorithm: 'SHA-256',
    cdhash: appCdhash
  };
  const appLog = {
    logFormatVersion: 1,
    jobId: appSubmissionId,
    status: 'Accepted',
    statusCode: 0,
    archiveFilename: appArchiveFilename,
    uploadDate: startedAt,
    sha256: appArchiveSha256,
    issues: [],
    ticketContents: [{
      path: appTicket.path,
      arch: 'arm64',
      digestAlgorithm: 'SHA-256',
      cdhash: appCdhash
    }]
  };
  const appLogPath = path.join(release, 'app-notary-log.json');
  writeCanonicalJson(appLogPath, appLog);
  const appEvidence = {
    schema: 'automixer.app_notarization.v1',
    pathPolicy: 'artifact_root_relative_v1',
    status: 'Accepted',
    generatedAt: completedAt,
    app: { path: appRelativePath, treeSha256: appTreeSha256 },
    mainExecutable: {
      path: 'Contents/MacOS/AUTO MIXER Tubeslave',
      architecture: 'arm64',
      cdhash: appCdhash
    },
    submission: {
      id: appSubmissionId,
      archiveFilename: appArchiveFilename,
      archive: { size: 123, sha256: appArchiveSha256 },
      startedAt,
      completedAt
    },
    rawLog: artifactFileIdentity(release, appLogPath, 'Synthetic application notary log'),
    appleLog: {
      logFormatVersion: 1,
      jobId: appSubmissionId,
      status: 'Accepted',
      statusCode: 0,
      archiveFilename: appArchiveFilename,
      uploadDate: startedAt,
      sha256: appArchiveSha256,
      issuesCount: 0,
      ticketCount: 1
    },
    ticket: appTicket
  };
  const appEvidencePath = path.join(release, 'app-notarization.json');
  writeCanonicalJson(appEvidencePath, appEvidence);

  const dmgArchiveFilename = path.basename(dmgPath);
  const dmgTicket = {
    path: dmgArchiveFilename,
    digestAlgorithm: 'SHA-256',
    cdhash: dmgCdhash
  };
  const dmgLog = {
    logFormatVersion: 1,
    jobId: dmgSubmissionId,
    status: 'Accepted',
    statusCode: 0,
    archiveFilename: dmgArchiveFilename,
    uploadDate: startedAt,
    sha256: dmgIdentity.sha256,
    issues: [],
    ticketContents: [{ ...dmgTicket, arch: null }]
  };
  const dmgLogPath = path.join(release, 'dmg-notary-log.json');
  writeCanonicalJson(dmgLogPath, dmgLog);
  const dmgEvidence = {
    schema: 'automixer.dmg_notarization.v2',
    pathPolicy: 'artifact_root_relative_v1',
    status: 'Accepted',
    generatedAt: completedAt,
    dmg: dmgIdentity,
    submission: {
      id: dmgSubmissionId,
      archiveFilename: dmgArchiveFilename,
      archive: { size: dmgIdentity.size, sha256: dmgIdentity.sha256 },
      startedAt,
      completedAt
    },
    signature: {
      cdhashBeforeStaple: dmgCdhash,
      cdhashAfterStaple: dmgCdhash
    },
    rawLog: artifactFileIdentity(release, dmgLogPath, 'Synthetic DMG notary log'),
    appleLog: {
      logFormatVersion: 1,
      jobId: dmgSubmissionId,
      status: 'Accepted',
      statusCode: 0,
      archiveFilename: dmgArchiveFilename,
      uploadDate: startedAt,
      sha256: dmgIdentity.sha256,
      issuesCount: 0,
      ticketCount: 1
    },
    ticket: dmgTicket
  };
  const dmgEvidencePath = path.join(release, 'dmg-notarization.json');
  writeCanonicalJson(dmgEvidencePath, dmgEvidence);

  const signatures = {
    application: { cdhash: appCdhash },
    'code:Contents/MacOS/AUTO MIXER Tubeslave': { cdhash: appCdhash }
  };
  const manifestInputs = {
    artifactDirectory: release,
    isolatedOutputsPresent: true,
    expectedHead: source.head,
    expectedTree: source.tree,
    buildMetadataPath,
    buildMetadata,
    appPath,
    appBefore: {
      infoPlist: { CFBundleIdentifier: 'com.automixer.wing' },
      treeSha256: appTreeSha256,
      packagedInputs: { frontendTreeSha256 },
      signatures,
      backendEntitlements: { 'com.apple.security.cs.disable-library-validation': true }
    },
    zipVerification: { treeSha256: appTreeSha256 },
    dmgVerification: { treeSha256: appTreeSha256 },
    dmgPath,
    zipPath,
    dmgSignature: { cdhash: dmgCdhash },
    appNotarizationEvidence: appEvidence,
    appNotarizationEvidencePath: appEvidencePath,
    appNotaryLogPath: appLogPath,
    dmgNotarizationEvidence: dmgEvidence,
    dmgNotarizationEvidencePath: dmgEvidencePath,
    dmgNotaryLogPath: dmgLogPath,
    smokeResultPath
  };
  const initialVerificationPath = path.join(
    release,
    'release-verification-initial.json'
  );
  const initialManifest = createPortableReleaseManifest({
    ...manifestInputs,
    retainedArtifacts: false,
    initialVerificationPath: null
  });
  fs.writeFileSync(
    initialVerificationPath,
    `${JSON.stringify(initialManifest, null, 2)}\n`
  );
  const finalManifest = createPortableReleaseManifest({
    ...manifestInputs,
    retainedArtifacts: true,
    isolatedOutputsPresent: false,
    initialVerificationPath
  });
  fs.writeFileSync(
    path.join(release, 'release-manifest.json'),
    `${JSON.stringify(finalManifest, null, 2)}\n`
  );
  return { zipPath, appTreeSha256 };
}

function createBundleIntegrationFixture() {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'automixer-public-bundle-'));
  const repository = path.join(root, 'repository');
  const release = path.join(root, 'release');
  const inputs = path.join(root, 'inputs');
  fs.mkdirSync(repository);
  fs.mkdirSync(release);
  fs.mkdirSync(inputs);
  runGit(repository, ['init', '-q']);
  const tracked = {
    'tools/wing_attended_soundcheck_runner.py': 'print("retained runner")\n',
    'backend/server.py': 'print("retained backend")\n',
    'backend/contracts/m110_m006_restore_20260711.json': fs.readFileSync(m006ContractPath),
    'ops/m110/CP05_HARDWARE_GATE.json': '{}\n',
    'Docs/reports/M110_CP05_SOFTWARE_VALIDATION.json': '{}\n'
  };
  for (const authorityPath of publicReleaseEvidenceTestHooks.authoritySourcePaths) {
    tracked[authorityPath] = fs.readFileSync(path.join(workspaceRoot, authorityPath));
  }
  for (const [relativePath, bytes] of Object.entries(tracked)) {
    const target = path.join(repository, relativePath);
    fs.mkdirSync(path.dirname(target), { recursive: true });
    fs.writeFileSync(target, bytes);
    if (publicReleaseEvidenceTestHooks.authoritySourcePaths.includes(relativePath)) {
      fs.chmodSync(
        target,
        fs.statSync(path.join(workspaceRoot, relativePath)).mode & 0o777
      );
    }
  }
  runGit(repository, ['add', '.']);
  runGit(repository, [
    '-c', 'user.name=AUTO MIXER Test',
    '-c', 'user.email=automixer-test@example.invalid',
    '-c', 'commit.gpgsign=false',
    'commit', '-q', '-m', 'fixture source'
  ]);
  const initialBinding = computeCp05SourceBinding(repository);
  const cp05Evidence = {
    checks: {
      focused_tests: true,
      frontend_build_and_tests: true,
      full_backend_suite: true,
      independent_review: true,
      no_wing_release_check: true,
      released_development_lease: true,
      static_scan: true
    },
    schema: 'automixer.m110_cp05_software_validation.v1',
    source_binding_sha256: initialBinding.sha256,
    status: 'passed'
  };
  const cp05EvidenceBytes = canonicalBytes(cp05Evidence);
  const cp05Gate = {
    checkpoint: 'M110-CP05',
    hardware_enabled: true,
    schema: 'automixer.m110_cp05_hardware_gate.v1',
    software_evidence_sha256: `sha256:${sha256(cp05EvidenceBytes)}`
  };
  writeCanonicalJson(
    path.join(repository, 'Docs/reports/M110_CP05_SOFTWARE_VALIDATION.json'),
    cp05Evidence
  );
  writeCanonicalJson(path.join(repository, 'ops/m110/CP05_HARDWARE_GATE.json'), cp05Gate);
  runGit(repository, ['add', '.']);
  runGit(repository, [
    '-c', 'user.name=AUTO MIXER Test',
    '-c', 'user.email=automixer-test@example.invalid',
    '-c', 'commit.gpgsign=false',
    'commit', '-q', '-m', 'fixture CP05 markers'
  ]);
  const finalBinding = computeCp05SourceBinding(repository);
  assert.deepEqual(finalBinding, initialBinding);
  const cp05 = validateCp05Markers({
    gate: cp05Gate,
    gateBytes: canonicalBytes(cp05Gate),
    evidence: cp05Evidence,
    evidenceBytes: cp05EvidenceBytes,
    sourceBinding: finalBinding
  });
  const source = {
    head: runGit(repository, ['rev-parse', 'HEAD']),
    tree: runGit(repository, ['rev-parse', 'HEAD^{tree}']),
    cleanCheckout: true
  };
  const m110 = acceptedM110Fixture({ source, cp05 });
  const summaryPath = path.join(inputs, 'execution-summary.json');
  const routeProofPath = path.join(inputs, `${m110.routeProof.proof_id}.json`);
  fs.writeFileSync(summaryPath, canonicalBytes(m110.summary));
  fs.writeFileSync(routeProofPath, m110.routeBytes);
  const retainedManifestBytes = Buffer.from('{"fixture":true}\n');
  fs.writeFileSync(path.join(release, 'release-manifest.json'), retainedManifestBytes);
  const packageRelease = {
    detachedEvidenceReplayComplete: true,
    nativeTrustReexecuted: false,
    gatekeeperReexecuted: false,
    archiveContentsReexecuted: false,
    manifestIdentity: {
      path: 'release-manifest.json',
      size: retainedManifestBytes.length,
      sha256: sha256(retainedManifestBytes)
    },
    generatedAt: '2026-07-15T10:00:00.000Z',
    source
  };
  const verifyPackage = () => structuredClone(packageRelease);
  return {
    root,
    repository,
    release,
    summaryPath,
    routeProofPath,
    packageRelease,
    verifyPackage,
    cleanup() { fs.rmSync(root, { recursive: true, force: true }); }
  };
}

function rehashBundleManifest(manifest) {
  const projection = structuredClone(manifest);
  delete projection.bundleSha256;
  manifest.bundleSha256 = `sha256:${sha256(Buffer.from(canonicalJson(projection), 'utf8'))}`;
  return manifest;
}

test('CP05 markers require canonical bytes, seven checks, and exact source binding', () => {
  const fixture = cp05Fixture();
  assert.equal(fixture.verified.verified, true);
  assert.throws(() => validateCp05Markers({
    ...fixture,
    evidenceBytes: Buffer.from(`${JSON.stringify(fixture.evidence, null, 2)}\n`)
  }), /not canonical/);
  assert.throws(() => validateCp05Markers({
    ...fixture,
    sourceBinding: { ...fixture.sourceBinding, sha256: `sha256:${'9'.repeat(64)}` }
  }), /source binding/);
});

test('M110 summary parser matches Python ensure_ascii false canonical JSON', () => {
  const value = {
    control: 'line\n\t"\\',
    emoji: '😀',
    nested: { channel: 'Вокал', device: 'Пульт WING' },
    numbers: [
      -0.0,
      0.0,
      Number.MIN_VALUE,
      2.2250738585072014e-308,
      1e-7,
      1e-6,
      1e-5,
      0.0001,
      0.0009999999999999999,
      1.0,
      1.2345678901234567,
      999999999999999.0,
      1e15,
      1e16,
      1e20,
      1e21,
      1e23,
      1000000000000000128.0,
      Number.MAX_VALUE
    ],
    русский: 'саундчек'
  };
  const completed = spawnSync('python3', ['-c', [
    'import json,sys',
    'value=json.loads(sys.stdin.read())',
    'value["numbers"]=[-0.0,0.0,5e-324,2.2250738585072014e-308,1e-7,1e-6,1e-5,0.0001,0.0009999999999999999,1.0,1.2345678901234567,999999999999999.0,1e15,1e16,1e20,1e21,1e23,1000000000000000128.0,1.7976931348623157e308]',
    'text=json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(",",":"),allow_nan=False)',
    'sys.stdout.buffer.write((text+"\\n").encode("utf-8"))'
  ].join(';')], {
    input: JSON.stringify(value),
    encoding: null
  });
  assert.equal(completed.status, 0, Buffer.from(completed.stderr || '').toString('utf8'));
  assert.equal(
    publicReleaseEvidenceTestHooks.validatePythonCanonicalJson(
      value,
      Buffer.from(completed.stdout)
    ),
    true
  );
  assert.throws(
    () => publicReleaseEvidenceTestHooks.validatePythonCanonicalJson(
      { x: 3 },
      Buffer.from('{"x":3.0000000000000001}\n')
    ),
    /exact canonical Python JSON representation/
  );
  assert.throws(
    () => publicReleaseEvidenceTestHooks.validatePythonCanonicalJson(
      { x: 0.1 },
      Buffer.from('{"x":0.10000000000000001}\n')
    ),
    /exact canonical Python JSON representation/
  );

  const routeValue = {
    allowance: 3,
    ceiling: -1,
    exponent: 1e-7,
    unicode: 'Пульт WING'
  };
  const routeCompleted = spawnSync('python3', ['-c', [
    'import json,sys',
    'value=json.loads(sys.stdin.read())',
    'value["allowance"]=3.0',
    'value["ceiling"]=-1.0',
    'value["exponent"]=1e-7',
    'text=json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(",",":"),allow_nan=False)',
    'sys.stdout.buffer.write(text.encode("utf-8"))'
  ].join(';')], {
    input: JSON.stringify(routeValue),
    encoding: null
  });
  assert.equal(
    routeCompleted.status,
    0,
    Buffer.from(routeCompleted.stderr || '').toString('utf8')
  );
  assert.equal(
    publicReleaseEvidenceTestHooks.validatePythonCanonicalJson(
      routeValue,
      Buffer.from(routeCompleted.stdout),
      { finalNewline: false }
    ),
    true
  );
});

test('detached canonical reset and M006 digests match the Python golden vectors', () => {
  const fixture = acceptedM110Fixture();
  const digests = computeM110FixedOperationDigests({
    routeProof: fixture.routeProof,
    routeIdentity: { size: 1, sha256: 'd'.repeat(64) },
    source: fixture.source,
    m006Contract: fixture.m006Contract
  });
  assert.equal(
    digests.reset,
    'sha256:cb79fdb7ab0c827d3a3f7274eba8b3005e314fe54e1175117aa0e06865565c96'
  );
  assert.deepEqual(digests.m006, {
    foundation_level: 'sha256:657ce905f6e75d848e19ce43bfb0f98024260b8967fbc47a92326b91fbaecfd9',
    tonal_dynamics: 'sha256:cf7e1003b664584fd665765b79e84f8fe3b29e7193d8c8e9eb547f3a1fdec439',
    spatial_balance: 'sha256:4e112ddf1e46a327da219a2e21b3d9ea0cf59e4bc533f2c9bbd166d77e320b64'
  });
});

test('M006 receipt scopes preserve the Python server canonical bundle order', () => {
  const { contract, contractIdentity } = immutableM006ContractFixture();
  const expected = validateM006Contract({ contract, contractIdentity });
  const scopeHashes = Object.fromEntries(
    Object.entries(expected).map(([stage, value]) => [
      stage,
      sha256(Buffer.from(JSON.stringify(value.scopeAddresses), 'utf8'))
    ])
  );

  // Golden values emitted by canonical_bundle_scope() for the retained fixture.
  assert.deepEqual(scopeHashes, {
    foundation_level: 'cb3100f4d4014fefceeb7798083d736522e584642ae4fb3cf4d625c6f743d6af',
    tonal_dynamics: 'b4b254b1b697fbf685b948b9b829044af88afe6de1944e2c39ee1bc4f3f4e306',
    spatial_balance: '4420997eb9f16995897d3a0fb97a6c55af06d6f30c18ea630d6200a8f6e4718f'
  });
  assert.deepEqual(expected.foundation_level.scopeAddresses.slice(0, 3), [
    '/ch/1/in/set/trim',
    '/ch/1/flt/lcf',
    '/ch/1/flt/lc'
  ]);
  assert.deepEqual(expected.spatial_balance.scopeAddresses.slice(8, 11), [
    '/ch/9/pan',
    '/ch/10/pan',
    '/ch/11/pan'
  ]);
});

test('accepted M110 validator binds source, route, M006, listening, and teardown', () => {
  const fixture = acceptedM110Fixture();
  const verified = validateM110AttendedEvidence(fixture);
  assert.equal(verified.status, 'show_mix_ready');
  assert.equal(verified.console.serial, '0100Q1V0604AAE');

  const mutate = callback => {
    const value = structuredClone(fixture);
    value.routeBytes = Buffer.from(fixture.routeBytes);
    callback(value);
    assert.throws(() => validateM110AttendedEvidence(value));
  };
  mutate(value => { value.summary.accepted = false; });
  mutate(value => { value.routeProof.evidence.runtime.head = '9'.repeat(40); });
  mutate(value => { value.routeProof.evidence.capture.conservative_true_peak_dbtp = 0; });
  mutate(value => { value.summary.execution.m006_restore.m006_restore_complete = false; });
  mutate(value => { value.summary.execution.hw05_operator_listening_ack.operator_response_matched = false; });
  mutate(value => { value.summary.execution.teardown.backend_shutdown.forced = true; });
  mutate(value => { delete value.summary.execution.lanes.commit_count; });
  mutate(value => { value.summary.execution.lanes.commit_count = 0; });
  mutate(value => {
    value.summary.execution.lanes.commits.soundcheck[0]
      .commit.terminal.proof_receipt_id = value.summary.execution.lanes
        .commits.processor[0].commit.terminal.proof_receipt_id;
  });
  mutate(value => { value.summary.execution.soundcheck_readiness.plan_revision = 99; });
  mutate(value => { value.summary.leaked_token = `m110cap_v1_${'f'.repeat(64)}`; });

  const sameFinalSecond = acceptedM110Fixture();
  const summarySecond = Math.floor(
    Date.parse(sameFinalSecond.summary.timestamp_utc) / 1000
  );
  sameFinalSecond.summary.timestamp_utc = new Date(summarySecond * 1000)
    .toISOString()
    .replace('.000Z', 'Z');
  sameFinalSecond.summary.execution.reset.proof
    .operator_checkpoint.captured_at_epoch = summarySecond + 0.789;
  sameFinalSecond.summary.execution.hw05_operator_listening_ack
    .captured_at_epoch = summarySecond + 0.789;
  assert.equal(validateM110AttendedEvidence(sameFinalSecond).verified, true);

  const forgedAllowance = acceptedM110Fixture();
  forgedAllowance.routeProof.evidence.capture.true_peak_allowance_db = 0;
  forgedAllowance.routeProof.evidence.capture.conservative_true_peak_dbtp = -6;
  rebindRouteBytes(forgedAllowance);
  assert.throws(
    () => validateM110AttendedEvidence(forgedAllowance),
    /Fresh physical WING capture/
  );
});

test('route replay rejects digest drift, noncanonical numbers, oversized bytes, and fake hardware', () => {
  const withRouteBytes = value => {
    value.routeBytes = Buffer.from(pythonRouteCanonicalJson(value.routeProof), 'utf8');
    value.routeIdentity = {
      size: value.routeBytes.length,
      sha256: sha256(value.routeBytes)
    };
    return value;
  };

  const digestDrift = structuredClone(acceptedM110Fixture());
  digestDrift.routeProof.evidence_sha256 = `sha256:${'0'.repeat(64)}`;
  withRouteBytes(digestDrift);
  assert.throws(
    () => validateM110AttendedEvidence(digestDrift),
    /evidence digest/
  );

  const noncanonical = acceptedM110Fixture();
  noncanonical.routeBytes = Buffer.from(
    noncanonical.routeBytes.toString('utf8').replace(
      '1700000000.789',
      '1.700000000789e+09'
    ),
    'utf8'
  );
  noncanonical.routeIdentity = {
    size: noncanonical.routeBytes.length,
    sha256: sha256(noncanonical.routeBytes)
  };
  assert.throws(
    () => validateM110AttendedEvidence(noncanonical),
    /number exponent is not canonical Python JSON/
  );

  const overprecise = acceptedM110Fixture();
  overprecise.routeBytes = Buffer.from(
    overprecise.routeBytes.toString('utf8').replace(
      '1700000000.789',
      '1700000000.7890000000001'
    ),
    'utf8'
  );
  overprecise.routeIdentity = {
    size: overprecise.routeBytes.length,
    sha256: sha256(overprecise.routeBytes)
  };
  assert.throws(
    () => validateM110AttendedEvidence(overprecise),
    /exact canonical Python JSON representation/
  );

  const negativeZero = acceptedM110Fixture();
  negativeZero.routeBytes = Buffer.from(
    negativeZero.routeBytes.toString('utf8').replace(
      '"issued_at_epoch":1700000000.789',
      '"issued_at_epoch":-0'
    ),
    'utf8'
  );
  negativeZero.routeIdentity = {
    size: negativeZero.routeBytes.length,
    sha256: sha256(negativeZero.routeBytes)
  };
  assert.throws(
    () => validateM110AttendedEvidence(negativeZero),
    /negative zero/
  );

  const oversized = acceptedM110Fixture();
  oversized.routeBytes = Buffer.alloc(64 * 1024 + 1, 0x20);
  oversized.routeIdentity = {
    size: oversized.routeBytes.length,
    sha256: sha256(oversized.routeBytes)
  };
  assert.throws(
    () => validateM110AttendedEvidence(oversized),
    /artifact byte limit/
  );

  const fakeHardware = structuredClone(acceptedM110Fixture());
  fakeHardware.routeProof.evidence.capture.hardware_identity.device_index = '1';
  fakeHardware.routeProof.evidence.capture.device_index = '1';
  rebindRouteBytes(fakeHardware);
  assert.throws(
    () => validateM110AttendedEvidence(fakeHardware),
    /Fresh physical WING capture/
  );
});

test('production build, public verify, and retained-package verifier fail closed on incomplete roots', () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'automixer-public-entrypoints-'));
  const release = path.join(root, 'release');
  const repository = path.join(root, 'repository');
  fs.mkdirSync(release);
  fs.mkdirSync(repository);
  try {
    assert.throws(
      () => verifyRetainedReleaseManifestEvidence(release),
      /release-manifest\.json/
    );
    withProductionRuntime(() => {
      assert.throws(
        () => verifyPublicReleaseEvidenceBundle(release),
        /public-release-evidence\.json|completely clean checkout/
      );
      assert.throws(() => buildPublicReleaseEvidenceBundle({
        repoRoot: repository,
        releaseRoot: release,
        m110SummaryPath: path.join(root, 'execution-summary.json'),
        routeProofPath: path.join(root, 'route.json')
      }), /Production bundle authority must execute from the selected source repository/);
    });
    assert.equal(fs.existsSync(path.join(release, 'public-release-evidence.json')), false);
  } finally {
    fs.rmSync(root, { recursive: true, force: true });
  }
});

test('builder rejects raw M110 inputs anywhere under the distributable release root', () => {
  const fixture = createBundleIntegrationFixture();
  try {
    for (const [field, filename] of [
      ['summaryPath', 'leaked-execution-summary.json'],
      ['routeProofPath', 'leaked-route-proof.json']
    ]) {
      const leakedPath = path.join(fixture.release, filename);
      fs.copyFileSync(fixture[field], leakedPath);
      const options = {
        repoRoot: fixture.repository,
        releaseRoot: fixture.release,
        m110SummaryPath: fixture.summaryPath,
        routeProofPath: fixture.routeProofPath,
        generatedAt: '2026-07-15T11:00:00.000Z'
      };
      options[field === 'summaryPath' ? 'm110SummaryPath' : 'routeProofPath'] = leakedPath;
      assert.throws(
        () => publicReleaseEvidenceTestHooks.build(
          options,
          fixture.verifyPackage
        ),
        /stable single-link regular file outside the distributable release root/
      );
      fs.unlinkSync(leakedPath);
    }
    for (const [field, filename] of [
      ['summaryPath', 'hardlinked-execution-summary.json'],
      ['routeProofPath', 'hardlinked-route-proof.json']
    ]) {
      const leakedPath = path.join(fixture.release, filename);
      fs.linkSync(fixture[field], leakedPath);
      assert.throws(
        () => publicReleaseEvidenceTestHooks.build({
          repoRoot: fixture.repository,
          releaseRoot: fixture.release,
          m110SummaryPath: fixture.summaryPath,
          routeProofPath: fixture.routeProofPath,
          generatedAt: '2026-07-15T11:00:00.000Z'
        }, fixture.verifyPackage),
        /stable single-link regular file outside the distributable release root/
      );
      fs.unlinkSync(leakedPath);
    }
    assert.equal(
      fs.existsSync(path.join(fixture.release, 'public-release-evidence')),
      false
    );
  } finally {
    fixture.cleanup();
  }
});

test('raw-input closures bind descriptor-pinned bytes to the copied JSON snapshot', () => {
  const source = fs.readFileSync(bundlePath, 'utf8');
  assert.match(source, /fs\.readFileSync\(descriptor\)/);
  assert.match(
    source,
    /identitiesEqual\(stable\.identity, inputClosure\.content\)/
  );
});

test('builder rejects a raw-input hardlink created after the private snapshot', () => {
  const fixture = createBundleIntegrationFixture();
  const leakedPath = path.join(fixture.release, 'late-hardlinked-summary.json');
  let replayCount = 0;
  const leakingVerifier = () => {
    replayCount += 1;
    if (replayCount === 1) fs.linkSync(fixture.summaryPath, leakedPath);
    return structuredClone(fixture.packageRelease);
  };
  try {
    assert.throws(
      () => publicReleaseEvidenceTestHooks.build({
        repoRoot: fixture.repository,
        releaseRoot: fixture.release,
        m110SummaryPath: fixture.summaryPath,
        routeProofPath: fixture.routeProofPath,
        generatedAt: '2026-07-15T11:00:00.000Z'
      }, leakingVerifier),
      /stable single-link regular file outside the distributable release root/
    );
    assert.equal(fs.existsSync(path.join(fixture.release, 'public-release-evidence.json')), false);
    assert.equal(fs.existsSync(path.join(fixture.release, '.public-release-evidence.candidate')), false);
  } finally {
    fixture.cleanup();
  }
});

test('builder rejects a public hardlink alias to a copied private payload', () => {
  const fixture = createBundleIntegrationFixture();
  const privateSummaryPath = path.join(
    fixture.release,
    'public-release-evidence',
    'm110-execution-summary.json'
  );
  const leakedPath = path.join(fixture.release, 'leaked-private-summary.json');
  let replayCount = 0;
  const leakingVerifier = () => {
    replayCount += 1;
    if (replayCount === 1) fs.linkSync(privateSummaryPath, leakedPath);
    return structuredClone(fixture.packageRelease);
  };
  try {
    assert.throws(
      () => publicReleaseEvidenceTestHooks.build({
        repoRoot: fixture.repository,
        releaseRoot: fixture.release,
        m110SummaryPath: fixture.summaryPath,
        routeProofPath: fixture.routeProofPath,
        generatedAt: '2026-07-15T11:00:00.000Z'
      }, leakingVerifier),
      /private single-link 0600 file|exact canonical private 0600 file set/
    );
    assert.equal(fs.existsSync(leakedPath), true);
    assert.equal(fs.existsSync(path.join(fixture.release, 'public-release-evidence.json')), false);
    assert.equal(fs.existsSync(path.join(fixture.release, '.public-release-evidence.candidate')), false);
  } finally {
    fixture.cleanup();
  }
});

test('builder rechecks raw-input single-link closure after its final retained replay', () => {
  const fixture = createBundleIntegrationFixture();
  const leakedPath = path.join(fixture.release, 'late-return-cut-summary.json');
  let replayCount = 0;
  const leakingVerifier = () => {
    replayCount += 1;
    if (replayCount === 7) fs.linkSync(fixture.summaryPath, leakedPath);
    return structuredClone(fixture.packageRelease);
  };
  try {
    assert.throws(
      () => publicReleaseEvidenceTestHooks.build({
        repoRoot: fixture.repository,
        releaseRoot: fixture.release,
        m110SummaryPath: fixture.summaryPath,
        routeProofPath: fixture.routeProofPath,
        generatedAt: '2026-07-15T11:00:00.000Z'
      }, leakingVerifier),
      /changed at bundle return cut|stable single-link regular file/
    );
    assert.equal(replayCount, 7);
    assert.equal(fs.existsSync(leakedPath), true);
    assert.equal(fs.existsSync(path.join(fixture.release, 'public-release-evidence.json')), false);
  } finally {
    fixture.cleanup();
  }
});

test('full synthetic build verifies before publication and remains valid after relocation', () => {
  const fixture = createBundleIntegrationFixture();
  try {
    const built = publicReleaseEvidenceTestHooks.build({
      repoRoot: fixture.repository,
      releaseRoot: fixture.release,
      m110SummaryPath: fixture.summaryPath,
      routeProofPath: fixture.routeProofPath,
      generatedAt: '2026-07-15T11:00:00.000Z'
    }, fixture.verifyPackage);
    assert.equal(built.verified, true);
    assert.equal(built.soundcheck.claimedM110Status, 'show_mix_ready');
    assert.equal(built.soundcheck.structuralEvidenceReplayComplete, true);
    assert.equal(built.soundcheck.externalAuthenticityVerified, false);
    assert.equal(built.soundcheck.readinessClosureAllowed, false);
    assert.equal(
      fs.statSync(path.join(fixture.release, 'public-release-evidence')).mode & 0o777,
      0o700
    );
    assert.equal(
      fs.statSync(path.join(
        fixture.release,
        'public-release-evidence',
        'm110-execution-summary.json'
      )).mode & 0o777,
      0o600
    );
    const publicManifestText = fs.readFileSync(
      path.join(fixture.release, 'public-release-evidence.json'),
      'utf8'
    );
    assert.doesNotMatch(publicManifestText, /0100Q1V0604AAE/);
    assert.doesNotMatch(publicManifestText, /usb:1397:050b/);
    assert.equal(publicReleaseEvidenceTestHooks.verify(
      fixture.release,
      fixture.verifyPackage
    ).verified, true);
    const relocated = path.join(fixture.root, 'relocated-release');
    fs.renameSync(fixture.release, relocated);
    assert.equal(publicReleaseEvidenceTestHooks.verify(
      relocated,
      fixture.verifyPackage
    ).verified, true);
  } finally {
    fixture.cleanup();
  }
});

test('production retained-v2 replay drives public build and detects archive drift', () => {
  const fixture = createBundleIntegrationFixture();
  try {
    const retained = createRealRetainedPackageRelease(
      fixture.release,
      fixture.packageRelease.source
    );
    const retainedManifestPath = path.join(fixture.release, 'release-manifest.json');
    const retainedManifestBytes = fs.readFileSync(retainedManifestPath);
    fs.writeFileSync(
      retainedManifestPath,
      retainedManifestBytes.toString('utf8').replace(
        /^\{\n/,
        '{\n  "releaseEligible": false,\n'
      )
    );
    assert.throws(
      () => verifyRetainedReleaseManifestEvidence(fixture.release),
      /exact canonical serialized bytes/
    );
    fs.writeFileSync(retainedManifestPath, retainedManifestBytes);
    const packageReplay = verifyRetainedReleaseManifestEvidence(fixture.release);
    assert.equal(packageReplay.detachedEvidenceReplayComplete, true);
    assert.equal(packageReplay.packageReleaseEligibleEvidence, false);
    assert.equal(packageReplay.packageReleaseStructuralConsistencyVerified, true);
    assert.equal(packageReplay.externalAuthenticityVerified, false);
    assert.equal(packageReplay.readinessClosureAllowed, false);
    const built = publicReleaseEvidenceTestHooks.build({
      repoRoot: fixture.repository,
      releaseRoot: fixture.release,
      m110SummaryPath: fixture.summaryPath,
      routeProofPath: fixture.routeProofPath,
      generatedAt: '2026-07-15T11:00:00.000Z'
    }, verifyRetainedReleaseManifestEvidence);
    assert.equal(built.verified, true);
    assert.equal(built.status, 'structural_consistency_verified');
    assert.equal(built.evidenceComplete, false);
    assert.equal(built.externalAuthenticityVerified, false);
    assert.equal(built.externalAuthenticatedAttestationRequired, true);
    assert.equal(built.readinessClosureAllowed, false);
    assert.equal(
      withProductionRuntime(() => verifyPublicReleaseEvidenceBundle(fixture.release)).verified,
      true
    );

    fs.appendFileSync(retained.zipPath, 'forged ZIP bytes\n');
    assert.throws(
      () => withProductionRuntime(() => verifyPublicReleaseEvidenceBundle(fixture.release)),
      /Retained release ZIP identity does not match retained bytes/
    );
  } finally {
    fixture.cleanup();
  }
});

test('detached verifier rejects every formerly unbound public projection claim', () => {
  const fixture = createBundleIntegrationFixture();
  try {
    publicReleaseEvidenceTestHooks.build({
      repoRoot: fixture.repository,
      releaseRoot: fixture.release,
      m110SummaryPath: fixture.summaryPath,
      routeProofPath: fixture.routeProofPath,
      generatedAt: '2026-07-15T11:00:00.000Z'
    }, fixture.verifyPackage);
    const manifestPath = path.join(fixture.release, 'public-release-evidence.json');
    const original = JSON.parse(fs.readFileSync(manifestPath, 'utf8'));
    const mutations = [
      value => { value.capture.hardwareIdentitySha256 = `sha256:${'0'.repeat(64)}`; },
      value => { value.soundcheck.routeProofId = '0'.repeat(32); },
      value => { value.soundcheck.routeSemanticSha256 = `sha256:${'0'.repeat(64)}`; },
      value => { value.soundcheck.readinessClosureAllowed = true; },
      value => { value.readinessClosureAllowed = true; },
      value => { value.externalAuthenticityVerified = true; },
      value => { value.status = 'accepted'; },
      value => { value.packageRelease.generatedAt = '2026-07-15T10:00:01.000Z'; },
      value => { value.packageRelease.detachedEvidenceReplayComplete = false; },
      value => { value.retained.committedSources.runner.logicalPath = 'tools/forged.py'; },
      value => { value.retained.m110ExecutionSummary.path = 'public-release-evidence/alternate.json'; },
      value => { value.unreviewed = true; }
    ];
    for (const mutate of mutations) {
      const tampered = structuredClone(original);
      mutate(tampered);
      rehashBundleManifest(tampered);
      fs.writeFileSync(manifestPath, `${JSON.stringify(tampered, null, 2)}\n`);
      assert.throws(
        () => publicReleaseEvidenceTestHooks.verify(fixture.release, fixture.verifyPackage),
        /projection drifted|manifest contract or bundle hash|exact builder serialization/
      );
    }
    fs.writeFileSync(manifestPath, `${JSON.stringify(original, null, 2)}\n`);
    assert.equal(publicReleaseEvidenceTestHooks.verify(
      fixture.release,
      fixture.verifyPackage
    ).verified, true);
  } finally {
    fixture.cleanup();
  }
});

test('detached verifier rejects publicized or redirected private payload bytes', () => {
  const fixture = createBundleIntegrationFixture();
  try {
    publicReleaseEvidenceTestHooks.build({
      repoRoot: fixture.repository,
      releaseRoot: fixture.release,
      m110SummaryPath: fixture.summaryPath,
      routeProofPath: fixture.routeProofPath,
      generatedAt: '2026-07-15T11:00:00.000Z'
    }, fixture.verifyPackage);
    const payloadRoot = path.join(fixture.release, 'public-release-evidence');
    const summaryPath = path.join(payloadRoot, 'm110-execution-summary.json');
    fs.chmodSync(payloadRoot, 0o755);
    assert.throws(
      () => publicReleaseEvidenceTestHooks.verify(fixture.release, fixture.verifyPackage),
      /fixed real 0700 directory/
    );
    fs.chmodSync(payloadRoot, 0o700);
    fs.chmodSync(summaryPath, 0o644);
    assert.throws(
      () => publicReleaseEvidenceTestHooks.verify(fixture.release, fixture.verifyPackage),
      /fixed private single-link 0600 file|exact canonical private 0600 file set/
    );
    fs.chmodSync(summaryPath, 0o600);
    const realSummary = path.join(payloadRoot, 'm110-summary.real.json');
    fs.renameSync(summaryPath, realSummary);
    fs.symlinkSync(realSummary, summaryPath);
    assert.throws(
      () => publicReleaseEvidenceTestHooks.verify(fixture.release, fixture.verifyPackage),
      /fixed private single-link 0600 file|exact canonical private 0600 file set/
    );
  } finally {
    fixture.cleanup();
  }
});

test('detached verifier holds a real 0700 private payload directory across its full walk', () => {
  const fixture = createBundleIntegrationFixture();
  const payloadRoot = path.join(fixture.release, 'public-release-evidence');
  const originalLstatSync = fs.lstatSync;
  const originalRealpathSync = fs.realpathSync;
  let mutated = false;
  try {
    publicReleaseEvidenceTestHooks.build({
      repoRoot: fixture.repository,
      releaseRoot: fixture.release,
      m110SummaryPath: fixture.summaryPath,
      routeProofPath: fixture.routeProofPath,
      generatedAt: '2026-07-15T11:00:00.000Z'
    }, fixture.verifyPackage);
    const staleRootStat = originalLstatSync(payloadRoot, { bigint: true });
    fs.lstatSync = (target, options) => {
      if (
        path.resolve(String(target)) === path.resolve(payloadRoot)
        && options?.bigint === true
      ) return staleRootStat;
      return originalLstatSync(target, options);
    };
    fs.realpathSync = (target, options) => {
      const resolved = originalRealpathSync(target, options);
      if (!mutated && path.resolve(String(target)) === path.resolve(payloadRoot)) {
        mutated = true;
        fs.chmodSync(payloadRoot, 0o755);
      }
      return resolved;
    };
    assert.throws(
      () => publicReleaseEvidenceTestHooks.verify(fixture.release, fixture.verifyPackage),
      /Private evidence payload directory changed during canonical identity capture/
    );
    assert.equal(mutated, true);
    assert.equal(originalLstatSync(payloadRoot).mode & 0o777, 0o755);
  } finally {
    fs.lstatSync = originalLstatSync;
    fs.realpathSync = originalRealpathSync;
    if (fs.existsSync(payloadRoot)) fs.chmodSync(payloadRoot, 0o700);
    fixture.cleanup();
  }
});

test('detached verifier rejects retained source bytes outside the selected Git tree', () => {
  const fixture = createBundleIntegrationFixture();
  try {
    publicReleaseEvidenceTestHooks.build({
      repoRoot: fixture.repository,
      releaseRoot: fixture.release,
      m110SummaryPath: fixture.summaryPath,
      routeProofPath: fixture.routeProofPath,
      generatedAt: '2026-07-15T11:00:00.000Z'
    }, fixture.verifyPackage);
    const runnerPath = path.join(
      fixture.release,
      'public-release-evidence',
      'wing-attended-soundcheck-runner.py'
    );
    fs.appendFileSync(runnerPath, '# forged retained bytes\n');
    fs.chmodSync(runnerPath, 0o600);
    const manifestPath = path.join(fixture.release, 'public-release-evidence.json');
    const manifest = JSON.parse(fs.readFileSync(manifestPath, 'utf8'));
    manifest.retained.committedSources.runner.artifact = artifactFileIdentity(
      fixture.release,
      runnerPath,
      'Forged retained runner source'
    );
    rehashBundleManifest(manifest);
    fs.writeFileSync(manifestPath, `${JSON.stringify(manifest, null, 2)}\n`);
    assert.throws(
      () => publicReleaseEvidenceTestHooks.verify(fixture.release, fixture.verifyPackage),
      /not a blob in the selected Git tree/
    );
  } finally {
    fixture.cleanup();
  }
});

test('detached verifier rejects noncanonical M110 summary bytes with duplicate keys', () => {
  const fixture = createBundleIntegrationFixture();
  try {
    publicReleaseEvidenceTestHooks.build({
      repoRoot: fixture.repository,
      releaseRoot: fixture.release,
      m110SummaryPath: fixture.summaryPath,
      routeProofPath: fixture.routeProofPath,
      generatedAt: '2026-07-15T11:00:00.000Z'
    }, fixture.verifyPackage);
    const summaryPath = path.join(
      fixture.release,
      'public-release-evidence',
      'm110-execution-summary.json'
    );
    const original = fs.readFileSync(summaryPath, 'utf8');
    assert.match(original, /^\{"accepted":true,/);
    fs.writeFileSync(
      summaryPath,
      original.replace(/^\{"accepted":true,/, '{"accepted":false,"accepted":true,')
    );
    fs.chmodSync(summaryPath, 0o600);
    const manifestPath = path.join(fixture.release, 'public-release-evidence.json');
    const manifest = JSON.parse(fs.readFileSync(manifestPath, 'utf8'));
    manifest.retained.m110ExecutionSummary = artifactFileIdentity(
      fixture.release,
      summaryPath,
      'Forged retained M110 summary'
    );
    rehashBundleManifest(manifest);
    fs.writeFileSync(manifestPath, `${JSON.stringify(manifest, null, 2)}\n`);
    assert.throws(
      () => publicReleaseEvidenceTestHooks.verify(fixture.release, fixture.verifyPackage),
      /duplicated or not sorted/
    );
  } finally {
    fixture.cleanup();
  }
});

test('detached verifier rejects reordered keys and duplicate public readiness fields', () => {
  const fixture = createBundleIntegrationFixture();
  try {
    publicReleaseEvidenceTestHooks.build({
      repoRoot: fixture.repository,
      releaseRoot: fixture.release,
      m110SummaryPath: fixture.summaryPath,
      routeProofPath: fixture.routeProofPath,
      generatedAt: '2026-07-15T11:00:00.000Z'
    }, fixture.verifyPackage);
    const manifestPath = path.join(fixture.release, 'public-release-evidence.json');
    const original = fs.readFileSync(manifestPath, 'utf8');
    const entries = Object.entries(JSON.parse(original));
    const reordered = Object.fromEntries([...entries.slice(1), entries[0]]);
    fs.writeFileSync(manifestPath, `${JSON.stringify(reordered, null, 2)}\n`);
    assert.throws(
      () => publicReleaseEvidenceTestHooks.verify(fixture.release, fixture.verifyPackage),
      /exact builder serialization/
    );
    fs.writeFileSync(
      manifestPath,
      original.replace(
        /^\{\n/,
        '{\n  "readinessClosureAllowed": true,\n  "status": "accepted",\n'
      )
    );
    assert.throws(
      () => publicReleaseEvidenceTestHooks.verify(fixture.release, fixture.verifyPackage),
      /exact canonical serialized bytes/
    );
  } finally {
    fixture.cleanup();
  }
});

test('candidate and published-manifest failures leave no accepted public manifest', () => {
  for (const failOnCall of [3, 5, 7]) {
    const fixture = createBundleIntegrationFixture();
    let calls = 0;
    const failingVerifier = () => {
      calls += 1;
      if (calls === failOnCall) throw new Error(`package replay failure ${failOnCall}`);
      return structuredClone(fixture.packageRelease);
    };
    try {
      assert.throws(() => publicReleaseEvidenceTestHooks.build({
        repoRoot: fixture.repository,
        releaseRoot: fixture.release,
        m110SummaryPath: fixture.summaryPath,
        routeProofPath: fixture.routeProofPath,
        generatedAt: '2026-07-15T11:00:00.000Z'
      }, failingVerifier), /package replay failure/);
      assert.equal(
        fs.existsSync(path.join(fixture.release, 'public-release-evidence.json')),
        false
      );
      assert.equal(
        fs.existsSync(path.join(fixture.release, '.public-release-evidence.candidate')),
        false
      );
    } finally {
      fixture.cleanup();
    }
  }
});

test('post-publication throw removes unchanged final bytes without a publisher return value', () => {
  const fixture = createBundleIntegrationFixture();
  const publishThenThrow = (...args) => {
    publishNewArtifactFileNoFollow(...args);
    throw new Error('injected failure after publication helper success');
  };
  try {
    assert.throws(() => publicReleaseEvidenceTestHooks.build({
      repoRoot: fixture.repository,
      releaseRoot: fixture.release,
      m110SummaryPath: fixture.summaryPath,
      routeProofPath: fixture.routeProofPath,
      generatedAt: '2026-07-15T11:00:00.000Z'
    }, fixture.verifyPackage, {
      publishManifest: publishThenThrow
    }), /injected failure after publication helper success/);
    assert.equal(
      fs.existsSync(path.join(fixture.release, 'public-release-evidence.json')),
      false
    );
    assert.equal(
      fs.existsSync(path.join(fixture.release, '.public-release-evidence.candidate')),
      false
    );
  } finally {
    fixture.cleanup();
  }
});

test('post-publication authority drift is rejected after final replay and unpublished', () => {
  const fixture = createBundleIntegrationFixture();
  const authorityPath = publicReleaseEvidenceTestHooks.authoritySourcePaths.find(
    value => value.endsWith('verify-macos-release.mjs')
  );
  const publishThenMutateAuthority = (...args) => {
    publishNewArtifactFileNoFollow(...args);
    fs.appendFileSync(
      path.join(fixture.repository, authorityPath),
      '// injected post-publication authority drift\n'
    );
  };
  try {
    assert.throws(() => publicReleaseEvidenceTestHooks.build({
      repoRoot: fixture.repository,
      releaseRoot: fixture.release,
      m110SummaryPath: fixture.summaryPath,
      routeProofPath: fixture.routeProofPath,
      generatedAt: '2026-07-15T11:00:00.000Z'
    }, fixture.verifyPackage, {
      publishManifest: publishThenMutateAuthority
    }), /completely clean checkout/);
    assert.equal(
      fs.existsSync(path.join(fixture.release, 'public-release-evidence.json')),
      false
    );
    assert.equal(
      fs.existsSync(path.join(fixture.release, '.public-release-evidence.candidate')),
      false
    );
  } finally {
    fixture.cleanup();
  }
});

test('committed source snapshots remain pinned when symbolic HEAD changes', () => {
  const fixture = createBundleIntegrationFixture();
  const sourcePath = 'tools/wing_attended_soundcheck_runner.py';
  const sourceTree = runGit(fixture.repository, ['rev-parse', 'HEAD^{tree}']);
  const workingPath = path.join(fixture.repository, sourcePath);
  const destinationPath = path.join(fixture.release, 'pinned-runner.py');
  try {
    fs.writeFileSync(workingPath, 'print("different symbolic HEAD")\n');
    runGit(fixture.repository, ['add', sourcePath]);
    runGit(fixture.repository, [
      '-c', 'user.name=AUTO MIXER Test',
      '-c', 'user.email=automixer-test@example.invalid',
      '-c', 'commit.gpgsign=false',
      'commit', '-q', '-m', 'different head'
    ]);

    const retained = publicReleaseEvidenceTestHooks.snapshotGitBlob({
      repoRoot: fixture.repository,
      sourceTree,
      sourcePath,
      destinationRoot: fixture.release,
      destinationPath
    });

    assert.equal(fs.readFileSync(destinationPath, 'utf8'), 'print("retained runner")\n');
    assert.equal(retained.sha256, sha256(Buffer.from('print("retained runner")\n')));
  } finally {
    fixture.cleanup();
  }
});

test('authority closure rejects assume-unchanged worktree bytes', () => {
  const fixture = createBundleIntegrationFixture();
  const authorityPath = publicReleaseEvidenceTestHooks.authoritySourcePaths.find(
    value => value.endsWith('verify-macos-release.mjs')
  );
  const workingPath = path.join(fixture.repository, authorityPath);
  const launcherAuthorityPath = 'frontend/scripts/run-public-release-evidence-bundle';
  const launcherWorkingPath = path.join(fixture.repository, launcherAuthorityPath);
  try {
    assert.equal(
      publicReleaseEvidenceTestHooks.validateAuthority(fixture.repository).pathCount,
      publicReleaseEvidenceTestHooks.authoritySourcePaths.length
    );
    runGit(fixture.repository, ['config', 'core.filemode', 'false']);
    fs.chmodSync(launcherWorkingPath, 0o644);
    assert.equal(runGit(fixture.repository, ['status', '--porcelain=v1', '-uall']), '');
    assert.throws(
      () => publicReleaseEvidenceTestHooks.validateAuthority(fixture.repository),
      /executable mode|unsafe value: core\.filemode/
    );
    fs.chmodSync(launcherWorkingPath, 0o755);
    runGit(fixture.repository, ['config', 'core.filemode', 'true']);
    runGit(fixture.repository, ['update-index', '--assume-unchanged', '--', authorityPath]);
    fs.appendFileSync(workingPath, '// hidden authority mutation\n');
    assert.equal(runGit(fixture.repository, ['status', '--porcelain=v1', '-uall']), '');
    assert.throws(
      () => publicReleaseEvidenceTestHooks.validateAuthority(fixture.repository),
      /hidden Git index flags/
    );
  } finally {
    fixture.cleanup();
  }
});

test('pinned helper Git blobs ignore a transient helper pathname replacement', async () => {
  const fixture = createBundleIntegrationFixture();
  const isolatedReleaseEvidencePath = path.join(
    fixture.repository,
    'frontend/scripts/release-evidence.mjs'
  );
  const isolated = await import(
    `${pathToFileURL(isolatedReleaseEvidencePath).href}?pin=${crypto.randomUUID()}`
  );
  const sources = Object.fromEntries(
    isolated.RELEASE_EVIDENCE_HELPER_SOURCE_PATHS.map(sourcePath => [
      sourcePath,
      fs.readFileSync(path.join(fixture.repository, sourcePath))
    ])
  );
  const artifactRoot = path.join(fixture.root, 'pinned-helper-artifacts');
  fs.mkdirSync(artifactRoot);
  const candidatePath = path.join(artifactRoot, 'candidate.json');
  const finalPath = path.join(artifactRoot, 'final.json');
  const markerPath = path.join(fixture.root, 'mutable-helper-executed');
  const publisherPath = path.join(
    fixture.repository,
    'frontend/scripts/publish-new-artifact-helper.mjs'
  );
  const originalPublisher = fs.readFileSync(publisherPath);
  const originalMode = fs.statSync(publisherPath).mode & 0o777;
  try {
    isolated.pinReleaseEvidenceHelperSources(sources);
    const candidate = isolated.writeNewArtifactFileNoFollow(
      artifactRoot,
      candidatePath,
      '{"pinned":true}\n',
      'Pinned helper candidate'
    );
    const maliciousPublisher = originalPublisher.toString('utf8').replace(
      'try {',
      `fs.writeFileSync(${JSON.stringify(markerPath)}, 'executed');\ntry {`
    );
    assert.notEqual(maliciousPublisher, originalPublisher.toString('utf8'));
    fs.writeFileSync(publisherPath, maliciousPublisher);
    fs.chmodSync(publisherPath, originalMode);
    try {
      isolated.publishNewArtifactFileNoFollow(
        artifactRoot,
        candidatePath,
        finalPath,
        candidate,
        'Pinned helper publication'
      );
    } finally {
      fs.writeFileSync(publisherPath, originalPublisher);
      fs.chmodSync(publisherPath, originalMode);
    }
    assert.equal(fs.existsSync(markerPath), false);
    assert.equal(fs.readFileSync(finalPath, 'utf8'), '{"pinned":true}\n');
  } finally {
    fixture.cleanup();
  }
});

test('stable tree digest remains hashTree-compatible and guard detects inode replacement', () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'automixer-stable-tree-'));
  try {
    const releaseEvidenceSource = fs.readFileSync(releaseEvidencePath, 'utf8');
    const stableTreeHelperSource = fs.readFileSync(
      path.join(workspaceRoot, 'frontend/scripts/stable-tree-identity-helper.mjs'),
      'utf8'
    );
    assert.match(releaseEvidenceSource, /cwd: resolvedRoot/);
    assert.match(stableTreeHelperSource, /const rootPath = '\.'/);
    assert.doesNotMatch(stableTreeHelperSource, /path\.join\(parentPath, basename\)/);
    const nested = path.join(root, 'nested');
    fs.mkdirSync(nested);
    const file = path.join(nested, 'value.txt');
    fs.writeFileSync(file, 'same bytes');
    const first = stableTreeIdentityNoFollow(root, 'Unit stable tree');
    assert.equal(first.sha256, hashTree(root));
    fs.unlinkSync(file);
    fs.writeFileSync(file, 'same bytes');
    const replacement = stableTreeIdentityNoFollow(root, 'Unit replacement tree');
    assert.equal(replacement.sha256, first.sha256);
    assert.notEqual(replacement.guardSha256, first.guardSha256);
    fs.chmodSync(file, 0o755);
    assert.notEqual(stableTreeIdentityNoFollow(root).sha256, first.sha256);
  } finally {
    fs.rmSync(root, { recursive: true, force: true });
  }
});

test('flat directory identity stays bound to the selected root inode', () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'automixer-flat-root-'));
  const selectedParent = path.join(root, 'selected');
  const heldParent = path.join(root, 'held');
  const replacementParent = path.join(root, 'replacement');
  const selected = path.join(selectedParent, 'payload');
  const replacement = path.join(replacementParent, 'payload');
  fs.mkdirSync(selected, { recursive: true });
  fs.mkdirSync(replacement, { recursive: true });
  fs.writeFileSync(path.join(selected, 'value.json'), '{"source":"A"}\n', { mode: 0o600 });
  fs.writeFileSync(path.join(replacement, 'value.json'), '{"source":"B"}\n', { mode: 0o600 });
  const stat = fs.lstatSync(selected, { bigint: true });
  const expectedRoot = {
    realPath: fs.realpathSync(selected),
    stat: Object.fromEntries([
      'dev', 'ino', 'size', 'mode', 'nlink', 'mtimeNs', 'ctimeNs'
    ].map(name => [name, stat[name].toString()]))
  };
  try {
    fs.renameSync(selectedParent, heldParent);
    fs.renameSync(replacementParent, selectedParent);
    assert.throws(
      () => stableFlatDirectoryFileIdentitiesNoFollow(
        selected,
        expectedRoot,
        ['value.json'],
        'Selected flat root'
      ),
      /selected real directory|changed around stable tree helper/
    );
  } finally {
    fs.rmSync(root, { recursive: true, force: true });
  }
});

test('pinned helper runtime rejects process executable path drift', () => {
  const pinned = pinReleaseEvidenceHelperRuntime();
  const previousExecPath = process.execPath;
  try {
    process.execPath = '/definitely/not/the/pinned/node';
    assert.throws(
      () => pinReleaseEvidenceHelperRuntime(),
      /ENOENT|helper runtime/
    );
  } finally {
    process.execPath = previousExecPath;
  }
  assert.equal(pinReleaseEvidenceHelperRuntime().path, pinned.path);
});

test('stable tree rejects newline row delimiters in names and symlink targets', () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'automixer-stable-tree-row-'));
  try {
    fs.writeFileSync(path.join(root, 'bad\nname'), 'x');
    assert.throws(() => stableTreeIdentityNoFollow(root), /row delimiter/);
    fs.rmSync(path.join(root, 'bad\nname'));
    fs.symlinkSync('bad\ntarget', path.join(root, 'link'));
    assert.throws(() => stableTreeIdentityNoFollow(root), /row delimiter/);
  } finally {
    fs.rmSync(root, { recursive: true, force: true });
  }
});

test('public-release evidence rejects missing launcher marker and visible preload surfaces', () => {
  const completed = spawnSync(process.execPath, [bundlePath, 'verify', '--release-root', '/tmp'], {
    encoding: 'utf8',
    env: { PATH: process.env.PATH || '' }
  });
  assert.equal(completed.status, 1);
  assert.match(completed.stderr, /fixed shell launcher/);

  const preloadRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'automixer-node-preload-'));
  const preloadPath = path.join(preloadRoot, 'preload.mjs');
  try {
    fs.writeFileSync(preloadPath, 'globalThis.__AUTOMIXER_PRELOADED__ = true;\n');
    const preloaded = spawnSync(
      process.execPath,
      [bundlePath, 'verify', '--release-root', '/tmp'],
      {
        cwd: path.join(workspaceRoot, 'frontend'),
        encoding: 'utf8',
        env: productionRuntimeEnvironment({
          NODE_OPTIONS: `--import=${preloadPath}`
        })
      }
    );
    assert.equal(preloaded.status, 1);
    assert.match(preloaded.stderr, /fixed clean Node runtime/);

    const cliPreloaded = spawnSync(
      process.execPath,
      [`--import=${preloadPath}`, bundlePath, 'verify', '--release-root', '/tmp'],
      {
        cwd: path.join(workspaceRoot, 'frontend'),
        encoding: 'utf8',
        env: productionRuntimeEnvironment()
      }
    );
    assert.equal(cliPreloaded.status, 1);
    assert.match(cliPreloaded.stderr, /fixed clean Node runtime/);

    const gitPoisoned = spawnSync(
      process.execPath,
      [bundlePath, 'verify', '--release-root', '/tmp'],
      {
        cwd: path.join(workspaceRoot, 'frontend'),
        encoding: 'utf8',
        env: productionRuntimeEnvironment({
          GIT_DIR: '/definitely/not/a/repository'
        })
      }
    );
    assert.equal(gitPoisoned.status, 1);
    assert.match(gitPoisoned.stderr, /fixed clean Node runtime/);
    assert.doesNotMatch(gitPoisoned.stderr, /not a git repository/);
  } finally {
    fs.rmSync(preloadRoot, { recursive: true, force: true });
  }
});

test('public-release evidence launcher is executable, fixed-shell, and clean-environment only', () => {
  const source = fs.readFileSync(launcherPath, 'utf8');
  assert.equal(fs.statSync(launcherPath).mode & 0o111, 0o111);
  assert.match(source, /^#!\/bin\/sh -p/);
  assert.match(source, /exec \/usr\/bin\/env -i/);
  assert.match(source, /HOME=\/var\/empty/);
  assert.match(source, /TMPDIR=\/tmp/);
  assert.match(source, /LANG=C/);
  assert.match(source, /LC_ALL=C/);
  assert.match(source, /AUTOMIXER_PUBLIC_RELEASE_EVIDENCE_CLEAN_ENV=1/);
  assert.match(source, /public-release-evidence-bundle\.mjs" "\$@"/);
  assert.doesNotMatch(source, /npm|npx|bash/);

  const hostileRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'automixer-hostile-home-'));
  const hostileHome = path.join(hostileRoot, 'home');
  const hookPath = path.join(hostileRoot, 'fsmonitor-hook');
  const markerPath = path.join(hostileRoot, 'hook-executed');
  fs.mkdirSync(hostileHome);
  try {
    fs.writeFileSync(
      hookPath,
      `#!/bin/sh\n/usr/bin/touch ${JSON.stringify(markerPath)}\nexit 0\n`
    );
    fs.chmodSync(hookPath, 0o755);
    fs.writeFileSync(
      path.join(hostileHome, '.gitconfig'),
      `[core]\n\tfsmonitor = ${hookPath}\n`
    );
    const completed = spawnSync(
      launcherPath,
      ['verify', '--release-root', '/tmp'],
      {
        cwd: path.join(workspaceRoot, 'frontend'),
        encoding: 'utf8',
        env: {
          HOME: hostileHome,
          AUTOMIXER_NODE_BIN: process.execPath
        }
      }
    );
    assert.equal(completed.status, 1);
    assert.match(
      completed.stderr,
      /completely clean checkout|public-release evidence manifest|public-release-evidence\.json/
    );
    assert.doesNotMatch(completed.stderr, /fixed clean Node runtime/);
    assert.equal(fs.existsSync(markerPath), false);
  } finally {
    fs.rmSync(hostileRoot, { recursive: true, force: true });
  }
});

test('release-tool Git reads reject repository command-execution config', () => {
  const fixture = createBundleIntegrationFixture();
  const hookPath = path.join(fixture.root, 'repository-fsmonitor-hook');
  const markerPath = path.join(fixture.root, 'repository-hook-executed');
  const attributesPath = path.join(fixture.root, 'hostile-attributes');
  try {
    fs.writeFileSync(
      hookPath,
      `#!/bin/sh\n/usr/bin/touch ${JSON.stringify(markerPath)}\n/bin/cat\n`
    );
    fs.chmodSync(hookPath, 0o755);
    runGit(fixture.repository, ['config', 'core.fsmonitor', hookPath]);
    assert.throws(
      () => publicReleaseEvidenceTestHooks.validateAuthority(fixture.repository),
      /unsafe for evidence reads: core\.fsmonitor/
    );
    assert.equal(fs.existsSync(markerPath), false);

    runGit(fixture.repository, ['config', '--unset', 'core.fsmonitor']);
    fs.writeFileSync(attributesPath, '*.mjs filter=hostile\n');
    runGit(fixture.repository, ['config', 'core.attributesFile', attributesPath]);
    runGit(fixture.repository, ['config', 'filter.hostile.clean', hookPath]);
    runGit(fixture.repository, ['config', 'filter.hostile.required', 'true']);
    assert.throws(
      () => publicReleaseEvidenceTestHooks.validateAuthority(fixture.repository),
      /unsafe for evidence reads: core\.attributesfile|unsafe for evidence reads: filter\.hostile/
    );
    assert.equal(fs.existsSync(markerPath), false);
  } finally {
    fixture.cleanup();
  }
});

test('release-tool Git status detects case-only and mode-only tracked drift', () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'automixer-git-case-'));
  try {
    runGit(root, ['init', '-q']);
    fs.writeFileSync(path.join(root, 'Alpha.txt'), 'case-bound bytes\n');
    const toolPath = path.join(root, 'tool.sh');
    fs.writeFileSync(toolPath, '#!/bin/sh\nexit 0\n');
    fs.chmodSync(toolPath, 0o755);
    runGit(root, ['add', 'Alpha.txt', 'tool.sh']);
    runGit(root, [
      '-c', 'user.name=AUTO MIXER Test',
      '-c', 'user.email=automixer-test@example.invalid',
      '-c', 'commit.gpgsign=false',
      'commit', '-q', '-m', 'case fixture'
    ]);
    runGit(root, ['config', 'core.ignorecase', 'true']);
    fs.renameSync(path.join(root, 'Alpha.txt'), path.join(root, 'alpha.txt'));
    const status = runReleaseEvidenceGit(root, ['status', '--porcelain=v1', '-uall'])
      .toString('utf8').trim();
    assert.notEqual(status, '');
    fs.renameSync(path.join(root, 'alpha.txt'), path.join(root, 'Alpha.txt'));
    fs.chmodSync(toolPath, 0o644);
    const modeStatus = runReleaseEvidenceGit(root, ['status', '--porcelain=v1', '-uall'])
      .toString('utf8').trim();
    assert.notEqual(modeStatus, '');
  } finally {
    fs.rmSync(root, { recursive: true, force: true });
  }
});
