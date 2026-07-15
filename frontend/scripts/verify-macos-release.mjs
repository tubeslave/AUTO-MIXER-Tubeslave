#!/usr/bin/env node

import crypto from 'node:crypto';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import process from 'node:process';
import { spawnSync } from 'node:child_process';
import { createRequire } from 'node:module';
import { fileURLToPath } from 'node:url';

import {
  artifactFileIdentity,
  artifactRelativePath,
  assertIdentityUnchanged,
  copyStableRegularFileNoFollow,
  fileIdentity,
  hashTree,
  identitiesEqual,
  logicalFileIdentity,
  publishNewArtifactFileNoFollow,
  readStableJsonFileNoFollow,
  removeArtifactFileNoFollow,
  resolveArtifactPath,
  writeNewArtifactFileNoFollow,
  sha256File,
  verifyPackagedRuntimeSmokeManifest
} from './release-evidence.mjs';
import {
  assertAcceptedNotaryLog,
  assertDmgTicket,
  assertMainExecutableTicket,
  normalizeCdHash
} from './notarytool-evidence.mjs';

const scriptPath = fileURLToPath(import.meta.url);
const require = createRequire(import.meta.url);
const { extractAll: extractAsar } = require('@electron/asar');
const {
  BACKEND_NETWORK_SANDBOX_PROBES,
  BACKEND_SANDBOX_PROFILE_SHA256
} = require('../public/electronSecurity.js');
const frontendRoot = path.resolve(path.dirname(scriptPath), '..');
const repoRoot = path.resolve(frontendRoot, '..');
const BACKEND_ENTITLEMENT = 'com.apple.security.cs.disable-library-validation';
const RUNTIME_SOURCE_FILES = ['electron.js', 'electronSecurity.js', 'preload.js'];
const CONFIG_SOURCE_FILES = [
  'default_config.json',
  'automixer.yaml',
  'muq_eval.yaml',
  'perceptual.yaml',
  'source_knowledge.yaml'
];
export const EXPECTED_RUNTIME_CRITICAL_SOURCE_PATHS = [
  'backend/auth.py',
  'backend/audio_capture.py',
  'backend/auto_soundcheck_engine.py',
  'backend/contracts/m110_m006_restore_20260711.json',
  'backend/decision_queue_wire_contract.py',
  'backend/handlers/__init__.py',
  'backend/handlers/automation_handlers.py',
  'backend/handlers/connection_handlers.py',
  'backend/handlers/mixer_handlers.py',
  'backend/handlers/product_state_handlers.py',
  'backend/handlers/soundcheck_handlers.py',
  'backend/m110_cp05_hardware_gate.py',
  'backend/m110_manual_write_wal.py',
  'backend/m110_m006_restore.py',
  'backend/m110_processor_enable.py',
  'backend/maintenance_dsp_reset.py',
  'backend/master_capture_route_proof.py',
  'backend/mixer_client_base.py',
  'backend/operator_mode_policy.py',
  'backend/operator_proposal_queue.py',
  'backend/operator_proposal_result_store.py',
  'backend/osc/enhanced_osc_client.py',
  'backend/runtime_identity.py',
  'backend/server.py',
  'backend/soundcheck_mix_readiness.py',
  'backend/soundcheck_transaction_plan.py',
  'backend/supervised_console_parameter_registry.py',
  'backend/supervised_console_transaction.py',
  'backend/supervised_family_proof_plan.py',
  'backend/supervised_proof_capability.py',
  'backend/wing_capture_hardware_identity.py',
  'backend/wing_client.py',
  'backend/wing_supervised_console_adapters.py',
  'backend/ws_transport.py',
  'tools/wing_attended_soundcheck_runner.py',
  'tools/wing_supervised_family_proof_runner.py'
].sort();

function canonicalJson(value) {
  if (Array.isArray(value)) return `[${value.map(canonicalJson).join(',')}]`;
  if (value && typeof value === 'object') {
    return `{${Object.keys(value).sort().map(key => (
      `${JSON.stringify(key)}:${canonicalJson(value[key])}`
    )).join(',')}}`;
  }
  return JSON.stringify(value);
}

function canonicalSha256(value) {
  return crypto.createHash('sha256').update(canonicalJson(value)).digest('hex');
}

export function runtimeIdentityManifestSha256(identity) {
  const canonical = JSON.parse(JSON.stringify(identity));
  delete canonical.manifest_sha256;
  delete canonical.request_challenge_nonce;
  return canonicalSha256(canonical);
}

export function packagedSourceManifestSha256(manifest) {
  const canonical = JSON.parse(JSON.stringify(manifest));
  delete canonical.manifest_sha256;
  return canonicalSha256(canonical);
}

function run(command, args, options = {}) {
  const completed = spawnSync(command, args, {
    cwd: options.cwd || frontendRoot,
    env: options.env || process.env,
    encoding: 'utf8',
    input: options.input,
    maxBuffer: 64 * 1024 * 1024
  });
  if (completed.error) throw completed.error;
  if (completed.status !== 0 && options.check !== false) {
    throw new Error(
      `${path.basename(command)} failed with exit code ${completed.status}: `
      + `${completed.stderr || completed.stdout}`.trim()
    );
  }
  return {
    status: completed.status,
    stdout: completed.stdout || '',
    stderr: completed.stderr || ''
  };
}

function combined(result) {
  return `${result.stdout}\n${result.stderr}`.trim();
}

function plistJson(xml) {
  return JSON.parse(run(
    '/usr/bin/plutil',
    ['-convert', 'json', '-o', '-', '-'],
    { input: xml }
  ).stdout);
}

function readPlist(filePath) {
  return JSON.parse(run(
    '/usr/bin/plutil',
    ['-convert', 'json', '-o', '-', filePath]
  ).stdout);
}

function readJson(filePath) {
  return JSON.parse(fs.readFileSync(filePath, 'utf8'));
}

export function parseCodesignDetails(output) {
  const text = String(output || '');
  return {
    identifier: text.match(/^Identifier=(.+)$/m)?.[1]?.trim() || '',
    authorities: [...text.matchAll(/^Authority=(.+)$/gm)].map(match => match[1].trim()),
    teamId: text.match(/^TeamIdentifier=(.+)$/m)?.[1]?.trim() || '',
    cdhash: text.match(/^CDHash=([0-9a-f]+)$/mi)?.[1]?.trim().toLowerCase() || '',
    runtime: /flags=.*\(runtime\)/.test(text),
    timestamp: text.match(/^Timestamp=(.+)$/m)?.[1]?.trim() || ''
  };
}

export function assertDeveloperIdSignature(
  details,
  { label, teamId, requireRuntime = true }
) {
  if (!details.authorities.some(value => value.startsWith('Developer ID Application:'))) {
    throw new Error(`${label} is not signed with Developer ID Application`);
  }
  if (details.teamId !== teamId) throw new Error(`${label} Team ID mismatch`);
  if (requireRuntime && !details.runtime) {
    throw new Error(`${label} does not use hardened runtime`);
  }
  if (!details.timestamp) throw new Error(`${label} has no secure signing timestamp`);
}

export function assertBackendEntitlements(entitlements) {
  const keys = Object.keys(entitlements || {}).sort();
  if (
    keys.length !== 1
    || keys[0] !== BACKEND_ENTITLEMENT
    || entitlements[BACKEND_ENTITLEMENT] !== true
  ) {
    throw new Error('Backend entitlements do not match the exact release allowlist');
  }
}

export function assertExactEntitlements(actual, expected, label) {
  const normalize = value => Object.fromEntries(
    Object.entries(value || {}).sort(([left], [right]) => left.localeCompare(right))
  );
  if (JSON.stringify(normalize(actual)) !== JSON.stringify(normalize(expected))) {
    throw new Error(`${label} entitlements do not match the exact release allowlist`);
  }
}

export function assertDmgNotarizationEvidence(
  evidence,
  expectedDmg,
  {
    artifactRoot = path.dirname(expectedDmg?.path || ''),
    rawLog = null,
    rawLogIdentity = null,
    dmgCdhash = '',
    allowLegacy = false
  } = {}
) {
  const legacyAllowed = evidence?.schema === 'automixer.dmg_notarization.v1'
    && allowLegacy === true
    && (
      evidence?.pathPolicy == null
      || evidence.pathPolicy === 'artifact_root_relative_v1'
    );
  const verifiedDmgPath = assertRegularContainedFile(
    artifactRoot,
    expectedDmg?.path,
    'Notarized DMG'
  );
  let recordedDmgPath = null;
  try {
    recordedDmgPath = resolveArtifactPath(
      artifactRoot,
      evidence?.dmg?.path,
      'DMG notarization evidence path',
      {
        allowLegacyAbsoluteWithinRoot: legacyAllowed && evidence?.pathPolicy == null,
        mustExist: true
      }
    );
  } catch {
    recordedDmgPath = null;
  }
  if (legacyAllowed) {
    if (
      recordedDmgPath !== verifiedDmgPath
      || evidence.status !== 'Accepted'
      || !/^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i.test(
        String(evidence.submissionId || '')
      )
      || !Number.isFinite(new Date(evidence.generatedAt).getTime())
      || evidence.dmg?.size !== expectedDmg.size
      || evidence.dmg?.sha256 !== expectedDmg.sha256
    ) {
      throw new Error('Legacy DMG notarization evidence does not bind to the final DMG');
    }
    return;
  }
  if (evidence?.schema !== 'automixer.dmg_notarization.v2') {
    throw new Error('DMG notarization evidence schema mismatch');
  }
  const expectedRawLogPath = path.join(path.resolve(artifactRoot), 'dmg-notary-log.json');
  let recordedRawLogPath = null;
  try {
    recordedRawLogPath = resolveArtifactPath(
      artifactRoot,
      evidence?.rawLog?.path,
      'DMG retained Apple notary log',
      { mustExist: true }
    );
  } catch {
    recordedRawLogPath = null;
  }
  const summary = assertAcceptedNotaryLog(rawLog, {
    submissionId: evidence?.submission?.id,
    archiveFilename: evidence?.submission?.archiveFilename,
    archiveSha256: evidence?.submission?.archive?.sha256,
    startedAt: evidence?.submission?.startedAt,
    completedAt: evidence?.submission?.completedAt
  });
  const verifiedDmgCdhash = normalizeCdHash(
    dmgCdhash,
    'Verified final DMG CDHash'
  );
  const ticket = assertDmgTicket(rawLog, {
    expectedPath: evidence?.submission?.archiveFilename,
    expectedCdHash: verifiedDmgCdhash
  });
  if (
    evidence.pathPolicy !== 'artifact_root_relative_v1'
    || recordedDmgPath !== verifiedDmgPath
    || recordedRawLogPath !== expectedRawLogPath
    || evidence.status !== 'Accepted'
    || !Number.isFinite(new Date(evidence.generatedAt).getTime())
    || !Number.isSafeInteger(evidence?.submission?.archive?.size)
    || evidence.submission.archive.size <= 0
    || evidence.dmg?.size !== expectedDmg.size
    || evidence.dmg?.sha256 !== expectedDmg.sha256
    || normalizeCdHash(evidence?.signature?.cdhashBeforeStaple) !== verifiedDmgCdhash
    || normalizeCdHash(evidence?.signature?.cdhashAfterStaple) !== verifiedDmgCdhash
    || !identitiesEqual(evidence.rawLog, rawLogIdentity)
    || canonicalJson(evidence.appleLog) !== canonicalJson(summary)
    || canonicalJson(evidence.ticket) !== canonicalJson(ticket)
  ) {
    throw new Error('DMG notarization evidence does not bind to the final DMG');
  }
}

export function assertAppNotarizationEvidence(evidence, context) {
  const artifactRoot = path.resolve(context.artifactRoot);
  let recordedAppPath = null;
  let recordedRawLogPath = null;
  try {
    recordedAppPath = resolveArtifactPath(
      artifactRoot,
      evidence?.app?.path,
      'Notarized application evidence path',
      { mustExist: true }
    );
    recordedRawLogPath = resolveArtifactPath(
      artifactRoot,
      evidence?.rawLog?.path,
      'Application retained Apple notary log',
      { mustExist: true }
    );
  } catch {
    recordedAppPath = null;
    recordedRawLogPath = null;
  }
  const executableRelativePath = `Contents/MacOS/${context.executableName}`;
  const mainSignature = context.signatures?.[`code:${executableRelativePath}`];
  const mainCdhash = normalizeCdHash(
    mainSignature?.cdhash,
    'Verified application main executable CDHash'
  );
  const appCdhash = normalizeCdHash(
    context.signatures?.application?.cdhash,
    'Verified application bundle CDHash'
  );
  const summary = assertAcceptedNotaryLog(context.rawLog, {
    submissionId: evidence?.submission?.id,
    archiveFilename: evidence?.submission?.archiveFilename,
    archiveSha256: evidence?.submission?.archive?.sha256,
    startedAt: evidence?.submission?.startedAt,
    completedAt: evidence?.submission?.completedAt
  });
  const ticket = assertMainExecutableTicket(context.rawLog, {
    expectedPathSuffix: path.basename(context.appPath),
    expectedCdHash: appCdhash,
    architecture: 'arm64',
    expectedArchiveFilename: evidence?.submission?.archiveFilename
  });
  if (
    evidence?.schema !== 'automixer.app_notarization.v1'
    || evidence.pathPolicy !== 'artifact_root_relative_v1'
    || evidence.status !== 'Accepted'
    || !Number.isFinite(new Date(evidence.generatedAt).getTime())
    || recordedAppPath !== path.resolve(context.appPath)
    || recordedRawLogPath !== path.join(artifactRoot, 'app-notary-log.json')
    || evidence.app?.treeSha256 !== context.appTreeSha256
    || evidence.mainExecutable?.path !== executableRelativePath
    || evidence.mainExecutable?.architecture !== 'arm64'
    || normalizeCdHash(evidence.mainExecutable?.cdhash) !== mainCdhash
    || appCdhash !== mainCdhash
    || !Number.isSafeInteger(evidence?.submission?.archive?.size)
    || evidence.submission.archive.size <= 0
    || !identitiesEqual(evidence.rawLog, context.rawLogIdentity)
    || canonicalJson(evidence.appleLog) !== canonicalJson(summary)
    || canonicalJson(evidence.ticket) !== canonicalJson(ticket)
  ) {
    throw new Error('Application notarization evidence does not bind to the verified app');
  }
}

export function assertFreshArtifact(stat, notBefore, label) {
  const lowerBound = new Date(notBefore).getTime();
  if (!Number.isFinite(lowerBound)) throw new Error('--not-before must be an ISO timestamp');
  if (stat.mtimeMs < lowerBound) throw new Error(`${label} predates this release build`);
}

export function assertSourceBinding(expected, actual, label = 'release verification') {
  if (actual.head !== expected.head) throw new Error(`Source HEAD mismatch during ${label}`);
  if (actual.tree !== expected.tree) throw new Error(`Source tree mismatch during ${label}`);
  if (actual.status) throw new Error(`Source checkout is dirty during ${label}`);
}

export function assertPassingSmokeEvidence(result, context) {
  if (result?.schema !== 'automixer.packaged_runtime_smoke.v1') {
    throw new Error('Packaged runtime smoke schema mismatch');
  }
  const legacyAllowed = result.pathPolicy == null && context.allowLegacy === true;
  if (
    result.pathPolicy !== 'packaged_build_artifact_root_relative_v1'
    && !legacyAllowed
  ) {
    throw new Error('Packaged runtime smoke path policy mismatch');
  }
  const artifactRoot = path.resolve(
    context.artifactRoot || path.dirname(context.buildProfile?.path || '')
  );
  const resolveRecordedPath = (recordedPath, label, mustExist = false) => (
    resolveArtifactPath(artifactRoot, recordedPath, label, {
      allowLegacyAbsoluteWithinRoot: legacyAllowed,
      mustExist
    })
  );
  if (result.pass !== true) throw new Error('Packaged runtime smoke did not pass');
  if (
    result.readOnly !== true
    || result.contactsWing !== false
    || result.liveWrites !== false
    || result.liveAuthorityGranted !== false
  ) {
    throw new Error('Packaged runtime smoke does not prove a read-only no-WING run');
  }
  if (
    result.provenance?.head !== context.expectedHead
    || result.provenance?.tree !== context.expectedTree
    || result.provenance?.worktreeDirty !== false
    || result.provenance?.sourceBindingStrength !== 'clean_checkout'
  ) {
    throw new Error('Packaged runtime smoke source binding mismatch');
  }
  if (
    resolveRecordedPath(result.appPath, 'Packaged runtime smoke app')
      !== path.resolve(context.appPath)
  ) {
    throw new Error('Packaged runtime smoke targets a different application');
  }
  if (
    result.before?.appTree?.sha256 !== context.appTreeSha256
    || result.after?.appTree?.sha256 !== context.appTreeSha256
    || result.appTreeUnchanged !== true
  ) {
    throw new Error('Packaged runtime smoke application-tree binding mismatch');
  }
  const expectedBlockedOperations = ['auto_connect', 'connect_wing', 'scan_mixers'];
  const runtimeIdentity = result.runtimeProbe?.runtimeIdentity;
  const runtimeCodeFiles = Array.isArray(runtimeIdentity?.code?.files)
    ? runtimeIdentity.code.files
    : [];
  const runtimeCriticalPaths = runtimeCodeFiles.map(row => row?.path).sort();
  const packagedManifest = {
    schema: runtimeIdentity?.launch_binding?.packaged_manifest_schema,
    source: runtimeIdentity?.source,
    code: runtimeIdentity?.code
  };
  const runtimeIdentityValid = Boolean(
    runtimeIdentity?.schema === 'automixer.backend_runtime_identity.v2'
    && runtimeIdentity?.runtime_kind === 'packaged'
    && Number.isInteger(runtimeIdentity?.process?.pid)
    && runtimeIdentity?.process?.instance_nonce === runtimeIdentity?.instance_nonce
    && runtimeIdentity?.process?.started_at_utc === runtimeIdentity?.started_at_utc
    && result.processGroup?.some(row => (
      row?.role === 'backend' && row?.pid === runtimeIdentity.process.pid
    ))
    && runtimeIdentity?.request_challenge_nonce
      === result.runtimeProbe?.runtimeIdentityChallengeNonce
    && runtimeIdentity?.manifest_sha256
      === runtimeIdentityManifestSha256(runtimeIdentity)
    && runtimeIdentity?.source?.schema === 'automixer.backend_runtime_source.v1'
    && runtimeIdentity?.source?.observation === 'packaged_build_manifest'
    && runtimeIdentity?.source?.head === context.expectedHead
    && runtimeIdentity?.source?.tree === context.expectedTree
    && runtimeIdentity?.source?.worktree_dirty === false
    && JSON.stringify(runtimeCriticalPaths)
      === JSON.stringify(EXPECTED_RUNTIME_CRITICAL_SOURCE_PATHS)
    && runtimeCodeFiles.every(row => (
      row?.exists === true
      && Number.isInteger(row.mode)
      && row.mode >= 0
      && row.mode <= 0o777
      && Number.isInteger(row.size)
      && row.size >= 0
      && /^[0-9a-f]{64}$/.test(row.sha256 || '')
    ))
    && /^[0-9a-f]{64}$/.test(runtimeIdentity?.code?.files_sha256 || '')
    && runtimeIdentity?.source?.source_content_sha256
      === runtimeIdentity?.code?.files_sha256
    && runtimeIdentity?.launch_binding?.mode === 'packaged_build_manifest'
    && runtimeIdentity?.launch_binding?.bound_before_process_start === true
    && runtimeIdentity?.launch_binding?.proof_eligible === false
    && runtimeIdentity?.launch_binding?.authenticated_launcher_bound === true
    && runtimeIdentity?.launch_binding?.launch_token_sha256
      === result.bootstrap?.tokenSha256
    && /^[0-9a-f]{64}$/.test(result.bootstrap?.tokenSha256 || '')
    && runtimeIdentity?.launch_binding?.packaged_manifest_schema
      === 'automixer.packaged_runtime_source_manifest.v1'
    && runtimeIdentity?.launch_binding?.packaged_manifest_name
      === 'automixer-runtime-source-manifest.json'
    && runtimeIdentity?.launch_binding?.packaged_manifest_sha256
      === packagedSourceManifestSha256(packagedManifest)
    && runtimeIdentity?.launch_binding?.packaged_manifest_file_sha256
      === result.runtimeIdentityValidation?.embeddedManifest?.sha256
    && runtimeIdentity?.launch_binding?.packaged_manifest_size
      === result.runtimeIdentityValidation?.embeddedManifest?.size
    && runtimeIdentity?.backend?.path === 'backend/server'
    && runtimeIdentity?.backend?.exists === true
    && runtimeIdentity?.backend?.size === result.before?.packageFiles?.backend?.size
    && runtimeIdentity?.backend?.sha256 === result.before?.packageFiles?.backend?.sha256
    && runtimeIdentity?.websocket?.host === '127.0.0.1'
    && runtimeIdentity?.websocket?.port === 8765
    && runtimeIdentity?.websocket?.loopback_only === true
    && runtimeIdentity?.read_only_identity === true
    && runtimeIdentity?.live_authority_granted === false
  );
  const expectedSandboxProbeNames = Object.keys(BACKEND_NETWORK_SANDBOX_PROBES).sort();
  const exactProbeResults = value => (
    JSON.stringify(Object.keys(value || {}).sort()) === JSON.stringify(expectedSandboxProbeNames)
    && expectedSandboxProbeNames.every(name => value?.[name] === true)
  );
  if (
    result.runtime?.cdpReady !== true
    || result.runtime?.backendReadyWithinBudget !== true
    || result.runtime?.backendReadyBudgetMs !== 20_000
    || !Number.isFinite(result.runtime?.backendReadyMs)
    || result.runtime.backendReadyMs < 0
    || result.runtime.backendReadyMs > result.runtime.backendReadyBudgetMs
    || result.bootstrap?.validShape !== true
    || result.bootstrap?.packagedOfflineLock !== true
    || result.bootstrap?.backendNetworkSandboxed !== true
    || result.bootstrap?.backendNetworkSandboxProfileSha256
      !== result.networkEnforcement?.profileSha256
    || !exactProbeResults(result.bootstrap?.backendNetworkSandboxProbeResults)
    || result.invalidToken?.rejected !== true
    || result.runtimeProbe?.authenticated !== true
    || runtimeIdentityValid !== true
    || result.runtimeIdentityValidation?.valid !== true
    || !result.runtimeIdentityValidation?.checks
    || Object.values(result.runtimeIdentityValidation.checks).some(value => value !== true)
    || JSON.stringify(result.runtimeProbe?.runtimeIdentity)
      !== JSON.stringify(result.runtimeIdentityValidation?.identity)
    || result.runtimeProbe?.mode?.mode !== 'analyze'
    || result.runtimeProbe?.mode?.packagedOfflineLock !== true
    || result.runtimeProbe?.mode?.mixerTransportLocked !== true
    || result.blockedOperationsPass !== true
    || JSON.stringify(result.runtimeProbe?.blockedOperations) !== JSON.stringify(expectedBlockedOperations)
    || result.reloadRecovery?.staleSessionTokenRemoved !== true
    || result.reloadRecovery?.secureBootstrapFailureShown !== false
    || result.reloadRecovery?.offlineLockShown !== true
    || result.reloadRecovery?.hardwareControlsHidden !== true
  ) {
    throw new Error('Packaged runtime smoke product-surface evidence is incomplete');
  }
  if (
    result.networkEnforcement?.schema
      !== 'automixer.packaged_runtime_network_enforcement.v1'
    || result.networkEnforcement?.mechanism !== 'macos_sandbox_exec'
    || result.networkEnforcement?.scope !== 'packaged_backend_process_tree'
    || result.networkEnforcement?.nonLoopbackOutboundDenied !== true
    || result.networkEnforcement?.loopbackOutboundAllowed !== true
    || result.networkEnforcement?.preflightPassed !== true
    || result.networkEnforcement?.backendBoundaryReported !== true
    || result.networkEnforcement?.profileMatchesBackend !== true
    || result.networkEnforcement?.probeMatrixMatchesBackend !== true
    || result.networkEnforcement?.profileSha256 !== BACKEND_SANDBOX_PROFILE_SHA256
    || !exactProbeResults(result.networkEnforcement?.probeResults)
    || !/^[0-9a-f]{64}$/.test(String(result.networkEnforcement?.profileSha256 || ''))
    || result.launchCopy?.signatureVerifiedBefore !== true
    || result.launchCopy?.signatureVerifiedAfter !== true
    || result.launchCopy?.matchesSourceApp !== true
    || result.launchCopy?.unchanged !== true
    || result.launchCopy?.stillMatchesSourceApp !== true
    || result.launchCopy?.treeSha256Before !== context.appTreeSha256
    || result.launchCopy?.treeSha256After !== context.appTreeSha256
  ) {
    throw new Error('Packaged runtime smoke launch-copy or network enforcement is incomplete');
  }
  if (
    result.canonicalLogUnchanged !== true
    || result.canonicalDecisionTraceUnchanged !== true
    || result.canonicalDecisionMemoryUnchanged !== true
    || result.consoleWalUnchanged !== true
    || result.backendBundleUnchanged !== true
    || result.packageFilesUnchanged !== true
    || result.network?.externalConnections?.length !== 0
    || result.network?.wingPortSockets?.length !== 0
    || result.network?.monitorErrors?.length !== 0
    || !(result.network?.sampleCount > 1)
    || result.tempWriteArtifactsAbsent !== true
  ) {
    throw new Error('Packaged runtime smoke safety invariants are incomplete');
  }
  if (
    result.buildProfile?.schema !== 'automixer.macos_build_profile.v1'
    || result.buildProfile?.profile !== 'release'
    || result.buildProfile?.releaseEligible !== false
    || result.buildProfile?.bindingValid !== true
    || result.buildProfile?.appTreeMatches !== true
    || resolveRecordedPath(
      result.buildProfile?.path,
      'Packaged runtime smoke build profile'
    ) !== path.resolve(context.buildProfile?.path || '')
    || result.buildProfile?.size !== context.buildProfile?.size
    || result.buildProfile?.sha256 !== context.buildProfile?.sha256
    || result.buildProfile?.source?.head !== context.expectedHead
    || result.buildProfile?.source?.tree !== context.expectedTree
    || result.buildProfile?.source?.worktreeDirty !== false
    || resolveRecordedPath(
      result.buildProfile?.declaredApp?.path,
      'Packaged runtime smoke declared app'
    ) !== path.resolve(context.appPath)
    || result.buildProfile?.declaredApp?.treeSha256 !== context.appTreeSha256
  ) {
    throw new Error('Packaged runtime smoke build-profile binding is incomplete');
  }
  const expectedBackendBundlePath = path.join(
    context.appPath, 'Contents', 'Resources', 'backend'
  );
  if (
    resolveRecordedPath(
      result.before?.backendBundle?.path,
      'Packaged runtime smoke before backend bundle'
    ) !== path.resolve(expectedBackendBundlePath)
    || resolveRecordedPath(
      result.after?.backendBundle?.path,
      'Packaged runtime smoke after backend bundle'
    ) !== path.resolve(expectedBackendBundlePath)
    || result.before?.backendBundle?.sha256 !== context.backendBundleTreeSha256
    || result.after?.backendBundle?.sha256 !== context.backendBundleTreeSha256
  ) {
    throw new Error('Packaged runtime smoke backend-bundle binding mismatch');
  }
}

export function assertBuildMetadata(metadata, context) {
  if (metadata?.schema !== 'automixer.macos_release_build_metadata.v1') {
    throw new Error('Release build metadata schema mismatch');
  }
  const legacyAllowed = metadata.pathPolicy == null && context.allowLegacy === true;
  if (
    metadata.pathPolicy !== 'artifact_root_relative_and_logical_v1'
    && !legacyAllowed
  ) {
    throw new Error('Release build metadata path policy mismatch');
  }
  if (
    metadata.source?.head !== context.expectedHead
    || metadata.source?.tree !== context.expectedTree
    || metadata.source?.cleanCheckout !== true
  ) {
    throw new Error('Release build metadata source binding mismatch');
  }
  if (
    !identitiesEqual(metadata.locks?.frontendPackageLock, context.frontendPackageLock)
    || !identitiesEqual(metadata.locks?.backendRequirementsLock, context.backendRequirementsLock)
    || !identitiesEqual(
      metadata.locks?.backendBuildRequirementsLock,
      context.backendBuildRequirementsLock
    )
    || !identitiesEqual(
      metadata.toolchainInputs?.pythonExecutable,
      context.pythonExecutable
    )
  ) {
    throw new Error('Release build metadata dependency-lock binding mismatch');
  }
  const requiredTools = [
    'node', 'npm', 'python', 'pip', 'pyinstaller', 'electronBuilder', 'electron',
    'osxSign', 'macOS', 'xcode'
  ];
  for (const name of requiredTools) {
    if (!String(metadata.tools?.[name] || '').trim()) {
      throw new Error(`Release build metadata is missing tool version: ${name}`);
    }
  }
  if (!legacyAllowed) {
    const exactIdentityPath = (identity, expectedPath, expectedBase) => Boolean(
      identity?.path === expectedPath && identity?.pathBase === expectedBase
    );
    if (
      !exactIdentityPath(
        metadata.locks?.frontendPackageLock,
        'frontend/package-lock.json',
        'source_checkout'
      )
      || !exactIdentityPath(
        metadata.locks?.backendRequirementsLock,
        'backend/requirements-macos-arm64.lock.txt',
        'source_checkout'
      )
      || !exactIdentityPath(
        metadata.locks?.backendBuildRequirementsLock,
        'backend/requirements-macos-arm64-build.lock.txt',
        'source_checkout'
      )
      || metadata.toolchainInputs?.pythonExecutable?.pathBase !== 'toolchain_runtime'
      || !/^cpython\/[^/]+\/python$/.test(
        metadata.toolchainInputs?.pythonExecutable?.path || ''
      )
      || !exactIdentityPath(
        metadata.buildOutputs?.backendServer,
        '.build/backend-dist/server/server',
        undefined
      )
      || !/^\d+(?:\.\d+)+(?:[a-z0-9.+-]*)?$/i.test(metadata.tools?.pip || '')
    ) {
      throw new Error('Release build metadata portable path contract mismatch');
    }
  }
  if (
    (context.backendServer
      && !identitiesEqual(metadata.buildOutputs?.backendServer, context.backendServer))
    || (context.backendBundleTreeSha256
      && metadata.buildOutputs?.backendBundleTreeSha256
        !== context.backendBundleTreeSha256)
    || (context.frontendTreeSha256
      && metadata.buildOutputs?.frontendTreeSha256 !== context.frontendTreeSha256)
    || !/^[0-9a-f]{64}$/.test(String(metadata.buildOutputs?.backendServer?.sha256 || ''))
    || !(metadata.buildOutputs?.backendServer?.size > 0)
    || !/^[0-9a-f]{64}$/.test(
      String(metadata.buildOutputs?.backendBundleTreeSha256 || '')
    )
    || !/^[0-9a-f]{64}$/.test(String(metadata.buildOutputs?.frontendTreeSha256 || ''))
  ) {
    throw new Error('Release build metadata isolated-output binding mismatch');
  }
}

export function assertReleaseBuildAttestation(attestation, context) {
  if (attestation?.schema !== 'automixer.macos_release_build_attestation.v1') {
    throw new Error('Signed release build attestation schema mismatch');
  }
  if (
    attestation.source?.head !== context.expectedHead
    || attestation.source?.tree !== context.expectedTree
    || attestation.buildOutputs?.backendSha256 !== context.backendSha256
    || attestation.buildOutputs?.backendBundleTreeSha256
      !== context.backendBundleTreeSha256
    || attestation.buildOutputs?.frontendTreeSha256 !== context.frontendTreeSha256
  ) {
    throw new Error('Signed release build attestation does not match isolated build outputs');
  }
}

export function assertInitialReleaseVerification(manifest, context) {
  const legacyAllowed = manifest?.pathPolicy == null && context.allowLegacy === true;
  const artifactRoot = path.resolve(
    context.artifactRoot || path.dirname(context.buildMetadata?.path || '')
  );
  const artifactIdentityMatches = (recorded, current, label) => {
    if (!identitiesEqual(recorded, current)) return false;
    try {
      const currentPath = path.isAbsolute(String(current?.path || ''))
        ? path.resolve(current.path)
        : path.resolve(artifactRoot, String(current?.path || ''));
      return resolveArtifactPath(artifactRoot, recorded?.path, label, {
        allowLegacyAbsoluteWithinRoot: legacyAllowed,
        mustExist: false
      }) === currentPath;
    } catch {
      return false;
    }
  };
  let recordedAppPath = null;
  try {
    recordedAppPath = resolveArtifactPath(
      artifactRoot,
      manifest?.app?.path,
      'Initial verification app',
      { allowLegacyAbsoluteWithinRoot: legacyAllowed, mustExist: false }
    );
  } catch {
    recordedAppPath = null;
  }
  if (
    manifest?.schema !== 'automixer.macos_release_manifest.v2'
    || (
      manifest.pathPolicy !== 'artifact_root_relative_and_logical_v1'
      && !legacyAllowed
    )
    || manifest.releaseEligible !== false
    || manifest.verification?.mode !== 'isolated_outputs'
    || manifest.verification?.complete !== false
    || manifest.verification?.eligibleAfterRetainedVerification !== true
    || manifest.verification?.isolatedOutputsPresent !== true
    || manifest.source?.head !== context.expectedHead
    || manifest.source?.tree !== context.expectedTree
    || !artifactIdentityMatches(
      manifest.build?.metadata,
      context.buildMetadata,
      'Initial verification build metadata'
    )
    || recordedAppPath !== path.resolve(context.appPath)
    || manifest.app?.treeSha256 !== context.appTreeSha256
    || !artifactIdentityMatches(
      manifest.app?.buildAttestation,
      context.buildAttestation,
      'Initial verification build attestation'
    )
    || !artifactIdentityMatches(
      manifest.artifacts?.dmg,
      context.dmg,
      'Initial verification DMG'
    )
    || !artifactIdentityMatches(
      manifest.artifacts?.zip,
      context.zip,
      'Initial verification ZIP'
    )
    || !artifactIdentityMatches(
      manifest.runtimeSmoke?.result,
      context.smokeResult,
      'Initial verification smoke result'
    )
    || !artifactIdentityMatches(
      manifest.notarization?.app?.evidence,
      context.appNotarizationEvidence,
      'Initial verification application notarization evidence'
    )
    || !artifactIdentityMatches(
      manifest.notarization?.app?.rawAppleLog,
      context.appNotaryLog,
      'Initial verification application Apple log'
    )
    || !artifactIdentityMatches(
      manifest.notarization?.dmg?.evidence,
      context.dmgNotarizationEvidence,
      'Initial verification DMG notarization evidence'
    )
    || !artifactIdentityMatches(
      manifest.notarization?.dmg?.rawAppleLog,
      context.dmgNotaryLog,
      'Initial verification DMG Apple log'
    )
  ) {
    throw new Error('Initial isolated-output release verification is invalid');
  }
}

function codesignDetails(targetPath) {
  return parseCodesignDetails(combined(run(
    '/usr/bin/codesign',
    ['-d', '--verbose=4', targetPath]
  )));
}

function readEntitlements(targetPath) {
  const result = run(
    '/usr/bin/codesign',
    ['-d', '--entitlements', ':-', targetPath],
    { check: false }
  );
  const text = combined(result);
  const start = Math.max(text.indexOf('<?xml'), text.indexOf('<plist'));
  const end = text.lastIndexOf('</plist>');
  if (start < 0 || end < start) return {};
  return plistJson(text.slice(start, end + '</plist>'.length));
}

function exactOne(paths, label) {
  if (paths.length !== 1) {
    throw new Error(`Expected exactly one ${label}, found ${paths.length}`);
  }
  return paths[0];
}

export function assertRegularContainedFile(artifactRoot, candidate, label) {
  const resolvedRoot = path.resolve(artifactRoot);
  const candidateText = String(candidate || '');
  const resolvedCandidate = path.isAbsolute(candidateText)
    ? path.resolve(candidateText)
    : path.resolve(resolvedRoot, candidateText);
  const relative = path.relative(resolvedRoot, resolvedCandidate);
  if (
    !relative
    || path.isAbsolute(relative)
    || relative === '..'
    || relative.startsWith(`..${path.sep}`)
  ) {
    throw new Error(`${label} escapes the release artifact directory`);
  }
  const stat = fs.lstatSync(resolvedCandidate);
  if (stat.isSymbolicLink() || !stat.isFile()) {
    throw new Error(`${label} must be a regular file, not a symlink or special file`);
  }
  const contained = resolveArtifactPath(
    artifactRoot,
    relative,
    label,
    { mustExist: true }
  );
  if (contained !== resolvedCandidate) {
    throw new Error(`${label} real path does not match its artifact path`);
  }
  return resolvedCandidate;
}

export function releasePublicationPolicy({
  allowLegacy = false,
  retainedArtifacts = false
} = {}) {
  if (allowLegacy && retainedArtifacts) {
    throw new Error('Legacy compatibility is inspect-only and cannot publish retained release evidence');
  }
  return {
    publishManifest: !allowLegacy,
    releaseEligible: !allowLegacy && retainedArtifacts,
    verificationMode: allowLegacy
      ? 'legacy_inspection'
      : retainedArtifacts
        ? 'retained_artifacts'
        : 'isolated_outputs'
  };
}

export function findTopLevelArtifact(directory, extension) {
  const candidate = exactOne(
    fs.readdirSync(directory)
      .filter(name => name.toLowerCase().endsWith(extension))
      .map(name => path.join(directory, name)),
    extension
  );
  return assertRegularContainedFile(directory, candidate, `Top-level ${extension} artifact`);
}

export function findReleaseArtifactInputs(
  artifactDirectory,
  { allowLegacy = false } = {}
) {
  const optionalV2Sidecars = [
    ['app-notarization.json', 'Application notarization evidence'],
    ['app-notary-log.json', 'Application retained Apple notary log'],
    ['dmg-notary-log.json', 'DMG retained Apple notary log']
  ];
  const sidecarPresence = optionalV2Sidecars.map(([name]) => Boolean(
    fs.lstatSync(path.join(artifactDirectory, name), { throwIfNoEntry: false })
  ));
  const legacySidecarsAbsent = allowLegacy && sidecarPresence.every(value => !value);
  if (allowLegacy && !legacySidecarsAbsent && sidecarPresence.some(value => !value)) {
    throw new Error('Legacy inspection refuses a partial v2 notarization sidecar set');
  }
  const optionalSidecar = (name, label) => {
    if (legacySidecarsAbsent) return null;
    const candidate = path.join(artifactDirectory, name);
    if (!fs.lstatSync(candidate, { throwIfNoEntry: false })) {
      throw new Error(`${label} is missing`);
    }
    return assertRegularContainedFile(artifactDirectory, candidate, label);
  };
  return {
    dmgPath: findTopLevelArtifact(artifactDirectory, '.dmg'),
    zipPath: findTopLevelArtifact(artifactDirectory, '.zip'),
    appNotarizationEvidencePath: optionalSidecar(
      'app-notarization.json',
      'Application notarization evidence'
    ),
    appNotaryLogPath: optionalSidecar(
      'app-notary-log.json',
      'Application retained Apple notary log'
    ),
    dmgNotarizationEvidencePath: assertRegularContainedFile(
      artifactDirectory,
      path.join(artifactDirectory, 'dmg-notarization.json'),
      'DMG notarization evidence'
    ),
    dmgNotaryLogPath: optionalSidecar(
      'dmg-notary-log.json',
      'DMG retained Apple notary log'
    )
  };
}

function findApp(directory) {
  return exactOne(
    fs.readdirSync(directory, { withFileTypes: true })
      .filter(entry => entry.isDirectory() && entry.name.endsWith('.app'))
      .map(entry => path.join(directory, entry.name)),
    'application bundle'
  );
}

function collectNestedBundles(frameworksRoot) {
  const bundles = [];
  function visit(directory) {
    for (const entry of fs.readdirSync(directory, { withFileTypes: true })) {
      const candidate = path.join(directory, entry.name);
      if (!entry.isDirectory() || entry.isSymbolicLink()) continue;
      if (entry.name.endsWith('.app') || entry.name.endsWith('.framework')) {
        bundles.push(candidate);
        if (entry.name.endsWith('.app')) continue;
      }
      visit(candidate);
    }
  }
  visit(frameworksRoot);
  return bundles;
}

function isMachOFile(filePath) {
  const descriptor = fs.openSync(filePath, 'r');
  try {
    const magic = Buffer.alloc(4);
    if (fs.readSync(descriptor, magic, 0, magic.length, 0) !== magic.length) return false;
    return new Set([
      0xfeedface,
      0xcefaedfe,
      0xfeedfacf,
      0xcffaedfe,
      0xcafebabe,
      0xbebafeca,
      0xcafebabf,
      0xbfbafeca
    ]).has(magic.readUInt32BE(0));
  } finally {
    fs.closeSync(descriptor);
  }
}

function collectMachOFiles(root) {
  const files = [];
  function visit(candidate) {
    const stat = fs.lstatSync(candidate);
    if (stat.isSymbolicLink()) return;
    if (stat.isDirectory()) {
      for (const name of fs.readdirSync(candidate).sort()) visit(path.join(candidate, name));
      return;
    }
    if (stat.isFile() && isMachOFile(candidate)) files.push(candidate);
  }
  visit(root);
  return files;
}

export function machOEntitlementProfile(relativePath, mainExecutableName) {
  const normalized = String(relativePath || '').split(path.sep).join('/');
  if (normalized === `Contents/MacOS/${mainExecutableName}`) return 'application';
  if (normalized === 'Contents/Resources/backend/server') return 'backend';
  if (normalized.startsWith('Contents/Resources/backend/')) return 'empty';
  if (/\.app\/Contents\/MacOS\//.test(normalized)) return 'inherited';
  if (/\.(?:dylib|node)$/.test(normalized)) return 'empty';
  const frameworkMain = normalized.match(
    /(?:^|\/)([^/]+)\.framework\/Versions\/[^/]+\/([^/]+)$/
  );
  if (frameworkMain && frameworkMain[1] === frameworkMain[2]) return 'empty';
  return 'inherited';
}

function assertArm64Only(executable, label) {
  const architectures = run('/usr/bin/lipo', ['-archs', executable]).stdout
    .trim()
    .split(/\s+/)
    .filter(Boolean);
  if (architectures.length !== 1 || architectures[0] !== 'arm64') {
    throw new Error(`${label} must be arm64-only`);
  }
}

function assertGatekeeper(commandArgs, label) {
  const output = combined(run('/usr/sbin/spctl', commandArgs));
  if (!/accepted/i.test(output) || !/source=Notarized Developer ID/i.test(output)) {
    throw new Error(`${label} is not accepted as Notarized Developer ID`);
  }
}

function parseArgs(argv) {
  const values = {};
  for (let index = 0; index < argv.length; index += 1) {
    const argument = argv[index];
    if (argument === '--artifact-dir') values.artifactDirectory = path.resolve(argv[++index] || '');
    else if (argument === '--team-id') values.teamId = String(argv[++index] || '').trim();
    else if (argument === '--not-before') values.notBefore = String(argv[++index] || '').trim();
    else if (argument === '--expected-head') values.expectedHead = String(argv[++index] || '').trim();
    else if (argument === '--expected-tree') values.expectedTree = String(argv[++index] || '').trim();
    else if (argument === '--smoke-result') values.smokeResultPath = path.resolve(argv[++index] || '');
    else if (argument === '--build-metadata') values.buildMetadataPath = path.resolve(argv[++index] || '');
    else if (argument === '--retained-artifacts') values.retainedArtifacts = true;
    else if (argument === '--allow-legacy') values.allowLegacy = true;
    else throw new Error(`Unknown argument: ${argument}`);
  }
  if (!values.artifactDirectory || !fs.existsSync(values.artifactDirectory)) {
    throw new Error('--artifact-dir must exist');
  }
  if (!/^[A-Z0-9]{10}$/.test(values.teamId || '')) {
    throw new Error('--team-id must be a 10-character Apple Team ID');
  }
  for (const [name, value] of [
    ['--expected-head', values.expectedHead],
    ['--expected-tree', values.expectedTree]
  ]) {
    if (!/^[0-9a-f]{40,64}$/.test(value || '')) throw new Error(`${name} must be a Git object ID`);
  }
  if (!values.smokeResultPath || !fs.existsSync(values.smokeResultPath)) {
    throw new Error('--smoke-result must exist');
  }
  if (!values.buildMetadataPath || !fs.existsSync(values.buildMetadataPath)) {
    throw new Error('--build-metadata must exist');
  }
  const notBeforeMs = new Date(values.notBefore).getTime();
  if (!Number.isFinite(notBeforeMs)) throw new Error('--not-before must be an ISO timestamp');
  return values;
}

function assertWithinArtifactDirectory(candidate, artifactDirectory, label) {
  const relative = path.relative(fs.realpathSync(artifactDirectory), fs.realpathSync(candidate));
  if (relative.startsWith('..') || path.isAbsolute(relative)) {
    throw new Error(`${label} must be inside the release artifact directory`);
  }
}

function sourceSnapshot() {
  return {
    head: run('/usr/bin/git', ['rev-parse', 'HEAD'], { cwd: repoRoot }).stdout.trim(),
    tree: run('/usr/bin/git', ['rev-parse', 'HEAD^{tree}'], { cwd: repoRoot }).stdout.trim(),
    status: run('/usr/bin/git', ['status', '--porcelain=v1', '-uall'], { cwd: repoRoot }).stdout.trim()
  };
}

function copyStableApplicationSnapshot(sourceAppPath, tempRoot) {
  const sourceStatBefore = fs.lstatSync(sourceAppPath, { bigint: true });
  if (sourceStatBefore.isSymbolicLink() || !sourceStatBefore.isDirectory()) {
    throw new Error('Application snapshot source must be a real directory');
  }
  const sourceTreeBefore = hashTree(sourceAppPath);
  const snapshotPath = path.join(tempRoot, path.basename(sourceAppPath));
  fs.cpSync(sourceAppPath, snapshotPath, {
    recursive: true,
    preserveTimestamps: true,
    verbatimSymlinks: true,
    errorOnExist: true,
    force: false
  });
  const snapshotStat = fs.lstatSync(snapshotPath, { bigint: true });
  const sourceStatAfter = fs.lstatSync(sourceAppPath, { bigint: true });
  const sourceTreeAfter = hashTree(sourceAppPath);
  const snapshotTree = hashTree(snapshotPath);
  if (
    snapshotStat.isSymbolicLink()
    || !snapshotStat.isDirectory()
    || sourceStatBefore.dev !== sourceStatAfter.dev
    || sourceStatBefore.ino !== sourceStatAfter.ino
    || sourceStatBefore.mode !== sourceStatAfter.mode
    || sourceTreeBefore !== sourceTreeAfter
    || snapshotTree !== sourceTreeBefore
  ) {
    throw new Error('Application changed while creating its verification snapshot');
  }
  return { path: snapshotPath, treeSha256: snapshotTree };
}

function extractZip(zipPath, tempRoot) {
  const destination = path.join(tempRoot, 'zip');
  fs.mkdirSync(destination);
  run('/usr/bin/ditto', ['-x', '-k', zipPath, destination]);
  return findApp(destination);
}

export function assertMatchingFileContents(sourcePath, packagedPath, label) {
  if (!fs.existsSync(packagedPath)) throw new Error(`${label} is missing from the packaged app`);
  const source = fileIdentity(sourcePath);
  const packaged = fileIdentity(packagedPath);
  if (!identitiesEqual(source, packaged)) {
    throw new Error(`${label} does not match the clean source input`);
  }
  return { source, packaged };
}

function verifyPackagedInputs(appPath, tempRoot, expectedTreeSha256, artifactRoot) {
  const archivePath = path.join(appPath, 'Contents', 'Resources', 'app.asar');
  const destination = path.join(tempRoot, 'asar');
  fs.mkdirSync(destination);
  extractAsar(archivePath, destination);
  const packagedBuild = path.join(destination, 'build');
  if (!fs.existsSync(packagedBuild)) {
    throw new Error('Packaged frontend build is missing from app.asar');
  }
  const actual = hashTree(packagedBuild);
  if (actual !== expectedTreeSha256) {
    throw new Error('Packaged frontend does not match the isolated Vite build output');
  }
  const runtime = Object.fromEntries(RUNTIME_SOURCE_FILES.map(name => {
    const sourcePath = path.join(frontendRoot, 'public', name);
    const packagedPath = path.join(destination, 'public', name);
    assertMatchingFileContents(sourcePath, packagedPath, `Runtime source ${name}`);
    return [name, {
      source: logicalFileIdentity(
        sourcePath,
        `frontend/public/${name}`,
        'source_checkout',
        `Runtime source ${name}`
      ),
      packaged: logicalFileIdentity(
        packagedPath,
        `public/${name}`,
        'packaged_app_asar',
        `Packaged runtime source ${name}`
      )
    }];
  }));
  const config = Object.fromEntries(CONFIG_SOURCE_FILES.map(name => {
    const sourcePath = path.join(repoRoot, 'config', name);
    const packagedPath = path.join(appPath, 'Contents', 'Resources', 'config', name);
    assertMatchingFileContents(sourcePath, packagedPath, `Runtime config ${name}`);
    return [name, {
      source: logicalFileIdentity(
        sourcePath,
        `config/${name}`,
        'source_checkout',
        `Runtime config source ${name}`
      ),
      packaged: artifactFileIdentity(
        artifactRoot,
        packagedPath,
        `Packaged runtime config ${name}`
      )
    }];
  }));
  return { frontendTreeSha256: actual, runtime, config };
}

export function withMountedDmg(
  dmgPath,
  tempRoot,
  callback,
  dependencies = {}
) {
  const runCommand = dependencies.runCommand || run;
  const parseAttachPlist = dependencies.parsePlist || plistJson;
  const findMountedApp = dependencies.findMountedApp || findApp;
  const mountRoot = path.join(tempRoot, 'dmg-mount');
  fs.mkdirSync(mountRoot);
  let attached = false;
  let value;
  let primaryError;
  let cleanupError;
  let detachTarget = mountRoot;
  try {
    const attach = runCommand('/usr/bin/hdiutil', [
      'attach', '-readonly', '-nobrowse', '-mountpoint', mountRoot, '-plist', dmgPath
    ]);
    attached = true;
    const attachInfo = parseAttachPlist(attach.stdout);
    detachTarget = attachInfo?.['system-entities']?.find(row => (
      row?.['mount-point'] === mountRoot && String(row?.['dev-entry'] || '').startsWith('/dev/')
    ))?.['dev-entry'] || mountRoot;
    value = callback(findMountedApp(mountRoot));
  } catch (error) {
    primaryError = error;
  } finally {
    if (attached) {
      try {
        const attempts = [
          ['detach', detachTarget],
          ...Array.from({ length: 5 }, () => ['detach', '-force', detachTarget])
        ];
        let detached = false;
        let lastResult;
        for (const args of attempts) {
          lastResult = runCommand('/usr/bin/hdiutil', args, { check: false });
          if (lastResult?.status === 0) {
            detached = true;
            break;
          }
        }
        if (!detached) {
          cleanupError = new Error(
            `Failed to detach mounted DMG ${detachTarget}: `
            + `${lastResult?.stderr || lastResult?.stdout || 'unknown hdiutil error'}`.trim()
          );
        }
      } catch (error) {
        cleanupError = error;
      }
    }
  }
  if (primaryError && cleanupError) {
    throw new AggregateError([primaryError, cleanupError], 'DMG verification and cleanup both failed');
  }
  if (primaryError) throw primaryError;
  if (cleanupError) throw cleanupError;
  return value;
}

function verifyApplication(appPath, teamId, label) {
  const infoPlist = readPlist(path.join(appPath, 'Contents', 'Info.plist'));
  if (infoPlist.CFBundleIdentifier !== 'com.automixer.wing') {
    throw new Error(`${label} bundle identifier mismatch`);
  }
  const ats = infoPlist.NSAppTransportSecurity || {};
  if (ats.NSAllowsArbitraryLoads !== false || ats.NSAllowsLocalNetworking !== true) {
    throw new Error(`${label} App Transport Security policy is not release-safe`);
  }

  const electronExecutable = path.join(
    appPath, 'Contents', 'MacOS', infoPlist.CFBundleExecutable
  );
  const backendExecutable = path.join(
    appPath, 'Contents', 'Resources', 'backend', 'server'
  );
  const appMachOFiles = collectMachOFiles(appPath);
  if (appMachOFiles.length < 3) {
    throw new Error(`${label} application has no nested Mach-O runtime`);
  }
  for (const codePath of appMachOFiles) {
    assertArm64Only(
      codePath,
      `${label} ${path.relative(appPath, codePath)}`
    );
  }
  const backendMachOFiles = collectMachOFiles(
    path.join(appPath, 'Contents', 'Resources', 'backend')
  );
  if (backendMachOFiles.length < 2) {
    throw new Error(`${label} backend onedir bundle has no nested Mach-O runtime`);
  }
  for (const backendMachO of backendMachOFiles) {
    assertArm64Only(
      backendMachO,
      `${label} backend ${path.relative(path.dirname(backendExecutable), backendMachO)}`
    );
  }
  run('/usr/bin/codesign', ['--verify', '--deep', '--strict', '--verbose=4', appPath]);
  run('/usr/bin/codesign', ['--verify', '--strict', '--verbose=4', backendExecutable]);

  const applicationEntitlements = readPlist(path.join(
    frontendRoot, 'public', 'entitlements.mac.plist'
  ));
  const inheritedEntitlements = readPlist(path.join(
    frontendRoot, 'public', 'entitlements.mac.inherit.plist'
  ));
  const signedComponents = [
    { label: 'application', path: appPath, expectedEntitlements: applicationEntitlements },
    { label: 'backend', path: backendExecutable, expectedEntitlements: {
      [BACKEND_ENTITLEMENT]: true
    } },
    ...collectNestedBundles(path.join(appPath, 'Contents', 'Frameworks')).map(bundle => ({
      label: path.relative(appPath, bundle),
      path: bundle,
      expectedEntitlements: bundle.endsWith('.app') ? inheritedEntitlements : {}
    })),
    ...appMachOFiles.map(codePath => {
      const relativePath = path.relative(appPath, codePath);
      const profile = machOEntitlementProfile(relativePath, infoPlist.CFBundleExecutable);
      const profiles = {
        application: applicationEntitlements,
        backend: { [BACKEND_ENTITLEMENT]: true },
        inherited: inheritedEntitlements,
        empty: {}
      };
      return {
        label: `code:${relativePath}`,
        path: codePath,
        expectedEntitlements: profiles[profile]
      };
    })
  ];
  const signatures = {};
  for (const component of signedComponents) {
    const details = codesignDetails(component.path);
    assertDeveloperIdSignature(details, { label: `${label} ${component.label}`, teamId });
    signatures[component.label] = details;
  }

  const backendEntitlements = readEntitlements(backendExecutable);
  assertBackendEntitlements(backendEntitlements);
  for (const component of signedComponents) {
    const actualEntitlements = readEntitlements(component.path);
    if (component.label === 'backend' || component.label === 'code:Contents/Resources/backend/server') {
      assertBackendEntitlements(actualEntitlements);
    } else {
      assertExactEntitlements(
        actualEntitlements,
        component.expectedEntitlements,
        `${label} ${component.label}`
      );
    }
  }
  const attestationPath = path.join(
    appPath, 'Contents', 'Resources', 'release-build-attestation.json'
  );
  if (!fs.existsSync(attestationPath)) {
    throw new Error(`${label} signed release build attestation is missing`);
  }
  const stableAttestation = readStableJsonFileNoFollow(
    attestationPath,
    `${label} signed release build attestation`,
    { containmentRoot: appPath }
  );
  return {
    infoPlist,
    signatures,
    backendEntitlements,
    treeSha256: hashTree(appPath),
    buildAttestation: stableAttestation.value,
    buildAttestationIdentity: {
      path: attestationPath,
      ...stableAttestation.identity
    }
  };
}

function loadSmokeEvidence({
  smokeResultPath,
  appPath,
  expectedHead,
  expectedTree,
  appTreeSha256,
  backendBundleTreeSha256,
  buildProfilePath,
  allowLegacy = false
}) {
  const artifactRoot = path.dirname(path.resolve(buildProfilePath));
  const stableBuildProfile = readStableJsonFileNoFollow(
    buildProfilePath,
    'Packaged runtime build profile'
  );
  const buildProfileIdentity = {
    path: path.resolve(buildProfilePath),
    ...stableBuildProfile.identity
  };

  const smokeDirectory = path.dirname(smokeResultPath);
  const manifestPath = path.join(smokeDirectory, 'artifact-manifest.json');
  const markdownPath = path.join(smokeDirectory, 'result.md');
  if (!fs.existsSync(manifestPath) || !fs.existsSync(markdownPath)) {
    throw new Error('Packaged runtime smoke artifact is incomplete');
  }
  const detached = verifyPackagedRuntimeSmokeManifest(manifestPath, { allowLegacy });
  const result = detached.result;
  const smokeResultIdentity = {
    path: path.resolve(smokeResultPath),
    ...detached.identities['result.json']
  };
  assertPassingSmokeEvidence(result, {
    artifactRoot,
    appPath,
    expectedHead,
    expectedTree,
    appTreeSha256,
    backendBundleTreeSha256,
    buildProfile: buildProfileIdentity,
    allowLegacy
  });
  const artifactManifest = detached.manifest;
  const resolvedBuildProfilePath = resolveArtifactPath(
    artifactRoot,
    artifactManifest.buildProfile?.path,
    'Packaged runtime smoke manifest build profile',
    {
      allowLegacyAbsoluteWithinRoot: artifactManifest.pathPolicy == null && allowLegacy,
      mustExist: true
    }
  );
  if (
    artifactManifest.schema !== 'automixer.packaged_runtime_smoke_manifest.v1'
    || artifactManifest.readOnly !== true
    || artifactManifest.contactsWing !== false
    || artifactManifest.liveAuthorityGranted !== false
    || artifactManifest.buildProfile?.profile !== 'release'
    || artifactManifest.buildProfile?.releaseEligible !== false
    || resolvedBuildProfilePath !== path.resolve(buildProfilePath)
    || artifactManifest.buildProfile?.path !== result.buildProfile.path
    || artifactManifest.buildProfile?.sha256 !== result.buildProfile.sha256
    || artifactManifest.buildProfile?.bindingValid !== true
    || artifactManifest.buildProfile?.appTreeMatches !== true
    || !identitiesEqual(detached.identities['result.json'], smokeResultIdentity)
  ) {
    throw new Error('Packaged runtime smoke manifest binding mismatch');
  }

  const packageFiles = {
    electron: path.join(appPath, 'Contents', 'MacOS', 'AUTO MIXER Tubeslave'),
    backend: path.join(appPath, 'Contents', 'Resources', 'backend', 'server'),
    asar: path.join(appPath, 'Contents', 'Resources', 'app.asar'),
    infoPlist: path.join(appPath, 'Contents', 'Info.plist'),
    runtimeSourceManifest: path.join(
      appPath,
      'Contents',
      'Resources',
      'backend',
      '_internal',
      'automixer-runtime-source-manifest.json'
    )
  };
  for (const [name, filePath] of Object.entries(packageFiles)) {
    const current = fileIdentity(filePath);
    const beforeRecordedPath = resolveArtifactPath(
      artifactRoot,
      result.before?.packageFiles?.[name]?.path,
      `Packaged runtime smoke before file ${name}`,
      {
        allowLegacyAbsoluteWithinRoot: result.pathPolicy == null && allowLegacy,
        mustExist: true
      }
    );
    const afterRecordedPath = resolveArtifactPath(
      artifactRoot,
      result.after?.packageFiles?.[name]?.path,
      `Packaged runtime smoke after file ${name}`,
      {
        allowLegacyAbsoluteWithinRoot: result.pathPolicy == null && allowLegacy,
        mustExist: true
      }
    );
    if (
      beforeRecordedPath !== path.resolve(filePath)
      || afterRecordedPath !== path.resolve(filePath)
      || !identitiesEqual(result.before?.packageFiles?.[name], current)
      || !identitiesEqual(result.after?.packageFiles?.[name], current)
    ) {
      throw new Error(`Packaged runtime smoke file binding mismatch: ${name}`);
    }
  }
  return {
    result,
    resultIdentity: smokeResultIdentity,
    manifest: {
      path: path.resolve(manifestPath),
      ...detached.manifestIdentity
    },
    markdown: {
      path: path.resolve(markdownPath),
      ...detached.identities['result.md']
    }
  };
}

export function createPortableReleaseManifest({
  artifactDirectory,
  retainedArtifacts,
  isolatedOutputsPresent,
  initialVerificationPath = null,
  expectedHead,
  expectedTree,
  buildMetadataPath,
  buildMetadata,
  appPath,
  appBefore,
  zipVerification,
  dmgVerification,
  dmgPath,
  zipPath,
  dmgSignature,
  appNotarizationEvidence,
  appNotarizationEvidencePath,
  appNotaryLogPath,
  dmgNotarizationEvidence,
  dmgNotarizationEvidencePath,
  dmgNotaryLogPath,
  smokeResultPath,
  verifiedIdentities = null
}) {
  const identities = verifiedIdentities || {
    initialVerification: initialVerificationPath
      ? artifactFileIdentity(
        artifactDirectory,
        initialVerificationPath,
        'Initial isolated release verification'
      )
      : null,
    buildMetadata: artifactFileIdentity(
      artifactDirectory,
      buildMetadataPath,
      'Release build metadata'
    ),
    buildAttestation: artifactFileIdentity(
      artifactDirectory,
      path.join(appPath, 'Contents', 'Resources', 'release-build-attestation.json'),
      'Signed release build attestation'
    ),
    dmg: artifactFileIdentity(artifactDirectory, dmgPath, 'Release DMG'),
    zip: artifactFileIdentity(artifactDirectory, zipPath, 'Release ZIP'),
    appNotarizationEvidence: artifactFileIdentity(
      artifactDirectory,
      appNotarizationEvidencePath,
      'Application notarization evidence'
    ),
    appNotaryLog: artifactFileIdentity(
      artifactDirectory,
      appNotaryLogPath,
      'Application retained Apple notary log'
    ),
    dmgNotarizationEvidence: artifactFileIdentity(
      artifactDirectory,
      dmgNotarizationEvidencePath,
      'DMG notarization evidence'
    ),
    dmgNotaryLog: artifactFileIdentity(
      artifactDirectory,
      dmgNotaryLogPath,
      'DMG retained Apple notary log'
    ),
    smokeResult: artifactFileIdentity(
      artifactDirectory,
      smokeResultPath,
      'Packaged runtime smoke result'
    ),
    smokeManifest: artifactFileIdentity(
      artifactDirectory,
      path.join(path.dirname(smokeResultPath), 'artifact-manifest.json'),
      'Packaged runtime smoke manifest'
    ),
    smokeMarkdown: artifactFileIdentity(
      artifactDirectory,
      path.join(path.dirname(smokeResultPath), 'result.md'),
      'Packaged runtime smoke Markdown'
    )
  };
  return {
    schema: 'automixer.macos_release_manifest.v2',
    pathPolicy: 'artifact_root_relative_and_logical_v1',
    generatedAt: new Date().toISOString(),
    profile: 'release',
    releaseEligible: retainedArtifacts === true,
    verification: {
      mode: retainedArtifacts ? 'retained_artifacts' : 'isolated_outputs',
      complete: retainedArtifacts === true,
      eligibleAfterRetainedVerification: retainedArtifacts !== true,
      isolatedOutputsPresent,
      initialIsolatedVerification: initialVerificationPath
        ? identities.initialVerification
        : null
    },
    source: { head: expectedHead, tree: expectedTree, cleanCheckout: true },
    build: {
      metadata: identities.buildMetadata,
      locks: buildMetadata.locks,
      tools: buildMetadata.tools,
      toolchainInputs: buildMetadata.toolchainInputs
    },
    app: {
      path: artifactRelativePath(artifactDirectory, appPath, 'Release application'),
      bundleId: appBefore.infoPlist.CFBundleIdentifier,
      treeSha256: appBefore.treeSha256,
      architecture: 'arm64',
      buildAttestation: identities.buildAttestation
    },
    packagedInputs: appBefore.packagedInputs,
    archives: {
      zipAppTreeSha256: zipVerification.treeSha256,
      dmgAppTreeSha256: dmgVerification.treeSha256
    },
    artifacts: {
      dmg: identities.dmg,
      zip: identities.zip
    },
    signatures: appBefore.signatures,
    dmgSignature,
    backendEntitlements: appBefore.backendEntitlements,
    notarization: {
      appStapled: true,
      dmgStapled: true,
      app: {
        submissionId: appNotarizationEvidence.submission.id,
        evidence: identities.appNotarizationEvidence,
        rawAppleLog: identities.appNotaryLog,
        ticket: appNotarizationEvidence.ticket
      },
      dmg: {
        submissionId: dmgNotarizationEvidence.submission.id,
        evidence: identities.dmgNotarizationEvidence,
        rawAppleLog: identities.dmgNotaryLog,
        ticket: dmgNotarizationEvidence.ticket
      },
      gatekeeperSource: 'Notarized Developer ID'
    },
    runtimeSmoke: {
      result: identities.smokeResult,
      manifest: identities.smokeManifest,
      markdown: identities.smokeMarkdown,
      readOnly: true,
      contactsWing: false,
      liveAuthorityGranted: false,
      pass: true
    }
  };
}

function main() {
  if (process.platform !== 'darwin' || process.arch !== 'arm64') {
    throw new Error('Release verification requires macOS arm64');
  }
  const options = parseArgs(process.argv.slice(2));
  const {
    artifactDirectory,
    teamId,
    notBefore,
    expectedHead,
    expectedTree,
    smokeResultPath,
    buildMetadataPath,
    allowLegacy = false
  } = options;
  const publicationPolicy = releasePublicationPolicy({
    allowLegacy,
    retainedArtifacts: options.retainedArtifacts === true
  });
  const releaseArtifactInputs = findReleaseArtifactInputs(
    artifactDirectory,
    { allowLegacy }
  );
  assertWithinArtifactDirectory(smokeResultPath, artifactDirectory, 'Smoke result');
  assertWithinArtifactDirectory(buildMetadataPath, artifactDirectory, 'Build metadata');
  const expectedSource = { head: expectedHead, tree: expectedTree };
  assertSourceBinding(expectedSource, sourceSnapshot(), 'verification start');

  const packageLock = fileIdentity(path.join(frontendRoot, 'package-lock.json'));
  const requirementsLock = fileIdentity(path.join(
    repoRoot, 'backend', 'requirements-macos-arm64.lock.txt'
  ));
  const buildRequirementsLock = fileIdentity(path.join(
    repoRoot, 'backend', 'requirements-macos-arm64-build.lock.txt'
  ));
  const pythonExecutablePath = String(process.env.AUTOMIXER_PYTHON_BIN || '').trim();
  if (!path.isAbsolute(pythonExecutablePath) || !fs.existsSync(pythonExecutablePath)) {
    throw new Error('Release verification requires the exact Python executable path');
  }
  const pythonExecutable = fileIdentity(pythonExecutablePath);
  const stableBuildMetadata = readStableJsonFileNoFollow(
    buildMetadataPath,
    'Release build metadata'
  );
  const buildMetadataIdentity = artifactFileIdentity(
    artifactDirectory,
    buildMetadataPath,
    'Release build metadata'
  );
  assertIdentityUnchanged(
    stableBuildMetadata.identity,
    buildMetadataIdentity,
    'build metadata stable read'
  );
  const buildMetadata = stableBuildMetadata.value;
  const expectedBackendBundle = path.join(
    artifactDirectory, '.build', 'backend-dist', 'server'
  );
  const expectedBackendServer = path.join(expectedBackendBundle, 'server');
  const expectedFrontendBuild = path.join(
    artifactDirectory, '.build', 'frontend-build'
  );
  const backendOutputPresent = fs.existsSync(expectedBackendServer);
  const frontendOutputPresent = fs.existsSync(expectedFrontendBuild);
  if (backendOutputPresent !== frontendOutputPresent) {
    throw new Error('Isolated release build outputs are only partially retained');
  }
  const isolatedOutputsPresent = backendOutputPresent && frontendOutputPresent;
  if (!isolatedOutputsPresent && !options.retainedArtifacts) {
    throw new Error('Isolated release build outputs are missing');
  }
  assertBuildMetadata(buildMetadata, {
    expectedHead,
    expectedTree,
    frontendPackageLock: packageLock,
    backendRequirementsLock: requirementsLock,
    backendBuildRequirementsLock: buildRequirementsLock,
    pythonExecutable,
    backendServer: isolatedOutputsPresent ? fileIdentity(expectedBackendServer) : undefined,
    backendBundleTreeSha256: isolatedOutputsPresent
      ? hashTree(expectedBackendBundle)
      : undefined,
    frontendTreeSha256: isolatedOutputsPresent ? hashTree(expectedFrontendBuild) : undefined,
    artifactRoot: artifactDirectory,
    allowLegacy
  });

  const macDirectory = path.join(artifactDirectory, 'mac-arm64');
  const appPath = findApp(macDirectory);
  const {
    dmgPath,
    zipPath,
    appNotarizationEvidencePath,
    appNotaryLogPath,
    dmgNotarizationEvidencePath,
    dmgNotaryLogPath
  } = releaseArtifactInputs;
  const dmgBefore = artifactFileIdentity(artifactDirectory, dmgPath, 'Release DMG');
  const zipBefore = artifactFileIdentity(artifactDirectory, zipPath, 'Release ZIP');
  assertFreshArtifact(fs.statSync(dmgPath), notBefore, 'DMG');
  assertFreshArtifact(fs.statSync(zipPath), notBefore, 'ZIP');
  assertFreshArtifact(fs.statSync(smokeResultPath), notBefore, 'runtime smoke');

  const initialVerificationPath = path.join(
    artifactDirectory, 'release-verification-initial.json'
  );
  const tempRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'automixer-release-verify-'));
  fs.chmodSync(tempRoot, 0o700);
  let appBefore;
  let smokeEvidence;
  let dmgSignature;
  let appNotarizationEvidenceIdentity;
  let appNotaryLogIdentity;
  let appNotarizationEvidence;
  let dmgNotarizationEvidenceIdentity;
  let dmgNotaryLogIdentity;
  let dmgNotarizationEvidence;
  let initialVerificationIdentity;
  let zipVerification;
  let dmgVerification;
  try {
    const appSnapshot = copyStableApplicationSnapshot(appPath, tempRoot);
    const dmgSnapshotPath = path.join(tempRoot, 'release-verification.dmg');
    const zipSnapshotPath = path.join(tempRoot, 'release-verification.zip');
    const dmgSnapshotIdentity = copyStableRegularFileNoFollow(
      dmgPath,
      dmgSnapshotPath,
      'Private release DMG verification snapshot'
    );
    const zipSnapshotIdentity = copyStableRegularFileNoFollow(
      zipPath,
      zipSnapshotPath,
      'Private release ZIP verification snapshot'
    );
    assertIdentityUnchanged(dmgBefore, dmgSnapshotIdentity, 'DMG snapshot binding');
    assertIdentityUnchanged(zipBefore, zipSnapshotIdentity, 'ZIP snapshot binding');

    const verifiedAppSnapshot = verifyApplication(
      appSnapshot.path,
      teamId,
      'application snapshot'
    );
    if (verifiedAppSnapshot.treeSha256 !== appSnapshot.treeSha256) {
      throw new Error('Application verification snapshot changed during signature checks');
    }
    const originalBuildAttestationPath = path.join(
      appPath,
      'Contents',
      'Resources',
      'release-build-attestation.json'
    );
    const stableOriginalBuildAttestation = readStableJsonFileNoFollow(
      originalBuildAttestationPath,
      'Original signed release build attestation',
      { containmentRoot: artifactDirectory }
    );
    const originalBuildAttestationArtifactIdentity = artifactFileIdentity(
      artifactDirectory,
      originalBuildAttestationPath,
      'Original signed release build attestation'
    );
    assertIdentityUnchanged(
      stableOriginalBuildAttestation.identity,
      originalBuildAttestationArtifactIdentity,
      'original signed release build attestation stable read'
    );
    assertIdentityUnchanged(
      verifiedAppSnapshot.buildAttestationIdentity,
      originalBuildAttestationArtifactIdentity,
      'application snapshot build attestation'
    );
    const originalBuildAttestationIdentity = {
      path: originalBuildAttestationPath,
      size: originalBuildAttestationArtifactIdentity.size,
      sha256: originalBuildAttestationArtifactIdentity.sha256
    };
    appBefore = {
      ...verifiedAppSnapshot,
      buildAttestation: stableOriginalBuildAttestation.value,
      buildAttestationIdentity: originalBuildAttestationIdentity
    };
    assertReleaseBuildAttestation(appBefore.buildAttestation, {
      expectedHead,
      expectedTree,
      backendSha256: buildMetadata.buildOutputs.backendServer.sha256,
      backendBundleTreeSha256: buildMetadata.buildOutputs.backendBundleTreeSha256,
      frontendTreeSha256: buildMetadata.buildOutputs.frontendTreeSha256
    });
    run('/usr/bin/xcrun', ['stapler', 'validate', appSnapshot.path]);
    if (hashTree(appSnapshot.path) !== appBefore.treeSha256) {
      throw new Error('Application snapshot changed during stapler validation');
    }
    if (appNotaryLogPath && appNotarizationEvidencePath) {
      const stableAppNotaryLog = readStableJsonFileNoFollow(
        appNotaryLogPath,
        'Application retained Apple notary log'
      );
      appNotaryLogIdentity = artifactFileIdentity(
        artifactDirectory,
        appNotaryLogPath,
        'Application retained Apple notary log'
      );
      assertIdentityUnchanged(
        stableAppNotaryLog.identity,
        appNotaryLogIdentity,
        'application retained Apple notary log stable read'
      );
      const stableAppNotarizationEvidence = readStableJsonFileNoFollow(
        appNotarizationEvidencePath,
        'Application notarization evidence'
      );
      appNotarizationEvidenceIdentity = artifactFileIdentity(
        artifactDirectory,
        appNotarizationEvidencePath,
        'Application notarization evidence'
      );
      assertIdentityUnchanged(
        stableAppNotarizationEvidence.identity,
        appNotarizationEvidenceIdentity,
        'application notarization evidence stable read'
      );
      appNotarizationEvidence = stableAppNotarizationEvidence.value;
      assertAppNotarizationEvidence(appNotarizationEvidence, {
        artifactRoot: artifactDirectory,
        appPath,
        appTreeSha256: appBefore.treeSha256,
        executableName: appBefore.infoPlist.CFBundleExecutable,
        signatures: appBefore.signatures,
        rawLog: stableAppNotaryLog.value,
        rawLogIdentity: appNotaryLogIdentity
      });
    } else if (!allowLegacy) {
      throw new Error('Application notarization v2 evidence is required');
    }
    smokeEvidence = loadSmokeEvidence({
      smokeResultPath,
      appPath,
      expectedHead,
      expectedTree,
      appTreeSha256: appBefore.treeSha256,
      backendBundleTreeSha256: hashTree(path.join(
        appPath, 'Contents', 'Resources', 'backend'
      )),
      buildProfilePath: path.join(artifactDirectory, 'build-profile.json'),
      allowLegacy
    });

    dmgSignature = codesignDetails(dmgSnapshotPath);
    assertDeveloperIdSignature(dmgSignature, {
      label: 'DMG', teamId, requireRuntime: false
    });
    run('/usr/bin/xcrun', ['stapler', 'validate', dmgSnapshotPath]);
    assertIdentityUnchanged(
      dmgSnapshotIdentity,
      artifactFileIdentity(tempRoot, dmgSnapshotPath, 'Post-stapler DMG snapshot'),
      'DMG verification snapshot'
    );
    assertGatekeeper([
      '--assess', '--type', 'execute', '--verbose=4', appSnapshot.path
    ], 'Application');
    if (hashTree(appSnapshot.path) !== appBefore.treeSha256) {
      throw new Error('Application snapshot changed during Gatekeeper assessment');
    }
    assertGatekeeper([
      '--assess', '--type', 'open', '--context', 'context:primary-signature',
      '--verbose=4', dmgSnapshotPath
    ], 'DMG');
    assertIdentityUnchanged(
      dmgSnapshotIdentity,
      artifactFileIdentity(tempRoot, dmgSnapshotPath, 'Post-Gatekeeper DMG snapshot'),
      'DMG verification snapshot'
    );

    let stableDmgNotaryLog = null;
    if (dmgNotaryLogPath) {
      stableDmgNotaryLog = readStableJsonFileNoFollow(
        dmgNotaryLogPath,
        'DMG retained Apple notary log'
      );
      dmgNotaryLogIdentity = artifactFileIdentity(
        artifactDirectory,
        dmgNotaryLogPath,
        'DMG retained Apple notary log'
      );
      assertIdentityUnchanged(
        stableDmgNotaryLog.identity,
        dmgNotaryLogIdentity,
        'DMG retained Apple notary log stable read'
      );
    }
    const stableDmgNotarizationEvidence = readStableJsonFileNoFollow(
      dmgNotarizationEvidencePath,
      'DMG notarization evidence'
    );
    dmgNotarizationEvidenceIdentity = artifactFileIdentity(
      artifactDirectory,
      dmgNotarizationEvidencePath,
      'DMG notarization evidence'
    );
    assertIdentityUnchanged(
      stableDmgNotarizationEvidence.identity,
      dmgNotarizationEvidenceIdentity,
      'DMG notarization evidence stable read'
    );
    dmgNotarizationEvidence = stableDmgNotarizationEvidence.value;
    assertDmgNotarizationEvidence(dmgNotarizationEvidence, dmgBefore, {
      artifactRoot: artifactDirectory,
      rawLog: stableDmgNotaryLog?.value,
      rawLogIdentity: dmgNotaryLogIdentity,
      dmgCdhash: dmgSignature.cdhash,
      allowLegacy
    });

    if (options.retainedArtifacts) {
      if (!fs.existsSync(initialVerificationPath)) {
        throw new Error('Initial isolated-output release verification is missing');
      }
      const stableInitialVerification = readStableJsonFileNoFollow(
        initialVerificationPath,
        'Initial isolated-output release verification'
      );
      initialVerificationIdentity = artifactFileIdentity(
        artifactDirectory,
        initialVerificationPath,
        'Initial isolated-output release verification'
      );
      assertIdentityUnchanged(
        stableInitialVerification.identity,
        initialVerificationIdentity,
        'initial isolated-output release verification stable read'
      );
      const initialVerification = stableInitialVerification.value;
      assertInitialReleaseVerification(initialVerification, {
        expectedHead,
        expectedTree,
        buildMetadata: buildMetadataIdentity,
        appPath,
        appTreeSha256: appBefore.treeSha256,
        buildAttestation: appBefore.buildAttestationIdentity,
        dmg: dmgBefore,
        zip: zipBefore,
        appNotarizationEvidence: appNotarizationEvidenceIdentity,
        appNotaryLog: appNotaryLogIdentity,
        dmgNotarizationEvidence: dmgNotarizationEvidenceIdentity,
        dmgNotaryLog: dmgNotaryLogIdentity,
        smokeResult: smokeEvidence.resultIdentity,
        artifactRoot: artifactDirectory,
        allowLegacy
      });
    }

    const packagedInputs = verifyPackagedInputs(
      appPath,
      tempRoot,
      buildMetadata.buildOutputs.frontendTreeSha256,
      artifactDirectory
    );
    const zipAppPath = extractZip(zipSnapshotPath, tempRoot);
    zipVerification = verifyApplication(zipAppPath, teamId, 'ZIP application');
    run('/usr/bin/xcrun', ['stapler', 'validate', zipAppPath]);
    if (zipVerification.treeSha256 !== appBefore.treeSha256) {
      throw new Error('ZIP does not contain the verified mode-preserving application tree');
    }

    dmgVerification = withMountedDmg(dmgSnapshotPath, tempRoot, dmgAppPath => {
      const verification = verifyApplication(dmgAppPath, teamId, 'DMG application');
      run('/usr/bin/xcrun', ['stapler', 'validate', dmgAppPath]);
      if (verification.treeSha256 !== appBefore.treeSha256) {
        throw new Error('DMG does not contain the verified mode-preserving application tree');
      }
      return verification;
    });
    if (packagedInputs.frontendTreeSha256 !== buildMetadata.buildOutputs.frontendTreeSha256) {
      throw new Error('Packaged frontend tree verification drifted');
    }
    appBefore.packagedInputs = packagedInputs;
    assertIdentityUnchanged(
      zipSnapshotIdentity,
      artifactFileIdentity(tempRoot, zipSnapshotPath, 'Final ZIP verification snapshot'),
      'ZIP verification snapshot'
    );
    assertIdentityUnchanged(
      dmgSnapshotIdentity,
      artifactFileIdentity(tempRoot, dmgSnapshotPath, 'Final DMG verification snapshot'),
      'DMG verification snapshot'
    );
    if (hashTree(appSnapshot.path) !== appBefore.treeSha256) {
      throw new Error('Application verification snapshot changed before completion');
    }
  } finally {
    fs.rmSync(tempRoot, { recursive: true, force: true });
  }

  assertSourceBinding(expectedSource, sourceSnapshot(), 'verification completion');
  if (hashTree(appPath) !== appBefore.treeSha256) {
    throw new Error('Application tree changed during release verification');
  }
  const finalReleaseArtifactInputs = findReleaseArtifactInputs(
    artifactDirectory,
    { allowLegacy }
  );
  if (
    finalReleaseArtifactInputs.dmgPath !== dmgPath
    || finalReleaseArtifactInputs.zipPath !== zipPath
    || finalReleaseArtifactInputs.appNotarizationEvidencePath
      !== appNotarizationEvidencePath
    || finalReleaseArtifactInputs.appNotaryLogPath !== appNotaryLogPath
    || finalReleaseArtifactInputs.dmgNotarizationEvidencePath
      !== dmgNotarizationEvidencePath
    || finalReleaseArtifactInputs.dmgNotaryLogPath !== dmgNotaryLogPath
  ) {
    throw new Error('Top-level release artifact paths changed during verification');
  }
  const buildAttestationPath = path.join(
    appPath,
    'Contents',
    'Resources',
    'release-build-attestation.json'
  );
  const smokeManifestPath = path.join(
    path.dirname(smokeResultPath),
    'artifact-manifest.json'
  );
  const smokeMarkdownPath = path.join(path.dirname(smokeResultPath), 'result.md');
  const verifiedIdentities = {
    initialVerification: initialVerificationIdentity,
    buildMetadata: artifactFileIdentity(
      artifactDirectory,
      buildMetadataPath,
      'Final release build metadata'
    ),
    buildAttestation: artifactFileIdentity(
      artifactDirectory,
      buildAttestationPath,
      'Final signed release build attestation'
    ),
    dmg: artifactFileIdentity(artifactDirectory, dmgPath, 'Final release DMG'),
    zip: artifactFileIdentity(artifactDirectory, zipPath, 'Final release ZIP'),
    appNotarizationEvidence: appNotarizationEvidencePath
      ? artifactFileIdentity(
        artifactDirectory,
        appNotarizationEvidencePath,
        'Final application notarization evidence'
      )
      : null,
    appNotaryLog: appNotaryLogPath
      ? artifactFileIdentity(
        artifactDirectory,
        appNotaryLogPath,
        'Final application retained Apple notary log'
      )
      : null,
    dmgNotarizationEvidence: artifactFileIdentity(
      artifactDirectory,
      dmgNotarizationEvidencePath,
      'Final DMG notarization evidence'
    ),
    dmgNotaryLog: dmgNotaryLogPath
      ? artifactFileIdentity(
        artifactDirectory,
        dmgNotaryLogPath,
        'Final DMG retained Apple notary log'
      )
      : null,
    smokeResult: artifactFileIdentity(
      artifactDirectory,
      smokeResultPath,
      'Final packaged runtime smoke result'
    ),
    smokeManifest: artifactFileIdentity(
      artifactDirectory,
      smokeManifestPath,
      'Final packaged runtime smoke manifest'
    ),
    smokeMarkdown: artifactFileIdentity(
      artifactDirectory,
      smokeMarkdownPath,
      'Final packaged runtime smoke Markdown'
    )
  };
  assertIdentityUnchanged(dmgBefore, verifiedIdentities.dmg, 'DMG');
  assertIdentityUnchanged(zipBefore, verifiedIdentities.zip, 'ZIP');
  assertIdentityUnchanged(
    buildMetadataIdentity,
    verifiedIdentities.buildMetadata,
    'build metadata'
  );
  assertIdentityUnchanged(
    appBefore.buildAttestationIdentity,
    verifiedIdentities.buildAttestation,
    'signed release build attestation'
  );
  assertIdentityUnchanged(
    smokeEvidence.resultIdentity,
    verifiedIdentities.smokeResult,
    'runtime smoke result'
  );
  assertIdentityUnchanged(
    smokeEvidence.manifest,
    verifiedIdentities.smokeManifest,
    'runtime smoke manifest'
  );
  assertIdentityUnchanged(
    smokeEvidence.markdown,
    verifiedIdentities.smokeMarkdown,
    'runtime smoke Markdown'
  );
  if (appNotarizationEvidenceIdentity) {
    assertIdentityUnchanged(
      appNotarizationEvidenceIdentity,
      verifiedIdentities.appNotarizationEvidence,
      'application notarization evidence'
    );
  }
  if (appNotaryLogIdentity) {
    assertIdentityUnchanged(
      appNotaryLogIdentity,
      verifiedIdentities.appNotaryLog,
      'application retained Apple notary log'
    );
  }
  assertIdentityUnchanged(
    dmgNotarizationEvidenceIdentity,
    verifiedIdentities.dmgNotarizationEvidence,
    'DMG notarization evidence'
  );
  if (dmgNotaryLogIdentity) {
    assertIdentityUnchanged(
      dmgNotaryLogIdentity,
      verifiedIdentities.dmgNotaryLog,
      'DMG retained Apple notary log'
    );
  }

  const revalidateRetainedIdentities = () => {
    const checks = [
      ['DMG', verifiedIdentities.dmg, dmgPath],
      ['ZIP', verifiedIdentities.zip, zipPath],
      ['build metadata', verifiedIdentities.buildMetadata, buildMetadataPath],
      ['build attestation', verifiedIdentities.buildAttestation, buildAttestationPath],
      ['runtime smoke result', verifiedIdentities.smokeResult, smokeResultPath],
      ['runtime smoke manifest', verifiedIdentities.smokeManifest, smokeManifestPath],
      ['runtime smoke Markdown', verifiedIdentities.smokeMarkdown, smokeMarkdownPath],
      [
        'DMG notarization evidence',
        verifiedIdentities.dmgNotarizationEvidence,
        dmgNotarizationEvidencePath
      ]
    ];
    if (appNotarizationEvidencePath) {
      checks.push([
        'application notarization evidence',
        verifiedIdentities.appNotarizationEvidence,
        appNotarizationEvidencePath
      ]);
    }
    if (appNotaryLogPath) {
      checks.push([
        'application retained Apple notary log',
        verifiedIdentities.appNotaryLog,
        appNotaryLogPath
      ]);
    }
    if (dmgNotaryLogPath) {
      checks.push([
        'DMG retained Apple notary log',
        verifiedIdentities.dmgNotaryLog,
        dmgNotaryLogPath
      ]);
    }
    if (initialVerificationIdentity) {
      checks.push([
        'initial isolated-output release verification',
        initialVerificationIdentity,
        initialVerificationPath
      ]);
    }
    for (const [label, expected, filePath] of checks) {
      assertIdentityUnchanged(
        expected,
        artifactFileIdentity(artifactDirectory, filePath, `Postcondition ${label}`),
        label
      );
    }
    if (hashTree(appPath) !== appBefore.treeSha256) {
      throw new Error('Application tree changed before release manifest publication');
    }
  };

  if (!publicationPolicy.publishManifest) {
    assertSourceBinding(expectedSource, sourceSnapshot(), 'legacy inspection completion');
    revalidateRetainedIdentities();
    process.stdout.write(`${JSON.stringify({
      status: 'passed',
      releaseEligible: false,
      verificationMode: publicationPolicy.verificationMode,
      manifestPublished: false
    })}\n`);
    return;
  }

  const manifest = createPortableReleaseManifest({
    artifactDirectory,
    retainedArtifacts: options.retainedArtifacts,
    isolatedOutputsPresent,
    initialVerificationPath: initialVerificationIdentity ? initialVerificationPath : null,
    expectedHead,
    expectedTree,
    buildMetadataPath,
    buildMetadata,
    appPath,
    appBefore,
    zipVerification,
    dmgVerification,
    dmgPath,
    zipPath,
    dmgSignature,
    appNotarizationEvidence,
    appNotarizationEvidencePath,
    appNotaryLogPath,
    dmgNotarizationEvidence,
    dmgNotarizationEvidencePath,
    dmgNotaryLogPath,
    smokeResultPath,
    verifiedIdentities
  });
  const manifestPath = path.join(
    artifactDirectory,
    options.retainedArtifacts ? 'release-manifest.json' : 'release-verification-initial.json'
  );
  const candidatePath = path.join(
    artifactDirectory,
    `.${path.basename(manifestPath)}.${crypto.randomUUID()}.candidate`
  );
  const candidateIdentity = writeNewArtifactFileNoFollow(
    artifactDirectory,
    candidatePath,
    `${JSON.stringify(manifest, null, 2)}\n`,
    options.retainedArtifacts
      ? 'Final release manifest candidate'
      : 'Initial release verification candidate'
  );
  let publishedIdentity = null;
  try {
    assertSourceBinding(expectedSource, sourceSnapshot(), 'release manifest candidate');
    revalidateRetainedIdentities();
    publishedIdentity = publishNewArtifactFileNoFollow(
      artifactDirectory,
      candidatePath,
      manifestPath,
      candidateIdentity,
      options.retainedArtifacts
        ? 'Final release manifest publication'
        : 'Initial release verification publication'
    );
    assertSourceBinding(expectedSource, sourceSnapshot(), 'release manifest publication');
    revalidateRetainedIdentities();
  } catch (error) {
    if (
      publishedIdentity
      && fs.lstatSync(manifestPath, { throwIfNoEntry: false })
    ) {
      try {
        removeArtifactFileNoFollow(
          artifactDirectory,
          manifestPath,
          'Failed published release manifest cleanup',
          publishedIdentity
        );
      } catch (cleanupError) {
        throw new AggregateError(
          [error, cleanupError],
          'Published release manifest failed postconditions and could not be removed safely'
        );
      }
    }
    if (fs.lstatSync(candidatePath, { throwIfNoEntry: false })) {
      try {
        removeArtifactFileNoFollow(
          artifactDirectory,
          candidatePath,
          'Failed release manifest candidate cleanup',
          candidateIdentity
        );
      } catch (cleanupError) {
        throw new AggregateError(
          [error, cleanupError],
          'Release manifest failed and its candidate could not be removed safely'
        );
      }
    }
    throw error;
  }
  process.stdout.write(`${JSON.stringify({
    status: 'passed',
    releaseEligible: publicationPolicy.releaseEligible,
    verificationMode: publicationPolicy.verificationMode,
    manifestPublished: true
  })}\n`);
}

if (process.argv[1] && path.resolve(process.argv[1]) === scriptPath) {
  try {
    main();
  } catch (error) {
    process.stderr.write(`${error?.stack || error?.message || error}\n`);
    process.exitCode = 1;
  }
}
