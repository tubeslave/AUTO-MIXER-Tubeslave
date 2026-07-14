#!/usr/bin/env node

import crypto from 'node:crypto';
import fs from 'node:fs';
import fsp from 'node:fs/promises';
import http from 'node:http';
import net from 'node:net';
import os from 'node:os';
import path from 'node:path';
import { spawn, spawnSync } from 'node:child_process';
import { createRequire } from 'node:module';
import { fileURLToPath } from 'node:url';

import { chromium } from 'playwright';
import {
  artifactFileIdentity,
  artifactRelativePath,
  hashTree,
  readStableJsonFileNoFollow,
  resolveArtifactPath,
  verifyPackagedRuntimeSmokeManifest
} from './release-evidence.mjs';

export { verifyPackagedRuntimeSmokeManifest };

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const require = createRequire(import.meta.url);
const {
  BACKEND_NETWORK_SANDBOX_PROBES,
  BACKEND_SANDBOX_PROFILE_SHA256,
  LOOPBACK_ONLY_BACKEND_SANDBOX_PROFILE
} = require('../public/electronSecurity.js');
const frontendRoot = path.resolve(__dirname, '..');
const repoRoot = path.resolve(frontendRoot, '..');
const defaultAppPath = path.join(
  frontendRoot,
  'dist',
  'mac-arm64',
  'AUTO MIXER Tubeslave.app'
);
const PACKAGED_RUNTIME_SOURCE_MANIFEST_FILENAME = 'automixer-runtime-source-manifest.json';
const EXPECTED_RUNTIME_CRITICAL_SOURCE_PATHS = [
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
function parseArgs(argv) {
  const options = {
    appPath: defaultAppPath,
    buildProfilePath: null,
    artifactRoot: path.join(repoRoot, 'artifacts', 'packaged_runtime_smoke'),
    timeoutMs: 120_000,
    maxBackendReadyMs: 20_000,
    keepTemp: false,
    allowLegacy: false
  };
  for (let index = 0; index < argv.length; index += 1) {
    const argument = argv[index];
    if (argument === '--app') options.appPath = path.resolve(argv[++index]);
    else if (argument === '--build-profile') options.buildProfilePath = path.resolve(argv[++index]);
    else if (argument === '--artifact-root') options.artifactRoot = path.resolve(argv[++index]);
    else if (argument === '--timeout-ms') options.timeoutMs = Number(argv[++index]);
    else if (argument === '--max-backend-ready-ms') {
      options.maxBackendReadyMs = Number(argv[++index]);
    }
    else if (argument === '--keep-temp') options.keepTemp = true;
    else if (argument === '--allow-legacy') options.allowLegacy = true;
    else throw new Error(`Unknown argument: ${argument}`);
  }
  if (!Number.isFinite(options.timeoutMs) || options.timeoutMs < 5_000) {
    throw new Error('--timeout-ms must be at least 5000');
  }
  if (!Number.isFinite(options.maxBackendReadyMs) || options.maxBackendReadyMs < 1_000) {
    throw new Error('--max-backend-ready-ms must be at least 1000');
  }
  return options;
}

export function loadBuildProfileBinding(
  appPath,
  explicitPath = null,
  { allowLegacy = false } = {}
) {
  const resolvedAppPath = path.resolve(appPath);
  const profilePath = explicitPath
    ? path.resolve(explicitPath)
    : path.join(path.dirname(path.dirname(resolvedAppPath)), 'build-profile.json');
  if (!fs.existsSync(profilePath)) {
    throw new Error(`Packaged runtime smoke build profile is missing: ${profilePath}`);
  }
  const {
    bytes: profileBytes,
    value: profile
  } = readStableJsonFileNoFollow(profilePath, 'Packaged runtime build profile');
  const artifactRoot = path.dirname(profilePath);
  const legacyAllowed = profile.pathPolicy == null && allowLegacy === true;
  let declaredAppPath = null;
  try {
    declaredAppPath = resolveArtifactPath(
      artifactRoot,
      profile.app?.path,
      'Build profile app path',
      { allowLegacyAbsoluteWithinRoot: legacyAllowed }
    );
  } catch {
    declaredAppPath = null;
  }
  const embeddedPath = path.join(
    resolvedAppPath,
    'Contents',
    'Resources',
    'local-build-attestation.json'
  );
  const embedded = fs.existsSync(embeddedPath)
    ? readStableJsonFileNoFollow(
      embeddedPath,
      'Embedded local build attestation'
    ).value
    : null;
  const localBindingValid = profile.profile !== 'local' || Boolean(
    profile.releaseEligible === false
    && embedded?.schema === 'automixer.macos_local_build_attestation.v1'
    && embedded?.profile === 'local'
    && embedded?.releaseEligible === false
    && JSON.stringify(embedded?.source) === JSON.stringify(profile.source)
  );
  const bindingValid = Boolean(
    profile.schema === 'automixer.macos_build_profile.v1'
    && (profile.pathPolicy === 'artifact_root_relative_v1' || legacyAllowed)
    && ['local', 'release'].includes(profile.profile)
    && typeof profile.releaseEligible === 'boolean'
    && declaredAppPath === resolvedAppPath
    && /^[0-9a-f]{64}$/.test(profile.app?.treeSha256 || '')
    && localBindingValid
  );
  const binding = {
    path: artifactRelativePath(artifactRoot, profilePath, 'Build profile'),
    pathPolicy: profile.pathPolicy || (
      legacyAllowed ? 'legacy_absolute_within_artifact_root_v1' : 'missing_strict_path_policy'
    ),
    size: profileBytes.length,
    sha256: sha256Buffer(profileBytes),
    schema: profile.schema,
    profile: profile.profile,
    releaseEligible: profile.releaseEligible,
    reason: profile.reason,
    source: profile.source,
    declaredApp: {
      ...profile.app,
      path: declaredAppPath
        ? artifactRelativePath(artifactRoot, declaredAppPath, 'Declared packaged app')
        : null
    },
    embeddedAttestationPath: embedded
      ? artifactRelativePath(artifactRoot, embeddedPath, 'Embedded build attestation')
      : null,
    embeddedAttestation: embedded,
    bindingValid,
    appTreeMatches: null
  };
  Object.defineProperties(binding, {
    artifactRoot: { value: artifactRoot, enumerable: false },
    resolvedPath: { value: profilePath, enumerable: false },
    declaredAppPath: { value: declaredAppPath, enumerable: false }
  });
  return binding;
}

function utcTimestamp() {
  return new Date().toISOString().replace(/[-:]/g, '').replace(/\.\d{3}Z$/, 'Z');
}

function sha256Buffer(value) {
  return crypto.createHash('sha256').update(value).digest('hex');
}

function canonicalJson(value) {
  if (Array.isArray(value)) return `[${value.map(canonicalJson).join(',')}]`;
  if (value && typeof value === 'object') {
    return `{${Object.keys(value).sort().map(key => (
      `${JSON.stringify(key)}:${canonicalJson(value[key])}`
    )).join(',')}}`;
  }
  return JSON.stringify(value);
}

function runtimeIdentityManifestSha256(identity) {
  const canonical = JSON.parse(JSON.stringify(identity));
  delete canonical.manifest_sha256;
  delete canonical.request_challenge_nonce;
  return sha256Buffer(Buffer.from(canonicalJson(canonical)));
}

function packagedSourceManifestSha256(manifest) {
  const canonical = JSON.parse(JSON.stringify(manifest));
  delete canonical.manifest_sha256;
  return sha256Buffer(Buffer.from(canonicalJson(canonical)));
}

async function fileIdentity(filePath, recordedPath = filePath, pathBase = undefined) {
  try {
    const value = await fsp.readFile(filePath);
    const identity = {
      path: recordedPath,
      size: value.length,
      sha256: sha256Buffer(value)
    };
    if (pathBase) identity.pathBase = pathBase;
    return identity;
  } catch (error) {
    if (error?.code === 'ENOENT') return null;
    throw error;
  }
}

function run(command, args, options = {}) {
  const completed = spawnSync(command, args, {
    cwd: options.cwd || repoRoot,
    encoding: 'utf8',
    env: options.env || process.env,
    maxBuffer: 64 * 1024 * 1024
  });
  if (completed.error) throw completed.error;
  if (completed.status !== 0 && options.check !== false) {
    throw new Error(
      `${command} ${args.join(' ')} failed (${completed.status}): `
      + `${completed.stderr || completed.stdout}`.trim()
    );
  }
  return {
    status: completed.status,
    stdout: completed.stdout || '',
    stderr: completed.stderr || ''
  };
}

function enforceLoopbackOnlySandbox(profilePath) {
  if (sha256Buffer(LOOPBACK_ONLY_BACKEND_SANDBOX_PROFILE) !== BACKEND_SANDBOX_PROFILE_SHA256) {
    throw new Error('Canonical backend network sandbox profile hash mismatch');
  }
  fs.writeFileSync(profilePath, LOOPBACK_ONLY_BACKEND_SANDBOX_PROFILE, { mode: 0o600 });
  const probeResults = {};
  for (const [name, probe] of Object.entries(BACKEND_NETWORK_SANDBOX_PROBES)) {
    run('/usr/bin/sandbox-exec', [
      '-f', profilePath, process.execPath, '-e', probe
    ], { env: { ...process.env, ELECTRON_RUN_AS_NODE: '1' } });
    probeResults[name] = true;
  }
  return {
    schema: 'automixer.packaged_runtime_network_enforcement.v1',
    mechanism: 'macos_sandbox_exec',
    scope: 'packaged_backend_process_tree',
    profileSha256: BACKEND_SANDBOX_PROFILE_SHA256,
    probeResults,
    nonLoopbackOutboundDenied: true,
    loopbackOutboundAllowed: true,
    preflightPassed: true
  };
}

async function freeLoopbackPort() {
  return await new Promise((resolve, reject) => {
    const server = net.createServer();
    server.on('error', reject);
    server.listen(0, '127.0.0.1', () => {
      const address = server.address();
      const port = typeof address === 'object' && address ? address.port : null;
      server.close(error => {
        if (error) reject(error);
        else if (!port) reject(new Error('Could not allocate a loopback CDP port'));
        else resolve(port);
      });
    });
  });
}

async function waitForJson(url, timeoutMs) {
  const started = Date.now();
  return await new Promise((resolve, reject) => {
    const attempt = () => {
      const request = http.get(url, response => {
        let body = '';
        response.on('data', chunk => { body += chunk; });
        response.on('end', () => {
          if (response.statusCode === 200) {
            try {
              resolve(JSON.parse(body));
              return;
            } catch {
              // The endpoint may be visible before its complete response is ready.
            }
          }
          if (Date.now() - started >= timeoutMs) {
            reject(new Error(`Timed out waiting for ${url}`));
          } else {
            setTimeout(attempt, 200);
          }
        });
      });
      request.on('error', () => {
        if (Date.now() - started >= timeoutMs) {
          reject(new Error(`Timed out waiting for ${url}`));
        } else {
          setTimeout(attempt, 200);
        }
      });
      request.setTimeout(1000, () => request.destroy());
    };
    attempt();
  });
}

async function waitForJsonOrChildExit(url, timeoutMs, child) {
  let onExit;
  const childExit = new Promise((resolve, reject) => {
    onExit = (code, signal) => reject(new Error(
      `Packaged app exited before CDP became ready (code=${code}, signal=${signal})`
    ));
    child.once('exit', onExit);
  });
  try {
    return await Promise.race([waitForJson(url, timeoutMs), childExit]);
  } finally {
    child.off('exit', onExit);
  }
}

async function waitForLoopbackPort(port, timeoutMs, child) {
  const started = Date.now();
  while (Date.now() - started < timeoutMs) {
    if (child.exitCode !== null || child.signalCode !== null) {
      throw new Error(
        `Packaged app exited before backend port ${port} became ready `
        + `(code=${child.exitCode}, signal=${child.signalCode})`
      );
    }
    const connected = await new Promise(resolve => {
      const socket = net.createConnection({ host: '127.0.0.1', port });
      socket.setTimeout(500);
      socket.once('connect', () => {
        socket.destroy();
        resolve(true);
      });
      socket.once('error', () => resolve(false));
      socket.once('timeout', () => {
        socket.destroy();
        resolve(false);
      });
    });
    if (connected) return Date.now() - started;
    await new Promise(resolve => setTimeout(resolve, 250));
  }
  throw new Error(`Timed out waiting for loopback backend port ${port}`);
}

async function settleWithin(promise, timeoutMs) {
  return await Promise.race([
    promise.then(() => true, () => false),
    new Promise(resolve => setTimeout(() => resolve(false), timeoutMs))
  ]);
}

async function findMountedAppPage(browser, timeoutMs) {
  const started = Date.now();
  let diagnostics = [];
  while (Date.now() - started < timeoutMs) {
    diagnostics = [];
    const pages = browser.contexts().flatMap(context => context.pages());
    for (const page of pages) {
      try {
        const state = await page.evaluate(() => {
          const root = document.getElementById('root');
          const text = root?.innerText || document.body?.innerText || '';
          const normalizedText = text.toLowerCase();
          return {
            mounted: Boolean(root?.childElementCount),
            appLabelPresent: normalizedText.includes('automixer')
              && normalizedText.includes('tubeslave'),
            textPreview: text.slice(0, 240)
          };
        });
        diagnostics.push({ url: page.url(), ...state });
        if (state.mounted && state.appLabelPresent) return page;
      } catch (error) {
        diagnostics.push({
          url: page.url(),
          evaluationError: redact(error?.message || error)
        });
      }
    }
    await new Promise(resolve => setTimeout(resolve, 200));
  }
  throw new Error(
    `Timed out waiting for mounted Automixer renderer: ${JSON.stringify(diagnostics)}`
  );
}

function redact(text) {
  return String(text || '')
    .replace(/[A-Za-z0-9_-]{43}/g, '[REDACTED_TOKEN]')
    .replace(/("token"\s*:\s*")[^"]+("?)/gi, '$1[REDACTED_TOKEN]$2');
}

function processGroupMembers(processGroupId) {
  const listing = run('/bin/ps', ['-axo', 'pid=,pgid=,command=']).stdout;
  return listing
    .split('\n')
    .map(line => {
      const match = line.match(/^\s*(\d+)\s+(\d+)\s+(.*)$/);
      return match ? {
        pid: Number(match[1]),
        pgid: Number(match[2]),
        command: match[3]
      } : null;
    })
    .filter(row => row && row.pgid === processGroupId);
}

function inspectNetwork(processes) {
  const sockets = [];
  for (const processInfo of processes) {
    const completed = run(
      '/usr/sbin/lsof',
      ['-nP', '-a', '-p', String(processInfo.pid), '-i'],
      { check: false }
    );
    for (const line of completed.stdout.split('\n').slice(1)) {
      const trimmed = line.trim();
      if (!trimmed) continue;
      sockets.push({ pid: processInfo.pid, line: trimmed });
    }
  }
  const externalConnections = sockets.filter(({ line }) => {
    const arrow = line.indexOf('->');
    if (arrow === -1) return false;
    const remote = line.slice(arrow + 2).split(/\s|\(/, 1)[0];
    return !(
      remote.startsWith('127.0.0.1:')
      || remote.startsWith('[::1]:')
      || remote.startsWith('localhost:')
    );
  });
  return { sockets, externalConnections };
}

function createContinuousNetworkMonitor(child, intervalMs = 250) {
  const startedAt = Date.now();
  const socketObservations = new Map();
  const externalConnections = new Map();
  const wingPortSockets = new Map();
  const errors = [];
  let sampleCount = 0;
  const sample = () => {
    try {
      const processes = processGroupMembers(child.pid);
      const snapshot = inspectNetwork(processes);
      sampleCount += 1;
      for (const row of snapshot.sockets) {
        socketObservations.set(`${row.pid}:${row.line}`, row);
        if (row.line.includes(':2222') || row.line.includes(':2223')) {
          wingPortSockets.set(`${row.pid}:${row.line}`, row);
        }
      }
      for (const row of snapshot.externalConnections) {
        externalConnections.set(`${row.pid}:${row.line}`, row);
      }
    } catch (error) {
      errors.push(redact(error?.message || error));
    }
  };
  sample();
  const timer = setInterval(sample, intervalMs);
  return {
    stop() {
      clearInterval(timer);
      sample();
      return {
        continuous: true,
        intervalMs,
        durationMs: Date.now() - startedAt,
        sampleCount,
        sockets: [...socketObservations.values()],
        externalConnections: [...externalConnections.values()],
        wingPortSockets: [...wingPortSockets.values()],
        monitorErrors: errors
      };
    }
  };
}

function listFilesRecursively(root) {
  if (!fs.existsSync(root)) return [];
  const files = [];
  function visit(candidate, relative) {
    const stat = fs.lstatSync(candidate);
    if (stat.isSymbolicLink()) {
      files.push({ path: relative, type: 'symlink', target: fs.readlinkSync(candidate) });
      return;
    }
    if (stat.isDirectory()) {
      for (const name of fs.readdirSync(candidate).sort()) {
        visit(path.join(candidate, name), relative ? `${relative}/${name}` : name);
      }
      return;
    }
    if (stat.isFile()) {
      files.push({
        path: relative,
        type: 'file',
        size: stat.size,
        sha256: sha256Buffer(fs.readFileSync(candidate))
      });
    }
  }
  visit(root, '');
  return files;
}

async function authenticatedRuntimeProbe(page, bootstrap, timeoutMs) {
  const identityChallengeNonce = crypto.randomBytes(16).toString('hex');
  return await page.evaluate(
    ({ bootstrapPayload, probeTimeoutMs, identityChallenge }) => new Promise(resolve => {
      const result = {
        authenticated: false,
        mode: null,
        runtimeIdentity: null,
        runtimeIdentityChallengeNonce: identityChallenge,
        blockedOperations: []
      };
      const expectedOperations = new Set(['connect_wing', 'scan_mixers', 'auto_connect']);
      const websocket = new WebSocket(bootstrapPayload.websocketUrl);
      const timer = setTimeout(() => {
        websocket.close();
        resolve({ ...result, timeout: true });
      }, probeTimeoutMs);

      const finishIfComplete = () => {
        if (
          result.authenticated
          && result.mode
          && result.runtimeIdentity
          && result.blockedOperations.length === expectedOperations.size
        ) {
          clearTimeout(timer);
          websocket.close();
          resolve(result);
        }
      };

      websocket.onopen = () => websocket.send(JSON.stringify({
        type: 'authenticate',
        token: bootstrapPayload.token,
        client_id: bootstrapPayload.clientId
      }));
      websocket.onmessage = event => {
        const payload = JSON.parse(event.data);
        if (payload.type === 'authentication_status' && payload.authenticated === true) {
          result.authenticated = true;
          websocket.send(JSON.stringify({ type: 'get_operator_mode' }));
          websocket.send(JSON.stringify({
            type: 'get_runtime_identity',
            challenge_nonce: identityChallenge
          }));
          return;
        }
        if (payload.type === 'runtime_identity') {
          result.runtimeIdentity = payload;
          finishIfComplete();
          return;
        }
        if (payload.type === 'operator_mode_status') {
          result.mode = {
            mode: payload.mode,
            packagedOfflineLock: payload.packaged_offline_lock === true,
            mixerTransportLocked: payload.mixer_transport_locked === true
          };
          websocket.send(JSON.stringify({
            type: 'connect_wing',
            ip: '192.168.1.102',
            observation_only: false,
            record_telemetry: true
          }));
          websocket.send(JSON.stringify({ type: 'scan_mixers', full_scan: true }));
          websocket.send(JSON.stringify({ type: 'auto_connect', full_scan: true }));
          return;
        }
        if (
          payload.type === 'connection_status'
          && payload.reason === 'packaged_offline_lock_active'
          && expectedOperations.has(payload.operation)
          && !result.blockedOperations.includes(payload.operation)
        ) {
          result.blockedOperations.push(payload.operation);
          result.blockedOperations.sort();
          finishIfComplete();
        }
      };
      websocket.onerror = () => {};
      websocket.onclose = event => {
        if (
          result.authenticated
          && result.runtimeIdentity
          && result.blockedOperations.length === expectedOperations.size
        ) {
          return;
        }
        clearTimeout(timer);
        resolve({ ...result, closedEarly: true, code: event.code, reason: event.reason });
      };
    }),
    {
      bootstrapPayload: bootstrap,
      probeTimeoutMs: Math.min(timeoutMs, 10_000),
      identityChallenge: identityChallengeNonce
    }
  );
}

async function rejectedTokenProbe(page) {
  return await page.evaluate(() => new Promise(resolve => {
    const websocket = new WebSocket('ws://127.0.0.1:8765');
    const timer = setTimeout(() => resolve({ timeout: true }), 5000);
    websocket.onopen = () => websocket.send(JSON.stringify({
      type: 'authenticate',
      token: 'invalid-or-expired-token',
      client_id: 'automixer-ui'
    }));
    websocket.onmessage = event => {
      clearTimeout(timer);
      resolve({ unexpectedMessageType: JSON.parse(event.data)?.type || 'unknown' });
      websocket.close();
    };
    websocket.onclose = event => {
      clearTimeout(timer);
      resolve({
        rejected: event.code === 1008,
        code: event.code,
        reason: event.reason
      });
    };
    websocket.onerror = () => {};
  }));
}

function markdown(result) {
  const yesNo = value => value ? 'PASS' : 'FAIL';
  return [
    '# Packaged runtime smoke',
    '',
    `- Overall: ${yesNo(result.pass)}`,
    `- Started packaged app: ${yesNo(result.runtime?.cdpReady)}`,
    `- Backend ready within ${result.runtime?.backendReadyBudgetMs ?? 'unknown'} ms: ${yesNo(result.runtime?.backendReadyWithinBudget)}`,
    `- Secure IPC bootstrap: ${yesNo(result.bootstrap?.validShape)}`,
    `- Invalid or expired token rejected: ${yesNo(result.invalidToken?.rejected)}`,
    `- Fresh token recovery authenticated: ${yesNo(result.runtimeProbe?.authenticated)}`,
    `- Packaged runtime identity source/code bound: ${yesNo(result.runtimeIdentityValidation?.valid)}`,
    `- Immutable offline lock reported: ${yesNo(result.runtimeProbe?.mode?.packagedOfflineLock)}`,
    `- Mixer connect/discovery operations blocked: ${yesNo(result.blockedOperationsPass)}`,
    `- Stale session token removed on reload: ${yesNo(result.reloadRecovery?.staleSessionTokenRemoved)}`,
    `- Hardware controls hidden: ${yesNo(result.reloadRecovery?.hardwareControlsHidden)}`,
    `- OS-enforced backend non-loopback network deny: ${yesNo(result.networkEnforcement?.preflightPassed && result.networkEnforcement?.backendBoundaryReported)}`,
    `- Launched copy matches signed app before/after: ${yesNo(result.launchCopy?.unchanged && result.launchCopy?.stillMatchesSourceApp)}`,
    `- External process-group connections: ${result.network?.externalConnections?.length ?? 'unknown'}`,
    `- Continuous network samples: ${result.network?.sampleCount ?? 'unknown'}`,
    `- WING-port socket observations: ${result.network?.wingPortSockets?.length ?? 'unknown'}`,
    `- Canonical WING log unchanged: ${yesNo(result.canonicalLogUnchanged)}`,
    `- Canonical decision trace log unchanged: ${yesNo(result.canonicalDecisionTraceUnchanged)}`,
    `- Canonical decision memory log unchanged: ${yesNo(result.canonicalDecisionMemoryUnchanged)}`,
    `- Console WAL unchanged: ${yesNo(result.consoleWalUnchanged)}`,
    `- Complete application tree unchanged: ${yesNo(result.appTreeUnchanged)}`,
    `- Backend onedir bundle unchanged: ${yesNo(result.backendBundleUnchanged)}`,
    `- Temporary write artifacts absent: ${yesNo(result.tempWriteArtifactsAbsent)}`,
    `- Source binding: ${result.provenance?.sourceBindingStrength || 'unknown'}`,
    `- Build profile: ${result.buildProfile?.profile || 'unknown'}`,
    `- Release eligible: ${String(result.buildProfile?.releaseEligible)}`,
    `- Build profile bound to app: ${yesNo(result.buildProfile?.bindingValid && result.buildProfile?.appTreeMatches)}`,
    '',
    'This artifact proves an offline packaged runtime launch only. It grants no live authority and is not macOS notarization evidence.',
    ''
  ].join('\n');
}

async function writeArtifact(result, artifactRoot, runId) {
  const runDirectory = path.join(artifactRoot, runId);
  await fsp.mkdir(artifactRoot, { recursive: true });
  await fsp.mkdir(runDirectory);
  const jsonPath = path.join(runDirectory, 'result.json');
  const markdownPath = path.join(runDirectory, 'result.md');
  await fsp.writeFile(jsonPath, `${JSON.stringify(result, null, 2)}\n`, { mode: 0o600 });
  await fsp.writeFile(markdownPath, markdown(result), { mode: 0o600 });
  const manifest = {
    schema: 'automixer.packaged_runtime_smoke_manifest.v1',
    pathPolicy: 'manifest_directory_relative_v1',
    generatedAt: new Date().toISOString(),
    readOnly: result.readOnly,
    contactsWing: result.contactsWing,
    liveAuthorityGranted: result.liveAuthorityGranted,
    buildProfile: {
      profile: result.buildProfile?.profile,
      releaseEligible: result.buildProfile?.releaseEligible,
      path: result.buildProfile?.path,
      pathBase: 'packaged_build_artifact_root',
      sha256: result.buildProfile?.sha256,
      bindingValid: result.buildProfile?.bindingValid,
      appTreeMatches: result.buildProfile?.appTreeMatches
    },
    files: {
      'result.json': artifactFileIdentity(runDirectory, jsonPath, 'Smoke result JSON'),
      'result.md': artifactFileIdentity(runDirectory, markdownPath, 'Smoke result Markdown')
    }
  };
  const manifestPath = path.join(runDirectory, 'artifact-manifest.json');
  await fsp.writeFile(manifestPath, `${JSON.stringify(manifest, null, 2)}\n`, { mode: 0o600 });
  return { runDirectory, jsonPath, markdownPath, manifestPath };
}

async function main() {
  if (process.platform !== 'darwin' || process.arch !== 'arm64') {
    throw new Error('Packaged runtime smoke requires macOS arm64');
  }
  const options = parseArgs(process.argv.slice(2));
  const runId = `${utcTimestamp()}_${crypto.randomUUID()}_packaged_runtime_smoke`;
  const result = {
    schema: 'automixer.packaged_runtime_smoke.v1',
    pathPolicy: 'packaged_build_artifact_root_relative_v1',
    generatedAt: new Date().toISOString(),
    runId,
    readOnly: null,
    contactsWing: null,
    liveWrites: null,
    liveAuthorityGranted: null,
    pass: false,
    appPath: null
  };
  result.buildProfile = loadBuildProfileBinding(
    options.appPath,
    options.buildProfilePath,
    { allowLegacy: options.allowLegacy }
  );
  const buildArtifactRoot = result.buildProfile.artifactRoot;
  result.appPath = artifactRelativePath(
    buildArtifactRoot,
    options.appPath,
    'Packaged runtime smoke app'
  );
  const packagedFileIdentity = async (filePath, label) => await fileIdentity(
    filePath,
    artifactRelativePath(buildArtifactRoot, filePath, label),
    'packaged_build_artifact_root'
  );
  const tempRoot = await fsp.mkdtemp(path.join(os.tmpdir(), 'automixer-packaged-runtime-'));
  const copiedAppPath = path.join(tempRoot, path.basename(options.appPath));
  const sandboxProfilePath = path.join(tempRoot, 'loopback-only.sb');
  const originalFiles = {
    electron: path.join(options.appPath, 'Contents', 'MacOS', 'AUTO MIXER Tubeslave'),
    backend: path.join(options.appPath, 'Contents', 'Resources', 'backend', 'server'),
    asar: path.join(options.appPath, 'Contents', 'Resources', 'app.asar'),
    infoPlist: path.join(options.appPath, 'Contents', 'Info.plist'),
    runtimeSourceManifest: path.join(
      options.appPath,
      'Contents',
      'Resources',
      'backend',
      '_internal',
      PACKAGED_RUNTIME_SOURCE_MANIFEST_FILENAME
    )
  };
  const backendBundlePath = path.join(
    options.appPath,
    'Contents',
    'Resources',
    'backend'
  );
  const canonicalLogPath = path.join(
    repoRoot,
    'artifacts',
    'wing_telemetry',
    'manual_write_gate',
    'manual_write_events.jsonl'
  );
  const consoleWalPath = path.join(
    os.homedir(),
    '.automixer',
    'runtime',
    'manual_write_wal.jsonl'
  );
  const canonicalDecisionTracePath = path.join(
    repoRoot,
    'artifacts',
    'wing_telemetry',
    'manual_write_gate',
    'decision_traces.jsonl'
  );
  const canonicalDecisionMemoryPath = path.join(
    repoRoot,
    'artifacts',
    'wing_telemetry',
    'manual_write_gate',
    'decision_memory.jsonl'
  );
  const before = {
    canonicalLog: await fileIdentity(
      canonicalLogPath,
      'artifacts/wing_telemetry/manual_write_gate/manual_write_events.jsonl',
      'repository_root'
    ),
    canonicalDecisionTrace: await fileIdentity(
      canonicalDecisionTracePath,
      'artifacts/wing_telemetry/manual_write_gate/decision_traces.jsonl',
      'repository_root'
    ),
    canonicalDecisionMemory: await fileIdentity(
      canonicalDecisionMemoryPath,
      'artifacts/wing_telemetry/manual_write_gate/decision_memory.jsonl',
      'repository_root'
    ),
    consoleWal: await fileIdentity(
      consoleWalPath,
      '.automixer/runtime/manual_write_wal.jsonl',
      'operator_home'
    ),
    appTree: {
      path: result.appPath,
      pathBase: 'packaged_build_artifact_root',
      sha256: hashTree(options.appPath)
    },
    backendBundle: {
      path: artifactRelativePath(
        buildArtifactRoot,
        backendBundlePath,
        'Packaged backend bundle'
      ),
      pathBase: 'packaged_build_artifact_root',
      sha256: hashTree(backendBundlePath)
    },
    packageFiles: Object.fromEntries(
      await Promise.all(Object.entries(originalFiles).map(async ([key, value]) => [
        key,
        await packagedFileIdentity(value, `Packaged file ${key}`)
      ]))
    )
  };
  result.before = before;
  result.buildProfile.appTreeMatches = (
    result.buildProfile.declaredApp?.treeSha256 === before.appTree.sha256
  );
  if (!result.buildProfile.bindingValid || !result.buildProfile.appTreeMatches) {
    throw new Error('Packaged runtime smoke build profile does not bind to the app tree');
  }

  const head = run('/usr/bin/git', ['rev-parse', 'HEAD']).stdout.trim();
  const status = run('/usr/bin/git', ['status', '--porcelain=v1', '-uall']).stdout;
  const trackedDiff = run('/usr/bin/git', ['diff', '--binary']).stdout;
  result.provenance = {
    head,
    tree: run('/usr/bin/git', ['rev-parse', 'HEAD^{tree}']).stdout.trim(),
    worktreeDirty: Boolean(status.trim()),
    statusSha256: sha256Buffer(status),
    trackedDiffSha256: sha256Buffer(trackedDiff),
    sourceBindingStrength: status.trim()
      ? 'dirty_tree_non_clean_checkout'
      : 'clean_checkout'
  };

  let child;
  let browser;
  let networkMonitor;
  const stderr = [];
  const consoleErrors = [];
  try {
    run('/usr/bin/codesign', ['--verify', '--deep', '--strict', '--verbose=2', options.appPath]);
    result.networkEnforcement = enforceLoopbackOnlySandbox(sandboxProfilePath);
    run('/usr/bin/ditto', [options.appPath, copiedAppPath]);
    run('/usr/bin/codesign', ['--verify', '--deep', '--strict', '--verbose=2', copiedAppPath]);
    const copiedTreeBefore = hashTree(copiedAppPath);
    if (copiedTreeBefore !== before.appTree.sha256) {
      throw new Error('Packaged runtime launch copy does not match the verified source app tree');
    }
    result.launchCopy = {
      path: path.basename(copiedAppPath),
      pathBase: 'ephemeral_smoke_root',
      signatureVerifiedBefore: true,
      treeSha256Before: copiedTreeBefore,
      matchesSourceApp: true
    };
    const copiedBinary = path.join(
      copiedAppPath,
      'Contents',
      'MacOS',
      'AUTO MIXER Tubeslave'
    );
    const cdpPort = await freeLoopbackPort();
    const launchStarted = Date.now();
    child = spawn(copiedBinary, [
      `--remote-debugging-port=${cdpPort}`,
      `--user-data-dir=${path.join(tempRoot, 'profile')}`,
      '--disable-background-networking',
      '--disable-component-update',
      '--no-default-browser-check'
    ], {
      detached: true,
      stdio: ['ignore', 'pipe', 'pipe'],
      env: {
        PATH: '/usr/bin:/bin:/usr/sbin:/sbin',
        HOME: tempRoot,
        USER: process.env.USER || 'automixer-smoke',
        LOGNAME: process.env.LOGNAME || process.env.USER || 'automixer-smoke',
        TMPDIR: tempRoot,
        LANG: 'en_US.UTF-8',
        AUTOMIXER_PACKAGED_OFFLINE_LOCK: 'true',
        AUTOMIXER_MANUAL_WRITE_ARTIFACTS_DIR: path.join(tempRoot, 'artifacts', 'manual'),
        AUTOMIXER_SHADOW_REPORTS_DIR: path.join(tempRoot, 'artifacts', 'shadow'),
        AUTOMIXER_REPLAY_ARTIFACTS_DIR: path.join(tempRoot, 'artifacts', 'replay'),
        AUTOMIXER_WING_REPLAY_READINESS_DIR: path.join(tempRoot, 'artifacts', 'readiness'),
        AUTOMIXER_WING_TELEMETRY_DIR: path.join(tempRoot, 'artifacts', 'wing'),
        HF_HUB_OFFLINE: '1',
        TRANSFORMERS_OFFLINE: '1',
        HF_DATASETS_OFFLINE: '1',
        HF_HUB_DISABLE_TELEMETRY: '1'
      }
    });
    child.stderr.on('data', data => {
      const lines = redact(data).split('\n').map(line => line.trim()).filter(Boolean);
      stderr.push(...lines);
      if (stderr.length > 200) stderr.splice(0, stderr.length - 200);
    });
    child.stdout.on('data', () => {});
    networkMonitor = createContinuousNetworkMonitor(child);

    const cdpVersion = await waitForJsonOrChildExit(
      `http://127.0.0.1:${cdpPort}/json/version`,
      options.timeoutMs,
      child
    );
    result.runtime = {
      cdpReady: true,
      cdpReadyMs: Date.now() - launchStarted,
      browser: cdpVersion.Browser,
      protocolVersion: cdpVersion['Protocol-Version']
    };
    browser = await chromium.connectOverCDP(`http://127.0.0.1:${cdpPort}`, {
      timeout: options.timeoutMs
    });
    const page = await findMountedAppPage(browser, Math.min(options.timeoutMs, 20_000));
    result.runtime.pageUrls = browser.contexts().flatMap(context => (
      context.pages().map(candidate => candidate.url())
    ));
    page.on('console', message => {
      if (message.type() === 'error') consoleErrors.push(redact(message.text()));
    });
    page.on('pageerror', error => consoleErrors.push(redact(error.message)));

    const bootstrap = await page.evaluate(() => window.electron.getWebSocketAuthBootstrap());
    const expectedSandboxProbeNames = Object.keys(BACKEND_NETWORK_SANDBOX_PROBES).sort();
    const bootstrapSandboxProbeNames = Object.keys(
      bootstrap.backendNetworkSandboxProbeResults || {}
    ).sort();
    const sandboxProbeMatrixMatches = (
      JSON.stringify(bootstrapSandboxProbeNames) === JSON.stringify(expectedSandboxProbeNames)
      && expectedSandboxProbeNames.every(name => (
        bootstrap.backendNetworkSandboxProbeResults?.[name] === true
        && result.networkEnforcement.probeResults?.[name] === true
      ))
    );
    result.bootstrap = {
      validShape: Boolean(
        bootstrap
        && /^[A-Za-z0-9_-]{43}$/.test(bootstrap.token)
        && bootstrap.clientId === 'automixer-ui'
        && bootstrap.websocketUrl === 'ws://127.0.0.1:8765'
        && bootstrap.packagedOfflineLock === true
        && bootstrap.backendNetworkSandboxed === true
        && bootstrap.backendNetworkSandboxProfileSha256
          === result.networkEnforcement.profileSha256
        && sandboxProbeMatrixMatches
      ),
      tokenLength: bootstrap.token.length,
      tokenSha256: sha256Buffer(bootstrap.token),
      clientId: bootstrap.clientId,
      websocketUrl: bootstrap.websocketUrl,
      packagedOfflineLock: bootstrap.packagedOfflineLock,
      backendNetworkSandboxed: bootstrap.backendNetworkSandboxed,
      backendNetworkSandboxProfileSha256: bootstrap.backendNetworkSandboxProfileSha256,
      backendNetworkSandboxProbeResults: bootstrap.backendNetworkSandboxProbeResults
    };
    result.networkEnforcement.backendBoundaryReported = (
      bootstrap.backendNetworkSandboxed === true
    );
    result.networkEnforcement.profileMatchesBackend = (
      bootstrap.backendNetworkSandboxProfileSha256
      === result.networkEnforcement.profileSha256
    );
    result.networkEnforcement.probeMatrixMatchesBackend = sandboxProbeMatrixMatches;
    const backendWaitMs = await waitForLoopbackPort(8765, options.timeoutMs, child);
    result.runtime.backendReadyMs = Date.now() - launchStarted;
    result.runtime.backendWaitAfterUiMs = backendWaitMs;
    result.runtime.backendReadyBudgetMs = options.maxBackendReadyMs;
    result.runtime.backendReadyWithinBudget = (
      result.runtime.backendReadyMs <= options.maxBackendReadyMs
    );
    result.invalidToken = await rejectedTokenProbe(page);
    result.runtimeProbe = await authenticatedRuntimeProbe(page, bootstrap, options.timeoutMs);
    const runtimeIdentity = result.runtimeProbe.runtimeIdentity;
    const runtimeSourceManifestBytes = fs.readFileSync(
      originalFiles.runtimeSourceManifest
    );
    const runtimeSourceManifest = JSON.parse(runtimeSourceManifestBytes.toString('utf8'));
    const runtimeCodeFiles = Array.isArray(runtimeIdentity?.code?.files)
      ? runtimeIdentity.code.files
      : [];
    const runtimeCriticalPaths = runtimeCodeFiles.map(row => row?.path).sort();
    const runtimeFileIdentitiesValid = runtimeCodeFiles.every(row => (
      row
      && row.exists === true
      && Number.isInteger(row.mode)
      && row.mode >= 0
      && row.mode <= 0o777
      && Number.isInteger(row.size)
      && row.size >= 0
      && /^[0-9a-f]{64}$/.test(row.sha256 || '')
    ));
    const identityChecks = {
      schema: runtimeIdentity?.schema === 'automixer.backend_runtime_identity.v2',
      runtimeKind: runtimeIdentity?.runtime_kind === 'packaged',
      challenge: runtimeIdentity?.request_challenge_nonce
        === result.runtimeProbe.runtimeIdentityChallengeNonce,
      manifest: runtimeIdentity?.manifest_sha256
        === runtimeIdentityManifestSha256(runtimeIdentity || {}),
      sourceSchema: runtimeIdentity?.source?.schema === 'automixer.backend_runtime_source.v1',
      sourceObservation: runtimeIdentity?.source?.observation === 'packaged_build_manifest',
      sourceBuildProfile: Boolean(
        runtimeIdentity?.source?.head === result.buildProfile.source?.head
        && runtimeIdentity?.source?.tree === result.buildProfile.source?.tree
        && runtimeIdentity?.source?.worktree_dirty
          === result.buildProfile.source?.worktreeDirty
      ),
      criticalPaths: JSON.stringify(runtimeCriticalPaths)
        === JSON.stringify(EXPECTED_RUNTIME_CRITICAL_SOURCE_PATHS),
      criticalFiles: runtimeFileIdentitiesValid,
      sourceCodeDigest: Boolean(
        /^[0-9a-f]{64}$/.test(runtimeIdentity?.code?.files_sha256 || '')
        && runtimeIdentity?.source?.source_content_sha256
          === runtimeIdentity?.code?.files_sha256
      ),
      embeddedManifestSchema: runtimeSourceManifest?.schema
        === 'automixer.packaged_runtime_source_manifest.v1',
      embeddedManifestDigest: runtimeSourceManifest?.manifest_sha256
        === packagedSourceManifestSha256(runtimeSourceManifest || {}),
      embeddedComponents: Boolean(
        JSON.stringify(runtimeIdentity?.source) === JSON.stringify(runtimeSourceManifest?.source)
        && JSON.stringify(runtimeIdentity?.code) === JSON.stringify(runtimeSourceManifest?.code)
      ),
      packagedLaunchBinding: Boolean(
        runtimeIdentity?.launch_binding?.mode === 'packaged_build_manifest'
        && runtimeIdentity?.launch_binding?.bound_before_process_start === true
        && runtimeIdentity?.launch_binding?.proof_eligible === false
        && runtimeIdentity?.launch_binding?.packaged_manifest_schema
          === 'automixer.packaged_runtime_source_manifest.v1'
        && runtimeIdentity?.launch_binding?.packaged_manifest_name
          === PACKAGED_RUNTIME_SOURCE_MANIFEST_FILENAME
        && runtimeIdentity?.launch_binding?.packaged_manifest_size
          === runtimeSourceManifestBytes.length
        && runtimeIdentity?.launch_binding?.packaged_manifest_file_sha256
          === sha256Buffer(runtimeSourceManifestBytes)
        && runtimeIdentity?.launch_binding?.packaged_manifest_sha256
          === runtimeSourceManifest.manifest_sha256
        && runtimeIdentity?.launch_binding?.authenticated_launcher_bound === true
        && runtimeIdentity?.launch_binding?.launch_token_sha256
          === sha256Buffer(bootstrap.token)
      ),
      backendBinary: Boolean(
        runtimeIdentity?.backend?.path === 'backend/server'
        && runtimeIdentity?.backend?.exists === true
        && runtimeIdentity?.backend?.size === before.packageFiles.backend?.size
        && runtimeIdentity?.backend?.sha256 === before.packageFiles.backend?.sha256
      ),
      endpoint: Boolean(
        runtimeIdentity?.websocket?.host === '127.0.0.1'
        && runtimeIdentity?.websocket?.port === 8765
        && runtimeIdentity?.websocket?.loopback_only === true
      ),
      authorityDenied: runtimeIdentity?.read_only_identity === true
        && runtimeIdentity?.live_authority_granted === false
    };
    result.runtimeIdentityValidation = {
      valid: Object.values(identityChecks).every(Boolean),
      checks: identityChecks,
      identity: runtimeIdentity,
      embeddedManifest: {
        path: artifactRelativePath(
          buildArtifactRoot,
          originalFiles.runtimeSourceManifest,
          'Embedded runtime source manifest'
        ),
        pathBase: 'packaged_build_artifact_root',
        size: runtimeSourceManifestBytes.length,
        sha256: sha256Buffer(runtimeSourceManifestBytes),
        schema: runtimeSourceManifest.schema,
        manifestSha256: runtimeSourceManifest.manifest_sha256
      }
    };
    result.blockedOperationsPass = (
      JSON.stringify(result.runtimeProbe.blockedOperations)
      === JSON.stringify(['auto_connect', 'connect_wing', 'scan_mixers'])
    );

    await page.evaluate(() => {
      sessionStorage.setItem('automixer_ws_auth_token', 'stale-expired-token');
      localStorage.setItem('automixer_ws_url', 'ws://192.0.2.10:8765');
    });
    await page.reload({ waitUntil: 'domcontentloaded', timeout: options.timeoutMs });
    await page.waitForFunction(
      () => {
        const root = document.getElementById('root');
        const text = (root?.innerText || '').toLowerCase();
        return Boolean(
          root?.childElementCount
          && text.includes('automixer')
          && text.includes('tubeslave')
        );
      },
      undefined,
      { timeout: Math.min(options.timeoutMs, 20_000) }
    );
    result.reloadRecovery = await page.evaluate(() => {
      const bodyText = document.body.innerText;
      const normalizedBodyText = bodyText.toLowerCase();
      return {
        staleSessionTokenRemoved: sessionStorage.getItem('automixer_ws_auth_token') === null,
        secureBootstrapFailureShown: bodyText.includes(
          'Secure local backend bootstrap failed'
        ),
        offlineLockShown: normalizedBodyText.includes('offline lock')
          || normalizedBodyText.includes('offline smoke lock'),
        connectWingControlCount: document.querySelectorAll(
          '[aria-label="Connect WING read-only"]'
        ).length,
        startAudioControlCount: document.querySelectorAll(
          '[aria-label="Start read-only audio meters"]'
        ).length
      };
    });
    result.reloadRecovery.hardwareControlsHidden = (
      result.reloadRecovery.connectWingControlCount === 0
      && result.reloadRecovery.startAudioControlCount === 0
    );

    const processGroup = processGroupMembers(child.pid);
    result.processGroup = processGroup.map(row => ({
      pid: row.pid,
      role: row.command.includes('/backend/server')
        ? 'backend'
        : row.command.includes('Helper')
          ? 'electron_helper'
          : 'electron_main'
    }));
    result.runtimeIdentityValidation.checks.processPid = Boolean(
      Number.isInteger(runtimeIdentity?.process?.pid)
      && runtimeIdentity?.process?.instance_nonce === runtimeIdentity?.instance_nonce
      && runtimeIdentity?.process?.started_at_utc === runtimeIdentity?.started_at_utc
      && result.processGroup.some(row => (
        row.role === 'backend' && row.pid === runtimeIdentity.process.pid
      ))
    );
    result.runtimeIdentityValidation.valid = Object.values(
      result.runtimeIdentityValidation.checks
    ).every(Boolean);
    result.consoleErrors = consoleErrors;
  } catch (error) {
    result.error = redact(error?.stack || error);
  } finally {
    if (networkMonitor) result.network = networkMonitor.stop();
    if (child?.pid) {
      try { process.kill(-child.pid, 'SIGTERM'); } catch {}
    }
    if (browser) {
      result.browserCloseCompleted = await settleWithin(browser.close(), 2_000);
    }
    if (child?.pid) {
      await new Promise(resolve => setTimeout(resolve, 1_500));
      try { process.kill(-child.pid, 'SIGKILL'); } catch {}
    }
  }

  const after = {
    canonicalLog: await fileIdentity(
      canonicalLogPath,
      'artifacts/wing_telemetry/manual_write_gate/manual_write_events.jsonl',
      'repository_root'
    ),
    canonicalDecisionTrace: await fileIdentity(
      canonicalDecisionTracePath,
      'artifacts/wing_telemetry/manual_write_gate/decision_traces.jsonl',
      'repository_root'
    ),
    canonicalDecisionMemory: await fileIdentity(
      canonicalDecisionMemoryPath,
      'artifacts/wing_telemetry/manual_write_gate/decision_memory.jsonl',
      'repository_root'
    ),
    consoleWal: await fileIdentity(
      consoleWalPath,
      '.automixer/runtime/manual_write_wal.jsonl',
      'operator_home'
    ),
    appTree: {
      path: result.appPath,
      pathBase: 'packaged_build_artifact_root',
      sha256: hashTree(options.appPath)
    },
    backendBundle: {
      path: artifactRelativePath(
        buildArtifactRoot,
        backendBundlePath,
        'Packaged backend bundle'
      ),
      pathBase: 'packaged_build_artifact_root',
      sha256: hashTree(backendBundlePath)
    },
    packageFiles: Object.fromEntries(
      await Promise.all(Object.entries(originalFiles).map(async ([key, value]) => [
        key,
        await packagedFileIdentity(value, `Packaged file ${key}`)
      ]))
    )
  };
  result.after = after;
  try {
    run('/usr/bin/codesign', ['--verify', '--deep', '--strict', '--verbose=2', copiedAppPath]);
    const copiedTreeAfter = hashTree(copiedAppPath);
    result.launchCopy = {
      ...result.launchCopy,
      signatureVerifiedAfter: true,
      treeSha256After: copiedTreeAfter,
      unchanged: copiedTreeAfter === result.launchCopy?.treeSha256Before,
      stillMatchesSourceApp: copiedTreeAfter === after.appTree.sha256
    };
  } catch (error) {
    result.launchCopy = {
      ...result.launchCopy,
      signatureVerifiedAfter: false,
      unchanged: false,
      stillMatchesSourceApp: false,
      verificationError: redact(error?.message || error)
    };
  }
  result.canonicalLogUnchanged = JSON.stringify(before.canonicalLog) === JSON.stringify(after.canonicalLog);
  result.canonicalDecisionTraceUnchanged = (
    JSON.stringify(before.canonicalDecisionTrace)
    === JSON.stringify(after.canonicalDecisionTrace)
  );
  result.canonicalDecisionMemoryUnchanged = (
    JSON.stringify(before.canonicalDecisionMemory)
    === JSON.stringify(after.canonicalDecisionMemory)
  );
  result.consoleWalUnchanged = JSON.stringify(before.consoleWal) === JSON.stringify(after.consoleWal);
  result.packageFilesUnchanged = JSON.stringify(before.packageFiles) === JSON.stringify(after.packageFiles);
  result.appTreeUnchanged = before.appTree.sha256 === after.appTree.sha256;
  result.backendBundleUnchanged = (
    before.backendBundle.sha256 === after.backendBundle.sha256
  );
  result.temporaryArtifactFiles = listFilesRecursively(path.join(tempRoot, 'artifacts'));
  result.temporaryConsoleRuntimeFiles = listFilesRecursively(
    path.join(tempRoot, '.automixer', 'runtime')
  );
  result.tempWriteArtifactsAbsent = (
    result.temporaryArtifactFiles.length === 0
    && result.temporaryConsoleRuntimeFiles.length === 0
  );
  result.contactsWing = !(
    result.networkEnforcement?.nonLoopbackOutboundDenied === true
    && result.networkEnforcement?.preflightPassed === true
    && result.networkEnforcement?.backendBoundaryReported === true
    && result.networkEnforcement?.profileMatchesBackend === true
    && result.networkEnforcement?.probeMatrixMatchesBackend === true
    && result.network?.wingPortSockets?.length === 0
  );
  result.liveWrites = !(
    result.canonicalLogUnchanged
    && result.canonicalDecisionTraceUnchanged
    && result.canonicalDecisionMemoryUnchanged
    && result.consoleWalUnchanged
    && result.tempWriteArtifactsAbsent
  );
  result.liveAuthorityGranted = !(
    result.runtimeProbe?.mode?.packagedOfflineLock
    && result.runtimeProbe?.mode?.mixerTransportLocked
  );
  result.readOnly = result.liveWrites === false;
  result.stderrTail = stderr.slice(-30);
  result.pass = Boolean(
    !result.error
    && result.networkEnforcement?.mechanism === 'macos_sandbox_exec'
    && result.networkEnforcement?.scope === 'packaged_backend_process_tree'
    && result.networkEnforcement?.nonLoopbackOutboundDenied
    && result.networkEnforcement?.loopbackOutboundAllowed
    && result.networkEnforcement?.preflightPassed
    && result.networkEnforcement?.backendBoundaryReported
    && result.networkEnforcement?.profileMatchesBackend
    && result.networkEnforcement?.probeMatrixMatchesBackend
    && result.launchCopy?.signatureVerifiedBefore
    && result.launchCopy?.signatureVerifiedAfter
    && result.launchCopy?.matchesSourceApp
    && result.launchCopy?.unchanged
    && result.launchCopy?.stillMatchesSourceApp
    && result.runtime?.cdpReady
    && result.runtime?.backendReadyWithinBudget
    && result.bootstrap?.validShape
    && result.invalidToken?.rejected
    && result.runtimeProbe?.authenticated
    && result.runtimeIdentityValidation?.valid
    && result.runtimeProbe?.mode?.mode === 'analyze'
    && result.runtimeProbe?.mode?.packagedOfflineLock
    && result.runtimeProbe?.mode?.mixerTransportLocked
    && result.blockedOperationsPass
    && result.reloadRecovery?.staleSessionTokenRemoved
    && !result.reloadRecovery?.secureBootstrapFailureShown
    && result.reloadRecovery?.offlineLockShown
    && result.reloadRecovery?.hardwareControlsHidden
    && result.network?.externalConnections?.length === 0
    && result.network?.wingPortSockets?.length === 0
    && result.network?.monitorErrors?.length === 0
    && result.network?.sampleCount > 1
    && result.canonicalLogUnchanged
    && result.canonicalDecisionTraceUnchanged
    && result.canonicalDecisionMemoryUnchanged
    && result.consoleWalUnchanged
    && result.packageFilesUnchanged
    && result.appTreeUnchanged
    && result.backendBundleUnchanged
    && result.tempWriteArtifactsAbsent
    && result.contactsWing === false
    && result.liveWrites === false
    && result.liveAuthorityGranted === false
    && result.readOnly === true
    && result.buildProfile?.bindingValid === true
    && result.buildProfile?.appTreeMatches === true
  );

  const artifact = await writeArtifact(result, options.artifactRoot, runId);
  result.artifact = artifact;
  console.log(JSON.stringify({
    pass: result.pass,
    runId,
    artifactDirectory: artifact.runDirectory,
    invalidTokenRejected: result.invalidToken?.rejected === true,
    freshTokenAuthenticated: result.runtimeProbe?.authenticated === true,
    runtimeIdentityBound: result.runtimeIdentityValidation?.valid === true,
    backendReadyMs: result.runtime?.backendReadyMs ?? null,
    backendReadyWithinBudget: result.runtime?.backendReadyWithinBudget === true,
    blockedOperations: result.runtimeProbe?.blockedOperations || [],
    externalConnections: result.network?.externalConnections?.length ?? null,
    canonicalLogUnchanged: result.canonicalLogUnchanged,
    canonicalDecisionTraceUnchanged: result.canonicalDecisionTraceUnchanged,
    canonicalDecisionMemoryUnchanged: result.canonicalDecisionMemoryUnchanged,
    consoleWalUnchanged: result.consoleWalUnchanged,
    sourceBindingStrength: result.provenance.sourceBindingStrength,
    buildProfile: result.buildProfile.profile,
    releaseEligible: result.buildProfile.releaseEligible,
    error: result.error || null
  }, null, 2));

  if (!options.keepTemp) await fsp.rm(tempRoot, { recursive: true, force: true });
  else console.error(`Temporary launch copy retained at ${tempRoot}`);
  if (!result.pass) process.exitCode = 1;
}

if (path.resolve(process.argv[1] || '') === fileURLToPath(import.meta.url)) {
  main().catch(async error => {
    console.error(redact(error?.stack || error));
    process.exitCode = 1;
  });
}
