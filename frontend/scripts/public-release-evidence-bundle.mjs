#!/usr/bin/env node

import crypto from 'node:crypto';
import fs from 'node:fs';
import path from 'node:path';
import process from 'node:process';
import { fileURLToPath } from 'node:url';

import {
  artifactFileIdentity,
  identitiesEqual,
  pinReleaseEvidenceHelperRuntime,
  pinReleaseEvidenceHelperSources,
  publishNewArtifactFileNoFollow,
  readStableJsonFileNoFollow,
  RELEASE_EVIDENCE_HELPER_SOURCE_PATHS,
  removeArtifactFileNoFollow,
  resolveArtifactPath,
  runReleaseEvidenceGit,
  stableFlatDirectoryFileIdentitiesNoFollow,
  writeNewArtifactFileNoFollow
} from './release-evidence.mjs';
import { verifyRetainedReleaseManifestEvidence } from './verify-macos-release.mjs';

const scriptPath = fileURLToPath(import.meta.url);
const executingRepositoryRoot = path.resolve(path.dirname(scriptPath), '..', '..');
const PAYLOAD_DIRECTORY = 'public-release-evidence';
const PUBLIC_MANIFEST = 'public-release-evidence.json';
const CANDIDATE_MANIFEST = '.public-release-evidence.candidate';
const CP05_GATE_PATH = 'ops/m110/CP05_HARDWARE_GATE.json';
const CP05_EVIDENCE_PATH = 'Docs/reports/M110_CP05_SOFTWARE_VALIDATION.json';
const CP05_MARKER_PATHS = new Set([CP05_GATE_PATH, CP05_EVIDENCE_PATH]);
const CP05_CHECKS = [
  'focused_tests',
  'frontend_build_and_tests',
  'full_backend_suite',
  'independent_review',
  'no_wing_release_check',
  'released_development_lease',
  'static_scan'
].sort();
const ROUTE_ADDRESSES = [
  '/$syscfg/$serial',
  '/$syscfg/$firmware',
  '/io/out/USB/47/grp',
  '/io/out/USB/47/in',
  '/io/out/USB/48/grp',
  '/io/out/USB/48/in'
];
const ROUTE_EXPECTED = {
  '/io/out/USB/47/grp': 'MAIN',
  '/io/out/USB/47/in': 0,
  '/io/out/USB/48/grp': 'MAIN',
  '/io/out/USB/48/in': 1
};
const SOURCE_BLOBS = {
  runner: 'tools/wing_attended_soundcheck_runner.py',
  backendServer: 'backend/server.py',
  m006Contract: 'backend/contracts/m110_m006_restore_20260711.json'
};
const COMMITTED_SOURCE_PATHS = [
  ...Object.values(SOURCE_BLOBS),
  CP05_GATE_PATH,
  CP05_EVIDENCE_PATH
].sort();
const AUTHORITY_SOURCE_PATHS = [
  'frontend/public/electronSecurity.js',
  'frontend/scripts/notarytool-evidence.mjs',
  'frontend/scripts/public-release-evidence-bundle.mjs',
  'frontend/scripts/publish-new-artifact-helper.mjs',
  'frontend/scripts/release-evidence.mjs',
  'frontend/scripts/remove-artifact-helper.mjs',
  'frontend/scripts/replace-artifact-helper.mjs',
  'frontend/scripts/run-public-release-evidence-bundle',
  'frontend/scripts/stable-file-identity-helper.mjs',
  'frontend/scripts/stable-json-reader-helper.mjs',
  'frontend/scripts/stable-tree-identity-helper.mjs',
  'frontend/scripts/verify-macos-release.mjs',
  'frontend/scripts/write-new-artifact-helper.mjs'
].sort();
const SOURCE_MEMBERSHIP_PATHS = [
  ...new Set([...COMMITTED_SOURCE_PATHS, ...AUTHORITY_SOURCE_PATHS])
].sort();
const PAYLOAD_FILES = {
  summary: 'm110-execution-summary.json',
  route: 'm110-route-proof.json',
  cp05Gate: 'm110-cp05-gate.json',
  cp05Evidence: 'm110-cp05-software-validation.json',
  runnerSource: 'wing-attended-soundcheck-runner.py',
  backendSource: 'backend-server.py',
  m006Contract: 'm110-m006-restore-contract.json',
  sourceMembership: 'committed-source-membership.json'
};
const M006_CONTRACT_SHA256 = 'e75ced24084dfe1526c707122d837bcc51c1fd0829313faf6abce7f44840b274';
const M006_SOURCE_TRANSACTION_ID = 'soundcheck:normal:13c43cc98a3c487c';
const M006_SOURCE_EVENT_SHA256 = '0ce7efb80f19e38f0a5e926467f77845b05ebee9820c78e7ec362b2443128d51';
const M006_STAGES = ['foundation_level', 'tonal_dynamics', 'spatial_balance'];
const M006_STAGE_OPERATION_COUNTS = [23, 14, 14];
const M006_BUNDLE_KIND = 'm006_exact_preimage_restore_bundle';
const RESET_BUNDLE_KIND = 'maintenance_dsp_reset_bundle';
const RESET_SUFFIXES = [
  ['eq/on', 'eq_enabled'],
  ['flt/lc', 'hpf_enabled'],
  ['gate/on', 'gate_enabled'],
  ['dyn/on', 'compressor_enabled']
];

function canonicalJson(value) {
  if (Array.isArray(value)) return `[${value.map(canonicalJson).join(',')}]`;
  if (value && typeof value === 'object') {
    return `{${Object.keys(value).sort().map(key => (
      `${JSON.stringify(key)}:${canonicalJson(value[key])}`
    )).join(',')}}`;
  }
  return JSON.stringify(value);
}

function pythonCanonicalJsonString(value) {
  for (let index = 0; index < value.length; index += 1) {
    const code = value.charCodeAt(index);
    if (code >= 0xd800 && code <= 0xdbff) {
      const next = value.charCodeAt(index + 1);
      assertCondition(
        next >= 0xdc00 && next <= 0xdfff,
        'Python canonical JSON string contains an unpaired high surrogate'
      );
      index += 1;
    } else {
      assertCondition(
        code < 0xdc00 || code > 0xdfff,
        'Python canonical JSON string contains an unpaired low surrogate'
      );
    }
  }
  return JSON.stringify(value);
}

function pythonCanonicalJsonFloat(value) {
  if (Object.is(value, -0)) return '-0.0';
  if (value === 0) return '0.0';
  assertCondition(Number.isFinite(value), 'Python canonical JSON float is not finite');

  const sign = value < 0 ? '-' : '';
  const serialized = Math.abs(value).toString();
  let digits;
  let scientificExponent;
  if (serialized.includes('e')) {
    const [coefficient, exponent] = serialized.split('e');
    digits = coefficient.replace('.', '');
    scientificExponent = Number(exponent);
  } else if (serialized.includes('.')) {
    const [whole, fraction] = serialized.split('.');
    if (whole !== '0') {
      digits = `${whole}${fraction}`;
      scientificExponent = whole.length - 1;
    } else {
      const firstSignificant = fraction.search(/[1-9]/);
      assertCondition(firstSignificant >= 0, 'Python canonical JSON float has no significant digit');
      digits = fraction.slice(firstSignificant);
      scientificExponent = -(firstSignificant + 1);
    }
  } else {
    digits = serialized;
    scientificExponent = digits.length - 1;
  }
  digits = digits.replace(/0+$/, '');

  if (scientificExponent < -4 || scientificExponent >= 16) {
    const coefficient = digits.length === 1
      ? digits
      : `${digits[0]}.${digits.slice(1)}`;
    const exponentSign = scientificExponent < 0 ? '-' : '+';
    const exponentMagnitude = String(Math.abs(scientificExponent)).padStart(2, '0');
    return `${sign}${coefficient}e${exponentSign}${exponentMagnitude}`;
  }

  const decimalPosition = scientificExponent + 1;
  if (decimalPosition <= 0) {
    return `${sign}0.${'0'.repeat(-decimalPosition)}${digits}`;
  }
  if (decimalPosition >= digits.length) {
    return `${sign}${digits}${'0'.repeat(decimalPosition - digits.length)}.0`;
  }
  return `${sign}${digits.slice(0, decimalPosition)}.${digits.slice(decimalPosition)}`;
}

function parseCompactCanonicalJsonBytes(
  bytes,
  label,
  { pythonWriter = false, finalNewline = pythonWriter } = {}
) {
  assertCondition(Buffer.isBuffer(bytes) && bytes.length > 0, `${label} bytes are required`);
  let text = new TextDecoder('utf-8', { fatal: true }).decode(bytes);
  if (finalNewline) {
    assertCondition(
      text.endsWith('\n') && !text.slice(0, -1).includes('\n'),
      `${label} must contain exactly one final newline`
    );
    text = text.slice(0, -1);
  }
  let index = 0;

  const parseString = () => {
    const start = index;
    assertCondition(text[index] === '"', `${label} string is malformed`);
    index += 1;
    while (index < text.length) {
      const code = text.charCodeAt(index);
      if (text[index] === '"') {
        index += 1;
        const token = text.slice(start, index);
        const value = JSON.parse(token);
        assertCondition(
          (pythonWriter ? pythonCanonicalJsonString(value) : JSON.stringify(value)) === token,
          `${label} string is not canonical`
        );
        return value;
      }
      if (text[index] === '\\') {
        index += 1;
        const escape = text[index];
        assertCondition(
          ['"', '\\', '/', 'b', 'f', 'n', 'r', 't', 'u'].includes(escape),
          `${label} string escape is invalid`
        );
        if (escape === 'u') {
          assertCondition(
            /^[0-9a-f]{4}$/.test(text.slice(index + 1, index + 5)),
            `${label} Unicode escape is not canonical`
          );
          index += 4;
        }
        index += 1;
        continue;
      }
      assertCondition(code >= 0x20, `${label} string contains a control character`);
      index += 1;
    }
    throw new Error(`${label} string is unterminated`);
  };

  const parseNumber = () => {
    const token = text.slice(index).match(pythonWriter
      ? /^-?(?:0|[1-9]\d*)(?:\.\d+)?(?:e[+-]\d+)?/
      : /^-?(?:0|[1-9]\d*)(?:\.\d+)?/)?.[0];
    assertCondition(Boolean(token), `${label} number is malformed`);
    const next = text[index + token.length];
    assertCondition(
      next === undefined || [',', ']', '}'].includes(next),
      `${label} number has a non-canonical suffix`
    );
    const coefficient = token.split('e')[0];
    if (coefficient.includes('.')) {
      const fraction = coefficient.split('.')[1];
      assertCondition(
        fraction === '0' || !fraction.endsWith('0'),
        `${label} number has redundant fractional zeros`
      );
    }
    const value = Number(token);
    assertCondition(Number.isFinite(value), `${label} number is not finite`);
    if (pythonWriter) {
      const exponent = token.match(/e([+-])(\d+)$/);
      if (exponent) {
        const magnitude = Number(exponent[2]);
        const signedExponent = exponent[1] === '-' ? -magnitude : magnitude;
        assertCondition(
          value !== 0
          && exponent[2] === String(magnitude).padStart(2, '0')
          && (signedExponent <= -5 || signedExponent >= 16)
          && !coefficient.endsWith('.0'),
          `${label} number exponent is not canonical Python JSON`
        );
      } else if (!coefficient.includes('.')) {
        assertCondition(
          !Object.is(value, -0) && Number.isSafeInteger(value),
          `${label} integer is unsafe or negative zero`
        );
      }
      if (coefficient.includes('.') || exponent) {
        assertCondition(
          pythonCanonicalJsonFloat(value) === token,
          `${label} float is not the exact canonical Python JSON representation`
        );
      }
    } else {
      assertCondition(!Object.is(value, -0), `${label} number must not be negative zero`);
      assertCondition(
        JSON.stringify(value) === token,
        `${label} number is not the exact canonical JSON representation`
      );
    }
    index += token.length;
    return value;
  };

  let parseValue;
  const parseArray = () => {
    const values = [];
    index += 1;
    if (text[index] === ']') {
      index += 1;
      return values;
    }
    while (true) {
      values.push(parseValue());
      if (text[index] === ']') {
        index += 1;
        return values;
      }
      assertCondition(text[index] === ',', `${label} array separator is invalid`);
      index += 1;
    }
  };
  const parseObject = captureSlices => {
    const value = {};
    const slices = {};
    let previousKey = null;
    index += 1;
    if (text[index] === '}') {
      index += 1;
      return captureSlices ? { value, slices } : value;
    }
    while (true) {
      const key = parseString();
      assertCondition(
        previousKey === null || previousKey < key,
        `${label} object keys are duplicated or not sorted`
      );
      previousKey = key;
      assertCondition(text[index] === ':', `${label} object separator is invalid`);
      index += 1;
      const start = index;
      value[key] = parseValue();
      if (captureSlices) slices[key] = text.slice(start, index);
      if (text[index] === '}') {
        index += 1;
        return captureSlices ? { value, slices } : value;
      }
      assertCondition(text[index] === ',', `${label} object separator is invalid`);
      index += 1;
    }
  };
  parseValue = () => {
    const character = text[index];
    if (character === '"') return parseString();
    if (character === '[') return parseArray();
    if (character === '{') return parseObject(false);
    for (const [token, value] of [['true', true], ['false', false], ['null', null]]) {
      if (text.startsWith(token, index)) {
        index += token.length;
        return value;
      }
    }
    return parseNumber();
  };

  assertCondition(text[index] === '{', `${label} root must be an object`);
  const parsed = parseObject(true);
  assertCondition(index === text.length, `${label} has trailing or non-canonical bytes`);
  return parsed;
}

function assertPythonCanonicalJsonBytes(value, bytes, label) {
  const parsed = parseCompactCanonicalJsonBytes(bytes, label, {
    pythonWriter: true
  });
  assertCondition(
    deepEqual(parsed.value, value),
    `${label} parsed value does not match its retained bytes`
  );
}

function sha256Bytes(bytes) {
  return crypto.createHash('sha256').update(bytes).digest('hex');
}

function canonicalSha256(value) {
  return sha256Bytes(Buffer.from(canonicalJson(value), 'utf8'));
}

function deepEqual(left, right) {
  return canonicalJson(left) === canonicalJson(right);
}

function assertCondition(value, message) {
  if (!value) throw new Error(message);
}

function assertExactKeys(value, keys, label) {
  assertCondition(
    value && typeof value === 'object' && !Array.isArray(value),
    `${label} must be an object`
  );
  assertCondition(
    deepEqual(Object.keys(value).sort(), [...keys].sort()),
    `${label} keys are not exact`
  );
}

function parseIso(value, label) {
  const text = String(value || '');
  const epoch = Date.parse(text);
  const normalized = Number.isFinite(epoch) ? new Date(epoch).toISOString() : '';
  assertCondition(
    /^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{3})?Z$/.test(text)
    && (normalized === text || normalized.replace('.000Z', 'Z') === text),
    `${label} must be a canonical ISO timestamp`
  );
  return epoch / 1000;
}

function sha256Prefixed(value) {
  return /^sha256:[0-9a-f]{64}$/.test(String(value || ''));
}

function bareSha256(value) {
  return /^[0-9a-f]{64}$/.test(String(value || ''));
}

function sourceObjectId(value) {
  return /^[0-9a-f]{40,64}$/.test(String(value || ''));
}

function runGit(repoRoot, args, { maxBuffer = 16 * 1024 * 1024 } = {}) {
  return runReleaseEvidenceGit(repoRoot, args, { maxBuffer });
}

function sourceSnapshot(repoRoot) {
  const head = runGit(repoRoot, ['rev-parse', 'HEAD']).toString('utf8').trim();
  const tree = runGit(repoRoot, ['rev-parse', `${head}^{tree}`]).toString('utf8').trim();
  const status = runGit(repoRoot, ['status', '--porcelain=v1', '-uall'])
    .toString('utf8')
    .trim();
  const confirmedHead = runGit(repoRoot, ['rev-parse', 'HEAD']).toString('utf8').trim();
  assertCondition(sourceObjectId(head) && sourceObjectId(tree), 'Git source identity is invalid');
  assertCondition(confirmedHead === head, 'Git HEAD changed during source identity capture');
  assertCondition(!status, 'Public-release evidence requires a completely clean checkout');
  return { head, tree, cleanCheckout: true };
}

let releaseEvidenceAuthorityPin = null;

function pinReleaseEvidenceAuthorityHelpers(repoRoot, sourceTree) {
  assertCondition(sourceObjectId(sourceTree), 'Release-evidence helper source tree is invalid');
  assertCondition(
    RELEASE_EVIDENCE_HELPER_SOURCE_PATHS.every(sourcePath => (
      AUTHORITY_SOURCE_PATHS.includes(sourcePath)
    )),
    'Release-evidence helper source is outside the authority closure'
  );
  const sources = {};
  for (const sourcePath of RELEASE_EVIDENCE_HELPER_SOURCE_PATHS) {
    const objectId = runGit(repoRoot, ['rev-parse', `${sourceTree}:${sourcePath}`])
      .toString('utf8')
      .trim();
    assertCondition(sourceObjectId(objectId), `Release-evidence helper blob is invalid: ${sourcePath}`);
    sources[sourcePath] = runGit(repoRoot, ['cat-file', 'blob', objectId], {
      maxBuffer: 1024 * 1024
    });
  }
  const identities = pinReleaseEvidenceHelperSources(sources);
  releaseEvidenceAuthorityPin = Object.freeze({
    sourceTree,
    identities
  });
  return releaseEvidenceAuthorityPin;
}

function assertExecutingAuthorityMatchesSource(repoRoot, source) {
  assertCondition(
    sourceObjectId(source?.head) && sourceObjectId(source?.tree),
    'Authority source identity is invalid'
  );
  const taggedRows = runGit(repoRoot, [
    'ls-files',
    '-v',
    '--',
    ...AUTHORITY_SOURCE_PATHS
  ]).toString('utf8').trim().split('\n').filter(Boolean);
  const tagged = taggedRows.map(row => ({
    tag: row.slice(0, 1),
    path: row.slice(2)
  }));
  assertCondition(
    tagged.length === AUTHORITY_SOURCE_PATHS.length
    && deepEqual(tagged.map(row => row.path), AUTHORITY_SOURCE_PATHS)
    && tagged.every(row => row.tag === 'H'),
    'Release authority files are missing or carry hidden Git index flags'
  );
  const rows = [];
  for (const relativePath of AUTHORITY_SOURCE_PATHS) {
    const treeRow = runGit(repoRoot, ['ls-tree', source.tree, '--', relativePath])
      .toString('utf8')
      .trim();
    const match = treeRow.match(/^([0-7]{6}) blob ([0-9a-f]{40,64})\t(.+)$/);
    assertCondition(
      match
      && ['100644', '100755'].includes(match[1])
      && match[3] === relativePath,
      `Release authority path is not one regular Git blob: ${relativePath}`
    );
    const expectedBytes = runGit(repoRoot, ['cat-file', 'blob', match[2]], {
      maxBuffer: 16 * 1024 * 1024
    });
    const currentPath = path.join(repoRoot, relativePath);
    const currentIdentity = artifactFileIdentity(
      repoRoot,
      currentPath,
      `Executing release authority ${relativePath}`
    );
    const current = readPayloadFile(
      repoRoot,
      currentIdentity,
      `Executing release authority ${relativePath}`
    );
    const executableBits = Number(
      fs.lstatSync(current.path, { bigint: true }).mode & 0o111n
    );
    assertCondition(
      current.bytes.equals(expectedBytes)
      && (
        (match[1] === '100755' && executableBits === 0o111)
        || (match[1] === '100644' && executableBits === 0)
      ),
      `Executing release authority bytes or executable mode differ from selected Git blob: ${relativePath}`
    );
    rows.push({
      path: relativePath,
      mode: match[1],
      blobObjectId: match[2],
      size: current.identity.size,
      sha256: current.identity.sha256
    });
  }
  return {
    schema: 'automixer.public_release_authority_closure.v1',
    sourceHead: source.head,
    sourceTree: source.tree,
    pathCount: rows.length,
    closureSha256: `sha256:${canonicalSha256({
      schema: 'automixer.public_release_authority_closure.v1',
      rows
    })}`
  };
}

function assertProductionRuntimeEnvironment() {
  const requiredEnvironmentNames = [
    'AUTOMIXER_PUBLIC_RELEASE_EVIDENCE_CLEAN_ENV',
    'HOME',
    'LANG',
    'LC_ALL',
    'PATH',
    'TMPDIR'
  ];
  const allowedEnvironmentNames = new Set([
    ...requiredEnvironmentNames,
    // macOS injects this process-local locale hint even through `env -i`.
    '__CF_USER_TEXT_ENCODING'
  ]);
  const environmentNames = Object.keys(process.env);
  const unexpectedEnvironmentNames = environmentNames.filter(
    name => !allowedEnvironmentNames.has(name)
  );
  const missingEnvironmentNames = requiredEnvironmentNames.filter(
    name => !Object.hasOwn(process.env, name)
  );
  const executable = path.resolve(process.execPath || '');
  const executableStat = fs.lstatSync(executable);
  const expectedPath = `${path.dirname(executable)}:/usr/bin:/bin:/usr/sbin:/sbin`;
  let executingFromFrontendRoot = false;
  try {
    executingFromFrontendRoot = fs.realpathSync(process.cwd()) === fs.realpathSync(
      path.join(executingRepositoryRoot, 'frontend')
    );
  } catch {
    executingFromFrontendRoot = false;
  }
  assertCondition(
    process.env.AUTOMIXER_PUBLIC_RELEASE_EVIDENCE_CLEAN_ENV === '1'
    && process.version === 'v22.22.3'
    && process.execArgv.length === 0
    && unexpectedEnvironmentNames.length === 0
    && missingEnvironmentNames.length === 0
    && process.env.HOME === '/var/empty'
    && process.env.TMPDIR === '/tmp'
    && process.env.LANG === 'C'
    && process.env.LC_ALL === 'C'
    && process.env.PATH === expectedPath
    && executingFromFrontendRoot
    && path.isAbsolute(process.execPath || '')
    && !executableStat.isSymbolicLink()
    && executableStat.isFile(),
    'Production public-release evidence requires the fixed clean Node runtime'
  );
  pinReleaseEvidenceHelperRuntime();
}

export function computeCp05SourceBinding(repoRoot, sourceTree = 'HEAD') {
  assertCondition(
    sourceTree === 'HEAD' || sourceObjectId(sourceTree),
    'CP05 source tree reference is invalid'
  );
  const raw = runGit(repoRoot, ['ls-tree', '-r', '-z', '--full-tree', sourceTree], {
    maxBuffer: 16 * 1024 * 1024
  });
  const rows = [];
  const decoded = new TextDecoder('utf-8', { fatal: true }).decode(raw);
  for (const entry of decoded.split('\0')) {
    if (!entry) continue;
    const tab = entry.indexOf('\t');
    assertCondition(tab > 0, 'CP05 Git tree entry is malformed');
    const [mode, objectType, objectId] = entry.slice(0, tab).split(' ');
    const relativePath = entry.slice(tab + 1);
    assertCondition(
      /^[0-7]{6}$/.test(mode)
      && ['blob', 'commit'].includes(objectType)
      && sourceObjectId(objectId)
      && relativePath
      && !relativePath.startsWith('/')
      && !relativePath.includes('\\')
      && !relativePath.split('/').some(part => !part || part === '.' || part === '..'),
      'CP05 Git tree entry is invalid'
    );
    if (CP05_MARKER_PATHS.has(relativePath)) continue;
    rows.push({ mode, object_id: objectId, object_type: objectType, path: relativePath });
  }
  assertCondition(rows.length > 0, 'CP05 Git source tree is empty');
  const payload = {
    schema: 'automixer.m110_cp05_git_tree_binding.v1',
    entries: rows
  };
  return {
    schema: payload.schema,
    sha256: `sha256:${canonicalSha256(payload)}`,
    tracked_entry_count: rows.length
  };
}

function assertCanonicalJsonBytes(value, bytes, label) {
  const canonical = Buffer.from(`${canonicalJson(value)}\n`, 'utf8');
  assertCondition(bytes.equals(canonical), `${label} is not canonical JSON plus one newline`);
}

export function validateCp05Markers({
  gate,
  gateBytes,
  evidence,
  evidenceBytes,
  sourceBinding
}) {
  assertCanonicalJsonBytes(gate, gateBytes, 'CP05 hardware gate');
  assertCanonicalJsonBytes(evidence, evidenceBytes, 'CP05 software evidence');
  assertExactKeys(
    gate,
    ['schema', 'checkpoint', 'hardware_enabled', 'software_evidence_sha256'],
    'CP05 hardware gate'
  );
  assertCondition(
    gate.schema === 'automixer.m110_cp05_hardware_gate.v1'
    && gate.checkpoint === 'M110-CP05'
    && gate.hardware_enabled === true
    && sha256Prefixed(gate.software_evidence_sha256),
    'CP05 hardware gate is closed'
  );
  assertExactKeys(
    evidence,
    ['schema', 'status', 'checks', 'source_binding_sha256'],
    'CP05 software evidence'
  );
  assertCondition(
    evidence.schema === 'automixer.m110_cp05_software_validation.v1'
    && evidence.status === 'passed'
    && deepEqual(Object.keys(evidence.checks || {}).sort(), CP05_CHECKS)
    && CP05_CHECKS.every(name => evidence.checks[name] === true)
    && sha256Prefixed(evidence.source_binding_sha256),
    'CP05 software evidence is incomplete'
  );
  const evidenceSha256 = `sha256:${sha256Bytes(evidenceBytes)}`;
  assertCondition(
    gate.software_evidence_sha256 === evidenceSha256,
    'CP05 software evidence byte digest mismatch'
  );
  assertCondition(
    sourceBinding?.schema === 'automixer.m110_cp05_git_tree_binding.v1'
    && sha256Prefixed(sourceBinding.sha256)
    && Number.isSafeInteger(sourceBinding.tracked_entry_count)
    && sourceBinding.tracked_entry_count > 0
    && evidence.source_binding_sha256 === sourceBinding.sha256,
    'CP05 source binding does not match the selected source tree'
  );
  return {
    verified: true,
    reason: 'm110_cp05_hardware_gate_verified',
    software_evidence_sha256: evidenceSha256,
    source_binding_sha256: sourceBinding.sha256,
    source_binding_tracked_entry_count: sourceBinding.tracked_entry_count
  };
}

function validateRoutePass(pass, console, label) {
  assertCondition(
    pass?.verified === true
    && pass.reason === 'fixed_master_route_pass_verified'
    && pass.confirmed_round_trips_required === 2
    && Array.isArray(pass.errors)
    && pass.errors.length === 0
    && deepEqual(pass.addresses, ROUTE_ADDRESSES),
    `${label} is incomplete`
  );
  const expectedValues = {
    '/$syscfg/$serial': console.serial,
    '/$syscfg/$firmware': console.firmware,
    ...ROUTE_EXPECTED
  };
  assertCondition(deepEqual(pass.values, expectedValues), `${label} values drifted`);
  for (const address of ROUTE_ADDRESSES) {
    const row = pass.readbacks?.[address];
    assertCondition(
      row?.address === address
      && row.status === 'received'
      && row.confirmed_round_trips === 2
      && row.value === expectedValues[address],
      `${label} readback is invalid: ${address}`
    );
  }
}

function validateRouteArtifact(
  routeProof,
  routeBytes,
  routeIdentity,
  routeEnvelope,
  source,
  summaryEpoch
) {
  assertCondition(
    Buffer.isBuffer(routeBytes) && routeBytes.length > 0 && routeBytes.length <= 64 * 1024,
    'M110 route proof exceeds the backend artifact byte limit'
  );
  const parsedRoute = parseCompactCanonicalJsonBytes(routeBytes, 'M110 route proof', {
    pythonWriter: true,
    finalNewline: false
  });
  assertCondition(
    deepEqual(parsedRoute.value, routeProof)
    && routeBytes.length === routeIdentity?.size
    && sha256Bytes(routeBytes) === routeIdentity?.sha256,
    'M110 route proof raw bytes do not match the stable retained identity'
  );
  const evidenceDigest = `sha256:${sha256Bytes(
    Buffer.from(parsedRoute.slices.evidence || '', 'utf8')
  )}`;
  assertCondition(
    evidenceDigest === routeProof?.evidence_sha256,
    'M110 route proof evidence digest does not match the canonical evidence bytes'
  );
  assertExactKeys(
    routeProof,
    [
      'accepted',
      'evidence',
      'evidence_sha256',
      'expires_at_epoch',
      'issued_at_epoch',
      'issued_at_utc',
      'proof_id',
      'schema'
    ],
    'M110 route proof'
  );
  const issued = routeProof.issued_at_epoch;
  const expires = routeProof.expires_at_epoch;
  assertCondition(
    routeProof.schema === 'automixer.wing_master_capture_route_proof.v1'
    && routeProof.accepted === true
    && /^[0-9a-f]{32}$/.test(String(routeProof.proof_id || ''))
    && sha256Prefixed(routeProof.evidence_sha256)
    && Number.isFinite(issued)
    && Number.isFinite(expires)
    && issued < expires
    && expires - issued <= 1800
    && Math.abs(
      parseIso(routeProof.issued_at_utc, 'Route issued_at_utc') - Math.floor(issued)
    ) < 0.001
    && issued <= summaryEpoch
    && summaryEpoch <= expires,
    'M110 route proof envelope or historical TTL is invalid'
  );
  const evidence = routeProof.evidence;
  assertExactKeys(evidence, ['request', 'runtime', 'console', 'route', 'capture', 'safety'], 'Route evidence');
  const runtime = evidence.runtime;
  assertCondition(
    runtime?.head === source.head
    && runtime.tree === source.tree
    && runtime.tracked_tree_clean === true
    && runtime.identity_schema === 'automixer.backend_runtime_identity.v2'
    && bareSha256(runtime.manifest_sha256)
    && bareSha256(runtime.launch_manifest_sha256)
    && typeof runtime.instance_nonce === 'string'
    && runtime.instance_nonce.length >= 16,
    'Route runtime does not match the selected release source'
  );
  const console = evidence.console;
  assertCondition(
    typeof console?.serial === 'string'
    && /^[A-Z0-9]{8,32}$/.test(console.serial)
    && /^3\.0\.5(?:[-+ ][A-Za-z0-9._+-]+(?::release)?)?$/.test(String(console.firmware || ''))
    && console.client_type === 'EnhancedOSCClient',
    'Route console identity is invalid'
  );
  const route = evidence.route;
  assertCondition(
    route?.consistent === true
    && route.complete_passes_required === 2
    && route.confirmed_round_trips_required === 2
    && route.logical_query_count === 12
    && deepEqual(route.addresses, ROUTE_ADDRESSES)
    && deepEqual(route.expected, ROUTE_EXPECTED),
    'Master 47/48 route evidence is incomplete'
  );
  validateRoutePass(route.pass_1, console, 'Route pass 1');
  validateRoutePass(route.pass_2, console, 'Route pass 2');
  const capture = evidence.capture;
  const hardware = capture?.hardware_identity;
  const coreaudio = hardware?.coreaudio;
  const usb = hardware?.usb;
  assertExactKeys(
    hardware,
    [
      'binding_sha256',
      'console_serial',
      'coreaudio',
      'device_index',
      'device_name',
      'hardware_uid',
      'reason',
      'schema',
      'serial_binding',
      'usb',
      'verified'
    ],
    'WING capture hardware identity'
  );
  assertExactKeys(
    coreaudio,
    ['input_channels', 'manufacturer', 'output_channels', 'sample_rate_hz', 'transport'],
    'WING CoreAudio identity'
  );
  assertExactKeys(
    usb,
    ['location_id', 'product_id', 'serial', 'vendor_id'],
    'WING USB identity'
  );
  const usbSerialBinding = usb.serial === console.serial
    ? 'exact'
    : new RegExp(`^${console.serial.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}[0-9]{3}$`).test(
      String(usb.serial || '')
    )
      ? 'chassis_serial_plus_three_digit_usb_interface'
      : null;
  const expectedHardwareUid = Number.isSafeInteger(usb.location_id) && usb.location_id >= 0
    ? `usb:${usb.vendor_id.toString(16).padStart(4, '0')}:`
      + `${usb.product_id.toString(16).padStart(4, '0')}:`
      + `${usb.serial}:${usb.location_id.toString(16).padStart(8, '0')}`
    : null;
  const hardwareBinding = {
    schema: hardware.schema,
    device_name: hardware.device_name,
    device_index: hardware.device_index,
    console_serial: hardware.console_serial,
    coreaudio,
    usb,
    serial_binding: hardware.serial_binding
  };
  const requiredSamples = capture?.required_fresh_sample_count;
  const peak47 = capture?.sample_peaks_dbfs?.['47'];
  const peak48 = capture?.sample_peaks_dbfs?.['48'];
  const allowance = capture?.true_peak_allowance_db;
  const computedPeak = Math.max(peak47, peak48) + allowance;
  assertCondition(
    capture?.verified === true
    && capture.reason === 'fresh_real_master_capture_verified'
    && capture.source_type === 'sounddevice'
    && typeof capture.device_name === 'string'
    && capture.device_name.length > 0
    && Number.isSafeInteger(capture.device_index)
    && capture.device_index >= 0
    && Number.isSafeInteger(capture.capture_instance_id)
    && capture.capture_instance_id > 0
    && Number.isSafeInteger(capture.capture_stream_instance_id)
    && capture.capture_stream_instance_id > 0
    && capture.sample_rate_hz === 48000
    && capture.channel_count === 48
    && deepEqual(capture.master_channels, [47, 48])
    && Number.isSafeInteger(requiredSamples)
    && requiredSamples === 12000
    && Number.isSafeInteger(capture.sample_counts?.['47'])
    && Number.isSafeInteger(capture.sample_counts?.['48'])
    && capture.sample_counts['47'] === requiredSamples
    && capture.sample_counts['48'] === requiredSamples
    && Number.isSafeInteger(capture.captured_sample_count_before)
    && capture.captured_sample_count_before >= 0
    && Number.isSafeInteger(capture.captured_sample_count_snapshot)
    && capture.captured_sample_count_snapshot
      >= capture.captured_sample_count_before + requiredSamples
    && Number.isSafeInteger(capture.captured_sample_count_after)
    && capture.captured_sample_count_after >= capture.captured_sample_count_snapshot
    && capture.captured_sample_count_after - capture.captured_sample_count_before >= requiredSamples
    && Number.isFinite(peak47)
    && Number.isFinite(peak48)
    && peak47 > -90
    && peak48 > -90
    && allowance === 3
    && Math.abs(computedPeak - capture.conservative_true_peak_dbtp) < 1e-6
    && capture.headroom_ceiling_dbtp === -1
    && capture.conservative_true_peak_dbtp <= capture.headroom_ceiling_dbtp
    && hardware?.schema === 'automixer.wing_capture_hardware_identity.v1'
    && hardware.verified === true
    && hardware.reason === 'physical_wing_usb_capture_bound_to_console'
    && hardware.console_serial === console.serial
    && hardware.device_name === capture.device_name
    && hardware.device_index === capture.device_index
    && hardware.serial_binding === usbSerialBinding
    && hardware.hardware_uid === expectedHardwareUid
    && hardware.binding_sha256 === canonicalSha256(hardwareBinding)
    && coreaudio?.manufacturer === 'BEHRINGER'
    && coreaudio.transport === 'usb'
    && coreaudio.input_channels === 48
    && coreaudio.output_channels === 48
    && coreaudio.sample_rate_hz === 48000
    && usb?.vendor_id === 5015
    && usb.product_id === 1291
    && typeof usb.serial === 'string'
    && usbSerialBinding !== null
    && Number.isSafeInteger(usb.location_id)
    && usb.location_id >= 0,
    'Fresh physical WING capture or headroom evidence is invalid'
  );
  const safety = evidence.safety;
  assertCondition(
    safety?.authenticated_control_session === true
    && safety.console_writer_scope_acquired === true
    && safety.console_mutation_write_attempt_count === 0
    && safety.operator_mode === 'analyze'
    && safety.observation_only === true
    && safety.live_authority_granted === false
    && safety.readiness_closure_allowed === false
    && safety.normal_live_allowlist_changed === false
    && safety.emergency_generation_stable === true
    && safety.recovery_transaction_count === 0
    && safety.wal_integrity_ok === true,
    'Route safety evidence is invalid'
  );
  const routeResult = routeEnvelope?.route_result;
  const artifact = routeResult?.artifact;
  const verification = routeEnvelope?.artifact_verification;
  assertCondition(
    routeEnvelope?.verified === true
    && routeEnvelope.reason === 'persistent_proof_owned_master_route_session_established'
    && routeEnvelope.same_websocket_retained === true
    && routeEnvelope.backend_process_retained === true
    && routeEnvelope.live_authority_granted === false
    && routeEnvelope.readiness_closure_allowed === false
    && routeResult?.type === 'wing_master_route_proof_result'
    && routeResult.status === 'issued'
    && routeResult.success === true
    && routeResult.accepted === true
    && routeResult.reason === 'process_private_master_capture_route_authority_issued'
    && routeResult.proof_id === routeProof.proof_id
    && routeResult.issued_at_epoch === issued
    && routeResult.expires_at_epoch === expires
    && routeResult.console_serial === console.serial
    && routeResult.firmware_version === console.firmware
    && routeResult.capture_device === capture.device_name
    && routeResult.capture_device_index === capture.device_index
    && routeResult.capture_hardware_uid === hardware.hardware_uid
    && routeResult.capture_usb_serial === usb.serial
    && routeResult.conservative_true_peak_dbtp === capture.conservative_true_peak_dbtp
    && deepEqual(routeResult.master_channels, [47, 48])
    && routeResult.console_mutation_write_attempt_count === 0
    && routeResult.live_authority_granted === false
    && routeResult.readiness_closure_allowed === false
    && artifact?.schema === routeProof.schema
    && artifact.size === routeIdentity.size
    && artifact.sha256 === routeIdentity.sha256
    && artifact.evidence_sha256 === routeProof.evidence_sha256
    && path.basename(String(artifact.path || '')) === `${routeProof.proof_id}.json`
    && verification?.verified === true
    && verification.reason === 'runner_independently_verified_master_route_artifact'
    && verification.request_binding_verified === true
    && verification.hardware_binding_verified === true
    && verification.live_authority_granted === false
    && verification.proof_id === routeProof.proof_id
    && verification.sha256 === routeIdentity.sha256
    && verification.evidence_sha256 === routeProof.evidence_sha256,
    'Runner route result does not bind the copied raw route proof'
  );
  return {
    proofId: routeProof.proof_id,
    issuedAtEpoch: issued,
    expiresAtEpoch: expires,
    consoleSerial: console.serial,
    firmware: console.firmware,
    captureDevice: capture.device_name,
    captureHardwareUid: hardware.hardware_uid,
    conservativeTruePeakDbtp: capture.conservative_true_peak_dbtp,
    headroomCeilingDbtp: capture.headroom_ceiling_dbtp,
    semanticSha256: `sha256:${canonicalSha256({
      source,
      console: { serial: console.serial, firmware: console.firmware },
      route: ROUTE_EXPECTED,
      capture: {
        device: capture.device_name,
        hardwareUid: hardware.hardware_uid,
        sampleRateHz: capture.sample_rate_hz,
        masterChannels: capture.master_channels,
        conservativeTruePeakDbtp: capture.conservative_true_peak_dbtp
      }
    })}`
  };
}

function validateFinalAudit(audit, label) {
  assertCondition(
    audit?.type === 'm110_soundcheck_final_audit_result'
    && audit.contract_version === 1
    && audit.status === 'verified'
    && audit.success === true
    && audit.verified === true
    && audit.reason === 'm110_show_mix_full_reconciliation_verified'
    && typeof audit.run_id === 'string'
    && audit.run_id.length > 0
    && audit.mix_readiness === 'show_mix_ready'
    && Number.isSafeInteger(audit.audited_address_count)
    && audit.audited_address_count > 0
    && audit.audit_pass_count === 2
    && audit.operator_listening_required === true
    && audit.console_mutation_write_attempt_count === 0
    && audit.live_mixer_writes === false
    && audit.live_authority_granted === false
    && audit.readiness_closure_allowed === false
    && sha256Prefixed(audit.verified_state_sha256)
    && sha256Prefixed(audit.committed_surface_ledger_sha256),
    `${label} is invalid`
  );
}

function validateEmbeddedCp05(value, expected, label) {
  assertCondition(deepEqual(value, expected), `${label} does not match CP05 builder verification`);
}

function validateMainCheckpoint(value, checkpoint, reason, label) {
  assertCondition(
    value?.verified === true
    && value.checkpoint === checkpoint
    && value.reason === reason
    && value.console_mutation_write_attempt_count === 0
    && value.terminal?.verified === true
    && value.terminal.checkpoint === checkpoint
    && value.terminal.reason === reason
    && value.terminal.audio_status === 'passed'
    && value.terminal.console_mutation_write_attempt_count === 0
    && value.terminal.live_mixer_writes === false
    && value.terminal.live_authority_granted === false
    && value.terminal.readiness_closure_allowed === false,
    `${label} is invalid`
  );
}

function resetReviewOperations() {
  const rows = [];
  for (let channel = 1; channel <= 16; channel += 1) {
    for (const [suffix, parameter] of RESET_SUFFIXES) {
      rows.push({
        operation_id: `m110-reset-ch${String(channel).padStart(2, '0')}-${suffix.replace('/', '-')}`,
        family_id: 'maintenance_dsp_reset',
        write_kind: 'processor_bypass',
        channel,
        parameter,
        address: `/ch/${channel}/${suffix}`,
        target_value: 0
      });
    }
  }
  return rows;
}

function m006StageForWriteKind(writeKind) {
  if (['gain', 'trim', 'hpf', 'fader'].includes(writeKind)) return 'foundation_level';
  if (['eq', 'gate', 'compressor'].includes(writeKind)) return 'tonal_dynamics';
  if (writeKind === 'pan') return 'spatial_balance';
  throw new Error(`M006 contract has unsupported write kind: ${String(writeKind || '')}`);
}

function m006ScopeAddresses(operation) {
  const channel = operation.channel;
  const base = `/ch/${channel}`;
  const writeKind = operation.write_kind;
  if (['gain', 'trim'].includes(writeKind)) return [`${base}/in/set/trim`];
  if (writeKind === 'fader') return [`${base}/fdr`];
  if (writeKind === 'pan') return [`${base}/pan`];
  if (writeKind === 'hpf') return [`${base}/flt/lcf`, `${base}/flt/lc`];
  if (writeKind === 'eq') {
    const band = operation.target_value?.band;
    assertCondition(Number.isSafeInteger(band) && band >= 1 && band <= 6, 'M006 EQ band is invalid');
    return [`${base}/eq/${band}f`, `${base}/eq/${band}g`, `${base}/eq/${band}q`];
  }
  if (writeKind === 'gate') {
    return ['thr', 'range', 'att', 'hld', 'rel', 'on'].map(name => `${base}/gate/${name}`);
  }
  if (writeKind === 'compressor') {
    return ['thr', 'ratio', 'att', 'rel', 'gain', 'on'].map(name => `${base}/dyn/${name}`);
  }
  throw new Error(`M006 scope cannot be derived for: ${String(writeKind || '')}`);
}

export function validateM006Contract({ contract, contractIdentity }) {
  assertCondition(
    contractIdentity?.sha256 === M006_CONTRACT_SHA256,
    'M006 retained contract byte digest is not the immutable reviewed fixture'
  );
  assertExactKeys(contract, ['operation_count', 'operations', 'schema', 'source'], 'M006 contract');
  assertCondition(
    contract.schema === 'm006_exact_transaction_fixture/v1'
    && contract.operation_count === 51
    && Array.isArray(contract.operations)
    && contract.operations.length === 51
    && contract.source?.transaction_id === M006_SOURCE_TRANSACTION_ID
    && contract.source?.event_line_sha256 === M006_SOURCE_EVENT_SHA256,
    'M006 retained contract identity or source binding is invalid'
  );
  const grouped = Object.fromEntries(M006_STAGES.map(stage => [stage, []]));
  const sourceOperationIds = new Set();
  for (const operation of contract.operations) {
    assertExactKeys(
      operation,
      ['channel', 'family_id', 'metadata', 'operation_id', 'parameter', 'snapshot_values', 'target_value', 'write_kind'],
      'M006 source operation'
    );
    const sourceOperationId = String(operation.operation_id || '');
    const snapshots = operation.snapshot_values;
    assertCondition(
      sourceOperationId.length > 0
      && !sourceOperationIds.has(sourceOperationId)
      && Number.isSafeInteger(operation.channel)
      && operation.channel >= 1
      && operation.channel <= 16
      && typeof operation.family_id === 'string'
      && operation.family_id.length > 0
      && typeof operation.parameter === 'string'
      && operation.parameter.length > 0
      && snapshots
      && typeof snapshots === 'object'
      && !Array.isArray(snapshots)
      && Object.keys(snapshots).length > 0
      && Object.keys(snapshots).every(address => (
        address.startsWith(`/ch/${operation.channel}/`)
        && !address.includes('..')
      )),
      'M006 source operation contract is malformed'
    );
    sourceOperationIds.add(sourceOperationId);
    const stage = m006StageForWriteKind(operation.write_kind);
    const scopeAddresses = m006ScopeAddresses(operation);
    assertCondition(
      scopeAddresses.every(address => Object.hasOwn(snapshots, address)),
      'M006 immutable preimage omits a canonical operation address'
    );
    grouped[stage].push({
      operationId: `m110:m006-restore:${sourceOperationId}`,
      sourceOperationId,
      scopeAddresses,
      sourceOperation: operation
    });
  }
  assertCondition(
    deepEqual(M006_STAGES.map(stage => grouped[stage].length), M006_STAGE_OPERATION_COUNTS),
    'M006 retained contract stage counts drifted'
  );
  const result = Object.fromEntries(M006_STAGES.map(stage => [stage, {
    operationIds: grouped[stage].map(row => row.operationId),
    sourceOperationIds: grouped[stage].map(row => row.sourceOperationId),
    sourceOperations: grouped[stage].map(row => row.sourceOperation),
    scopeAddresses: [...new Set(grouped[stage].flatMap(row => row.scopeAddresses))]
  }]));
  assertCondition(
    deepEqual(M006_STAGES.map(stage => result[stage].scopeAddresses.length), [33, 63, 14]),
    'M006 retained contract canonical scope sizes drifted'
  );
  return result;
}

function canonicalDecimalString(value) {
  assertCondition(
    typeof value === 'number' && Number.isFinite(value),
    'Canonical operation decimal must be finite'
  );
  if (Object.is(value, -0) || value === 0) return '0';
  let text = String(value);
  if (/[eE]/.test(text)) {
    const [coefficient, exponentText] = text.toLowerCase().split('e');
    const exponent = Number(exponentText);
    const negative = coefficient.startsWith('-');
    const digits = coefficient.replace('-', '').replace('.', '');
    const decimalIndex = coefficient.replace('-', '').indexOf('.') === -1
      ? coefficient.replace('-', '').length
      : coefficient.replace('-', '').indexOf('.');
    const shifted = decimalIndex + exponent;
    text = shifted <= 0
      ? `0.${'0'.repeat(-shifted)}${digits}`
      : shifted >= digits.length
        ? `${digits}${'0'.repeat(shifted - digits.length)}`
        : `${digits.slice(0, shifted)}.${digits.slice(shifted)}`;
    if (negative) text = `-${text}`;
  }
  if (text.includes('.')) text = text.replace(/0+$/, '').replace(/\.$/, '');
  return text === '-0' ? '0' : text;
}

function canonicalTarget(value) {
  if (typeof value === 'boolean' || typeof value === 'string') return value;
  return canonicalDecimalString(value);
}

function canonicalWrite(address, value, tolerance) {
  return {
    address,
    target_value: canonicalTarget(value),
    tolerance: canonicalDecimalString(tolerance)
  };
}

function nearestWingRatio(value) {
  const labels = ['1.1', '1.2', '1.3', '1.5', '1.7', '2.0', '2.5', '3.0', '3.5', '4.0', '5.0', '6.0', '8.0', '10', '20', '50', '100'];
  const values = labels.map(Number);
  let selected = 0;
  for (let index = 1; index < values.length; index += 1) {
    if (Math.abs(values[index] - value) < Math.abs(values[selected] - value)) selected = index;
  }
  return labels[selected];
}

function canonicalM006Operation(operation) {
  const base = `/ch/${operation.channel}`;
  const snapshots = operation.snapshot_values;
  const kind = operation.write_kind;
  let writes;
  if (['gain', 'trim'].includes(kind)) {
    writes = [canonicalWrite(`${base}/in/set/trim`, snapshots[`${base}/in/set/trim`], 0.05)];
  } else if (kind === 'fader') {
    writes = [canonicalWrite(`${base}/fdr`, snapshots[`${base}/fdr`], 0.05)];
  } else if (kind === 'pan') {
    writes = [canonicalWrite(`${base}/pan`, snapshots[`${base}/pan`], 0.5)];
  } else if (kind === 'hpf') {
    const frequency = snapshots[`${base}/flt/lcf`];
    writes = [
      canonicalWrite(`${base}/flt/lcf`, frequency, Math.max(1, frequency * 0.005)),
      canonicalWrite(`${base}/flt/lc`, 0, 0)
    ];
  } else if (kind === 'eq') {
    const band = operation.target_value.band;
    const frequency = snapshots[`${base}/eq/${band}f`];
    writes = [
      canonicalWrite(`${base}/eq/${band}f`, frequency, Math.max(1, frequency * 0.005)),
      canonicalWrite(`${base}/eq/${band}g`, snapshots[`${base}/eq/${band}g`], 0.05),
      canonicalWrite(`${base}/eq/${band}q`, snapshots[`${base}/eq/${band}q`], 0.05)
    ];
  } else if (kind === 'gate') {
    const attack = snapshots[`${base}/gate/att`];
    const hold = snapshots[`${base}/gate/hld`];
    const release = snapshots[`${base}/gate/rel`];
    writes = [
      canonicalWrite(`${base}/gate/thr`, snapshots[`${base}/gate/thr`], 0.26),
      canonicalWrite(`${base}/gate/range`, snapshots[`${base}/gate/range`], 0.51),
      canonicalWrite(`${base}/gate/att`, attack, Math.max(0.5, attack * 0.03)),
      canonicalWrite(`${base}/gate/hld`, hold, Math.max(1, hold * 0.03)),
      canonicalWrite(`${base}/gate/rel`, release, Math.max(1, release * 0.03)),
      canonicalWrite(`${base}/gate/on`, 0, 0)
    ];
  } else if (kind === 'compressor') {
    const attack = snapshots[`${base}/dyn/att`];
    const release = snapshots[`${base}/dyn/rel`];
    writes = [
      canonicalWrite(`${base}/dyn/thr`, snapshots[`${base}/dyn/thr`], 0.26),
      canonicalWrite(`${base}/dyn/ratio`, nearestWingRatio(snapshots[`${base}/dyn/ratio`]), 0.05),
      canonicalWrite(`${base}/dyn/att`, attack, Math.max(0.5, attack * 0.03)),
      canonicalWrite(`${base}/dyn/rel`, release, Math.max(1, release * 0.03)),
      canonicalWrite(`${base}/dyn/gain`, snapshots[`${base}/dyn/gain`], 0.26),
      canonicalWrite(`${base}/dyn/on`, 0, 0)
    ];
  } else {
    throw new Error(`Cannot canonicalize M006 operation kind: ${kind}`);
  }
  return {
    operation_id: `m110:m006-restore:${operation.operation_id}`,
    family_id: operation.family_id,
    write_kind: operation.write_kind,
    channel: operation.channel,
    parameter: operation.parameter,
    writes,
    audible_state_reads: []
  };
}

function canonicalSafetyBinding(routeProof, routeIdentity, source, adapterRevision) {
  const evidence = routeProof.evidence;
  const runtime = evidence.runtime;
  const console = evidence.console;
  return {
    adapter_revision: adapterRevision,
    protocol_revision: 'wing-osc-fw3.0.5',
    runtime_instance_nonce: runtime.instance_nonce,
    runtime_manifest_sha256: `sha256:${runtime.manifest_sha256}`,
    source_head: source.head,
    source_tree: source.tree,
    console_serial: console.serial,
    firmware_version: console.firmware,
    route_proof_id: routeProof.proof_id,
    route_artifact_sha256: `sha256:${routeIdentity.sha256}`,
    capture_hardware_uid: evidence.capture.hardware_identity.hardware_uid,
    connection_generation: console.connection_generation,
    transport_generation: console.transport_generation,
    emergency_generation: evidence.safety.emergency_generation
  };
}

function canonicalBundleDigest(bundleKind, operations, safetyBinding) {
  return `sha256:${canonicalSha256({
    schema: 'automixer.canonical_attended_operation_bundle.v1',
    bundle_kind: bundleKind,
    operations,
    safety_binding: safetyBinding
  })}`;
}

export function computeM110FixedOperationDigests({
  routeProof,
  routeIdentity,
  source,
  m006Contract
}) {
  const runtime = routeProof?.evidence?.runtime;
  const console = routeProof?.evidence?.console;
  const safety = routeProof?.evidence?.safety;
  const generations = [
    console?.connection_generation,
    console?.transport_generation,
    safety?.emergency_generation
  ];
  assertCondition(
    bareSha256(routeIdentity?.sha256)
    && /^[0-9a-f]{32}$/.test(String(runtime?.instance_nonce || ''))
    && bareSha256(runtime?.manifest_sha256)
    && /^[0-9a-f]{40}$/.test(String(source?.head || ''))
    && /^[0-9a-f]{40}$/.test(String(source?.tree || ''))
    && /^[0-9a-f]{32}$/.test(String(routeProof?.proof_id || ''))
    && typeof console?.serial === 'string'
    && console.serial.length > 0
    && typeof console?.firmware === 'string'
    && console.firmware.length > 0
    && typeof routeProof?.evidence?.capture?.hardware_identity?.hardware_uid === 'string'
    && routeProof.evidence.capture.hardware_identity.hardware_uid.length > 0
    && generations.every(value => Number.isSafeInteger(value) && value >= 0),
    'M110 canonical operation safety binding is invalid'
  );
  const expectedM006 = validateM006Contract(m006Contract);
  const resetOperations = resetReviewOperations().map(row => ({
    operation_id: row.operation_id,
    family_id: row.family_id,
    write_kind: row.write_kind,
    channel: row.channel,
    parameter: row.parameter,
    writes: [canonicalWrite(row.address, 0, 0)],
    audible_state_reads: []
  }));
  const reset = canonicalBundleDigest(
    RESET_BUNDLE_KIND,
    resetOperations,
    canonicalSafetyBinding(
      routeProof,
      routeIdentity,
      source,
      'm110-maintenance-dsp-reset.v1'
    )
  );
  const m006 = Object.fromEntries(M006_STAGES.map(stage => [
    stage,
    canonicalBundleDigest(
      M006_BUNDLE_KIND,
      expectedM006[stage].sourceOperations.map(canonicalM006Operation),
      canonicalSafetyBinding(
        routeProof,
        routeIdentity,
        source,
        'm110-m006-exact-preimage-restore.v1'
      )
    )
  ]));
  return { reset, m006 };
}

function assertNoCapabilitySecret(value, label) {
  const serialized = JSON.stringify(value) ?? '';
  assertCondition(
    !serialized.includes('m110cap_v1_')
    && !serialized.includes('supervised_capability_token'),
    `${label} leaks capability material`
  );
}

function validateOperatorCheckpoint(checkpoint, {
  proposalId,
  capabilityKind,
  operationDigest,
  summaryEpoch,
  label
}) {
  assertCondition(
    checkpoint?.verified === true
    && checkpoint.expected_literal === operationDigest
    && checkpoint.operator_response_matched === true
    && Number.isFinite(checkpoint.captured_at_epoch)
    && checkpoint.captured_at_epoch > 0
    && Math.floor(checkpoint.captured_at_epoch) <= summaryEpoch
    && checkpoint.deadline_epoch === null
    && checkpoint.proposal_id === proposalId
    && checkpoint.capability_kind === capabilityKind
    && checkpoint.operation_digest === operationDigest,
    `${label} operator digest acknowledgement is invalid`
  );
}

function validateReceiptTerminal(terminal, {
  proposalId,
  capabilityKind,
  operationDigest,
  reviewContentDigest,
  bundleKind,
  operationIds,
  reviewPreimageSha256,
  reset,
  label
}) {
  assertNoCapabilitySecret(terminal, label);
  const proofRun = capabilityKind === 'restore_only_proof';
  const expectedStatus = proofRun ? 'proof_passed' : 'applied';
  const expectedWal = proofRun ? 'proof_restored' : 'committed';
  const applied = terminal?.applied_operation_ids;
  const rollback = terminal?.rollback_operation_ids;
  const already = terminal?.already_satisfied_operation_ids;
  const expectedIds = new Set(operationIds);
  const identifierListsValid = (
    Array.isArray(applied)
    && Array.isArray(rollback)
    && Array.isArray(already)
    && [...applied, ...rollback, ...already].every(value => typeof value === 'string' && value)
    && applied.length === new Set(applied).size
    && rollback.length === new Set(rollback).size
    && already.length === new Set(already).size
    && applied.every(value => !already.includes(value))
  );
  const exactCoverage = identifierListsValid
    && new Set([...applied, ...already]).size === expectedIds.size
    && [...expectedIds].every(value => applied.includes(value) || already.includes(value))
    && (proofRun
      ? rollback.length === applied.length && rollback.every(value => applied.includes(value))
      : rollback.length === 0);
  const allAlreadySatisfied = exactCoverage
    && applied.length === 0
    && rollback.length === 0
    && already.length === expectedIds.size;
  const expectedRollback = proofRun && !allAlreadySatisfied;
  const expectedRollbackStatus = proofRun
    ? (allAlreadySatisfied ? 'not_required_already_satisfied' : 'verified')
    : 'not_attempted';
  const proposal = terminal?.proposal;
  assertCondition(
    terminal?.type === 'operator_proposal_apply_result'
    && terminal.contract_version === 2
    && exactCoverage
    && terminal.status === expectedStatus
    && terminal.success === true
    && terminal.proposal_id === proposalId
    && terminal.proposal_status === expectedStatus
    && terminal.apply_status === expectedStatus
    && terminal.accepted === (reset ? false : !proofRun)
    && terminal.owner_review_required === (reset ? true : proofRun)
    && terminal.live_authority_granted === false
    && terminal.readiness_closure_allowed === false
    && terminal.proof_run === proofRun
    && terminal.restore_after_verify === proofRun
    && terminal.capability_kind === capabilityKind
    && terminal.operation_digest === operationDigest
    && terminal.bundle_kind === bundleKind
    && terminal.review_preimage_sha256 === (reset ? reviewPreimageSha256 : null)
    && terminal.proof_target_verified === proofRun
    && terminal.no_console_send_proven === allAlreadySatisfied
    && terminal.physical_verified === true
    && terminal.physical_state_restored === proofRun
    && terminal.terminal_after_console_send === !allAlreadySatisfied
    && terminal.write_state_unknown === false
    && terminal.rollback_attempted === expectedRollback
    && terminal.rollback_succeeded === expectedRollback
    && terminal.rollback_state_unknown === false
    && terminal.rollback_status === expectedRollbackStatus
    && terminal.manual_review_required === false
    && terminal.emergency_stop_activated === false
    && terminal.emergency_stop_required === false
    && terminal.wal_terminal === expectedWal
    && terminal.audio_recheck_passed === true
    && terminal.result_detail_available === true
    && terminal.result_evidence_pinned === false
    && (proofRun || terminal.parent_commit_recorded === true)
    && sha256Prefixed(terminal.result_digest)
    && proposal?.id === proposalId
    && proposal.status === expectedStatus
    && proposal.review_contract_version === 2
    && proposal.review_content_digest === reviewContentDigest
    && proposal.result_digest === terminal.result_digest,
    `${label} physical apply terminal is invalid or incomplete`
  );
  if (reset) {
    assertCondition(
      sha256Prefixed(terminal.before_manifest_sha256)
      && sha256Prefixed(terminal.after_manifest_sha256)
      && sha256Prefixed(terminal.verification_cut_reset_surface_sha256)
      && sha256Prefixed(terminal.verification_cut_preservation_manifest_sha256)
      && terminal.verification_cut_preservation_manifest_sha256
        === terminal.after_manifest_sha256
      && terminal.no_outside_mutation_attempt_proven === true
      && terminal.current_state_guaranteed_at_response === false
      && terminal.fresh_readback_required_before_next_stage === true
      && terminal.attended_no_touch_required_during_execution === true
      && terminal.later_console_event_invalidates_readiness_freshness === true
      && proposal.result_detail_available === true
      && proposal.result_evidence_pinned === false,
      `${label} reset preservation proof is incomplete`
    );
  } else {
    assertCondition(
      proofRun
        ? terminal.proof_receipt_id == null
        : /^m110proof_v1_[0-9a-f]{64}$/.test(String(terminal.proof_receipt_id || '')),
      `${label} restored-proof receipt binding is invalid`
    );
  }
}

function validateReceipt(receipt, {
  role,
  operationDigest,
  bundleKind,
  operationIds,
  scopeAddresses,
  sourceOperationIds = [],
  reviewPreimageSha256 = null,
  proposalPattern,
  reset = false,
  commitCapabilityKind = 'maintenance_dsp_reset_commit_once',
  summaryEpoch,
  label
}) {
  const capabilityKind = role === 'proof'
    ? 'restore_only_proof'
    : commitCapabilityKind;
  const proposalId = receipt?.proposal_id;
  assertCondition(
    receipt?.verified === true
    && typeof proposalId === 'string'
    && proposalPattern.test(proposalId)
    && receipt.capability_kind === capabilityKind
    && receipt.review_contract_version === 2
    && sha256Prefixed(receipt.review_content_digest)
    && receipt.operation_digest === operationDigest
    && receipt.bundle_kind === bundleKind
    && receipt.operation_count === operationIds.length
    && deepEqual(receipt.scope_addresses, scopeAddresses)
    && deepEqual(receipt.review_operation_ids, operationIds)
    && (reset
      ? receipt.source_operation_ids === undefined
      : receipt.review_preimage_sha256 === ''
        && deepEqual(receipt.source_operation_ids, sourceOperationIds))
    && receipt.token_retained === false
    && (reset
      ? receipt.review_preimage_sha256 === reviewPreimageSha256
      : true),
    `${label} durable receipt envelope is invalid`
  );
  validateOperatorCheckpoint(receipt.operator_checkpoint, {
    proposalId,
    capabilityKind,
    operationDigest,
    summaryEpoch,
    label
  });
  validateReceiptTerminal(receipt.terminal, {
    proposalId,
    capabilityKind,
    operationDigest,
    reviewContentDigest: receipt.review_content_digest,
    bundleKind,
    operationIds,
    reviewPreimageSha256,
    reset,
    label
  });
}

function validateAnalysisEnvelope(analysis, label) {
  assertCondition(
    analysis?.verified === true
    && analysis.reason === 'm110_soundcheck_manifest_registered'
    && typeof analysis.run_id === 'string'
    && analysis.run_id.length > 0
    && Number.isSafeInteger(analysis.plan_revision)
    && analysis.plan_revision >= 1
    && (analysis.imported_count == null
      || (Number.isSafeInteger(analysis.imported_count) && analysis.imported_count >= 0))
    && analysis.shared_capture_retained === true,
    `${label} is invalid`
  );
}

function validateCompletedLaneState(state, lane, readiness) {
  const expectedReason = lane === 'processor'
    ? 'm110_no_unresolved_processor_candidate'
    : 'm110_no_unresolved_soundcheck_child';
  assertCondition(
    state?.state === 'complete'
    && state.success === true
    && state.reason === expectedReason
    && typeof state.run_id === 'string'
    && state.run_id.length > 0
    && Array.isArray(state.residuals)
    && state.residuals.length === 0
    && state.replan_available === false
    && state.replan_from_plan_revision == null
    && state.replan_to_plan_revision == null
    && state.replan_split_operation_count === 0
    && state.replan_safe_alternative_operation_count === 0
    && (lane !== 'soundcheck'
      || deepEqual(state.soundcheck_readiness, readiness)),
    `M110 ${lane} completion envelope is invalid`
  );
}

function validateLaneCommit(step, lane, summaryEpoch) {
  assertExactKeys(
    step,
    [
      'commit',
      'lane',
      'operation_digest',
      'proof',
      'source_operation_ids',
      'state',
      'success'
    ],
    `M110 ${lane} committed step`
  );
  const proof = step.proof;
  const commit = step.commit;
  const operationIds = proof?.review_operation_ids;
  const scopeAddresses = proof?.scope_addresses;
  const sourceOperationIds = step.source_operation_ids;
  const bundleKind = proof?.bundle_kind;
  const operationCount = proof?.operation_count;
  const validBundle = lane === 'processor'
    ? bundleKind === 'processor_enable_bundle' && operationCount === 1
    : ['single_soundcheck_operation', 'soundcheck_atomic_group'].includes(bundleKind)
      && Number.isSafeInteger(operationCount)
      && (bundleKind === 'single_soundcheck_operation'
        ? operationCount === 1
        : operationCount >= 2 && operationCount <= 64);
  assertCondition(
    step.state === 'committed'
    && step.success === true
    && step.lane === lane
    && sha256Prefixed(step.operation_digest)
    && Array.isArray(operationIds)
    && operationIds.length === operationCount
    && operationIds.length === new Set(operationIds).size
    && operationIds.every(value => typeof value === 'string' && value.length > 0)
    && Array.isArray(scopeAddresses)
    && scopeAddresses.length > 0
    && scopeAddresses.length === new Set(scopeAddresses).size
    && scopeAddresses.every(value => typeof value === 'string' && value.startsWith('/'))
    && Array.isArray(sourceOperationIds)
    && sourceOperationIds.length > 0
    && sourceOperationIds.length === new Set(sourceOperationIds).size
    && sourceOperationIds.every(value => typeof value === 'string' && value.length > 0)
    && validBundle
    && proof.operation_digest === step.operation_digest
    && commit?.operation_digest === step.operation_digest
    && commit.bundle_kind === bundleKind
    && commit.operation_count === operationCount
    && deepEqual(commit.scope_addresses, scopeAddresses)
    && deepEqual(commit.review_operation_ids, operationIds)
    && deepEqual(proof.source_operation_ids, sourceOperationIds)
    && deepEqual(commit.source_operation_ids, sourceOperationIds)
    && (lane !== 'soundcheck' || deepEqual(sourceOperationIds, operationIds)),
    `M110 ${lane} committed step is not bound to one exact proof/commit pair`
  );
  const proposalPrefix = lane === 'processor'
    ? 'm110-processor'
    : 'm110-soundcheck-child';
  const commitCapabilityKind = lane === 'processor'
    ? 'processor_enable_commit_once'
    : 'soundcheck_child_commit_once';
  for (const role of ['proof', 'commit']) {
    validateReceipt(step[role], {
      role,
      operationDigest: step.operation_digest,
      bundleKind,
      operationIds,
      scopeAddresses,
      sourceOperationIds,
      proposalPattern: new RegExp(
        `^${proposalPrefix}-${role}-[0-9a-f]{32}$`
      ),
      commitCapabilityKind,
      summaryEpoch,
      label: `M110 ${lane} ${role}`
    });
  }
  return {
    proposalIds: [proof.proposal_id, commit.proposal_id],
    proofReceiptId: commit.terminal.proof_receipt_id
  };
}

function validateInterleavedLanes(lanes, readiness, initialAnalysis, summaryEpoch) {
  assertExactKeys(
    lanes,
    [
      'commit_count',
      'commits',
      'lane_states',
      'processor',
      'reanalyses',
      'reason',
      'replans',
      'soundcheck',
      'soundcheck_readiness',
      'status',
      'verified'
    ],
    'M110 interleaved lanes'
  );
  assertCondition(
    lanes.verified === true
    && lanes.status === 'complete'
    && lanes.reason === 'm110_interleaved_show_mix_ready'
    && deepEqual(lanes.lane_states?.processor, lanes.processor)
    && deepEqual(lanes.lane_states?.soundcheck, lanes.soundcheck)
    && deepEqual(lanes.soundcheck_readiness, readiness),
    'M110 interleaved lane aliases or completion state are invalid'
  );
  validateCompletedLaneState(lanes.processor, 'processor', readiness);
  validateCompletedLaneState(lanes.soundcheck, 'soundcheck', readiness);
  assertExactKeys(lanes.commits, ['processor', 'soundcheck'], 'M110 lane commits');
  assertCondition(
    Array.isArray(lanes.commits.processor)
    && Array.isArray(lanes.commits.soundcheck)
    && Number.isSafeInteger(lanes.commit_count)
    && lanes.commit_count >= 0
    && lanes.commit_count <= 512
    && lanes.commit_count
      === lanes.commits.processor.length + lanes.commits.soundcheck.length
    && Array.isArray(lanes.reanalyses)
    && lanes.reanalyses.length <= 16
    && Array.isArray(lanes.replans)
    && lanes.replans.length <= 64,
    'M110 interleaved lane counts or arrays are invalid'
  );
  const operationDigests = new Set();
  const proposalIds = new Set();
  const proofReceiptIds = new Set();
  for (const lane of ['processor', 'soundcheck']) {
    for (const step of lanes.commits[lane]) {
      assertCondition(
        !operationDigests.has(step?.operation_digest),
        'M110 lane operation digest was repeated after commit'
      );
      operationDigests.add(step.operation_digest);
      const validated = validateLaneCommit(step, lane, summaryEpoch);
      for (const proposalId of validated.proposalIds) {
        assertCondition(!proposalIds.has(proposalId), 'M110 lane proposal ID was reused');
        proposalIds.add(proposalId);
      }
      assertCondition(
        !proofReceiptIds.has(validated.proofReceiptId),
        'M110 restored-proof receipt ID was reused by multiple commits'
      );
      proofReceiptIds.add(validated.proofReceiptId);
    }
  }
  const analyses = [initialAnalysis, ...lanes.reanalyses];
  assertCondition(
    new Set(analyses.map(analysis => analysis?.run_id)).size === analyses.length,
    'M110 soundcheck analysis run IDs must be unique'
  );
  for (const [index, analysis] of analyses.entries()) {
    validateAnalysisEnvelope(analysis, `M110 soundcheck analysis ${index + 1}`);
  }
  const currentRevisionByRun = new Map(
    analyses.map(analysis => [analysis.run_id, analysis.plan_revision])
  );
  const seenReplans = new Set();
  for (const replan of lanes.replans) {
    assertExactKeys(
      replan,
      [
        'from_plan_revision',
        'run_id',
        'safe_alternative_operation_count',
        'split_operation_count',
        'state',
        'success',
        'to_plan_revision'
      ],
      'M110 bounded replan'
    );
    const safeCount = replan.safe_alternative_operation_count;
    const key = `${replan.run_id}:${replan.from_plan_revision}:${replan.to_plan_revision}`;
    assertCondition(
      replan.state === 'advanced'
      && replan.success === true
      && typeof replan.run_id === 'string'
      && replan.run_id.length > 0
      && Number.isSafeInteger(replan.from_plan_revision)
      && replan.from_plan_revision >= 1
      && replan.to_plan_revision === replan.from_plan_revision + 1
      && Number.isSafeInteger(replan.split_operation_count)
      && replan.split_operation_count >= 0
      && Number.isSafeInteger(safeCount)
      && (safeCount === 0 || safeCount >= 2)
      && replan.split_operation_count + safeCount >= 1
      && currentRevisionByRun.get(replan.run_id) === replan.from_plan_revision
      && !seenReplans.has(key),
      'M110 bounded replan receipt is invalid or discontinuous'
    );
    seenReplans.add(key);
    currentRevisionByRun.set(replan.run_id, replan.to_plan_revision);
  }
  const finalAnalysis = analyses.at(-1);
  assertCondition(
    finalAnalysis.run_id === readiness?.run_id
    && Number.isSafeInteger(readiness.plan_revision)
    && readiness.plan_revision >= 1
    && bareSha256(readiness.plan_fingerprint)
    && currentRevisionByRun.get(readiness.run_id) === readiness.plan_revision
    && lanes.processor.run_id === readiness?.run_id
    && lanes.soundcheck.run_id === readiness?.run_id,
    'M110 final analysis and lane completion do not bind the ready run'
  );
  return finalAnalysis;
}

export function validateM110AttendedEvidence({
  summary,
  routeProof,
  routeBytes,
  routeIdentity,
  source,
  cp05,
  m006Contract
}) {
  assertNoCapabilitySecret(summary, 'M110 execution summary');
  const summaryEpoch = parseIso(summary?.timestamp_utc, 'M110 summary timestamp');
  assertCondition(
    summary?.schema === 'automixer.m110_attended_runner_summary.v1'
    && summary.status === 'show_mix_ready'
    && summary.reason === 'm110_attended_show_mix_committed'
    && Array.isArray(summary.authorization_failures)
    && summary.authorization_failures.length === 0
    && summary.required_confirmation === 'EXECUTE_M110_ATTENDED_WING_SOUNDCHECK'
    && summary.route_confirmation === 'CAPTURE_WING_MAIN_USB_47_48_ROUTE_PROOF'
    && summary.reset_containment_confirmation
      === 'CONTAIN_ALL_PA_MONITOR_OUTPUTS_AND_MUTE_MAIN_1_FOR_M110_RESET'
    && summary.soundcheck_unmute_confirmation === 'UNMUTE_MAIN_1_FOR_M110_SOUNDCHECK'
    && summary.interactive_tty_required === true
    && summary.contacts_backend === true
    && summary.contacts_wing === true
    && summary.opens_udp_or_osc === true
    && summary.runner_opens_udp_or_osc === false
    && summary.proof_owned_backend_opens_udp_or_osc === true
    && summary.live_write === true
    && summary.accepted === true
    && summary.readiness_closure_allowed === true,
    'M110 summary is not one accepted attended full execution'
  );
  const execution = summary.execution;
  assertCondition(
    execution?.verified === true
    && execution.status === 'show_mix_ready'
    && execution.reason === 'm110_attended_show_mix_hw05_verified_and_torn_down'
    && execution.live_authority_granted === false
    && execution.readiness_closure_allowed === true
    && deepEqual(summary.launch_probe, execution.identity)
    && deepEqual(summary.route_probe, execution.route),
    'M110 execution envelope is inconsistent'
  );
  validateEmbeddedCp05(execution.cp05_hardware_gate, cp05, 'M110 execution CP05');
  validateEmbeddedCp05(execution.route?.cp05_hardware_gate, cp05, 'M110 route CP05');
  const route = validateRouteArtifact(
    routeProof,
    routeBytes,
    routeIdentity,
    execution.route,
    source,
    summaryEpoch
  );
  assertCondition(
    execution.identity?.manifest_sha256 === routeProof.evidence.runtime.manifest_sha256
    && execution.identity?.instance_nonce === routeProof.evidence.runtime.instance_nonce,
    'M110 launch identity does not bind the route runtime'
  );
  assertCondition(
    execution.arm?.verified === true
    && execution.arm.reason === 'm110_supervised_capability_only_armed'
    && execution.arm.route_proof_id === route.proofId
    && execution.arm.normal_operator_mode_unchanged === true
    && execution.arm.normal_live_allowlist_changed === false
    && execution.arm.live_authority_granted === false
    && execution.arm.readiness_closure_allowed === false,
    'M110 supervised-only control arm is invalid'
  );
  const containment = execution.reset_containment_operator_ack;
  assertCondition(
    containment?.verified === true
    && containment.operator_response_matched === true
    && containment.expected_literal
      === 'CONTAIN_ALL_PA_MONITOR_OUTPUTS_AND_MUTE_MAIN_1_FOR_M110_RESET'
    && containment.operator_attested_external_output_containment === true,
    'M110 physical output-containment acknowledgement is invalid'
  );
  validateMainCheckpoint(
    execution.reset_muted_checkpoint,
    'reset_muted',
    'm110_reset_muted_containment_verified',
    'M110 reset-muted checkpoint'
  );
  const resetOperations = resetReviewOperations();
  const resetScope = resetOperations.map(row => row.address);
  const resetOperationIds = resetOperations.map(row => row.operation_id);
  const fixedOperationDigests = computeM110FixedOperationDigests({
    routeProof,
    routeIdentity,
    source,
    m006Contract
  });
  const reset = execution.reset;
  assertCondition(
    reset?.verified === true
    && reset.reason === 'm110_fixed_maintenance_reset_committed'
    && reset.reset_applied === true
    && reset.same_websocket_retained === true
    && reset.backend_process_retained === true
    && reset.token_retained === false
    && reset.live_authority_granted === false
    && reset.readiness_closure_allowed === false
    && sha256Prefixed(reset.operation_digest)
    && reset.operation_digest === fixedOperationDigests.reset
    && sha256Prefixed(reset.review_preimage_sha256)
    && deepEqual(reset.scope_addresses, resetScope),
    'M110 fixed maintenance reset is incomplete'
  );
  validateReceipt(reset.proof, {
    role: 'proof',
    operationDigest: reset.operation_digest,
    bundleKind: RESET_BUNDLE_KIND,
    operationIds: resetOperationIds,
    scopeAddresses: resetScope,
    reviewPreimageSha256: reset.review_preimage_sha256,
    proposalPattern: /^m110-maintenance-reset-proof-[0-9a-f]{32}$/,
    reset: true,
    summaryEpoch,
    label: 'M110 reset proof'
  });
  validateReceipt(reset.commit, {
    role: 'commit',
    operationDigest: reset.operation_digest,
    bundleKind: RESET_BUNDLE_KIND,
    operationIds: resetOperationIds,
    scopeAddresses: resetScope,
    reviewPreimageSha256: reset.review_preimage_sha256,
    proposalPattern: /^m110-maintenance-reset-commit-[0-9a-f]{32}$/,
    reset: true,
    summaryEpoch,
    label: 'M110 reset commit'
  });
  const expectedM006 = validateM006Contract(m006Contract);
  const m006 = execution.m006_restore;
  assertCondition(
    m006?.verified === true
    && m006.reason === 'm110_m006_exact_preimage_restored'
    && m006.source_transaction_id === M006_SOURCE_TRANSACTION_ID
    && m006.source_event_sha256 === M006_SOURCE_EVENT_SHA256
    && m006.stage_count === 3
    && Array.isArray(m006.stages)
    && deepEqual(
      (m006.stages || []).map(row => row?.stage),
      M006_STAGES
    )
    && m006.m006_restore_complete === true
    && m006.same_websocket_retained === true
    && m006.backend_process_retained === true
    && m006.token_retained === false
    && m006.live_authority_granted === false
    && m006.readiness_closure_allowed === false
    && m006.stages.every(row => row && sha256Prefixed(row.operation_digest)),
    'M006 exact preimage restoration is incomplete'
  );
  assertCondition(
    new Set(m006.stages.map(row => row.operation_digest)).size === M006_STAGES.length,
    'M006 stage operation digests must be pairwise distinct'
  );
  for (const row of m006.stages) {
    const expected = expectedM006[row.stage];
    assertCondition(
      expected
      && deepEqual(row.scope_addresses, expected.scopeAddresses)
      && row.operation_digest === fixedOperationDigests.m006[row.stage],
      `M006 ${row.stage} scope or canonical digest does not match the immutable contract`
    );
    for (const role of ['proof', 'commit']) {
      validateReceipt(row[role], {
        role,
        operationDigest: row.operation_digest,
        bundleKind: M006_BUNDLE_KIND,
        operationIds: expected.operationIds,
        scopeAddresses: expected.scopeAddresses,
        sourceOperationIds: expected.sourceOperationIds,
        proposalPattern: new RegExp(
          `^m110-m006-restore-${row.stage}-${role}-[0-9a-f]{32}$`
        ),
        summaryEpoch,
        label: `M006 ${row.stage} ${role}`
      });
    }
  }
  const unmute = execution.soundcheck_unmute_operator_ack;
  assertCondition(
    unmute?.verified === true
    && unmute.operator_response_matched === true
    && unmute.expected_literal === 'UNMUTE_MAIN_1_FOR_M110_SOUNDCHECK',
    'M110 soundcheck-unmute acknowledgement is invalid'
  );
  validateMainCheckpoint(
    execution.soundcheck_unmuted_checkpoint,
    'soundcheck_unmuted',
    'm110_soundcheck_unmuted_audio_verified',
    'M110 soundcheck-unmuted checkpoint'
  );
  validateMainCheckpoint(
    execution.final_soundcheck_unmuted_checkpoint,
    'soundcheck_unmuted',
    'm110_soundcheck_unmuted_audio_verified',
    'M110 final soundcheck-unmuted checkpoint'
  );
  const readiness = execution.soundcheck_readiness;
  validateAnalysisEnvelope(execution.analysis, 'M110 initial soundcheck analysis');
  const finalLaneAnalysis = validateInterleavedLanes(
    execution.lanes,
    readiness,
    execution.analysis,
    summaryEpoch
  );
  assertCondition(
    deepEqual(execution.processors, execution.lanes.processor)
    && deepEqual(execution.soundcheck, execution.lanes.soundcheck)
    && deepEqual(readiness, execution.soundcheck?.soundcheck_readiness),
    'M110 outer lane aliases do not match the retained scheduler result'
  );
  assertCondition(
    readiness?.mix_readiness === 'show_mix_ready'
    && readiness.status === 'show_mix_ready'
    && readiness.safety_state === 'clear'
    && readiness.execution_status === 'complete'
    && readiness.proposal_only === false
    && readiness.live_authority_changed === false
    && Array.isArray(readiness.global_blockers)
    && readiness.global_blockers.length === 0
    && Array.isArray(readiness.unresolved_required_operation_ids)
    && readiness.unresolved_required_operation_ids.length === 0
    && (!readiness.unresolved_reasons
      || Object.keys(readiness.unresolved_reasons).length === 0)
    && readiness.readiness_evidence_complete === true
    && readiness.source_coverage_fresh === true
    && readiness.main_47_48_evidence_fresh === true
    && readiness.safe_core_stage_coverage_complete === true
    && readiness.show_mix_stage_coverage_complete === true,
    'M110 show-mix readiness still has unresolved work or safety blockers'
  );
  const before = execution.final_audit_before_listening;
  const after = execution.final_audit_after_listening;
  validateFinalAudit(before, 'Pre-listening final audit');
  validateFinalAudit(after, 'Post-listening final audit');
  assertCondition(
    finalLaneAnalysis.run_id === readiness.run_id
    && before.run_id === readiness.run_id
    && after.verified_state_sha256 === before.verified_state_sha256
    && after.committed_surface_ledger_sha256 === before.committed_surface_ledger_sha256
    && after.run_id === before.run_id,
    'Verified mix state changed across physical listening'
  );
  const listening = execution.hw05_operator_listening_ack;
  assertCondition(
    listening?.verified === true
    && listening.operator_response_matched === true
    && listening.verified_state_sha256 === before.verified_state_sha256
    && listening.expected_literal
      === `CONFIRM_M110_HW05_LISTENING_${before.verified_state_sha256}`
    && Number.isFinite(listening.captured_at_epoch)
    && Number.isFinite(listening.deadline_epoch)
    && listening.captured_at_epoch < listening.deadline_epoch
    && Math.floor(listening.captured_at_epoch) <= summaryEpoch,
    'HW05 physical listening acknowledgement is invalid'
  );
  const teardown = execution.teardown;
  assertCondition(
    teardown?.verified === true
    && teardown.capture_stopped === true
    && teardown.mixer_disconnected === true
    && teardown.websocket_closed === true
    && teardown.backend_process_terminated === true
    && teardown.route_authority_retained === false
    && Array.isArray(teardown.errors)
    && teardown.errors.length === 0
    && teardown.backend_shutdown?.terminated === true
    && teardown.backend_shutdown.clean_shutdown === true
    && teardown.backend_shutdown.forced === false
    && (!Array.isArray(teardown.backend_shutdown.errors)
      || teardown.backend_shutdown.errors.length === 0),
    'M110 proof-owned teardown is incomplete'
  );
  return {
    verified: true,
    status: 'show_mix_ready',
    runId: before.run_id,
    verifiedStateSha256: before.verified_state_sha256,
    committedSurfaceLedgerSha256: before.committed_surface_ledger_sha256,
    console: { serial: route.consoleSerial, firmware: route.firmware },
    capture: {
      device: route.captureDevice,
      hardwareUid: route.captureHardwareUid,
      conservativeTruePeakDbtp: route.conservativeTruePeakDbtp,
      headroomCeilingDbtp: route.headroomCeilingDbtp
    },
    route
  };
}

function readStableInputJson(filePath, label) {
  const resolved = path.resolve(filePath);
  const root = path.dirname(resolved);
  const stable = readStableJsonFileNoFollow(resolved, label, { containmentRoot: root });
  const identity = artifactFileIdentity(root, resolved, label);
  assertCondition(identitiesEqual(stable.identity, identity), `${label} changed during read`);
  return { value: stable.value, bytes: stable.bytes, identity, path: resolved };
}

function readPayloadJson(releaseRoot, recorded, label) {
  const resolved = resolveArtifactPath(releaseRoot, recorded?.path, label, { mustExist: true });
  const stable = readStableJsonFileNoFollow(resolved, label, { containmentRoot: releaseRoot });
  const identity = artifactFileIdentity(releaseRoot, resolved, label);
  assertCondition(
    identitiesEqual(recorded, identity) && identitiesEqual(stable.identity, identity),
    `${label} identity mismatch`
  );
  return { value: stable.value, bytes: stable.bytes, identity, path: resolved };
}

function readPayloadFile(releaseRoot, recorded, label) {
  const resolved = resolveArtifactPath(releaseRoot, recorded?.path, label, { mustExist: true });
  const initialIdentity = artifactFileIdentity(releaseRoot, resolved, label);
  assertCondition(
    identitiesEqual(recorded, initialIdentity)
    && Number.isSafeInteger(initialIdentity.size)
    && initialIdentity.size >= 0
    && initialIdentity.size <= 16 * 1024 * 1024,
    `${label} identity mismatch or byte size is invalid`
  );
  const descriptor = fs.openSync(
    resolved,
    fs.constants.O_RDONLY | (fs.constants.O_NOFOLLOW || 0)
  );
  try {
    const before = fs.fstatSync(descriptor, { bigint: true });
    assertCondition(before.isFile(), `${label} must be a real file`);
    const bytes = fs.readFileSync(descriptor);
    const after = fs.fstatSync(descriptor, { bigint: true });
    const pathStat = fs.lstatSync(resolved, { bigint: true });
    const identity = artifactFileIdentity(releaseRoot, resolved, label);
    const stableStat = left => (
      left.dev === before.dev
      && left.ino === before.ino
      && left.size === before.size
      && left.mode === before.mode
      && left.nlink === before.nlink
      && left.mtimeNs === before.mtimeNs
      && left.ctimeNs === before.ctimeNs
    );
    assertCondition(
      !pathStat.isSymbolicLink()
      && pathStat.isFile()
      && stableStat(after)
      && stableStat(pathStat)
      && bytes.length === initialIdentity.size
      && sha256Bytes(bytes) === initialIdentity.sha256
      && identitiesEqual(initialIdentity, identity),
      `${label} changed during pinned byte read`
    );
    return { identity, path: resolved, bytes };
  } finally {
    fs.closeSync(descriptor);
  }
}

function writeGitBlob(
  repoRoot,
  sourceTree,
  sourcePath,
  destinationRoot,
  destinationPath,
  label
) {
  assertCondition(sourceObjectId(sourceTree), `${label} source tree is invalid`);
  assertCondition(
    typeof sourcePath === 'string'
    && sourcePath.length > 0
    && !sourcePath.startsWith('/')
    && !sourcePath.includes('\\')
    && !sourcePath.split('/').some(part => !part || part === '.' || part === '..'),
    `${label} source path is invalid`
  );
  const objectId = runGit(repoRoot, ['rev-parse', `${sourceTree}:${sourcePath}`])
    .toString('utf8')
    .trim();
  assertCondition(sourceObjectId(objectId), `${label} Git blob identity is invalid`);
  const bytes = runGit(repoRoot, ['cat-file', 'blob', objectId], {
    maxBuffer: 8 * 1024 * 1024
  });
  assertCondition(bytes.length > 0, `${label} Git blob is empty`);
  const written = writeNewArtifactFileNoFollow(
    destinationRoot,
    destinationPath,
    bytes,
    label
  );
  assertCondition(written.sha256 === sha256Bytes(bytes), `${label} copied Git blob identity drifted`);
  return written;
}

function gitObjectId(objectFormat, objectType, bytes) {
  assertCondition(['sha1', 'sha256'].includes(objectFormat), 'Git object format is unsupported');
  assertCondition(['blob', 'tree', 'commit'].includes(objectType), 'Git object type is unsupported');
  return crypto.createHash(objectFormat)
    .update(Buffer.from(`${objectType} ${bytes.length}\0`, 'utf8'))
    .update(bytes)
    .digest('hex');
}

function parseGitTreeObject(bytes, objectFormat, label) {
  const objectIdBytes = objectFormat === 'sha1' ? 20 : 32;
  const entries = [];
  let offset = 0;
  while (offset < bytes.length) {
    const space = bytes.indexOf(0x20, offset);
    const nul = bytes.indexOf(0x00, space + 1);
    assertCondition(space > offset && nul > space + 1, `${label} is malformed`);
    const mode = bytes.subarray(offset, space).toString('ascii');
    const name = new TextDecoder('utf-8', { fatal: true }).decode(
      bytes.subarray(space + 1, nul)
    );
    const objectStart = nul + 1;
    const objectEnd = objectStart + objectIdBytes;
    assertCondition(
      /^[0-7]{5,6}$/.test(mode)
      && name.length > 0
      && !name.includes('/')
      && name !== '.'
      && name !== '..'
      && objectEnd <= bytes.length,
      `${label} contains an invalid entry`
    );
    entries.push({
      mode,
      name,
      objectId: bytes.subarray(objectStart, objectEnd).toString('hex')
    });
    offset = objectEnd;
  }
  assertCondition(
    entries.length > 0
    && entries.length === new Set(entries.map(entry => entry.name)).size,
    `${label} entries are empty or duplicated`
  );
  return entries;
}

function canonicalBase64(bytes, label) {
  const encoded = bytes.toString('base64');
  assertCondition(encoded.length > 0, `${label} is empty`);
  return encoded;
}

function decodeCanonicalBase64(value, label) {
  assertCondition(
    typeof value === 'string'
    && value.length > 0
    && /^[A-Za-z0-9+/]+={0,2}$/.test(value),
    `${label} is not canonical base64`
  );
  const bytes = Buffer.from(value, 'base64');
  assertCondition(bytes.toString('base64') === value, `${label} base64 spelling is invalid`);
  return bytes;
}

function buildGitSourceMembershipProof(repoRoot, source) {
  const objectFormat = runGit(repoRoot, ['rev-parse', '--show-object-format'])
    .toString('utf8')
    .trim();
  const objectIdPattern = objectFormat === 'sha1'
    ? /^[0-9a-f]{40}$/
    : objectFormat === 'sha256'
      ? /^[0-9a-f]{64}$/
      : null;
  assertCondition(
    objectIdPattern
    && objectIdPattern.test(source.head)
    && objectIdPattern.test(source.tree),
    'Git source object format does not match the captured source IDs'
  );
  const commitBytes = runGit(repoRoot, ['cat-file', 'commit', source.head], {
    maxBuffer: 8 * 1024 * 1024
  });
  const treeHeader = Buffer.from(`tree ${source.tree}\n`, 'ascii');
  assertCondition(
    gitObjectId(objectFormat, 'commit', commitBytes) === source.head
    && commitBytes.subarray(0, treeHeader.length).equals(treeHeader),
    'Git source commit does not bind the captured root tree'
  );
  const treeObjects = new Map();
  const paths = [];
  for (const logicalPath of SOURCE_MEMBERSHIP_PATHS) {
    let treeObjectId = source.tree;
    const components = logicalPath.split('/');
    let finalEntry = null;
    for (const [index, component] of components.entries()) {
      let treeBytes = treeObjects.get(treeObjectId);
      if (!treeBytes) {
        treeBytes = runGit(repoRoot, ['cat-file', 'tree', treeObjectId], {
          maxBuffer: 16 * 1024 * 1024
        });
        assertCondition(
          gitObjectId(objectFormat, 'tree', treeBytes) === treeObjectId,
          `Git tree object identity mismatch: ${treeObjectId}`
        );
        treeObjects.set(treeObjectId, treeBytes);
      }
      const entry = parseGitTreeObject(
        treeBytes,
        objectFormat,
        `Git tree ${treeObjectId}`
      ).find(candidate => candidate.name === component);
      assertCondition(entry, `Committed source path is absent from Git tree: ${logicalPath}`);
      if (index < components.length - 1) {
        assertCondition(entry.mode === '40000', `Committed source parent is not a tree: ${logicalPath}`);
        treeObjectId = entry.objectId;
      } else {
        assertCondition(
          ['100644', '100755'].includes(entry.mode),
          `Committed source path is not a regular blob: ${logicalPath}`
        );
        finalEntry = entry;
      }
    }
    assertCondition(finalEntry, `Committed source blob traversal is incomplete: ${logicalPath}`);
    paths.push({
      path: logicalPath,
      mode: finalEntry.mode,
      blobObjectId: finalEntry.objectId
    });
  }
  return {
    schema: 'automixer.git_source_membership_proof.v1',
    objectFormat,
    sourceHead: source.head,
    sourceTree: source.tree,
    commitObjectBase64: canonicalBase64(commitBytes, 'Git commit object'),
    treeObjects: [...treeObjects.entries()]
      .sort(([left], [right]) => left.localeCompare(right))
      .map(([objectId, bytes]) => ({
        objectId,
        bytesBase64: canonicalBase64(bytes, `Git tree ${objectId}`)
      })),
    paths
  };
}

function validateGitSourceMembershipProof(proof, proofBytes, source, retainedBytesByPath) {
  assertCondition(
    Buffer.isBuffer(proofBytes)
    && proofBytes.length > 0
    && proofBytes.length <= 4 * 1024 * 1024,
    'Committed source membership proof byte size is invalid'
  );
  assertCanonicalJsonBytes(proof, proofBytes, 'Committed source membership proof');
  assertExactKeys(
    proof,
    [
      'commitObjectBase64',
      'objectFormat',
      'paths',
      'schema',
      'sourceHead',
      'sourceTree',
      'treeObjects'
    ],
    'Committed source membership proof'
  );
  const objectFormat = proof.objectFormat;
  const objectIdPattern = objectFormat === 'sha1'
    ? /^[0-9a-f]{40}$/
    : objectFormat === 'sha256'
      ? /^[0-9a-f]{64}$/
      : null;
  const commitBytes = decodeCanonicalBase64(
    proof.commitObjectBase64,
    'Committed source commit object'
  );
  const treeHeader = Buffer.from(`tree ${proof.sourceTree}\n`, 'ascii');
  assertCondition(
    proof.schema === 'automixer.git_source_membership_proof.v1'
    && objectIdPattern
    && proof.sourceHead === source.head
    && proof.sourceTree === source.tree
    && objectIdPattern.test(proof.sourceHead)
    && objectIdPattern.test(proof.sourceTree)
    && gitObjectId(objectFormat, 'commit', commitBytes) === proof.sourceHead
    && commitBytes.subarray(0, treeHeader.length).equals(treeHeader),
    'Committed source commit object does not bind the selected source tree'
  );
  assertCondition(
    Array.isArray(proof.treeObjects)
    && proof.treeObjects.length > 0
    && proof.treeObjects.length
      === new Set(proof.treeObjects.map(row => row?.objectId)).size,
    'Committed source tree object set is invalid'
  );
  const trees = new Map();
  assertCondition(
    deepEqual(
      proof.treeObjects.map(row => row?.objectId),
      proof.treeObjects.map(row => row?.objectId).sort()
    ),
    'Committed source tree objects are not canonically ordered'
  );
  for (const row of proof.treeObjects) {
    assertExactKeys(row, ['bytesBase64', 'objectId'], 'Committed Git tree object');
    const bytes = decodeCanonicalBase64(row.bytesBase64, `Committed Git tree ${row.objectId}`);
    assertCondition(
      objectIdPattern.test(row.objectId)
      && gitObjectId(objectFormat, 'tree', bytes) === row.objectId,
      'Committed Git tree object identity is invalid'
    );
    trees.set(row.objectId, parseGitTreeObject(bytes, objectFormat, `Committed Git tree ${row.objectId}`));
  }
  assertCondition(
    Array.isArray(proof.paths)
    && deepEqual(proof.paths.map(row => row?.path), SOURCE_MEMBERSHIP_PATHS)
    && deepEqual(Object.keys(retainedBytesByPath).sort(), COMMITTED_SOURCE_PATHS),
    'Committed source membership path set is not exact'
  );
  const traversedTreeObjectIds = new Set();
  for (const row of proof.paths) {
    assertExactKeys(row, ['blobObjectId', 'mode', 'path'], 'Committed source path proof');
    let treeObjectId = proof.sourceTree;
    const components = row.path.split('/');
    let entry = null;
    for (const [index, component] of components.entries()) {
      const entries = trees.get(treeObjectId);
      traversedTreeObjectIds.add(treeObjectId);
      assertCondition(entries, `Committed source tree object is missing: ${treeObjectId}`);
      entry = entries.find(candidate => candidate.name === component);
      assertCondition(entry, `Committed source path traversal failed: ${row.path}`);
      if (index < components.length - 1) {
        assertCondition(entry.mode === '40000', `Committed source parent mode is invalid: ${row.path}`);
        treeObjectId = entry.objectId;
      }
    }
    const bytes = retainedBytesByPath[row.path];
    assertCondition(
      ['100644', '100755'].includes(row.mode)
      && entry.mode === row.mode
      && entry.objectId === row.blobObjectId,
      `Committed source path does not bind a regular blob in the selected Git tree: ${row.path}`
    );
    if (Object.hasOwn(retainedBytesByPath, row.path)) {
      assertCondition(
        Buffer.isBuffer(bytes)
        && gitObjectId(objectFormat, 'blob', bytes) === row.blobObjectId,
        `Retained committed source bytes are not a blob in the selected Git tree: ${row.path}`
      );
    }
  }
  assertCondition(
    deepEqual([...traversedTreeObjectIds].sort(), [...trees.keys()].sort()),
    'Committed source membership proof contains unused Git tree objects'
  );
  return {
    verified: true,
    schema: proof.schema,
    objectFormat,
    sourceHead: proof.sourceHead,
    sourceTree: proof.sourceTree,
    pathCount: proof.paths.length
  };
}

function validateRetainedGitSourceMembership(releaseRoot, payload, source, label) {
  const membership = readPayloadJson(
    releaseRoot,
    payload.sourceMembership,
    `${label} committed source membership proof`
  );
  const retained = {
    [SOURCE_BLOBS.runner]: readPayloadFile(
      releaseRoot,
      payload.runnerSource,
      `${label} committed runner source`
    ),
    [SOURCE_BLOBS.backendServer]: readPayloadFile(
      releaseRoot,
      payload.backendSource,
      `${label} committed backend source`
    ),
    [SOURCE_BLOBS.m006Contract]: readPayloadFile(
      releaseRoot,
      payload.m006Contract,
      `${label} committed M006 contract`
    ),
    [CP05_GATE_PATH]: readPayloadFile(
      releaseRoot,
      payload.cp05Gate,
      `${label} committed CP05 hardware gate`
    ),
    [CP05_EVIDENCE_PATH]: readPayloadFile(
      releaseRoot,
      payload.cp05Evidence,
      `${label} committed CP05 software evidence`
    )
  };
  const verification = validateGitSourceMembershipProof(
    membership.value,
    membership.bytes,
    source,
    Object.fromEntries(
      Object.entries(retained).map(([logicalPath, file]) => [logicalPath, file.bytes])
    )
  );
  return { membership, retained, verification };
}

function ensureRealDirectory(directory, label) {
  const resolved = path.resolve(directory);
  const stat = fs.lstatSync(resolved);
  assertCondition(!stat.isSymbolicLink() && stat.isDirectory(), `${label} must be a real directory`);
  return resolved;
}

function privateInputStatIdentity(stat) {
  return Object.fromEntries([
    'dev',
    'ino',
    'size',
    'mode',
    'nlink',
    'mtimeNs',
    'ctimeNs'
  ].map(name => [name, stat[name].toString()]));
}

function capturePrivateInputClosure(inputPath, releaseRoot, label) {
  const resolvedInput = path.resolve(inputPath);
  const resolvedRelease = path.resolve(releaseRoot);
  const isOutside = (root, candidate) => {
    const relative = path.relative(root, candidate);
    return path.isAbsolute(relative)
      || relative === '..'
      || relative.startsWith(`..${path.sep}`);
  };
  const before = fs.lstatSync(resolvedInput, { bigint: true });
  const descriptor = fs.openSync(
    resolvedInput,
    fs.constants.O_RDONLY | (fs.constants.O_NOFOLLOW || 0)
  );
  try {
    const pinnedBefore = fs.fstatSync(descriptor, { bigint: true });
    const bytes = fs.readFileSync(descriptor);
    const pinnedAfter = fs.fstatSync(descriptor, { bigint: true });
    const realInput = fs.realpathSync(resolvedInput);
    const after = fs.lstatSync(resolvedInput, { bigint: true });
    const pinnedIdentity = privateInputStatIdentity(pinnedBefore);
    assertCondition(
      !before.isSymbolicLink()
      && before.isFile()
      && before.nlink === 1n
      && pinnedBefore.isFile()
      && pinnedBefore.nlink === 1n
      && deepEqual(privateInputStatIdentity(before), pinnedIdentity)
      && deepEqual(privateInputStatIdentity(pinnedAfter), pinnedIdentity)
      && deepEqual(privateInputStatIdentity(after), pinnedIdentity)
      && isOutside(resolvedRelease, resolvedInput)
      && isOutside(fs.realpathSync(resolvedRelease), realInput),
      `${label} must remain a stable single-link regular file outside the distributable release root`
    );
    return {
      label,
      path: resolvedInput,
      realPath: realInput,
      stat: pinnedIdentity,
      content: {
        size: bytes.length,
        sha256: sha256Bytes(bytes)
      }
    };
  } finally {
    fs.closeSync(descriptor);
  }
}

function assertPrivateInputClosure(closure, releaseRoot, stage) {
  const current = capturePrivateInputClosure(
    closure.path,
    releaseRoot,
    closure.label
  );
  assertCondition(
    deepEqual(current, closure),
    `${closure.label} changed ${stage}`
  );
}

function assertManifestPortable(manifest, repoRoot, releaseRoot) {
  const serialized = canonicalJson(manifest);
  assertCondition(
    !serialized.includes(path.resolve(repoRoot))
    && !serialized.includes(path.resolve(releaseRoot))
    && !/(?:^|["'])\/(?:Users|private|tmp|var|Volumes)\//.test(serialized),
    'Public evidence manifest contains a workstation-absolute path'
  );
}

function privatePayloadRoot(releaseRoot) {
  return path.join(path.resolve(releaseRoot), PAYLOAD_DIRECTORY);
}

function openPrivatePayloadDirectory(releaseRoot) {
  const release = path.resolve(releaseRoot);
  const payloadRoot = privatePayloadRoot(release);
  const before = fs.lstatSync(payloadRoot, { bigint: true });
  const descriptor = fs.openSync(
    payloadRoot,
    fs.constants.O_RDONLY
      | (fs.constants.O_NOFOLLOW || 0)
      | (fs.constants.O_DIRECTORY || 0)
  );
  try {
    const pinned = fs.fstatSync(descriptor, { bigint: true });
    const releaseRealPath = fs.realpathSync(release);
    const realPath = fs.realpathSync(payloadRoot);
    const after = fs.lstatSync(payloadRoot, { bigint: true });
    const pinnedIdentity = privateInputStatIdentity(pinned);
    assertCondition(
      !before.isSymbolicLink()
      && before.isDirectory()
      && pinned.isDirectory()
      && !after.isSymbolicLink()
      && after.isDirectory()
      && (pinned.mode & 0o777n) === 0o700n
      && deepEqual(privateInputStatIdentity(before), pinnedIdentity)
      && deepEqual(privateInputStatIdentity(after), pinnedIdentity)
      && path.relative(releaseRealPath, realPath) === PAYLOAD_DIRECTORY,
      'Private evidence payload must remain the fixed real 0700 directory'
    );
    return {
      descriptor,
      path: payloadRoot,
      realPath,
      releaseRealPath,
      stat: pinnedIdentity
    };
  } catch (error) {
    fs.closeSync(descriptor);
    throw error;
  }
}

function assertPrivatePayloadDirectoryClosure(closure, stage) {
  const pinned = fs.fstatSync(closure.descriptor, { bigint: true });
  const current = fs.lstatSync(closure.path, { bigint: true });
  assertCondition(
    pinned.isDirectory()
    && !current.isSymbolicLink()
    && current.isDirectory()
    && (pinned.mode & 0o777n) === 0o700n
    && (current.mode & 0o777n) === 0o700n
    && deepEqual(privateInputStatIdentity(pinned), closure.stat)
    && deepEqual(privateInputStatIdentity(current), closure.stat)
    && fs.realpathSync(closure.path) === closure.realPath
    && fs.realpathSync(path.dirname(closure.path)) === closure.releaseRealPath,
    `Private evidence payload directory changed ${stage}`
  );
}

function sealPrivatePayloadFile(filePath, label) {
  const descriptor = fs.openSync(
    filePath,
    fs.constants.O_RDONLY | (fs.constants.O_NOFOLLOW || 0)
  );
  try {
    const stat = fs.fstatSync(descriptor, { bigint: true });
    assertCondition(stat.isFile(), `${label} must be a real file`);
    fs.fchmodSync(descriptor, 0o600);
    const sealed = fs.fstatSync(descriptor, { bigint: true });
    const pathStat = fs.lstatSync(filePath, { bigint: true });
    assertCondition(
      !pathStat.isSymbolicLink()
      && pathStat.isFile()
      && sealed.dev === stat.dev
      && sealed.ino === stat.ino
      && pathStat.dev === sealed.dev
      && pathStat.ino === sealed.ino
      && sealed.nlink === 1n
      && pathStat.nlink === 1n
      && (sealed.mode & 0o777n) === 0o600n
      && (pathStat.mode & 0o777n) === 0o600n,
      `${label} could not be sealed as a private single-link file`
    );
  } finally {
    fs.closeSync(descriptor);
  }
}

function canonicalPayloadIdentities(releaseRoot) {
  const release = path.resolve(releaseRoot);
  const payloadDirectory = openPrivatePayloadDirectory(release);
  const payloadRoot = payloadDirectory.path;
  try {
    const identities = stableFlatDirectoryFileIdentitiesNoFollow(
      payloadRoot,
      payloadDirectory,
      Object.values(PAYLOAD_FILES),
      'Private evidence payload'
    );
    const identity = (filename, label) => {
      const value = identities[filename];
      assertCondition(
        value
        && Number.isSafeInteger(value.size)
        && /^[0-9a-f]{64}$/.test(value.sha256),
        `${label} is absent from the descriptor-pinned private payload`
      );
      return {
        path: `${PAYLOAD_DIRECTORY}/${filename}`,
        ...value
      };
    };
    const payload = {
      summary: identity(PAYLOAD_FILES.summary, 'Private M110 summary payload'),
      routeProof: identity(PAYLOAD_FILES.route, 'Private M110 route payload'),
      cp05Gate: identity(PAYLOAD_FILES.cp05Gate, 'Private CP05 gate payload'),
      cp05Evidence: identity(PAYLOAD_FILES.cp05Evidence, 'Private CP05 evidence payload'),
      runnerSource: identity(PAYLOAD_FILES.runnerSource, 'Private committed runner payload'),
      backendSource: identity(PAYLOAD_FILES.backendSource, 'Private committed backend payload'),
      m006Contract: identity(
        PAYLOAD_FILES.m006Contract,
        'Private committed M006 contract payload'
      ),
      sourceMembership: identity(
        PAYLOAD_FILES.sourceMembership,
        'Private committed source membership payload'
      )
    };
    assertPrivatePayloadDirectoryClosure(
      payloadDirectory,
      'during canonical identity capture'
    );
    return payload;
  } finally {
    fs.closeSync(payloadDirectory.descriptor);
  }
}

function bundleHash(manifest) {
  const value = JSON.parse(JSON.stringify(manifest));
  delete value.bundleSha256;
  return `sha256:${canonicalSha256(value)}`;
}

export function createPublicReleaseEvidenceManifest({
  generatedAt,
  source,
  releaseEvidence,
  cp05,
  m110,
  payload,
  sourceMembership
}) {
  const manifest = {
    schema: 'automixer.public_release_evidence_bundle.v1',
    pathPolicy: 'release_root_relative_v1',
    generatedAt,
    status: 'structural_consistency_verified',
    evidenceComplete: false,
    structuralConsistencyVerified: true,
    externalAuthenticityVerified: false,
    externalAuthenticatedAttestationRequired: true,
    readinessClosureAllowed: false,
    authenticityBoundary: 'self_consistent_detached_evidence_without_public_trust_anchor',
    distribution: {
      manifest: 'public_sanitized_projection',
      retainedPayload: 'private_audit_evidence_do_not_publish_raw'
    },
    source: { ...source },
    packageRelease: {
      releaseManifest: releaseEvidence.manifestIdentity,
      generatedAt: releaseEvidence.generatedAt,
      detachedEvidenceReplayComplete: releaseEvidence.detachedEvidenceReplayComplete,
      structuralConsistencyVerified: true,
      externalAuthenticityVerified: false,
      externalAuthenticatedAttestationRequired: true,
      nativeTrustReexecuted: releaseEvidence.nativeTrustReexecuted,
      gatekeeperReexecuted: releaseEvidence.gatekeeperReexecuted,
      archiveContentsReexecuted: releaseEvidence.archiveContentsReexecuted
    },
    cp05: {
      ...cp05,
      sourceBindingRecomputedAtBuild: true,
      detachedSourceBindingRecomputed: false,
      gate: payload.cp05Gate,
      softwareEvidence: payload.cp05Evidence
    },
    console: {
      identitySha256: `sha256:${canonicalSha256({
        domain: 'automixer.public.console_identity.v1',
        serial: m110.console.serial
      })}`,
      firmware: m110.console.firmware,
      rawIdentityRetainedPrivately: true
    },
    capture: {
      deviceClass: 'WING_USB_48x48',
      hardwareIdentitySha256: `sha256:${canonicalSha256({
        domain: 'automixer.public.capture_hardware_identity.v1',
        hardwareUid: m110.capture.hardwareUid
      })}`,
      conservativeTruePeakDbtp: m110.capture.conservativeTruePeakDbtp,
      headroomCeilingDbtp: m110.capture.headroomCeilingDbtp,
      rawIdentityRetainedPrivately: true
    },
    soundcheck: {
      claimedM110Status: m110.status,
      structuralEvidenceReplayComplete: true,
      runId: m110.runId,
      verifiedStateSha256: m110.verifiedStateSha256,
      committedSurfaceLedgerSha256: m110.committedSurfaceLedgerSha256,
      routeProofId: m110.route.proofId,
      routeSemanticSha256: m110.route.semanticSha256,
      liveAuthorityGranted: false,
      readinessClosureAllowed: false,
      externalAuthenticityVerified: false,
      teardownClaimStructurallyVerified: true,
      physicalListeningClaimStructurallyVerified: true
    },
    retained: {
      m110ExecutionSummary: payload.summary,
      routeProof: payload.routeProof,
      committedSources: {
        runner: { logicalPath: SOURCE_BLOBS.runner, artifact: payload.runnerSource },
        backendServer: { logicalPath: SOURCE_BLOBS.backendServer, artifact: payload.backendSource },
        m006Contract: { logicalPath: SOURCE_BLOBS.m006Contract, artifact: payload.m006Contract },
        membershipProof: {
          artifact: payload.sourceMembership,
          schema: sourceMembership.schema,
          objectFormat: sourceMembership.objectFormat,
          sourceHead: sourceMembership.sourceHead,
          sourceTree: sourceMembership.sourceTree,
          pathCount: sourceMembership.pathCount
        }
      }
    },
    bundleSha256: null
  };
  manifest.bundleSha256 = bundleHash(manifest);
  return manifest;
}

function copyInput(inputClosure, releaseRoot, payloadRoot, filename, label) {
  assertPrivateInputClosure(inputClosure, releaseRoot, 'before stable read');
  const stable = readStableJsonFileNoFollow(
    inputClosure.path,
    `${label} source`
  );
  assertCondition(
    identitiesEqual(stable.identity, inputClosure.content),
    `${label} stable read does not match the descriptor-pinned raw input bytes`
  );
  assertPrivateInputClosure(inputClosure, releaseRoot, 'after stable read');
  const copied = writeNewArtifactFileNoFollow(
    payloadRoot,
    path.join(payloadRoot, filename),
    stable.bytes,
    label
  );
  assertCondition(
    copied.size === stable.identity.size
    && copied.sha256 === stable.identity.sha256,
    `${label} copied bytes do not match the pinned source`
  );
  assertPrivateInputClosure(inputClosure, releaseRoot, 'after private snapshot write');
  return copied;
}

function buildPublicReleaseEvidenceBundleCore({
  repoRoot,
  releaseRoot,
  m110SummaryPath,
  routeProofPath,
  generatedAt = new Date().toISOString()
}, {
  verifyReleaseEvidence,
  publishManifest = publishNewArtifactFileNoFollow
}) {
  const repository = ensureRealDirectory(repoRoot, 'Repository root');
  const release = ensureRealDirectory(releaseRoot, 'Release root');
  const rawInputClosures = [
    capturePrivateInputClosure(
      m110SummaryPath,
      release,
      'Raw M110 execution summary input'
    ),
    capturePrivateInputClosure(
      routeProofPath,
      release,
      'Raw M110 route proof input'
    )
  ];
  const assertRawInputClosures = stage => {
    for (const closure of rawInputClosures) {
      assertPrivateInputClosure(closure, release, stage);
    }
  };
  const initialSource = sourceSnapshot(repository);
  pinReleaseEvidenceAuthorityHelpers(repository, initialSource.tree);
  const initialAuthority = assertExecutingAuthorityMatchesSource(
    repository,
    initialSource
  );
  const assertCurrentAuthorityClosure = stage => {
    const currentSource = sourceSnapshot(repository);
    const currentAuthority = assertExecutingAuthorityMatchesSource(
      repository,
      currentSource
    );
    assertCondition(
      deepEqual(currentSource, initialSource)
      && deepEqual(currentAuthority, initialAuthority),
      `Source or executing release authority changed ${stage}`
    );
    return { source: currentSource, authority: currentAuthority };
  };
  parseIso(generatedAt, 'Bundle generatedAt');
  const payloadRoot = path.join(release, PAYLOAD_DIRECTORY);
  const finalManifestPath = path.join(release, PUBLIC_MANIFEST);
  const candidateManifestPath = path.join(release, CANDIDATE_MANIFEST);
  for (const candidate of [payloadRoot, finalManifestPath, candidateManifestPath]) {
    assertCondition(!fs.existsSync(candidate), `Refusing to reuse existing bundle path: ${path.basename(candidate)}`);
  }
  fs.mkdirSync(payloadRoot, { mode: 0o700 });
  fs.chmodSync(payloadRoot, 0o700);

  const copiedSummary = copyInput(
    rawInputClosures[0],
    release,
    payloadRoot,
    PAYLOAD_FILES.summary,
    'Private M110 execution summary snapshot'
  );
  const copiedRoute = copyInput(
    rawInputClosures[1],
    release,
    payloadRoot,
    PAYLOAD_FILES.route,
    'Private M110 route proof snapshot'
  );
  const copiedGate = writeGitBlob(
    repository,
    initialSource.tree,
    CP05_GATE_PATH,
    payloadRoot,
    path.join(payloadRoot, PAYLOAD_FILES.cp05Gate),
    'CP05 hardware gate snapshot'
  );
  const copiedEvidence = writeGitBlob(
    repository,
    initialSource.tree,
    CP05_EVIDENCE_PATH,
    payloadRoot,
    path.join(payloadRoot, PAYLOAD_FILES.cp05Evidence),
    'CP05 software evidence snapshot'
  );
  const copiedRunnerSource = writeGitBlob(
    repository,
    initialSource.tree,
    SOURCE_BLOBS.runner,
    payloadRoot,
    path.join(payloadRoot, PAYLOAD_FILES.runnerSource),
    'Committed M110 runner source snapshot'
  );
  const copiedBackendSource = writeGitBlob(
    repository,
    initialSource.tree,
    SOURCE_BLOBS.backendServer,
    payloadRoot,
    path.join(payloadRoot, PAYLOAD_FILES.backendSource),
    'Committed backend server source snapshot'
  );
  const copiedM006Contract = writeGitBlob(
    repository,
    initialSource.tree,
    SOURCE_BLOBS.m006Contract,
    payloadRoot,
    path.join(payloadRoot, PAYLOAD_FILES.m006Contract),
    'Committed M006 restore contract snapshot'
  );
  const sourceMembershipProof = buildGitSourceMembershipProof(repository, initialSource);
  const copiedSourceMembership = writeNewArtifactFileNoFollow(
    payloadRoot,
    path.join(payloadRoot, PAYLOAD_FILES.sourceMembership),
    `${canonicalJson(sourceMembershipProof)}\n`,
    'Committed source membership proof snapshot'
  );
  for (const [label, copied] of Object.entries({
    summary: copiedSummary,
    route: copiedRoute,
    cp05Gate: copiedGate,
    cp05Evidence: copiedEvidence,
    runnerSource: copiedRunnerSource,
    backendSource: copiedBackendSource,
    m006Contract: copiedM006Contract,
    sourceMembership: copiedSourceMembership
  })) {
    sealPrivatePayloadFile(copied.path, `Private bundle ${label}`);
  }

  const releaseEvidence = verifyReleaseEvidence(release);
  assertRawInputClosures('after retained package replay');
  assertCondition(
    releaseEvidence.source.head === initialSource.head
    && releaseEvidence.source.tree === initialSource.tree,
    'Retained package release source does not match the clean checkout'
  );
  const gate = readStableInputJson(copiedGate.path, 'Copied CP05 hardware gate');
  const evidence = readStableInputJson(copiedEvidence.path, 'Copied CP05 software evidence');
  const sourceBinding = computeCp05SourceBinding(repository, initialSource.tree);
  const cp05 = validateCp05Markers({
    gate: gate.value,
    gateBytes: gate.bytes,
    evidence: evidence.value,
    evidenceBytes: evidence.bytes,
    sourceBinding
  });
  const summary = readStableInputJson(copiedSummary.path, 'Copied M110 execution summary');
  assertPythonCanonicalJsonBytes(
    summary.value,
    summary.bytes,
    'Copied M110 execution summary'
  );
  const route = readStableInputJson(copiedRoute.path, 'Copied M110 route proof');
  const m006Contract = readStableInputJson(
    copiedM006Contract.path,
    'Copied immutable M006 restore contract'
  );
  const m110 = validateM110AttendedEvidence({
    summary: summary.value,
    routeProof: route.value,
    routeBytes: route.bytes,
    routeIdentity: route.identity,
    source: initialSource,
    cp05,
    m006Contract: {
      contract: m006Contract.value,
      contractIdentity: m006Contract.identity
    }
  });
  assertCondition(
    path.basename(path.resolve(m110SummaryPath)) === 'execution-summary.json',
    'M110 input must be execution-summary.json, never a plan/HW01 summary'
  );
  const payload = canonicalPayloadIdentities(release);
  const retainedSourceMembership = validateRetainedGitSourceMembership(
    release,
    payload,
    initialSource,
    'Bundled'
  );
  const manifest = createPublicReleaseEvidenceManifest({
    generatedAt,
    source: initialSource,
    releaseEvidence,
    cp05,
    m110,
    payload,
    sourceMembership: retainedSourceMembership.verification
  });
  assertManifestPortable(manifest, repository, release);
  assertCurrentAuthorityClosure('during bundle construction');
  const finalReleaseEvidence = verifyReleaseEvidence(release);
  assertCondition(
    identitiesEqual(finalReleaseEvidence.manifestIdentity, releaseEvidence.manifestIdentity)
    && finalReleaseEvidence.source.head === initialSource.head
    && finalReleaseEvidence.source.tree === initialSource.tree,
    'Retained package evidence changed during bundle construction'
  );
  for (const [name, recorded] of Object.entries(payload)) {
    const current = artifactFileIdentity(
      release,
      resolveArtifactPath(release, recorded.path, `Final bundle payload ${name}`),
      `Final bundle payload ${name}`
    );
    assertCondition(identitiesEqual(recorded, current), `Bundle payload changed: ${name}`);
  }

  let candidate = null;
  try {
    assertRawInputClosures('before candidate manifest creation');
    candidate = writeNewArtifactFileNoFollow(
      release,
      candidateManifestPath,
      `${JSON.stringify(manifest, null, 2)}\n`,
      'Public-release evidence manifest candidate'
    );
    const candidateVerification = verifyPublicReleaseEvidenceManifest(
      release,
      CANDIDATE_MANIFEST,
      { verifyReleaseEvidence }
    );
    assertCondition(
      identitiesEqual(candidate, candidateVerification.manifestIdentity),
      'Pre-publication verification did not retain the exact candidate bytes'
    );
    assertRawInputClosures('before manifest publication');
    assertCurrentAuthorityClosure('before manifest publication');
    publishManifest(
      release,
      candidateManifestPath,
      finalManifestPath,
      candidate,
      'Public-release evidence manifest publication'
    );
    const publishedVerification = verifyPublicReleaseEvidenceManifest(
      release,
      PUBLIC_MANIFEST,
      { verifyReleaseEvidence }
    );
    assertCurrentAuthorityClosure('after final published-manifest replay');
    assertRawInputClosures('after final published-manifest replay');
    const returnReleaseEvidence = verifyReleaseEvidence(release);
    assertCondition(
      deepEqual(finalReleaseEvidence, returnReleaseEvidence),
      'Retained package evidence changed at bundle return cut'
    );
    assertRawInputClosures('at bundle return cut');
    const finalPublishedPayload = canonicalPayloadIdentities(release);
    assertCondition(
      deepEqual(payload, finalPublishedPayload),
      'Private evidence payload changed after final published-manifest replay'
    );
    const finalPublishedManifest = artifactFileIdentity(
      release,
      finalManifestPath,
      'Final public-release evidence manifest return cut'
    );
    assertCondition(
      identitiesEqual(publishedVerification.manifestIdentity, finalPublishedManifest),
      'Public-release evidence manifest changed after final published-manifest replay'
    );
    return publishedVerification;
  } catch (error) {
    const cleanupErrors = [];
    const removeUnchangedManifest = (manifestPath, label) => {
      if (!fs.lstatSync(manifestPath, { throwIfNoEntry: false })) return;
      if (!candidate) {
        throw new Error(`${label} has no immutable candidate identity`);
      }
      const current = artifactFileIdentity(release, manifestPath, label);
      if (!identitiesEqual(current, candidate)) {
        throw new Error(`${label} refuses to remove bytes that differ from the candidate`);
      }
      removeArtifactFileNoFollow(
        release,
        manifestPath,
        label,
        candidate
      );
    };
    for (const [manifestPath, label] of [
      [finalManifestPath, 'Failed public-release manifest cleanup'],
      [candidateManifestPath, 'Failed public-release candidate cleanup']
    ]) {
      try {
        removeUnchangedManifest(manifestPath, label);
      } catch (cleanupFailure) {
        cleanupErrors.push(cleanupFailure);
      }
    }
    if (cleanupErrors.length > 0) {
      throw new AggregateError(
        [error, ...cleanupErrors],
        'Public-release evidence verification failed and immutable cleanup was refused'
      );
    }
    throw error;
  }
}

export function buildPublicReleaseEvidenceBundle(options) {
  assertProductionRuntimeEnvironment();
  assertCondition(options && typeof options === 'object', 'Bundle build options are required');
  const selectedRepository = ensureRealDirectory(options.repoRoot, 'Repository root');
  const authorityRepository = ensureRealDirectory(
    executingRepositoryRoot,
    'Executing bundle authority repository'
  );
  assertCondition(
    fs.realpathSync(selectedRepository) === fs.realpathSync(authorityRepository),
    'Production bundle authority must execute from the selected source repository'
  );
  return buildPublicReleaseEvidenceBundleCore(options, {
    verifyReleaseEvidence: verifyRetainedReleaseManifestEvidence
  });
}

function verifyPublicReleaseEvidenceManifest(
  releaseRoot,
  manifestName,
  { verifyReleaseEvidence }
) {
  const release = ensureRealDirectory(releaseRoot, 'Release root');
  assertCondition(
    [PUBLIC_MANIFEST, CANDIDATE_MANIFEST].includes(manifestName),
    'Unsupported public-release evidence manifest name'
  );
  const manifestPath = path.join(release, manifestName);
  const stableManifest = readStableJsonFileNoFollow(
    manifestPath,
    'Public-release evidence manifest',
    { containmentRoot: release }
  );
  const manifestIdentity = artifactFileIdentity(
    release,
    manifestPath,
    'Public-release evidence manifest'
  );
  assertCondition(
    identitiesEqual(stableManifest.identity, manifestIdentity),
    'Public-release evidence manifest changed during read'
  );
  const manifest = stableManifest.value;
  assertCondition(
    stableManifest.bytes.equals(
      Buffer.from(`${JSON.stringify(manifest, null, 2)}\n`, 'utf8')
    ),
    'Public-release evidence manifest is not the exact canonical serialized bytes'
  );
  parseIso(manifest?.generatedAt, 'Bundle generatedAt');
  assertCondition(
    manifest?.schema === 'automixer.public_release_evidence_bundle.v1'
    && manifest.pathPolicy === 'release_root_relative_v1'
    && manifest.status === 'structural_consistency_verified'
    && manifest.evidenceComplete === false
    && manifest.structuralConsistencyVerified === true
    && manifest.externalAuthenticityVerified === false
    && manifest.externalAuthenticatedAttestationRequired === true
    && manifest.readinessClosureAllowed === false
    && manifest.authenticityBoundary
      === 'self_consistent_detached_evidence_without_public_trust_anchor'
    && sha256Prefixed(manifest.bundleSha256)
    && manifest.bundleSha256 === bundleHash(manifest)
    && manifest.distribution?.manifest === 'public_sanitized_projection'
    && manifest.distribution?.retainedPayload === 'private_audit_evidence_do_not_publish_raw'
    && sourceObjectId(manifest.source?.head)
    && sourceObjectId(manifest.source?.tree)
    && manifest.source?.cleanCheckout === true,
    'Public-release evidence manifest contract or bundle hash is invalid'
  );
  assertManifestPortable(manifest, release, release);
  const packageRelease = verifyReleaseEvidence(release);
  const payload = canonicalPayloadIdentities(release);
  const gate = readPayloadJson(release, payload.cp05Gate, 'Bundled CP05 hardware gate');
  const evidence = readPayloadJson(
    release,
    payload.cp05Evidence,
    'Bundled CP05 software evidence'
  );
  const cp05 = validateCp05Markers({
    gate: gate.value,
    gateBytes: gate.bytes,
    evidence: evidence.value,
    evidenceBytes: evidence.bytes,
    sourceBinding: {
      schema: 'automixer.m110_cp05_git_tree_binding.v1',
      sha256: manifest.cp05?.source_binding_sha256,
      tracked_entry_count: manifest.cp05?.source_binding_tracked_entry_count
    }
  });
  const summary = readPayloadJson(
    release,
    payload.summary,
    'Bundled M110 execution summary'
  );
  assertPythonCanonicalJsonBytes(
    summary.value,
    summary.bytes,
    'Bundled M110 execution summary'
  );
  const route = readPayloadJson(
    release,
    payload.routeProof,
    'Bundled M110 route proof'
  );
  const m006Contract = readPayloadJson(
    release,
    payload.m006Contract,
    'Bundled immutable M006 restore contract'
  );
  const m110 = validateM110AttendedEvidence({
    summary: summary.value,
    routeProof: route.value,
    routeBytes: route.bytes,
    routeIdentity: route.identity,
    source: packageRelease.source,
    cp05,
    m006Contract: {
      contract: m006Contract.value,
      contractIdentity: m006Contract.identity
    }
  });
  const retainedSourceMembership = validateRetainedGitSourceMembership(
    release,
    payload,
    packageRelease.source,
    'Bundled'
  );
  const expectedManifest = createPublicReleaseEvidenceManifest({
    generatedAt: manifest.generatedAt,
    source: packageRelease.source,
    releaseEvidence: packageRelease,
    cp05,
    m110,
    payload,
    sourceMembership: retainedSourceMembership.verification
  });
  assertCondition(
    stableManifest.bytes.equals(
      Buffer.from(`${JSON.stringify(expectedManifest, null, 2)}\n`, 'utf8')
    ),
    'Public-release evidence manifest is not the exact builder serialization'
  );
  assertCondition(
    deepEqual(manifest, expectedManifest),
    'Public-release evidence projection drifted from replayed private evidence'
  );
  for (const [name, retained] of Object.entries({
    gate,
    evidence,
    summary,
    route,
    m006Contract,
    sourceMembership: retainedSourceMembership.membership,
    ...retainedSourceMembership.retained
  })) {
    const current = artifactFileIdentity(
      release,
      retained.path,
      `Final bundled input ${name}`
    );
    assertCondition(
      identitiesEqual(retained.identity, current),
      `Bundled input changed during detached verification: ${name}`
    );
  }
  const finalPayload = canonicalPayloadIdentities(release);
  assertCondition(
    deepEqual(payload, finalPayload),
    'Private evidence payload set or identities changed during detached verification'
  );
  const finalSourceMembership = validateRetainedGitSourceMembership(
    release,
    finalPayload,
    packageRelease.source,
    'Final bundled'
  );
  assertCondition(
    deepEqual(
      retainedSourceMembership.verification,
      finalSourceMembership.verification
    ),
    'Committed source membership proof changed during detached verification'
  );
  const finalManifestIdentity = artifactFileIdentity(
    release,
    manifestPath,
    'Final public-release evidence manifest'
  );
  assertCondition(
    identitiesEqual(manifestIdentity, finalManifestIdentity),
    'Public-release evidence manifest changed during detached verification'
  );
  const returnPackageRelease = verifyReleaseEvidence(release);
  assertCondition(
    deepEqual(packageRelease, returnPackageRelease),
    'Retained package evidence changed at detached verification return cut'
  );
  const returnPayload = canonicalPayloadIdentities(release);
  assertCondition(
    deepEqual(finalPayload, returnPayload),
    'Private evidence payload changed at detached verification return cut'
  );
  const returnManifestIdentity = artifactFileIdentity(
    release,
    manifestPath,
    'Public-release evidence manifest return cut'
  );
  assertCondition(
    identitiesEqual(finalManifestIdentity, returnManifestIdentity),
    'Public-release evidence manifest changed at detached verification return cut'
  );
  return {
    verified: true,
    status: 'structural_consistency_verified',
    evidenceComplete: false,
    structuralConsistencyVerified: true,
    externalAuthenticityVerified: false,
    externalAuthenticatedAttestationRequired: true,
    readinessClosureAllowed: false,
    source: { ...manifest.source },
    console: { ...manifest.console },
    soundcheck: { ...manifest.soundcheck },
    manifestIdentity: returnManifestIdentity,
    cp05SourceBindingRecomputed: false,
    applicationProcessStarted: false,
    backendProcessStarted: false,
    liveAuthorityGranted: false
  };
}

export function verifyPublicReleaseEvidenceBundle(releaseRoot) {
  assertProductionRuntimeEnvironment();
  if (!releaseEvidenceAuthorityPin) {
    const repository = ensureRealDirectory(
      executingRepositoryRoot,
      'Executing bundle authority repository'
    );
    const source = sourceSnapshot(repository);
    pinReleaseEvidenceAuthorityHelpers(repository, source.tree);
    assertExecutingAuthorityMatchesSource(repository, source);
  }
  return verifyPublicReleaseEvidenceManifest(releaseRoot, PUBLIC_MANIFEST, {
    verifyReleaseEvidence: verifyRetainedReleaseManifestEvidence
  });
}

// The native package verifier is intentionally not replaceable through the
// public build/verify functions or the CLI.  Tests use this separate hook to
// exercise relocation, publication, and cleanup around a deterministic
// package-proof stub without weakening the production authority path.
export const publicReleaseEvidenceTestHooks = Object.freeze({
  authoritySourcePaths: Object.freeze([...AUTHORITY_SOURCE_PATHS]),
  pythonCanonicalFloat(value) {
    return pythonCanonicalJsonFloat(value);
  },
  validatePythonCanonicalJson(value, bytes, { finalNewline = true } = {}) {
    const parsed = parseCompactCanonicalJsonBytes(bytes, 'Test Python canonical JSON', {
      pythonWriter: true,
      finalNewline
    });
    assertCondition(
      deepEqual(parsed.value, value),
      'Test Python canonical JSON parsed value does not match its retained bytes'
    );
    return true;
  },
  validateAuthority(repoRoot) {
    const repository = ensureRealDirectory(repoRoot, 'Test authority repository');
    const source = sourceSnapshot(repository);
    pinReleaseEvidenceAuthorityHelpers(repository, source.tree);
    return assertExecutingAuthorityMatchesSource(repository, source);
  },
  build(options, verifyReleaseEvidence, { publishManifest } = {}) {
    assertCondition(typeof verifyReleaseEvidence === 'function', 'Test package verifier is required');
    assertCondition(
      publishManifest === undefined || typeof publishManifest === 'function',
      'Test manifest publisher must be a function'
    );
    return buildPublicReleaseEvidenceBundleCore(options, {
      verifyReleaseEvidence,
      publishManifest: publishManifest || publishNewArtifactFileNoFollow
    });
  },
  verify(releaseRoot, verifyReleaseEvidence) {
    assertCondition(typeof verifyReleaseEvidence === 'function', 'Test package verifier is required');
    return verifyPublicReleaseEvidenceManifest(releaseRoot, PUBLIC_MANIFEST, {
      verifyReleaseEvidence
    });
  },
  snapshotGitBlob(options) {
    return writeGitBlob(
      options.repoRoot,
      options.sourceTree,
      options.sourcePath,
      options.destinationRoot,
      options.destinationPath,
      options.label || 'Test committed blob snapshot'
    );
  }
});

function parseArgs(argv) {
  const command = argv[0];
  assertCondition(['build', 'verify'].includes(command), 'Expected build or verify command');
  const values = { command };
  for (let index = 1; index < argv.length; index += 2) {
    const name = argv[index];
    const value = argv[index + 1];
    assertCondition(name?.startsWith('--') && value !== undefined, `Invalid argument: ${name || ''}`);
    assertCondition(values[name.slice(2)] === undefined, `Duplicate argument: ${name}`);
    values[name.slice(2)] = value;
  }
  const allowed = command === 'build'
    ? ['repo-root', 'release-root', 'm110-summary', 'route-proof']
    : ['release-root'];
  assertCondition(
    Object.keys(values).filter(name => name !== 'command').every(name => allowed.includes(name)),
    'Unknown public-release evidence argument'
  );
  for (const name of allowed) {
    assertCondition(typeof values[name] === 'string' && path.isAbsolute(values[name]), `--${name} must be absolute`);
  }
  return values;
}

function main() {
  assertCondition(
    process.env.AUTOMIXER_PUBLIC_RELEASE_EVIDENCE_CLEAN_ENV === '1',
    'Direct public-release evidence Node entry is forbidden; use the fixed shell launcher'
  );
  const args = parseArgs(process.argv.slice(2));
  const result = args.command === 'build'
    ? buildPublicReleaseEvidenceBundle({
      repoRoot: args['repo-root'],
      releaseRoot: args['release-root'],
      m110SummaryPath: args['m110-summary'],
      routeProofPath: args['route-proof']
    })
    : verifyPublicReleaseEvidenceBundle(args['release-root']);
  process.stdout.write(`${JSON.stringify(result, null, 2)}\n`);
}

if (path.resolve(process.argv[1] || '') === path.resolve(scriptPath)) {
  try {
    main();
  } catch (error) {
    process.stderr.write(`public_release_evidence_failed:${error?.message || error}\n`);
    process.exitCode = 1;
  }
}
