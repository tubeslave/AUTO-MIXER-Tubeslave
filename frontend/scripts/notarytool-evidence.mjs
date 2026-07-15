import fs from 'node:fs';
import path from 'node:path';

import {
  artifactFileIdentity,
  assertIdentityUnchanged,
  writeNewArtifactFileNoFollow
} from './release-evidence.mjs';

const UUID_PATTERN = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;
const SHA256_PATTERN = /^[0-9a-f]{64}$/;
const CDHASH_PATTERN = /^[0-9a-f]{40,64}$/;
const MAX_NOTARY_JSON_BYTES = 2 * 1024 * 1024;
const NOTARY_WAIT_TIMEOUT = '30m';

function outputBytes(value) {
  if (Buffer.isBuffer(value)) return Buffer.from(value);
  return Buffer.from(String(value || ''), 'utf8');
}

function isoTimestamp(value, label) {
  const text = value instanceof Date ? value.toISOString() : String(value || '');
  const epoch = Date.parse(text);
  if (!Number.isFinite(epoch) || new Date(epoch).toISOString() !== text) {
    throw new Error(`${label} must be an exact ISO-8601 UTC timestamp`);
  }
  return { text, epoch };
}

function appleUtcTimestamp(value, label) {
  const text = String(value || '');
  const epoch = Date.parse(text);
  if (
    !/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,9})?Z$/.test(text)
    || !Number.isFinite(epoch)
  ) {
    throw new Error(`${label} must be an RFC3339 UTC timestamp`);
  }
  return { text, epoch };
}

function nextTimestamp(clock, label) {
  return isoTimestamp(clock(), label).text;
}

export function normalizeNotarySha256(value, label = 'Notary SHA-256') {
  const normalized = String(value || '').trim().toLowerCase();
  if (!SHA256_PATTERN.test(normalized)) {
    throw new Error(`${label} is not a 64-character SHA-256 digest`);
  }
  return normalized;
}

export function normalizeCdHash(value, label = 'Code Directory hash') {
  const normalized = String(value || '').trim().toLowerCase();
  if (!CDHASH_PATTERN.test(normalized)) {
    throw new Error(`${label} is not a supported hexadecimal CDHash`);
  }
  return normalized;
}

export function parseCodesignCdHash(output) {
  const matches = [...String(output || '').matchAll(/^CDHash=([0-9a-f]+)$/gmi)];
  if (matches.length !== 1) {
    throw new Error(`Expected exactly one codesign CDHash, found ${matches.length}`);
  }
  return normalizeCdHash(matches[0][1], 'codesign CDHash');
}

export function parseNotarytoolJson(rawOutput, label = 'notarytool JSON') {
  const bytes = outputBytes(rawOutput);
  if (bytes.length === 0 || bytes.length > MAX_NOTARY_JSON_BYTES) {
    throw new Error(`${label} size is outside the accepted range`);
  }
  if (bytes.includes(0)) throw new Error(`${label} contains a NUL byte`);
  let value;
  try {
    value = JSON.parse(bytes.toString('utf8'));
  } catch (error) {
    throw new Error(`${label} is not valid JSON: ${error.message}`);
  }
  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    throw new Error(`${label} must contain one JSON object`);
  }
  return { bytes, value };
}

export function assertAcceptedNotaryLog(log, context) {
  const started = isoTimestamp(context.startedAt, 'Notary submission start');
  const completed = isoTimestamp(context.completedAt, 'Notary submission completion');
  const upload = appleUtcTimestamp(log?.uploadDate, 'Apple notary log uploadDate');
  const expectedSha256 = normalizeNotarySha256(
    context.archiveSha256,
    'Submitted archive SHA-256'
  );
  const actualSha256 = normalizeNotarySha256(log?.sha256, 'Apple notary log SHA-256');
  const archiveFilename = String(context.archiveFilename || '');
  const issuesEmpty = log?.issues === null
    || (Array.isArray(log?.issues) && log.issues.length === 0);
  if (
    completed.epoch < started.epoch
    || !UUID_PATTERN.test(String(context.submissionId || ''))
    || !archiveFilename
    || path.basename(archiveFilename) !== archiveFilename
    || upload.epoch < started.epoch
    || upload.epoch > completed.epoch
    || log?.logFormatVersion !== 1
    || log?.jobId !== context.submissionId
    || log?.status !== 'Accepted'
    || log?.statusCode !== 0
    || log?.archiveFilename !== archiveFilename
    || actualSha256 !== expectedSha256
    || !issuesEmpty
    || !Array.isArray(log?.ticketContents)
  ) {
    throw new Error('Apple notary log does not bind to the accepted submission');
  }
  return {
    logFormatVersion: 1,
    jobId: log.jobId,
    status: 'Accepted',
    statusCode: 0,
    archiveFilename: log.archiveFilename,
    uploadDate: upload.text,
    sha256: actualSha256,
    issuesCount: 0,
    ticketCount: log.ticketContents.length
  };
}

export function assertMainExecutableTicket(log, {
  expectedPathSuffix,
  expectedCdHash,
  architecture = 'arm64',
  expectedArchiveFilename = null
}) {
  const suffix = String(expectedPathSuffix || '').replaceAll('\\', '/');
  if (!suffix || suffix.startsWith('/') || suffix.includes('../')) {
    throw new Error('Expected application ticket path suffix is invalid');
  }
  const archiveFilename = expectedArchiveFilename == null
    ? null
    : String(expectedArchiveFilename);
  if (
    archiveFilename != null
    && (!archiveFilename || path.basename(archiveFilename) !== archiveFilename)
  ) {
    throw new Error('Expected application ticket archive filename is invalid');
  }
  const expectedTicketPath = archiveFilename == null
    ? null
    : `${archiveFilename}/${suffix}`;
  const cdhash = normalizeCdHash(expectedCdHash, 'Expected application CDHash');
  const matches = (log?.ticketContents || []).filter(ticket => {
    const ticketPath = String(ticket?.path || '').replaceAll('\\', '/');
    return ticket?.arch === architecture
      && ticket?.digestAlgorithm === 'SHA-256'
      && (
        expectedTicketPath == null
          ? ticketPath === suffix || ticketPath.endsWith(`/${suffix}`)
          : ticketPath === expectedTicketPath
      )
      && normalizeCdHash(ticket?.cdhash, 'Apple ticket CDHash') === cdhash;
  });
  if (matches.length !== 1) {
    throw new Error(
      `Expected exactly one ${architecture} Apple ticket for the main executable, found ${matches.length}`
    );
  }
  return {
    ...(expectedTicketPath == null ? {} : { path: expectedTicketPath }),
    pathSuffix: suffix,
    architecture,
    digestAlgorithm: 'SHA-256',
    cdhash
  };
}

export function assertDmgTicket(log, {
  expectedPath,
  expectedCdHash
}) {
  const ticketPath = String(expectedPath || '').replaceAll('\\', '/');
  if (
    !ticketPath
    || ticketPath.startsWith('/')
    || ticketPath.includes('/')
    || ticketPath === '.'
    || ticketPath === '..'
  ) {
    throw new Error('Expected DMG ticket path must be one archive basename');
  }
  const cdhash = normalizeCdHash(expectedCdHash, 'Expected DMG CDHash');
  const matches = (log?.ticketContents || []).filter(ticket => (
    ticket?.path === ticketPath
    && ticket?.digestAlgorithm === 'SHA-256'
    && ticket?.arch == null
    && normalizeCdHash(ticket?.cdhash, 'Apple DMG ticket CDHash') === cdhash
  ));
  if (matches.length !== 1) {
    throw new Error(`Expected exactly one Apple ticket for DMG ${ticketPath}, found ${matches.length}`);
  }
  return {
    path: ticketPath,
    digestAlgorithm: 'SHA-256',
    cdhash
  };
}

export function submitAndRetainNotaryLog({
  artifactRoot,
  submissionPath,
  rawLogPath,
  profile,
  keychain = '',
  runCommand,
  clock = () => new Date()
}) {
  const resolvedRoot = path.resolve(artifactRoot);
  const resolvedSubmission = path.resolve(submissionPath);
  const resolvedLog = path.resolve(rawLogPath);
  if (path.dirname(resolvedLog) !== resolvedRoot) {
    throw new Error('Retained Apple notary log must be a top-level artifact');
  }
  if (fs.lstatSync(resolvedLog, { throwIfNoEntry: false })) {
    throw new Error('Retained Apple notary log already exists; refusing replacement');
  }
  const archiveBefore = artifactFileIdentity(
    resolvedRoot,
    resolvedSubmission,
    'Submitted notarization archive'
  );
  const startedAt = nextTimestamp(clock, 'Notary submission start');
  const submitArgs = [
    'notarytool', 'submit', resolvedSubmission,
    '--keychain-profile', profile,
    '--wait',
    '--timeout', NOTARY_WAIT_TIMEOUT,
    '--output-format', 'json'
  ];
  if (keychain) submitArgs.push('--keychain', keychain);
  const submit = runCommand('/usr/bin/xcrun', submitArgs, { check: false });
  const parsedSubmission = parseNotarytoolJson(
    submit?.stdout,
    'notarytool submit response'
  ).value;
  const submissionId = String(parsedSubmission?.id || '');
  if (!UUID_PATTERN.test(submissionId)) {
    throw new Error('notarytool submit response has no valid submission ID');
  }

  const logArgs = [
    'notarytool', 'log', submissionId,
    '--keychain-profile', profile,
    '--output-format', 'json'
  ];
  if (keychain) logArgs.push('--keychain', keychain);
  const logResult = runCommand('/usr/bin/xcrun', logArgs, { check: false });
  const completedAt = nextTimestamp(clock, 'Notary submission completion');
  assertIdentityUnchanged(
    archiveBefore,
    artifactFileIdentity(
      resolvedRoot,
      resolvedSubmission,
      'Post-notary submitted archive'
    ),
    'Submitted notarization archive'
  );
  if (
    submit?.status !== 0
    || parsedSubmission?.status !== 'Accepted'
    || logResult?.status !== 0
  ) {
    throw new Error(
      `Notarization was not accepted or its Apple log was unavailable (status=${parsedSubmission?.status || 'unknown'})`
    );
  }
  const parsedLog = parseNotarytoolJson(logResult.stdout, 'Apple notary log');
  const summary = assertAcceptedNotaryLog(parsedLog.value, {
    submissionId,
    archiveFilename: path.basename(resolvedSubmission),
    archiveSha256: archiveBefore.sha256,
    startedAt,
    completedAt
  });
  writeNewArtifactFileNoFollow(
    resolvedRoot,
    resolvedLog,
    parsedLog.bytes,
    'Retained raw Apple notary log'
  );
  const rawLog = artifactFileIdentity(
    resolvedRoot,
    resolvedLog,
    'Retained raw Apple notary log'
  );
  return {
    submission: {
      id: submissionId,
      startedAt,
      completedAt,
      archiveFilename: path.basename(resolvedSubmission),
      archive: {
        size: archiveBefore.size,
        sha256: archiveBefore.sha256
      }
    },
    log: parsedLog.value,
    logSummary: summary,
    rawLog
  };
}
