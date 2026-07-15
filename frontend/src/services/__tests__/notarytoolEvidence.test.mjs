import test from 'node:test';
import assert from 'node:assert/strict';
import crypto from 'node:crypto';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { createRequire } from 'node:module';
import { spawnSync } from 'node:child_process';
import { fileURLToPath, pathToFileURL } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const frontendRoot = path.resolve(__dirname, '../../..');
const require = createRequire(import.meta.url);
const {
  assertAcceptedNotaryLog,
  assertDmgTicket,
  assertMainExecutableTicket,
  parseCodesignCdHash,
  parseNotarytoolJson,
  submitAndRetainNotaryLog
} = await import(pathToFileURL(path.join(
  frontendRoot,
  'scripts',
  'notarytool-evidence.mjs'
)));
const { notarizeApp } = await import(pathToFileURL(path.join(
  frontendRoot,
  'scripts',
  'notarize-app.mjs'
)));
const pinnedAppStaplerPath = path.join(
  frontendRoot,
  'scripts',
  'pinned-app-stapler.mjs'
);
const { notarizeSignedApp } = require(path.join(
  frontendRoot,
  'scripts',
  'after-sign-macos.cjs'
));

const submissionId = '12345678-1234-1234-1234-123456789abc';

function sha256(bytes) {
  return crypto.createHash('sha256').update(bytes).digest('hex');
}

function acceptedLog({ archivePath, uploadDate, ticketContents = [] }) {
  return {
    logFormatVersion: 1,
    jobId: submissionId,
    status: 'Accepted',
    statusCode: 0,
    archiveFilename: path.basename(archivePath),
    uploadDate,
    sha256: sha256(fs.readFileSync(archivePath)),
    ticketContents,
    issues: null
  };
}

test('notarytool evidence always fetches and retains the exact accepted Apple log', () => {
  const artifactRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'automixer-notary-log-'));
  const archivePath = path.join(artifactRoot, 'release.zip');
  const rawLogPath = path.join(artifactRoot, 'app-notary-log.json');
  const commands = [];
  const times = [
    new Date('2026-07-15T09:00:00.000Z'),
    new Date('2026-07-15T09:00:02.000Z')
  ];
  try {
    fs.writeFileSync(archivePath, 'signed app archive');
    const rawLogBytes = Buffer.from(`${JSON.stringify(acceptedLog({
      archivePath,
      uploadDate: '2026-07-15T09:00:01.000Z'
    }), null, 2)}\n`);
    const result = submitAndRetainNotaryLog({
      artifactRoot,
      submissionPath: archivePath,
      rawLogPath,
      profile: 'unit-test-profile',
      runCommand(command, args, options = {}) {
        commands.push([command, ...args]);
        if (args[0] === 'notarytool' && args[1] === 'submit') {
          return {
            status: 0,
            stdout: JSON.stringify({ status: 'Accepted', id: submissionId }),
            stderr: ''
          };
        }
        return { status: 0, stdout: rawLogBytes, stderr: '' };
      },
      clock: () => times.shift()
    });

    assert.deepEqual(commands.map(row => row.slice(1, 3)), [
      ['notarytool', 'submit'],
      ['notarytool', 'log']
    ]);
    assert.deepEqual(
      commands[0].slice(commands[0].indexOf('--timeout'), commands[0].indexOf('--timeout') + 2),
      ['--timeout', '30m']
    );
    assert.deepEqual(fs.readFileSync(rawLogPath), rawLogBytes);
    assert.equal(result.submission.id, submissionId);
    assert.equal(result.logSummary.sha256, sha256(fs.readFileSync(archivePath)));
    assert.equal(result.rawLog.sha256, sha256(rawLogBytes));
  } finally {
    fs.rmSync(artifactRoot, { recursive: true, force: true });
  }
});

test('Apple log binding rejects status, hash, time, issues, and CDHash drift', () => {
  const base = {
    logFormatVersion: 1,
    jobId: submissionId,
    status: 'Accepted',
    statusCode: 0,
    archiveFilename: 'release.zip',
    uploadDate: '2026-07-15T09:00:01.000Z',
    sha256: 'a'.repeat(64),
    ticketContents: [{
      path: 'release.zip/App.app',
      digestAlgorithm: 'SHA-256',
      arch: 'arm64',
      cdhash: 'b'.repeat(40)
    }],
    issues: null
  };
  const context = {
    submissionId,
    archiveFilename: 'release.zip',
    archiveSha256: 'a'.repeat(64),
    startedAt: '2026-07-15T09:00:00.000Z',
    completedAt: '2026-07-15T09:00:02.000Z'
  };
  assert.doesNotThrow(() => assertAcceptedNotaryLog(base, context));
  assert.doesNotThrow(() => assertAcceptedNotaryLog({
    ...base,
    uploadDate: '2026-07-15T09:00:01Z'
  }, context));
  for (const mutation of [
    { status: 'Invalid' },
    { statusCode: 1 },
    { logFormatVersion: 2 },
    { archiveFilename: 'swapped.zip' },
    { sha256: 'c'.repeat(64) },
    { uploadDate: '2026-07-15T08:59:59.000Z' },
    { issues: [{}] },
    { jobId: '22345678-1234-1234-1234-123456789abc' }
  ]) {
    assert.throws(
      () => assertAcceptedNotaryLog({ ...base, ...mutation }, context),
      /does not bind/
    );
  }
  assert.doesNotThrow(() => assertMainExecutableTicket(base, {
    expectedPathSuffix: 'App.app',
    expectedCdHash: 'b'.repeat(40),
    expectedArchiveFilename: 'release.zip'
  }));
  assert.throws(() => assertMainExecutableTicket(base, {
    expectedPathSuffix: 'App.app',
    expectedCdHash: 'c'.repeat(40)
  }), /exactly one/);
  assert.throws(() => assertMainExecutableTicket({
    ...base,
    ticketContents: [{ ...base.ticketContents[0], path: 'release.zip/NotApp.app' }]
  }, {
    expectedPathSuffix: 'App.app',
    expectedCdHash: 'b'.repeat(40)
  }), /exactly one/);
  assert.throws(() => assertMainExecutableTicket({
    ...base,
    ticketContents: [{ ...base.ticketContents[0], path: 'other.zip/App.app' }]
  }, {
    expectedPathSuffix: 'App.app',
    expectedCdHash: 'b'.repeat(40),
    expectedArchiveFilename: 'release.zip'
  }), /exactly one/);
  const dmgLog = {
    ...base,
    archiveFilename: 'release.dmg',
    ticketContents: [{
      path: 'release.dmg',
      digestAlgorithm: 'SHA-256',
      cdhash: 'd'.repeat(40)
    }]
  };
  assert.deepEqual(assertDmgTicket(dmgLog, {
    expectedPath: 'release.dmg',
    expectedCdHash: 'd'.repeat(40)
  }), {
    path: 'release.dmg',
    digestAlgorithm: 'SHA-256',
    cdhash: 'd'.repeat(40)
  });
  for (const ticketContents of [
    [],
    [{ ...dmgLog.ticketContents[0], cdhash: 'e'.repeat(40) }],
    [{ ...dmgLog.ticketContents[0], arch: 'arm64' }],
    [dmgLog.ticketContents[0], { ...dmgLog.ticketContents[0] }]
  ]) {
    assert.throws(() => assertDmgTicket({ ...dmgLog, ticketContents }, {
      expectedPath: 'release.dmg',
      expectedCdHash: 'd'.repeat(40)
    }), /exactly one/);
  }
  assert.equal(parseCodesignCdHash(`CDHash=${'B'.repeat(40)}`), 'b'.repeat(40));
});

test('notarytool JSON parser rejects missing, malformed, NUL, and oversized evidence', () => {
  assert.throws(() => parseNotarytoolJson(''), /size/);
  assert.throws(() => parseNotarytoolJson('{'), /valid JSON/);
  assert.throws(() => parseNotarytoolJson(Buffer.from([0x7b, 0x00, 0x7d])), /NUL/);
  assert.throws(() => parseNotarytoolJson('x'.repeat(2 * 1024 * 1024 + 1)), /size/);
});

test('pinned application stapler rejects a different working-directory inode', () => {
  const tempRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'automixer-pinned-stapler-'));
  const expectedApp = path.join(tempRoot, 'Expected.app');
  const substitutedApp = path.join(tempRoot, 'Substituted.app');
  try {
    fs.mkdirSync(expectedApp);
    fs.mkdirSync(substitutedApp);
    const expectedStat = fs.statSync(expectedApp, { bigint: true });
    const result = spawnSync(process.execPath, [
      pinnedAppStaplerPath,
      'validate',
      expectedStat.dev.toString(),
      expectedStat.ino.toString(),
      fs.realpathSync(expectedApp)
    ], {
      cwd: substitutedApp,
      encoding: 'utf8'
    });
    assert.notEqual(result.status, 0);
    assert.match(result.stderr, /does not match the pinned application/);
  } finally {
    fs.rmSync(tempRoot, { recursive: true, force: true });
  }
});

test('application notarization staples only after raw Apple log and CDHash binding', () => {
  const artifactRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'automixer-app-notary-'));
  const appPath = path.join(artifactRoot, 'mac-arm64', 'App.app');
  const executablePath = path.join(appPath, 'Contents', 'MacOS', 'App');
  const infoPlistPath = path.join(appPath, 'Contents', 'Info.plist');
  const cdhash = 'b'.repeat(40);
  const times = [
    new Date('2026-07-15T09:00:00.000Z'),
    new Date('2026-07-15T09:00:02.000Z')
  ];
  let submittedPath = '';
  const commands = [];
  try {
    fs.mkdirSync(path.dirname(executablePath), { recursive: true });
    fs.writeFileSync(executablePath, 'arm64 executable');
    fs.writeFileSync(infoPlistPath, 'plist');
    const result = notarizeApp({
      artifactRoot,
      appPath,
      environment: {
        AUTOMIXER_MAC_BUILD_PROFILE: 'release',
        APPLE_KEYCHAIN_PROFILE: 'unit-test-profile'
      },
      runCommand(command, args, options = {}) {
        commands.push([command, ...args]);
        if (command === '/usr/bin/plutil') {
          return { status: 0, stdout: 'App\n', stderr: '' };
        }
        if (command === '/usr/bin/ditto') {
          assert.equal(args.at(-1), '/dev/fd/3');
          fs.writeSync(options.pinnedOutput.descriptor, Buffer.from('private app archive'));
          return { status: 0, stdout: '', stderr: '' };
        }
        if (args[0] === 'notarytool' && args[1] === 'submit') {
          submittedPath = args[2];
          return {
            status: 0,
            stdout: JSON.stringify({ status: 'Accepted', id: submissionId }),
            stderr: ''
          };
        }
        if (args[0] === 'notarytool' && args[1] === 'log') {
          return {
            status: 0,
            stdout: JSON.stringify(acceptedLog({
              archivePath: submittedPath,
              uploadDate: '2026-07-15T09:00:01.000Z',
              ticketContents: [{
                path: `${path.basename(submittedPath)}/App.app`,
                digestAlgorithm: 'SHA-256',
                arch: 'arm64',
                cdhash
              }]
            })),
            stderr: ''
          };
        }
        if (command === '/usr/bin/codesign' && args[0] === '-d') {
          return { status: 0, stdout: '', stderr: `CDHash=${cdhash}\n` };
        }
        return { status: 0, stdout: '', stderr: '' };
      },
      clock: () => times.shift()
    });

    const evidence = JSON.parse(fs.readFileSync(
      path.join(artifactRoot, 'app-notarization.json'),
      'utf8'
    ));
    assert.equal(result.cdhash, cdhash);
    assert.equal(evidence.rawLog.path, 'app-notary-log.json');
    assert.equal(evidence.mainExecutable.cdhash, cdhash);
    assert.equal(evidence.ticket.cdhash, cdhash);
    assert.equal(
      commands.findIndex(row => row[1] === 'notarytool' && row[2] === 'log')
        < commands.findIndex(row => (
          String(row[1]).endsWith('/pinned-app-stapler.mjs') && row[2] === 'staple'
        )),
      true
    );
    assert.equal(
      fs.readdirSync(artifactRoot).some(name => name.startsWith('.notarize-app-')),
      false
    );
  } finally {
    fs.rmSync(artifactRoot, { recursive: true, force: true });
  }
});

test('application notarization detects restored-byte mutation before submission', () => {
  const artifactRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'automixer-app-notary-race-'));
  const appPath = path.join(artifactRoot, 'mac-arm64', 'App.app');
  const executablePath = path.join(appPath, 'Contents', 'MacOS', 'App');
  const infoPlistPath = path.join(appPath, 'Contents', 'Info.plist');
  const cdhash = 'b'.repeat(40);
  let submitCalls = 0;
  try {
    fs.mkdirSync(path.dirname(executablePath), { recursive: true });
    fs.writeFileSync(executablePath, 'arm64 executable');
    fs.writeFileSync(infoPlistPath, 'plist');
    assert.throws(() => notarizeApp({
      artifactRoot,
      appPath,
      environment: {
        AUTOMIXER_MAC_BUILD_PROFILE: 'release',
        APPLE_KEYCHAIN_PROFILE: 'unit-test-profile'
      },
      runCommand(command, args, options = {}) {
        if (command === '/usr/bin/plutil') {
          return { status: 0, stdout: 'App\n', stderr: '' };
        }
        if (command === '/usr/bin/codesign' && args[0] === '-d') {
          return { status: 0, stdout: '', stderr: `CDHash=${cdhash}\n` };
        }
        if (command === '/usr/bin/ditto') {
          fs.writeSync(options.pinnedOutput.descriptor, Buffer.from('private app archive'));
          const original = fs.readFileSync(executablePath);
          fs.writeFileSync(executablePath, 'temporary attacker bytes');
          fs.writeFileSync(executablePath, original);
        }
        if (args[0] === 'notarytool' && args[1] === 'submit') submitCalls += 1;
        return { status: 0, stdout: '', stderr: '' };
      }
    }), /changed application identity or tree state/);
    assert.equal(submitCalls, 0);
    assert.equal(fs.existsSync(path.join(artifactRoot, 'app-notary-log.json')), false);
  } finally {
    fs.rmSync(artifactRoot, { recursive: true, force: true });
  }
});

test('application notarization rejects archive-path replacement before submission', () => {
  const tempRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'automixer-app-archive-race-'));
  const artifactRoot = path.join(tempRoot, 'artifact');
  const appPath = path.join(artifactRoot, 'mac-arm64', 'App.app');
  const executablePath = path.join(appPath, 'Contents', 'MacOS', 'App');
  const infoPlistPath = path.join(appPath, 'Contents', 'Info.plist');
  const outsidePath = path.join(tempRoot, 'outside.zip');
  const cdhash = 'b'.repeat(40);
  let submitCalls = 0;
  try {
    fs.mkdirSync(path.dirname(executablePath), { recursive: true });
    fs.writeFileSync(executablePath, 'arm64 executable');
    fs.writeFileSync(infoPlistPath, 'plist');
    fs.writeFileSync(outsidePath, 'DO NOT WRITE');
    assert.throws(() => notarizeApp({
      artifactRoot,
      appPath,
      environment: {
        AUTOMIXER_MAC_BUILD_PROFILE: 'release',
        APPLE_KEYCHAIN_PROFILE: 'unit-test-profile'
      },
      runCommand(command, args, options = {}) {
        if (command === '/usr/bin/plutil') {
          return { status: 0, stdout: 'App\n', stderr: '' };
        }
        if (command === '/usr/bin/codesign' && args[0] === '-d') {
          return { status: 0, stdout: '', stderr: `CDHash=${cdhash}\n` };
        }
        if (command === '/usr/bin/ditto') {
          fs.rmSync(options.pinnedOutput.path);
          fs.symlinkSync(outsidePath, options.pinnedOutput.path);
          fs.writeSync(options.pinnedOutput.descriptor, Buffer.from('private app archive'));
        }
        if (args[0] === 'notarytool' && args[1] === 'submit') submitCalls += 1;
        return { status: 0, stdout: '', stderr: '' };
      }
    }), /must remain a real file|changed pinned (?:file|descriptor) identity/);
    assert.equal(submitCalls, 0);
    assert.equal(fs.readFileSync(outsidePath, 'utf8'), 'DO NOT WRITE');
    assert.equal(fs.existsSync(path.join(artifactRoot, 'app-notary-log.json')), false);
  } finally {
    fs.rmSync(tempRoot, { recursive: true, force: true });
  }
});

test('application notarization rejects app replacement immediately after Info.plist read', () => {
  const tempRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'automixer-app-path-race-'));
  const artifactRoot = path.join(tempRoot, 'artifact');
  const appPath = path.join(artifactRoot, 'mac-arm64', 'App.app');
  const originalBackup = path.join(artifactRoot, 'mac-arm64', 'Original.app');
  const outsideApp = path.join(tempRoot, 'Outside.app');
  const originalExecutable = path.join(appPath, 'Contents', 'MacOS', 'App');
  const originalPlist = path.join(appPath, 'Contents', 'Info.plist');
  const outsideExecutable = path.join(outsideApp, 'Contents', 'MacOS', 'App');
  const outsidePlist = path.join(outsideApp, 'Contents', 'Info.plist');
  let staplerCalls = 0;
  try {
    fs.mkdirSync(path.dirname(originalExecutable), { recursive: true });
    fs.mkdirSync(path.dirname(outsideExecutable), { recursive: true });
    fs.writeFileSync(originalExecutable, 'original executable');
    fs.writeFileSync(originalPlist, 'original plist');
    fs.writeFileSync(outsideExecutable, 'outside executable');
    fs.writeFileSync(outsidePlist, 'DO NOT MODIFY');
    assert.throws(() => notarizeApp({
      artifactRoot,
      appPath,
      environment: {
        AUTOMIXER_MAC_BUILD_PROFILE: 'release',
        APPLE_KEYCHAIN_PROFILE: 'unit-test-profile'
      },
      runCommand(command, args) {
        if (command === '/usr/bin/plutil') {
          fs.renameSync(appPath, originalBackup);
          fs.symlinkSync(outsideApp, appPath);
          return { status: 0, stdout: 'App\n', stderr: '' };
        }
        if (args[0] === 'stapler') staplerCalls += 1;
        return { status: 0, stdout: '', stderr: '' };
      }
    }), /must remain a real directory|changed pinned directory identity/);
    assert.equal(staplerCalls, 0);
    assert.equal(fs.readFileSync(outsidePlist, 'utf8'), 'DO NOT MODIFY');
    assert.equal(fs.existsSync(path.join(artifactRoot, 'app-notary-log.json')), false);
  } finally {
    fs.rmSync(tempRoot, { recursive: true, force: true });
  }
});

test('electron-builder afterSign invokes only the contained explicit app notarizer', () => {
  const artifactRoot = '/tmp/automixer-after-sign-artifact';
  const calls = [];
  const result = notarizeSignedApp({
    electronPlatformName: 'darwin',
    appOutDir: path.join(artifactRoot, 'mac-arm64'),
    packager: { appInfo: { productFilename: 'App' } }
  }, {
    environment: {
      AUTOMIXER_MAC_BUILD_PROFILE: 'release',
      AUTOMIXER_MAC_OUTPUT_DIR: artifactRoot
    },
    spawn(command, args, options) {
      calls.push({ command, args, options });
      return {
        status: 0,
        stdout: JSON.stringify({
          status: 'Accepted',
          submissionId,
          cdhash: 'b'.repeat(40)
        }),
        stderr: ''
      };
    }
  });
  assert.equal(result.status, 'Accepted');
  assert.equal(calls.length, 1);
  assert.equal(calls[0].command, process.execPath);
  assert.deepEqual(calls[0].args.slice(-4), [
    '--artifact-root', artifactRoot,
    '--app', path.join(artifactRoot, 'mac-arm64', 'App.app')
  ]);
  assert.throws(() => notarizeSignedApp({
    electronPlatformName: 'darwin',
    appOutDir: '/tmp/elsewhere',
    packager: { appInfo: { productFilename: 'App' } }
  }, {
    environment: {
      AUTOMIXER_MAC_BUILD_PROFILE: 'release',
      AUTOMIXER_MAC_OUTPUT_DIR: artifactRoot
    },
    spawn() { throw new Error('must not run'); }
  }), /does not match/);
});
