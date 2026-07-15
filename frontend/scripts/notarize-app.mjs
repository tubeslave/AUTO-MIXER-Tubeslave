#!/usr/bin/env node

import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import process from 'node:process';
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';

import {
  artifactFileIdentity,
  artifactRelativePath,
  hashTree,
  removeArtifactFileNoFollow,
  writeNewArtifactFileNoFollow
} from './release-evidence.mjs';
import {
  assertMainExecutableTicket,
  parseCodesignCdHash,
  submitAndRetainNotaryLog
} from './notarytool-evidence.mjs';

const scriptPath = fileURLToPath(import.meta.url);
const pinnedAppStaplerPath = fileURLToPath(
  new URL('./pinned-app-stapler.mjs', import.meta.url)
);

function textOutput(value) {
  return Buffer.isBuffer(value) ? value.toString('utf8') : String(value || '');
}

function sanitizeOutput(value) {
  return textOutput(value)
    .replaceAll(os.homedir(), '~')
    .replace(/\b(password|token|secret|api[_-]?key)\s*[:=]\s*\S+/gi, '$1=[REDACTED]')
    .slice(-12_000);
}

function run(command, args, options = {}) {
  const completed = spawnSync(command, args, {
    encoding: null,
    maxBuffer: 32 * 1024 * 1024,
    ...(options.stdio ? { stdio: options.stdio } : {}),
    ...(options.cwd ? { cwd: options.cwd } : {})
  });
  if (completed.error) throw completed.error;
  if (completed.status !== 0 && options.check !== false) {
    const detail = sanitizeOutput(Buffer.concat([
      Buffer.from(completed.stderr || ''),
      Buffer.from('\n'),
      Buffer.from(completed.stdout || '')
    ]));
    throw new Error(
      `${path.basename(command)} failed with exit code ${completed.status}`
      + (detail ? `: ${detail}` : '')
    );
  }
  return {
    status: completed.status,
    stdout: Buffer.from(completed.stdout || ''),
    stderr: Buffer.from(completed.stderr || '')
  };
}

function parseArgs(argv) {
  let appPath = '';
  let artifactRoot = '';
  for (let index = 0; index < argv.length; index += 1) {
    if (argv[index] === '--app') appPath = path.resolve(argv[++index] || '');
    else if (argv[index] === '--artifact-root') {
      artifactRoot = path.resolve(argv[++index] || '');
    } else throw new Error(`Unknown argument: ${argv[index]}`);
  }
  if (!artifactRoot || !fs.existsSync(artifactRoot)) {
    throw new Error('--artifact-root must point to the release artifact directory');
  }
  return { artifactRoot, appPath };
}

export function resolveNotarizationApp(artifactRoot, appPath) {
  const resolvedRoot = path.resolve(artifactRoot);
  const resolvedApp = path.resolve(appPath);
  const appParent = path.dirname(resolvedApp);
  const expectedParent = path.join(resolvedRoot, 'mac-arm64');
  const realRoot = fs.realpathSync(resolvedRoot);
  const realParent = fs.realpathSync(expectedParent);
  const realApp = fs.realpathSync(resolvedApp);
  if (
    !resolvedApp.endsWith('.app')
    || appParent !== expectedParent
    || realParent !== path.join(realRoot, 'mac-arm64')
    || realApp !== path.join(realParent, path.basename(resolvedApp))
  ) {
    throw new Error('--app must identify one real app bundle in artifact-root/mac-arm64');
  }
  const rootStat = fs.lstatSync(resolvedRoot);
  const parentStat = fs.lstatSync(expectedParent);
  const appStat = fs.lstatSync(resolvedApp);
  if (
    rootStat.isSymbolicLink()
    || parentStat.isSymbolicLink()
    || appStat.isSymbolicLink()
    || !rootStat.isDirectory()
    || !parentStat.isDirectory()
    || !appStat.isDirectory()
  ) {
    throw new Error('Application notarization requires real contained directories');
  }
  return { artifactRoot: resolvedRoot, appPath: resolvedApp };
}

function combined(result) {
  return `${textOutput(result?.stdout)}\n${textOutput(result?.stderr)}`.trim();
}

function metadataIdentity(candidate) {
  const stat = fs.lstatSync(candidate, { bigint: true });
  return {
    dev: stat.dev.toString(),
    ino: stat.ino.toString(),
    mode: stat.mode.toString(),
    nlink: stat.nlink.toString(),
    size: stat.size.toString(),
    mtimeNs: stat.mtimeNs.toString(),
    ctimeNs: stat.ctimeNs.toString()
  };
}

function pinnedNodeIdentity(candidate, label, expectedType) {
  const stat = fs.lstatSync(candidate, { bigint: true });
  const isExpected = expectedType === 'directory' ? stat.isDirectory() : stat.isFile();
  if (stat.isSymbolicLink() || !isExpected) {
    throw new Error(`${label} must remain a real ${expectedType}`);
  }
  return {
    realPath: fs.realpathSync(candidate),
    dev: stat.dev.toString(),
    ino: stat.ino.toString(),
    mode: stat.mode.toString(),
    nlink: stat.nlink.toString()
  };
}

function assertPinnedNodeIdentity(
  expected,
  candidate,
  label,
  expectedType,
  { ignoreNlink = false } = {}
) {
  const current = pinnedNodeIdentity(candidate, label, expectedType);
  const comparable = value => ignoreNlink
    ? Object.fromEntries(Object.entries(value).filter(([key]) => key !== 'nlink'))
    : value;
  if (JSON.stringify(comparable(current)) !== JSON.stringify(comparable(expected))) {
    throw new Error(`${label} changed pinned ${expectedType} identity`);
  }
}

function pinnedDescriptorIdentity(descriptor, label, expectedType = 'file') {
  const stat = fs.fstatSync(descriptor, { bigint: true });
  const isExpected = expectedType === 'directory' ? stat.isDirectory() : stat.isFile();
  if (!isExpected) throw new Error(`${label} must remain a ${expectedType} descriptor`);
  return {
    dev: stat.dev.toString(),
    ino: stat.ino.toString(),
    mode: stat.mode.toString(),
    nlink: stat.nlink.toString()
  };
}

function captureApplicationState(appPath) {
  const parentPath = path.dirname(appPath);
  const rows = [];
  function visit(candidate, relative) {
    const stat = fs.lstatSync(candidate);
    const row = {
      path: relative,
      metadata: metadataIdentity(candidate),
      type: stat.isSymbolicLink()
        ? 'symlink'
        : stat.isDirectory()
          ? 'directory'
          : stat.isFile()
            ? 'file'
            : 'unsupported'
    };
    if (row.type === 'unsupported') {
      throw new Error(`Unsupported application object: ${relative}`);
    }
    if (row.type === 'symlink') row.target = fs.readlinkSync(candidate);
    rows.push(row);
    if (row.type === 'directory') {
      for (const name of fs.readdirSync(candidate).sort()) {
        visit(path.join(candidate, name), relative ? `${relative}/${name}` : name);
      }
    }
  }
  visit(appPath, '');
  return {
    parentPath,
    parentRealPath: fs.realpathSync(parentPath),
    parentMetadata: metadataIdentity(parentPath),
    appRealPath: fs.realpathSync(appPath),
    treeSha256: hashTree(appPath),
    rows
  };
}

function assertApplicationState(expected, appPath, label) {
  const current = captureApplicationState(appPath);
  if (JSON.stringify(current) !== JSON.stringify(expected)) {
    throw new Error(`${label} changed application identity or tree state`);
  }
}

function assertCriticalApplicationLayout(appPath, executableName = null) {
  for (const [relative, expectedType] of [
    ['', 'directory'],
    ['Contents', 'directory'],
    ['Contents/MacOS', 'directory'],
    ['Contents/Info.plist', 'file'],
    ...(executableName ? [[`Contents/MacOS/${executableName}`, 'file']] : [])
  ]) {
    pinnedNodeIdentity(
      relative ? path.join(appPath, ...relative.split('/')) : appPath,
      `Application critical path ${relative || '.'}`,
      expectedType
    );
  }
}

export function notarizeApp({
  artifactRoot,
  appPath,
  environment = process.env,
  runCommand = run,
  clock = () => new Date()
}) {
  if (environment.AUTOMIXER_MAC_BUILD_PROFILE !== 'release') {
    throw new Error('Application notarization is available only in release profile');
  }
  const profile = String(environment.APPLE_KEYCHAIN_PROFILE || '').trim();
  if (!profile) throw new Error('Application notarization requires APPLE_KEYCHAIN_PROFILE');
  const keychain = String(environment.APPLE_KEYCHAIN || '').trim();
  ({ artifactRoot, appPath } = resolveNotarizationApp(artifactRoot, appPath));
  const appDescriptor = fs.openSync(
    appPath,
    fs.constants.O_RDONLY | fs.constants.O_DIRECTORY | fs.constants.O_NOFOLLOW
  );
  const appNodeIdentity = pinnedNodeIdentity(
    appPath,
    'Notarized application',
    'directory'
  );
  const appDescriptorIdentity = pinnedDescriptorIdentity(
    appDescriptor,
    'Notarized application',
    'directory'
  );
  if (
    appNodeIdentity.dev !== appDescriptorIdentity.dev
    || appNodeIdentity.ino !== appDescriptorIdentity.ino
  ) {
    fs.closeSync(appDescriptor);
    throw new Error('Notarized application path does not match its pinned descriptor');
  }
  assertCriticalApplicationLayout(appPath);
  const stateBefore = captureApplicationState(appPath);
  let appDescriptorOpen = true;
  const assertPinnedApplicationBoundary = label => {
    assertPinnedNodeIdentity(appNodeIdentity, appPath, label, 'directory');
    const descriptorIdentity = pinnedDescriptorIdentity(
      appDescriptor,
      label,
      'directory'
    );
    if (JSON.stringify(descriptorIdentity) !== JSON.stringify(appDescriptorIdentity)) {
      throw new Error(`${label} changed pinned application descriptor identity`);
    }
  };
  const assertOriginalApplication = label => {
    assertPinnedApplicationBoundary(label);
    assertApplicationState(stateBefore, appPath, label);
  };
  try {

  const evidencePath = path.join(artifactRoot, 'app-notarization.json');
  const rawLogPath = path.join(artifactRoot, 'app-notary-log.json');
  for (const outputPath of [evidencePath, rawLogPath]) {
    if (fs.lstatSync(outputPath, { throwIfNoEntry: false })) {
      throw new Error(`Application notarization output already exists: ${path.basename(outputPath)}`);
    }
  }

  const infoPlistPath = path.join(appPath, 'Contents', 'Info.plist');
  assertOriginalApplication('Application before Info.plist read');
  const executableResult = runCommand('/usr/bin/plutil', [
    '-extract', 'CFBundleExecutable', 'raw', '-o', '-', infoPlistPath
  ]);
  assertOriginalApplication('Application after Info.plist read');
  const executableName = textOutput(executableResult.stdout).trim();
  if (!executableName || path.basename(executableName) !== executableName) {
    throw new Error('Application CFBundleExecutable is invalid');
  }
  const executablePath = path.join(appPath, 'Contents', 'MacOS', executableName);
  assertCriticalApplicationLayout(appPath, executableName);

  assertOriginalApplication('Application before signature verification');
  runCommand('/usr/bin/codesign', ['--verify', '--deep', '--strict', '--verbose=4', appPath]);
  assertOriginalApplication('Application after signature verification');
  const appCdhashBefore = parseCodesignCdHash(combined(runCommand(
    '/usr/bin/codesign',
    ['-d', '--verbose=4', appPath]
  )));
  assertOriginalApplication('Application after bundle CDHash read');
  const mainCdhashBefore = parseCodesignCdHash(combined(runCommand(
    '/usr/bin/codesign',
    ['-d', '--verbose=4', executablePath]
  )));
  assertOriginalApplication('Application after executable CDHash read');
  if (appCdhashBefore !== mainCdhashBefore) {
    throw new Error('Application bundle and main executable CDHash differ before notarization');
  }
  const workingDirectoryPath = fs.mkdtempSync(path.join(artifactRoot, '.notarize-app-'));
  const workingDirectoryDescriptor = fs.openSync(
    workingDirectoryPath,
    fs.constants.O_RDONLY | fs.constants.O_DIRECTORY | fs.constants.O_NOFOLLOW
  );
  fs.fchmodSync(workingDirectoryDescriptor, 0o700);
  const workingArchivePath = path.join(workingDirectoryPath, 'submission.zip');
  const archiveDescriptor = fs.openSync(
    workingArchivePath,
    fs.constants.O_CREAT
      | fs.constants.O_EXCL
      | fs.constants.O_RDWR
      | fs.constants.O_NOFOLLOW,
    0o600
  );
  fs.fchmodSync(archiveDescriptor, 0o600);
  const workingDirectoryIdentity = pinnedNodeIdentity(
    workingDirectoryPath,
    'Private application notarization directory',
    'directory'
  );
  const workingDirectoryDescriptorIdentity = pinnedDescriptorIdentity(
    workingDirectoryDescriptor,
    'Private application notarization directory',
    'directory'
  );
  const reservedArchiveIdentity = pinnedNodeIdentity(
    workingArchivePath,
    'Reserved application notarization archive',
    'file'
  );
  const reservedDescriptorIdentity = pinnedDescriptorIdentity(
    archiveDescriptor,
    'Reserved application notarization archive'
  );
  let archiveDescriptorOpen = true;
  let workingDirectoryDescriptorOpen = true;
  let workingArchiveIdentity = null;
  try {
    assertPinnedNodeIdentity(
      workingDirectoryIdentity,
      workingDirectoryPath,
      'Private application notarization directory before archive creation',
      'directory'
    );
    if (
      JSON.stringify(pinnedDescriptorIdentity(
        workingDirectoryDescriptor,
        'Private application notarization directory before archive creation',
        'directory'
      )) !== JSON.stringify(workingDirectoryDescriptorIdentity)
    ) {
      throw new Error('Private application notarization directory changed pinned descriptor identity');
    }
    assertPinnedNodeIdentity(
      reservedArchiveIdentity,
      workingArchivePath,
      'Reserved application notarization archive before archive creation',
      'file'
    );
    runCommand('/usr/bin/ditto', [
      '-c', '-k', '--sequesterRsrc', '--keepParent', appPath, '/dev/fd/3'
    ], {
      stdio: ['ignore', 'pipe', 'pipe', archiveDescriptor],
      pinnedOutput: { descriptor: archiveDescriptor, path: workingArchivePath }
    });
    fs.fsyncSync(archiveDescriptor);
    if (
      JSON.stringify(pinnedDescriptorIdentity(
        archiveDescriptor,
        'Application notarization archive after archive creation'
      )) !== JSON.stringify(reservedDescriptorIdentity)
    ) {
      throw new Error('Application notarization archive changed pinned descriptor identity');
    }
    assertPinnedNodeIdentity(
      workingDirectoryIdentity,
      workingDirectoryPath,
      'Private application notarization directory after archive creation',
      'directory'
    );
    assertPinnedNodeIdentity(
      reservedArchiveIdentity,
      workingArchivePath,
      'Application notarization archive after archive creation',
      'file'
    );
    fs.closeSync(archiveDescriptor);
    archiveDescriptorOpen = false;
    workingArchiveIdentity = artifactFileIdentity(
      artifactRoot,
      workingArchivePath,
      'Private application notarization archive'
    );
    assertApplicationState(
      stateBefore,
      appPath,
      'Application notarization archive creation'
    );

    const notary = submitAndRetainNotaryLog({
      artifactRoot,
      submissionPath: workingArchivePath,
      rawLogPath,
      profile,
      keychain,
      runCommand,
      clock
    });
    const ticket = assertMainExecutableTicket(notary.log, {
      expectedPathSuffix: path.basename(appPath),
      expectedCdHash: appCdhashBefore,
      architecture: 'arm64',
      expectedArchiveFilename: notary.submission.archiveFilename
    });
    assertApplicationState(stateBefore, appPath, 'Apple notarization submission');

    assertOriginalApplication('Application before stapling');
    runCommand(process.execPath, [
      pinnedAppStaplerPath,
      'staple',
      appDescriptorIdentity.dev,
      appDescriptorIdentity.ino,
      appNodeIdentity.realPath
    ], {
      cwd: appPath,
      pinnedInput: { descriptor: appDescriptor, path: appPath }
    });
    assertPinnedApplicationBoundary('Application after stapling');
    const stapledState = captureApplicationState(appPath);
    runCommand(process.execPath, [
      pinnedAppStaplerPath,
      'validate',
      appDescriptorIdentity.dev,
      appDescriptorIdentity.ino,
      appNodeIdentity.realPath
    ], {
      cwd: appPath,
      pinnedInput: { descriptor: appDescriptor, path: appPath }
    });
    assertPinnedApplicationBoundary('Application after stapler validation');
    assertApplicationState(stapledState, appPath, 'Application stapler validation');
    runCommand('/usr/bin/codesign', ['--verify', '--deep', '--strict', '--verbose=4', appPath]);
    const appCdhashAfter = parseCodesignCdHash(combined(runCommand(
      '/usr/bin/codesign',
      ['-d', '--verbose=4', appPath]
    )));
    const mainCdhashAfter = parseCodesignCdHash(combined(runCommand(
      '/usr/bin/codesign',
      ['-d', '--verbose=4', executablePath]
    )));
    assertApplicationState(stapledState, appPath, 'Post-staple signature verification');
    if (
      appCdhashAfter !== appCdhashBefore
      || mainCdhashAfter !== mainCdhashBefore
      || appCdhashAfter !== mainCdhashAfter
    ) {
      throw new Error('Application main executable CDHash changed after stapling');
    }

    const evidence = {
      schema: 'automixer.app_notarization.v1',
      pathPolicy: 'artifact_root_relative_v1',
      generatedAt: new Date().toISOString(),
      status: 'Accepted',
      submission: notary.submission,
      appleLog: notary.logSummary,
      rawLog: notary.rawLog,
      ticket,
      app: {
        path: artifactRelativePath(artifactRoot, appPath, 'Notarized application'),
        treeSha256: stapledState.treeSha256
      },
      mainExecutable: {
        path: `Contents/MacOS/${executableName}`,
        architecture: 'arm64',
        cdhash: mainCdhashAfter
      }
    };
    writeNewArtifactFileNoFollow(
      artifactRoot,
      evidencePath,
      `${JSON.stringify(evidence, null, 2)}\n`,
      'Application notarization evidence'
    );
    return {
      status: 'Accepted',
      submissionId: notary.submission.id,
      cdhash: mainCdhashAfter
    };
  } finally {
    if (archiveDescriptorOpen) {
      try {
        fs.closeSync(archiveDescriptor);
      } catch {}
      archiveDescriptorOpen = false;
    }
    if (fs.lstatSync(workingArchivePath, { throwIfNoEntry: false })) {
      try {
        assertPinnedNodeIdentity(
          workingDirectoryIdentity,
          workingDirectoryPath,
          'Private application notarization directory before archive cleanup',
          'directory'
        );
        removeArtifactFileNoFollow(
          artifactRoot,
          workingArchivePath,
          'Private application notarization archive cleanup',
          workingArchiveIdentity
        );
      } catch (cleanupError) {
        if (workingArchiveIdentity) throw cleanupError;
      }
    }
    if (fs.lstatSync(workingDirectoryPath, { throwIfNoEntry: false })) {
      try {
        assertPinnedNodeIdentity(
          workingDirectoryIdentity,
          workingDirectoryPath,
          'Private application notarization directory cleanup',
          'directory',
          { ignoreNlink: true }
        );
        if (fs.readdirSync(workingDirectoryPath).length === 0) {
          if (workingDirectoryDescriptorOpen) {
            fs.closeSync(workingDirectoryDescriptor);
            workingDirectoryDescriptorOpen = false;
          }
          fs.rmdirSync(workingDirectoryPath);
        }
      } catch (cleanupError) {
        if (workingArchiveIdentity) throw cleanupError;
      }
    }
    if (workingDirectoryDescriptorOpen) {
      try {
        fs.closeSync(workingDirectoryDescriptor);
      } catch {}
      workingDirectoryDescriptorOpen = false;
    }
  }
  } finally {
    if (appDescriptorOpen) {
      try {
        fs.closeSync(appDescriptor);
      } catch {}
      appDescriptorOpen = false;
    }
  }
}

function main() {
  const result = notarizeApp({
    ...parseArgs(process.argv.slice(2)),
    environment: process.env
  });
  process.stdout.write(`${JSON.stringify(result)}\n`);
}

if (process.argv[1] && path.resolve(process.argv[1]) === scriptPath) {
  try {
    main();
  } catch (error) {
    process.stderr.write(`${error?.message || error}\n`);
    process.exitCode = 1;
  }
}
