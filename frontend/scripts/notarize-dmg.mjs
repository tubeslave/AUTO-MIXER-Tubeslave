#!/usr/bin/env node

import fs from 'node:fs';
import crypto from 'node:crypto';
import os from 'node:os';
import path from 'node:path';
import process from 'node:process';
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';

import {
  artifactFileIdentity,
  assertIdentityUnchanged,
  copyStableRegularFileNoFollow,
  removeArtifactFileNoFollow,
  replaceArtifactFileNoFollow,
  resolveArtifactPath,
  writeNewArtifactFileNoFollow
} from './release-evidence.mjs';
import {
  assertDmgTicket,
  parseCodesignCdHash,
  submitAndRetainNotaryLog
} from './notarytool-evidence.mjs';

const scriptPath = fileURLToPath(import.meta.url);

function textOutput(value) {
  return Buffer.isBuffer(value) ? value.toString('utf8') : String(value || '');
}

function combined(result) {
  return `${textOutput(result?.stdout)}\n${textOutput(result?.stderr)}`.trim();
}

function sanitizeOutput(value) {
  return String(value || '')
    .replaceAll(os.homedir(), '~')
    .replace(/\b(password|token|secret|api[_-]?key)\s*[:=]\s*\S+/gi, '$1=[REDACTED]')
    .slice(-12_000);
}

function run(command, args, options = {}) {
  const completed = spawnSync(command, args, {
    encoding: null,
    maxBuffer: 32 * 1024 * 1024,
    ...(options.stdio ? { stdio: options.stdio } : {})
  });
  if (completed.error) throw completed.error;
  if (completed.status !== 0 && options.check !== false) {
    const detail = sanitizeOutput(`${completed.stderr || ''}\n${completed.stdout || ''}`.trim());
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
  let dmgPath = '';
  let artifactRoot = '';
  for (let index = 0; index < argv.length; index += 1) {
    if (argv[index] === '--dmg') dmgPath = path.resolve(argv[++index] || '');
    else if (argv[index] === '--artifact-root') {
      artifactRoot = path.resolve(argv[++index] || '');
    }
    else throw new Error(`Unknown argument: ${argv[index]}`);
  }
  if (!artifactRoot || !fs.existsSync(artifactRoot)) {
    throw new Error('--artifact-root must point to the release artifact directory');
  }
  return { artifactRoot, dmgPath };
}

function captureFileState(artifactRoot, filePath, label) {
  const stat = fs.lstatSync(filePath, { bigint: true });
  if (stat.isSymbolicLink() || !stat.isFile()) {
    throw new Error(`${label} must remain a regular file`);
  }
  return {
    realPath: fs.realpathSync(filePath),
    dev: stat.dev.toString(),
    ino: stat.ino.toString(),
    mode: stat.mode.toString(),
    nlink: stat.nlink.toString(),
    size: stat.size.toString(),
    mtimeNs: stat.mtimeNs.toString(),
    ctimeNs: stat.ctimeNs.toString(),
    content: artifactFileIdentity(artifactRoot, filePath, label)
  };
}

function assertFileState(expected, artifactRoot, filePath, label) {
  const current = captureFileState(artifactRoot, filePath, label);
  if (JSON.stringify(current) !== JSON.stringify(expected)) {
    throw new Error(`${label} changed file identity or state`);
  }
}

function pinnedDescriptorIdentity(descriptor, label) {
  const stat = fs.fstatSync(descriptor, { bigint: true });
  if (!stat.isFile()) throw new Error(`${label} must remain a regular file descriptor`);
  return {
    dev: stat.dev.toString(),
    ino: stat.ino.toString(),
    mode: stat.mode.toString(),
    nlink: stat.nlink.toString()
  };
}

export function resolveNotarizationDmg(artifactRoot, dmgPath) {
  const resolvedRoot = path.resolve(artifactRoot);
  fs.realpathSync(resolvedRoot);
  const resolvedDmg = path.resolve(dmgPath);
  if (
    !resolvedDmg.toLowerCase().endsWith('.dmg')
    || path.dirname(resolvedDmg) !== resolvedRoot
  ) {
    throw new Error('--dmg must identify one top-level DMG in --artifact-root');
  }
  const stat = fs.lstatSync(resolvedDmg);
  if (stat.isSymbolicLink() || !stat.isFile()) {
    throw new Error('DMG notarization input must be a regular file, not a symlink');
  }
  const contained = resolveArtifactPath(
    resolvedRoot,
    path.basename(resolvedDmg),
    'DMG notarization input',
    { mustExist: true }
  );
  if (contained !== resolvedDmg) {
    throw new Error('DMG notarization input real path does not match its artifact path');
  }
  return { artifactRoot: resolvedRoot, dmgPath: resolvedDmg };
}

export function notarizeDmg({
  artifactRoot,
  dmgPath,
  environment = process.env,
  runCommand = run
}) {
  if (environment.AUTOMIXER_MAC_BUILD_PROFILE !== 'release') {
    throw new Error('DMG notarization is available only in release profile');
  }
  const profile = String(environment.APPLE_KEYCHAIN_PROFILE || '').trim();
  if (!profile) throw new Error('DMG notarization requires APPLE_KEYCHAIN_PROFILE');
  ({ artifactRoot, dmgPath } = resolveNotarizationDmg(artifactRoot, dmgPath));
  const evidencePath = path.join(artifactRoot, 'dmg-notarization.json');
  const rawLogPath = path.join(artifactRoot, 'dmg-notary-log.json');
  if (
    fs.lstatSync(evidencePath, { throwIfNoEntry: false })
    || fs.lstatSync(rawLogPath, { throwIfNoEntry: false })
  ) {
    throw new Error('DMG notarization output already exists; refusing replacement');
  }

  const originalDmgIdentity = artifactFileIdentity(
    artifactRoot,
    dmgPath,
    'Original DMG before notarization'
  );
  const originalDmgState = captureFileState(
    artifactRoot,
    dmgPath,
    'Original DMG before notarization'
  );
  const workingDmgPath = path.join(
    artifactRoot,
    `.notarize-working-${crypto.randomUUID()}.dmg`
  );
  let workingDmgDescriptor = null;
  let workingDmgDescriptorOpen = false;
  try {
    const workingSnapshot = copyStableRegularFileNoFollow(
      dmgPath,
      workingDmgPath,
      'Private notarization DMG snapshot'
    );
    const snapshotSha256 = workingSnapshot.sha256;
    const workingDmgIdentity = artifactFileIdentity(
      artifactRoot,
      workingDmgPath,
      'Private notarization DMG snapshot'
    );
    const workingDmgState = captureFileState(
      artifactRoot,
      workingDmgPath,
      'Private notarization DMG snapshot'
    );
    workingDmgDescriptor = fs.openSync(
      workingDmgPath,
      fs.constants.O_RDWR | fs.constants.O_NOFOLLOW
    );
    workingDmgDescriptorOpen = true;
    const workingDmgDescriptorIdentity = pinnedDescriptorIdentity(
      workingDmgDescriptor,
      'Private notarization DMG snapshot'
    );
    if (
      workingDmgDescriptorIdentity.dev !== workingDmgState.dev
      || workingDmgDescriptorIdentity.ino !== workingDmgState.ino
    ) {
      throw new Error('Private notarization DMG path does not match its pinned descriptor');
    }
    if (
      workingDmgIdentity.size !== originalDmgIdentity.size
      || workingDmgIdentity.sha256 !== originalDmgIdentity.sha256
      || workingDmgIdentity.sha256 !== snapshotSha256
    ) {
      throw new Error('Private notarization DMG snapshot does not bind to the original');
    }
    const assertWorkingDmgUnchanged = label => {
      assertIdentityUnchanged(
        workingDmgIdentity,
        artifactFileIdentity(artifactRoot, workingDmgPath, label),
        'Private notarization DMG snapshot'
      );
      assertFileState(
        workingDmgState,
        artifactRoot,
        workingDmgPath,
        label
      );
      if (
        JSON.stringify(pinnedDescriptorIdentity(workingDmgDescriptor, label))
        !== JSON.stringify(workingDmgDescriptorIdentity)
      ) {
        throw new Error(`${label} changed pinned descriptor identity`);
      }
    };

    runCommand('/usr/bin/codesign', [
      '--verify', '--strict', '--verbose=2', workingDmgPath
    ]);
    assertWorkingDmgUnchanged('Post-codesign verification notarization DMG snapshot');
    const dmgCdhashBeforeStaple = parseCodesignCdHash(combined(runCommand(
      '/usr/bin/codesign',
      ['-d', '--verbose=4', workingDmgPath]
    )));
    assertWorkingDmgUnchanged('Post-CDHash notarization DMG snapshot');
    const keychain = String(environment.APPLE_KEYCHAIN || '').trim();
    const notary = submitAndRetainNotaryLog({
      artifactRoot,
      submissionPath: workingDmgPath,
      rawLogPath,
      profile,
      keychain,
      runCommand
    });
    const ticket = assertDmgTicket(notary.log, {
      expectedPath: path.basename(workingDmgPath),
      expectedCdHash: dmgCdhashBeforeStaple
    });
    assertWorkingDmgUnchanged('Post-notary notarization DMG snapshot');

    runCommand('/usr/bin/xcrun', ['stapler', 'staple', '/dev/fd/3'], {
      stdio: ['ignore', 'pipe', 'pipe', workingDmgDescriptor],
      pinnedInput: { descriptor: workingDmgDescriptor, path: workingDmgPath }
    });
    fs.fsyncSync(workingDmgDescriptor);
    const stapledWorkingDmgIdentity = artifactFileIdentity(
      artifactRoot,
      workingDmgPath,
      'Stapled notarized DMG snapshot'
    );
    const stapledWorkingDmgState = captureFileState(
      artifactRoot,
      workingDmgPath,
      'Stapled notarized DMG snapshot'
    );
    runCommand('/usr/bin/xcrun', ['stapler', 'validate', '/dev/fd/3'], {
      stdio: ['ignore', 'pipe', 'pipe', workingDmgDescriptor],
      pinnedInput: { descriptor: workingDmgDescriptor, path: workingDmgPath }
    });
    runCommand('/usr/bin/codesign', [
      '--verify', '--strict', '--verbose=2', workingDmgPath
    ]);
    const dmgCdhashAfterStaple = parseCodesignCdHash(combined(runCommand(
      '/usr/bin/codesign',
      ['-d', '--verbose=4', workingDmgPath]
    )));
    if (dmgCdhashAfterStaple !== dmgCdhashBeforeStaple) {
      throw new Error('DMG CDHash changed after stapling');
    }
    const validatedWorkingDmgIdentity = artifactFileIdentity(
      artifactRoot,
      workingDmgPath,
      'Validated notarized DMG snapshot'
    );
    assertIdentityUnchanged(
      stapledWorkingDmgIdentity,
      validatedWorkingDmgIdentity,
      'Stapled notarization DMG snapshot'
    );
    assertFileState(
      stapledWorkingDmgState,
      artifactRoot,
      workingDmgPath,
      'Validated notarized DMG snapshot'
    );
    if (
      JSON.stringify(pinnedDescriptorIdentity(
        workingDmgDescriptor,
        'Validated notarized DMG snapshot'
      )) !== JSON.stringify(workingDmgDescriptorIdentity)
    ) {
      throw new Error('Validated notarized DMG changed pinned descriptor identity');
    }

    assertIdentityUnchanged(
      originalDmgIdentity,
      artifactFileIdentity(
        artifactRoot,
        dmgPath,
        'Original DMG before notarized replacement'
      ),
      'Original DMG'
    );
    assertFileState(
      originalDmgState,
      artifactRoot,
      dmgPath,
      'Original DMG before notarized replacement'
    );
    fs.closeSync(workingDmgDescriptor);
    workingDmgDescriptorOpen = false;
    replaceArtifactFileNoFollow(
      artifactRoot,
      workingDmgPath,
      dmgPath,
      validatedWorkingDmgIdentity,
      'Notarized DMG publication'
    );

    const evidence = {
      schema: 'automixer.dmg_notarization.v2',
      pathPolicy: 'artifact_root_relative_v1',
      generatedAt: new Date().toISOString(),
      status: 'Accepted',
      submission: notary.submission,
      appleLog: notary.logSummary,
      rawLog: notary.rawLog,
      ticket,
      signature: {
        cdhashBeforeStaple: dmgCdhashBeforeStaple,
        cdhashAfterStaple: dmgCdhashAfterStaple
      },
      dmg: artifactFileIdentity(artifactRoot, dmgPath, 'Notarized DMG')
    };
    writeNewArtifactFileNoFollow(
      artifactRoot,
      evidencePath,
      `${JSON.stringify(evidence, null, 2)}\n`,
      'DMG notarization evidence'
    );
    return { status: 'Accepted', submissionId: notary.submission.id };
  } finally {
    if (workingDmgDescriptorOpen) {
      try {
        fs.closeSync(workingDmgDescriptor);
      } catch {}
      workingDmgDescriptorOpen = false;
    }
    if (fs.lstatSync(workingDmgPath, { throwIfNoEntry: false })) {
      try {
        removeArtifactFileNoFollow(
          artifactRoot,
          workingDmgPath,
          'Private notarization snapshot cleanup'
        );
      } catch {}
    }
  }
}

function main() {
  const result = notarizeDmg({
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
