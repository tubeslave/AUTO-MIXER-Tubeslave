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

const scriptPath = fileURLToPath(import.meta.url);

function sanitizeOutput(value) {
  return String(value || '')
    .replaceAll(os.homedir(), '~')
    .replace(/\b(password|token|secret|api[_-]?key)\s*[:=]\s*\S+/gi, '$1=[REDACTED]')
    .slice(-12_000);
}

function run(command, args, options = {}) {
  const completed = spawnSync(command, args, {
    encoding: 'utf8',
    maxBuffer: 32 * 1024 * 1024
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
    stdout: completed.stdout || '',
    stderr: completed.stderr || ''
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
  if (fs.lstatSync(evidencePath, { throwIfNoEntry: false })) {
    throw new Error('DMG notarization evidence already exists; refusing replacement');
  }

  const originalDmgIdentity = artifactFileIdentity(
    artifactRoot,
    dmgPath,
    'Original DMG before notarization'
  );
  const workingDmgPath = path.join(
    artifactRoot,
    `.notarize-working-${crypto.randomUUID()}.dmg`
  );
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
    if (
      workingDmgIdentity.size !== originalDmgIdentity.size
      || workingDmgIdentity.sha256 !== originalDmgIdentity.sha256
      || workingDmgIdentity.sha256 !== snapshotSha256
    ) {
      throw new Error('Private notarization DMG snapshot does not bind to the original');
    }

    runCommand('/usr/bin/codesign', [
      '--verify', '--strict', '--verbose=2', workingDmgPath
    ]);
    assertIdentityUnchanged(
      workingDmgIdentity,
      artifactFileIdentity(
        artifactRoot,
        workingDmgPath,
        'Post-codesign notarization DMG snapshot'
      ),
      'Private notarization DMG snapshot'
    );
    const submitArgs = [
      'notarytool',
      'submit',
      workingDmgPath,
      '--keychain-profile',
      profile,
      '--wait',
      '--output-format',
      'json'
    ];
    const keychain = String(environment.APPLE_KEYCHAIN || '').trim();
    if (keychain) submitArgs.push('--keychain', keychain);
    const submit = runCommand('/usr/bin/xcrun', submitArgs, { check: false });
    assertIdentityUnchanged(
      workingDmgIdentity,
      artifactFileIdentity(
        artifactRoot,
        workingDmgPath,
        'Post-submit notarization DMG snapshot'
      ),
      'Private notarization DMG snapshot'
    );
    let submission = {};
    try {
      submission = JSON.parse(submit.stdout || '{}');
    } catch {}
    if (submit.status !== 0 || submission.status !== 'Accepted' || !submission.id) {
      let notaryLog = '';
      if (submission.id) {
        const logArgs = [
          'notarytool', 'log', submission.id,
          '--keychain-profile', profile,
          '--output-format', 'json'
        ];
        if (keychain) logArgs.push('--keychain', keychain);
        const log = runCommand('/usr/bin/xcrun', logArgs, { check: false });
        notaryLog = sanitizeOutput(`${log.stderr}\n${log.stdout}`.trim());
      }
      const submitDetail = sanitizeOutput(`${submit.stderr}\n${submit.stdout}`.trim());
      throw new Error(
        `DMG notarization was not accepted (status=${submission.status || 'unknown'})`
        + (submitDetail ? `; submit=${submitDetail}` : '')
        + (notaryLog ? `; log=${notaryLog}` : '')
      );
    }

    runCommand('/usr/bin/xcrun', ['stapler', 'staple', workingDmgPath]);
    const stapledWorkingDmgIdentity = artifactFileIdentity(
      artifactRoot,
      workingDmgPath,
      'Stapled notarized DMG snapshot'
    );
    runCommand('/usr/bin/xcrun', ['stapler', 'validate', workingDmgPath]);
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

    assertIdentityUnchanged(
      originalDmgIdentity,
      artifactFileIdentity(
        artifactRoot,
        dmgPath,
        'Original DMG before notarized replacement'
      ),
      'Original DMG'
    );
    replaceArtifactFileNoFollow(
      artifactRoot,
      workingDmgPath,
      dmgPath,
      validatedWorkingDmgIdentity,
      'Notarized DMG publication'
    );

    const evidence = {
      schema: 'automixer.dmg_notarization.v1',
      pathPolicy: 'artifact_root_relative_v1',
      generatedAt: new Date().toISOString(),
      status: 'Accepted',
      submissionId: submission.id,
      dmg: artifactFileIdentity(artifactRoot, dmgPath, 'Notarized DMG')
    };
    writeNewArtifactFileNoFollow(
      artifactRoot,
      evidencePath,
      `${JSON.stringify(evidence, null, 2)}\n`,
      'DMG notarization evidence'
    );
    return { status: 'Accepted', submissionId: submission.id };
  } finally {
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
