import test from 'node:test';
import assert from 'node:assert/strict';
import crypto from 'node:crypto';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { createRequire } from 'node:module';
import { fileURLToPath, pathToFileURL } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const frontendRoot = path.resolve(__dirname, '../../..');
const require = createRequire(import.meta.url);
const { notarizeDmg } = await import(pathToFileURL(
  path.join(frontendRoot, 'scripts', 'notarize-dmg.mjs')
));

test('DMG notarization rejects an escaped symlink before every external command', () => {
  const tempRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'automixer-notarize-symlink-'));
  const artifactRoot = path.join(tempRoot, 'artifact');
  const outsideDmg = path.join(tempRoot, 'outside.dmg');
  const dmgPath = path.join(artifactRoot, 'release.dmg');
  const commands = [];
  try {
    fs.mkdirSync(artifactRoot);
    fs.writeFileSync(outsideDmg, 'outside DMG');
    fs.symlinkSync(outsideDmg, dmgPath);

    assert.throws(() => notarizeDmg({
      artifactRoot,
      dmgPath,
      environment: {
        AUTOMIXER_MAC_BUILD_PROFILE: 'release',
        APPLE_KEYCHAIN_PROFILE: 'unit-test-profile'
      },
      runCommand(command, args) {
        commands.push([command, ...args]);
        return { status: 0, stdout: '', stderr: '' };
      }
    }), /regular file, not a symlink/);

    assert.deepEqual(commands, []);
    assert.equal(
      fs.existsSync(path.join(artifactRoot, 'dmg-notarization.json')),
      false
    );
  } finally {
    fs.rmSync(tempRoot, { recursive: true, force: true });
  }
});

test('DMG notarization refuses a symlink evidence output before external commands', () => {
  const tempRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'automixer-notarize-output-'));
  const artifactRoot = path.join(tempRoot, 'artifact');
  const dmgPath = path.join(artifactRoot, 'release.dmg');
  const outsideEvidence = path.join(tempRoot, 'outside.json');
  const evidencePath = path.join(artifactRoot, 'dmg-notarization.json');
  const commands = [];
  try {
    fs.mkdirSync(artifactRoot);
    fs.writeFileSync(dmgPath, 'regular DMG');
    fs.writeFileSync(outsideEvidence, 'DO NOT REPLACE');
    fs.symlinkSync(outsideEvidence, evidencePath);

    assert.throws(() => notarizeDmg({
      artifactRoot,
      dmgPath,
      environment: {
        AUTOMIXER_MAC_BUILD_PROFILE: 'release',
        APPLE_KEYCHAIN_PROFILE: 'unit-test-profile'
      },
      runCommand(command, args) {
        commands.push([command, ...args]);
        return { status: 0, stdout: '', stderr: '' };
      }
    }), /output already exists/);

    assert.deepEqual(commands, []);
    assert.equal(fs.readFileSync(outsideEvidence, 'utf8'), 'DO NOT REPLACE');
  } finally {
    fs.rmSync(tempRoot, { recursive: true, force: true });
  }
});

test('DMG notarization commands operate on a private snapshot during input replacement race', () => {
  const tempRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'automixer-notarize-race-'));
  const artifactRoot = path.join(tempRoot, 'artifact');
  const dmgPath = path.join(artifactRoot, 'release.dmg');
  const outsideDmg = path.join(tempRoot, 'outside.dmg');
  const commands = [];
  const dmgCdhash = 'b'.repeat(40);
  let replaced = false;
  let submittedPath = '';
  try {
    fs.mkdirSync(artifactRoot);
    fs.writeFileSync(dmgPath, 'trusted DMG');
    fs.writeFileSync(outsideDmg, 'outside DMG');

    assert.throws(() => notarizeDmg({
      artifactRoot,
      dmgPath,
      environment: {
        AUTOMIXER_MAC_BUILD_PROFILE: 'release',
        APPLE_KEYCHAIN_PROFILE: 'unit-test-profile'
      },
      runCommand(command, args) {
        commands.push([command, ...args]);
        if (!replaced) {
          replaced = true;
          fs.rmSync(dmgPath);
          fs.symlinkSync(outsideDmg, dmgPath);
        }
        if (args[0] === 'notarytool' && args[1] === 'submit') {
          submittedPath = args[2];
          return {
            status: 0,
            stdout: JSON.stringify({
              status: 'Accepted',
              id: '12345678-1234-1234-1234-123456789abc'
            }),
            stderr: ''
          };
        }
        if (args[0] === 'notarytool' && args[1] === 'log') {
          return {
            status: 0,
            stdout: JSON.stringify({
              logFormatVersion: 1,
              jobId: '12345678-1234-1234-1234-123456789abc',
              status: 'Accepted',
              statusCode: 0,
              archiveFilename: path.basename(submittedPath),
              uploadDate: new Date().toISOString(),
              sha256: crypto.createHash('sha256')
                .update(fs.readFileSync(submittedPath)).digest('hex'),
              ticketContents: [{
                path: path.basename(submittedPath),
                digestAlgorithm: 'SHA-256',
                cdhash: dmgCdhash
              }],
              issues: null
            }),
            stderr: ''
          };
        }
        if (command === '/usr/bin/codesign' && args[0] === '-d') {
          return { status: 0, stdout: '', stderr: `CDHash=${dmgCdhash}\n` };
        }
        return { status: 0, stdout: '', stderr: '' };
      }
    }), /existing regular file|changed identity|resolves outside/);

    assert.equal(commands.length, 8);
    assert.equal(
      commands.some(command => command.includes(dmgPath) || command.includes(outsideDmg)),
      false
    );
    assert.equal(
      commands
        .filter(command => !(command[1] === 'notarytool' && command[2] === 'log'))
        .every(command => command.some(value => (
          String(value).includes('.notarize-working-') || value === '/dev/fd/3'
        ))),
      true
    );
    assert.equal(fs.readFileSync(outsideDmg, 'utf8'), 'outside DMG');
    assert.equal(fs.lstatSync(dmgPath).isSymbolicLink(), true);
    assert.equal(
      fs.readdirSync(artifactRoot).some(name => name.startsWith('.notarize-working-')),
      false
    );
  } finally {
    fs.rmSync(tempRoot, { recursive: true, force: true });
  }
});

test('DMG notarization rejects a mutated working snapshot before publication', () => {
  const tempRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'automixer-notarize-snapshot-race-'));
  const artifactRoot = path.join(tempRoot, 'artifact');
  const dmgPath = path.join(artifactRoot, 'release.dmg');
  let commandCalls = 0;
  try {
    fs.mkdirSync(artifactRoot);
    fs.writeFileSync(dmgPath, 'trusted DMG');

    assert.throws(() => notarizeDmg({
      artifactRoot,
      dmgPath,
      environment: {
        AUTOMIXER_MAC_BUILD_PROFILE: 'release',
        APPLE_KEYCHAIN_PROFILE: 'unit-test-profile'
      },
      runCommand(_command, args) {
        commandCalls += 1;
        const workingPath = args.find(value => (
          String(value).includes('.notarize-working-')
        ));
        assert.ok(workingPath);
        fs.writeFileSync(workingPath, 'ATTACKER-DMG');
        return { status: 0, stdout: '', stderr: '' };
      }
    }), /changed during release verification/);

    assert.equal(commandCalls, 1);
    assert.equal(fs.readFileSync(dmgPath, 'utf8'), 'trusted DMG');
    assert.equal(fs.existsSync(path.join(artifactRoot, 'dmg-notarization.json')), false);
  } finally {
    fs.rmSync(tempRoot, { recursive: true, force: true });
  }
});

test('DMG notarization detects restored-byte mutation through file metadata identity', () => {
  const tempRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'automixer-notarize-restore-'));
  const artifactRoot = path.join(tempRoot, 'artifact');
  const dmgPath = path.join(artifactRoot, 'release.dmg');
  try {
    fs.mkdirSync(artifactRoot);
    fs.writeFileSync(dmgPath, 'trusted DMG');
    assert.throws(() => notarizeDmg({
      artifactRoot,
      dmgPath,
      environment: {
        AUTOMIXER_MAC_BUILD_PROFILE: 'release',
        APPLE_KEYCHAIN_PROFILE: 'unit-test-profile'
      },
      runCommand(_command, args) {
        const workingPath = args.find(value => (
          String(value).includes('.notarize-working-')
        ));
        assert.ok(workingPath);
        const original = fs.readFileSync(workingPath);
        fs.writeFileSync(workingPath, 'temporary attacker bytes');
        fs.writeFileSync(workingPath, original);
        return { status: 0, stdout: '', stderr: '' };
      }
    }), /changed file identity or state/);
    assert.equal(fs.readFileSync(dmgPath, 'utf8'), 'trusted DMG');
    assert.equal(fs.existsSync(path.join(artifactRoot, 'dmg-notary-log.json')), false);
  } finally {
    fs.rmSync(tempRoot, { recursive: true, force: true });
  }
});

test('DMG stapling stays on the pinned descriptor during pathname replacement', () => {
  const tempRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'automixer-dmg-staple-race-'));
  const artifactRoot = path.join(tempRoot, 'artifact');
  const dmgPath = path.join(artifactRoot, 'release.dmg');
  const outsideDmg = path.join(tempRoot, 'outside.dmg');
  const cdhash = 'b'.repeat(40);
  const submissionId = '12345678-1234-1234-1234-123456789abc';
  let submittedPath = '';
  let stapleCalls = 0;
  try {
    fs.mkdirSync(artifactRoot);
    fs.writeFileSync(dmgPath, 'trusted DMG');
    fs.writeFileSync(outsideDmg, 'DO NOT WRITE');
    assert.throws(() => notarizeDmg({
      artifactRoot,
      dmgPath,
      environment: {
        AUTOMIXER_MAC_BUILD_PROFILE: 'release',
        APPLE_KEYCHAIN_PROFILE: 'unit-test-profile'
      },
      runCommand(command, args, options = {}) {
        if (command === '/usr/bin/codesign' && args[0] === '-d') {
          return { status: 0, stdout: '', stderr: `CDHash=${cdhash}\n` };
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
            stdout: JSON.stringify({
              logFormatVersion: 1,
              jobId: submissionId,
              status: 'Accepted',
              statusCode: 0,
              archiveFilename: path.basename(submittedPath),
              uploadDate: new Date().toISOString(),
              sha256: crypto.createHash('sha256')
                .update(fs.readFileSync(submittedPath)).digest('hex'),
              ticketContents: [{
                path: path.basename(submittedPath),
                digestAlgorithm: 'SHA-256',
                cdhash
              }],
              issues: null
            }),
            stderr: ''
          };
        }
        if (args[0] === 'stapler' && args[1] === 'staple') {
          stapleCalls += 1;
          fs.rmSync(options.pinnedInput.path);
          fs.symlinkSync(outsideDmg, options.pinnedInput.path);
          fs.writeSync(options.pinnedInput.descriptor, Buffer.from('ticket'));
        }
        return { status: 0, stdout: '', stderr: '' };
      }
    }), /regular file|symlink|resolves outside|identity/);
    assert.equal(stapleCalls, 1);
    assert.equal(fs.readFileSync(outsideDmg, 'utf8'), 'DO NOT WRITE');
    assert.equal(fs.readFileSync(dmgPath, 'utf8'), 'trusted DMG');
    assert.equal(fs.existsSync(path.join(artifactRoot, 'dmg-notarization.json')), false);
  } finally {
    fs.rmSync(tempRoot, { recursive: true, force: true });
  }
});

test('macOS signing assigns a dedicated PyInstaller backend entitlement profile', () => {
  const packageJson = JSON.parse(fs.readFileSync(
    path.join(frontendRoot, 'package.json'),
    'utf8'
  ));
  const backendEntitlements = fs.readFileSync(
    path.join(frontendRoot, 'public', 'entitlements.backend.plist'),
    'utf8'
  );
  const inheritedEntitlements = fs.readFileSync(
    path.join(frontendRoot, 'public', 'entitlements.mac.inherit.plist'),
    'utf8'
  );
  const signer = fs.readFileSync(
    path.join(frontendRoot, 'scripts', 'sign-macos.cjs'),
    'utf8'
  );

  assert.equal(packageJson.build.mac.entitlements, 'public/entitlements.mac.plist');
  assert.equal(
    packageJson.build.mac.entitlementsInherit,
    'public/entitlements.mac.inherit.plist'
  );
  assert.equal(packageJson.build.mac.sign, 'scripts/sign-macos.cjs');
  assert.equal(packageJson.build.mac.strictVerify, true);
  assert.notEqual(
    packageJson.build.mac.entitlements,
    packageJson.build.mac.entitlementsInherit
  );

  assert.match(
    backendEntitlements,
    /com\.apple\.security\.cs\.disable-library-validation/
  );
  assert.doesNotMatch(backendEntitlements, /allow-jit/);
  assert.doesNotMatch(backendEntitlements, /allow-unsigned-executable-memory/);
  assert.match(inheritedEntitlements, /allow-jit/);
  assert.match(inheritedEntitlements, /allow-unsigned-executable-memory/);

  assert.match(signer, /Contents[\s\S]*Resources[\s\S]*backend[\s\S]*server/);
  assert.match(signer, /scopedBackendEntitlements/);
  assert.match(signer, /return backendEntitlements/);
  assert.match(signer, /backendVisited/);
  assert.match(signer, /MACH_O_MAGICS/);
  assert.match(signer, /buildSignIgnore\(options\.ignore\)/);
  assert.match(signer, /validateOptsIgnore\(\) has no array return branch/);
  assert.match(signer, /AUTOMIXER_MAC_BUILD_PROFILE === 'release'/);
  assert.match(signer, /AUTOMIXER_RELEASE_BACKEND_SHA256/);
  assert.match(signer, /AUTOMIXER_RELEASE_BACKEND_BUNDLE_TREE_SHA256/);
  assert.match(signer, /actualBackendSha256 !== required\.backendSha256/);
  assert.match(signer, /automixer\.macos_release_build_attestation\.v1/);
  assert.match(signer, /Refusing to replace an existing release build attestation/);
  assert.match(signer, /options\.type !== 'distribution'/);
  assert.match(signer, /explicit distribution identity/);
  assert.match(
    signer,
    /options\.strictVerify === true \? undefined : options\.strictVerify/
  );
  assert.match(signer, /signAsync\(signOptions\)/);
});

test('macOS signing gives nested backend Mach-O files empty entitlements', () => {
  const signer = require(path.join(frontendRoot, 'scripts', 'sign-macos.cjs'));
  const tempRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'automixer-backend-entitlements-'));
  const backendPath = path.join(tempRoot, 'backend', 'server');
  const nestedMachO = path.join(tempRoot, 'backend', '_internal', 'runtime.so');
  const nestedData = path.join(tempRoot, 'backend', '_internal', 'data.json');
  fs.mkdirSync(path.dirname(nestedMachO), { recursive: true });
  fs.writeFileSync(backendPath, Buffer.from([0xfe, 0xed, 0xfa, 0xcf, 0x00]));
  fs.writeFileSync(nestedMachO, Buffer.from([0xfe, 0xed, 0xfa, 0xcf, 0x00]));
  fs.writeFileSync(nestedData, '{}');
  const context = {
    backendPath,
    backendEntitlements: '/tmp/backend.plist',
    emptyEntitlements: '/tmp/empty.plist'
  };
  try {
    assert.equal(
      signer.scopedBackendEntitlements(backendPath, context),
      context.backendEntitlements
    );
    assert.equal(
      signer.scopedBackendEntitlements(nestedMachO, context),
      context.emptyEntitlements
    );
    assert.equal(signer.scopedBackendEntitlements(nestedData, context), null);
  } finally {
    fs.rmSync(tempRoot, { recursive: true, force: true });
  }
});

test('local macOS package receives a verified ad-hoc signature before archiving', () => {
  const signer = require(path.join(frontendRoot, 'scripts', 'sign-macos.cjs'));
  const calls = [];
  const appPath = '/tmp/AUTO MIXER Tubeslave.app';
  const result = signer.adHocSignLocalApp(appPath, (command, args) => {
    calls.push([command, ...args]);
    return { status: 0, stdout: '', stderr: '' };
  });

  assert.deepEqual(result, { signed: true, identity: '-', verified: true });
  assert.deepEqual(calls, [
    ['/usr/bin/codesign', '--force', '--deep', '--sign', '-', appPath],
    ['/usr/bin/codesign', '--verify', '--deep', '--strict', '--verbose=2', appPath]
  ]);
});

test('macOS signing ignore rule preserves inherited rules and excludes non-Mach-O data', () => {
  const signer = require(path.join(frontendRoot, 'scripts', 'sign-macos.cjs'));
  const temporaryDirectory = fs.mkdtempSync(path.join(os.tmpdir(), 'automixer-sign-ignore-'));
  try {
    const machO = path.join(temporaryDirectory, 'backend');
    const locale = path.join(temporaryDirectory, 'locale.pak');
    const inherited = path.join(temporaryDirectory, 'inherited.bin');
    fs.writeFileSync(machO, Buffer.from([0xfe, 0xed, 0xfa, 0xcf, 0x00]));
    fs.writeFileSync(locale, Buffer.from([0x05, 0x00, 0x00, 0x00]));
    fs.writeFileSync(inherited, Buffer.from([0xfe, 0xed, 0xfa, 0xcf, 0x00]));

    const shouldIgnore = signer.buildSignIgnore([
      filePath => filePath === inherited,
      'never-matches-this-path'
    ]);
    assert.equal(shouldIgnore(machO), false);
    assert.equal(shouldIgnore(locale), true);
    assert.equal(shouldIgnore(inherited), true);
    assert.equal(shouldIgnore(temporaryDirectory), false);
  } finally {
    fs.rmSync(temporaryDirectory, { recursive: true, force: true });
  }
});
