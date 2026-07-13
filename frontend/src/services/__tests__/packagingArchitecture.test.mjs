import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';

import {
  assertExactArchitecture,
  assertMatchingArchitectures,
  normalizeArchitecture,
  parseArchitectures,
  verifyPackagedArchitecture
} from '../../../scripts/verify-packaged-architecture.mjs';

function writeMachO(filePath) {
  fs.mkdirSync(path.dirname(filePath), { recursive: true });
  fs.writeFileSync(filePath, Buffer.from([0xfe, 0xed, 0xfa, 0xcf, 0, 0, 0, 0]));
}

test('packaged architecture parser normalizes native macOS names', () => {
  assert.equal(normalizeArchitecture('x86_64'), 'x64');
  assert.equal(normalizeArchitecture('aarch64'), 'arm64');
  assert.deepEqual(parseArchitectures('x86_64 arm64\n'), ['arm64', 'x64']);
});

test('every onedir backend Mach-O must be exact target architecture', () => {
  assert.equal(assertExactArchitecture('arm64', 'arm64', 'nested backend'), 'arm64');
  assert.throws(
    () => assertExactArchitecture('x86_64 arm64', 'arm64', 'nested backend'),
    /arm64-only/
  );
  assert.throws(
    () => assertExactArchitecture('x86_64', 'arm64', 'nested backend'),
    /arm64-only/
  );
});

test('matching native and universal Electron/backend architecture sets pass', () => {
  assert.deepEqual(
    assertMatchingArchitectures({
      appArchitectures: 'arm64',
      backendArchitectures: 'arm64',
      expectedArchitecture: 'arm64'
    }),
    { app: ['arm64'], backend: ['arm64'], expected: 'arm64' }
  );

  assert.doesNotThrow(() => assertMatchingArchitectures({
    appArchitectures: 'x86_64 arm64',
    backendArchitectures: 'arm64 x86_64',
    expectedArchitecture: 'arm64'
  }));
});

test('mismatched Electron and backend architectures fail closed', () => {
  assert.throws(
    () => assertMatchingArchitectures({
      appArchitectures: 'x86_64',
      backendArchitectures: 'arm64',
      expectedArchitecture: 'x64'
    }),
    /Packaged architecture mismatch/
  );
});

test('packaged architecture rejects a mismatched nested Electron helper', () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'automixer-arch-contract-'));
  try {
    const frontendRoot = path.join(root, 'frontend');
    const appPath = path.join(root, 'AUTO MIXER Tubeslave.app');
    fs.mkdirSync(frontendRoot, { recursive: true });
    fs.writeFileSync(path.join(frontendRoot, 'package.json'), JSON.stringify({
      build: { productName: 'AUTO MIXER Tubeslave' }
    }));
    const electron = path.join(appPath, 'Contents', 'MacOS', 'AUTO MIXER Tubeslave');
    const backend = path.join(appPath, 'Contents', 'Resources', 'backend', 'server');
    const backendRuntime = path.join(
      appPath, 'Contents', 'Resources', 'backend', '_internal', 'libpython.dylib'
    );
    const helper = path.join(
      appPath, 'Contents', 'Frameworks', 'Electron Helper.app', 'Contents', 'MacOS', 'Electron Helper'
    );
    for (const candidate of [electron, backend, backendRuntime, helper]) writeMachO(candidate);

    assert.throws(() => verifyPackagedArchitecture({
      frontendRoot,
      targetArchitecture: 'arm64',
      appPath,
      runLipo: candidate => candidate === helper ? 'x86_64' : 'arm64'
    }), /Electron Helper.*arm64-only/);

    const result = verifyPackagedArchitecture({
      frontendRoot,
      targetArchitecture: 'arm64',
      appPath,
      runLipo: () => 'arm64'
    });
    assert.equal(result.appMachOCount, 4);
    assert.equal(result.backendMachOCount, 2);
  } finally {
    fs.rmSync(root, { recursive: true, force: true });
  }
});
