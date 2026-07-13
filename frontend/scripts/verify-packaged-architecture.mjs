import { execFileSync } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';
import process from 'node:process';
import { fileURLToPath } from 'node:url';

export function normalizeArchitecture(value) {
  const normalized = String(value || '').trim().toLowerCase();
  if (['x64', 'x86_64', 'amd64'].includes(normalized)) return 'x64';
  if (['arm64', 'aarch64'].includes(normalized)) return 'arm64';
  return normalized;
}

export function parseArchitectures(output) {
  const architectures = String(output || '')
    .trim()
    .split(/\s+/)
    .map(normalizeArchitecture)
    .filter(Boolean);
  return [...new Set(architectures)].sort();
}

export function assertMatchingArchitectures({
  appArchitectures,
  backendArchitectures,
  expectedArchitecture
}) {
  const app = parseArchitectures(appArchitectures);
  const backend = parseArchitectures(backendArchitectures);
  const expected = normalizeArchitecture(expectedArchitecture);
  const sameSet = app.length === backend.length
    && app.every((architecture, index) => architecture === backend[index]);

  if (!expected || !app.includes(expected) || !backend.includes(expected) || !sameSet) {
    throw new Error(
      `Packaged architecture mismatch: expected=${expected || 'unset'} `
      + `electron=${app.join(',') || 'missing'} backend=${backend.join(',') || 'missing'}`
    );
  }

  return { app, backend, expected };
}

export function assertExactArchitecture(output, expectedArchitecture, label) {
  const actual = parseArchitectures(output);
  const expected = normalizeArchitecture(expectedArchitecture);
  if (actual.length !== 1 || actual[0] !== expected) {
    throw new Error(
      `${label} must be ${expected}-only; found ${actual.join(',') || 'missing'}`
    );
  }
  return actual[0];
}

function isMachOFile(filePath) {
  const stat = fs.lstatSync(filePath);
  if (!stat.isFile()) return false;
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

export function collectMachOFiles(root) {
  const files = [];
  function visit(candidate) {
    const stat = fs.lstatSync(candidate);
    if (stat.isSymbolicLink()) return;
    if (stat.isDirectory()) {
      for (const name of fs.readdirSync(candidate).sort()) visit(path.join(candidate, name));
      return;
    }
    if (isMachOFile(candidate)) files.push(candidate);
  }
  visit(root);
  return files;
}

function requirePath(targetPath, label) {
  if (!fs.existsSync(targetPath)) {
    throw new Error(`Missing packaged ${label}: ${targetPath}`);
  }
  return targetPath;
}

export function verifyPackagedArchitecture({
  frontendRoot,
  targetArchitecture,
  appPath,
  runLipo = targetPath => execFileSync('/usr/bin/lipo', ['-archs', targetPath], {
    encoding: 'utf8'
  })
}) {
  const expected = normalizeArchitecture(targetArchitecture);
  const packageJson = JSON.parse(
    fs.readFileSync(path.join(frontendRoot, 'package.json'), 'utf8')
  );
  const productName = packageJson.build.productName;
  const outputDirectory = expected === 'arm64' ? 'mac-arm64' : 'mac';
  const appRoot = appPath
    ? path.resolve(appPath)
    : path.join(frontendRoot, 'dist', outputDirectory, `${productName}.app`);
  const electronExecutable = requirePath(
    path.join(appRoot, 'Contents', 'MacOS', productName),
    'Electron executable'
  );
  const backendExecutable = requirePath(
    path.join(appRoot, 'Contents', 'Resources', 'backend', 'server'),
    'backend executable'
  );
  const primary = assertMatchingArchitectures({
    appArchitectures: runLipo(electronExecutable),
    backendArchitectures: runLipo(backendExecutable),
    expectedArchitecture: expected
  });
  const appMachOFiles = collectMachOFiles(appRoot);
  if (appMachOFiles.length < 3) {
    throw new Error('Packaged application has no nested Mach-O runtime');
  }
  for (const candidate of appMachOFiles) {
    assertExactArchitecture(
      runLipo(candidate),
      expected,
      `Packaged application ${path.relative(appRoot, candidate)}`
    );
  }
  const backendBundle = path.dirname(backendExecutable);
  const backendMachOFiles = collectMachOFiles(backendBundle);
  if (backendMachOFiles.length < 2) {
    throw new Error('Packaged backend onedir bundle has no nested Mach-O runtime');
  }
  for (const candidate of backendMachOFiles) {
    assertExactArchitecture(
      runLipo(candidate),
      expected,
      `Packaged backend ${path.relative(backendBundle, candidate)}`
    );
  }
  return {
    ...primary,
    appMachOCount: appMachOFiles.length,
    backendMachOCount: backendMachOFiles.length
  };
}

function main() {
  const scriptDirectory = path.dirname(fileURLToPath(import.meta.url));
  const frontendRoot = path.resolve(scriptDirectory, '..');
  const targetArchitecture = process.env.AUTOMIXER_TARGET_ARCH;
  const result = verifyPackagedArchitecture({
    frontendRoot,
    targetArchitecture,
    appPath: process.env.AUTOMIXER_APP_PATH
  });
  process.stdout.write(`${JSON.stringify({ status: 'passed', ...result })}\n`);
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  main();
}
