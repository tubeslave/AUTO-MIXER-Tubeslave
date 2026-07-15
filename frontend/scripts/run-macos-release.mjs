#!/usr/bin/env node

import fs from 'node:fs';
import path from 'node:path';
import process from 'node:process';
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';

const scriptPath = fileURLToPath(import.meta.url);
const frontendRoot = path.resolve(path.dirname(scriptPath), '..');

export const RELEASE_NODE_VERSION = 'v22.22.3';
export const RELEASE_NPM_VERSION = '10.9.8';

function realExistingFile(candidate, label) {
  if (!path.isAbsolute(String(candidate || '')) || /[\0\r\n]/.test(String(candidate))) {
    throw new Error(`${label} must be an absolute path without control characters`);
  }
  const stat = fs.lstatSync(candidate);
  if (stat.isSymbolicLink() || !stat.isFile()) {
    throw new Error(`${label} must be a real file`);
  }
  return fs.realpathSync(candidate);
}

function realExistingDirectory(candidate, label) {
  if (!path.isAbsolute(String(candidate || '')) || /[\0\r\n]/.test(String(candidate))) {
    throw new Error(`${label} must be an absolute path without control characters`);
  }
  const stat = fs.lstatSync(candidate);
  if (stat.isSymbolicLink() || !stat.isDirectory()) {
    throw new Error(`${label} must be a real directory`);
  }
  return fs.realpathSync(candidate);
}

export function validateDirectReleaseEnvironment(
  environment,
  {
    frontendDirectory = frontendRoot,
    workingDirectory = process.cwd(),
    nodeExecutable = process.execPath,
    nodeVersion = process.version,
    platform = process.platform,
    architecture = process.arch
  } = {}
) {
  if (environment.AUTOMIXER_RELEASE_CLEAN_ENV !== '1') {
    throw new Error(
      'macOS release must enter through /bin/sh -p scripts/run-macos-release'
    );
  }
  const inheritedLifecycleInputs = Object.keys(environment).filter(name => (
    /^npm_/i.test(name)
    || name === 'INIT_CWD'
    || name === 'NODE'
    || name === 'NODE_OPTIONS'
  ));
  if (inheritedLifecycleInputs.length) {
    throw new Error(
      'Authoritative macOS release must be launched directly, outside npm lifecycle: '
      + inheritedLifecycleInputs.sort().join(', ')
    );
  }
  if (nodeVersion !== RELEASE_NODE_VERSION || platform !== 'darwin' || architecture !== 'arm64') {
    throw new Error(`macOS release requires Node ${RELEASE_NODE_VERSION} on darwin arm64`);
  }

  const expectedFrontend = realExistingDirectory(frontendDirectory, 'frontend root');
  if (realExistingDirectory(workingDirectory, 'release launcher cwd') !== expectedFrontend) {
    throw new Error('Authoritative macOS release launcher must run from the frontend root');
  }
  const expectedNode = realExistingFile(nodeExecutable, 'release Node executable');
  const nodePrefix = path.resolve(path.dirname(expectedNode), '..');
  const npmRoot = path.join(nodePrefix, 'lib', 'node_modules', 'npm');
  const npmCli = realExistingFile(
    path.join(npmRoot, 'bin', 'npm-cli.js'),
    'pinned npm CLI'
  );
  const npmPackage = JSON.parse(fs.readFileSync(path.join(npmRoot, 'package.json'), 'utf8'));
  if (npmPackage.version !== RELEASE_NPM_VERSION) {
    throw new Error(`macOS release requires npm ${RELEASE_NPM_VERSION}`);
  }
  return {
    frontendRoot: expectedFrontend,
    nodeExecutable: expectedNode,
    npmCli
  };
}

export function createDirectReleaseEnvironment(
  environment,
  { nodeExecutable = process.execPath } = {}
) {
  const child = { ...environment };
  delete child.AUTOMIXER_RELEASE_CLEAN_ENV;
  child.PATH = [...new Set([
    path.dirname(fs.realpathSync(nodeExecutable)),
    '/opt/homebrew/bin',
    '/usr/local/bin',
    '/usr/bin',
    '/bin',
    '/usr/sbin',
    '/sbin'
  ])].join(':');
  return child;
}

export function runDirectRelease({
  environment = process.env,
  frontendDirectory = frontendRoot,
  workingDirectory = process.cwd(),
  nodeExecutable = process.execPath,
  nodeVersion = process.version,
  platform = process.platform,
  architecture = process.arch,
  runCommand = spawnSync
} = {}) {
  const validated = validateDirectReleaseEnvironment(environment, {
    frontendDirectory,
    workingDirectory,
    nodeExecutable,
    nodeVersion,
    platform,
    architecture
  });
  const completed = runCommand(validated.nodeExecutable, [
    path.join(validated.frontendRoot, 'scripts', 'build-macos.mjs'),
    '--profile',
    'release'
  ], {
    cwd: validated.frontendRoot,
    env: createDirectReleaseEnvironment(environment, {
      nodeExecutable: validated.nodeExecutable
    }),
    stdio: 'inherit',
    shell: false
  });
  if (completed.error) throw completed.error;
  if (completed.status !== 0) {
    throw new Error(`macOS release builder exited with status ${completed.status}`);
  }
}

if (process.argv[1] && path.resolve(process.argv[1]) === scriptPath) {
  try {
    runDirectRelease();
  } catch (error) {
    process.stderr.write(`${error?.message || error}\n`);
    process.exitCode = 1;
  }
}
