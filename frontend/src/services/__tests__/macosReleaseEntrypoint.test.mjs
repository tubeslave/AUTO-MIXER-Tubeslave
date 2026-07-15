import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { spawnSync } from 'node:child_process';
import { fileURLToPath, pathToFileURL } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const frontendRoot = path.resolve(__dirname, '../../..');
const wrapperPath = path.join(frontendRoot, 'scripts', 'run-macos-release.mjs');
const {
  createDirectReleaseEnvironment,
  runDirectRelease,
  validateDirectReleaseEnvironment
} = await import(pathToFileURL(wrapperPath));

function fixture() {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'automixer-release-entrypoint-'));
  const prefix = path.join(root, 'node-prefix');
  const nodeExecutable = path.join(prefix, 'bin', 'node');
  const npmRoot = path.join(prefix, 'lib', 'node_modules', 'npm');
  const npmCli = path.join(npmRoot, 'bin', 'npm-cli.js');
  fs.mkdirSync(path.dirname(nodeExecutable), { recursive: true });
  fs.mkdirSync(path.dirname(npmCli), { recursive: true });
  fs.writeFileSync(nodeExecutable, 'test node');
  fs.writeFileSync(npmCli, 'test npm cli');
  fs.writeFileSync(path.join(npmRoot, 'package.json'), JSON.stringify({ version: '10.9.8' }));
  return {
    root,
    nodeExecutable,
    environment: {
      HOME: path.join(root, 'home'),
      PATH: `${path.join(root, 'hostile-bin')}:/usr/bin:/bin`,
      AUTOMIXER_RELEASE_CLEAN_ENV: '1'
    }
  };
}

function validationOptions(current) {
  return {
    frontendDirectory: frontendRoot,
    workingDirectory: frontendRoot,
    nodeExecutable: current.nodeExecutable,
    nodeVersion: 'v22.22.3',
    platform: 'darwin',
    architecture: 'arm64'
  };
}

test('direct release launcher pins Node/npm and replaces inherited PATH', () => {
  const current = fixture();
  try {
    assert.doesNotThrow(() => validateDirectReleaseEnvironment(
      current.environment,
      validationOptions(current)
    ));
    const child = createDirectReleaseEnvironment(current.environment, {
      nodeExecutable: current.nodeExecutable
    });
    assert.equal(
      child.PATH.split(':')[0],
      path.dirname(fs.realpathSync(current.nodeExecutable))
    );
    assert.doesNotMatch(child.PATH, /hostile-bin/);
    assert.equal(child.AUTOMIXER_RELEASE_CLEAN_ENV, undefined);
  } finally {
    fs.rmSync(current.root, { recursive: true, force: true });
  }
});

test('direct release launcher invokes the core builder with no shell', () => {
  const current = fixture();
  const calls = [];
  try {
    runDirectRelease({
      environment: current.environment,
      ...validationOptions(current),
      runCommand(command, args, options) {
        calls.push({ command, args, options });
        return { status: 0 };
      }
    });
    assert.equal(calls.length, 1);
    assert.equal(calls[0].command, fs.realpathSync(current.nodeExecutable));
    assert.deepEqual(calls[0].args, [
      path.join(frontendRoot, 'scripts', 'build-macos.mjs'),
      '--profile',
      'release'
    ]);
    assert.equal(calls[0].options.cwd, fs.realpathSync(frontendRoot));
    assert.equal(calls[0].options.shell, false);
  } finally {
    fs.rmSync(current.root, { recursive: true, force: true });
  }
});

test('direct release launcher rejects every npm and pre-exec Node surface', () => {
  const current = fixture();
  const options = validationOptions(current);
  try {
    for (const name of [
      'npm_config_script_shell',
      'NPM_CONFIG_CACHE',
      'npm_lifecycle_event',
      'INIT_CWD',
      'NODE',
      'NODE_OPTIONS'
    ]) {
      assert.throws(() => validateDirectReleaseEnvironment({
        ...current.environment,
        [name]: name === 'NODE_OPTIONS' ? '--require /tmp/inject.js' : '/tmp/value'
      }, options), /outside npm lifecycle/);
    }
  } finally {
    fs.rmSync(current.root, { recursive: true, force: true });
  }
});

test('inner release verifier rejects direct JavaScript entry without clean-env marker', () => {
  const current = fixture();
  try {
    const environment = { ...current.environment };
    delete environment.AUTOMIXER_RELEASE_CLEAN_ENV;
    assert.throws(() => validateDirectReleaseEnvironment(
      environment,
      validationOptions(current)
    ), /\/bin\/sh -p scripts\/run-macos-release/);
  } finally {
    fs.rmSync(current.root, { recursive: true, force: true });
  }
});

test('outer fixed-shell launcher clears NODE_OPTIONS before Node starts', () => {
  if (
    process.platform !== 'darwin'
    || process.arch !== 'arm64'
    || process.version !== 'v22.22.3'
  ) return;
  const marker = path.join(
    os.tmpdir(),
    `automixer-node-options-marker-${process.pid}-${Date.now()}`
  );
  const shellMarker = `${marker}.shell`;
  const shellStartup = `${marker}.sh`;
  const hostilePath = `${marker}.path`;
  const hostilePathMarker = `${marker}.path-executed`;
  const preload = `import fs from 'node:fs';fs.writeFileSync(${JSON.stringify(marker)},'executed')`;
  try {
    fs.writeFileSync(shellStartup, `echo executed > ${JSON.stringify(shellMarker)}\n`);
    fs.mkdirSync(hostilePath);
    fs.writeFileSync(
      path.join(hostilePath, 'dirname'),
      `#!/bin/sh\necho executed > ${JSON.stringify(hostilePathMarker)}\nexec /usr/bin/dirname "$@"\n`
    );
    fs.chmodSync(path.join(hostilePath, 'dirname'), 0o755);
    const completed = spawnSync('/bin/sh', [
      '-p',
      path.join(frontendRoot, 'scripts', 'run-macos-release')
    ], {
      cwd: frontendRoot,
      env: {
        ...process.env,
        AUTOMIXER_NODE_BIN: process.execPath,
        BASH_ENV: shellStartup,
        ENV: shellStartup,
        PATH: hostilePath,
        NODE_OPTIONS: `--import=data:text/javascript,${encodeURIComponent(preload)}`
      },
      encoding: 'utf8'
    });
    assert.notEqual(completed.status, 0);
    assert.equal(fs.existsSync(marker), false);
    assert.equal(fs.existsSync(shellMarker), false);
    assert.equal(fs.existsSync(hostilePathMarker), false);
    assert.match(
      `${completed.stdout}\n${completed.stderr}`,
      /Release environment is incomplete/
    );
  } finally {
    fs.rmSync(marker, { force: true });
    fs.rmSync(shellMarker, { force: true });
    fs.rmSync(shellStartup, { force: true });
    fs.rmSync(hostilePathMarker, { force: true });
    fs.rmSync(hostilePath, { recursive: true, force: true });
  }
});

test('npm cannot false-green a removed mac:release script through script-shell', () => {
  const packageJson = JSON.parse(fs.readFileSync(path.join(frontendRoot, 'package.json'), 'utf8'));
  assert.equal(Object.hasOwn(packageJson.scripts, 'mac:release'), false);
  const npmExecutable = path.join(
    path.dirname(process.execPath),
    process.platform === 'win32' ? 'npm.cmd' : 'npm'
  );
  if (!fs.existsSync(npmExecutable)) return;
  const completed = spawnSync(npmExecutable, ['run', 'mac:release'], {
    cwd: frontendRoot,
    env: {
      ...process.env,
      npm_config_script_shell: '/usr/bin/true'
    },
    encoding: 'utf8'
  });
  assert.notEqual(completed.status, 0);
  assert.match(`${completed.stdout}\n${completed.stderr}`, /Missing script: "mac:release"/);
});
