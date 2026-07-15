#!/usr/bin/env node

import crypto from 'node:crypto';
import fs from 'node:fs';
import path from 'node:path';
import process from 'node:process';
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';

import {
  artifactFileIdentity,
  artifactRelativePath,
  fileIdentity,
  hashTree,
  logicalFileIdentity,
  readStableJsonFileNoFollow,
  removeArtifactFileNoFollow,
  resolveArtifactPath,
  writeNewArtifactFileNoFollow
} from './release-evidence.mjs';

const scriptPath = fileURLToPath(import.meta.url);
const frontendRoot = path.resolve(path.dirname(scriptPath), '..');
const repoRoot = path.resolve(frontendRoot, '..');
const releasePythonVersion = '3.14.3';

function timestamp() {
  return new Date().toISOString().replace(/[-:]/g, '').replace(/\.\d{3}Z$/, 'Z');
}

function sha256Text(value) {
  return crypto.createHash('sha256').update(String(value)).digest('hex');
}

function run(command, args, options = {}) {
  const completed = spawnSync(command, args, {
    cwd: options.cwd || frontendRoot,
    env: options.env || process.env,
    encoding: 'utf8',
    stdio: options.stdio || 'inherit',
    maxBuffer: 64 * 1024 * 1024
  });
  if (completed.error) throw completed.error;
  if (completed.status !== 0) {
    const detail = `${completed.stderr || completed.stdout || ''}`.trim();
    throw new Error(
      `${path.basename(command)} failed with exit code ${completed.status}`
      + (detail ? `: ${detail}` : '')
    );
  }
  return { stdout: completed.stdout || '', stderr: completed.stderr || '' };
}

function gitOutput(args) {
  return run('/usr/bin/git', args, { cwd: repoRoot, stdio: 'pipe' }).stdout.trim();
}

function sha256FileStreaming(filePath) {
  const hash = crypto.createHash('sha256');
  const descriptor = fs.openSync(filePath, 'r');
  const buffer = Buffer.allocUnsafe(1024 * 1024);
  try {
    let bytesRead;
    do {
      bytesRead = fs.readSync(descriptor, buffer, 0, buffer.length, null);
      if (bytesRead) hash.update(buffer.subarray(0, bytesRead));
    } while (bytesRead);
  } finally {
    fs.closeSync(descriptor);
  }
  return hash.digest('hex');
}

function isSensitiveSourcePath(relativePath) {
  return String(relativePath).split('/').some(part => (
    /^\.env(?:\.|$)/i.test(part)
    || /(?:^|[-_.])(secret|token|credential)(?:[-_.]|$)/i.test(part)
    || /\.(?:key|pem|p12|pfx)$/i.test(part)
  ));
}

function isProductionSourceInput(relativePath) {
  const normalized = String(relativePath).split(path.sep).join('/');
  if (
    normalized.startsWith('frontend/dist/')
    || normalized.startsWith('frontend/node_modules/')
    || normalized.startsWith('backend/venv/')
    || normalized.startsWith('artifacts/')
    || normalized.startsWith('reports/')
  ) return false;
  const [top] = normalized.split('/');
  return new Set([
    '.github', 'backend', 'config', 'frontend', 'scripts', 'tests', 'tools'
  ]).has(top) || !normalized.includes('/');
}

function snapshotSourcePath(relativePath, rootDirectory) {
  if (isSensitiveSourcePath(relativePath)) {
    throw new Error(`Refusing to inspect sensitive source path: ${relativePath}`);
  }
  const absolutePath = path.resolve(rootDirectory, relativePath);
  const expectedPrefix = `${path.resolve(rootDirectory)}${path.sep}`;
  if (!absolutePath.startsWith(expectedPrefix)) {
    throw new Error(`Source path escapes checkout: ${relativePath}`);
  }
  let stat;
  try {
    stat = fs.lstatSync(absolutePath);
  } catch (error) {
    if (error?.code === 'ENOENT') return { path: relativePath, type: 'absent' };
    throw error;
  }
  const mode = stat.mode & 0o777;
  if (stat.isSymbolicLink()) {
    return {
      path: relativePath,
      type: 'symlink',
      mode,
      targetSha256: sha256Text(fs.readlinkSync(absolutePath))
    };
  }
  if (!stat.isFile()) return { path: relativePath, type: 'non_file', mode };
  return {
    path: relativePath,
    type: 'file',
    mode,
    size: stat.size,
    sha256: sha256FileStreaming(absolutePath)
  };
}

function nulPaths(value) {
  return String(value || '').split('\0').map(item => item.trim()).filter(Boolean).sort();
}

export function captureSourceSnapshot(readGit = gitOutput, rootDirectory = repoRoot) {
  const trackedPaths = nulPaths(
    readGit(['diff', 'HEAD', '--name-only', '-z'])
  ).filter(isProductionSourceInput);
  const untrackedPaths = nulPaths(
    readGit(['ls-files', '--others', '--exclude-standard', '-z'])
  ).filter(isProductionSourceInput);
  const contentManifest = {
    tracked: trackedPaths.map(relativePath => snapshotSourcePath(relativePath, rootDirectory)),
    untrackedSourceInputs: untrackedPaths.map(
      relativePath => snapshotSourcePath(relativePath, rootDirectory)
    )
  };
  return {
    head: readGit(['rev-parse', 'HEAD']),
    tree: readGit(['rev-parse', 'HEAD^{tree}']),
    status: readGit(['status', '--porcelain=v1', '-uall']),
    sourceContentSha256: sha256Text(JSON.stringify(contentManifest)),
    trackedChangedFileCount: contentManifest.tracked.length,
    untrackedSourceFileCount: contentManifest.untrackedSourceInputs.length
  };
}

export function assertSourceSnapshot(expected, actual, stage = 'release stage') {
  if (actual.head !== expected.head) {
    throw new Error(`Source HEAD changed during ${stage}`);
  }
  if (actual.tree !== expected.tree) {
    throw new Error(`Committed source tree changed during ${stage}`);
  }
  if (actual.status !== expected.status) {
    if (!expected.status && actual.status) {
      throw new Error(`Source checkout became dirty during ${stage}`);
    }
    throw new Error(`Source worktree status changed during ${stage}`);
  }
  if (
    expected.sourceContentSha256
    && actual.sourceContentSha256 !== expected.sourceContentSha256
  ) {
    throw new Error(`Source file contents changed during ${stage}`);
  }
}

export function runGuardedStage(
  stage,
  action,
  {
    expectedSource,
    readSource = captureSourceSnapshot,
    onPostconditionFailure = null
  }
) {
  assertSourceSnapshot(expectedSource, readSource(), `before ${stage}`);
  let result;
  try {
    result = action();
  } catch (error) {
    if (typeof onPostconditionFailure === 'function') {
      onPostconditionFailure(error);
    }
    throw error;
  }
  try {
    assertSourceSnapshot(expectedSource, readSource(), `after ${stage}`);
  } catch (error) {
    if (typeof onPostconditionFailure === 'function') {
      onPostconditionFailure(error);
    }
    throw error;
  }
  return result;
}

export function validateProfile(profile) {
  if (!['local', 'release'].includes(profile)) {
    throw new Error('--profile must be local or release');
  }
  return profile;
}

export function validateReleaseEnvironment(environment, worktreeStatus = '') {
  const required = [
    'AUTOMIXER_APPLE_TEAM_ID',
    'AUTOMIXER_DEVELOPER_ID_IDENTITY',
    'APPLE_KEYCHAIN_PROFILE'
  ];
  const missing = required.filter(name => !String(environment[name] || '').trim());
  if (missing.length) {
    throw new Error(`Release environment is incomplete: missing ${missing.join(', ')}`);
  }
  if (!/^[A-Z0-9]{10}$/.test(String(environment.AUTOMIXER_APPLE_TEAM_ID))) {
    throw new Error('AUTOMIXER_APPLE_TEAM_ID must be a 10-character Apple Team ID');
  }
  if (!/^Developer ID Application:/.test(String(environment.AUTOMIXER_DEVELOPER_ID_IDENTITY))) {
    throw new Error('AUTOMIXER_DEVELOPER_ID_IDENTITY must name a Developer ID Application identity');
  }
  if (String(worktreeStatus).trim()) {
    throw new Error('Release build requires a clean checkout including untracked files');
  }
  const forbiddenPatterns = [
    /^NODE_OPTIONS$/i,
    /^NPM_CONFIG_/i,
    /^npm_config_/,
    /^PIP_/i,
    /^PYTHON(PATH|HOME)$/i,
    /^ELECTRON_/i,
    /^CSC_/i,
    /^APPLE_ID$/i,
    /^APPLE_APP_SPECIFIC_PASSWORD$/i,
    /^APPLE_API_/i,
    /^API_KEY_/i
  ];
  const conflicting = Object.keys(environment).filter(name => (
    String(environment[name] || '').trim()
    && forbiddenPatterns.some(pattern => pattern.test(name))
  ));
  if (conflicting.length) {
    throw new Error(
      `Release environment contains conflicting build or Apple credential inputs: `
      + conflicting.sort().join(', ')
    );
  }
}

export function createReleaseChildEnvironment(environment) {
  const allowed = [
    'HOME', 'TMPDIR', 'LANG', 'LC_ALL', 'USER', 'LOGNAME', 'SHELL', 'SSH_AUTH_SOCK',
    'AUTOMIXER_APPLE_TEAM_ID', 'AUTOMIXER_DEVELOPER_ID_IDENTITY',
    'APPLE_KEYCHAIN_PROFILE', 'APPLE_KEYCHAIN'
  ];
  const child = Object.fromEntries(allowed
    .filter(name => String(environment[name] || '').trim())
    .map(name => [name, environment[name]]));
  child.PATH = [...new Set([
    path.dirname(process.execPath),
    '/opt/homebrew/bin',
    '/usr/local/bin',
    '/usr/bin',
    '/bin',
    '/usr/sbin',
    '/sbin'
  ])].join(':');
  child.CI = '1';
  child.PYTHONHASHSEED = '0';
  child.PIP_DISABLE_PIP_VERSION_CHECK = '1';
  child.PIP_NO_INPUT = '1';
  child.npm_config_update_notifier = 'false';
  return child;
}

export function acquireBuildLock(lockPath = path.join(
  frontendRoot, 'dist', '.automixer-macos-build.lock'
)) {
  fs.mkdirSync(path.dirname(lockPath), { recursive: true });
  const token = crypto.randomUUID();
  const payload = { schema: 'automixer.macos_build_lock.v1', pid: process.pid, token };
  try {
    const descriptor = fs.openSync(lockPath, 'wx', 0o600);
    try {
      fs.writeFileSync(descriptor, `${JSON.stringify(payload)}\n`);
    } finally {
      fs.closeSync(descriptor);
    }
  } catch (error) {
    if (error?.code !== 'EEXIST') throw error;
    let owner = {};
    try { owner = JSON.parse(fs.readFileSync(lockPath, 'utf8')); } catch {}
    throw new Error(
      `Another or stale macOS build lock exists at ${lockPath}`
      + (owner.pid ? ` (pid ${owner.pid})` : '')
    );
  }
  return () => {
    try {
      const current = JSON.parse(fs.readFileSync(lockPath, 'utf8'));
      if (current.token === token && current.pid === process.pid) fs.unlinkSync(lockPath);
    } catch (error) {
      if (error?.code !== 'ENOENT') throw error;
    }
  };
}

function parseArgs(argv) {
  let profile = '';
  for (let index = 0; index < argv.length; index += 1) {
    if (argv[index] === '--profile') profile = argv[++index] || '';
    else throw new Error(`Unknown argument: ${argv[index]}`);
  }
  return { profile: validateProfile(profile) };
}

function uniqueArtifact(outputDirectory, extension) {
  const matches = fs.readdirSync(outputDirectory)
    .filter(name => name.toLowerCase().endsWith(extension))
    .map(name => path.join(outputDirectory, name));
  if (matches.length !== 1) {
    throw new Error(`Expected exactly one ${extension} artifact, found ${matches.length}`);
  }
  const candidate = matches[0];
  const stat = fs.lstatSync(candidate);
  if (stat.isSymbolicLink() || !stat.isFile()) {
    throw new Error(`${extension} artifact must be a regular file, not a symlink`);
  }
  return resolveArtifactPath(
    outputDirectory,
    path.basename(candidate),
    `${extension} artifact`,
    { mustExist: true }
  );
}

function singleSmokeResult(smokeRoot) {
  const runDirectories = fs.readdirSync(smokeRoot, { withFileTypes: true })
    .filter(entry => entry.isDirectory())
    .map(entry => path.join(smokeRoot, entry.name));
  if (runDirectories.length !== 1) {
    throw new Error(`Expected exactly one packaged runtime smoke run, found ${runDirectories.length}`);
  }
  const resultPath = path.join(runDirectories[0], 'result.json');
  if (!fs.existsSync(resultPath)) throw new Error('Packaged runtime smoke result is missing');
  const result = readStableJsonFileNoFollow(
    resultPath,
    'Packaged runtime smoke result'
  ).value;
  if (result.pass !== true) throw new Error('Packaged runtime smoke did not pass');
  return resultPath;
}

function commandVersion(command, args, options = {}) {
  const result = run(command, args, { ...options, stdio: 'pipe' });
  return `${result.stdout || result.stderr}`.trim();
}

export function resolvePythonBinary(expectedVersion, environment = process.env) {
  const explicit = String(environment.AUTOMIXER_PYTHON_BIN || '').trim();
  if (explicit && !path.isAbsolute(explicit)) {
    throw new Error('AUTOMIXER_PYTHON_BIN must be an absolute path');
  }
  if (explicit && !fs.existsSync(explicit)) {
    throw new Error('AUTOMIXER_PYTHON_BIN does not exist');
  }
  const pathCandidates = String(environment.PATH || '')
    .split(path.delimiter)
    .filter(Boolean)
    .flatMap(directory => [path.join(directory, 'python3.14'), path.join(directory, 'python3')]);
  const candidates = [
    explicit,
    '/Library/Frameworks/Python.framework/Versions/3.14/bin/python3.14',
    ...pathCandidates
  ].filter(Boolean);
  const visited = new Set();
  for (const candidate of candidates) {
    if (!fs.existsSync(candidate)) continue;
    let resolved;
    try { resolved = fs.realpathSync(candidate); } catch { continue; }
    if (visited.has(resolved)) continue;
    visited.add(resolved);
    const probe = spawnSync(resolved, [
      '-c', 'import platform; print(platform.python_version())'
    ], { encoding: 'utf8', timeout: 5000 });
    if (!probe.error && probe.status === 0 && probe.stdout.trim() === expectedVersion) {
      return resolved;
    }
    if (explicit && resolved === fs.realpathSync(explicit)) {
      throw new Error(`AUTOMIXER_PYTHON_BIN is not CPython ${expectedVersion}`);
    }
  }
  throw new Error(`Release build requires an explicit native CPython ${expectedVersion}`);
}

function writeJson(artifactRoot, filePath, value, label) {
  writeNewArtifactFileNoFollow(
    artifactRoot,
    filePath,
    `${JSON.stringify(value, null, 2)}\n`,
    label
  );
}

export function writeLocalManifest({ outputDirectory, appPath, zipPath, source }) {
  const embeddedAttestationPath = path.join(
    appPath,
    'Contents',
    'Resources',
    'local-build-attestation.json'
  );
  if (!fs.existsSync(embeddedAttestationPath)) {
    throw new Error('Local package is missing its embedded non-release attestation');
  }
  const embeddedAttestation = readStableJsonFileNoFollow(
    embeddedAttestationPath,
    'Embedded local non-release attestation'
  ).value;
  if (
    embeddedAttestation.schema !== 'automixer.macos_local_build_attestation.v1'
    || embeddedAttestation.profile !== 'local'
    || embeddedAttestation.releaseEligible !== false
    || embeddedAttestation.source?.head !== source.head
    || embeddedAttestation.source?.tree !== source.tree
    || embeddedAttestation.source?.worktreeDirty !== Boolean(source.status)
    || embeddedAttestation.source?.statusSha256 !== sha256Text(source.status)
    || embeddedAttestation.source?.sourceContentSha256 !== source.sourceContentSha256
  ) {
    throw new Error('Embedded local non-release attestation does not match build source');
  }
  writeJson(outputDirectory, path.join(outputDirectory, 'build-profile.json'), {
    schema: 'automixer.macos_build_profile.v1',
    pathPolicy: 'artifact_root_relative_v1',
    generatedAt: new Date().toISOString(),
    profile: 'local',
    releaseEligible: false,
    reason: 'Local validation artifact; Developer ID, notarization and Gatekeeper acceptance are not asserted.',
    source: {
      head: source.head,
      tree: source.tree,
      worktreeDirty: Boolean(source.status),
      statusSha256: sha256Text(source.status),
      sourceContentSha256: source.sourceContentSha256
    },
    app: {
      path: artifactRelativePath(outputDirectory, appPath, 'Packaged app'),
      treeSha256: hashTree(appPath)
    },
    embeddedAttestation: artifactFileIdentity(
      outputDirectory,
      embeddedAttestationPath,
      'Embedded local build attestation'
    ),
    artifacts: {
      zip: artifactFileIdentity(outputDirectory, zipPath, 'Local ZIP artifact')
    }
  }, 'Local build profile');
}

function writeReleaseBuildProfile({ outputDirectory, appPath, source }) {
  const attestationPath = path.join(
    appPath,
    'Contents',
    'Resources',
    'release-build-attestation.json'
  );
  if (!fs.existsSync(attestationPath)) {
    throw new Error('Release package is missing its signed build attestation');
  }
  writeJson(outputDirectory, path.join(outputDirectory, 'build-profile.json'), {
    schema: 'automixer.macos_build_profile.v1',
    pathPolicy: 'artifact_root_relative_v1',
    generatedAt: new Date().toISOString(),
    profile: 'release',
    releaseEligible: false,
    reason: 'Release candidate pending runtime smoke, notarization, stapling, Gatekeeper and retained-artifact verification.',
    source: {
      head: source.head,
      tree: source.tree,
      worktreeDirty: false,
      statusSha256: sha256Text(source.status),
      sourceContentSha256: source.sourceContentSha256
    },
    app: {
      path: artifactRelativePath(outputDirectory, appPath, 'Packaged app'),
      treeSha256: hashTree(appPath)
    },
    embeddedAttestation: artifactFileIdentity(
      outputDirectory,
      attestationPath,
      'Embedded release build attestation'
    )
  }, 'Release build profile');
}

export function createReleaseBuildMetadata({
  outputDirectory,
  generatedAt = new Date().toISOString(),
  startedAt,
  source,
  packageLock,
  requirementsLock,
  buildRequirementsLock,
  pythonExecutable,
  pythonVersion,
  tools,
  backendServer,
  backendBundleTreeSha256,
  frontendTreeSha256
}) {
  return {
    schema: 'automixer.macos_release_build_metadata.v1',
    pathPolicy: 'artifact_root_relative_and_logical_v1',
    generatedAt,
    startedAt,
    source: { head: source.head, tree: source.tree, cleanCheckout: true },
    locks: {
      frontendPackageLock: logicalFileIdentity(
        packageLock,
        'frontend/package-lock.json',
        'source_checkout',
        'Frontend package lock'
      ),
      backendRequirementsLock: logicalFileIdentity(
        requirementsLock,
        'backend/requirements-macos-arm64.lock.txt',
        'source_checkout',
        'Backend runtime requirements lock'
      ),
      backendBuildRequirementsLock: logicalFileIdentity(
        buildRequirementsLock,
        'backend/requirements-macos-arm64-build.lock.txt',
        'source_checkout',
        'Backend build requirements lock'
      )
    },
    toolchainInputs: {
      pythonExecutable: logicalFileIdentity(
        pythonExecutable,
        `cpython/${pythonVersion}/python`,
        'toolchain_runtime',
        'Release Python executable'
      )
    },
    tools,
    buildOutputs: {
      backendServer: artifactFileIdentity(
        outputDirectory,
        backendServer,
        'Isolated backend server'
      ),
      backendBundleTreeSha256,
      frontendTreeSha256
    }
  };
}

function requirePackage() {
  return JSON.parse(fs.readFileSync(path.join(frontendRoot, 'package.json'), 'utf8'));
}

export function runReleaseVerificationLifecycle({
  stage,
  artifactDirectory,
  buildWorkspace,
  verifyInitial,
  verifyRetained,
  removeWorkspace = directory => fs.rmSync(directory, { recursive: true, force: true })
}) {
  const initialPath = path.join(artifactDirectory, 'release-verification-initial.json');
  const finalPath = path.join(artifactDirectory, 'release-manifest.json');
  let initial;
  let finalManifestIdentity = null;
  stage('isolated-output release verification', () => {
    verifyInitial();
    if (!fs.lstatSync(initialPath, { throwIfNoEntry: false })) {
      throw new Error('Initial isolated-output release verification was not produced');
    }
    initial = readStableJsonFileNoFollow(
      initialPath,
      'Initial isolated-output release verification'
    ).value;
    if (
      initial.schema !== 'automixer.macos_release_manifest.v2'
      || initial.pathPolicy !== 'artifact_root_relative_and_logical_v1'
      || initial.releaseEligible !== false
      || initial.verification?.mode !== 'isolated_outputs'
      || initial.verification?.complete !== false
      || initial.verification?.eligibleAfterRetainedVerification !== true
      || initial.source?.cleanCheckout !== true
    ) {
      throw new Error('Initial release verification is not a fail-closed intermediate proof');
    }
  });
  stage('release workspace cleanup', () => removeWorkspace(buildWorkspace));
  if (fs.existsSync(buildWorkspace)) {
    throw new Error('Release build workspace still exists after cleanup');
  }
  stage('retained-artifact release verification', () => {
    verifyRetained();
    if (!fs.lstatSync(finalPath, { throwIfNoEntry: false })) {
      throw new Error('Final retained-artifact release manifest was not produced');
    }
    const stableFinalManifest = readStableJsonFileNoFollow(
      finalPath,
      'Final retained-artifact release manifest'
    );
    const finalManifest = stableFinalManifest.value;
    finalManifestIdentity = {
      path: finalPath,
      ...stableFinalManifest.identity
    };
    if (
      finalManifest.schema !== 'automixer.macos_release_manifest.v2'
      || finalManifest.pathPolicy !== 'artifact_root_relative_and_logical_v1'
      || finalManifest.releaseEligible !== true
      || finalManifest.verification?.mode !== 'retained_artifacts'
      || finalManifest.verification?.complete !== true
      || finalManifest.source?.cleanCheckout !== true
      || finalManifest.source?.head !== initial.source?.head
      || finalManifest.source?.tree !== initial.source?.tree
      || finalManifest.build?.metadata?.sha256 !== initial.build?.metadata?.sha256
      || finalManifest.artifacts?.dmg?.sha256 !== initial.artifacts?.dmg?.sha256
      || finalManifest.artifacts?.zip?.sha256 !== initial.artifacts?.zip?.sha256
      || finalManifest.runtimeSmoke?.result?.sha256
        !== initial.runtimeSmoke?.result?.sha256
      || finalManifest.notarization?.app?.evidence?.sha256
        !== initial.notarization?.app?.evidence?.sha256
      || finalManifest.notarization?.app?.rawAppleLog?.sha256
        !== initial.notarization?.app?.rawAppleLog?.sha256
      || finalManifest.notarization?.dmg?.evidence?.sha256
        !== initial.notarization?.dmg?.evidence?.sha256
      || finalManifest.notarization?.dmg?.rawAppleLog?.sha256
        !== initial.notarization?.dmg?.rawAppleLog?.sha256
    ) {
      throw new Error('Final retained-artifact release manifest is not fully bound');
    }
  }, {
    onPostconditionFailure: () => {
      if (
        finalManifestIdentity
        && fs.lstatSync(finalPath, { throwIfNoEntry: false })
      ) {
        removeArtifactFileNoFollow(
          artifactDirectory,
          finalPath,
          'Invalid final release manifest cleanup',
          finalManifestIdentity
        );
      }
    }
  });
  return { initialPath, finalPath, finalManifestIdentity };
}

function main() {
  const { profile } = parseArgs(process.argv.slice(2));
  if (process.platform !== 'darwin' || process.arch !== 'arm64') {
    throw new Error('macOS packaging requires a native arm64 macOS runner');
  }

  const initialSource = captureSourceSnapshot();
  if (profile === 'release') validateReleaseEnvironment(process.env, initialSource.status);
  const releasePythonBinary = profile === 'release'
    ? resolvePythonBinary(releasePythonVersion, process.env)
    : null;
  const packageDefinition = requirePackage();
  if (profile === 'release') {
    const expectedNode = `v${packageDefinition.engines?.node || ''}`;
    const expectedNpm = packageDefinition.engines?.npm || '';
    const actualNpm = commandVersion('npm', ['--version']);
    if (process.version !== expectedNode || actualNpm !== expectedNpm) {
      throw new Error(
        `Release toolchain mismatch: required Node ${expectedNode} / npm ${expectedNpm}; `
        + `actual Node ${process.version} / npm ${actualNpm}`
      );
    }
  }

  const runId = `${timestamp()}_${crypto.randomUUID()}_${profile}`;
  let finalReleaseManifestIdentity = null;
  const profileOutputRoot = path.join(frontendRoot, 'dist', profile);
  fs.mkdirSync(profileOutputRoot, { recursive: true });
  const outputDirectory = path.join(frontendRoot, 'dist', profile, runId);
  try {
    fs.mkdirSync(outputDirectory);
  } catch (error) {
    if (error?.code === 'EEXIST') {
      throw new Error(`Refusing to reuse build output directory: ${outputDirectory}`);
    }
    throw error;
  }

  const buildWorkspace = path.join(outputDirectory, '.build');
  const backendWorkDirectory = path.join(buildWorkspace, 'pyinstaller-work');
  const backendDistDirectory = path.join(buildWorkspace, 'backend-dist');
  const frontendBuildDirectory = path.join(buildWorkspace, 'frontend-build');
  const backendBundlePath = path.join(backendDistDirectory, 'server');
  const backendServerPath = path.join(backendBundlePath, 'server');

  const requirementsLock = path.join(repoRoot, 'backend', 'requirements-macos-arm64.lock.txt');
  const buildRequirementsLock = path.join(
    repoRoot, 'backend', 'requirements-macos-arm64-build.lock.txt'
  );
  const packageLock = path.join(frontendRoot, 'package-lock.json');
  if (
    !fs.existsSync(requirementsLock)
    || !fs.existsSync(buildRequirementsLock)
    || !fs.existsSync(packageLock)
  ) {
    throw new Error('Release dependency locks are missing');
  }

  const backendVenv = profile === 'release'
    ? path.join(buildWorkspace, 'backend-venv')
    : path.join(repoRoot, 'backend', 'venv');
  const environment = {
    ...(profile === 'release' ? createReleaseChildEnvironment(process.env) : process.env),
    AUTOMIXER_MAC_BUILD_PROFILE: profile,
    AUTOMIXER_MAC_OUTPUT_DIR: outputDirectory,
    AUTOMIXER_TARGET_ARCH: 'arm64',
    AUTOMIXER_BACKEND_WORK_DIR: backendWorkDirectory,
    AUTOMIXER_BACKEND_DIST_DIR: backendDistDirectory,
    AUTOMIXER_BACKEND_BUNDLE_PATH: backendBundlePath,
    AUTOMIXER_BACKEND_SERVER_PATH: backendServerPath,
    AUTOMIXER_FRONTEND_BUILD_DIR: frontendBuildDirectory,
    AUTOMIXER_BUILD_SOURCE_HEAD: initialSource.head,
    AUTOMIXER_BUILD_SOURCE_TREE: initialSource.tree,
    AUTOMIXER_BUILD_SOURCE_WORKTREE_DIRTY: String(Boolean(initialSource.status)),
    AUTOMIXER_BUILD_SOURCE_STATUS_SHA256: sha256Text(initialSource.status),
    AUTOMIXER_BUILD_SOURCE_CONTENT_SHA256: initialSource.sourceContentSha256
  };
  if (profile === 'release') {
    Object.assign(environment, {
      AUTOMIXER_BACKEND_VENV_DIR: backendVenv,
      AUTOMIXER_BACKEND_REQUIREMENTS_FILE: requirementsLock,
      AUTOMIXER_BACKEND_BUILD_REQUIREMENTS_FILE: buildRequirementsLock,
      AUTOMIXER_PYTHON_BIN: releasePythonBinary,
      AUTOMIXER_PIP_VERSION: '25.3',
      AUTOMIXER_PYINSTALLER_VERSION: '6.20.0',
      AUTOMIXER_REQUIRED_PYTHON_VERSION: releasePythonVersion,
      AUTOMIXER_VERIFY_LOCKED_ENV: 'true'
    });
  }
  let npmEnvironment = environment;
  if (profile === 'release') {
    const npmHome = path.join(buildWorkspace, 'npm-home');
    const npmCache = path.join(buildWorkspace, 'npm-cache');
    const npmUserConfig = path.join(buildWorkspace, 'release.npmrc');
    fs.mkdirSync(npmHome, { recursive: true });
    fs.writeFileSync(npmUserConfig, [
      'registry=https://registry.npmjs.org/',
      'audit=false',
      'fund=false',
      'update-notifier=false',
      ''
    ].join('\n'), { mode: 0o600 });
    npmEnvironment = {
      ...environment,
      HOME: npmHome,
      npm_config_cache: npmCache,
      npm_config_userconfig: npmUserConfig,
      NPM_CONFIG_GLOBALCONFIG: '/dev/null'
    };
  }

  const stage = (name, action, stageOptions = {}) => runGuardedStage(
    name,
    action,
    { expectedSource: initialSource, ...stageOptions }
  );
  const startedAt = new Date().toISOString();

  if (profile === 'release') {
    stage('npm ci', () => run('npm', ['ci', '--no-audit', '--no-fund'], {
      env: npmEnvironment
    }));
  }
  stage('backend build', () => run('/bin/bash', [path.join(repoRoot, 'build_backend.sh')], {
    cwd: repoRoot,
    env: environment
  }));
  stage('frontend build', () => run('npm', ['run', 'build'], { env: environment }));

  const backendBuildIdentity = fileIdentity(backendServerPath);
  const backendBundleTreeSha256 = hashTree(backendBundlePath);
  const frontendBuildTreeSha256 = hashTree(frontendBuildDirectory);
  if (profile === 'release') {
    Object.assign(environment, {
      AUTOMIXER_RELEASE_SOURCE_HEAD: initialSource.head,
      AUTOMIXER_RELEASE_SOURCE_TREE: initialSource.tree,
      AUTOMIXER_RELEASE_BACKEND_SHA256: backendBuildIdentity.sha256,
      AUTOMIXER_RELEASE_BACKEND_BUNDLE_TREE_SHA256: backendBundleTreeSha256,
      AUTOMIXER_RELEASE_FRONTEND_TREE_SHA256: frontendBuildTreeSha256
    });
  }

  const electronBuilder = path.join(frontendRoot, 'node_modules', '.bin', 'electron-builder');
  if (!fs.existsSync(electronBuilder)) {
    throw new Error('Local electron-builder binary is missing; run npm ci');
  }
  stage('electron-builder packaging', () => run(electronBuilder, [
    '--config',
    `electron-builder.${profile}.cjs`,
    '--mac',
    '--arm64',
    '--publish',
    'never'
  ], { env: environment }));

  const productName = packageDefinition.build.productName;
  const appPath = path.join(outputDirectory, 'mac-arm64', `${productName}.app`);
  if (!fs.existsSync(appPath)) throw new Error(`Packaged app is missing: ${appPath}`);
  stage('packaged architecture verification', () => run(
    process.execPath,
    ['scripts/verify-packaged-architecture.mjs'],
    { env: { ...environment, AUTOMIXER_APP_PATH: appPath } }
  ));

  const zipPath = uniqueArtifact(outputDirectory, '.zip');
  if (profile === 'local') {
    writeLocalManifest({
      outputDirectory,
      appPath,
      zipPath,
      source: initialSource
    });
  } else {
    writeReleaseBuildProfile({
      outputDirectory,
      appPath,
      source: initialSource
    });
    const buildMetadataPath = path.join(outputDirectory, 'build-metadata.json');
    const buildMetadata = createReleaseBuildMetadata({
      outputDirectory,
      startedAt,
      source: initialSource,
      packageLock,
      requirementsLock,
      buildRequirementsLock,
      pythonExecutable: releasePythonBinary,
      pythonVersion: releasePythonVersion,
      tools: {
        node: process.version,
        npm: commandVersion('npm', ['--version']),
        python: commandVersion(path.join(backendVenv, 'bin', 'python'), ['--version']),
        pip: commandVersion(path.join(backendVenv, 'bin', 'python'), [
          '-c',
          "import importlib.metadata; print(importlib.metadata.version('pip'))"
        ]),
        pyinstaller: commandVersion(path.join(backendVenv, 'bin', 'python'), ['-m', 'PyInstaller', '--version']),
        electronBuilder: commandVersion(electronBuilder, ['--version']),
        electron: readJsonVersion(path.join(frontendRoot, 'node_modules', 'electron', 'package.json')),
        osxSign: readJsonVersion(path.join(frontendRoot, 'node_modules', '@electron', 'osx-sign', 'package.json')),
        macOS: commandVersion('/usr/bin/sw_vers', ['-productVersion']),
        xcode: commandVersion('/usr/bin/xcodebuild', ['-version']).replace(/\n/g, '; ')
      },
      backendServer: backendServerPath,
      backendBundleTreeSha256,
      frontendTreeSha256: frontendBuildTreeSha256
    });
    writeJson(
      outputDirectory,
      buildMetadataPath,
      buildMetadata,
      'Release build metadata'
    );
    assertSourceSnapshot(initialSource, captureSourceSnapshot(), 'before packaged runtime smoke');

    const smokeRoot = path.join(outputDirectory, 'runtime-smoke');
    stage('packaged runtime smoke', () => run(process.execPath, [
      'scripts/smoke-packaged-runtime.mjs',
      '--app', appPath,
      '--artifact-root', smokeRoot
    ], { env: environment }));
    const smokeResultPath = singleSmokeResult(smokeRoot);

    const dmgPath = uniqueArtifact(outputDirectory, '.dmg');
    stage('DMG notarization', () => run(process.execPath, [
      'scripts/notarize-dmg.mjs',
      '--artifact-root', outputDirectory,
      '--dmg', dmgPath
    ], { env: environment }));
    const verificationArgs = [
      'scripts/verify-macos-release.mjs',
      '--artifact-dir', outputDirectory,
      '--team-id', environment.AUTOMIXER_APPLE_TEAM_ID,
      '--not-before', startedAt,
      '--expected-head', initialSource.head,
      '--expected-tree', initialSource.tree,
      '--smoke-result', smokeResultPath,
      '--build-metadata', buildMetadataPath
    ];
    const releaseVerification = runReleaseVerificationLifecycle({
      stage,
      artifactDirectory: outputDirectory,
      buildWorkspace,
      verifyInitial: () => run(process.execPath, verificationArgs, { env: environment }),
      verifyRetained: () => run(
        process.execPath,
        [...verificationArgs, '--retained-artifacts'],
        { env: environment }
      )
    });
    finalReleaseManifestIdentity = releaseVerification.finalManifestIdentity;
  }

  if (profile === 'local') {
    fs.rmSync(buildWorkspace, { recursive: true, force: true });
  }

  try {
    assertSourceSnapshot(initialSource, captureSourceSnapshot(), `${profile} build completion`);
  } catch (error) {
    if (profile === 'release') {
      const finalManifestPath = path.join(outputDirectory, 'release-manifest.json');
      if (
        finalReleaseManifestIdentity
        && fs.lstatSync(finalManifestPath, { throwIfNoEntry: false })
      ) {
        removeArtifactFileNoFollow(
          outputDirectory,
          finalManifestPath,
          'Release completion drift manifest cleanup',
          finalReleaseManifestIdentity
        );
      }
    }
    throw error;
  }

  process.stdout.write(`${JSON.stringify({ profile, outputDirectory, appPath })}\n`);
}

function readJsonVersion(filePath) {
  const version = JSON.parse(fs.readFileSync(filePath, 'utf8')).version;
  if (!version) throw new Error(`Package version is missing: ${filePath}`);
  return version;
}

if (process.argv[1] && path.resolve(process.argv[1]) === scriptPath) {
  let releaseBuildLock;
  try {
    releaseBuildLock = acquireBuildLock();
    main();
  } catch (error) {
    process.stderr.write(`${error?.message || error}\n`);
    process.exitCode = 1;
  } finally {
    try {
      releaseBuildLock?.();
    } catch (error) {
      process.stderr.write(`Failed to release macOS build lock: ${error?.message || error}\n`);
      process.exitCode = 1;
    }
  }
}
