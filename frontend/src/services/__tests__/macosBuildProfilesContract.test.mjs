import test from 'node:test';
import assert from 'node:assert/strict';
import crypto from 'node:crypto';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { spawnSync } from 'node:child_process';
import { createRequire } from 'node:module';
import { fileURLToPath, pathToFileURL } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const frontendRoot = path.resolve(__dirname, '../../..');
const buildScriptPath = path.join(frontendRoot, 'scripts', 'build-macos.mjs');
const smokeScriptPath = path.join(frontendRoot, 'scripts', 'smoke-packaged-runtime.mjs');
const smokeVerifierPath = path.join(
  frontendRoot,
  'scripts',
  'verify-packaged-runtime-smoke.mjs'
);
const releaseEvidencePath = path.join(
  frontendRoot,
  'scripts',
  'release-evidence.mjs'
);
const require = createRequire(import.meta.url);
const { applyMacAtsPolicy } = require(path.join(frontendRoot, 'scripts', 'after-pack-macos.cjs'));
const {
  acquireBuildLock,
  assertSourceSnapshot,
  captureSourceSnapshot,
  createReleaseChildEnvironment,
  createReleaseBuildMetadata,
  resolvePythonBinary,
  runGuardedStage,
  runReleaseVerificationLifecycle,
  validateProfile,
  validateReleaseEnvironment,
  writeLocalManifest
} = await import(pathToFileURL(buildScriptPath));
const {
  loadBuildProfileBinding,
  verifyPackagedRuntimeSmokeManifest
} = await import(pathToFileURL(smokeScriptPath));
const {
  resolveBuildProfileAppPath
} = await import(pathToFileURL(releaseEvidencePath));

test('macOS build profiles are explicit and use non-overlapping unique outputs', () => {
  const packageJson = JSON.parse(fs.readFileSync(path.join(frontendRoot, 'package.json'), 'utf8'));
  const buildSource = fs.readFileSync(buildScriptPath, 'utf8');
  const smokeSource = fs.readFileSync(smokeScriptPath, 'utf8');
  const releaseVerifierSource = fs.readFileSync(
    path.join(frontendRoot, 'scripts', 'verify-macos-release.mjs'),
    'utf8'
  );
  const releaseEvidenceSource = fs.readFileSync(releaseEvidencePath, 'utf8');
  const notarizeSource = fs.readFileSync(
    path.join(frontendRoot, 'scripts', 'notarize-dmg.mjs'),
    'utf8'
  );
  const localConfig = fs.readFileSync(path.join(frontendRoot, 'electron-builder.local.cjs'), 'utf8');
  const releaseConfig = fs.readFileSync(path.join(frontendRoot, 'electron-builder.release.cjs'), 'utf8');
  const backendSpec = fs.readFileSync(path.join(frontendRoot, '..', 'backend', 'server.spec'), 'utf8');
  const backendBuild = fs.readFileSync(path.join(frontendRoot, '..', 'build_backend.sh'), 'utf8');

  assert.equal(packageJson.scripts['mac:local'], 'node scripts/build-macos.mjs --profile local');
  assert.match(smokeSource, /size: result\.buildProfile\?\.size/);
  assert.equal(Object.hasOwn(packageJson.scripts, 'mac:release'), false);
  for (const source of [buildSource, smokeSource, releaseVerifierSource]) {
    assert.match(source, /runReleaseEvidenceGit/);
    assert.doesNotMatch(source, /run\(['"]\/usr\/bin\/git['"]|spawnSync\(['"]\/usr\/bin\/git['"]/);
  }
  assert.match(releaseEvidenceSource, /GIT_CONFIG_NOSYSTEM: '1'/);
  assert.match(releaseEvidenceSource, /core\.attributesFile=\/dev\/null/);
  assert.match(releaseEvidenceSource, /core\.filemode=true/);
  assert.match(releaseEvidenceSource, /Repository Git configuration is unsafe for evidence reads/);
  assert.equal(
    fs.existsSync(path.join(frontendRoot, 'scripts', 'run-macos-release.mjs')),
    true
  );
  assert.equal(
    fs.existsSync(path.join(frontendRoot, 'scripts', 'run-macos-release')),
    true
  );
  assert.equal(packageJson.build.afterPack, 'scripts/after-pack-macos.cjs');
  assert.equal(packageJson.devDependencies.electron, '43.1.0');
  assert.equal(packageJson.devDependencies['electron-builder'], '24.13.3');
  assert.equal(
    packageJson.build.extraResources.find(entry => entry.to === 'backend')?.from,
    '../backend/dist/server'
  );
  assert.match(backendSpec, /exclude_binaries=True/);
  assert.match(backendSpec, /coll = COLLECT\(/);
  assert.match(backendSpec, /PACKAGED_RUNTIME_SOURCE_MANIFEST_FILENAME/);
  assert.match(backendSpec, /datas=\[\(str\(runtime_manifest_path\), "\."\)\]/);
  assert.match(backendBuild, /BACKEND_BUNDLE="\$DIST_DIR\/server"/);
  assert.match(backendBuild, /BACKEND_BINARY="\$BACKEND_BUNDLE\/server"/);
  assert.match(backendBuild, /automixer-runtime-source-manifest\.json/);
  assert.match(buildSource, /dist', profile, runId/);
  assert.match(buildSource, /crypto\.randomUUID\(\)/);
  assert.match(buildSource, /fs\.mkdirSync\(outputDirectory\)/);
  assert.match(buildSource, /Refusing to reuse build output directory/);
  assert.match(buildSource, /releaseEligible: false/);
  assert.match(buildSource, /AUTOMIXER_PIP_VERSION: '25\.3'/);
  assert.match(buildSource, /registry=https:\/\/registry\.npmjs\.org\//);
  assert.match(buildSource, /npm_config_userconfig: npmUserConfig/);
  assert.match(buildSource, /NPM_CONFIG_GLOBALCONFIG: '\/dev\/null'/);
  assert.match(buildSource, /isolated-output release verification/);
  assert.match(buildSource, /retained-artifact release verification/);
  assert.match(buildSource, /--retained-artifacts/);
  assert.match(buildSource, /artifact_root_relative_and_logical_v1/);
  assert.match(releaseVerifierSource, /artifact_root_relative_and_logical_v1/);
  assert.match(releaseVerifierSource, /artifactRelativePath\(artifactDirectory, appPath/);
  assert.doesNotMatch(releaseVerifierSource, /app:\s*\{\s*path:\s*appPath/);
  assert.match(notarizeSource, /pathPolicy: 'artifact_root_relative_v1'/);
  assert.doesNotMatch(notarizeSource, /path:\s*dmgPath/);
  assert.match(notarizeSource, /resolveNotarizationDmg/);
  assert.match(buildSource, /--artifact-root', outputDirectory/);
  assert.ok(
    buildSource.indexOf("stage('release workspace cleanup'")
      < buildSource.indexOf("stage('retained-artifact release verification'")
  );
  assert.match(localConfig, /notarize: false/);
  assert.match(localConfig, /LOCAL-NOT-FOR-DISTRIBUTION/);
  assert.match(localConfig, /AUTOMIXER_BACKEND_BUNDLE_PATH/);
  assert.match(localConfig, /files: \[\s*'package\.json'/);
  assert.match(releaseConfig, /forceCodeSigning: true/);
  assert.match(releaseConfig, /AUTOMIXER_BACKEND_BUNDLE_PATH/);
  assert.match(releaseConfig, /files: \[\s*'package\.json'/);
  assert.match(releaseConfig, /type: 'distribution'/);
  assert.match(releaseConfig, /Developer ID Application:/);
  assert.match(releaseConfig, /timestamp\.apple\.com/);
  assert.match(releaseConfig, /notarize: false/);
  assert.match(releaseConfig, /afterSign: 'scripts\/after-sign-macos\.cjs'/);
  assert.doesNotMatch(releaseConfig, /staple-notarized-app/);
  assert.match(releaseConfig, /sign: true/);
  assert.match(releaseConfig, /writeUpdateInfo: false/);
});

test('clean macOS CI launches the package and retains detached no-WING evidence', () => {
  const workflow = fs.readFileSync(
    path.join(frontendRoot, '..', '.github', 'workflows', 'test.yml'),
    'utf8'
  );

  assert.match(workflow, /smoke-packaged-runtime\.mjs/);
  assert.match(workflow, /--build-profile "\$AUTOMIXER_CI_BUILD_PROFILE"/);
  assert.match(workflow, /--artifact-root "\$RUNNER_TEMP\/automixer-packaged-runtime-smoke"/);
  assert.match(workflow, /tools\/no_wing_release_check\.py/);
  assert.match(workflow, /--quick/);
  assert.match(workflow, /tools\/verify_no_wing_artifact_manifest\.py/);
  assert.match(workflow, /actions\/upload-artifact@v4/);
  assert.match(workflow, /artifact-manifest-verifier\.json/);
  assert.match(workflow, /automixer-locked-macos-pip-freeze\.txt/);
  assert.match(workflow, /resolve-build-profile-app\.mjs "\$PROFILE"/);
  assert.match(workflow, /resolveBuildProfileAppPath\(profilePath\)/);
  assert.doesNotMatch(workflow, /existsSync\(manifest\.app\.path\)/);
});

test('macOS afterPack hook overrides builder ATS mutation and verifies readback', () => {
  const calls = [];
  const context = {
    electronPlatformName: 'darwin',
    appOutDir: '/tmp/automixer-mac-arm64',
    packager: { appInfo: { productFilename: 'AUTO MIXER Tubeslave' } }
  };
  const spawn = (_command, args) => {
    calls.push(args);
    if (args[0] === '-convert') {
      return {
        status: 0,
        stdout: JSON.stringify({
          NSAppTransportSecurity: {
            NSAllowsArbitraryLoads: false,
            NSAllowsLocalNetworking: true
          }
        }),
        stderr: ''
      };
    }
    return { status: 0, stdout: '', stderr: '' };
  };

  const result = applyMacAtsPolicy(context, { spawn });

  assert.equal(result.NSAllowsArbitraryLoads, false);
  assert.equal(result.NSAllowsLocalNetworking, true);
  assert.deepEqual(calls[0].slice(0, 4), [
    '-replace',
    'NSAppTransportSecurity.NSAllowsArbitraryLoads',
    '-bool',
    'NO'
  ]);
  assert.deepEqual(calls[1].slice(0, 4), ['-convert', 'json', '-o', '-']);
  assert.throws(
    () => applyMacAtsPolicy(context, {
      spawn: (_command, args) => args[0] === '-convert'
        ? {
            status: 0,
            stdout: JSON.stringify({
              NSAppTransportSecurity: {
                NSAllowsArbitraryLoads: true,
                NSAllowsLocalNetworking: true
              }
            }),
            stderr: ''
          }
        : { status: 0, stdout: '', stderr: '' }
    }),
    /ATS readback/
  );
  assert.deepEqual(
    applyMacAtsPolicy({ electronPlatformName: 'linux' }),
    { skipped: true, reason: 'non_macos_package' }
  );
});

test('local afterPack embeds a sealed non-release attestation', () => {
  const tempRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'automixer-local-attestation-'));
  const appOutDir = path.join(tempRoot, 'mac-arm64');
  const contents = path.join(appOutDir, 'AUTO MIXER Tubeslave.app', 'Contents');
  fs.mkdirSync(path.join(contents, 'Resources'), { recursive: true });
  const plist = {
    NSAppTransportSecurity: {
      NSAllowsArbitraryLoads: false,
      NSAllowsLocalNetworking: true
    }
  };
  try {
    const result = applyMacAtsPolicy({
      electronPlatformName: 'darwin',
      appOutDir,
      packager: { appInfo: { productFilename: 'AUTO MIXER Tubeslave' } }
    }, {
      spawn: (_command, args) => ({
        status: 0,
        stdout: args[0] === '-convert' ? JSON.stringify(plist) : '',
        stderr: ''
      }),
      environment: {
        AUTOMIXER_MAC_BUILD_PROFILE: 'local',
        AUTOMIXER_BUILD_SOURCE_HEAD: 'a'.repeat(40),
        AUTOMIXER_BUILD_SOURCE_TREE: 'b'.repeat(40),
        AUTOMIXER_BUILD_SOURCE_STATUS_SHA256: 'c'.repeat(64),
        AUTOMIXER_BUILD_SOURCE_CONTENT_SHA256: 'd'.repeat(64),
        AUTOMIXER_BUILD_SOURCE_WORKTREE_DIRTY: 'true'
      }
    });
    const embeddedPath = result.localBuildAttestation.path;
    const embedded = JSON.parse(fs.readFileSync(embeddedPath, 'utf8'));
    assert.equal(embedded.profile, 'local');
    assert.equal(embedded.releaseEligible, false);
    assert.match(embedded.reason, /NOT FOR DISTRIBUTION/);
    assert.equal(embedded.source.worktreeDirty, true);
    assert.equal(embedded.source.statusSha256, 'c'.repeat(64));
    assert.equal(embedded.source.sourceContentSha256, 'd'.repeat(64));
    assert.equal(embeddedPath, path.join(contents, 'Resources', 'local-build-attestation.json'));
  } finally {
    fs.rmSync(tempRoot, { recursive: true, force: true });
  }
});

test('packaged smoke binds local releaseEligible false to the exact app', () => {
  const tempRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'automixer-smoke-profile-'));
  const artifactRoot = path.join(tempRoot, 'original');
  const appPath = path.join(artifactRoot, 'mac-arm64', 'AUTO MIXER Tubeslave.app');
  const embeddedPath = path.join(
    appPath,
    'Contents',
    'Resources',
    'local-build-attestation.json'
  );
  const source = {
    head: 'a'.repeat(40),
    tree: 'b'.repeat(40),
    worktreeDirty: true,
    statusSha256: 'c'.repeat(64)
  };
  fs.mkdirSync(path.dirname(embeddedPath), { recursive: true });
  fs.writeFileSync(embeddedPath, JSON.stringify({
    schema: 'automixer.macos_local_build_attestation.v1',
    profile: 'local',
    releaseEligible: false,
    source
  }));
  const profilePath = path.join(artifactRoot, 'build-profile.json');
  fs.writeFileSync(profilePath, JSON.stringify({
    schema: 'automixer.macos_build_profile.v1',
    pathPolicy: 'artifact_root_relative_v1',
    profile: 'local',
    releaseEligible: false,
    reason: 'local only',
    source,
    app: {
      path: 'mac-arm64/AUTO MIXER Tubeslave.app',
      treeSha256: 'd'.repeat(64)
    }
  }));
  try {
    const binding = loadBuildProfileBinding(appPath, profilePath);
    assert.equal(binding.profile, 'local');
    assert.equal(binding.releaseEligible, false);
    assert.equal(binding.bindingValid, true);
    assert.equal(binding.path, 'build-profile.json');
    assert.equal(
      binding.embeddedAttestationPath,
      'mac-arm64/AUTO MIXER Tubeslave.app/Contents/Resources/local-build-attestation.json'
    );

    const relocatedRoot = path.join(tempRoot, 'relocated');
    fs.cpSync(artifactRoot, relocatedRoot, { recursive: true });
    const relocatedApp = path.join(
      relocatedRoot,
      'mac-arm64',
      'AUTO MIXER Tubeslave.app'
    );
    assert.equal(loadBuildProfileBinding(
      relocatedApp,
      path.join(relocatedRoot, 'build-profile.json')
    ).bindingValid, true);

    const legacyProfile = JSON.parse(fs.readFileSync(profilePath, 'utf8'));
    delete legacyProfile.pathPolicy;
    legacyProfile.app.path = appPath;
    fs.writeFileSync(profilePath, JSON.stringify(legacyProfile));
    assert.equal(loadBuildProfileBinding(appPath, profilePath).bindingValid, false);
    const legacyBinding = loadBuildProfileBinding(
      appPath,
      profilePath,
      { allowLegacy: true }
    );
    assert.equal(legacyBinding.bindingValid, true);
    assert.equal(
      legacyBinding.declaredApp.path,
      'mac-arm64/AUTO MIXER Tubeslave.app'
    );

    fs.writeFileSync(embeddedPath, JSON.stringify({
      schema: 'automixer.macos_local_build_attestation.v1',
      profile: 'local',
      releaseEligible: true,
      source
    }));
    assert.equal(loadBuildProfileBinding(
      appPath,
      profilePath,
      { allowLegacy: true }
    ).bindingValid, false);

    const escaped = JSON.parse(fs.readFileSync(profilePath, 'utf8'));
    escaped.app.path = '../outside.app';
    fs.writeFileSync(profilePath, JSON.stringify(escaped));
    assert.equal(loadBuildProfileBinding(
      appPath,
      profilePath,
      { allowLegacy: true }
    ).bindingValid, false);
  } finally {
    fs.rmSync(tempRoot, { recursive: true, force: true });
  }
});

test('local packaged build profile records only artifact-root-relative paths', () => {
  const outputDirectory = fs.mkdtempSync(path.join(os.tmpdir(), 'automixer-build-profile-'));
  const appPath = path.join(
    outputDirectory,
    'mac-arm64',
    'AUTO MIXER Tubeslave.app'
  );
  const embeddedPath = path.join(
    appPath,
    'Contents',
    'Resources',
    'local-build-attestation.json'
  );
  const zipPath = path.join(outputDirectory, 'AUTO MIXER Tubeslave.zip');
  const source = {
    head: 'a'.repeat(40),
    tree: 'b'.repeat(40),
    status: ' M frontend/example.js\n',
    sourceContentSha256: 'd'.repeat(64)
  };
  try {
    fs.mkdirSync(path.dirname(embeddedPath), { recursive: true });
    fs.writeFileSync(embeddedPath, JSON.stringify({
      schema: 'automixer.macos_local_build_attestation.v1',
      profile: 'local',
      releaseEligible: false,
      source: {
        head: source.head,
        tree: source.tree,
        worktreeDirty: true,
        statusSha256: crypto.createHash('sha256').update(source.status).digest('hex'),
        sourceContentSha256: source.sourceContentSha256
      }
    }));
    fs.writeFileSync(zipPath, 'zip bytes');
    writeLocalManifest({ outputDirectory, appPath, zipPath, source });
    const profile = JSON.parse(fs.readFileSync(
      path.join(outputDirectory, 'build-profile.json'),
      'utf8'
    ));
    assert.equal(profile.pathPolicy, 'artifact_root_relative_v1');
    assert.equal(profile.app.path, 'mac-arm64/AUTO MIXER Tubeslave.app');
    assert.equal(
      profile.embeddedAttestation.path,
      'mac-arm64/AUTO MIXER Tubeslave.app/Contents/Resources/local-build-attestation.json'
    );
    assert.equal(profile.artifacts.zip.path, 'AUTO MIXER Tubeslave.zip');
    const profilePath = path.join(outputDirectory, 'build-profile.json');
    const resolvedProfile = resolveBuildProfileAppPath(profilePath);
    const profileBytes = fs.readFileSync(profilePath);
    assert.equal(resolvedProfile.appPath, appPath);
    assert.deepEqual(resolvedProfile.profileIdentity, {
      size: profileBytes.length,
      sha256: crypto.createHash('sha256').update(profileBytes).digest('hex')
    });
    assert.doesNotMatch(JSON.stringify(profile), new RegExp(outputDirectory));
  } finally {
    fs.rmSync(outputDirectory, { recursive: true, force: true });
  }
});

test('build profile resolver rejects a symlink instead of following external JSON', () => {
  const tempRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'automixer-profile-symlink-'));
  const artifactRoot = path.join(tempRoot, 'artifact');
  const outsideProfile = path.join(tempRoot, 'outside-profile.json');
  const profilePath = path.join(artifactRoot, 'build-profile.json');
  try {
    fs.mkdirSync(artifactRoot);
    fs.writeFileSync(outsideProfile, JSON.stringify({
      schema: 'automixer.macos_build_profile.v1',
      pathPolicy: 'artifact_root_relative_v1',
      app: { path: 'app' }
    }));
    fs.symlinkSync(outsideProfile, profilePath);

    assert.throws(
      () => resolveBuildProfileAppPath(profilePath),
      /ELOOP|regular file/
    );
  } finally {
    fs.rmSync(tempRoot, { recursive: true, force: true });
  }
});

test('artifact writer helper rejects a swapped parent without writing outside root', () => {
  const tempRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'automixer-parent-swap-'));
  const artifactRoot = path.join(tempRoot, 'artifact');
  const originalParent = path.join(artifactRoot, 'evidence');
  const pinnedParent = path.join(artifactRoot, 'evidence-pinned');
  const outside = path.join(tempRoot, 'outside');
  const helper = path.join(frontendRoot, 'scripts', 'write-new-artifact-helper.mjs');
  try {
    fs.mkdirSync(originalParent, { recursive: true });
    fs.mkdirSync(outside);
    const rootStat = fs.statSync(artifactRoot, { bigint: true });
    const parentStat = fs.statSync(originalParent, { bigint: true });
    fs.renameSync(originalParent, pinnedParent);
    fs.symlinkSync(outside, originalParent);

    const completed = spawnSync(process.execPath, [
      helper,
      'release-manifest.json',
      artifactRoot,
      originalParent,
      rootStat.dev.toString(),
      rootStat.ino.toString(),
      parentStat.dev.toString(),
      parentStat.ino.toString(),
      'Parent-swap test'
    ], {
      cwd: pinnedParent,
      input: '{"releaseEligible":true}\n',
      encoding: 'utf8'
    });

    assert.notEqual(completed.status, 0);
    assert.equal(fs.existsSync(path.join(outside, 'release-manifest.json')), false);
    assert.equal(fs.existsSync(path.join(pinnedParent, 'release-manifest.json')), false);
  } finally {
    fs.rmSync(tempRoot, { recursive: true, force: true });
  }
});

test('release build metadata uses artifact-relative and logical identities only', () => {
  const tempRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'automixer-build-metadata-'));
  const outputDirectory = path.join(tempRoot, 'artifact');
  const paths = {
    packageLock: path.join(tempRoot, 'source', 'frontend', 'package-lock.json'),
    requirementsLock: path.join(tempRoot, 'source', 'backend', 'runtime.lock'),
    buildRequirementsLock: path.join(tempRoot, 'source', 'backend', 'build.lock'),
    pythonExecutable: path.join(tempRoot, 'toolchain', 'python'),
    backendServer: path.join(
      outputDirectory,
      '.build',
      'backend-dist',
      'server',
      'server'
    )
  };
  try {
    for (const filePath of Object.values(paths)) {
      fs.mkdirSync(path.dirname(filePath), { recursive: true });
      fs.writeFileSync(filePath, path.basename(filePath));
    }
    const tools = Object.fromEntries([
      'node', 'npm', 'python', 'pyinstaller', 'electronBuilder', 'electron',
      'osxSign', 'macOS', 'xcode'
    ].map(name => [name, `${name}-version`]));
    tools.pip = '25.3';
    const metadata = createReleaseBuildMetadata({
      outputDirectory,
      generatedAt: '2026-07-12T00:00:00.000Z',
      startedAt: '2026-07-12T00:00:00.000Z',
      source: { head: 'a'.repeat(40), tree: 'b'.repeat(40) },
      ...paths,
      pythonVersion: '3.14.3',
      tools,
      backendBundleTreeSha256: 'c'.repeat(64),
      frontendTreeSha256: 'd'.repeat(64)
    });
    assert.equal(metadata.pathPolicy, 'artifact_root_relative_and_logical_v1');
    assert.deepEqual(
      [metadata.locks.frontendPackageLock.path, metadata.locks.frontendPackageLock.pathBase],
      ['frontend/package-lock.json', 'source_checkout']
    );
    assert.deepEqual(
      [metadata.toolchainInputs.pythonExecutable.path,
        metadata.toolchainInputs.pythonExecutable.pathBase],
      ['cpython/3.14.3/python', 'toolchain_runtime']
    );
    assert.equal(
      metadata.buildOutputs.backendServer.path,
      '.build/backend-dist/server/server'
    );
    assert.doesNotMatch(JSON.stringify(metadata), new RegExp(tempRoot));
  } finally {
    fs.rmSync(tempRoot, { recursive: true, force: true });
  }
});

test('detached packaged smoke manifest survives relocation and rejects path escape', () => {
  const tempRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'automixer-smoke-manifest-'));
  const runRoot = path.join(tempRoot, 'original');
  const relocatedRoot = path.join(tempRoot, 'relocated');
  const digest = filePath => crypto.createHash('sha256')
    .update(fs.readFileSync(filePath))
    .digest('hex');
  const identity = (root, name) => {
    const filePath = path.join(root, name);
    return { path: name, size: fs.statSync(filePath).size, sha256: digest(filePath) };
  };
  try {
    fs.mkdirSync(runRoot);
    const result = {
      schema: 'automixer.packaged_runtime_smoke.v1',
      pathPolicy: 'packaged_build_artifact_root_relative_v1',
      pass: true,
      readOnly: true,
      contactsWing: false,
      liveWrites: false,
      liveAuthorityGranted: false,
      buildProfile: {
        profile: 'local',
        releaseEligible: false,
        path: 'build-profile.json',
        size: 512,
        sha256: 'a'.repeat(64),
        bindingValid: true,
        appTreeMatches: true
      }
    };
    fs.writeFileSync(path.join(runRoot, 'result.json'), JSON.stringify(result));
    fs.writeFileSync(path.join(runRoot, 'result.md'), '# smoke\n');
    const manifestPath = path.join(runRoot, 'artifact-manifest.json');
    const manifest = {
      schema: 'automixer.packaged_runtime_smoke_manifest.v1',
      pathPolicy: 'manifest_directory_relative_v1',
      readOnly: true,
      contactsWing: false,
      liveAuthorityGranted: false,
      buildProfile: {
        ...result.buildProfile,
        pathBase: 'packaged_build_artifact_root'
      },
      files: {
        'result.json': identity(runRoot, 'result.json'),
        'result.md': identity(runRoot, 'result.md')
      }
    };
    fs.writeFileSync(manifestPath, JSON.stringify(manifest));
    assert.doesNotThrow(() => verifyPackagedRuntimeSmokeManifest(manifestPath));

    fs.cpSync(runRoot, relocatedRoot, { recursive: true });
    const relocatedManifest = path.join(relocatedRoot, 'artifact-manifest.json');
    assert.doesNotThrow(() => verifyPackagedRuntimeSmokeManifest(relocatedManifest));
    const detachedCli = spawnSync(
      process.execPath,
      [smokeVerifierPath, '--manifest', relocatedManifest],
      { encoding: 'utf8' }
    );
    assert.equal(detachedCli.status, 0, detachedCli.stderr);
    assert.equal(JSON.parse(detachedCli.stdout).status, 'passed');

    const missingProfileSize = structuredClone(manifest);
    delete missingProfileSize.buildProfile.size;
    fs.writeFileSync(relocatedManifest, JSON.stringify(missingProfileSize));
    assert.throws(
      () => verifyPackagedRuntimeSmokeManifest(relocatedManifest),
      /manifest contract mismatch/
    );
    const mismatchedProfileSize = structuredClone(manifest);
    mismatchedProfileSize.buildProfile.size += 1;
    fs.writeFileSync(relocatedManifest, JSON.stringify(mismatchedProfileSize));
    assert.throws(
      () => verifyPackagedRuntimeSmokeManifest(relocatedManifest),
      /manifest\/result binding mismatch/
    );
    fs.writeFileSync(relocatedManifest, JSON.stringify(manifest));

    const legacyRoot = path.join(tempRoot, 'legacy');
    fs.cpSync(runRoot, legacyRoot, { recursive: true });
    const legacyManifestPath = path.join(legacyRoot, 'artifact-manifest.json');
    const legacyManifest = JSON.parse(fs.readFileSync(legacyManifestPath, 'utf8'));
    delete legacyManifest.pathPolicy;
    legacyManifest.files['result.json'].path = path.join(legacyRoot, 'result.json');
    legacyManifest.files['result.md'].path = path.join(legacyRoot, 'result.md');
    fs.writeFileSync(legacyManifestPath, JSON.stringify(legacyManifest));
    assert.throws(
      () => verifyPackagedRuntimeSmokeManifest(legacyManifestPath),
      /manifest contract mismatch/
    );
    assert.doesNotThrow(() => verifyPackagedRuntimeSmokeManifest(
      legacyManifestPath,
      { allowLegacy: true }
    ));
    const strictLegacyCli = spawnSync(
      process.execPath,
      [smokeVerifierPath, '--manifest', legacyManifestPath],
      { encoding: 'utf8' }
    );
    assert.notEqual(strictLegacyCli.status, 0);
    const explicitLegacyCli = spawnSync(
      process.execPath,
      [smokeVerifierPath, '--manifest', legacyManifestPath, '--allow-legacy'],
      { encoding: 'utf8' }
    );
    assert.equal(explicitLegacyCli.status, 0, explicitLegacyCli.stderr);
    assert.equal(
      JSON.parse(explicitLegacyCli.stdout).legacyCompatibilityEnabled,
      true
    );
    const relocatedLegacyRoot = path.join(tempRoot, 'relocated-legacy');
    fs.cpSync(legacyRoot, relocatedLegacyRoot, { recursive: true });
    assert.throws(
      () => verifyPackagedRuntimeSmokeManifest(
        path.join(relocatedLegacyRoot, 'artifact-manifest.json'),
        { allowLegacy: true }
      ),
      /escapes the artifact root/
    );

    const escaped = JSON.parse(fs.readFileSync(relocatedManifest, 'utf8'));
    escaped.files['result.json'].path = '../result.json';
    fs.writeFileSync(relocatedManifest, JSON.stringify(escaped));
    assert.throws(
      () => verifyPackagedRuntimeSmokeManifest(relocatedManifest),
      /escapes the artifact root/
    );

    escaped.files['result.json'].path = path.join(tempRoot, 'outside.json');
    fs.writeFileSync(path.join(tempRoot, 'outside.json'), 'outside');
    fs.writeFileSync(relocatedManifest, JSON.stringify(escaped));
    assert.throws(
      () => verifyPackagedRuntimeSmokeManifest(relocatedManifest),
      /must be artifact-root-relative/
    );

    fs.writeFileSync(relocatedManifest, JSON.stringify(manifest));
    fs.rmSync(path.join(relocatedRoot, 'result.json'));
    fs.symlinkSync(path.join(tempRoot, 'outside.json'), path.join(relocatedRoot, 'result.json'));
    assert.throws(
      () => verifyPackagedRuntimeSmokeManifest(relocatedManifest),
      /resolves outside the artifact root/
    );
  } finally {
    fs.rmSync(tempRoot, { recursive: true, force: true });
  }
});

test('release stage guard rejects source HEAD, tree, and dirty-status drift', () => {
  const expected = { head: 'a'.repeat(40), tree: 'b'.repeat(40), status: '' };
  assert.doesNotThrow(() => assertSourceSnapshot(expected, { ...expected }, 'test'));
  assert.throws(
    () => assertSourceSnapshot(expected, { ...expected, head: 'c'.repeat(40) }, 'test'),
    /HEAD changed/
  );
  assert.throws(
    () => assertSourceSnapshot(expected, { ...expected, tree: 'd'.repeat(40) }, 'test'),
    /tree changed/
  );
  assert.throws(
    () => assertSourceSnapshot(expected, { ...expected, status: ' M source' }, 'test'),
    /became dirty/
  );

  let reads = 0;
  assert.throws(() => runGuardedStage('mutating stage', () => {}, {
    expectedSource: expected,
    readSource: () => (++reads === 1 ? expected : { ...expected, status: ' M source' })
  }), /after mutating stage/);
});

test('authoritative source snapshot ignores hostile inherited Git config', () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'automixer-release-git-home-'));
  const home = path.join(root, 'home');
  const hookPath = path.join(root, 'fsmonitor-hook');
  const markerPath = path.join(root, 'hook-executed');
  const previousHome = process.env.HOME;
  fs.mkdirSync(home);
  try {
    fs.writeFileSync(
      hookPath,
      `#!/bin/sh\n/usr/bin/touch ${JSON.stringify(markerPath)}\nexit 0\n`
    );
    fs.chmodSync(hookPath, 0o755);
    fs.writeFileSync(
      path.join(home, '.gitconfig'),
      `[core]\n\tfsmonitor = ${hookPath}\n`
    );
    process.env.HOME = home;
    const snapshot = captureSourceSnapshot();
    assert.match(snapshot.head, /^[0-9a-f]{40,64}$/);
    assert.equal(fs.existsSync(markerPath), false);
  } finally {
    if (previousHome === undefined) delete process.env.HOME;
    else process.env.HOME = previousHome;
    fs.rmSync(root, { recursive: true, force: true });
  }
});

test('dirty local source snapshot binds changed and untracked source file bytes', () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'automixer-source-snapshot-'));
  try {
    fs.mkdirSync(path.join(root, 'frontend'), { recursive: true });
    fs.mkdirSync(path.join(root, 'tests'), { recursive: true });
    fs.writeFileSync(path.join(root, 'frontend', 'source.js'), 'AAAA');
    fs.writeFileSync(path.join(root, 'tests', 'new_test.py'), 'BBBB');
    const readGit = args => {
      const command = args.join(' ');
      if (command === 'diff HEAD --name-only -z') return 'frontend/source.js\0';
      if (command === 'ls-files --others --exclude-standard -z') return 'tests/new_test.py\0';
      if (command === 'rev-parse HEAD') return 'a'.repeat(40);
      if (command === 'rev-parse HEAD^{tree}') return 'b'.repeat(40);
      if (command === 'status --porcelain=v1 -uall') {
        return ' M frontend/source.js\n?? tests/new_test.py';
      }
      throw new Error(`unexpected git command: ${command}`);
    };
    const before = captureSourceSnapshot(readGit, root);
    fs.writeFileSync(path.join(root, 'frontend', 'source.js'), 'CCCC');
    const afterTrackedMutation = captureSourceSnapshot(readGit, root);
    assert.notEqual(afterTrackedMutation.sourceContentSha256, before.sourceContentSha256);
    assert.throws(
      () => assertSourceSnapshot(before, afterTrackedMutation, 'local build'),
      /file contents changed/
    );
    fs.writeFileSync(path.join(root, 'frontend', 'source.js'), 'AAAA');
    fs.writeFileSync(path.join(root, 'tests', 'new_test.py'), 'DDDD');
    const afterUntrackedMutation = captureSourceSnapshot(readGit, root);
    assert.notEqual(afterUntrackedMutation.sourceContentSha256, before.sourceContentSha256);
  } finally {
    fs.rmSync(root, { recursive: true, force: true });
  }
});

test('local source content binding excludes docs while the stage status guard remains global', () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'automixer-source-doc-snapshot-'));
  try {
    fs.mkdirSync(path.join(root, 'frontend'), { recursive: true });
    fs.mkdirSync(path.join(root, 'Docs'), { recursive: true });
    fs.writeFileSync(path.join(root, 'frontend', 'source.js'), 'AAAA');
    fs.writeFileSync(path.join(root, 'Docs', 'STATE.md'), 'before');
    const readGit = args => {
      const command = args.join(' ');
      if (command === 'diff HEAD --name-only -z') {
        return 'frontend/source.js\0Docs/STATE.md\0';
      }
      if (command === 'ls-files --others --exclude-standard -z') return '';
      if (command === 'rev-parse HEAD') return 'a'.repeat(40);
      if (command === 'rev-parse HEAD^{tree}') return 'b'.repeat(40);
      if (command === 'status --porcelain=v1 -uall') {
        return ' M frontend/source.js\n M Docs/STATE.md';
      }
      throw new Error(`unexpected git command: ${command}`);
    };
    const before = captureSourceSnapshot(readGit, root);
    fs.writeFileSync(path.join(root, 'Docs', 'STATE.md'), 'after');
    const afterDocsOnly = captureSourceSnapshot(readGit, root);

    assert.equal(afterDocsOnly.sourceContentSha256, before.sourceContentSha256);
    assert.equal(afterDocsOnly.trackedChangedFileCount, 1);
  } finally {
    fs.rmSync(root, { recursive: true, force: true });
  }
});

test('effective local builder configuration uses explicit ad-hoc signing by default', () => {
  const probeSource = `
    const config = require('./electron-builder.local.cjs');
    process.stdout.write(JSON.stringify({
      forceCodeSigning: config.forceCodeSigning,
      hasIdentity: Object.hasOwn(config.mac, 'identity'),
      identity: config.mac.identity,
      hasSign: Object.hasOwn(config.mac, 'sign'),
      notarize: config.mac.notarize,
      timestamp: config.mac.timestamp,
      hardenedRuntime: config.mac.hardenedRuntime
    }));
  `;
  const probe = spawnSync(process.execPath, ['-e', probeSource], {
    cwd: frontendRoot,
    encoding: 'utf8',
    env: {
      ...process.env,
      AUTOMIXER_MAC_BUILD_PROFILE: 'local',
      AUTOMIXER_MAC_OUTPUT_DIR: path.join(frontendRoot, 'dist', 'config-probe'),
      AUTOMIXER_LOCAL_SIGNING_IDENTITY: ''
    }
  });
  assert.equal(probe.status, 0, probe.stderr);
  const config = JSON.parse(probe.stdout);
  assert.deepEqual(config, {
    forceCodeSigning: true,
    hasIdentity: true,
    identity: '-',
    hasSign: true,
    notarize: false,
    timestamp: null,
    hardenedRuntime: false
  });
});

test('effective release builder delegates app notarization only to awaited afterSign', () => {
  const probeSource = `
    const config = require('./electron-builder.release.cjs');
    process.stdout.write(JSON.stringify({
      afterSign: config.afterSign,
      notarize: config.mac.notarize,
      forceCodeSigning: config.forceCodeSigning,
      targets: config.mac.target
    }));
  `;
  const probe = spawnSync(process.execPath, ['-e', probeSource], {
    cwd: frontendRoot,
    encoding: 'utf8',
    env: {
      ...process.env,
      AUTOMIXER_MAC_BUILD_PROFILE: 'release',
      AUTOMIXER_MAC_OUTPUT_DIR: '/tmp/automixer-release-config-probe',
      AUTOMIXER_APPLE_TEAM_ID: 'QZN7ZYVYDZ',
      AUTOMIXER_DEVELOPER_ID_IDENTITY:
        'Developer ID Application: Example (QZN7ZYVYDZ)',
      AUTOMIXER_BACKEND_BUNDLE_PATH: '/tmp/automixer-backend-bundle',
      AUTOMIXER_BACKEND_SERVER_PATH: '/tmp/automixer-backend-bundle/server',
      AUTOMIXER_FRONTEND_BUILD_DIR: '/tmp/automixer-frontend-build',
      APPLE_KEYCHAIN_PROFILE: 'unit-test-profile'
    }
  });
  assert.equal(probe.status, 0, probe.stderr);
  assert.deepEqual(JSON.parse(probe.stdout), {
    afterSign: 'scripts/after-sign-macos.cjs',
    notarize: false,
    forceCodeSigning: true,
    targets: [
      { target: 'dmg', arch: ['arm64'] },
      { target: 'zip', arch: ['arm64'] }
    ]
  });
});

test('release profile fails closed on credentials and dirty checkout', () => {
  const valid = {
    AUTOMIXER_APPLE_TEAM_ID: 'QZN7ZYVYDZ',
    AUTOMIXER_DEVELOPER_ID_IDENTITY: 'Developer ID Application: Example (QZN7ZYVYDZ)',
    APPLE_KEYCHAIN_PROFILE: 'notary-profile'
  };

  assert.equal(validateProfile('local'), 'local');
  assert.equal(validateProfile('release'), 'release');
  assert.throws(() => validateProfile('legacy'));
  assert.throws(() => validateReleaseEnvironment({}, ''), /missing/);
  assert.throws(
    () => validateReleaseEnvironment({ ...valid, AUTOMIXER_APPLE_TEAM_ID: 'bad' }, ''),
    /Team ID/
  );
  assert.throws(
    () => validateReleaseEnvironment({ ...valid, AUTOMIXER_DEVELOPER_ID_IDENTITY: 'Apple Development: Example' }, ''),
    /Developer ID Application/
  );
  assert.throws(
    () => validateReleaseEnvironment(valid, ' M src/main.tsx'),
    /clean checkout/
  );
  assert.doesNotThrow(() => validateReleaseEnvironment(valid, ''));
  assert.throws(
    () => validateReleaseEnvironment({ ...valid, NODE_OPTIONS: '--require /tmp/inject.js' }, ''),
    /conflicting build or Apple credential inputs/
  );
  assert.throws(
    () => validateReleaseEnvironment({ ...valid, APPLE_ID: 'alternate@example.invalid' }, ''),
    /conflicting build or Apple credential inputs/
  );

  const child = createReleaseChildEnvironment({
    ...valid,
    HOME: '/tmp/home',
    PATH: '/tmp/untrusted-bin',
    NODE_OPTIONS: '--require /tmp/inject.js',
    PIP_INDEX_URL: 'https://example.invalid/simple'
  });
  assert.equal(child.HOME, '/tmp/home');
  assert.equal(child.AUTOMIXER_APPLE_TEAM_ID, valid.AUTOMIXER_APPLE_TEAM_ID);
  assert.equal(child.NODE_OPTIONS, undefined);
  assert.equal(child.PIP_INDEX_URL, undefined);
  assert.doesNotMatch(child.PATH, /untrusted-bin/);

  const exactPython = '/Library/Frameworks/Python.framework/Versions/3.14/bin/python3.14';
  if (fs.existsSync(exactPython)) {
    assert.equal(resolvePythonBinary('3.14.3', {
      AUTOMIXER_PYTHON_BIN: exactPython,
      PATH: '/usr/bin:/bin'
    }), fs.realpathSync(exactPython));
  }
  if (fs.existsSync('/opt/homebrew/bin/python3')) {
    assert.throws(() => resolvePythonBinary('3.14.3', {
      AUTOMIXER_PYTHON_BIN: '/opt/homebrew/bin/python3',
      PATH: '/opt/homebrew/bin:/usr/bin'
    }), /not CPython 3\.14\.3/);
  }
});

test('macOS build lock serializes shared dependency and packaging state', () => {
  const tempRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'automixer-build-lock-test-'));
  const lockPath = path.join(tempRoot, 'build.lock');
  try {
    const release = acquireBuildLock(lockPath);
    assert.throws(() => acquireBuildLock(lockPath), /macOS build lock exists/);
    release();
    const releaseAgain = acquireBuildLock(lockPath);
    releaseAgain();
    assert.equal(fs.existsSync(lockPath), false);
    fs.writeFileSync(lockPath, '');
    assert.throws(() => acquireBuildLock(lockPath), /stale macOS build lock/);
    assert.equal(fs.existsSync(lockPath), true);
  } finally {
    fs.rmSync(tempRoot, { recursive: true, force: true });
  }
});

test('release Python dependency inputs are exact and hash-enforced', () => {
  const repoRoot = path.resolve(frontendRoot, '..');
  const runtimeLock = fs.readFileSync(path.join(
    repoRoot, 'backend', 'requirements-macos-arm64.lock.txt'
  ), 'utf8');
  const buildLock = fs.readFileSync(path.join(
    repoRoot, 'backend', 'requirements-macos-arm64-build.lock.txt'
  ), 'utf8');
  const backendBuild = fs.readFileSync(path.join(repoRoot, 'build_backend.sh'), 'utf8');
  const requirementLines = runtimeLock.split('\n')
    .map(line => line.trim())
    .filter(line => line && !line.startsWith('#') && !line.startsWith('--'));
  assert.equal(requirementLines.length, 131);
  for (const line of requirementLines) {
    assert.match(line, /^[A-Za-z0-9_.-]+==[^ ]+ --hash=sha256:[0-9a-f]{64}$/);
  }
  assert.match(runtimeLock, /^--only-binary=:all:$/m);
  assert.match(runtimeLock, /^--no-binary=watchdog$/m);
  assert.match(buildLock, /setuptools==81\.0\.0 --hash=sha256:[0-9a-f]{64}/);
  assert.match(backendBuild, /--require-hashes --no-build-isolation/);
  assert.match(backendBuild, /python -m pip --isolated/);
  assert.doesNotMatch(backendBuild, /pip install "pip==\$PIP_VERSION"/);
});

test('release verification lifecycle behavior cleans before retained eligibility', () => {
  const artifactDirectory = fs.mkdtempSync(path.join(os.tmpdir(), 'automixer-lifecycle-'));
  const buildWorkspace = path.join(artifactDirectory, '.build');
  fs.mkdirSync(buildWorkspace);
  const stages = [];
  try {
    const paths = runReleaseVerificationLifecycle({
      stage(name, action) {
        stages.push(name);
        return action();
      },
      artifactDirectory,
      buildWorkspace,
      verifyInitial() {
        assert.equal(fs.existsSync(buildWorkspace), true);
        fs.writeFileSync(path.join(
          artifactDirectory, 'release-verification-initial.json'
        ), JSON.stringify({
          schema: 'automixer.macos_release_manifest.v2',
          pathPolicy: 'artifact_root_relative_and_logical_v1',
          releaseEligible: false,
          source: { head: 'a', tree: 'b', cleanCheckout: true },
          build: { metadata: { sha256: '1'.repeat(64) } },
          artifacts: {
            dmg: { sha256: '2'.repeat(64) },
            zip: { sha256: '3'.repeat(64) }
          },
          runtimeSmoke: { result: { sha256: '4'.repeat(64) } },
          notarization: {
            app: {
              evidence: { sha256: '5'.repeat(64) },
              rawAppleLog: { sha256: '6'.repeat(64) }
            },
            dmg: {
              evidence: { sha256: '7'.repeat(64) },
              rawAppleLog: { sha256: '8'.repeat(64) }
            }
          },
          verification: {
            mode: 'isolated_outputs',
            complete: false,
            eligibleAfterRetainedVerification: true
          }
        }));
      },
      verifyRetained() {
        assert.equal(fs.existsSync(buildWorkspace), false);
        assert.equal(fs.existsSync(path.join(
          artifactDirectory, 'release-verification-initial.json'
        )), true);
        fs.writeFileSync(path.join(artifactDirectory, 'release-manifest.json'), JSON.stringify({
          schema: 'automixer.macos_release_manifest.v2',
          pathPolicy: 'artifact_root_relative_and_logical_v1',
          releaseEligible: true,
          source: { head: 'a', tree: 'b', cleanCheckout: true },
          build: { metadata: { sha256: '1'.repeat(64) } },
          artifacts: {
            dmg: { sha256: '2'.repeat(64) },
            zip: { sha256: '3'.repeat(64) }
          },
          runtimeSmoke: { result: { sha256: '4'.repeat(64) } },
          notarization: {
            app: {
              evidence: { sha256: '5'.repeat(64) },
              rawAppleLog: { sha256: '6'.repeat(64) }
            },
            dmg: {
              evidence: { sha256: '7'.repeat(64) },
              rawAppleLog: { sha256: '8'.repeat(64) }
            }
          },
          verification: { mode: 'retained_artifacts', complete: true }
        }));
      }
    });
    assert.deepEqual(stages, [
      'isolated-output release verification',
      'release workspace cleanup',
      'retained-artifact release verification'
    ]);
    assert.equal(fs.existsSync(paths.initialPath), true);
    assert.equal(fs.existsSync(paths.finalPath), true);
  } finally {
    fs.rmSync(artifactDirectory, { recursive: true, force: true });
  }
});

test('release lifecycle preserves workspace when initial proof is missing', () => {
  const artifactDirectory = fs.mkdtempSync(path.join(os.tmpdir(), 'automixer-lifecycle-fail-'));
  const buildWorkspace = path.join(artifactDirectory, '.build');
  fs.mkdirSync(buildWorkspace);
  try {
    assert.throws(() => runReleaseVerificationLifecycle({
      stage: (_name, action) => action(),
      artifactDirectory,
      buildWorkspace,
      verifyInitial() {},
      verifyRetained() { throw new Error('must not run'); }
    }), /Initial isolated-output/);
    assert.equal(fs.existsSync(buildWorkspace), true);
    assert.equal(fs.existsSync(path.join(artifactDirectory, 'release-manifest.json')), false);
  } finally {
    fs.rmSync(artifactDirectory, { recursive: true, force: true });
  }
});
