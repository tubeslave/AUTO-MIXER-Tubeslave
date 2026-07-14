import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { createRequire } from 'node:module';
import { fileURLToPath, pathToFileURL } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const require = createRequire(import.meta.url);
const {
  BACKEND_NETWORK_SANDBOX_PROBES,
  BACKEND_SANDBOX_PROFILE_SHA256
} = require('../../../public/electronSecurity.js');
const passingSandboxProbeResults = Object.fromEntries(
  Object.keys(BACKEND_NETWORK_SANDBOX_PROBES).map(name => [name, true])
);
const verifierPath = path.resolve(__dirname, '../../../scripts/verify-macos-release.mjs');
const {
  assertBackendEntitlements,
  assertBuildMetadata,
  assertDmgNotarizationEvidence,
  assertDeveloperIdSignature,
  assertExactEntitlements,
  assertFreshArtifact,
  assertInitialReleaseVerification,
  assertMatchingFileContents,
  assertPassingSmokeEvidence,
  createPortableReleaseManifest,
  EXPECTED_RUNTIME_CRITICAL_SOURCE_PATHS,
  findReleaseArtifactInputs,
  packagedSourceManifestSha256,
  assertReleaseBuildAttestation,
  machOEntitlementProfile,
  parseCodesignDetails,
  runtimeIdentityManifestSha256,
  withMountedDmg
} = await import(pathToFileURL(verifierPath));
const evidencePath = path.resolve(__dirname, '../../../scripts/release-evidence.mjs');
const {
  artifactFileIdentity,
  artifactRelativePath,
  assertIdentityUnchanged,
  copyStableRegularFileNoFollow,
  fileIdentity,
  hashTree,
  publishNewArtifactFileNoFollow,
  readStableJsonFileNoFollow,
  removeArtifactFileNoFollow,
  resolveArtifactPath,
  writeNewArtifactFileNoFollow
} = await import(pathToFileURL(evidencePath));

const teamId = 'QZN7ZYVYDZ';

test('release signature parser distinguishes Developer ID from development signing', () => {
  const release = parseCodesignDetails([
    'Identifier=com.automixer.wing',
    'CodeDirectory flags=0x10000(runtime)',
    'Authority=Developer ID Application: Example (QZN7ZYVYDZ)',
    `TeamIdentifier=${teamId}`,
    'Timestamp=12 Jul 2026 at 04:00:00'
  ].join('\n'));
  const development = parseCodesignDetails([
    'Identifier=com.automixer.wing',
    'CodeDirectory flags=0x10000(runtime)',
    'Authority=Apple Development: Example (QZN7ZYVYDZ)',
    `TeamIdentifier=${teamId}`,
    'Timestamp=12 Jul 2026 at 04:00:00'
  ].join('\n'));

  assert.doesNotThrow(() => assertDeveloperIdSignature(release, {
    label: 'app',
    teamId
  }));
  assert.throws(() => assertDeveloperIdSignature(development, {
    label: 'app',
    teamId
  }), /Developer ID Application/);
  assert.throws(() => assertDeveloperIdSignature({ ...release, teamId: 'AAAAAAAAAA' }, {
    label: 'app',
    teamId
  }), /Team ID/);
  assert.throws(() => assertDeveloperIdSignature({ ...release, runtime: false }, {
    label: 'app',
    teamId
  }), /hardened runtime/);
  assert.throws(() => assertDeveloperIdSignature({ ...release, timestamp: '' }, {
    label: 'app',
    teamId
  }), /timestamp/);
});

test('signed release attestation binds source, backend, and frontend build outputs', () => {
  const context = {
    expectedHead: 'a'.repeat(40),
    expectedTree: 'b'.repeat(40),
    backendSha256: 'c'.repeat(64),
    backendBundleTreeSha256: 'e'.repeat(64),
    frontendTreeSha256: 'd'.repeat(64)
  };
  const valid = {
    schema: 'automixer.macos_release_build_attestation.v1',
    source: { head: context.expectedHead, tree: context.expectedTree },
    buildOutputs: {
      backendSha256: context.backendSha256,
      backendBundleTreeSha256: context.backendBundleTreeSha256,
      frontendTreeSha256: context.frontendTreeSha256
    }
  };
  assert.doesNotThrow(() => assertReleaseBuildAttestation(valid, context));
  assert.throws(() => assertReleaseBuildAttestation({
    ...valid,
    buildOutputs: { ...valid.buildOutputs, backendSha256: 'e'.repeat(64) }
  }, context), /does not match/);
  assert.throws(() => assertReleaseBuildAttestation({
    ...valid,
    source: { ...valid.source, head: 'f'.repeat(40) }
  }, context), /does not match/);
});

test('release metadata binds both Python locks and the exact interpreter', () => {
  const identity = suffix => ({ path: `/tmp/${suffix}`, size: 10, sha256: suffix.repeat(64) });
  const context = {
    expectedHead: 'a'.repeat(40),
    expectedTree: 'b'.repeat(40),
    frontendPackageLock: identity('c'),
    backendRequirementsLock: identity('d'),
    backendBuildRequirementsLock: identity('e'),
    pythonExecutable: identity('f'),
    backendServer: identity('1'),
    backendBundleTreeSha256: '3'.repeat(64),
    frontendTreeSha256: '2'.repeat(64),
    artifactRoot: '/tmp'
  };
  const metadata = {
    schema: 'automixer.macos_release_build_metadata.v1',
    pathPolicy: 'artifact_root_relative_and_logical_v1',
    source: { head: context.expectedHead, tree: context.expectedTree, cleanCheckout: true },
    locks: {
      frontendPackageLock: {
        ...context.frontendPackageLock,
        path: 'frontend/package-lock.json',
        pathBase: 'source_checkout'
      },
      backendRequirementsLock: {
        ...context.backendRequirementsLock,
        path: 'backend/requirements-macos-arm64.lock.txt',
        pathBase: 'source_checkout'
      },
      backendBuildRequirementsLock: {
        ...context.backendBuildRequirementsLock,
        path: 'backend/requirements-macos-arm64-build.lock.txt',
        pathBase: 'source_checkout'
      }
    },
    toolchainInputs: {
      pythonExecutable: {
        ...context.pythonExecutable,
        path: 'cpython/3.14.3/python',
        pathBase: 'toolchain_runtime'
      }
    },
    tools: Object.fromEntries([
      'node', 'npm', 'python', 'pip', 'pyinstaller', 'electronBuilder', 'electron',
      'osxSign', 'macOS', 'xcode'
    ].map(name => [name, name === 'pip' ? '25.3' : `${name}-version`])),
    buildOutputs: {
      backendServer: {
        ...context.backendServer,
        path: '.build/backend-dist/server/server'
      },
      backendBundleTreeSha256: context.backendBundleTreeSha256,
      frontendTreeSha256: context.frontendTreeSha256
    }
  };
  assert.doesNotThrow(() => assertBuildMetadata(metadata, context));
  assert.throws(() => assertBuildMetadata({
    ...metadata,
    toolchainInputs: { pythonExecutable: identity('9') }
  }, context), /dependency-lock binding mismatch/);
  assert.throws(() => assertBuildMetadata({
    ...metadata,
    locks: { ...metadata.locks, backendBuildRequirementsLock: identity('8') }
  }, context), /dependency-lock binding mismatch/);
  const legacy = {
    ...metadata,
    pathPolicy: undefined,
    locks: {
      frontendPackageLock: context.frontendPackageLock,
      backendRequirementsLock: context.backendRequirementsLock,
      backendBuildRequirementsLock: context.backendBuildRequirementsLock
    },
    toolchainInputs: { pythonExecutable: context.pythonExecutable },
    buildOutputs: { ...metadata.buildOutputs, backendServer: context.backendServer }
  };
  assert.throws(
    () => assertBuildMetadata(legacy, context),
    /path policy mismatch/
  );
  assert.doesNotThrow(() => assertBuildMetadata(legacy, {
    ...context,
    allowLegacy: true
  }));
});

test('retained verification requires a matching initial isolated-output proof', () => {
  const identity = suffix => ({ path: `/tmp/${suffix}`, size: 12, sha256: suffix.repeat(64) });
  const context = {
    expectedHead: 'a'.repeat(40),
    expectedTree: 'b'.repeat(40),
    buildMetadata: identity('f'),
    appPath: '/tmp/AUTO MIXER Tubeslave.app',
    appTreeSha256: '1'.repeat(64),
    buildAttestation: identity('2'),
    dmg: identity('c'),
    zip: identity('d'),
    smokeResult: identity('e'),
    artifactRoot: '/tmp'
  };
  const portableIdentity = value => ({ ...value, path: path.basename(value.path) });
  const valid = {
    schema: 'automixer.macos_release_manifest.v1',
    pathPolicy: 'artifact_root_relative_and_logical_v1',
    releaseEligible: false,
    verification: {
      mode: 'isolated_outputs',
      complete: false,
      eligibleAfterRetainedVerification: true,
      isolatedOutputsPresent: true
    },
    source: { head: context.expectedHead, tree: context.expectedTree },
    build: { metadata: portableIdentity(context.buildMetadata) },
    app: {
      path: path.basename(context.appPath),
      treeSha256: context.appTreeSha256,
      buildAttestation: portableIdentity(context.buildAttestation)
    },
    artifacts: {
      dmg: portableIdentity(context.dmg),
      zip: portableIdentity(context.zip)
    },
    runtimeSmoke: { result: portableIdentity(context.smokeResult) }
  };
  assert.doesNotThrow(() => assertInitialReleaseVerification(valid, context));
  const legacy = {
    ...valid,
    pathPolicy: undefined,
    build: { metadata: context.buildMetadata },
    app: {
      ...valid.app,
      path: context.appPath,
      buildAttestation: context.buildAttestation
    },
    artifacts: { dmg: context.dmg, zip: context.zip },
    runtimeSmoke: { result: context.smokeResult }
  };
  assert.throws(
    () => assertInitialReleaseVerification(legacy, context),
    /Initial isolated-output/
  );
  assert.doesNotThrow(() => assertInitialReleaseVerification(legacy, {
    ...context,
    allowLegacy: true
  }));
  assert.throws(() => assertInitialReleaseVerification({
    ...valid,
    verification: { mode: 'retained_artifacts' }
  }, context), /Initial isolated-output/);
  assert.throws(() => assertInitialReleaseVerification({
    ...valid,
    artifacts: { ...valid.artifacts, dmg: identity('f') }
  }, context), /Initial isolated-output/);
  assert.throws(() => assertInitialReleaseVerification({
    ...valid,
    verification: { ...valid.verification, isolatedOutputsPresent: false }
  }, context), /Initial isolated-output/);
  assert.throws(() => assertInitialReleaseVerification({
    ...valid,
    build: { metadata: identity('7') }
  }, context), /Initial isolated-output/);
  assert.throws(() => assertInitialReleaseVerification({
    ...valid,
    app: { ...valid.app, treeSha256: '8'.repeat(64) }
  }, context), /Initial isolated-output/);
});

test('final release manifest serializes no builder, repository, or temp paths', () => {
  const tempRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'automixer-release-manifest-'));
  const artifactDirectory = path.join(tempRoot, 'artifact');
  const appPath = path.join(artifactDirectory, 'mac-arm64', 'AUTO MIXER Tubeslave.app');
  const paths = {
    buildMetadataPath: path.join(artifactDirectory, 'build-metadata.json'),
    initialVerificationPath: path.join(
      artifactDirectory,
      'release-verification-initial.json'
    ),
    attestationPath: path.join(
      appPath,
      'Contents',
      'Resources',
      'release-build-attestation.json'
    ),
    dmgPath: path.join(artifactDirectory, 'AUTO MIXER.dmg'),
    zipPath: path.join(artifactDirectory, 'AUTO MIXER.zip'),
    notarizationEvidencePath: path.join(artifactDirectory, 'dmg-notarization.json'),
    smokeResultPath: path.join(
      artifactDirectory,
      'runtime-smoke',
      'run',
      'result.json'
    )
  };
  try {
    for (const filePath of Object.values(paths)) {
      fs.mkdirSync(path.dirname(filePath), { recursive: true });
      fs.writeFileSync(filePath, path.basename(filePath));
    }
    fs.writeFileSync(
      path.join(path.dirname(paths.smokeResultPath), 'artifact-manifest.json'),
      'manifest'
    );
    fs.writeFileSync(path.join(path.dirname(paths.smokeResultPath), 'result.md'), 'markdown');
    const buildMetadata = {
      locks: {
        frontendPackageLock: {
          path: 'frontend/package-lock.json',
          pathBase: 'source_checkout',
          size: 1,
          sha256: '1'.repeat(64)
        }
      },
      tools: { pip: '25.3' },
      toolchainInputs: {
        pythonExecutable: {
          path: 'cpython/3.14.3/python',
          pathBase: 'toolchain_runtime',
          size: 1,
          sha256: '2'.repeat(64)
        }
      }
    };
    const manifest = createPortableReleaseManifest({
      artifactDirectory,
      retainedArtifacts: true,
      isolatedOutputsPresent: false,
      initialVerificationPath: paths.initialVerificationPath,
      expectedHead: 'a'.repeat(40),
      expectedTree: 'b'.repeat(40),
      buildMetadataPath: paths.buildMetadataPath,
      buildMetadata,
      appPath,
      appBefore: {
        infoPlist: { CFBundleIdentifier: 'com.automixer.wing' },
        treeSha256: 'c'.repeat(64),
        packagedInputs: {
          runtime: {
            'electron.js': {
              source: {
                path: 'frontend/public/electron.js',
                pathBase: 'source_checkout',
                size: 1,
                sha256: '3'.repeat(64)
              },
              packaged: {
                path: 'public/electron.js',
                pathBase: 'packaged_app_asar',
                size: 1,
                sha256: '3'.repeat(64)
              }
            }
          },
          config: {}
        },
        signatures: { application: { identifier: 'com.automixer.wing' } },
        backendEntitlements: { 'com.apple.security.cs.disable-library-validation': true }
      },
      zipVerification: { treeSha256: 'c'.repeat(64) },
      dmgVerification: { treeSha256: 'c'.repeat(64) },
      dmgPath: paths.dmgPath,
      zipPath: paths.zipPath,
      dmgSignature: { identifier: 'com.automixer.wing.dmg' },
      notarizationEvidence: { submissionId: '12345678-1234-1234-1234-123456789abc' },
      notarizationEvidencePath: paths.notarizationEvidencePath,
      smokeResultPath: paths.smokeResultPath
    });
    const serialized = JSON.stringify(manifest);
    assert.equal(manifest.pathPolicy, 'artifact_root_relative_and_logical_v1');
    assert.equal(manifest.app.path, 'mac-arm64/AUTO MIXER Tubeslave.app');
    assert.equal(manifest.build.metadata.path, 'build-metadata.json');
    assert.equal(manifest.artifacts.dmg.path, 'AUTO MIXER.dmg');
    assert.equal(
      manifest.runtimeSmoke.result.path,
      'runtime-smoke/run/result.json'
    );
    assert.doesNotMatch(serialized, new RegExp(tempRoot));
  } finally {
    fs.rmSync(tempRoot, { recursive: true, force: true });
  }
});

test('release smoke evidence is source-bound, app-bound, and no-WING', () => {
  const appPath = '/tmp/AUTO MIXER Tubeslave.app';
  const context = {
    expectedHead: 'a'.repeat(40),
    expectedTree: 'b'.repeat(40),
    appPath,
    appTreeSha256: 'c'.repeat(64),
    backendBundleTreeSha256: 'd'.repeat(64),
    artifactRoot: '/tmp',
    buildProfile: {
      path: '/tmp/build-profile.json',
      size: 512,
      sha256: 'e'.repeat(64)
    }
  };
  const runtimeCodeFiles = EXPECTED_RUNTIME_CRITICAL_SOURCE_PATHS.map(
    (runtimePath, index) => ({
      path: runtimePath,
      exists: true,
      mode: 0o644,
      size: 100 + index,
      sha256: index.toString(16).padStart(64, '0')
    })
  );
  const runtimeSource = {
    schema: 'automixer.backend_runtime_source.v1',
    observation: 'packaged_build_manifest',
    head: context.expectedHead,
    tree: context.expectedTree,
    worktree_dirty: false,
    status_sha256: '1'.repeat(64),
    tracked_diff_sha256: '2'.repeat(64),
    source_content_sha256: '3'.repeat(64),
    content_scope: 'automixer.production_source_inputs.v2'
  };
  const runtimeCode = {
    scope: 'automixer.live_safety_critical.v1',
    files: runtimeCodeFiles,
    files_sha256: runtimeSource.source_content_sha256
  };
  const runtimeChallenge = '4'.repeat(32);
  const embeddedManifestSha256 = packagedSourceManifestSha256({
    schema: 'automixer.packaged_runtime_source_manifest.v1',
    source: runtimeSource,
    code: runtimeCode
  });
  const runtimeIdentity = {
    type: 'runtime_identity',
    schema: 'automixer.backend_runtime_identity.v2',
    runtime_kind: 'packaged',
    instance_nonce: '5'.repeat(32),
    started_at_utc: '2026-07-12T00:00:00Z',
    process: {
      pid: 1234,
      instance_nonce: '5'.repeat(32),
      started_at_utc: '2026-07-12T00:00:00Z'
    },
    source: runtimeSource,
    code: runtimeCode,
    backend: {
      path: 'backend/server',
      exists: true,
      size: 4096,
      sha256: '6'.repeat(64)
    },
    launch_binding: {
      mode: 'packaged_build_manifest',
      packaged_manifest_schema: 'automixer.packaged_runtime_source_manifest.v1',
      packaged_manifest_name: 'automixer-runtime-source-manifest.json',
      packaged_manifest_size: 8192,
      packaged_manifest_file_sha256: '7'.repeat(64),
      packaged_manifest_sha256: embeddedManifestSha256,
      bound_before_process_start: true,
      proof_eligible: false,
      authenticated_launcher_bound: true,
      launch_token_sha256: '8'.repeat(64)
    },
    websocket: { host: '127.0.0.1', port: 8765, loopback_only: true },
    read_only_identity: true,
    live_authority_granted: false,
    request_challenge_nonce: runtimeChallenge
  };
  runtimeIdentity.manifest_sha256 = runtimeIdentityManifestSha256(runtimeIdentity);
  const valid = {
    schema: 'automixer.packaged_runtime_smoke.v1',
    pathPolicy: 'packaged_build_artifact_root_relative_v1',
    pass: true,
    readOnly: true,
    contactsWing: false,
    liveWrites: false,
    liveAuthorityGranted: false,
    appPath: path.basename(appPath),
    provenance: {
      head: context.expectedHead,
      tree: context.expectedTree,
      worktreeDirty: false,
      sourceBindingStrength: 'clean_checkout'
    },
    before: {
      appTree: { sha256: context.appTreeSha256 },
      packageFiles: {
        backend: { size: 4096, sha256: '6'.repeat(64) }
      },
      backendBundle: {
        path: path.relative(
          '/tmp',
          path.join(appPath, 'Contents', 'Resources', 'backend')
        ),
        sha256: context.backendBundleTreeSha256
      }
    },
    after: {
      appTree: { sha256: context.appTreeSha256 },
      backendBundle: {
        path: path.relative(
          '/tmp',
          path.join(appPath, 'Contents', 'Resources', 'backend')
        ),
        sha256: context.backendBundleTreeSha256
      }
    },
    runtime: {
      cdpReady: true,
      backendReadyMs: 5_900,
      backendReadyBudgetMs: 20_000,
      backendReadyWithinBudget: true
    },
    bootstrap: {
      validShape: true,
      tokenSha256: '8'.repeat(64),
      packagedOfflineLock: true,
      backendNetworkSandboxed: true,
      backendNetworkSandboxProfileSha256: BACKEND_SANDBOX_PROFILE_SHA256,
      backendNetworkSandboxProbeResults: passingSandboxProbeResults
    },
    invalidToken: { rejected: true },
    runtimeProbe: {
      authenticated: true,
      runtimeIdentityChallengeNonce: runtimeChallenge,
      runtimeIdentity,
      mode: {
        mode: 'analyze',
        packagedOfflineLock: true,
        mixerTransportLocked: true
      },
      blockedOperations: ['auto_connect', 'connect_wing', 'scan_mixers']
    },
    runtimeIdentityValidation: {
      valid: true,
      checks: { allRequiredChecksPassed: true },
      identity: runtimeIdentity,
      embeddedManifest: { size: 8192, sha256: '7'.repeat(64) }
    },
    processGroup: [{ pid: 1234, role: 'backend' }],
    blockedOperationsPass: true,
    reloadRecovery: {
      staleSessionTokenRemoved: true,
      secureBootstrapFailureShown: false,
      offlineLockShown: true,
      hardwareControlsHidden: true
    },
    networkEnforcement: {
      schema: 'automixer.packaged_runtime_network_enforcement.v1',
      mechanism: 'macos_sandbox_exec',
      scope: 'packaged_backend_process_tree',
      profileSha256: BACKEND_SANDBOX_PROFILE_SHA256,
      probeResults: passingSandboxProbeResults,
      nonLoopbackOutboundDenied: true,
      loopbackOutboundAllowed: true,
      preflightPassed: true,
      backendBoundaryReported: true,
      profileMatchesBackend: true,
      probeMatrixMatchesBackend: true
    },
    launchCopy: {
      signatureVerifiedBefore: true,
      signatureVerifiedAfter: true,
      treeSha256Before: context.appTreeSha256,
      treeSha256After: context.appTreeSha256,
      matchesSourceApp: true,
      unchanged: true,
      stillMatchesSourceApp: true
    },
    appTreeUnchanged: true,
    canonicalLogUnchanged: true,
    canonicalDecisionTraceUnchanged: true,
    canonicalDecisionMemoryUnchanged: true,
    consoleWalUnchanged: true,
    backendBundleUnchanged: true,
    packageFilesUnchanged: true,
    buildProfile: {
      schema: 'automixer.macos_build_profile.v1',
      profile: 'release',
      releaseEligible: false,
      bindingValid: true,
      appTreeMatches: true,
      path: path.basename(context.buildProfile.path),
      size: context.buildProfile.size,
      sha256: context.buildProfile.sha256,
      source: {
        head: context.expectedHead,
        tree: context.expectedTree,
        worktreeDirty: false
      },
      declaredApp: {
        path: path.basename(appPath),
        treeSha256: context.appTreeSha256
      }
    },
    tempWriteArtifactsAbsent: true,
    network: {
      externalConnections: [],
      wingPortSockets: [],
      monitorErrors: [],
      sampleCount: 3
    }
  };

  assert.doesNotThrow(() => assertPassingSmokeEvidence(valid, context));
  const legacy = {
    ...valid,
    pathPolicy: undefined,
    appPath,
    before: {
      ...valid.before,
      backendBundle: {
        ...valid.before.backendBundle,
        path: path.join(appPath, 'Contents', 'Resources', 'backend')
      }
    },
    after: {
      ...valid.after,
      backendBundle: {
        ...valid.after.backendBundle,
        path: path.join(appPath, 'Contents', 'Resources', 'backend')
      }
    },
    buildProfile: {
      ...valid.buildProfile,
      path: context.buildProfile.path,
      declaredApp: { ...valid.buildProfile.declaredApp, path: appPath }
    }
  };
  assert.throws(() => assertPassingSmokeEvidence(legacy, context), /path policy mismatch/);
  assert.doesNotThrow(() => assertPassingSmokeEvidence(legacy, {
    ...context,
    allowLegacy: true
  }));
  assert.throws(() => assertPassingSmokeEvidence({
    ...valid,
    appPath: '../other.app'
  }, context), /escapes the artifact root/);
  for (const mutation of [
    { pass: false },
    { contactsWing: true },
    { liveWrites: true },
    { runtime: { ...valid.runtime, backendReadyWithinBudget: false } },
    { runtime: { ...valid.runtime, backendReadyBudgetMs: 120_000 } },
    { runtime: { ...valid.runtime, backendReadyMs: 20_001 } },
    { appPath: '/tmp/other.app' },
    { provenance: { ...valid.provenance, head: 'd'.repeat(40) } },
    { before: { appTree: { sha256: 'e'.repeat(64) } } },
    { network: { externalConnections: ['198.51.100.1:443'] } },
    { invalidToken: { rejected: false } },
    { runtimeProbe: { ...valid.runtimeProbe, authenticated: false } },
    { runtimeIdentityValidation: { ...valid.runtimeIdentityValidation, valid: false } },
    { runtimeIdentityValidation: { ...valid.runtimeIdentityValidation, checks: null } },
    { runtimeProbe: {
      ...valid.runtimeProbe,
      runtimeIdentity: { ...runtimeIdentity, runtime_kind: 'source' }
    } },
    { runtimeProbe: { ...valid.runtimeProbe, blockedOperations: ['auto_connect'] } },
    { reloadRecovery: { ...valid.reloadRecovery, offlineLockShown: false } },
    { networkEnforcement: { ...valid.networkEnforcement, preflightPassed: false } },
    { networkEnforcement: {
      ...valid.networkEnforcement,
      probeResults: { ...valid.networkEnforcement.probeResults, externalTcp4Denied: false }
    } },
    { launchCopy: { ...valid.launchCopy, unchanged: false } },
    { canonicalDecisionTraceUnchanged: false },
    { canonicalDecisionMemoryUnchanged: false },
    { backendBundleUnchanged: false },
    { buildProfile: { ...valid.buildProfile, bindingValid: false } },
    { buildProfile: { ...valid.buildProfile, releaseEligible: true } },
    {
      before: {
        ...valid.before,
        backendBundle: { ...valid.before.backendBundle, sha256: 'f'.repeat(64) }
      }
    }
  ]) {
    assert.throws(() => assertPassingSmokeEvidence({ ...valid, ...mutation }, context));
  }
});

test('mode-aware tree hash and artifact identity reject chmod and replacement', () => {
  const tempRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'automixer-release-test-'));
  try {
    const tree = path.join(tempRoot, 'tree');
    fs.mkdirSync(tree);
    const executable = path.join(tree, 'server');
    fs.writeFileSync(executable, 'same bytes');
    fs.chmodSync(executable, 0o644);
    const beforeMode = hashTree(tree);
    fs.chmodSync(executable, 0o755);
    assert.notEqual(hashTree(tree), beforeMode);

    const archive = path.join(tempRoot, 'artifact.zip');
    fs.writeFileSync(archive, 'AAAA');
    const beforeIdentity = fileIdentity(archive);
    fs.writeFileSync(archive, 'BBBB');
    assert.throws(
      () => assertIdentityUnchanged(beforeIdentity, fileIdentity(archive), 'ZIP'),
      /changed/
    );
  } finally {
    fs.rmSync(tempRoot, { recursive: true, force: true });
  }
});

test('artifact evidence paths are portable and fail closed on traversal or symlink escape', () => {
  const tempRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'automixer-artifact-path-'));
  const artifactRoot = path.join(tempRoot, 'artifact');
  const filePath = path.join(artifactRoot, 'nested', 'result.json');
  const outsidePath = path.join(tempRoot, 'outside.json');
  try {
    fs.mkdirSync(path.dirname(filePath), { recursive: true });
    fs.writeFileSync(filePath, '{}\n');
    fs.writeFileSync(outsidePath, 'outside\n');
    assert.equal(
      artifactRelativePath(artifactRoot, filePath),
      'nested/result.json'
    );
    assert.equal(
      artifactFileIdentity(artifactRoot, filePath).path,
      'nested/result.json'
    );
    assert.equal(
      resolveArtifactPath(artifactRoot, 'nested/result.json'),
      filePath
    );
    assert.equal(
      resolveArtifactPath(
        artifactRoot,
        filePath,
        'legacy identity',
        { allowLegacyAbsoluteWithinRoot: true }
      ),
      filePath
    );
    assert.throws(
      () => resolveArtifactPath(artifactRoot, '../outside.json'),
      /escapes the artifact root/
    );
    assert.throws(
      () => resolveArtifactPath(artifactRoot, outsidePath),
      /must be artifact-root-relative/
    );
    assert.throws(
      () => resolveArtifactPath(
        artifactRoot,
        outsidePath,
        'legacy outside identity',
        { allowLegacyAbsoluteWithinRoot: true }
      ),
      /escapes the artifact root/
    );
    fs.symlinkSync(outsidePath, path.join(artifactRoot, 'escape.json'));
    assert.throws(
      () => resolveArtifactPath(artifactRoot, 'escape.json'),
      /resolves outside the artifact root/
    );
  } finally {
    fs.rmSync(tempRoot, { recursive: true, force: true });
  }
});

test('packaged runtime and config source binding compares exact bytes', () => {
  const tempRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'automixer-source-bind-test-'));
  try {
    const source = path.join(tempRoot, 'source.js');
    const packaged = path.join(tempRoot, 'packaged.js');
    fs.writeFileSync(source, 'trusted bytes\n');
    fs.writeFileSync(packaged, 'trusted bytes\n');
    assert.doesNotThrow(() => assertMatchingFileContents(source, packaged, 'runtime'));
    fs.writeFileSync(packaged, 'mutated bytes\n');
    assert.throws(
      () => assertMatchingFileContents(source, packaged, 'runtime'),
      /does not match the clean source input/
    );
  } finally {
    fs.rmSync(tempRoot, { recursive: true, force: true });
  }
});

test('release artifact preflight rejects external DMG ZIP and evidence symlinks before commands', () => {
  const tempRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'automixer-release-symlink-'));
  const artifactRoot = path.join(tempRoot, 'artifact');
  const outsideRoot = path.join(tempRoot, 'outside');
  let commandCalls = 0;
  const preflightThenCommand = () => {
    const inputs = findReleaseArtifactInputs(artifactRoot);
    commandCalls += 1;
    return inputs;
  };
  try {
    fs.mkdirSync(artifactRoot);
    fs.mkdirSync(outsideRoot);
    const outsideDmg = path.join(outsideRoot, 'outside.dmg');
    const outsideZip = path.join(outsideRoot, 'outside.zip');
    const outsideEvidence = path.join(outsideRoot, 'dmg-notarization.json');
    fs.writeFileSync(outsideDmg, 'dmg');
    fs.writeFileSync(outsideZip, 'zip');
    fs.writeFileSync(outsideEvidence, '{}');
    fs.writeFileSync(path.join(artifactRoot, 'release.zip'), 'zip');
    fs.writeFileSync(path.join(artifactRoot, 'dmg-notarization.json'), '{}');
    fs.symlinkSync(outsideDmg, path.join(artifactRoot, 'release.dmg'));

    assert.throws(preflightThenCommand, /must be a regular file/);
    assert.equal(commandCalls, 0);
    const symlinkDmgIdentity = fileIdentity(path.join(artifactRoot, 'release.dmg'));
    assert.throws(() => assertDmgNotarizationEvidence({
      schema: 'automixer.dmg_notarization.v1',
      pathPolicy: 'artifact_root_relative_v1',
      generatedAt: '2026-07-12T00:00:00.000Z',
      status: 'Accepted',
      submissionId: '12345678-1234-1234-1234-123456789abc',
      dmg: { ...symlinkDmgIdentity, path: 'release.dmg' }
    }, symlinkDmgIdentity, { artifactRoot }), /must be a regular file/);

    fs.rmSync(path.join(artifactRoot, 'release.dmg'));
    fs.writeFileSync(path.join(artifactRoot, 'release.dmg'), 'dmg');
    fs.rmSync(path.join(artifactRoot, 'release.zip'));
    fs.symlinkSync(outsideZip, path.join(artifactRoot, 'release.zip'));
    assert.throws(preflightThenCommand, /must be a regular file/);
    assert.equal(commandCalls, 0);

    fs.rmSync(path.join(artifactRoot, 'release.zip'));
    fs.writeFileSync(path.join(artifactRoot, 'release.zip'), 'zip');
    fs.rmSync(path.join(artifactRoot, 'dmg-notarization.json'));
    fs.symlinkSync(outsideEvidence, path.join(artifactRoot, 'dmg-notarization.json'));
    assert.throws(preflightThenCommand, /must be a regular file/);
    assert.equal(commandCalls, 0);
  } finally {
    fs.rmSync(tempRoot, { recursive: true, force: true });
  }
});

test('final artifact identity rejects an in-root identical-byte symlink replacement', () => {
  const artifactRoot = fs.mkdtempSync(
    path.join(os.tmpdir(), 'automixer-release-final-identity-')
  );
  const dmgPath = path.join(artifactRoot, 'release.dmg');
  const replacementPath = path.join(artifactRoot, 'replacement.dmg.bytes');
  try {
    fs.writeFileSync(dmgPath, 'identical bytes');
    fs.writeFileSync(replacementPath, 'identical bytes');
    const before = artifactFileIdentity(artifactRoot, dmgPath, 'Initial DMG');

    fs.rmSync(dmgPath);
    fs.symlinkSync(replacementPath, dmgPath);

    assert.throws(
      () => artifactFileIdentity(artifactRoot, dmgPath, 'Final DMG'),
      /existing regular file|changed identity/
    );
    assert.throws(
      () => findReleaseArtifactInputs(artifactRoot),
      /must be a regular file/
    );
    assert.equal(fileIdentity(dmgPath).sha256, before.sha256);
  } finally {
    fs.rmSync(artifactRoot, { recursive: true, force: true });
  }
});

test('artifact identity rejects same-inode mutation restored during the first hash pass', () => {
  const artifactRoot = fs.mkdtempSync(
    path.join(os.tmpdir(), 'automixer-release-stable-read-')
  );
  const artifactPath = path.join(artifactRoot, 'release.dmg');
  const originalBytes = Buffer.alloc(2 * 1024 * 1024, 0x41);
  const hookPath = path.join(artifactRoot, 'mutate-during-read.cjs');
  const markerPath = path.join(artifactRoot, 'mutation-ran');
  const originalNodeOptions = process.env.NODE_OPTIONS;
  try {
    fs.writeFileSync(artifactPath, originalBytes);
    fs.writeFileSync(hookPath, [
      "const fs = require('node:fs');",
      'const originalReadSync = fs.readSync;',
      'let mutated = false;',
      'fs.readSync = function(descriptor, buffer, offset, length, position) {',
      '  const count = originalReadSync(descriptor, buffer, offset, length, position);',
      '  const size = Number(process.env.AUTOMIXER_TEST_MUTATE_SIZE);',
      '  if (!mutated && Number.isInteger(position) && position + count >= size) {',
      '    mutated = true;',
      "    fs.writeFileSync(process.env.AUTOMIXER_TEST_MUTATE_PATH, Buffer.alloc(size, 0x42));",
      "    fs.writeFileSync(process.env.AUTOMIXER_TEST_MUTATE_PATH, Buffer.alloc(size, 0x41));",
      "    fs.writeFileSync(process.env.AUTOMIXER_TEST_MUTATE_MARKER, 'ran');",
      '  }',
      '  return count;',
      '};'
    ].join('\n'));
    process.env.NODE_OPTIONS = `--require=${hookPath}`;
    process.env.AUTOMIXER_TEST_MUTATE_PATH = artifactPath;
    process.env.AUTOMIXER_TEST_MUTATE_SIZE = String(originalBytes.length);
    process.env.AUTOMIXER_TEST_MUTATE_MARKER = markerPath;

    assert.throws(
      () => artifactFileIdentity(artifactRoot, artifactPath, 'Race-tested DMG'),
      /changed during pinned read/
    );
    assert.equal(fs.readFileSync(markerPath, 'utf8'), 'ran');
    assert.deepEqual(fs.readFileSync(artifactPath), originalBytes);
  } finally {
    if (originalNodeOptions === undefined) delete process.env.NODE_OPTIONS;
    else process.env.NODE_OPTIONS = originalNodeOptions;
    delete process.env.AUTOMIXER_TEST_MUTATE_PATH;
    delete process.env.AUTOMIXER_TEST_MUTATE_SIZE;
    delete process.env.AUTOMIXER_TEST_MUTATE_MARKER;
    fs.rmSync(artifactRoot, { recursive: true, force: true });
  }
});

test('initial and final release outputs cannot overwrite external symlink targets', () => {
  const tempRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'automixer-release-output-'));
  const artifactRoot = path.join(tempRoot, 'artifact');
  const outsidePath = path.join(tempRoot, 'outside.json');
  try {
    fs.mkdirSync(artifactRoot);
    fs.writeFileSync(outsidePath, 'DO NOT REPLACE');
    for (const name of [
      'release-verification-initial.json',
      'release-manifest.json'
    ]) {
      const outputPath = path.join(artifactRoot, name);
      fs.symlinkSync(outsidePath, outputPath);
      assert.throws(
        () => writeNewArtifactFileNoFollow(
          artifactRoot,
          outputPath,
          '{"releaseEligible":true}\n',
          name
        ),
        /already exists/
      );
      assert.equal(fs.readFileSync(outsidePath, 'utf8'), 'DO NOT REPLACE');
      fs.rmSync(outputPath);
    }
  } finally {
    fs.rmSync(tempRoot, { recursive: true, force: true });
  }
});

test('release publication binds the candidate to the exact intended JSON bytes', () => {
  const artifactRoot = fs.mkdtempSync(
    path.join(os.tmpdir(), 'automixer-release-candidate-')
  );
  const candidatePath = path.join(artifactRoot, '.release-manifest.candidate');
  const manifestPath = path.join(artifactRoot, 'release-manifest.json');
  try {
    const intended = writeNewArtifactFileNoFollow(
      artifactRoot,
      candidatePath,
      '{"releaseEligible":true,"source":"verified"}\n',
      'Release manifest candidate'
    );
    fs.writeFileSync(
      candidatePath,
      '{"releaseEligible":true,"source":"attacker"}\n'
    );

    assert.throws(
      () => publishNewArtifactFileNoFollow(
        artifactRoot,
        candidatePath,
        manifestPath,
        intended,
        'Release manifest publication'
      ),
      /candidate bytes changed/
    );
    assert.equal(fs.existsSync(manifestPath), false);
  } finally {
    fs.rmSync(artifactRoot, { recursive: true, force: true });
  }
});

test('published manifest cleanup refuses to unlink bytes changed after publication', () => {
  const artifactRoot = fs.mkdtempSync(
    path.join(os.tmpdir(), 'automixer-release-published-cleanup-')
  );
  const candidatePath = path.join(artifactRoot, '.release-manifest.candidate');
  const manifestPath = path.join(artifactRoot, 'release-manifest.json');
  try {
    const intended = writeNewArtifactFileNoFollow(
      artifactRoot,
      candidatePath,
      '{"releaseEligible":true}\n',
      'Release manifest candidate'
    );
    const published = publishNewArtifactFileNoFollow(
      artifactRoot,
      candidatePath,
      manifestPath,
      intended,
      'Release manifest publication'
    );
    fs.writeFileSync(manifestPath, '{"releaseEligible":false}\n');

    assert.throws(
      () => removeArtifactFileNoFollow(
        artifactRoot,
        manifestPath,
        'Postcondition cleanup',
        published
      ),
      /refuses to remove changed bytes/
    );
    assert.equal(fs.existsSync(manifestPath), true);
  } finally {
    fs.rmSync(artifactRoot, { recursive: true, force: true });
  }
});

test('private native-command snapshots reject symlink sources and bind exact bytes', () => {
  const tempRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'automixer-release-snapshot-'));
  const sourcePath = path.join(tempRoot, 'release.dmg');
  const snapshotPath = path.join(tempRoot, 'private.dmg');
  const symlinkPath = path.join(tempRoot, 'release-link.dmg');
  try {
    fs.writeFileSync(sourcePath, 'signed and notarized bytes');
    const snapshot = copyStableRegularFileNoFollow(
      sourcePath,
      snapshotPath,
      'Private DMG snapshot'
    );
    assert.equal(snapshot.sha256, fileIdentity(sourcePath).sha256);
    assert.equal(fs.readFileSync(snapshotPath, 'utf8'), 'signed and notarized bytes');

    fs.symlinkSync(sourcePath, symlinkPath);
    assert.throws(
      () => copyStableRegularFileNoFollow(
        symlinkPath,
        path.join(tempRoot, 'symlink-copy.dmg'),
        'Symlink DMG snapshot'
      ),
      /ELOOP|regular file/
    );
  } finally {
    fs.rmSync(tempRoot, { recursive: true, force: true });
  }
});

test('stable JSON reader rejects an intermediate parent swap before open', () => {
  const tempRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'automixer-json-parent-swap-'));
  const artifactRoot = path.join(tempRoot, 'artifact');
  const parentPath = path.join(artifactRoot, 'run');
  const heldParentPath = path.join(artifactRoot, 'run-held');
  const outsideParent = path.join(tempRoot, 'outside');
  const targetPath = path.join(parentPath, 'result.json');
  const originalRealpathSync = fs.realpathSync;
  let swapped = false;
  try {
    fs.mkdirSync(parentPath, { recursive: true });
    fs.mkdirSync(outsideParent);
    fs.writeFileSync(targetPath, '{"acceptedOrigin":"inside"}\n');
    fs.writeFileSync(
      path.join(outsideParent, 'result.json'),
      '{"acceptedOrigin":"outside"}\n'
    );
    fs.realpathSync = function patchedRealpathSync(candidate, ...args) {
      const result = originalRealpathSync(candidate, ...args);
      if (!swapped && path.resolve(candidate) === parentPath) {
        fs.renameSync(parentPath, heldParentPath);
        fs.symlinkSync(outsideParent, parentPath);
        swapped = true;
      }
      return result;
    };

    assert.throws(
      () => readStableJsonFileNoFollow(
        targetPath,
        'Race-tested JSON',
        { containmentRoot: artifactRoot }
      ),
      /parent changed|pinned read/
    );
    assert.equal(swapped, true);
  } finally {
    fs.realpathSync = originalRealpathSync;
    fs.rmSync(tempRoot, { recursive: true, force: true });
  }
});

test('mounted DMG is detached when application discovery fails', () => {
  const tempRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'automixer-dmg-test-'));
  const commands = [];
  try {
    assert.throws(() => withMountedDmg('/tmp/fake.dmg', tempRoot, () => {}, {
      runCommand(command, args) {
        commands.push([command, ...args]);
        return { status: 0, stdout: '<plist/>', stderr: '' };
      },
      parsePlist() {},
      findMountedApp() { throw new Error('application missing'); }
    }), /application missing/);
    assert.deepEqual(commands.at(-1).slice(0, 2), ['/usr/bin/hdiutil', 'detach']);
  } finally {
    fs.rmSync(tempRoot, { recursive: true, force: true });
  }
});

test('mounted DMG cleanup retries with force after a normal detach failure', () => {
  const tempRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'automixer-dmg-force-test-'));
  const commands = [];
  try {
    assert.throws(() => withMountedDmg('/tmp/fake.dmg', tempRoot, () => {}, {
      runCommand(command, args) {
        commands.push([command, ...args]);
        if (args[0] === 'attach') return { status: 0, stdout: '<plist/>', stderr: '' };
        if (args.includes('-force')) return { status: 0, stdout: '', stderr: '' };
        return { status: 1, stdout: '', stderr: 'busy' };
      },
      parsePlist() {},
      findMountedApp() { throw new Error('application missing'); }
    }), /application missing/);
    assert.deepEqual(commands.at(-1).slice(0, 3), [
      '/usr/bin/hdiutil', 'detach', '-force'
    ]);
  } finally {
    fs.rmSync(tempRoot, { recursive: true, force: true });
  }
});

test('backend entitlement and artifact freshness checks are exact', () => {
  assert.doesNotThrow(() => assertBackendEntitlements({
    'com.apple.security.cs.disable-library-validation': true
  }));
  assert.throws(() => assertBackendEntitlements({
    'com.apple.security.cs.disable-library-validation': true,
    'com.apple.security.cs.allow-jit': true
  }), /exact release allowlist/);
  assert.throws(() => assertBackendEntitlements({
    'com.apple.security.cs.disable-library-validation': false
  }), /exact release allowlist/);

  const notBefore = '2026-07-12T00:00:00.000Z';
  assert.doesNotThrow(() => assertFreshArtifact(
    { mtimeMs: Date.parse('2026-07-12T00:00:01.000Z') },
    notBefore,
    'ZIP'
  ));
  assert.throws(() => assertFreshArtifact(
    { mtimeMs: Date.parse('2026-07-11T23:59:59.000Z') },
    notBefore,
    'ZIP'
  ), /predates/);
});

test('application entitlement and DMG notarization allowlists are exact', () => {
  assert.doesNotThrow(() => assertExactEntitlements(
    { 'com.apple.security.network.client': true, 'com.apple.security.cs.allow-jit': true },
    { 'com.apple.security.cs.allow-jit': true, 'com.apple.security.network.client': true },
    'application'
  ));
  assert.throws(() => assertExactEntitlements(
    { 'com.apple.security.cs.allow-jit': true, 'com.apple.security.cs.debugger': true },
    { 'com.apple.security.cs.allow-jit': true },
    'application'
  ), /exact release allowlist/);

  const artifactRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'automixer-notary-evidence-'));
  const dmgPath = path.join(artifactRoot, 'AUTO MIXER.dmg');
  try {
    fs.writeFileSync(dmgPath, 'dmg bytes');
    const dmg = fileIdentity(dmgPath);
    const evidence = {
      schema: 'automixer.dmg_notarization.v1',
      pathPolicy: 'artifact_root_relative_v1',
      generatedAt: '2026-07-12T00:00:00.000Z',
      status: 'Accepted',
      submissionId: '12345678-1234-1234-1234-123456789abc',
      dmg: { ...dmg, path: path.basename(dmg.path) }
    };
    assert.doesNotThrow(() => assertDmgNotarizationEvidence(evidence, dmg, {
      artifactRoot
    }));
    const legacyEvidence = { ...evidence, pathPolicy: undefined, dmg };
    assert.throws(
      () => assertDmgNotarizationEvidence(legacyEvidence, dmg),
      /does not bind/
    );
    assert.doesNotThrow(() => assertDmgNotarizationEvidence(legacyEvidence, dmg, {
      artifactRoot,
      allowLegacy: true
    }));
    assert.throws(() => assertDmgNotarizationEvidence({
      ...evidence,
      submissionId: ''
    }, dmg, { artifactRoot }), /does not bind/);
    assert.throws(() => assertDmgNotarizationEvidence({
      ...evidence,
      dmg: { ...evidence.dmg, size: dmg.size + 1 }
    }, dmg, { artifactRoot }), /does not bind/);
  } finally {
    fs.rmSync(artifactRoot, { recursive: true, force: true });
  }
});

test('every packaged Mach-O kind maps to an explicit entitlement profile', () => {
  const main = 'AUTO MIXER Tubeslave';
  assert.equal(machOEntitlementProfile(`Contents/MacOS/${main}`, main), 'application');
  assert.equal(
    machOEntitlementProfile('Contents/Resources/backend/server', main),
    'backend'
  );
  assert.equal(machOEntitlementProfile(
    'Contents/Resources/backend/_internal/numpy/core/_multiarray_umath.so',
    main
  ), 'empty');
  assert.equal(machOEntitlementProfile(
    'Contents/Frameworks/AUTO MIXER Helper.app/Contents/MacOS/AUTO MIXER Helper',
    main
  ), 'inherited');
  assert.equal(machOEntitlementProfile(
    'Contents/Frameworks/Electron Framework.framework/Versions/A/Helpers/chrome_crashpad_handler',
    main
  ), 'inherited');
  assert.equal(machOEntitlementProfile(
    'Contents/Frameworks/Squirrel.framework/Versions/A/Resources/ShipIt',
    main
  ), 'inherited');
  assert.equal(machOEntitlementProfile(
    'Contents/Frameworks/Electron Framework.framework/Versions/A/Electron Framework',
    main
  ), 'empty');
  assert.equal(machOEntitlementProfile(
    'Contents/Frameworks/Electron Framework.framework/Versions/A/Libraries/libEGL.dylib',
    main
  ), 'empty');
});
