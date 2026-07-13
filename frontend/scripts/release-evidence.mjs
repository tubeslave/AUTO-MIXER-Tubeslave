import crypto from 'node:crypto';
import fs from 'node:fs';
import path from 'node:path';
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';

const artifactWriterHelper = fileURLToPath(
  new URL('./write-new-artifact-helper.mjs', import.meta.url)
);
const artifactPublisherHelper = fileURLToPath(
  new URL('./publish-new-artifact-helper.mjs', import.meta.url)
);
const artifactRemovalHelper = fileURLToPath(
  new URL('./remove-artifact-helper.mjs', import.meta.url)
);
const artifactReplacementHelper = fileURLToPath(
  new URL('./replace-artifact-helper.mjs', import.meta.url)
);
const stableFileIdentityHelper = fileURLToPath(
  new URL('./stable-file-identity-helper.mjs', import.meta.url)
);
const stableJsonReaderHelper = fileURLToPath(
  new URL('./stable-json-reader-helper.mjs', import.meta.url)
);

export function sha256File(filePath) {
  return crypto.createHash('sha256').update(fs.readFileSync(filePath)).digest('hex');
}

export function fileIdentity(filePath) {
  const stat = fs.statSync(filePath);
  return {
    path: filePath,
    size: stat.size,
    sha256: sha256File(filePath)
  };
}

function pinnedReadContext(filePath, containmentRoot, label) {
  const targetPath = path.resolve(filePath);
  const parentPath = path.dirname(targetPath);
  const rootPath = path.resolve(containmentRoot || parentPath);
  if (
    !isWithinRoot(rootPath, targetPath)
    || targetPath === rootPath
    || !isWithinRoot(rootPath, parentPath)
  ) {
    throw new Error(`${label} escapes its pinned read root`);
  }
  const rootStat = fs.lstatSync(rootPath, { bigint: true });
  const parentStat = fs.lstatSync(parentPath, { bigint: true });
  if (
    rootStat.isSymbolicLink()
    || parentStat.isSymbolicLink()
    || !rootStat.isDirectory()
    || !parentStat.isDirectory()
  ) {
    throw new Error(`${label} root or parent must be a real directory`);
  }
  const realRoot = fs.realpathSync(rootPath);
  const realParent = fs.realpathSync(parentPath);
  const expectedRealParent = path.join(
    realRoot,
    path.relative(rootPath, parentPath)
  );
  if (realParent !== expectedRealParent) {
    throw new Error(`${label} parent resolves through a symlink`);
  }
  return {
    targetPath,
    basename: path.basename(targetPath),
    rootPath,
    parentPath,
    rootStat,
    parentStat,
    realRoot,
    realParent
  };
}

function pinnedHelperArguments(context, label) {
  return [
    context.basename,
    context.rootPath,
    context.parentPath,
    context.rootStat.dev.toString(),
    context.rootStat.ino.toString(),
    context.parentStat.dev.toString(),
    context.parentStat.ino.toString(),
    context.realRoot,
    context.realParent,
    label
  ];
}

function regularFileIdentityNoFollow(filePath, label, containmentRoot = null) {
  const context = pinnedReadContext(filePath, containmentRoot, label);
  const completed = spawnSync(
    process.execPath,
    [stableFileIdentityHelper, ...pinnedHelperArguments(context, label)],
    {
      cwd: context.parentPath,
      encoding: 'utf8',
      maxBuffer: 1024 * 1024
    }
  );
  if (completed.error) {
    throw new Error(`${label} must be an existing regular file: ${completed.error.message}`);
  }
  if (completed.status !== 0) {
    throw new Error(
      `${label} must be an existing regular file: `
      + `${String(completed.stderr || '').trim() || 'pinned reader rejected input'}`
    );
  }
  const identity = JSON.parse(String(completed.stdout || '').trim());
  if (
    !Number.isSafeInteger(identity.size)
    || !/^[0-9a-f]{64}$/.test(String(identity.sha256 || ''))
  ) {
    throw new Error(`${label} pinned reader returned an invalid identity`);
  }
  return identity;
}

function stableStatEqual(left, right) {
  return Boolean(
    left
    && right
    && left.dev === right.dev
    && left.ino === right.ino
    && left.size === right.size
    && left.mode === right.mode
    && left.nlink === right.nlink
    && left.mtimeNs === right.mtimeNs
    && left.ctimeNs === right.ctimeNs
  );
}

function hashDescriptor(descriptor) {
  const hash = crypto.createHash('sha256');
  const buffer = Buffer.allocUnsafe(1024 * 1024);
  let position = 0;
  while (true) {
    const bytesRead = fs.readSync(
      descriptor,
      buffer,
      0,
      buffer.length,
      position
    );
    if (!bytesRead) break;
    hash.update(buffer.subarray(0, bytesRead));
    position += bytesRead;
  }
  return hash.digest('hex');
}

export function readStableJsonFileNoFollow(
  filePath,
  label = 'JSON file',
  { containmentRoot = null } = {}
) {
  const context = pinnedReadContext(filePath, containmentRoot, label);
  const completed = spawnSync(
    process.execPath,
    [stableJsonReaderHelper, ...pinnedHelperArguments(context, label)],
    {
      cwd: context.parentPath,
      encoding: null,
      maxBuffer: 96 * 1024 * 1024
    }
  );
  if (completed.error) throw completed.error;
  if (completed.status !== 0) {
    throw new Error(
      `${label} pinned read failed: `
      + `${Buffer.from(completed.stderr || '').toString('utf8').trim() || 'reader rejected input'}`
    );
  }
  const bytes = Buffer.from(
    Buffer.from(completed.stdout || '').toString('ascii'),
    'base64'
  );
  return {
    value: JSON.parse(bytes.toString('utf8')),
    bytes,
    identity: {
      size: bytes.length,
      sha256: crypto.createHash('sha256').update(bytes).digest('hex')
    }
  };
}

export function copyStableRegularFileNoFollow(
  sourcePath,
  destinationPath,
  label = 'Stable file snapshot'
) {
  const resolvedSource = path.resolve(sourcePath);
  const resolvedDestination = path.resolve(destinationPath);
  const expectedSourceIdentity = regularFileIdentityNoFollow(
    resolvedSource,
    `${label} source precondition`
  );
  const descriptor = fs.openSync(
    resolvedSource,
    fs.constants.O_RDONLY | (fs.constants.O_NOFOLLOW || 0)
  );
  let destinationDescriptor;
  let destinationStat;
  try {
    const before = fs.fstatSync(descriptor, { bigint: true });
    if (!before.isFile()) throw new Error(`${label} source must be a regular file`);
    destinationDescriptor = fs.openSync(
      resolvedDestination,
      fs.constants.O_WRONLY
        | fs.constants.O_CREAT
        | fs.constants.O_EXCL
        | (fs.constants.O_NOFOLLOW || 0),
      Number(before.mode & 0o777n)
    );
    destinationStat = fs.fstatSync(destinationDescriptor, { bigint: true });
    if (!destinationStat.isFile()) {
      throw new Error(`${label} destination must be a regular file`);
    }
    const copiedHash = crypto.createHash('sha256');
    const buffer = Buffer.allocUnsafe(1024 * 1024);
    let position = 0;
    while (true) {
      const bytesRead = fs.readSync(
        descriptor,
        buffer,
        0,
        buffer.length,
        position
      );
      if (!bytesRead) break;
      let written = 0;
      while (written < bytesRead) {
        written += fs.writeSync(
          destinationDescriptor,
          buffer,
          written,
          bytesRead - written,
          null
        );
      }
      copiedHash.update(buffer.subarray(0, bytesRead));
      position += bytesRead;
    }
    fs.fsyncSync(destinationDescriptor);
    destinationStat = fs.fstatSync(destinationDescriptor, { bigint: true });
    const middle = fs.fstatSync(descriptor, { bigint: true });
    const sourceHash = hashDescriptor(descriptor);
    const after = fs.fstatSync(descriptor, { bigint: true });
    const sourcePathStat = fs.lstatSync(resolvedSource, { bigint: true });
    const destinationPathStat = fs.lstatSync(resolvedDestination, { bigint: true });
    const copiedSha256 = copiedHash.digest('hex');
    if (
      sourcePathStat.isSymbolicLink()
      || destinationPathStat.isSymbolicLink()
      || !sourcePathStat.isFile()
      || !destinationPathStat.isFile()
      || !stableStatEqual(before, middle)
      || !stableStatEqual(middle, after)
      || !stableStatEqual(sourcePathStat, after)
      || !stableStatEqual(destinationPathStat, destinationStat)
      || copiedSha256 !== sourceHash
    ) {
      throw new Error(`${label} source or destination changed during copy`);
    }
  } catch (error) {
    if (destinationStat) {
      try {
        const current = fs.lstatSync(resolvedDestination, { bigint: true });
        if (
          !current.isSymbolicLink()
          && current.isFile()
          && current.dev === destinationStat.dev
          && current.ino === destinationStat.ino
        ) {
          fs.unlinkSync(resolvedDestination);
        }
      } catch {}
    }
    throw error;
  } finally {
    if (destinationDescriptor !== undefined) fs.closeSync(destinationDescriptor);
    fs.closeSync(descriptor);
  }
  const identity = regularFileIdentityNoFollow(resolvedDestination, label);
  const finalSourceIdentity = regularFileIdentityNoFollow(
    resolvedSource,
    `${label} source postcondition`
  );
  if (
    identity.size !== expectedSourceIdentity.size
    || identity.sha256 !== expectedSourceIdentity.sha256
    || finalSourceIdentity.size !== expectedSourceIdentity.size
    || finalSourceIdentity.sha256 !== expectedSourceIdentity.sha256
  ) {
    throw new Error(`${label} does not bind to the pinned source bytes`);
  }
  const directoryDescriptor = fs.openSync(
    path.dirname(resolvedDestination),
    fs.constants.O_RDONLY
  );
  try {
    fs.fsyncSync(directoryDescriptor);
  } finally {
    fs.closeSync(directoryDescriptor);
  }
  return { path: resolvedDestination, ...identity };
}

function isWithinRoot(rootPath, candidatePath) {
  const relative = path.relative(rootPath, candidatePath);
  return relative === '' || (
    !path.isAbsolute(relative)
    && relative !== '..'
    && !relative.startsWith(`..${path.sep}`)
  );
}

export function writeNewArtifactFileNoFollow(
  artifactRoot,
  targetPath,
  contents,
  label = 'Artifact output'
) {
  const resolvedRoot = path.resolve(artifactRoot);
  const resolvedTarget = path.resolve(targetPath);
  if (
    !isWithinRoot(resolvedRoot, resolvedTarget)
    || resolvedTarget === resolvedRoot
  ) {
    throw new Error(`${label} escapes the artifact root`);
  }
  const resolvedParent = path.dirname(resolvedTarget);
  const basename = path.basename(resolvedTarget);
  const rootStat = fs.lstatSync(resolvedRoot, { bigint: true });
  const parentStat = fs.lstatSync(resolvedParent, { bigint: true });
  if (
    rootStat.isSymbolicLink()
    || parentStat.isSymbolicLink()
    || !rootStat.isDirectory()
    || !parentStat.isDirectory()
    || !isWithinRoot(
      fs.realpathSync(resolvedRoot),
      fs.realpathSync(resolvedParent)
    )
  ) {
    throw new Error(`${label} root or parent is not a contained real directory`);
  }
  const intendedBytes = Buffer.isBuffer(contents)
    ? Buffer.from(contents)
    : Buffer.from(String(contents), 'utf8');
  const intendedIdentity = {
    size: intendedBytes.length,
    sha256: crypto.createHash('sha256').update(intendedBytes).digest('hex')
  };
  const completed = spawnSync(process.execPath, [
    artifactWriterHelper,
    basename,
    resolvedRoot,
    resolvedParent,
    rootStat.dev.toString(),
    rootStat.ino.toString(),
    parentStat.dev.toString(),
    parentStat.ino.toString(),
    label
  ], {
    cwd: resolvedParent,
    input: intendedBytes,
    encoding: 'utf8',
    maxBuffer: 4 * 1024 * 1024
  });
  if (completed.error) throw completed.error;
  if (completed.status !== 0) {
    throw new Error(
      `${label} creation failed: ${String(completed.stderr || '').trim() || 'writer rejected output'}`
    );
  }
  const createdIdentity = regularFileIdentityNoFollow(resolvedTarget, label);
  if (
    createdIdentity.size !== intendedIdentity.size
    || createdIdentity.sha256 !== intendedIdentity.sha256
  ) {
    try {
      removeArtifactFileNoFollow(
        resolvedRoot,
        resolvedTarget,
        `${label} mismatched-byte cleanup`,
        createdIdentity
      );
    } catch (cleanupError) {
      throw new AggregateError(
        [new Error(`${label} bytes changed during creation`), cleanupError],
        `${label} changed and could not be removed safely`
      );
    }
    throw new Error(`${label} bytes changed during creation`);
  }
  return { path: resolvedTarget, ...intendedIdentity };
}

export function publishNewArtifactFileNoFollow(
  artifactRoot,
  candidatePath,
  targetPath,
  expectedIdentity,
  label = 'Artifact publication'
) {
  const resolvedRoot = path.resolve(artifactRoot);
  const resolvedCandidate = path.resolve(candidatePath);
  const resolvedTarget = path.resolve(targetPath);
  const parentPath = path.dirname(resolvedTarget);
  if (
    path.dirname(resolvedCandidate) !== parentPath
    || !isWithinRoot(resolvedRoot, resolvedCandidate)
    || !isWithinRoot(resolvedRoot, resolvedTarget)
    || resolvedCandidate === resolvedTarget
  ) {
    throw new Error(`${label} paths must be distinct siblings inside the artifact root`);
  }
  const rootStat = fs.lstatSync(resolvedRoot, { bigint: true });
  const parentStat = fs.lstatSync(parentPath, { bigint: true });
  const candidateStat = fs.lstatSync(resolvedCandidate, { bigint: true });
  if (
    rootStat.isSymbolicLink()
    || parentStat.isSymbolicLink()
    || candidateStat.isSymbolicLink()
    || !rootStat.isDirectory()
    || !parentStat.isDirectory()
    || !candidateStat.isFile()
  ) {
    throw new Error(`${label} requires real root, parent, and candidate objects`);
  }
  if (
    !expectedIdentity
    || !Number.isSafeInteger(expectedIdentity.size)
    || !/^[0-9a-f]{64}$/.test(String(expectedIdentity.sha256 || ''))
  ) {
    throw new Error(`${label} immutable candidate identity is invalid`);
  }
  const candidateIdentity = regularFileIdentityNoFollow(
    resolvedCandidate,
    `${label} candidate`
  );
  if (
    candidateIdentity.size !== expectedIdentity.size
    || candidateIdentity.sha256 !== expectedIdentity.sha256
  ) {
    throw new Error(`${label} candidate bytes changed before publication`);
  }
  const completed = spawnSync(process.execPath, [
    artifactPublisherHelper,
    path.basename(resolvedCandidate),
    path.basename(resolvedTarget),
    resolvedRoot,
    parentPath,
    rootStat.dev.toString(),
    rootStat.ino.toString(),
    parentStat.dev.toString(),
    parentStat.ino.toString(),
    candidateStat.dev.toString(),
    candidateStat.ino.toString(),
    label
  ], {
    cwd: parentPath,
    encoding: 'utf8',
    maxBuffer: 1024 * 1024
  });
  if (completed.error) throw completed.error;
  if (completed.status !== 0) {
    throw new Error(
      `${label} failed: ${String(completed.stderr || '').trim() || 'publisher rejected output'}`
    );
  }
  const published = regularFileIdentityNoFollow(resolvedTarget, label);
  if (
    published.size !== expectedIdentity.size
    || published.sha256 !== expectedIdentity.sha256
  ) {
    try {
      removeArtifactFileNoFollow(
        resolvedRoot,
        resolvedTarget,
        `${label} mismatched-byte cleanup`,
        published
      );
    } catch (cleanupError) {
      throw new AggregateError(
        [new Error(`${label} bytes changed during publication`), cleanupError],
        `${label} changed and could not be removed safely`
      );
    }
    throw new Error(`${label} bytes changed during publication`);
  }
  return { path: resolvedTarget, ...published };
}

export function removeArtifactFileNoFollow(
  artifactRoot,
  targetPath,
  label = 'Artifact removal',
  expectedIdentity = null
) {
  const resolvedRoot = path.resolve(artifactRoot);
  const resolvedTarget = path.resolve(targetPath);
  const parentPath = path.dirname(resolvedTarget);
  if (!isWithinRoot(resolvedRoot, resolvedTarget) || resolvedTarget === resolvedRoot) {
    throw new Error(`${label} escapes the artifact root`);
  }
  const currentIdentity = regularFileIdentityNoFollow(resolvedTarget, label);
  if (
    expectedIdentity
    && (
      currentIdentity.size !== expectedIdentity.size
      || currentIdentity.sha256 !== expectedIdentity.sha256
    )
  ) {
    throw new Error(`${label} refuses to remove changed bytes`);
  }
  const rootStat = fs.lstatSync(resolvedRoot, { bigint: true });
  const parentStat = fs.lstatSync(parentPath, { bigint: true });
  const targetStat = fs.lstatSync(resolvedTarget, { bigint: true });
  const completed = spawnSync(process.execPath, [
    artifactRemovalHelper,
    path.basename(resolvedTarget),
    resolvedRoot,
    parentPath,
    rootStat.dev.toString(),
    rootStat.ino.toString(),
    parentStat.dev.toString(),
    parentStat.ino.toString(),
    targetStat.dev.toString(),
    targetStat.ino.toString(),
    label
  ], {
    cwd: parentPath,
    encoding: 'utf8',
    maxBuffer: 1024 * 1024
  });
  if (completed.error) throw completed.error;
  if (completed.status !== 0) {
    throw new Error(
      `${label} failed: ${String(completed.stderr || '').trim() || 'removal rejected'}`
    );
  }
  return true;
}

export function replaceArtifactFileNoFollow(
  artifactRoot,
  sourcePath,
  targetPath,
  expectedSourceIdentity,
  label = 'Artifact replacement'
) {
  const resolvedRoot = path.resolve(artifactRoot);
  const resolvedSource = path.resolve(sourcePath);
  const resolvedTarget = path.resolve(targetPath);
  if (
    !isWithinRoot(resolvedRoot, resolvedSource)
    || !isWithinRoot(resolvedRoot, resolvedTarget)
    || path.dirname(resolvedSource) !== resolvedRoot
    || path.dirname(resolvedTarget) !== resolvedRoot
    || resolvedSource === resolvedTarget
  ) {
    throw new Error(`${label} paths must be distinct and contained`);
  }
  if (
    !expectedSourceIdentity
    || !Number.isSafeInteger(expectedSourceIdentity.size)
    || !/^[0-9a-f]{64}$/.test(String(expectedSourceIdentity.sha256 || ''))
  ) {
    throw new Error(`${label} immutable source identity is invalid`);
  }
  const sourceParent = path.dirname(resolvedSource);
  const sourceParentStat = fs.lstatSync(sourceParent, { bigint: true });
  if (
    sourceParentStat.isSymbolicLink()
    || !sourceParentStat.isDirectory()
    || !isWithinRoot(fs.realpathSync(resolvedRoot), fs.realpathSync(sourceParent))
  ) {
    throw new Error(`${label} source parent is not a contained real directory`);
  }
  const sourceIdentity = regularFileIdentityNoFollow(resolvedSource, `${label} source`);
  if (
    sourceIdentity.size !== expectedSourceIdentity.size
    || sourceIdentity.sha256 !== expectedSourceIdentity.sha256
  ) {
    throw new Error(`${label} source bytes changed before replacement`);
  }
  regularFileIdentityNoFollow(resolvedTarget, `${label} target`);
  const rootStat = fs.lstatSync(resolvedRoot, { bigint: true });
  const sourceStat = fs.lstatSync(resolvedSource, { bigint: true });
  const targetStat = fs.lstatSync(resolvedTarget, { bigint: true });
  const relativeSource = path.relative(resolvedRoot, resolvedSource);
  const completed = spawnSync(process.execPath, [
    artifactReplacementHelper,
    relativeSource,
    path.basename(resolvedTarget),
    resolvedRoot,
    rootStat.dev.toString(),
    rootStat.ino.toString(),
    sourceStat.dev.toString(),
    sourceStat.ino.toString(),
    targetStat.dev.toString(),
    targetStat.ino.toString(),
    String(expectedSourceIdentity.size),
    expectedSourceIdentity.sha256,
    label
  ], {
    cwd: resolvedRoot,
    encoding: 'utf8',
    maxBuffer: 1024 * 1024
  });
  if (completed.error) throw completed.error;
  if (completed.status !== 0) {
    throw new Error(
      `${label} failed: ${String(completed.stderr || '').trim() || 'replacement rejected'}`
    );
  }
  const published = regularFileIdentityNoFollow(resolvedTarget, label);
  if (
    published.size !== expectedSourceIdentity.size
    || published.sha256 !== expectedSourceIdentity.sha256
  ) {
    throw new Error(`${label} bytes changed during replacement`);
  }
  return resolvedTarget;
}

function portablePath(relativePath) {
  return relativePath.split(path.sep).join('/');
}

function validateLogicalPath(recordedPath, label) {
  if (typeof recordedPath !== 'string' || !recordedPath || recordedPath.includes('\0')) {
    throw new Error(`${label} is missing or invalid`);
  }
  const normalized = recordedPath.replaceAll('\\', '/');
  if (
    path.isAbsolute(recordedPath)
    || normalized.startsWith('/')
    || normalized.split('/').some(part => part === '..' || part === '' || part === '.')
  ) {
    throw new Error(`${label} must be a normalized logical relative path`);
  }
  return normalized;
}

/**
 * Return a stable artifact-root-relative path and reject paths outside the root.
 */
export function artifactRelativePath(artifactRoot, targetPath, label = 'Artifact path') {
  const resolvedRoot = path.resolve(artifactRoot);
  const resolvedTarget = path.resolve(targetPath);
  if (!isWithinRoot(resolvedRoot, resolvedTarget)) {
    throw new Error(`${label} escapes the artifact root`);
  }
  if (fs.existsSync(resolvedRoot) && fs.existsSync(resolvedTarget)) {
    const realRoot = fs.realpathSync(resolvedRoot);
    const realTarget = fs.realpathSync(resolvedTarget);
    if (!isWithinRoot(realRoot, realTarget) || realTarget === realRoot) {
      throw new Error(`${label} resolves outside the artifact root`);
    }
  }
  const relative = path.relative(resolvedRoot, resolvedTarget);
  if (!relative) throw new Error(`${label} must identify an artifact below the root`);
  return portablePath(relative);
}

/**
 * Resolve a recorded artifact path without allowing lexical or symlink escape.
 *
 * Legacy absolute paths are accepted only when they still point inside the
 * current artifact root. They are never silently rebased after relocation.
 */
export function resolveArtifactPath(
  artifactRoot,
  recordedPath,
  label = 'Artifact path',
  { allowLegacyAbsoluteWithinRoot = false, mustExist = true } = {}
) {
  if (typeof recordedPath !== 'string' || !recordedPath || recordedPath.includes('\0')) {
    throw new Error(`${label} is missing or invalid`);
  }
  const resolvedRoot = path.resolve(artifactRoot);
  if (path.isAbsolute(recordedPath) && !allowLegacyAbsoluteWithinRoot) {
    throw new Error(`${label} must be artifact-root-relative`);
  }
  const resolvedTarget = path.isAbsolute(recordedPath)
    ? path.resolve(recordedPath)
    : path.resolve(resolvedRoot, recordedPath);
  if (!isWithinRoot(resolvedRoot, resolvedTarget) || resolvedTarget === resolvedRoot) {
    throw new Error(`${label} escapes the artifact root`);
  }
  if (mustExist && !fs.existsSync(resolvedTarget)) {
    throw new Error(`${label} does not exist inside the artifact root`);
  }
  if (mustExist) {
    const realRoot = fs.realpathSync(resolvedRoot);
    const realTarget = fs.realpathSync(resolvedTarget);
    if (!isWithinRoot(realRoot, realTarget) || realTarget === realRoot) {
      throw new Error(`${label} resolves outside the artifact root`);
    }
  }
  return resolvedTarget;
}

export function resolveBuildProfileAppPath(
  buildProfilePath,
  { allowLegacy = false } = {}
) {
  const resolvedProfilePath = path.resolve(buildProfilePath);
  const artifactRoot = path.dirname(resolvedProfilePath);
  const { value: profile } = readStableJsonFileNoFollow(
    resolvedProfilePath,
    'Build profile'
  );
  const legacy = profile.pathPolicy == null && allowLegacy === true;
  if (
    profile.schema !== 'automixer.macos_build_profile.v1'
    || (
      profile.pathPolicy !== 'artifact_root_relative_v1'
      && !legacy
    )
  ) {
    throw new Error('Build profile path policy is invalid');
  }
  const appPath = resolveArtifactPath(
    artifactRoot,
    profile.app?.path,
    'Build profile packaged app',
    { allowLegacyAbsoluteWithinRoot: legacy }
  );
  const appStat = fs.lstatSync(appPath);
  if (!appStat.isDirectory() || appStat.isSymbolicLink()) {
    throw new Error('Build profile packaged app must be a regular directory');
  }
  return {
    appPath,
    artifactRoot,
    profile,
    profilePath: resolvedProfilePath
  };
}

export function artifactFileIdentity(artifactRoot, filePath, label = 'Artifact file') {
  const relativePath = artifactRelativePath(artifactRoot, filePath, label);
  return {
    path: relativePath,
    ...regularFileIdentityNoFollow(filePath, label, artifactRoot)
  };
}

export function logicalFileIdentity(filePath, recordedPath, pathBase, label = 'Logical file') {
  if (typeof pathBase !== 'string' || !/^[a-z][a-z0-9_]*$/.test(pathBase)) {
    throw new Error(`${label} path base is invalid`);
  }
  return {
    ...fileIdentity(filePath),
    path: validateLogicalPath(recordedPath, `${label} path`),
    pathBase
  };
}

export function verifyPackagedRuntimeSmokeManifest(
  manifestPath,
  { allowLegacy = false } = {}
) {
  const resolvedManifestPath = path.resolve(manifestPath);
  const manifestRoot = path.dirname(resolvedManifestPath);
  resolveArtifactPath(
    manifestRoot,
    path.basename(resolvedManifestPath),
    'Packaged runtime smoke manifest'
  );
  const stableManifest = readStableJsonFileNoFollow(
    resolvedManifestPath,
    'Packaged runtime smoke manifest'
  );
  const manifest = stableManifest.value;
  const legacyPaths = manifest.pathPolicy == null && allowLegacy === true;
  if (
    manifest.schema !== 'automixer.packaged_runtime_smoke_manifest.v1'
    || (
      manifest.pathPolicy !== 'manifest_directory_relative_v1'
      && !(manifest.pathPolicy == null && allowLegacy === true)
    )
    || manifest.readOnly !== true
    || manifest.contactsWing !== false
    || manifest.liveAuthorityGranted !== false
    || !['local', 'release'].includes(manifest.buildProfile?.profile)
    || manifest.buildProfile?.releaseEligible !== false
    || manifest.buildProfile?.bindingValid !== true
    || manifest.buildProfile?.appTreeMatches !== true
  ) {
    throw new Error('Packaged runtime smoke manifest contract mismatch');
  }

  const identities = {};
  for (const name of ['result.json', 'result.md']) {
    const recorded = manifest.files?.[name];
    const resolved = resolveArtifactPath(
      manifestRoot,
      recorded?.path,
      `Packaged runtime smoke ${name}`,
      { allowLegacyAbsoluteWithinRoot: legacyPaths }
    );
    const expected = path.join(manifestRoot, name);
    if (resolved !== expected) {
      throw new Error(`Packaged runtime smoke ${name} is not the canonical sibling artifact`);
    }
    const actual = artifactFileIdentity(manifestRoot, resolved, `Smoke ${name}`);
    if (!identitiesEqual(recorded, actual)) {
      throw new Error(`Packaged runtime smoke ${name} identity mismatch`);
    }
    identities[name] = actual;
  }

  const stableResult = readStableJsonFileNoFollow(
    path.join(manifestRoot, 'result.json'),
    'Packaged runtime smoke result'
  );
  assertIdentityUnchanged(
    identities['result.json'],
    stableResult.identity,
    'Packaged runtime smoke result stable read'
  );
  const result = stableResult.value;
  if (
    result.schema !== 'automixer.packaged_runtime_smoke.v1'
    || result.pass !== true
    || result.readOnly !== manifest.readOnly
    || result.contactsWing !== manifest.contactsWing
    || result.liveWrites !== false
    || result.liveAuthorityGranted !== manifest.liveAuthorityGranted
    || result.buildProfile?.profile !== manifest.buildProfile?.profile
    || result.buildProfile?.releaseEligible !== manifest.buildProfile?.releaseEligible
    || result.buildProfile?.path !== manifest.buildProfile?.path
    || result.buildProfile?.sha256 !== manifest.buildProfile?.sha256
    || result.buildProfile?.bindingValid !== manifest.buildProfile?.bindingValid
    || result.buildProfile?.appTreeMatches !== manifest.buildProfile?.appTreeMatches
  ) {
    throw new Error('Packaged runtime smoke manifest/result binding mismatch');
  }
  if (manifest.pathPolicy === 'manifest_directory_relative_v1') {
    if (
      path.isAbsolute(manifest.buildProfile?.path || '')
      || manifest.buildProfile?.path !== 'build-profile.json'
      || manifest.buildProfile?.pathBase !== 'packaged_build_artifact_root'
    ) {
      throw new Error('Packaged runtime smoke build-profile path is not portable');
    }
  }
  return {
    manifest,
    result,
    identities,
    manifestIdentity: stableManifest.identity,
    manifestRoot
  };
}

export function identitiesEqual(left, right) {
  return Boolean(
    left
    && right
    && left.size === right.size
    && left.sha256 === right.sha256
  );
}

export function assertIdentityUnchanged(expected, actual, label) {
  if (!identitiesEqual(expected, actual)) {
    throw new Error(`${label} changed during release verification`);
  }
}

export function hashTree(root) {
  const rows = [];
  function visit(candidate, relative) {
    const stat = fs.lstatSync(candidate);
    const mode = (stat.mode & 0o7777).toString(8).padStart(4, '0');
    if (stat.isSymbolicLink()) {
      rows.push(`L\0${relative}\0${mode}\0${fs.readlinkSync(candidate)}`);
      return;
    }
    if (stat.isDirectory()) {
      rows.push(`D\0${relative}\0${mode}`);
      for (const name of fs.readdirSync(candidate).sort()) {
        visit(path.join(candidate, name), relative ? `${relative}/${name}` : name);
      }
      return;
    }
    if (stat.isFile()) {
      rows.push(`F\0${relative}\0${mode}\0${stat.size}\0${sha256File(candidate)}`);
      return;
    }
    throw new Error(`Unsupported file type in tree: ${relative}`);
  }
  visit(root, '');
  return crypto.createHash('sha256').update(rows.join('\n')).digest('hex');
}
