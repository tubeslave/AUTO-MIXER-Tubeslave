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
const stableTreeIdentityHelper = fileURLToPath(
  new URL('./stable-tree-identity-helper.mjs', import.meta.url)
);

const helperSourcesByPath = new Map([
  ['frontend/scripts/write-new-artifact-helper.mjs', artifactWriterHelper],
  ['frontend/scripts/publish-new-artifact-helper.mjs', artifactPublisherHelper],
  ['frontend/scripts/remove-artifact-helper.mjs', artifactRemovalHelper],
  ['frontend/scripts/replace-artifact-helper.mjs', artifactReplacementHelper],
  ['frontend/scripts/stable-file-identity-helper.mjs', stableFileIdentityHelper],
  ['frontend/scripts/stable-json-reader-helper.mjs', stableJsonReaderHelper],
  ['frontend/scripts/stable-tree-identity-helper.mjs', stableTreeIdentityHelper]
]);
const helperSourcePathByExecutable = new Map(
  [...helperSourcesByPath.entries()].map(([sourcePath, executablePath]) => [
    path.resolve(executablePath),
    sourcePath
  ])
);
export const RELEASE_EVIDENCE_HELPER_SOURCE_PATHS = Object.freeze(
  [...helperSourcesByPath.keys()].sort()
);
let pinnedHelperSources = null;
let pinnedHelperRuntime = null;

function fixedEvidenceGitEnvironment() {
  return {
    HOME: '/var/empty',
    TMPDIR: '/tmp',
    LANG: 'C',
    LC_ALL: 'C',
    PATH: '/usr/bin:/bin:/usr/sbin:/sbin',
    GIT_ATTR_NOSYSTEM: '1',
    GIT_CONFIG_GLOBAL: '/dev/null',
    GIT_CONFIG_NOSYSTEM: '1',
    GIT_NO_LAZY_FETCH: '1',
    GIT_NO_REPLACE_OBJECTS: '1',
    GIT_OPTIONAL_LOCKS: '0',
    GIT_TERMINAL_PROMPT: '0'
  };
}

function assertSafeEvidenceGitConfiguration(repoRoot) {
  const completed = spawnSync('/usr/bin/git', [
    'config', '--null', '--list', '--show-origin', '--includes'
  ], {
    cwd: repoRoot,
    encoding: null,
    env: fixedEvidenceGitEnvironment(),
    maxBuffer: 4 * 1024 * 1024
  });
  if (completed.error) throw completed.error;
  if (completed.status !== 0) {
    throw new Error(
      `Git configuration audit failed: ${Buffer.from(completed.stderr || '')
        .toString('utf8').trim() || 'unknown error'}`
    );
  }
  const decoded = new TextDecoder('utf-8', { fatal: true }).decode(
    Buffer.from(completed.stdout || '')
  );
  const fields = decoded.split('\0');
  if (fields.pop() !== '' || fields.length % 2 !== 0) {
    throw new Error('Git configuration audit fields are malformed');
  }
  const safeExactKeys = new Set([
    'commit.gpgsign',
    'core.bare',
    'core.filemode',
    'core.ignorecase',
    'core.logallrefupdates',
    'core.precomposeunicode',
    'core.repositoryformatversion',
    'extensions.objectformat',
    'extensions.worktreeconfig',
    'gpg.format',
    'lfs.repositoryformatversion',
    'user.email',
    'user.name',
    'user.signingkey'
  ]);
  const safeKeyPatterns = [
    /^branch\..+\.(?:merge|remote|vscode-merge-base)$/,
    /^remote\..+\.(?:fetch|url)$/
  ];
  for (let index = 0; index < fields.length; index += 2) {
    const origin = fields[index];
    const keyValue = fields[index + 1];
    const separator = keyValue.indexOf('\n');
    if (separator <= 0 || !origin) {
      throw new Error('Git configuration audit entry is malformed');
    }
    const key = keyValue.slice(0, separator).toLowerCase();
    const value = keyValue.slice(separator + 1);
    if (!safeExactKeys.has(key) && !safeKeyPatterns.some(pattern => pattern.test(key))) {
      throw new Error(`Repository Git configuration is unsafe for evidence reads: ${key}`);
    }
    if (
      (key === 'core.bare' && value !== 'false')
      || (key === 'core.filemode' && value !== 'true')
      || (key === 'core.repositoryformatversion' && value !== '0')
      || (key === 'extensions.worktreeconfig' && value !== 'true')
    ) {
      throw new Error(`Repository Git configuration has an unsafe value: ${key}`);
    }
  }
}

export function runReleaseEvidenceGit(
  repoRoot,
  args,
  { maxBuffer = 16 * 1024 * 1024 } = {}
) {
  const resolvedRepository = path.resolve(repoRoot);
  assertSafeEvidenceGitConfiguration(resolvedRepository);
  const completed = spawnSync('/usr/bin/git', [
    '-c', `core.worktree=${resolvedRepository}`,
    '-c', 'core.fsmonitor=false',
    '-c', 'core.hooksPath=/dev/null',
    '-c', 'core.untrackedCache=false',
    '-c', 'core.attributesFile=/dev/null',
    '-c', 'core.excludesFile=/dev/null',
    '-c', 'core.filemode=true',
    '-c', 'core.ignorecase=false',
    '-c', 'core.precomposeunicode=false',
    ...args
  ], {
    cwd: resolvedRepository,
    encoding: null,
    env: fixedEvidenceGitEnvironment(),
    maxBuffer
  });
  if (completed.error) throw completed.error;
  if (completed.status !== 0) {
    throw new Error(
      `git ${args[0]} failed: `
      + `${Buffer.from(completed.stderr || '').toString('utf8').trim() || 'unknown error'}`
    );
  }
  return Buffer.from(completed.stdout || '');
}

function helperRuntimeIdentity() {
  const runtimePath = path.resolve(process.execPath || '');
  const before = fs.lstatSync(runtimePath, { bigint: true });
  const realPath = fs.realpathSync(runtimePath);
  const after = fs.lstatSync(runtimePath, { bigint: true });
  const identity = stat => ({
    dev: stat.dev.toString(),
    ino: stat.ino.toString(),
    mode: stat.mode.toString(),
    nlink: stat.nlink.toString(),
    size: stat.size.toString(),
    mtimeNs: stat.mtimeNs.toString(),
    ctimeNs: stat.ctimeNs.toString()
  });
  const beforeIdentity = identity(before);
  const afterIdentity = identity(after);
  if (
    before.isSymbolicLink()
    || !before.isFile()
    || before.nlink !== 1n
    || JSON.stringify(beforeIdentity) !== JSON.stringify(afterIdentity)
  ) {
    throw new Error('Release-evidence helper runtime must be a stable single-link real file');
  }
  return { path: runtimePath, realPath, identity: afterIdentity };
}

function assertPinnedHelperRuntime(stage) {
  const current = helperRuntimeIdentity();
  if (
    !pinnedHelperRuntime
    || JSON.stringify(current) !== JSON.stringify(pinnedHelperRuntime)
  ) {
    throw new Error(`Release-evidence helper runtime changed ${stage}`);
  }
  return current.path;
}

export function pinReleaseEvidenceHelperRuntime() {
  const currentRuntime = helperRuntimeIdentity();
  if (pinnedHelperRuntime) {
    if (JSON.stringify(currentRuntime) !== JSON.stringify(pinnedHelperRuntime)) {
      throw new Error('Release-evidence helper runtime cannot change within one process');
    }
  } else {
    pinnedHelperRuntime = currentRuntime;
  }
  return Object.freeze({
    path: pinnedHelperRuntime.path,
    realPath: pinnedHelperRuntime.realPath,
    ...pinnedHelperRuntime.identity
  });
}

export function pinReleaseEvidenceHelperSources(sources) {
  if (!sources || typeof sources !== 'object' || Array.isArray(sources)) {
    throw new Error('Pinned release-evidence helper sources are required');
  }
  const suppliedPaths = Object.keys(sources).sort();
  if (
    suppliedPaths.length !== RELEASE_EVIDENCE_HELPER_SOURCE_PATHS.length
    || suppliedPaths.some((value, index) => (
      value !== RELEASE_EVIDENCE_HELPER_SOURCE_PATHS[index]
    ))
  ) {
    throw new Error('Pinned release-evidence helper source set is not exact');
  }
  const next = new Map();
  for (const sourcePath of RELEASE_EVIDENCE_HELPER_SOURCE_PATHS) {
    if (!Buffer.isBuffer(sources[sourcePath]) || sources[sourcePath].length === 0) {
      throw new Error(`Pinned release-evidence helper bytes are invalid: ${sourcePath}`);
    }
    const bytes = Buffer.from(sources[sourcePath]);
    const source = new TextDecoder('utf-8', { fatal: true }).decode(bytes);
    if (source.includes('\0')) {
      throw new Error(`Pinned release-evidence helper contains NUL: ${sourcePath}`);
    }
    next.set(sourcePath, { bytes, source });
  }
  if (pinnedHelperSources) {
    for (const sourcePath of RELEASE_EVIDENCE_HELPER_SOURCE_PATHS) {
      if (!pinnedHelperSources.get(sourcePath).bytes.equals(next.get(sourcePath).bytes)) {
        throw new Error('Release-evidence helper authority cannot change within one process');
      }
    }
  } else {
    pinnedHelperSources = next;
  }
  pinReleaseEvidenceHelperRuntime();
  return Object.freeze(Object.fromEntries(
    RELEASE_EVIDENCE_HELPER_SOURCE_PATHS.map(sourcePath => [
      sourcePath,
      Object.freeze({
        size: pinnedHelperSources.get(sourcePath).bytes.length,
        sha256: crypto.createHash('sha256')
          .update(pinnedHelperSources.get(sourcePath).bytes)
          .digest('hex')
      })
    ])
  ));
}

function spawnReleaseEvidenceHelper(executablePath, args, options) {
  const sourcePath = helperSourcePathByExecutable.get(path.resolve(executablePath));
  if (!sourcePath) throw new Error('Unknown release-evidence helper executable');
  const pinned = pinnedHelperSources?.get(sourcePath);
  if (!pinned) {
    return spawnSync(process.execPath, [executablePath, ...args], options);
  }
  const runtimePath = assertPinnedHelperRuntime('before helper spawn');
  const completed = spawnSync(runtimePath, [
    '--input-type=module',
    '--eval',
    pinned.source,
    'automixer-pinned-release-evidence-helper',
    ...args
  ], options);
  assertPinnedHelperRuntime('during helper spawn');
  return completed;
}

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
  const completed = spawnReleaseEvidenceHelper(
    stableFileIdentityHelper,
    pinnedHelperArguments(context, label),
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

function stableStatIdentity(stat) {
  return Object.fromEntries([
    'dev',
    'ino',
    'size',
    'mode',
    'nlink',
    'mtimeNs',
    'ctimeNs'
  ].map(name => [name, stat[name].toString()]));
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
  const completed = spawnReleaseEvidenceHelper(
    stableJsonReaderHelper,
    pinnedHelperArguments(context, label),
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
  const completed = spawnReleaseEvidenceHelper(artifactWriterHelper, [
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
  const completed = spawnReleaseEvidenceHelper(artifactPublisherHelper, [
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
  const completed = spawnReleaseEvidenceHelper(artifactRemovalHelper, [
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
  const completed = spawnReleaseEvidenceHelper(artifactReplacementHelper, [
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
  const stableProfile = readStableJsonFileNoFollow(
    resolvedProfilePath,
    'Build profile'
  );
  const profile = stableProfile.value;
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
    profileIdentity: stableProfile.identity,
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
    || (
      !legacyPaths
      && (
        !Number.isSafeInteger(manifest.buildProfile?.size)
        || manifest.buildProfile.size < 0
      )
    )
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
    || (
      !legacyPaths
      && result.buildProfile?.size !== manifest.buildProfile?.size
    )
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

/**
 * Hash regular files through pinned no-follow descriptors, guard directory and
 * symlink identities, and require two identical complete walks. This detached
 * evidence reader never executes anything in the tree.
 */
function runStableTreeIdentityHelper(
  root,
  label,
  { expectedRoot = null, outputMode = 'summary' } = {}
) {
  const resolvedRoot = path.resolve(root);
  const rootStat = fs.lstatSync(resolvedRoot, { bigint: true });
  const rootIdentity = stableStatIdentity(rootStat);
  const expectedIdentity = expectedRoot?.stat || rootIdentity;
  const realRoot = expectedRoot?.realPath || fs.realpathSync(resolvedRoot);
  if (
    rootStat.isSymbolicLink()
    || !rootStat.isDirectory()
    || JSON.stringify(rootIdentity) !== JSON.stringify(expectedIdentity)
    || fs.realpathSync(resolvedRoot) !== realRoot
    || !['summary', 'flat-files'].includes(outputMode)
  ) {
    throw new Error(`${label} root must remain the selected real directory`);
  }
  const completed = spawnReleaseEvidenceHelper(stableTreeIdentityHelper, [
    outputMode,
    rootStat.dev.toString(),
    rootStat.ino.toString(),
    realRoot,
    label
  ], {
    // The child process cwd is an OS-pinned reference to the selected root.
    // Walking `.` prevents an ancestor rename from redirecting both guarded
    // walks to a replacement tree at the former absolute pathname.
    cwd: resolvedRoot,
    encoding: 'utf8',
    maxBuffer: 1024 * 1024
  });
  if (completed.error) throw completed.error;
  if (completed.status !== 0) {
    throw new Error(
      `${label} stable tree read failed: `
      + `${String(completed.stderr || '').trim() || 'tree reader rejected input'}`
    );
  }
  const identity = JSON.parse(String(completed.stdout || '').trim());
  const finalRootStat = fs.lstatSync(resolvedRoot, { bigint: true });
  if (
    JSON.stringify(stableStatIdentity(finalRootStat)) !== JSON.stringify(expectedIdentity)
    || fs.realpathSync(resolvedRoot) !== realRoot
    || !/^[0-9a-f]{64}$/.test(String(identity.sha256 || ''))
    || !/^[0-9a-f]{64}$/.test(String(identity.guardSha256 || ''))
    || !Number.isSafeInteger(identity.entryCount)
    || identity.entryCount < 1
  ) {
    throw new Error(`${label} changed around stable tree helper execution`);
  }
  return identity;
}

export function stableTreeIdentityNoFollow(root, label = 'Stable tree') {
  const identity = runStableTreeIdentityHelper(root, label);
  return {
    sha256: identity.sha256,
    guardSha256: identity.guardSha256,
    entryCount: identity.entryCount
  };
}

export function stableFlatDirectoryFileIdentitiesNoFollow(
  root,
  expectedRoot,
  expectedNames,
  label = 'Stable flat directory'
) {
  if (
    !expectedRoot
    || typeof expectedRoot !== 'object'
    || !expectedRoot.stat
    || typeof expectedRoot.realPath !== 'string'
    || !Array.isArray(expectedNames)
    || expectedNames.length === 0
    || new Set(expectedNames).size !== expectedNames.length
    || expectedNames.some(name => path.basename(name) !== name || /[\r\n\0]/.test(name))
  ) {
    throw new Error(`${label} expected root and canonical basenames are required`);
  }
  const canonicalNames = [...expectedNames].sort();
  const identity = runStableTreeIdentityHelper(root, label, {
    expectedRoot,
    outputMode: 'flat-files'
  });
  const files = Array.isArray(identity.files) ? identity.files : [];
  if (
    identity.entryCount !== canonicalNames.length + 1
    || files.length !== canonicalNames.length
    || JSON.stringify(files.map(file => file.relative)) !== JSON.stringify(canonicalNames)
    || files.some(file => (
      !Number.isSafeInteger(file.size)
      || file.size < 0
      || !/^[0-9a-f]{64}$/.test(String(file.sha256 || ''))
      || !/^\d+$/.test(String(file.mode || ''))
      || !/^\d+$/.test(String(file.nlink || ''))
      || (BigInt(file.mode) & 0o777n) !== 0o600n
      || BigInt(file.nlink) !== 1n
    ))
  ) {
    throw new Error(`${label} is not the exact canonical private 0600 file set`);
  }
  return Object.fromEntries(files.map(file => [file.relative, {
    size: file.size,
    sha256: file.sha256
  }]));
}
