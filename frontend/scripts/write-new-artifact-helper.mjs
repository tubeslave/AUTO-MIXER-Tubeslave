#!/usr/bin/env node

import fs from 'node:fs';
import path from 'node:path';

function isWithinRoot(rootPath, candidatePath) {
  const relative = path.relative(rootPath, candidatePath);
  return relative === '' || (
    !path.isAbsolute(relative)
    && relative !== '..'
    && !relative.startsWith(`..${path.sep}`)
  );
}

function sameObject(stat, expectedDev, expectedIno) {
  return stat.dev.toString() === expectedDev && stat.ino.toString() === expectedIno;
}

const [
  basename,
  rootPath,
  parentPath,
  expectedRootDev,
  expectedRootIno,
  expectedParentDev,
  expectedParentIno,
  label = 'Artifact output'
] = process.argv.slice(2);

let descriptor;
let created = false;
let opened;
try {
  if (!basename || basename !== path.basename(basename) || basename.includes('\0')) {
    throw new Error(`${label} basename is invalid`);
  }
  const cwdStat = fs.statSync('.', { bigint: true });
  if (!cwdStat.isDirectory() || !sameObject(cwdStat, expectedParentDev, expectedParentIno)) {
    throw new Error(`${label} parent changed before helper startup`);
  }
  const rootStat = fs.lstatSync(rootPath, { bigint: true });
  const parentStat = fs.lstatSync(parentPath, { bigint: true });
  if (
    rootStat.isSymbolicLink()
    || parentStat.isSymbolicLink()
    || !rootStat.isDirectory()
    || !parentStat.isDirectory()
    || !sameObject(rootStat, expectedRootDev, expectedRootIno)
    || !sameObject(parentStat, expectedParentDev, expectedParentIno)
  ) {
    throw new Error(`${label} root or parent changed before creation`);
  }
  const realRoot = fs.realpathSync(rootPath);
  const realParent = fs.realpathSync(parentPath);
  if (!isWithinRoot(realRoot, realParent)) {
    throw new Error(`${label} parent resolves outside the artifact root`);
  }

  descriptor = fs.openSync(
    basename,
    fs.constants.O_WRONLY
      | fs.constants.O_CREAT
      | fs.constants.O_EXCL
      | (fs.constants.O_NOFOLLOW || 0),
    0o600
  );
  created = true;
  opened = fs.fstatSync(descriptor, { bigint: true });
  if (!opened.isFile()) throw new Error(`${label} is not a regular file`);
  fs.writeFileSync(descriptor, fs.readFileSync(0));
  fs.fsyncSync(descriptor);
  opened = fs.fstatSync(descriptor, { bigint: true });

  const finalCwd = fs.statSync('.', { bigint: true });
  const finalRoot = fs.lstatSync(rootPath, { bigint: true });
  const finalParent = fs.lstatSync(parentPath, { bigint: true });
  const relativeTarget = fs.lstatSync(basename, { bigint: true });
  const canonicalTarget = fs.lstatSync(path.join(parentPath, basename), { bigint: true });
  if (
    !sameObject(finalCwd, expectedParentDev, expectedParentIno)
    || finalRoot.isSymbolicLink()
    || finalParent.isSymbolicLink()
    || !sameObject(finalRoot, expectedRootDev, expectedRootIno)
    || !sameObject(finalParent, expectedParentDev, expectedParentIno)
    || relativeTarget.isSymbolicLink()
    || canonicalTarget.isSymbolicLink()
    || !relativeTarget.isFile()
    || !canonicalTarget.isFile()
    || relativeTarget.dev !== opened.dev
    || relativeTarget.ino !== opened.ino
    || canonicalTarget.dev !== opened.dev
    || canonicalTarget.ino !== opened.ino
    || !isWithinRoot(fs.realpathSync(rootPath), fs.realpathSync(parentPath))
  ) {
    throw new Error(`${label} pathname changed during creation`);
  }
  const directoryDescriptor = fs.openSync('.', fs.constants.O_RDONLY);
  try {
    fs.fsyncSync(directoryDescriptor);
  } finally {
    fs.closeSync(directoryDescriptor);
  }
  process.stdout.write(`${JSON.stringify({
    dev: opened.dev.toString(),
    ino: opened.ino.toString(),
    size: opened.size.toString()
  })}\n`);
} catch (error) {
  if (created && opened) {
    try {
      const current = fs.lstatSync(basename, { bigint: true });
      if (
        !current.isSymbolicLink()
        && current.isFile()
        && current.dev === opened.dev
        && current.ino === opened.ino
      ) {
        fs.unlinkSync(basename);
      }
    } catch {}
  }
  process.stderr.write(`${error?.message || error}\n`);
  process.exitCode = 1;
} finally {
  if (descriptor !== undefined) fs.closeSync(descriptor);
}
