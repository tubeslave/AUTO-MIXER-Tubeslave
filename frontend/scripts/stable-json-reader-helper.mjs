#!/usr/bin/env node

import fs from 'node:fs';
import path from 'node:path';

function sameObject(stat, expectedDev, expectedIno) {
  return stat.dev.toString() === expectedDev && stat.ino.toString() === expectedIno;
}

function stableStatEqual(left, right) {
  return Boolean(
    left.dev === right.dev
    && left.ino === right.ino
    && left.size === right.size
    && left.mode === right.mode
    && left.nlink === right.nlink
    && left.mtimeNs === right.mtimeNs
    && left.ctimeNs === right.ctimeNs
  );
}

function readDescriptorBytes(descriptor, label) {
  const stat = fs.fstatSync(descriptor, { bigint: true });
  const size = Number(stat.size);
  if (!Number.isSafeInteger(size) || size > 64 * 1024 * 1024) {
    throw new Error(`${label} exceeds the stable JSON read limit`);
  }
  const bytes = Buffer.alloc(size);
  let position = 0;
  while (position < size) {
    const count = fs.readSync(
      descriptor,
      bytes,
      position,
      size - position,
      position
    );
    if (!count) throw new Error(`${label} ended during stable read`);
    position += count;
  }
  return bytes;
}

const [
  basename,
  rootPath,
  parentPath,
  expectedRootDev,
  expectedRootIno,
  expectedParentDev,
  expectedParentIno,
  expectedRealRoot,
  expectedRealParent,
  label = 'Stable JSON file'
] = process.argv.slice(2);

let descriptor;
try {
  if (!basename || basename !== path.basename(basename) || basename.includes('\0')) {
    throw new Error(`${label} basename is invalid`);
  }
  const cwd = fs.statSync('.', { bigint: true });
  const root = fs.lstatSync(rootPath, { bigint: true });
  const parent = fs.lstatSync(parentPath, { bigint: true });
  if (
    root.isSymbolicLink()
    || parent.isSymbolicLink()
    || !root.isDirectory()
    || !parent.isDirectory()
    || !sameObject(cwd, expectedParentDev, expectedParentIno)
    || !sameObject(root, expectedRootDev, expectedRootIno)
    || !sameObject(parent, expectedParentDev, expectedParentIno)
    || fs.realpathSync(rootPath) !== expectedRealRoot
    || fs.realpathSync(parentPath) !== expectedRealParent
  ) {
    throw new Error(`${label} root or parent changed before pinned read`);
  }

  descriptor = fs.openSync(
    basename,
    fs.constants.O_RDONLY | (fs.constants.O_NOFOLLOW || 0)
  );
  const before = fs.fstatSync(descriptor, { bigint: true });
  if (!before.isFile()) throw new Error(`${label} must be a regular file`);
  const first = readDescriptorBytes(descriptor, label);
  const middle = fs.fstatSync(descriptor, { bigint: true });
  const second = readDescriptorBytes(descriptor, label);
  const after = fs.fstatSync(descriptor, { bigint: true });
  const relativePath = fs.lstatSync(basename, { bigint: true });
  const canonicalPath = fs.lstatSync(path.join(parentPath, basename), { bigint: true });
  const finalCwd = fs.statSync('.', { bigint: true });
  const finalRoot = fs.lstatSync(rootPath, { bigint: true });
  const finalParent = fs.lstatSync(parentPath, { bigint: true });
  if (
    relativePath.isSymbolicLink()
    || canonicalPath.isSymbolicLink()
    || !relativePath.isFile()
    || !canonicalPath.isFile()
    || !stableStatEqual(before, middle)
    || !stableStatEqual(middle, after)
    || !stableStatEqual(relativePath, after)
    || !stableStatEqual(canonicalPath, after)
    || !sameObject(finalCwd, expectedParentDev, expectedParentIno)
    || !sameObject(finalRoot, expectedRootDev, expectedRootIno)
    || !sameObject(finalParent, expectedParentDev, expectedParentIno)
    || fs.realpathSync(rootPath) !== expectedRealRoot
    || fs.realpathSync(parentPath) !== expectedRealParent
    || !first.equals(second)
  ) {
    throw new Error(`${label} changed during pinned read`);
  }
  JSON.parse(second.toString('utf8'));
  process.stdout.write(second.toString('base64'));
} catch (error) {
  process.stderr.write(`${error?.message || error}\n`);
  process.exitCode = 1;
} finally {
  if (descriptor !== undefined) fs.closeSync(descriptor);
}
