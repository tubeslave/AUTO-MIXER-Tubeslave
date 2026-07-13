#!/usr/bin/env node

import crypto from 'node:crypto';
import fs from 'node:fs';
import path from 'node:path';

function sameObject(stat, dev, ino) {
  return stat.dev.toString() === dev && stat.ino.toString() === ino;
}

const [
  source,
  target,
  rootPath,
  rootDev,
  rootIno,
  sourceDev,
  sourceIno,
  targetDev,
  targetIno,
  expectedSourceSize,
  expectedSourceSha256,
  label = 'Artifact replacement'
] = process.argv.slice(2);

function stableFileIdentity(filePath) {
  const descriptor = fs.openSync(
    filePath,
    fs.constants.O_RDONLY | (fs.constants.O_NOFOLLOW || 0)
  );
  const hashPass = () => {
    const hash = crypto.createHash('sha256');
    const buffer = Buffer.allocUnsafe(1024 * 1024);
    let position = 0;
    while (true) {
      const count = fs.readSync(descriptor, buffer, 0, buffer.length, position);
      if (!count) break;
      hash.update(buffer.subarray(0, count));
      position += count;
    }
    return hash.digest('hex');
  };
  const same = (left, right) => Boolean(
    left.dev === right.dev
    && left.ino === right.ino
    && left.size === right.size
    && left.mode === right.mode
    && left.nlink === right.nlink
    && left.mtimeNs === right.mtimeNs
    && left.ctimeNs === right.ctimeNs
  );
  try {
    const before = fs.fstatSync(descriptor, { bigint: true });
    const first = hashPass();
    const middle = fs.fstatSync(descriptor, { bigint: true });
    const second = hashPass();
    const after = fs.fstatSync(descriptor, { bigint: true });
    const pathname = fs.lstatSync(filePath, { bigint: true });
    if (
      !before.isFile()
      || pathname.isSymbolicLink()
      || !pathname.isFile()
      || !same(before, middle)
      || !same(middle, after)
      || !same(pathname, after)
      || first !== second
    ) {
      throw new Error(`${label} source changed during stable read`);
    }
    return { size: after.size.toString(), sha256: second };
  } finally {
    fs.closeSync(descriptor);
  }
}

const backup = `.replace-backup-${crypto.randomUUID()}`;
let backupCreated = false;
let replaced = false;
try {
  if (
    !source
    || source !== path.basename(source)
    || !target
    || target !== path.basename(target)
  ) {
    throw new Error(`${label} paths are invalid`);
  }
  const cwd = fs.statSync('.', { bigint: true });
  const root = fs.lstatSync(rootPath, { bigint: true });
  const sourceStat = fs.lstatSync(source, { bigint: true });
  const targetStat = fs.lstatSync(target, { bigint: true });
  if (
    root.isSymbolicLink()
    || sourceStat.isSymbolicLink()
    || targetStat.isSymbolicLink()
    || !root.isDirectory()
    || !sourceStat.isFile()
    || !targetStat.isFile()
    || !sameObject(cwd, rootDev, rootIno)
    || !sameObject(root, rootDev, rootIno)
    || !sameObject(sourceStat, sourceDev, sourceIno)
    || !sameObject(targetStat, targetDev, targetIno)
  ) {
    throw new Error(`${label} identity changed before replacement`);
  }
  const stableSource = stableFileIdentity(source);
  if (
    stableSource.size !== expectedSourceSize
    || stableSource.sha256 !== expectedSourceSha256
  ) {
    throw new Error(`${label} source bytes changed before replacement`);
  }
  fs.linkSync(target, backup);
  backupCreated = true;
  fs.renameSync(source, target);
  replaced = true;
  const published = fs.lstatSync(target, { bigint: true });
  const finalRoot = fs.lstatSync(rootPath, { bigint: true });
  const stablePublished = stableFileIdentity(target);
  if (
    published.isSymbolicLink()
    || !published.isFile()
    || !sameObject(published, sourceDev, sourceIno)
    || !sameObject(finalRoot, rootDev, rootIno)
    || stablePublished.size !== expectedSourceSize
    || stablePublished.sha256 !== expectedSourceSha256
  ) {
    throw new Error(`${label} changed during replacement`);
  }
  fs.unlinkSync(backup);
  backupCreated = false;
  const directoryDescriptor = fs.openSync('.', fs.constants.O_RDONLY);
  try {
    fs.fsyncSync(directoryDescriptor);
  } finally {
    fs.closeSync(directoryDescriptor);
  }
} catch (error) {
  if (replaced && backupCreated) {
    try {
      const current = fs.lstatSync(target, { bigint: true });
      if (!current.isSymbolicLink() && current.isFile() && sameObject(current, sourceDev, sourceIno)) {
        try { fs.linkSync(target, source); } catch {}
      }
      fs.renameSync(backup, target);
      backupCreated = false;
    } catch {}
  }
  if (backupCreated) {
    try {
      const current = fs.lstatSync(backup, { bigint: true });
      if (!current.isSymbolicLink() && current.isFile() && sameObject(current, targetDev, targetIno)) {
        fs.unlinkSync(backup);
      }
    } catch {}
  }
  process.stderr.write(`${error?.message || error}\n`);
  process.exitCode = 1;
}
