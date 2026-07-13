#!/usr/bin/env node

import fs from 'node:fs';
import path from 'node:path';

function sameObject(stat, dev, ino) {
  return stat.dev.toString() === dev && stat.ino.toString() === ino;
}

const [
  target,
  rootPath,
  parentPath,
  rootDev,
  rootIno,
  parentDev,
  parentIno,
  targetDev,
  targetIno,
  label = 'Artifact removal'
] = process.argv.slice(2);

try {
  if (!target || target !== path.basename(target) || target.includes('\0')) {
    throw new Error(`${label} basename is invalid`);
  }
  const cwd = fs.statSync('.', { bigint: true });
  const root = fs.lstatSync(rootPath, { bigint: true });
  const parent = fs.lstatSync(parentPath, { bigint: true });
  const artifact = fs.lstatSync(target, { bigint: true });
  if (
    !sameObject(cwd, parentDev, parentIno)
    || root.isSymbolicLink()
    || parent.isSymbolicLink()
    || artifact.isSymbolicLink()
    || !artifact.isFile()
    || !sameObject(root, rootDev, rootIno)
    || !sameObject(parent, parentDev, parentIno)
    || !sameObject(artifact, targetDev, targetIno)
  ) {
    throw new Error(`${label} identity changed before removal`);
  }
  fs.unlinkSync(target);
  const directoryDescriptor = fs.openSync('.', fs.constants.O_RDONLY);
  try {
    fs.fsyncSync(directoryDescriptor);
  } finally {
    fs.closeSync(directoryDescriptor);
  }
  const finalRoot = fs.lstatSync(rootPath, { bigint: true });
  const finalParent = fs.lstatSync(parentPath, { bigint: true });
  if (
    !sameObject(finalRoot, rootDev, rootIno)
    || !sameObject(finalParent, parentDev, parentIno)
  ) {
    throw new Error(`${label} root changed during removal`);
  }
} catch (error) {
  process.stderr.write(`${error?.message || error}\n`);
  process.exitCode = 1;
}
