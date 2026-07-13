#!/usr/bin/env node

import fs from 'node:fs';
import path from 'node:path';

function sameObject(stat, dev, ino) {
  return stat.dev.toString() === dev && stat.ino.toString() === ino;
}

const [
  candidate,
  target,
  rootPath,
  parentPath,
  rootDev,
  rootIno,
  parentDev,
  parentIno,
  candidateDev,
  candidateIno,
  label = 'Artifact publication'
] = process.argv.slice(2);

let linked = false;
try {
  for (const name of [candidate, target]) {
    if (!name || name !== path.basename(name) || name.includes('\0')) {
      throw new Error(`${label} basename is invalid`);
    }
  }
  const cwd = fs.statSync('.', { bigint: true });
  const root = fs.lstatSync(rootPath, { bigint: true });
  const parent = fs.lstatSync(parentPath, { bigint: true });
  if (
    !sameObject(cwd, parentDev, parentIno)
    || root.isSymbolicLink()
    || parent.isSymbolicLink()
    || !sameObject(root, rootDev, rootIno)
    || !sameObject(parent, parentDev, parentIno)
  ) {
    throw new Error(`${label} root or parent changed before publication`);
  }
  const candidateStat = fs.lstatSync(candidate, { bigint: true });
  if (
    candidateStat.isSymbolicLink()
    || !candidateStat.isFile()
    || !sameObject(candidateStat, candidateDev, candidateIno)
  ) {
    throw new Error(`${label} candidate changed before publication`);
  }
  if (fs.lstatSync(target, { throwIfNoEntry: false })) {
    throw new Error(`${label} target already exists`);
  }
  fs.linkSync(candidate, target);
  linked = true;
  const published = fs.lstatSync(target, { bigint: true });
  const canonical = fs.lstatSync(path.join(parentPath, target), { bigint: true });
  const finalRoot = fs.lstatSync(rootPath, { bigint: true });
  const finalParent = fs.lstatSync(parentPath, { bigint: true });
  if (
    published.isSymbolicLink()
    || !published.isFile()
    || !sameObject(published, candidateDev, candidateIno)
    || !sameObject(canonical, candidateDev, candidateIno)
    || !sameObject(finalRoot, rootDev, rootIno)
    || !sameObject(finalParent, parentDev, parentIno)
  ) {
    throw new Error(`${label} changed during publication`);
  }
  fs.unlinkSync(candidate);
  const directoryDescriptor = fs.openSync('.', fs.constants.O_RDONLY);
  try {
    fs.fsyncSync(directoryDescriptor);
  } finally {
    fs.closeSync(directoryDescriptor);
  }
} catch (error) {
  if (linked) {
    try {
      const current = fs.lstatSync(target, { bigint: true });
      if (
        !current.isSymbolicLink()
        && current.isFile()
        && sameObject(current, candidateDev, candidateIno)
      ) {
        fs.unlinkSync(target);
      }
    } catch {}
  }
  process.stderr.write(`${error?.message || error}\n`);
  process.exitCode = 1;
}
