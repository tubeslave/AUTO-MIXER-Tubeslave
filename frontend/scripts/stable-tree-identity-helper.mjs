#!/usr/bin/env node

import crypto from 'node:crypto';
import fs from 'node:fs';
import path from 'node:path';
import process from 'node:process';

const [
  outputMode,
  expectedDev,
  expectedIno,
  expectedRealRoot,
  label
] = process.argv.slice(2);

function fail(message) {
  throw new Error(`${label || 'Stable tree'}: ${message}`);
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
    const bytesRead = fs.readSync(descriptor, buffer, 0, buffer.length, position);
    if (!bytesRead) break;
    hash.update(buffer.subarray(0, bytesRead));
    position += bytesRead;
  }
  return hash.digest('hex');
}

function guard(relative, kind, stat, target = null) {
  return {
    relative,
    kind,
    dev: stat.dev.toString(),
    ino: stat.ino.toString(),
    mode: stat.mode.toString(),
    nlink: stat.nlink.toString(),
    size: stat.size.toString(),
    mtimeNs: stat.mtimeNs.toString(),
    ctimeNs: stat.ctimeNs.toString(),
    ...(target === null ? {} : { target })
  };
}

if (
  !['summary', 'flat-files'].includes(outputMode)
  || !/^\d+$/.test(expectedDev || '')
  || !/^\d+$/.test(expectedIno || '')
  || !path.isAbsolute(expectedRealRoot || '')
) {
  fail('invalid pinned-root arguments');
}

// The launcher sets cwd directly to the selected tree. The kernel retains that
// directory even if an ancestor pathname is renamed, so all relative reads
// stay rooted in the originally selected inode.
const rootPath = '.';
const initialRoot = fs.lstatSync(rootPath, { bigint: true });
const rootDescriptor = fs.openSync(
  rootPath,
  fs.constants.O_RDONLY
    | (fs.constants.O_NOFOLLOW || 0)
    | (fs.constants.O_DIRECTORY || 0)
);
const pinnedRoot = fs.fstatSync(rootDescriptor, { bigint: true });
if (
  initialRoot.isSymbolicLink()
  || !initialRoot.isDirectory()
  || !pinnedRoot.isDirectory()
  || !stableStatEqual(initialRoot, pinnedRoot)
  || initialRoot.dev.toString() !== expectedDev
  || initialRoot.ino.toString() !== expectedIno
  || fs.realpathSync(rootPath) !== expectedRealRoot
) {
  fs.closeSync(rootDescriptor);
  fail('root identity mismatch');
}

function walk() {
  const rows = [];
  const guards = [];
  const files = [];

  function visit(candidate, relative) {
    const before = fs.lstatSync(candidate, { bigint: true });
    const mode = (before.mode & 0o7777n).toString(8).padStart(4, '0');
    if (before.isSymbolicLink()) {
      const targetBefore = fs.readlinkSync(candidate);
      if (/[\r\n\0]/.test(targetBefore)) {
        fail(`symlink target contains a row delimiter: ${relative}`);
      }
      const targetAfter = fs.readlinkSync(candidate);
      const after = fs.lstatSync(candidate, { bigint: true });
      if (!stableStatEqual(before, after) || targetBefore !== targetAfter) {
        fail(`symlink changed during walk: ${relative}`);
      }
      rows.push(`L\0${relative}\0${mode}\0${targetBefore}`);
      guards.push(guard(relative, 'symlink', after, targetAfter));
      return;
    }
    if (before.isDirectory()) {
      const expectedRealPath = relative
        ? path.join(expectedRealRoot, ...relative.split('/'))
        : expectedRealRoot;
      if (fs.realpathSync(candidate) !== expectedRealPath) {
        fail(`directory resolves through a symlink: ${relative}`);
      }
      const namesBefore = fs.readdirSync(candidate).sort();
      if (namesBefore.some(name => /[\r\n\0]/.test(name))) {
        fail(`entry name contains a row delimiter: ${relative}`);
      }
      rows.push(`D\0${relative}\0${mode}`);
      for (const name of namesBefore) {
        visit(path.join(candidate, name), relative ? `${relative}/${name}` : name);
      }
      const namesAfter = fs.readdirSync(candidate).sort();
      const after = fs.lstatSync(candidate, { bigint: true });
      if (
        !stableStatEqual(before, after)
        || JSON.stringify(namesBefore) !== JSON.stringify(namesAfter)
      ) {
        fail(`directory changed during walk: ${relative}`);
      }
      guards.push(guard(relative, 'directory', after));
      return;
    }
    if (before.isFile()) {
      const descriptor = fs.openSync(
        candidate,
        fs.constants.O_RDONLY | (fs.constants.O_NOFOLLOW || 0)
      );
      let opened;
      let sha256;
      let closedOver;
      try {
        opened = fs.fstatSync(descriptor, { bigint: true });
        if (!opened.isFile() || !stableStatEqual(before, opened)) {
          fail(`file identity changed before read: ${relative}`);
        }
        sha256 = hashDescriptor(descriptor);
        closedOver = fs.fstatSync(descriptor, { bigint: true });
      } finally {
        fs.closeSync(descriptor);
      }
      const after = fs.lstatSync(candidate, { bigint: true });
      if (
        !stableStatEqual(opened, closedOver)
        || !stableStatEqual(before, after)
      ) {
        fail(`file changed during read: ${relative}`);
      }
      rows.push(`F\0${relative}\0${mode}\0${before.size}\0${sha256}`);
      const fileGuard = guard(relative, 'file', after);
      guards.push(fileGuard);
      files.push({
        relative,
        size: Number(after.size),
        sha256,
        mode: fileGuard.mode,
        nlink: fileGuard.nlink
      });
      return;
    }
    fail(`unsupported file type: ${relative}`);
  }

  visit(rootPath, '');
  return {
    rows,
    guards,
    files,
    sha256: crypto.createHash('sha256').update(rows.join('\n')).digest('hex'),
    guardSha256: crypto.createHash('sha256')
      .update(JSON.stringify(guards))
      .digest('hex')
  };
}

try {
  const first = walk();
  const second = walk();
  const finalRoot = fs.lstatSync(rootPath, { bigint: true });
  const finalPinnedRoot = fs.fstatSync(rootDescriptor, { bigint: true });
  if (
    !stableStatEqual(initialRoot, finalRoot)
    || !stableStatEqual(pinnedRoot, finalPinnedRoot)
    || !stableStatEqual(finalRoot, finalPinnedRoot)
    || fs.realpathSync(rootPath) !== expectedRealRoot
    || first.sha256 !== second.sha256
    || first.guardSha256 !== second.guardSha256
    || JSON.stringify(first.rows) !== JSON.stringify(second.rows)
    || JSON.stringify(first.guards) !== JSON.stringify(second.guards)
    || JSON.stringify(first.files) !== JSON.stringify(second.files)
  ) {
    fail('tree changed between guarded walks');
  }

  process.stdout.write(`${JSON.stringify({
    sha256: second.sha256,
    guardSha256: second.guardSha256,
    entryCount: second.rows.length,
    ...(outputMode === 'flat-files' ? { files: second.files } : {})
  })}\n`);
} finally {
  fs.closeSync(rootDescriptor);
}
