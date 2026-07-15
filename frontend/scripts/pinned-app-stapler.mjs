#!/usr/bin/env node

import fs from 'node:fs';
import path from 'node:path';
import process from 'node:process';
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';

const scriptPath = fileURLToPath(import.meta.url);

function assertPinnedCwd({ expectedDev, expectedIno, expectedRealPath, label }) {
  const stat = fs.statSync('.', { bigint: true });
  if (
    !stat.isDirectory()
    || stat.dev.toString() !== expectedDev
    || stat.ino.toString() !== expectedIno
    || fs.realpathSync('.') !== expectedRealPath
  ) {
    throw new Error(`${label} working directory does not match the pinned application`);
  }
}

export function runPinnedAppStapler({
  action,
  expectedDev,
  expectedIno,
  expectedRealPath,
  runCommand = (command, args) => spawnSync(command, args, {
    encoding: null,
    maxBuffer: 32 * 1024 * 1024
  })
}) {
  if (!['staple', 'validate'].includes(action)) {
    throw new Error('Pinned application stapler action is invalid');
  }
  const context = {
    expectedDev: String(expectedDev || ''),
    expectedIno: String(expectedIno || ''),
    expectedRealPath: path.resolve(String(expectedRealPath || '')),
    label: `Application stapler ${action}`
  };
  assertPinnedCwd(context);
  const completed = runCommand('/usr/bin/xcrun', ['stapler', action, '.']);
  if (completed?.error) throw completed.error;
  if (completed?.status !== 0) {
    const detail = Buffer.concat([
      Buffer.from(completed?.stderr || ''),
      Buffer.from('\n'),
      Buffer.from(completed?.stdout || '')
    ]).toString('utf8').trim().slice(-12_000);
    throw new Error(
      `Application stapler ${action} failed with exit code ${completed?.status}`
      + (detail ? `: ${detail}` : '')
    );
  }
  assertPinnedCwd(context);
  return { status: 0 };
}

function main() {
  const [action, expectedDev, expectedIno, expectedRealPath] = process.argv.slice(2);
  runPinnedAppStapler({ action, expectedDev, expectedIno, expectedRealPath });
}

if (process.argv[1] && path.resolve(process.argv[1]) === scriptPath) {
  try {
    main();
  } catch (error) {
    process.stderr.write(`${error?.message || error}\n`);
    process.exitCode = 1;
  }
}
