#!/usr/bin/env node

import path from 'node:path';
import process from 'node:process';
import { fileURLToPath } from 'node:url';

import { verifyPackagedRuntimeSmokeManifest } from './release-evidence.mjs';

const scriptPath = fileURLToPath(import.meta.url);

function parseArgs(argv) {
  const options = { manifestPath: null, allowLegacy: false };
  for (let index = 0; index < argv.length; index += 1) {
    const argument = argv[index];
    if (argument === '--manifest') options.manifestPath = path.resolve(argv[++index] || '');
    else if (argument === '--allow-legacy') options.allowLegacy = true;
    else throw new Error(`Unknown argument: ${argument}`);
  }
  if (!options.manifestPath) {
    throw new Error(
      'Usage: verify-packaged-runtime-smoke.mjs --manifest <artifact-manifest.json> '
      + '[--allow-legacy]'
    );
  }
  return options;
}

export function main(argv = process.argv.slice(2)) {
  const { manifestPath, allowLegacy } = parseArgs(argv);
  const verified = verifyPackagedRuntimeSmokeManifest(manifestPath, { allowLegacy });
  const report = {
    schema: 'automixer.packaged_runtime_smoke_manifest_verifier.v1',
    status: 'passed',
    readOnly: true,
    contactsWing: false,
    contactsBackend: false,
    liveAuthorityGranted: false,
    legacyCompatibilityEnabled: allowLegacy,
    manifest: path.basename(manifestPath),
    verifiedFiles: Object.keys(verified.identities).sort()
  };
  process.stdout.write(`${JSON.stringify(report, null, 2)}\n`);
  return report;
}

if (path.resolve(process.argv[1] || '') === scriptPath) {
  try {
    main();
  } catch (error) {
    process.stderr.write(`${error?.stack || error?.message || error}\n`);
    process.exitCode = 1;
  }
}
