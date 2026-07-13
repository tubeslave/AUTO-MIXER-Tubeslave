'use strict';

const path = require('node:path');
const fs = require('node:fs');
const { spawnSync } = require('node:child_process');

function runPlutil(args, spawn = spawnSync) {
  const result = spawn('/usr/bin/plutil', args, {
    encoding: 'utf8',
    maxBuffer: 4 * 1024 * 1024
  });
  if (result.error) throw result.error;
  if (result.status !== 0) {
    throw new Error(
      `/usr/bin/plutil ${args.join(' ')} failed (${result.status}): `
      + `${result.stderr || result.stdout || ''}`.trim()
    );
  }
  return result.stdout || '';
}

function writeLocalBuildAttestation(context, plistPath, environment = process.env) {
  if (environment.AUTOMIXER_MAC_BUILD_PROFILE !== 'local') return null;
  const required = [
    'AUTOMIXER_BUILD_SOURCE_HEAD',
    'AUTOMIXER_BUILD_SOURCE_TREE',
    'AUTOMIXER_BUILD_SOURCE_STATUS_SHA256',
    'AUTOMIXER_BUILD_SOURCE_CONTENT_SHA256',
    'AUTOMIXER_BUILD_SOURCE_WORKTREE_DIRTY'
  ];
  for (const name of required) {
    if (!String(environment[name] || '').trim()) {
      throw new Error(`local macOS afterPack attestation requires ${name}`);
    }
  }
  const attestation = {
    schema: 'automixer.macos_local_build_attestation.v1',
    generatedAt: new Date().toISOString(),
    profile: 'local',
    releaseEligible: false,
    reason: 'LOCAL NOT FOR DISTRIBUTION; Developer ID, notarization and Gatekeeper acceptance are not asserted.',
    source: {
      head: environment.AUTOMIXER_BUILD_SOURCE_HEAD,
      tree: environment.AUTOMIXER_BUILD_SOURCE_TREE,
      worktreeDirty: environment.AUTOMIXER_BUILD_SOURCE_WORKTREE_DIRTY === 'true',
      statusSha256: environment.AUTOMIXER_BUILD_SOURCE_STATUS_SHA256,
      sourceContentSha256: environment.AUTOMIXER_BUILD_SOURCE_CONTENT_SHA256
    }
  };
  const attestationPath = path.join(
    path.dirname(plistPath),
    'Resources',
    'local-build-attestation.json'
  );
  fs.mkdirSync(path.dirname(attestationPath), { recursive: true });
  fs.writeFileSync(attestationPath, `${JSON.stringify(attestation, null, 2)}\n`, {
    mode: 0o644
  });
  return { path: attestationPath, attestation };
}

function applyMacAtsPolicy(context, { spawn = spawnSync, environment = process.env } = {}) {
  if (context?.electronPlatformName !== 'darwin') {
    return { skipped: true, reason: 'non_macos_package' };
  }
  const appOutDir = String(context.appOutDir || '').trim();
  const productFilename = String(context.packager?.appInfo?.productFilename || '').trim();
  if (!appOutDir || !productFilename) {
    throw new Error('macOS afterPack ATS hook requires appOutDir and productFilename');
  }
  const plistPath = path.join(
    appOutDir,
    `${productFilename}.app`,
    'Contents',
    'Info.plist'
  );
  runPlutil([
    '-replace',
    'NSAppTransportSecurity.NSAllowsArbitraryLoads',
    '-bool',
    'NO',
    plistPath
  ], spawn);
  const plist = JSON.parse(runPlutil([
    '-convert',
    'json',
    '-o',
    '-',
    plistPath
  ], spawn));
  const ats = plist.NSAppTransportSecurity;
  if (
    !ats
    || ats.NSAllowsArbitraryLoads !== false
    || ats.NSAllowsLocalNetworking !== true
  ) {
    throw new Error(
      'macOS afterPack ATS readback requires NSAllowsArbitraryLoads=false '
      + 'and NSAllowsLocalNetworking=true'
    );
  }
  const localBuildAttestation = writeLocalBuildAttestation(
    context,
    plistPath,
    environment
  );
  return {
    skipped: false,
    plistPath,
    NSAllowsArbitraryLoads: false,
    NSAllowsLocalNetworking: true,
    localBuildAttestation
  };
}

async function afterPack(context) {
  return applyMacAtsPolicy(context);
}

module.exports = afterPack;
module.exports.applyMacAtsPolicy = applyMacAtsPolicy;
module.exports.writeLocalBuildAttestation = writeLocalBuildAttestation;
