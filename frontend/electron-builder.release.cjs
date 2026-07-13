'use strict';

const path = require('node:path');

function required(name) {
  const value = String(process.env[name] || '').trim();
  if (!value) throw new Error(`Release macOS builder requires ${name}`);
  return value;
}

if (process.env.AUTOMIXER_MAC_BUILD_PROFILE !== 'release') {
  throw new Error('Release macOS builder requires AUTOMIXER_MAC_BUILD_PROFILE=release');
}

const base = JSON.parse(JSON.stringify(require('./package.json').build));
const outputDirectory = required('AUTOMIXER_MAC_OUTPUT_DIR');
const teamId = required('AUTOMIXER_APPLE_TEAM_ID');
const identity = required('AUTOMIXER_DEVELOPER_ID_IDENTITY');
const backendBundle = required('AUTOMIXER_BACKEND_BUNDLE_PATH');
const backendServer = required('AUTOMIXER_BACKEND_SERVER_PATH');
const frontendBuild = required('AUTOMIXER_FRONTEND_BUILD_DIR');
required('APPLE_KEYCHAIN_PROFILE');

for (const [name, value] of [
  ['AUTOMIXER_MAC_OUTPUT_DIR', outputDirectory],
  ['AUTOMIXER_BACKEND_BUNDLE_PATH', backendBundle],
  ['AUTOMIXER_BACKEND_SERVER_PATH', backendServer],
  ['AUTOMIXER_FRONTEND_BUILD_DIR', frontendBuild]
]) {
  if (!path.isAbsolute(value)) throw new Error(`${name} must be absolute`);
}
if (!/^[A-Z0-9]{10}$/.test(teamId)) {
  throw new Error('AUTOMIXER_APPLE_TEAM_ID must be a 10-character Apple Team ID');
}
if (!/^Developer ID Application:/.test(identity)) {
  throw new Error('AUTOMIXER_DEVELOPER_ID_IDENTITY must name a Developer ID Application identity');
}

module.exports = {
  ...base,
  files: [
    'package.json',
    { from: frontendBuild, to: 'build', filter: ['**/*'] },
    ...base.files.filter(entry => entry !== 'build/**/*')
  ],
  extraResources: base.extraResources.map(entry => (
    entry.to === 'backend' ? { ...entry, from: backendBundle } : entry
  )),
  forceCodeSigning: true,
  directories: {
    ...base.directories,
    output: outputDirectory
  },
  mac: {
    ...base.mac,
    target: [
      { target: 'dmg', arch: ['arm64'] },
      { target: 'zip', arch: ['arm64'] }
    ],
    type: 'distribution',
    identity,
    timestamp: 'http://timestamp.apple.com/ts01',
    notarize: { teamId },
    gatekeeperAssess: false
  },
  dmg: {
    ...base.dmg,
    sign: true,
    writeUpdateInfo: false
  }
};
