'use strict';

const path = require('node:path');

const base = JSON.parse(JSON.stringify(require('./package.json').build));
const outputDirectory = process.env.AUTOMIXER_MAC_OUTPUT_DIR;
const localIdentity = String(process.env.AUTOMIXER_LOCAL_SIGNING_IDENTITY || '').trim();
const backendBundle = process.env.AUTOMIXER_BACKEND_BUNDLE_PATH
  || path.resolve(__dirname, '../backend/dist/server');
const frontendBuild = process.env.AUTOMIXER_FRONTEND_BUILD_DIR
  || path.resolve(__dirname, 'build');

if (process.env.AUTOMIXER_MAC_BUILD_PROFILE !== 'local') {
  throw new Error('Local macOS builder requires AUTOMIXER_MAC_BUILD_PROFILE=local');
}
if (!outputDirectory || !path.isAbsolute(outputDirectory)) {
  throw new Error('Local macOS builder requires an absolute AUTOMIXER_MAC_OUTPUT_DIR');
}
if (localIdentity && !/^Apple Development:/.test(localIdentity)) {
  throw new Error('AUTOMIXER_LOCAL_SIGNING_IDENTITY must name an Apple Development identity');
}

const localMac = {
  ...base.mac,
  target: [{ target: 'zip', arch: ['arm64'] }],
  type: 'development',
  identity: localIdentity || '-',
  hardenedRuntime: Boolean(localIdentity),
  notarize: false,
  timestamp: null,
  gatekeeperAssess: false
};

module.exports = {
  ...base,
  artifactName: '${productName}-${version}-LOCAL-NOT-FOR-DISTRIBUTION-${arch}.${ext}',
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
  mac: localMac
};
