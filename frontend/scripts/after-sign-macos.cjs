'use strict';

const path = require('node:path');
const process = require('node:process');
const { spawnSync } = require('node:child_process');

function sanitize(value) {
  return String(value || '')
    .replace(/\b(password|token|secret|api[_-]?key)\s*[:=]\s*\S+/gi, '$1=[REDACTED]')
    .slice(-12_000);
}

function notarizeSignedApp(context, dependencies = {}) {
  const environment = dependencies.environment || process.env;
  const spawn = dependencies.spawn || spawnSync;
  if (environment.AUTOMIXER_MAC_BUILD_PROFILE !== 'release') {
    throw new Error('afterSign notarization requires the release build profile');
  }
  if (context?.electronPlatformName !== 'darwin') {
    throw new Error('afterSign notarization requires a macOS package');
  }
  const artifactRoot = path.resolve(String(environment.AUTOMIXER_MAC_OUTPUT_DIR || ''));
  if (!path.isAbsolute(String(environment.AUTOMIXER_MAC_OUTPUT_DIR || ''))) {
    throw new Error('afterSign notarization requires absolute AUTOMIXER_MAC_OUTPUT_DIR');
  }
  const appOutDir = path.resolve(String(context?.appOutDir || ''));
  if (appOutDir !== path.join(artifactRoot, 'mac-arm64')) {
    throw new Error('afterSign app output does not match the release artifact root');
  }
  const productFilename = String(context?.packager?.appInfo?.productFilename || '');
  if (!productFilename || path.basename(productFilename) !== productFilename) {
    throw new Error('afterSign product filename is invalid');
  }
  const appPath = path.join(appOutDir, `${productFilename}.app`);
  const scriptPath = path.join(__dirname, 'notarize-app.mjs');
  const completed = spawn(process.execPath, [
    scriptPath,
    '--artifact-root', artifactRoot,
    '--app', appPath
  ], {
    cwd: path.resolve(__dirname, '..'),
    env: environment,
    encoding: 'utf8',
    maxBuffer: 32 * 1024 * 1024
  });
  if (completed.error) throw completed.error;
  if (completed.status !== 0) {
    throw new Error(
      `Application notarization hook failed with exit code ${completed.status}: `
      + sanitize(completed.stderr || completed.stdout)
    );
  }
  let result;
  try {
    result = JSON.parse(String(completed.stdout || '').trim());
  } catch {
    throw new Error('Application notarization hook returned invalid JSON');
  }
  if (result?.status !== 'Accepted' || !result?.submissionId || !result?.cdhash) {
    throw new Error('Application notarization hook did not return accepted evidence');
  }
  return result;
}

module.exports = async context => notarizeSignedApp(context);
module.exports.notarizeSignedApp = notarizeSignedApp;
