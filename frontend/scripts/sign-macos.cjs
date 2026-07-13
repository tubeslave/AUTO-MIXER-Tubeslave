'use strict';

const crypto = require('node:crypto');
const fs = require('node:fs');
const path = require('node:path');
const { spawnSync } = require('node:child_process');
const { signAsync } = require('@electron/osx-sign');

const MACH_O_MAGICS = new Set([
  0xfeedface,
  0xcefaedfe,
  0xfeedfacf,
  0xcffaedfe,
  0xcafebabe,
  0xbebafeca,
  0xcafebabf,
  0xbfbafeca
]);

function isMachOFile(filePath) {
  const stat = fs.lstatSync(filePath);
  if (!stat.isFile()) return false;
  const descriptor = fs.openSync(filePath, 'r');
  try {
    const magic = Buffer.alloc(4);
    if (fs.readSync(descriptor, magic, 0, magic.length, 0) !== magic.length) return false;
    return MACH_O_MAGICS.has(magic.readUInt32BE(0));
  } finally {
    fs.closeSync(descriptor);
  }
}

function hashTree(root) {
  const rows = [];
  function visit(candidate, relative) {
    const stat = fs.lstatSync(candidate);
    const mode = (stat.mode & 0o7777).toString(8).padStart(4, '0');
    if (stat.isSymbolicLink()) {
      rows.push(`L\0${relative}\0${mode}\0${fs.readlinkSync(candidate)}`);
      return;
    }
    if (stat.isDirectory()) {
      rows.push(`D\0${relative}\0${mode}`);
      for (const name of fs.readdirSync(candidate).sort()) {
        visit(path.join(candidate, name), relative ? `${relative}/${name}` : name);
      }
      return;
    }
    if (stat.isFile()) {
      const digest = crypto.createHash('sha256').update(fs.readFileSync(candidate)).digest('hex');
      rows.push(`F\0${relative}\0${mode}\0${stat.size}\0${digest}`);
      return;
    }
    throw new Error(`Unsupported backend bundle file type: ${relative}`);
  }
  visit(root, '');
  return crypto.createHash('sha256').update(rows.join('\n')).digest('hex');
}

function matchesIgnoreRule(rule, filePath) {
  if (typeof rule === 'function') return Boolean(rule(filePath));
  return Boolean(filePath.match(rule));
}

function buildSignIgnore(inheritedIgnore) {
  const inheritedRules = Array.isArray(inheritedIgnore)
    ? inheritedIgnore
    : inheritedIgnore ? [inheritedIgnore] : [];
  return filePath => {
    if (inheritedRules.some(rule => matchesIgnoreRule(rule, filePath))) return true;
    const stat = fs.lstatSync(filePath);
    return stat.isFile() && !isMachOFile(filePath);
  };
}

function scopedBackendEntitlements(filePath, {
  backendPath,
  backendEntitlements,
  emptyEntitlements
}) {
  const resolved = path.resolve(filePath);
  if (resolved === backendPath) return backendEntitlements;
  const backendBundlePath = path.dirname(backendPath);
  if (
    resolved.startsWith(`${backendBundlePath}${path.sep}`)
    && fs.existsSync(resolved)
    && isMachOFile(resolved)
  ) {
    return emptyEntitlements;
  }
  return null;
}

function adHocSignLocalApp(appPath, spawn = spawnSync) {
  const sign = spawn('/usr/bin/codesign', [
    '--force', '--deep', '--sign', '-', appPath
  ], { encoding: 'utf8', maxBuffer: 64 * 1024 * 1024 });
  if (sign.error) throw sign.error;
  if (sign.status !== 0) {
    throw new Error(
      `Local ad-hoc codesign failed (${sign.status}): `
      + `${sign.stderr || sign.stdout || ''}`.trim()
    );
  }
  const verify = spawn('/usr/bin/codesign', [
    '--verify', '--deep', '--strict', '--verbose=2', appPath
  ], { encoding: 'utf8', maxBuffer: 64 * 1024 * 1024 });
  if (verify.error) throw verify.error;
  if (verify.status !== 0) {
    throw new Error(
      `Local ad-hoc signature verification failed (${verify.status}): `
      + `${verify.stderr || verify.stdout || ''}`.trim()
    );
  }
  return { signed: true, identity: '-', verified: true };
}

module.exports = async function signMacWithScopedBackendEntitlements(options) {
  const releaseProfile = process.env.AUTOMIXER_MAC_BUILD_PROFILE === 'release';
  const localAdHocProfile = (
    process.env.AUTOMIXER_MAC_BUILD_PROFILE === 'local'
    && !String(process.env.AUTOMIXER_LOCAL_SIGNING_IDENTITY || '').trim()
  );
  if (releaseProfile) {
    if (options.type !== 'distribution' || !String(options.identity || '').trim()) {
      throw new Error(
        'Release signing requires an explicit distribution identity selected by electron-builder'
      );
    }
  }
  const backendPath = path.resolve(
    options.app,
    'Contents',
    'Resources',
    'backend',
    'server'
  );
  const backendBundlePath = path.dirname(backendPath);
  const backendEntitlements = path.resolve(
    __dirname,
    '..',
    'public',
    'entitlements.backend.plist'
  );
  const emptyEntitlements = path.resolve(
    __dirname,
    '..',
    'public',
    'entitlements.empty.plist'
  );
  if (!fs.existsSync(backendPath)) {
    throw new Error(`Packaged backend executable is missing: ${backendPath}`);
  }
  if (!fs.existsSync(backendEntitlements)) {
    throw new Error(`Backend entitlements file is missing: ${backendEntitlements}`);
  }
  if (!fs.existsSync(emptyEntitlements)) {
    throw new Error(`Empty nested-code entitlements file is missing: ${emptyEntitlements}`);
  }

  if (localAdHocProfile) {
    return adHocSignLocalApp(options.app);
  }

  if (releaseProfile) {
    const required = {
      head: process.env.AUTOMIXER_RELEASE_SOURCE_HEAD,
      tree: process.env.AUTOMIXER_RELEASE_SOURCE_TREE,
      backendSha256: process.env.AUTOMIXER_RELEASE_BACKEND_SHA256,
      backendBundleTreeSha256: process.env.AUTOMIXER_RELEASE_BACKEND_BUNDLE_TREE_SHA256,
      frontendTreeSha256: process.env.AUTOMIXER_RELEASE_FRONTEND_TREE_SHA256
    };
    for (const [name, value] of Object.entries(required)) {
      if (!/^[0-9a-f]{40,64}$/.test(String(value || ''))) {
        throw new Error(`Release signing attestation field is invalid: ${name}`);
      }
    }
    const actualBackendSha256 = crypto.createHash('sha256')
      .update(fs.readFileSync(backendPath))
      .digest('hex');
    if (actualBackendSha256 !== required.backendSha256) {
      throw new Error('Packaged backend does not match the isolated release build output');
    }
    if (hashTree(backendBundlePath) !== required.backendBundleTreeSha256) {
      throw new Error('Packaged backend bundle does not match the isolated release build output');
    }
    const attestationPath = path.resolve(
      options.app,
      'Contents',
      'Resources',
      'release-build-attestation.json'
    );
    if (fs.existsSync(attestationPath)) {
      throw new Error('Refusing to replace an existing release build attestation');
    }
    fs.writeFileSync(attestationPath, `${JSON.stringify({
      schema: 'automixer.macos_release_build_attestation.v1',
      source: { head: required.head, tree: required.tree },
      buildOutputs: {
        backendSha256: required.backendSha256,
        backendBundleTreeSha256: required.backendBundleTreeSha256,
        frontendTreeSha256: required.frontendTreeSha256
      }
    }, null, 2)}\n`, { mode: 0o600 });
  }

  const defaultOptionsForFile = options.optionsForFile;
  // @electron/osx-sign 1.0.5 drops an ignore value when it is already an
  // array (validateOptsIgnore() has no array return branch). Pass one
  // composite function so its validator wraps and preserves the rule.
  options.ignore = buildSignIgnore(options.ignore);
  let backendVisited = false;
  let nestedBackendMachOVisited = 0;
  options.optionsForFile = filePath => {
    const defaults = defaultOptionsForFile ? defaultOptionsForFile(filePath) : {};
    const resolved = path.resolve(filePath);
    const entitlements = scopedBackendEntitlements(resolved, {
      backendPath,
      backendEntitlements,
      emptyEntitlements
    });
    if (entitlements === backendEntitlements) {
      backendVisited = true;
    } else if (entitlements === emptyEntitlements) {
      nestedBackendMachOVisited += 1;
    }
    return entitlements ? { ...defaults, entitlements } : defaults;
  };

  // @electron/osx-sign 1.0.5 renders boolean `true` as the invalid
  // `codesign --strict=true`. Undefined keeps its strict-by-default behavior
  // and renders the valid `codesign --strict` form. Preserve every other
  // value so this narrow compatibility shim cannot weaken verification.
  const signOptions = {
    ...options,
    strictVerify: options.strictVerify === true ? undefined : options.strictVerify
  };

  await signAsync(signOptions);
  if (!backendVisited) {
    throw new Error('macOS signer did not visit the packaged backend executable');
  }
  if (nestedBackendMachOVisited < 1) {
    throw new Error('macOS signer did not visit any nested backend Mach-O runtime files');
  }
};

module.exports.buildSignIgnore = buildSignIgnore;
module.exports.hashTree = hashTree;
module.exports.scopedBackendEntitlements = scopedBackendEntitlements;
module.exports.adHocSignLocalApp = adHocSignLocalApp;
