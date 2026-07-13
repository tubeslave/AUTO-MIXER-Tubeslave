import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import http from 'node:http';
import os from 'node:os';
import path from 'node:path';
import { createRequire } from 'node:module';
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';

const require = createRequire(import.meta.url);
const __dirname = path.dirname(fileURLToPath(import.meta.url));
const frontendRoot = path.resolve(__dirname, '../../..');
const {
  BACKEND_NETWORK_SANDBOX_PROBES,
  BACKEND_SANDBOX_PROFILE_SHA256,
  LOOPBACK_ONLY_BACKEND_SANDBOX_PROFILE,
  assertLoopbackHttpOrigin,
  buildBackendEnvironment,
  createLaunchToken,
  explicitBoolean,
  startStaticServer,
  startViteDevServer
} = require(path.join(frontendRoot, 'public/electronSecurity.js'));

test('backend network sandbox policy and probe matrix are canonical', () => {
  assert.match(LOOPBACK_ONLY_BACKEND_SANDBOX_PROFILE, /deny network-outbound/);
  assert.match(BACKEND_SANDBOX_PROFILE_SHA256, /^[0-9a-f]{64}$/);
  assert.deepEqual(Object.keys(BACKEND_NETWORK_SANDBOX_PROBES).sort(), [
    'descendantUdp4Denied',
    'externalTcp4Denied',
    'externalTcp6Denied',
    'externalUdp4Denied',
    'externalUdp6Denied',
    'loopbackTcp4Allowed',
    'loopbackTcp6Allowed',
    'loopbackUdp4Allowed',
    'loopbackUdp6Allowed'
  ]);
  for (const probe of Object.values(BACKEND_NETWORK_SANDBOX_PROBES)) {
    assert.equal(typeof probe, 'string');
    assert.ok(probe.length > 40);
  }
});

test('canonical backend sandbox probe matrix is enforced by macOS', {
  skip: process.platform !== 'darwin'
}, () => {
  for (const [name, probe] of Object.entries(BACKEND_NETWORK_SANDBOX_PROBES)) {
    const completed = spawnSync('/usr/bin/sandbox-exec', [
      '-p', LOOPBACK_ONLY_BACKEND_SANDBOX_PROFILE,
      process.execPath, '-e', probe
    ], {
      encoding: 'utf8',
      env: { ...process.env, ELECTRON_RUN_AS_NODE: '1' },
      timeout: 5000
    });
    assert.equal(
      completed.status,
      0,
      `${name} failed: ${completed.error?.message || completed.stderr || completed.stdout}`
    );
  }
});

function request(url) {
  return new Promise((resolve, reject) => {
    http.get(url, response => {
      let body = '';
      response.setEncoding('utf8');
      response.on('data', chunk => { body += chunk; });
      response.on('end', () => resolve({
        body,
        headers: response.headers,
        statusCode: response.statusCode
      }));
    }).on('error', reject);
  });
}

test('launch token contains exactly 32 bytes of base64url entropy', () => {
  const token = createLaunchToken(size => Buffer.alloc(size, 0xab));

  assert.equal(token.length, 43);
  assert.match(token, /^[A-Za-z0-9_-]{43}$/);
  assert.equal(Buffer.from(token, 'base64url').length, 32);
});

test('backend child environment is isolated, loopback-only and memory-token authenticated', () => {
  const baseEnv = {
    PATH: '/usr/bin',
    HOME: '/Users/operator',
    AUTOMIXER_WS_AUTH_TOKEN: 'stale-parent-token',
    AUTOMIXER_WS_ALLOWED_ORIGINS: 'https://untrusted.example',
    AUTOMIXER_AUTO_MODE_UNLOCKED: 'true',
    AUTOMIXER_TRAINING_ENABLED: 'true',
    OPENAI_API_KEY: 'must-not-reach-child',
    AWS_SECRET_ACCESS_KEY: 'must-not-reach-child',
    PYTHONSTARTUP: '/tmp/inject.py'
  };
  const environment = buildBackendEnvironment({
    baseEnv,
    tokenFd: 3,
    rendererOrigin: 'http://127.0.0.1:49152',
    repoRoot: '/app/resources',
    pythonPath: '/app/backend:/app'
  });

  assert.equal('AUTOMIXER_WS_AUTH_TOKEN' in environment, false);
  assert.equal(environment.AUTOMIXER_WS_AUTH_TOKEN_FD, '3');
  assert.equal(environment.AUTOMIXER_WS_ALLOWED_ORIGINS, 'http://127.0.0.1:49152');
  assert.equal(environment.AUTOMIXER_WS_HOST, '127.0.0.1');
  assert.equal(environment.AUTOMIXER_WS_PORT, '8765');
  assert.equal(environment.AUTOMIXER_WS_CLIENT_ID, 'automixer-ui');
  assert.equal(environment.AUTOMIXER_WS_TOKEN_PROCESS_LIFETIME, 'true');
  assert.equal('AUTOMIXER_WS_TOKEN_TTL_HOURS' in environment, false);
  assert.equal(environment.AUTOMIXER_PACKAGED_APP, 'true');
  assert.equal(environment.AUTOMIXER_PACKAGED_OFFLINE_LOCK, 'false');
  assert.equal(environment.AUTOMIXER_AUTO_MODE_UNLOCKED, 'false');
  assert.equal(environment.AUTOMIXER_TRAINING_ENABLED, 'false');
  assert.equal(environment.AUTOMIXER_FORCE_CLEANUP_WS_PORT, 'false');
  assert.equal(environment.PYTHONPATH, '/app/backend:/app');
  assert.equal(environment.PATH, '/usr/bin');
  assert.equal(environment.HOME, '/Users/operator');
  assert.equal('OPENAI_API_KEY' in environment, false);
  assert.equal('AWS_SECRET_ACCESS_KEY' in environment, false);
  assert.equal('PYTHONSTARTUP' in environment, false);
  assert.equal(baseEnv.AUTOMIXER_WS_AUTH_TOKEN, 'stale-parent-token');
  assert.equal(baseEnv.AUTOMIXER_WS_ALLOWED_ORIGINS, 'https://untrusted.example');

  assert.equal(assertLoopbackHttpOrigin('http://localhost:3000'), 'http://localhost:3000');
  assert.throws(() => assertLoopbackHttpOrigin('file:///app/index.html'));
  assert.throws(() => assertLoopbackHttpOrigin('https://127.0.0.1:3000'));
  assert.throws(() => assertLoopbackHttpOrigin('http://example.test:3000'));
  assert.throws(() => assertLoopbackHttpOrigin('http://127.0.0.1:3000/path'));
});

test('packaged offline lock is allowlisted and forces immutable safe child settings', () => {
  assert.equal(explicitBoolean('YES'), true);
  assert.equal(explicitBoolean('false'), false);
  const environment = buildBackendEnvironment({
    baseEnv: {
      PATH: '/usr/bin',
      AUTOMIXER_PACKAGED_OFFLINE_LOCK: 'YES',
      AUTOMIXER_OPERATOR_MODE: 'supervised',
      AUTOMIXER_WING_OBSERVATION_MODE: 'false',
      AUTOMIXER_WING_RECORD_TELEMETRY: 'true'
    },
    tokenFd: 3,
    rendererOrigin: 'http://127.0.0.1:49153',
    repoRoot: '/app/resources'
  });

  assert.equal(environment.AUTOMIXER_PACKAGED_OFFLINE_LOCK, 'true');
  assert.equal(environment.AUTOMIXER_OPERATOR_MODE, 'analyze');
  assert.equal(environment.AUTOMIXER_AUTO_MODE_UNLOCKED, 'false');
  assert.equal(environment.AUTOMIXER_WING_OBSERVATION_MODE, 'true');
  assert.equal(environment.AUTOMIXER_WING_RECORD_TELEMETRY, 'false');
  assert.equal(environment.AUTOMIXER_WING_DEPLOYMENT_WRITE_BOUNDARY_ENABLED, 'true');
});

test('packaged renderer server binds an ephemeral loopback HTTP origin with security headers', async () => {
  const buildRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'automixer-electron-renderer-'));
  fs.writeFileSync(path.join(buildRoot, 'index.html'), '<!doctype html><title>Automixer</title>');
  let server;
  try {
    const started = await startStaticServer(buildRoot);
    server = started.server;
    assert.match(started.origin, /^http:\/\/127\.0\.0\.1:\d+$/);

    const response = await request(`${started.origin}/`);
    assert.equal(response.statusCode, 200);
    assert.match(response.body, /Automixer/);
    assert.equal(response.headers['cache-control'], 'no-store');
    assert.match(response.headers['content-security-policy'], /connect-src 'self' ws:\/\/127\.0\.0\.1:8765/);
    assert.equal(response.headers['x-content-type-options'], 'nosniff');
  } finally {
    if (server) await new Promise(resolve => server.close(resolve));
    fs.rmSync(buildRoot, { recursive: true, force: true });
  }
});

test('development Electron owns an ephemeral Vite server even when fixed port 3000 is hostile', async () => {
  const attacker = http.createServer((_request, response) => {
    response.writeHead(200, { 'Content-Type': 'text/html' });
    response.end('HOSTILE_RENDERER');
  });
  let viteServer;
  try {
    await new Promise((resolve, reject) => {
      attacker.once('error', reject);
      attacker.listen(3000, '127.0.0.1', () => {
        attacker.off('error', reject);
        resolve();
      });
    });
    const started = await startViteDevServer(frontendRoot);
    viteServer = started.server;
    assert.match(started.origin, /^http:\/\/127\.0\.0\.1:\d+$/);
    assert.notEqual(new URL(started.origin).port, '3000');
    const response = await request(`${started.origin}/`);
    assert.equal(response.statusCode, 200);
    assert.doesNotMatch(response.body, /HOSTILE_RENDERER/);
  } finally {
    if (viteServer) await viteServer.close();
    await new Promise(resolve => attacker.close(resolve));
  }
});

test('Electron entrypoint restricts bootstrap to the main frame and package scripts rebuild backend', () => {
  const electronSource = fs.readFileSync(path.join(frontendRoot, 'public/electron.js'), 'utf8');
  const preloadSource = fs.readFileSync(path.join(frontendRoot, 'public/preload.js'), 'utf8');
  const packageJson = JSON.parse(fs.readFileSync(path.join(frontendRoot, 'package.json'), 'utf8'));
  const viteSource = fs.readFileSync(path.join(frontendRoot, 'vite.config.js'), 'utf8');
  const macBuildSource = fs.readFileSync(
    path.join(frontendRoot, 'scripts', 'build-macos.mjs'),
    'utf8'
  );
  const backendBuildSource = fs.readFileSync(
    path.resolve(frontendRoot, '..', 'build_backend.sh'),
    'utf8'
  );

  assert.match(electronSource, /event\.senderFrame === mainWindow\.webContents\.mainFrame/);
  assert.match(electronSource, /setWindowOpenHandler/);
  assert.match(electronSource, /will-navigate/);
  assert.match(electronSource, /AUTO MIXER backend stopped/);
  assert.match(electronSource, /disable-background-networking/);
  assert.match(electronSource, /DnsOverHttps/);
  assert.match(electronSource, /disable-quic/);
  assert.match(electronSource, /webRequest\.onBeforeRequest/);
  assert.match(electronSource, /callback\(\{ cancel: !\(rendererTarget \|\| backendTarget\) \}\)/);
  assert.match(electronSource, /packagedOfflineLock/);
  assert.match(electronSource, /LOOPBACK_ONLY_BACKEND_SANDBOX_PROFILE/);
  assert.match(electronSource, /assertLoopbackOnlyBackendSandbox\(\)/);
  assert.match(electronSource, /childExecutable = '\/usr\/bin\/sandbox-exec'/);
  assert.match(electronSource, /backendNetworkSandboxed/);
  assert.match(electronSource, /backendNetworkSandboxProfileSha256/);
  assert.match(electronSource, /backendNetworkSandboxProbeResults/);
  assert.match(electronSource, /stdio:\s*\['ignore', 'pipe', 'pipe', 'pipe'\]/);
  assert.match(electronSource, /tokenPipe\.end\(launchToken\)/);
  assert.doesNotMatch(electronSource, /AUTOMIXER_WS_AUTH_TOKEN:\s*launchToken/);
  assert.match(electronSource, /ELECTRON_RUN_AS_NODE: '1'/);
  assert.doesNotMatch(electronSource, /\/usr\/bin\/python3/);
  assert.match(electronSource, /app\.quit\(\)/);
  assert.match(
    electronSource,
    /app\.on\('window-all-closed',[\s\S]*?app\.quit\(\);[\s\S]*?\}\);/
  );
  assert.match(electronSource, /child\.kill\('SIGTERM'\)/);
  assert.match(electronSource, /child\.kill\('SIGKILL'\)/);
  assert.doesNotMatch(electronSource, /child\.once\('error', finish\)/);
  assert.match(electronSource, /child\.once\('error', forceStop\)/);
  assert.match(electronSource, /event\.preventDefault\(\)/);
  assert.equal(electronSource.includes('file://'), false);
  assert.match(preloadSource, /getWebSocketAuthBootstrap/);
  assert.equal(preloadSource.includes('quitApp'), false);
  assert.equal(packageJson.scripts['dist:frontend'], 'npm run mac:local');
  assert.equal(packageJson.scripts.start, 'vite --host 127.0.0.1');
  assert.equal(packageJson.scripts['electron:dev'], 'electron .');
  assert.match(electronSource, /startViteDevServer/);
  assert.doesNotMatch(electronSource, /http:\/\/localhost:3000/);
  assert.match(viteSource, /host:\s*['"]127\.0\.0\.1['"]/);
  assert.doesNotMatch(viteSource, /host:\s*['"]0\.0\.0\.0['"]/);
  for (const target of packageJson.build.mac.target) {
    assert.deepEqual(target.arch, ['arm64']);
  }
  for (const scriptName of ['electron:build', 'electron:build:mac', 'dist:frontend']) {
    assert.equal(packageJson.scripts[scriptName], 'npm run mac:local');
  }
  assert.match(macBuildSource, /build_backend\.sh/);
  assert.match(macBuildSource, /verify-packaged-architecture\.mjs/);
  assert.match(macBuildSource, /AUTOMIXER_TARGET_ARCH: 'arm64'/);
  assert.match(packageJson.scripts['build:backend'], /AUTOMIXER_TARGET_ARCH=arm64/);
  assert.match(backendBuildSource, /Refusing cross-architecture backend build/);
  assert.match(backendBuildSource, /lipo -archs "\$BACKEND_BINARY"/);
  assert.ok(packageJson.build.files.includes('public/electronSecurity.js'));
});
