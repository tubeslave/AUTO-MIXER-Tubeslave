'use strict';

const crypto = require('crypto');
const fs = require('fs');
const http = require('http');
const path = require('path');

const MIME_TYPES = {
  '.css': 'text/css; charset=utf-8',
  '.html': 'text/html; charset=utf-8',
  '.ico': 'image/x-icon',
  '.jpeg': 'image/jpeg',
  '.jpg': 'image/jpeg',
  '.js': 'text/javascript; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
  '.map': 'application/json; charset=utf-8',
  '.png': 'image/png',
  '.svg': 'image/svg+xml',
  '.webp': 'image/webp',
  '.woff': 'font/woff',
  '.woff2': 'font/woff2'
};

const BACKEND_BASE_ENV_ALLOWLIST = new Set([
  'PATH',
  'HOME',
  'USER',
  'LOGNAME',
  'TMPDIR',
  'TEMP',
  'TMP',
  'LANG',
  'LC_ALL',
  'LC_CTYPE',
  'TZ',
  'SystemRoot',
  'WINDIR',
  'ComSpec',
  'PATHEXT',
  'USERPROFILE',
  'APPDATA',
  'LOCALAPPDATA',
  'PROGRAMDATA',
  'SSL_CERT_FILE',
  'SSL_CERT_DIR',
  'REQUESTS_CA_BUNDLE',
  'AUTOMIXER_PACKAGED_OFFLINE_LOCK',
  'AUTOMIXER_MANUAL_WRITE_ARTIFACTS_DIR',
  'AUTOMIXER_SHADOW_REPORTS_DIR',
  'AUTOMIXER_REPLAY_ARTIFACTS_DIR',
  'AUTOMIXER_WING_REPLAY_READINESS_DIR',
  'AUTOMIXER_WING_TELEMETRY_DIR',
  'HF_HUB_OFFLINE',
  'TRANSFORMERS_OFFLINE',
  'HF_DATASETS_OFFLINE',
  'HF_HUB_DISABLE_TELEMETRY'
]);

const LOOPBACK_ONLY_BACKEND_SANDBOX_PROFILE = [
  '(version 1)',
  '(allow default)',
  '(deny network-outbound)',
  '(allow network-outbound (remote ip "localhost:*"))'
].join('\n');
const BACKEND_SANDBOX_PROFILE_SHA256 = crypto.createHash('sha256')
  .update(LOOPBACK_ONLY_BACKEND_SANDBOX_PROFILE)
  .digest('hex');

function externalUdpProbe(socketType, host) {
  return [
    'const dgram=require("node:dgram")',
    'const timer=setTimeout(()=>process.exit(9),3000)',
    `const socket=dgram.createSocket(${JSON.stringify(socketType)})`,
    `socket.send(Buffer.from("x"),2222,${JSON.stringify(host)},error=>{`,
    'clearTimeout(timer);socket.close();',
    'process.exit(error&&(error.code==="EPERM"||error.code==="EACCES")?0:2)',
    '})'
  ].join(';');
}

function externalTcpProbe(host, family) {
  return [
    'const net=require("node:net")',
    'const timer=setTimeout(()=>process.exit(9),3000)',
    `const socket=net.createConnection({host:${JSON.stringify(host)},port:2222,family:${family}})`,
    'socket.once("connect",()=>{clearTimeout(timer);socket.destroy();process.exit(2)})',
    'socket.once("error",error=>{clearTimeout(timer);socket.destroy();',
    'process.exit(error&&(error.code==="EPERM"||error.code==="EACCES")?0:3)})'
  ].join(';');
}

function loopbackUdpProbe(socketType, host) {
  return [
    'const dgram=require("node:dgram")',
    'const timer=setTimeout(()=>process.exit(9),3000)',
    `const receiver=dgram.createSocket(${JSON.stringify(socketType)})`,
    'receiver.once("message",()=>{clearTimeout(timer);receiver.close();process.exit(0)})',
    `receiver.bind(0,${JSON.stringify(host)},()=>{`,
    `const sender=dgram.createSocket(${JSON.stringify(socketType)})`,
    `sender.send(Buffer.from("x"),receiver.address().port,${JSON.stringify(host)},error=>{`,
    'sender.close();if(error){clearTimeout(timer);receiver.close();process.exit(2)}',
    '})',
    '})'
  ].join(';');
}

function loopbackTcpProbe(host, family) {
  return [
    'const net=require("node:net")',
    'const timer=setTimeout(()=>process.exit(9),3000)',
    'const server=net.createServer(socket=>socket.end())',
    `server.listen({host:${JSON.stringify(host)},port:0},()=>{`,
    `const client=net.createConnection({host:${JSON.stringify(host)},port:server.address().port,family:${family}})`,
    'client.once("connect",()=>{clearTimeout(timer);client.destroy();server.close(()=>process.exit(0))})',
    'client.once("error",()=>{clearTimeout(timer);client.destroy();server.close(()=>process.exit(2))})',
    '})'
  ].join(';');
}

const externalUdp4DeniedProbe = externalUdpProbe('udp4', '192.0.2.1');
const BACKEND_NETWORK_SANDBOX_PROBES = Object.freeze({
  externalUdp4Denied: externalUdp4DeniedProbe,
  externalTcp4Denied: externalTcpProbe('192.0.2.1', 4),
  externalUdp6Denied: externalUdpProbe('udp6', '2001:db8::1'),
  externalTcp6Denied: externalTcpProbe('2001:db8::1', 6),
  loopbackUdp4Allowed: loopbackUdpProbe('udp4', '127.0.0.1'),
  loopbackTcp4Allowed: loopbackTcpProbe('127.0.0.1', 4),
  loopbackUdp6Allowed: loopbackUdpProbe('udp6', '::1'),
  loopbackTcp6Allowed: loopbackTcpProbe('::1', 6),
  descendantUdp4Denied: [
    'const {spawnSync}=require("node:child_process")',
    `const probe=${JSON.stringify(externalUdp4DeniedProbe)}`,
    'const child=spawnSync(process.execPath,["-e",probe],{env:{...process.env,ELECTRON_RUN_AS_NODE:"1"},timeout:5000,stdio:"ignore"})',
    'process.exit(child.status===0?0:2)'
  ].join(';')
});

function explicitBoolean(value) {
  return ['1', 'true', 'yes', 'on'].includes(String(value || '').trim().toLowerCase());
}

function allowlistedBackendEnvironment(baseEnv) {
  return Object.fromEntries(
    Object.entries(baseEnv || {}).filter(([key, value]) => (
      BACKEND_BASE_ENV_ALLOWLIST.has(key)
      && typeof value === 'string'
      && value.length > 0
    ))
  );
}

function createLaunchToken(randomBytes = crypto.randomBytes) {
  const bytes = randomBytes(32);
  if (!Buffer.isBuffer(bytes) || bytes.length !== 32) {
    throw new Error('Electron launch token source must return exactly 32 bytes');
  }
  return bytes.toString('base64url');
}

function assertLoopbackHttpOrigin(origin) {
  let parsed;
  try {
    parsed = new URL(String(origin || ''));
  } catch {
    throw new Error('Renderer origin must be an exact loopback HTTP origin');
  }
  if (
    parsed.protocol !== 'http:'
    || parsed.pathname !== '/'
    || parsed.search
    || parsed.hash
    || parsed.username
    || parsed.password
    || !['127.0.0.1', 'localhost'].includes(parsed.hostname)
  ) {
    throw new Error('Renderer origin must be an exact loopback HTTP origin');
  }
  return parsed.origin;
}

function buildBackendEnvironment({
  baseEnv = {},
  tokenFd = 3,
  rendererOrigin,
  repoRoot,
  pythonPath
}) {
  if (!Number.isInteger(tokenFd) || tokenFd < 3 || tokenFd > 999) {
    throw new Error('Electron backend launch token fd is invalid');
  }
  const allowedOrigin = assertLoopbackHttpOrigin(rendererOrigin);
  const packagedOfflineLock = explicitBoolean(baseEnv.AUTOMIXER_PACKAGED_OFFLINE_LOCK);
  const environment = {
    ...allowlistedBackendEnvironment(baseEnv),
    AUTOMIXER_REPO_ROOT: String(repoRoot || ''),
    AUTOMIXER_PACKAGED_APP: 'true',
    AUTOMIXER_PACKAGED_OFFLINE_LOCK: packagedOfflineLock ? 'true' : 'false',
    AUTOMIXER_WS_HOST: '127.0.0.1',
    AUTOMIXER_WS_PORT: '8765',
    AUTOMIXER_WS_ALLOWED_ORIGINS: allowedOrigin,
    AUTOMIXER_WS_AUTH_TOKEN_FD: String(tokenFd),
    AUTOMIXER_WS_CLIENT_ID: 'automixer-ui',
    AUTOMIXER_WS_TOKEN_PROCESS_LIFETIME: 'true',
    AUTOMIXER_AUTO_MODE_UNLOCKED: 'false',
    AUTOMIXER_FORCE_CLEANUP_WS_PORT: 'false',
    AUTOMIXER_TRAINING_ENABLED: 'false',
    AUTOMIXER_TRAINING_SAFE_AUTOSTART: 'false',
    AUTOMIXER_TRAINING_STUDY_ENABLED: 'false',
    AUTOMIXER_TRAINING_DISCOVERY_ENABLED: 'false'
  };
  if (packagedOfflineLock) {
    environment.AUTOMIXER_OPERATOR_MODE = 'analyze';
    environment.AUTOMIXER_WING_OBSERVATION_MODE = 'true';
    environment.AUTOMIXER_WING_RECORD_TELEMETRY = 'false';
    environment.AUTOMIXER_WING_DEPLOYMENT_WRITE_BOUNDARY_ENABLED = 'true';
  }
  if (pythonPath) environment.PYTHONPATH = String(pythonPath);
  return environment;
}

function isInsideRoot(root, candidate) {
  return candidate === root || candidate.startsWith(`${root}${path.sep}`);
}

async function resolveStaticFile(buildRoot, requestPath) {
  let decoded;
  try {
    decoded = decodeURIComponent(requestPath);
  } catch {
    throw Object.assign(new Error('Malformed request path'), { statusCode: 400 });
  }
  if (decoded.includes('\0')) {
    throw Object.assign(new Error('Malformed request path'), { statusCode: 400 });
  }

  const relativePath = decoded === '/' ? 'index.html' : decoded.replace(/^\/+/, '');
  let candidate = path.resolve(buildRoot, relativePath);
  if (!isInsideRoot(buildRoot, candidate)) {
    throw Object.assign(new Error('Path outside renderer root'), { statusCode: 403 });
  }

  try {
    const stat = await fs.promises.stat(candidate);
    if (stat.isDirectory()) candidate = path.join(candidate, 'index.html');
  } catch {
    if (path.extname(relativePath)) {
      throw Object.assign(new Error('Renderer asset not found'), { statusCode: 404 });
    }
    candidate = path.join(buildRoot, 'index.html');
  }

  const realCandidate = await fs.promises.realpath(candidate).catch(() => null);
  if (!realCandidate || !isInsideRoot(buildRoot, realCandidate)) {
    throw Object.assign(new Error('Renderer asset not found'), { statusCode: 404 });
  }
  const stat = await fs.promises.stat(realCandidate);
  if (!stat.isFile()) {
    throw Object.assign(new Error('Renderer asset not found'), { statusCode: 404 });
  }
  return realCandidate;
}

async function startStaticServer(buildDirectory) {
  const buildRoot = fs.realpathSync(String(buildDirectory));
  const server = http.createServer(async (request, response) => {
    if (!['GET', 'HEAD'].includes(request.method || '')) {
      response.writeHead(405, { Allow: 'GET, HEAD' });
      response.end();
      return;
    }

    try {
      const requestUrl = new URL(request.url || '/', 'http://127.0.0.1');
      const filePath = await resolveStaticFile(buildRoot, requestUrl.pathname);
      response.writeHead(200, {
        'Cache-Control': 'no-store',
        'Content-Security-Policy': (
          "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; "
          + "img-src 'self' data:; font-src 'self' data:; "
          + "connect-src 'self' ws://127.0.0.1:8765; object-src 'none'; base-uri 'none'"
        ),
        'Content-Type': MIME_TYPES[path.extname(filePath).toLowerCase()] || 'application/octet-stream',
        'X-Content-Type-Options': 'nosniff'
      });
      if (request.method === 'HEAD') {
        response.end();
        return;
      }
      const stream = fs.createReadStream(filePath);
      stream.on('error', () => response.destroy());
      stream.pipe(response);
    } catch (error) {
      const statusCode = Number(error?.statusCode) || 500;
      response.writeHead(statusCode, {
        'Cache-Control': 'no-store',
        'Content-Type': 'text/plain; charset=utf-8',
        'X-Content-Type-Options': 'nosniff'
      });
      response.end(statusCode === 500 ? 'Renderer unavailable' : 'Not found');
    }
  });

  return await new Promise((resolve, reject) => {
    const onError = error => reject(error);
    server.once('error', onError);
    server.listen(0, '127.0.0.1', () => {
      server.off('error', onError);
      const address = server.address();
      if (!address || typeof address === 'string') {
        server.close();
        reject(new Error('Could not allocate loopback renderer port'));
        return;
      }
      resolve({
        server,
        origin: `http://127.0.0.1:${address.port}`
      });
    });
  });
}

async function startViteDevServer(frontendDirectory) {
  const frontendRoot = fs.realpathSync(String(frontendDirectory));
  const { createServer } = await import('vite');
  const server = await createServer({
    root: frontendRoot,
    configFile: path.join(frontendRoot, 'vite.config.js'),
    logLevel: 'silent',
    server: {
      host: '127.0.0.1',
      port: 0,
      strictPort: true
    }
  });
  try {
    await server.listen();
    const address = server.httpServer?.address();
    if (
      !address
      || typeof address === 'string'
      || address.address !== '127.0.0.1'
      || !Number.isInteger(address.port)
      || address.port <= 0
    ) {
      throw new Error('Could not allocate Electron-owned Vite loopback port');
    }
    return {
      server,
      origin: `http://127.0.0.1:${address.port}`
    };
  } catch (error) {
    await server.close().catch(() => {});
    throw error;
  }
}

module.exports = {
  BACKEND_NETWORK_SANDBOX_PROBES,
  BACKEND_SANDBOX_PROFILE_SHA256,
  LOOPBACK_ONLY_BACKEND_SANDBOX_PROFILE,
  assertLoopbackHttpOrigin,
  buildBackendEnvironment,
  createLaunchToken,
  explicitBoolean,
  startStaticServer,
  startViteDevServer
};
