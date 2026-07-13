'use strict';

const { app, BrowserWindow, dialog, ipcMain, session } = require('electron');
const path = require('path');
const { spawn, spawnSync } = require('child_process');
const fs = require('fs');
const {
  BACKEND_NETWORK_SANDBOX_PROBES,
  BACKEND_SANDBOX_PROFILE_SHA256,
  LOOPBACK_ONLY_BACKEND_SANDBOX_PROFILE,
  buildBackendEnvironment,
  createLaunchToken,
  explicitBoolean,
  startStaticServer,
  startViteDevServer
} = require('./electronSecurity');

// Chromium must not perform DNS-over-HTTPS probes, connectivity checks, or
// other background traffic. The packaged product only needs its two explicit
// loopback origins; renderer requests are separately allowlisted below.
app.commandLine.appendSwitch('disable-background-networking');
app.commandLine.appendSwitch('disable-component-update');
app.commandLine.appendSwitch('disable-domain-reliability');
app.commandLine.appendSwitch('disable-quic');
app.commandLine.appendSwitch(
  'disable-features',
  [
    'AsyncDns',
    'DnsOverHttps',
    'UseDnsHttpsSvcbAlpn',
    'MediaRouter',
    'OptimizationHints',
    'AutofillServerCommunication',
    'CertificateTransparencyComponentUpdater',
    'NetworkTimeServiceQuerying'
  ].join(',')
);

let mainWindow = null;
let backendProcess = null;
let rendererServer = null;
let rendererOrigin = null;
let launchToken = null;
let quitting = false;
let shutdownComplete = false;
let shutdownPromise = null;
let backendNetworkSandboxed = false;
let backendNetworkSandboxProbeResults = null;

const isDev = !app.isPackaged;
const packagedOfflineLock = explicitBoolean(
  process.env.AUTOMIXER_PACKAGED_OFFLINE_LOCK
);
function assertLoopbackOnlyBackendSandbox() {
  if (process.platform !== 'darwin' || !fs.existsSync('/usr/bin/sandbox-exec')) {
    throw new Error('Packaged offline lock requires macOS sandbox-exec');
  }
  const results = {};
  for (const [name, probe] of Object.entries(BACKEND_NETWORK_SANDBOX_PROBES)) {
    const completed = spawnSync('/usr/bin/sandbox-exec', [
      '-p', LOOPBACK_ONLY_BACKEND_SANDBOX_PROFILE,
      process.execPath, '-e', probe
    ], {
      encoding: 'utf8',
      env: { ...process.env, ELECTRON_RUN_AS_NODE: '1' },
      timeout: 5000,
      stdio: 'ignore'
    });
    if (completed.error || completed.status !== 0) {
      throw new Error(`Packaged offline backend network sandbox preflight failed: ${name}`);
    }
    results[name] = true;
  }
  return results;
}

function isTrustedMainFrame(event) {
  return Boolean(
    mainWindow
    && !mainWindow.isDestroyed()
    && event.sender === mainWindow.webContents
    && event.senderFrame === mainWindow.webContents.mainFrame
  );
}

ipcMain.handle('get-websocket-auth-bootstrap', event => {
  if (!isTrustedMainFrame(event) || !launchToken || !rendererOrigin) {
    throw new Error('Secure Electron bootstrap unavailable');
  }
  return {
    token: launchToken,
    clientId: 'automixer-ui',
    websocketUrl: 'ws://127.0.0.1:8765',
    packagedOfflineLock,
    backendNetworkSandboxed,
    backendNetworkSandboxProfileSha256: backendNetworkSandboxed
      ? BACKEND_SANDBOX_PROFILE_SHA256
      : null,
    backendNetworkSandboxProbeResults: backendNetworkSandboxed
      ? backendNetworkSandboxProbeResults
      : null
  };
});

function configureRendererNetworkBoundary() {
  if (!rendererOrigin) throw new Error('Renderer origin is unavailable');
  const rendererUrl = new URL(rendererOrigin);
  session.defaultSession.setPermissionRequestHandler((_webContents, _permission, callback) => {
    callback(false);
  });
  session.defaultSession.webRequest.onBeforeRequest(
    { urls: ['http://*/*', 'https://*/*', 'ws://*/*', 'wss://*/*'] },
    (details, callback) => {
      try {
        const target = new URL(details.url);
        const rendererTarget = (
          ['http:', 'ws:'].includes(target.protocol)
          && target.hostname === rendererUrl.hostname
          && target.port === rendererUrl.port
        );
        const backendTarget = (
          target.protocol === 'ws:'
          && target.hostname === '127.0.0.1'
          && target.port === '8765'
        );
        callback({ cancel: !(rendererTarget || backendTarget) });
      } catch {
        callback({ cancel: true });
      }
    }
  );
}

function createWindow() {
  if (!rendererOrigin) throw new Error('Renderer origin is unavailable');
  mainWindow = new BrowserWindow({
    width: 1600,
    height: 1000,
    minWidth: 1200,
    minHeight: 800,
    backgroundColor: '#1a1a1a',
    webPreferences: {
      nodeIntegration: false,
      contextIsolation: true,
      sandbox: true,
      preload: path.join(__dirname, 'preload.js')
    },
    titleBarStyle: 'hiddenInset',
    show: false
  });

  mainWindow.once('ready-to-show', () => {
    mainWindow?.show();
  });
  mainWindow.webContents.setWindowOpenHandler(() => ({ action: 'deny' }));
  mainWindow.webContents.on('will-navigate', (event, targetUrl) => {
    try {
      if (new URL(targetUrl).origin !== rendererOrigin) event.preventDefault();
    } catch {
      event.preventDefault();
    }
  });
  mainWindow.loadURL(rendererOrigin);

  if (isDev) mainWindow.webContents.openDevTools();

  mainWindow.on('closed', () => {
    mainWindow = null;
  });
}

function redactBackendOutput(data) {
  const output = String(data || '');
  return launchToken ? output.split(launchToken).join('[REDACTED]') : output;
}

function startBackend() {
  const backendPath = isDev
    ? path.join(__dirname, '../../backend/server.py')
    : path.join(process.resourcesPath, 'backend', 'server');
  const repoRoot = isDev
    ? path.join(__dirname, '../..')
    : process.resourcesPath;
  const backendDirectory = isDev
    ? path.join(repoRoot, 'backend')
    : process.resourcesPath;

  if (!isDev && !fs.existsSync(backendPath)) {
    throw new Error(`Backend executable not found: ${backendPath}`);
  }

  const pythonPath = isDev
    ? [backendDirectory, repoRoot, process.env.PYTHONPATH]
      .filter(Boolean)
      .join(path.delimiter)
    : undefined;
  const childEnvironment = buildBackendEnvironment({
    baseEnv: process.env,
    tokenFd: 3,
    rendererOrigin,
    repoRoot,
    pythonPath
  });
  const executable = isDev
    ? (process.platform === 'win32' ? 'python' : 'python3')
    : backendPath;
  const args = isDev ? [backendPath] : [];
  let childExecutable = executable;
  let childArgs = args;
  if (packagedOfflineLock) {
    if (isDev) throw new Error('Packaged offline lock is unavailable in development mode');
    backendNetworkSandboxProbeResults = assertLoopbackOnlyBackendSandbox();
    childExecutable = '/usr/bin/sandbox-exec';
    childArgs = ['-p', LOOPBACK_ONLY_BACKEND_SANDBOX_PROFILE, executable, ...args];
    backendNetworkSandboxed = true;
  }

  return new Promise((resolve, reject) => {
    let spawned = false;
    const child = spawn(childExecutable, childArgs, {
      cwd: backendDirectory,
      env: childEnvironment,
      stdio: ['ignore', 'pipe', 'pipe', 'pipe']
    });
    backendProcess = child;
    const tokenPipe = child.stdio?.[3];
    if (!tokenPipe || !launchToken || !/^[A-Za-z0-9_-]{43}$/.test(launchToken)) {
      child.kill();
      reject(new Error('Backend launch token pipe is unavailable'));
      return;
    }
    tokenPipe.on('error', () => {
      if (!spawned) reject(new Error('Backend launch token pipe failed'));
    });
    tokenPipe.end(launchToken);

    child.once('spawn', () => {
      spawned = true;
      resolve();
    });
    child.once('error', () => {
      if (backendProcess === child) backendProcess = null;
      backendNetworkSandboxed = false;
      backendNetworkSandboxProbeResults = null;
      if (!spawned) reject(new Error('Backend process failed to start'));
      else console.error('Backend process error.');
    });
    child.stdout?.on('data', data => {
      console.log(`Backend: ${redactBackendOutput(data)}`);
    });
    child.stderr?.on('data', data => {
      console.error(`Backend Error: ${redactBackendOutput(data)}`);
    });
    child.on('close', code => {
      if (backendProcess === child) backendProcess = null;
      if (!quitting) {
        console.error(`Backend process exited with code ${code}`);
        dialog.showErrorBox(
          'AUTO MIXER backend stopped',
          'The secured local backend exited. The application will close without live authority.'
        );
        app.quit();
      }
    });
  });
}

function stopBackend() {
  const child = backendProcess;
  backendProcess = null;
  backendNetworkSandboxed = false;
  backendNetworkSandboxProbeResults = null;
  if (!child || child.exitCode !== null || child.signalCode !== null) {
    return Promise.resolve();
  }
  return new Promise(resolve => {
    let finished = false;
    let forceTimer;
    let verificationTimer;
    const childHasExited = () => {
      if (child.exitCode !== null || child.signalCode !== null) return true;
      if (!Number.isInteger(child.pid) || child.pid <= 0) return false;
      try {
        process.kill(child.pid, 0);
        return false;
      } catch (error) {
        return error?.code === 'ESRCH';
      }
    };
    const finish = () => {
      if (finished) return;
      finished = true;
      if (forceTimer) clearTimeout(forceTimer);
      if (verificationTimer) clearInterval(verificationTimer);
      child.off('close', finish);
      child.off('error', forceStop);
      resolve();
    };
    const forceStop = () => {
      if (finished) return;
      if (childHasExited()) {
        finish();
        return;
      }
      try {
        child.kill('SIGKILL');
      } catch {}
      if (!verificationTimer) {
        verificationTimer = setInterval(() => {
          if (childHasExited()) finish();
          else {
            try { child.kill('SIGKILL'); } catch {}
          }
        }, 250);
      }
    };
    child.once('close', finish);
    child.once('error', forceStop);
    try {
      child.kill('SIGTERM');
    } catch {
      forceStop();
    }
    forceTimer = setTimeout(forceStop, 2000);
  });
}

function stopRendererServer() {
  const server = rendererServer;
  rendererServer = null;
  if (!server) return Promise.resolve();
  return new Promise(resolve => {
    let finished = false;
    const finish = () => {
      if (finished) return;
      finished = true;
      resolve();
    };
    try {
      const result = server.close(finish);
      if (result && typeof result.then === 'function') {
        result.then(finish, finish);
      }
    } catch {
      finish();
    }
    setTimeout(finish, 2000);
  });
}

async function startApplication() {
  launchToken = createLaunchToken();
  if (isDev) {
    const viteServer = await startViteDevServer(path.join(__dirname, '..'));
    rendererServer = viteServer.server;
    rendererOrigin = viteServer.origin;
  } else {
    const staticServer = await startStaticServer(path.join(__dirname, '../build'));
    rendererServer = staticServer.server;
    rendererOrigin = staticServer.origin;
  }
  configureRendererNetworkBoundary();
  await startBackend();
  createWindow();
}

app.whenReady()
  .then(startApplication)
  .catch(() => {
    dialog.showErrorBox(
      'AUTO MIXER startup failed',
      'The secured local backend could not be started.'
    );
    app.quit();
  });

app.on('window-all-closed', () => {
  app.quit();
});

app.on('activate', () => {
  if (mainWindow === null && rendererOrigin) createWindow();
});

app.on('before-quit', event => {
  if (shutdownComplete) return;
  event.preventDefault();
  quitting = true;
  launchToken = null;
  if (!shutdownPromise) {
    shutdownPromise = Promise.allSettled([
      stopBackend(),
      stopRendererServer(),
    ]).finally(() => {
      shutdownComplete = true;
      app.quit();
    });
  }
});
