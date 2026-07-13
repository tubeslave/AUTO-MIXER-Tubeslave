import React from 'react';
import ReactDOM from 'react-dom/client';
import './index.css';
import AutomixerPrototype from './AutomixerPrototype';
import RemoteAuthGate from './components/RemoteAuthGate';
import websocketService from './services/websocket';

const rootElement = document.getElementById('root') as HTMLElement;

async function bootstrapElectronTransport() {
  const electron = window.electron;
  if (!electron) return false;
  if (typeof electron.getWebSocketAuthBootstrap !== 'function') {
    throw new Error('Secure Electron bootstrap API is unavailable');
  }
  const bootstrap = await electron.getWebSocketAuthBootstrap();
  if (
    !bootstrap
    || !/^[A-Za-z0-9_-]{43}$/.test(bootstrap.token)
    || bootstrap.clientId !== 'automixer-ui'
    || bootstrap.websocketUrl !== 'ws://127.0.0.1:8765'
    || typeof bootstrap.packagedOfflineLock !== 'boolean'
  ) {
    throw new Error('Secure Electron bootstrap payload is invalid');
  }
  websocketService.setUrl(bootstrap.websocketUrl, {
    reconnect: false
  });
  websocketService.setAuthToken(bootstrap.token, {
    reconnect: false
  });
  Object.defineProperty(window, 'automixerRuntimeSafety', {
    value: Object.freeze({
      packagedOfflineLock: bootstrap.packagedOfflineLock
    }),
    writable: false,
    configurable: false
  });
  return true;
}

async function startApplication() {
  let electronBootstrapped = false;
  try {
    electronBootstrapped = await bootstrapElectronTransport();
  } catch {
    rootElement.textContent = 'Secure local backend bootstrap failed. Application remains offline.';
    rootElement.setAttribute('role', 'alert');
    return;
  }

  ReactDOM.createRoot(rootElement).render(
    <React.StrictMode>
      {electronBootstrapped ? (
        <AutomixerPrototype />
      ) : (
        <RemoteAuthGate>
          <AutomixerPrototype />
        </RemoteAuthGate>
      )}
    </React.StrictMode>
  );
}

void startApplication();
