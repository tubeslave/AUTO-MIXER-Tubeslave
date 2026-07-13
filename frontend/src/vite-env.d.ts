/// <reference types="vite/client" />

interface ElectronWebSocketBootstrap {
  token: string;
  clientId: 'automixer-ui';
  websocketUrl: 'ws://127.0.0.1:8765';
  packagedOfflineLock: boolean;
}

interface Window {
  electron?: {
    getWebSocketAuthBootstrap: () => Promise<ElectronWebSocketBootstrap>;
  };
  automixerRuntimeSafety?: Readonly<{
    packagedOfflineLock: boolean;
  }>;
}
