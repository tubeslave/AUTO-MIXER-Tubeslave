'use strict';

const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('electron', {
  getWebSocketAuthBootstrap: () => ipcRenderer.invoke('get-websocket-auth-bootstrap')
});
