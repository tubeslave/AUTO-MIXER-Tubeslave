import {
  assertWebSocketConnectionPolicy
} from './remoteAuthPolicy.mjs';

const DEFAULT_WS_PORT = 8765;
const WS_URL_STORAGE_KEY = 'automixer_ws_url';
const WS_AUTH_TOKEN_STORAGE_KEY = 'automixer_ws_auth_token';

function safeLocalStorage() {
  try {
    return globalThis.localStorage || null;
  } catch {
    return null;
  }
}

function safeSessionStorage() {
  try {
    return globalThis.sessionStorage || null;
  } catch {
    return null;
  }
}

export function inferWebSocketUrl() {
  const location = globalThis.location;
  if (location?.protocol?.startsWith('http') && location.hostname) {
    if (location.protocol === 'https:') return `wss://${location.host}/ws`;
    return `ws://${location.hostname}:${DEFAULT_WS_PORT}`;
  }

  return `ws://localhost:${DEFAULT_WS_PORT}`;
}

function purgeLegacyConnectionStorage() {
  try {
    safeLocalStorage()?.removeItem?.(WS_URL_STORAGE_KEY);
    safeLocalStorage()?.removeItem?.(WS_AUTH_TOKEN_STORAGE_KEY);
    safeSessionStorage()?.removeItem?.(WS_AUTH_TOKEN_STORAGE_KEY);
  } catch {
    // Storage may be disabled by browser privacy policy. Runtime credentials
    // still remain memory-only and transport setup must continue fail-closed.
  }
}

function normalizeWebSocketUrl(url) {
  const trimmed = String(url || '').trim();
  if (!trimmed) return inferWebSocketUrl();
  if (trimmed.startsWith('ws://') || trimmed.startsWith('wss://')) return trimmed;
  if (trimmed.startsWith('http://')) return trimmed.replace(/^http:\/\//, 'ws://');
  if (trimmed.startsWith('https://')) return trimmed.replace(/^https:\/\//, 'wss://');
  return `ws://${trimmed}`;
}

export class BaseWebSocketTransport {
  constructor({
    url = inferWebSocketUrl(),
    reconnectInterval = 3000,
    WebSocketImpl = globalThis.WebSocket,
    authToken = '',
    clientId = 'automixer-ui',
    authenticationTimeout = 5000
  } = {}) {
    this.ws = null;
    this.url = normalizeWebSocketUrl(url);
    this.listeners = new Map();
    this.reconnectInterval = reconnectInterval;
    this.reconnectTimer = null;
    this.connectPromise = null;
    this.shouldReconnect = true;
    this.WebSocketImpl = WebSocketImpl;
    this.authToken = String(authToken || '');
    this.clientId = String(clientId || 'automixer-ui');
    this.authenticationTimeout = Math.max(1, Number(authenticationTimeout) || 5000);
    this.authenticationTimer = null;
    this.ready = false;
    this._pendingConnectReject = null;
    purgeLegacyConnectionStorage();
  }

  getUrl() {
    return this.url;
  }

  setUrl(url, { reconnect = true } = {}) {
    const nextUrl = normalizeWebSocketUrl(url);
    if (nextUrl === this.url) return;

    this.url = nextUrl;
    purgeLegacyConnectionStorage();

    if (reconnect) {
      this.disconnect();
      this.shouldReconnect = true;
      this.connect().catch(err => console.error('Reconnect failed:', err));
    }
  }

  setAuthToken(token, { reconnect = true } = {}) {
    this.authToken = String(token || '');
    // Bearer tokens and operator-selected endpoints are memory-only. Purge
    // values left by older builds from both browser storage scopes.
    purgeLegacyConnectionStorage();
    if (reconnect) {
      this.disconnect();
      this.shouldReconnect = true;
      this.connect().catch(err => console.error('Reconnect failed:', err));
    }
  }

  isReady() {
    return Boolean(
      this.ready
      && this.ws
      && this.ws.readyState === this.WebSocketImpl.OPEN
    );
  }

  clearAuthenticationTimer() {
    if (this.authenticationTimer) {
      clearTimeout(this.authenticationTimer);
      this.authenticationTimer = null;
    }
  }

  connect() {
    if (this.isReady()) {
      return Promise.resolve();
    }

    if (this.connectPromise) {
      return this.connectPromise;
    }

    if (!this.WebSocketImpl) {
      return Promise.reject(new Error('WebSocket API is not available'));
    }

    try {
      assertWebSocketConnectionPolicy({
        url: this.url,
        authToken: this.authToken,
        locationLike: globalThis.location
      });
    } catch {
      this.shouldReconnect = false;
      this.notifyListeners('connection_policy_blocked', {
        code: 'connection_policy_blocked'
      });
      return Promise.reject(new Error('WebSocket connection policy blocked'));
    }

    this.shouldReconnect = true;
    this.ready = false;
    const authToken = this.authToken;
    const clientId = this.clientId;
    let resolveConnect;
    let rejectConnect;
    let settled = false;
    let authenticationRejected = false;
    let authenticationFailureNotified = false;
    const promise = new Promise((resolve, reject) => {
      resolveConnect = resolve;
      rejectConnect = reject;
    });
    this.connectPromise = promise;

    const settleSuccess = () => {
      if (settled) return;
      settled = true;
      this._pendingConnectReject = null;
      if (this.connectPromise === promise) this.connectPromise = null;
      resolveConnect();
    };
    const settleFailure = (error) => {
      if (settled) return;
      settled = true;
      this._pendingConnectReject = null;
      if (this.connectPromise === promise) this.connectPromise = null;
      rejectConnect(error instanceof Error ? error : new Error('WebSocket connection failed'));
    };
    this._pendingConnectReject = settleFailure;

    let socket;
    try {
      socket = new this.WebSocketImpl(this.url);
      this.ws = socket;
    } catch (error) {
      settleFailure(error);
      return promise;
    }

    const markReady = () => {
      if (this.ws !== socket || !this.shouldReconnect) {
        settleFailure(new Error('WebSocket connection cancelled'));
        return;
      }
      this.clearAuthenticationTimer();
      this.ready = true;
      if (this.reconnectTimer) {
        clearTimeout(this.reconnectTimer);
        this.reconnectTimer = null;
      }
      this.notifyListeners('connected', {
        url: this.url,
        authenticated: Boolean(authToken)
      });
      settleSuccess();
    };

    const notifyAuthenticationFailure = () => {
      if (authenticationFailureNotified) return;
      authenticationFailureNotified = true;
      this.authToken = '';
      purgeLegacyConnectionStorage();
      this.notifyListeners('authentication_failed', {
        code: 'authentication_failed'
      });
    };

    const failAuthentication = (reason) => {
      this.ready = false;
      authenticationRejected = true;
      this.shouldReconnect = false;
      this.clearAuthenticationTimer();
      notifyAuthenticationFailure();
      settleFailure(new Error(reason));
      if (
        this.ws === socket
        && socket.readyState !== this.WebSocketImpl.CLOSED
        && socket.readyState !== this.WebSocketImpl.CLOSING
      ) {
        socket.close(1008, reason);
      }
    };

    socket.onopen = () => {
      if (this.ws !== socket || !this.shouldReconnect) {
        socket.close();
        settleFailure(new Error('WebSocket connection cancelled'));
        return;
      }
      if (!authToken) {
        markReady();
        return;
      }
      socket.send(JSON.stringify({
        type: 'authenticate',
        token: authToken,
        client_id: clientId
      }));
      this.authenticationTimer = setTimeout(() => {
        failAuthentication('WebSocket authentication timed out');
      }, this.authenticationTimeout);
    };

    socket.onmessage = (event) => {
      let data;
      try {
        data = JSON.parse(event.data);
      } catch {
        if (authToken && !this.ready) {
          failAuthentication('Invalid authentication response');
        } else {
          console.error('Invalid WebSocket JSON response.');
        }
        return;
      }

      if (authToken && !this.ready) {
        if (data.type !== 'authentication_status' || data.authenticated !== true) {
          failAuthentication('WebSocket authentication failed');
          return;
        }
        markReady();
        this.notifyListeners(data.type, data);
        return;
      }

      this.notifyListeners(data.type, data);
    };

    socket.onerror = (error) => {
      if (this.ws !== socket || !this.shouldReconnect) {
        settleFailure(error);
        return;
      }
      console.error('WebSocket transport error.');
      settleFailure(error);
    };

    socket.onclose = (event) => {
      if (this.ws !== socket) return;
      if (authToken && Number(event?.code) === 1008) {
        authenticationRejected = true;
        this.shouldReconnect = false;
        notifyAuthenticationFailure();
      }
      this.ready = false;
      this.ws = null;
      this.clearAuthenticationTimer();
      if (!settled) {
        settleFailure(new Error('WebSocket closed before becoming ready'));
      }
      this.notifyListeners('disconnected', {});
      if (this.shouldReconnect && !authenticationRejected) {
        this.scheduleReconnect();
      }
    };

    return promise;
  }

  scheduleReconnect() {
    if (!this.shouldReconnect) {
      return;
    }

    if (!this.reconnectTimer) {
      this.reconnectTimer = setTimeout(() => {
        this.reconnectTimer = null;
        console.log('Attempting to reconnect...');
        this.connect().catch(err => console.error('Reconnect failed:', err));
      }, this.reconnectInterval);
    }
  }

  disconnect() {
    this.shouldReconnect = false;
    this.ready = false;
    this.clearAuthenticationTimer();
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }
    if (this._pendingConnectReject) {
      this._pendingConnectReject(new Error('WebSocket connection cancelled'));
    }
    if (this.ws) {
      const socket = this.ws;
      this.ws = null;
      this.connectPromise = null;
      if (socket.readyState === this.WebSocketImpl.CONNECTING) {
        return;
      }
      socket.close();
      this.notifyListeners('disconnected', {});
    }
  }

  send(message) {
    if (this.isReady()) {
      try {
        const jsonMessage = JSON.stringify(message);
        this.ws.send(jsonMessage);
        return true;
      } catch (error) {
        console.error('Error sending WebSocket message:', error);
        return false;
      }
    } else {
      const state = this.ws ? this.ws.readyState : 'null';
      console.error('WebSocket is not authenticated and ready. ReadyState:', state);
      if (state === 0) {
        console.warn('WebSocket is still connecting...');
      } else if (state === 2 || state === 3) {
        console.warn('WebSocket is closed. Attempting to reconnect...');
        this.connect().catch(err => console.error('Reconnect failed:', err));
      }
      return false;
    }
  }

  on(eventType, callback) {
    if (!this.listeners.has(eventType)) {
      this.listeners.set(eventType, []);
    }
    this.listeners.get(eventType).push(callback);
  }

  off(eventType, callback) {
    if (this.listeners.has(eventType)) {
      const callbacks = this.listeners.get(eventType);
      const index = callbacks.indexOf(callback);
      if (index > -1) {
        callbacks.splice(index, 1);
      }
      if (callbacks.length === 0) {
        this.listeners.delete(eventType);
      }
    }
  }

  notifyListeners(eventType, data) {
    if (this.listeners.has(eventType)) {
      this.listeners.get(eventType).forEach(callback => {
        try {
          callback(data);
        } catch (error) {
          console.error(`Error in listener for ${eventType}:`, error);
        }
      });
    }
  }
}
