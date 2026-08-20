import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import { BaseWebSocketTransport, inferWebSocketUrl } from '../websocketCore.mjs';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const srcRoot = path.resolve(__dirname, '../..');

test('dLive UI and websocket preserve attended console identity fields', () => {
  const websocketSource = fs.readFileSync(path.join(srcRoot, 'services/websocket.js'), 'utf8');
  const mobileSource = fs.readFileSync(path.join(srcRoot, 'components/IPhoneControlSurface.js'), 'utf8');
  const settingsSource = fs.readFileSync(path.join(srcRoot, 'components/SettingsTab.js'), 'utf8');

  assert.match(websocketSource, /user_profile: userProfile/);
  assert.match(websocketSource, /midi_base_channel: midiBaseChannel/);
  assert.match(mobileSource, /dliveMidiBaseChannel/);
  assert.match(mobileSource, /dliveUserProfile/);
  assert.doesNotMatch(mobileSource, /connectDLive\([^;]*dliveTls,\s*0,\s*dliveEndpoint/s);
  assert.match(settingsSource, /dliveIp: '192\.168\.3\.71'/);
  assert.match(settingsSource, /dliveMidiChannel: 11/);
});

class FakeWebSocket {
  static CONNECTING = 0;
  static OPEN = 1;
  static CLOSING = 2;
  static CLOSED = 3;
  static instances = [];

  constructor(url) {
    this.url = url;
    this.readyState = FakeWebSocket.CONNECTING;
    this.sentMessages = [];
    this.onopen = null;
    this.onmessage = null;
    this.onerror = null;
    this.onclose = null;
    this.closeCode = null;
    this.closeReason = null;
    FakeWebSocket.instances.push(this);
  }

  send(message) {
    this.sentMessages.push(message);
  }

  close(code = 1000, reason = '') {
    this.closeCode = code;
    this.closeReason = reason;
    this.readyState = FakeWebSocket.CLOSED;
    if (this.onclose) {
      this.onclose({ code, reason });
    }
  }

  fail(error = new Error('fake socket error')) {
    if (this.onerror) {
      this.onerror(error);
    }
  }

  open() {
    this.readyState = FakeWebSocket.OPEN;
    if (this.onopen) {
      this.onopen();
    }
  }

  emitMessage(payload) {
    if (this.onmessage) {
      this.onmessage({ data: JSON.stringify(payload) });
    }
  }
}

function createTransport(overrides = {}) {
  return new BaseWebSocketTransport({
    reconnectInterval: 5,
    WebSocketImpl: FakeWebSocket,
    ...overrides
  });
}

test.beforeEach(() => {
  FakeWebSocket.instances = [];
  delete globalThis.location;
  delete globalThis.localStorage;
  delete globalThis.sessionStorage;
});

test('reuses the same pending connect promise and resolves on open', async () => {
  const transport = createTransport();

  const firstConnect = transport.connect();
  const secondConnect = transport.connect();

  assert.equal(firstConnect, secondConnect);
  assert.equal(FakeWebSocket.instances.length, 1);

  FakeWebSocket.instances[0].open();
  await firstConnect;
  assert.equal(transport.connectPromise, null);
});

test('notifies listeners and removes them with off()', async () => {
  const transport = createTransport();
  const received = [];
  const handler = (payload) => received.push(payload.status);

  transport.on('connection_status', handler);
  const connectPromise = transport.connect();
  FakeWebSocket.instances[0].open();
  await connectPromise;

  FakeWebSocket.instances[0].emitMessage({
    type: 'connection_status',
    status: 'connected'
  });
  transport.off('connection_status', handler);
  FakeWebSocket.instances[0].emitMessage({
    type: 'connection_status',
    status: 'disconnected'
  });

  assert.deepEqual(received, ['connected']);
});

test('disconnect() disables scheduled reconnect after close', async () => {
  const transport = createTransport();
  const connectPromise = transport.connect();
  const socket = FakeWebSocket.instances[0];

  socket.open();
  await connectPromise;
  transport.disconnect();

  assert.equal(transport.shouldReconnect, false);
  assert.equal(transport.reconnectTimer, null);

  await new Promise(resolve => setTimeout(resolve, 20));
  assert.equal(FakeWebSocket.instances.length, 1);
});

test('emits connected again after automatic reconnect', async () => {
  const transport = createTransport();
  const events = [];
  transport.on('connected', payload => events.push(`connected:${payload.url}`));
  transport.on('disconnected', () => events.push('disconnected'));

  const firstConnect = transport.connect();
  FakeWebSocket.instances[0].open();
  await firstConnect;
  FakeWebSocket.instances[0].close();

  await new Promise(resolve => setTimeout(resolve, 10));
  assert.equal(FakeWebSocket.instances.length, 2);
  FakeWebSocket.instances[1].open();
  await new Promise(resolve => setTimeout(resolve, 0));

  assert.deepEqual(events, [
    'connected:ws://localhost:8765',
    'disconnected',
    'connected:ws://localhost:8765'
  ]);
});

test('close 1009 preserves code and reason, stops reconnect, and requires manual retry', async () => {
  const transport = createTransport();
  const terminalEvents = [];
  const disconnectedEvents = [];
  transport.on('transport_terminal_close', payload => terminalEvents.push(payload));
  transport.on('disconnected', payload => disconnectedEvents.push(payload));

  const firstConnect = transport.connect();
  const firstSocket = FakeWebSocket.instances[0];
  firstSocket.open();
  await firstConnect;
  firstSocket.close(1009, 'decision_queue_frame_too_large');

  await new Promise(resolve => setTimeout(resolve, 15));
  assert.equal(FakeWebSocket.instances.length, 1);
  assert.equal(transport.shouldReconnect, false);
  assert.deepEqual(transport.getTerminalClose(), {
    code: 1009,
    reason: 'decision_queue_frame_too_large',
    manualRetryRequired: true
  });
  assert.deepEqual(terminalEvents, [{
    code: 1009,
    reason: 'decision_queue_frame_too_large',
    manual_retry_required: true
  }]);
  assert.deepEqual(disconnectedEvents.at(-1), {
    code: 1009,
    reason: 'decision_queue_frame_too_large',
    terminal: true,
    manual_retry_required: true
  });
  await assert.rejects(transport.connect(), /manual retry required/);
  assert.equal(transport.send({ type: 'get_decision_queue' }), false);

  const retry = transport.retryTerminalClose();
  const replacement = FakeWebSocket.instances[1];
  replacement.open();
  await retry;
  assert.equal(transport.isReady(), true);
  assert.equal(transport.getTerminalClose(), null);
});

test('stale close does not clear an active replacement socket', async () => {
  const transport = createTransport();

  const firstConnect = transport.connect();
  const staleSocket = FakeWebSocket.instances[0];
  staleSocket.open();
  await firstConnect;

  transport.ws = null;
  const secondConnect = transport.connect();
  const activeSocket = FakeWebSocket.instances[1];
  activeSocket.open();
  await secondConnect;

  staleSocket.close();

  assert.equal(transport.ws, activeSocket);
});

test('stale connecting socket error after disconnect is not logged', async () => {
  const transport = createTransport();
  const originalError = console.error;
  const consoleErrors = [];
  console.error = (...args) => consoleErrors.push(args);

  try {
    const connectPromise = transport.connect();
    const socket = FakeWebSocket.instances[0];
    transport.disconnect();
    assert.equal(socket.readyState, FakeWebSocket.CONNECTING);
    socket.fail();

    await assert.rejects(connectPromise);
    assert.deepEqual(consoleErrors, []);
  } finally {
    console.error = originalError;
  }
});

test('stale connecting socket open after disconnect closes without taking ownership', async () => {
  const transport = createTransport();

  const connectPromise = transport.connect();
  const socket = FakeWebSocket.instances[0];
  transport.disconnect();
  socket.open();

  await assert.rejects(connectPromise);
  assert.equal(socket.readyState, FakeWebSocket.CLOSED);
  assert.equal(transport.ws, null);
});

test('send() serializes messages when socket is open', async () => {
  const transport = createTransport();
  const connectPromise = transport.connect();
  const socket = FakeWebSocket.instances[0];

  socket.open();
  await connectPromise;
  transport.send({ type: 'ping', id: 7 });

  assert.deepEqual(socket.sentMessages, ['{"type":"ping","id":7}']);
});

test('authentication frame is always first and token persistence requests are ignored', async () => {
  const store = new Map();
  globalThis.sessionStorage = {
    getItem: key => store.get(key) || null,
    setItem: (key, value) => store.set(key, value),
    removeItem: key => store.delete(key)
  };
  const transport = createTransport({ authToken: 'secret-test-token' });
  const connectPromise = transport.connect();
  const socket = FakeWebSocket.instances[0];

  socket.open();
  assert.equal(transport.isReady(), false);
  assert.deepEqual(socket.sentMessages, [
    '{"type":"authenticate","token":"secret-test-token","client_id":"automixer-ui"}'
  ]);
  socket.emitMessage({ type: 'authentication_status', authenticated: true });
  await connectPromise;
  assert.equal(transport.isReady(), true);
  transport.send({ type: 'ping' });

  assert.deepEqual(socket.sentMessages, [
    '{"type":"authenticate","token":"secret-test-token","client_id":"automixer-ui"}',
    '{"type":"ping"}'
  ]);
  transport.setAuthToken('next-token', { persist: true, reconnect: false });
  assert.equal(transport.authToken, 'next-token');
  assert.equal(store.has('automixer_ws_auth_token'), false);
});

test('authenticated connection stays pending and blocks application frames before auth ACK', async () => {
  const transport = createTransport({ authToken: 'launch-only-token' });
  const connectedEvents = [];
  transport.on('connected', payload => connectedEvents.push(payload));
  const connectPromise = transport.connect();
  const socket = FakeWebSocket.instances[0];

  socket.open();
  assert.equal(transport.isReady(), false);
  assert.equal(transport.send({ type: 'get_dashboard_snapshot' }), false);
  assert.equal(connectedEvents.length, 0);
  assert.deepEqual(socket.sentMessages, [
    '{"type":"authenticate","token":"launch-only-token","client_id":"automixer-ui"}'
  ]);

  socket.emitMessage({ type: 'authentication_status', authenticated: true });
  await connectPromise;

  assert.equal(transport.isReady(), true);
  assert.equal(connectedEvents.length, 1);
  assert.equal(connectedEvents[0].authenticated, true);
  assert.equal(transport.send({ type: 'get_dashboard_snapshot' }), true);
  assert.equal(JSON.parse(socket.sentMessages[1]).type, 'get_dashboard_snapshot');
});

test('negative authentication status fails closed, clears token, and emits one safe event', async () => {
  const transport = createTransport({ authToken: 'wrong-token' });
  const connectedEvents = [];
  const authenticationFailures = [];
  transport.on('connected', payload => connectedEvents.push(payload));
  transport.on('authentication_failed', payload => authenticationFailures.push(payload));
  const connectPromise = transport.connect();
  const socket = FakeWebSocket.instances[0];

  socket.open();
  socket.emitMessage({ type: 'authentication_status', authenticated: false });

  await assert.rejects(connectPromise, /authentication failed/);
  assert.equal(transport.isReady(), false);
  assert.equal(transport.shouldReconnect, false);
  assert.equal(socket.closeCode, 1008);
  assert.equal(connectedEvents.length, 0);
  assert.equal(transport.authToken, '');
  assert.deepEqual(authenticationFailures, [{ code: 'authentication_failed' }]);
});

test('new memory-only token can recover after terminal authentication failure', async () => {
  const transport = createTransport({ authToken: 'expired-token' });
  const firstConnect = transport.connect();
  const firstSocket = FakeWebSocket.instances[0];
  firstSocket.open();
  firstSocket.emitMessage({ type: 'authentication_status', authenticated: false });
  await assert.rejects(firstConnect);

  transport.setAuthToken('fresh-operator-token', { persist: false, reconnect: false });
  const secondConnect = transport.connect();
  const secondSocket = FakeWebSocket.instances[1];
  secondSocket.open();
  secondSocket.emitMessage({ type: 'authentication_status', authenticated: true });
  await secondConnect;

  assert.equal(transport.isReady(), true);
  assert.equal(JSON.parse(secondSocket.sentMessages[0]).token, 'fresh-operator-token');
});

test('active authenticated session expiry clears token and returns to enrollment', async () => {
  const transport = createTransport({ authToken: 'e'.repeat(43) });
  const failures = [];
  transport.on('authentication_failed', payload => failures.push(payload));
  const connectPromise = transport.connect();
  const socket = FakeWebSocket.instances[0];
  socket.open();
  socket.emitMessage({ type: 'authentication_status', authenticated: true });
  await connectPromise;

  socket.close(1008, 'authentication_expired');
  await new Promise(resolve => setTimeout(resolve, 10));

  assert.equal(transport.isReady(), false);
  assert.equal(transport.authToken, '');
  assert.equal(transport.shouldReconnect, false);
  assert.deepEqual(failures, [{ code: 'authentication_failed' }]);
  assert.equal(FakeWebSocket.instances.length, 1);
});

test('backend close-only policy rejection is terminal and does not reconnect', async () => {
  const transport = createTransport({ authToken: 'backend-rejected-token' });
  const connectPromise = transport.connect();
  const socket = FakeWebSocket.instances[0];

  socket.open();
  socket.close(1008, 'authentication_failed');

  await assert.rejects(connectPromise, /closed before becoming ready/);
  await new Promise(resolve => setTimeout(resolve, 10));
  assert.equal(transport.isReady(), false);
  assert.equal(transport.shouldReconnect, false);
  assert.equal(FakeWebSocket.instances.length, 1);
});

test('missing authentication ACK times out and closes fail-closed', async () => {
  const transport = createTransport({
    authToken: 'unacknowledged-token',
    authenticationTimeout: 5
  });
  const connectPromise = transport.connect();
  const socket = FakeWebSocket.instances[0];

  socket.open();

  await assert.rejects(connectPromise, /authentication timed out/);
  assert.equal(transport.isReady(), false);
  assert.equal(transport.shouldReconnect, false);
  assert.equal(socket.closeCode, 1008);
});

test('pre-auth transport close retries and becomes ready after a later auth ACK', async () => {
  const transport = createTransport({ authToken: 'delayed-backend-token' });
  const events = [];
  transport.on('connected', () => events.push('connected'));
  const firstConnect = transport.connect();
  const firstSocket = FakeWebSocket.instances[0];

  firstSocket.open();
  firstSocket.close(1006, 'backend_not_ready');
  await assert.rejects(firstConnect, /closed before becoming ready/);

  await new Promise(resolve => setTimeout(resolve, 10));
  assert.equal(FakeWebSocket.instances.length, 2);
  const secondSocket = FakeWebSocket.instances[1];
  secondSocket.open();
  secondSocket.emitMessage({ type: 'authentication_status', authenticated: true });
  await new Promise(resolve => setTimeout(resolve, 0));

  assert.equal(transport.isReady(), true);
  assert.deepEqual(events, ['connected']);
  assert.equal(JSON.parse(secondSocket.sentMessages[0]).type, 'authenticate');
});

test('stale session token is ignored unless explicitly supplied by the caller', async () => {
  const store = new Map([['automixer_ws_auth_token', 'stale-token']]);
  globalThis.sessionStorage = {
    getItem: key => store.get(key) || null,
    setItem: (key, value) => store.set(key, value),
    removeItem: key => store.delete(key)
  };
  const transport = createTransport();
  const connectPromise = transport.connect();
  const socket = FakeWebSocket.instances[0];

  socket.open();
  await connectPromise;

  assert.equal(transport.authToken, '');
  assert.deepEqual(socket.sentMessages, []);
  assert.equal(transport.isReady(), true);
});

test('Electron-style memory-only token bootstrap never persists to session storage', () => {
  const store = new Map([['automixer_ws_auth_token', 'legacy-stored-token']]);
  globalThis.sessionStorage = {
    getItem: key => store.get(key) || null,
    setItem: (key, value) => store.set(key, value),
    removeItem: key => store.delete(key)
  };
  const transport = createTransport();

  transport.setAuthToken('memory-only-launch-token', {
    persist: false,
    reconnect: false
  });

  assert.equal(transport.authToken, 'memory-only-launch-token');
  assert.equal(store.has('automixer_ws_auth_token'), false);
});

test('infers websocket host from browser page host', () => {
  Object.defineProperty(globalThis, 'location', {
    configurable: true,
    value: { protocol: 'http:', hostname: '192.168.1.20' }
  });

  assert.equal(inferWebSocketUrl(), 'ws://192.168.1.20:8765');
});

test('setUrl normalizes in memory and purges legacy saved endpoint', () => {
  const store = new Map([['automixer_ws_url', 'ws://stale.example:8765']]);
  globalThis.localStorage = {
    getItem: key => store.get(key) || null,
    setItem: (key, value) => store.set(key, value),
    removeItem: key => store.delete(key)
  };

  const transport = createTransport({ url: 'ws://localhost:8765' });
  transport.setUrl('192.168.1.30:8765', { reconnect: false });

  assert.equal(transport.getUrl(), 'ws://192.168.1.30:8765');
  assert.equal(store.has('automixer_ws_url'), false);
});

test('insecure or unauthenticated remote endpoints are blocked before WebSocket construction', async () => {
  Object.defineProperty(globalThis, 'location', {
    configurable: true,
    value: {
      protocol: 'http:',
      hostname: '192.168.1.20',
      host: '192.168.1.20'
    }
  });
  const insecure = createTransport({ url: 'ws://192.168.1.20:8765' });
  await assert.rejects(insecure.connect(), /policy blocked/);
  assert.equal(FakeWebSocket.instances.length, 0);

  Object.defineProperty(globalThis, 'location', {
    configurable: true,
    value: {
      protocol: 'https:',
      hostname: 'mix.example',
      host: 'mix.example'
    }
  });
  const missingToken = createTransport({ url: 'wss://mix.example/ws' });
  await assert.rejects(missingToken.connect(), /policy blocked/);
  assert.equal(FakeWebSocket.instances.length, 0);
});

test('same-origin HTTPS WSS remote authenticates without writing browser storage', async () => {
  const localStore = new Map([['automixer_ws_url', 'ws://stale.example:8765']]);
  const sessionStore = new Map([['automixer_ws_auth_token', 'stale-token']]);
  globalThis.localStorage = {
    getItem: key => localStore.get(key) || null,
    setItem: (key, value) => localStore.set(key, value),
    removeItem: key => localStore.delete(key)
  };
  globalThis.sessionStorage = {
    getItem: key => sessionStore.get(key) || null,
    setItem: (key, value) => sessionStore.set(key, value),
    removeItem: key => sessionStore.delete(key)
  };
  Object.defineProperty(globalThis, 'location', {
    configurable: true,
    value: {
      protocol: 'https:',
      hostname: 'mix.example',
      host: 'mix.example'
    }
  });
  assert.equal(inferWebSocketUrl(), 'wss://mix.example/ws');
  const transport = createTransport({
    url: inferWebSocketUrl(),
    authToken: 'r'.repeat(43)
  });
  const connectPromise = transport.connect();
  const socket = FakeWebSocket.instances[0];
  socket.open();
  socket.emitMessage({ type: 'authentication_status', authenticated: true });
  await connectPromise;

  assert.equal(transport.isReady(), true);
  assert.equal(localStore.size, 0);
  assert.equal(sessionStore.size, 0);
});

test('remote page cannot redirect its bearer token to a loopback WebSocket', async () => {
  Object.defineProperty(globalThis, 'location', {
    configurable: true,
    value: {
      protocol: 'https:',
      hostname: 'mix.example',
      host: 'mix.example'
    }
  });
  const transport = createTransport({
    url: 'ws://127.0.0.1:8765',
    authToken: 'r'.repeat(43)
  });
  await assert.rejects(transport.connect(), /policy blocked/);
  assert.equal(FakeWebSocket.instances.length, 0);
});

test('privacy-restricted throwing storage cannot break memory-only bootstrap', () => {
  globalThis.localStorage = {
    removeItem: () => { throw new Error('storage blocked'); }
  };
  globalThis.sessionStorage = {
    removeItem: () => { throw new Error('storage blocked'); }
  };
  assert.doesNotThrow(() => createTransport({ authToken: 'e'.repeat(43) }));
});

test('main cockpit source does not call direct live write websocket methods', () => {
  const cockpitSources = [
    path.join(srcRoot, 'AutomixerPrototype.tsx'),
    path.join(srcRoot, 'App.js'),
    path.join(srcRoot, 'services/useOperatorBackend.ts'),
    path.join(srcRoot, 'components/GainStagingTab.js'),
    path.join(srcRoot, 'components/IPhoneControlSurface.js')
  ];
  const forbiddenCalls = [
    '.setFader(',
    '.setGain(',
    '.setEQ(',
    '.setCompressor(',
    '.routeOutput(',
    '.loadSnap(',
    '.saveSnap(',
    '.resetTrim(',
    '.bypassMixer('
  ];

  for (const sourcePath of cockpitSources) {
    const source = fs.readFileSync(sourcePath, 'utf8');
    for (const call of forbiddenCalls) {
      assert.equal(source.includes(call), false, `${path.basename(sourcePath)} must not call ${call}`);
    }
  }

  const operatorBackend = fs.readFileSync(path.join(srcRoot, 'services/useOperatorBackend.ts'), 'utf8');
  assert.match(operatorBackend, /applyOperatorProposal\(/);
  assert.match(operatorBackend, /websocketService\.emergencyStop\(\)/);

  const websocketSource = fs.readFileSync(path.join(srcRoot, 'services/websocket.js'), 'utf8');
  assert.match(websocketSource, /emergencyStop\(\)\s*\{\s*return this\.send\(\{\s*type: 'emergency_stop',\s*reason: 'frontend_operator_emergency_stop'/s);

  const cockpitShell = fs.readFileSync(path.join(srcRoot, 'components/prototype/AutomixerComponents.tsx'), 'utf8');
  assert.equal((cockpitShell.match(/<EmergencyStopControl/g) || []).length, 2, 'desktop and compact shells must both expose emergency stop');
  assert.match(cockpitShell, /Confirm system emergency stop/);
});
