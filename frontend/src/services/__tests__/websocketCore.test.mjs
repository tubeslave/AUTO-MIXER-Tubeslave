import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import { BaseWebSocketTransport, inferWebSocketUrl } from '../websocketCore.mjs';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const srcRoot = path.resolve(__dirname, '../..');

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
    FakeWebSocket.instances.push(this);
  }

  send(message) {
    this.sentMessages.push(message);
  }

  close() {
    this.readyState = FakeWebSocket.CLOSED;
    if (this.onclose) {
      this.onclose();
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

test('infers websocket host from browser page host', () => {
  Object.defineProperty(globalThis, 'location', {
    configurable: true,
    value: { protocol: 'http:', hostname: '192.168.1.20' }
  });

  assert.equal(inferWebSocketUrl(), 'ws://192.168.1.20:8765');
});

test('setUrl normalizes and persists manual websocket host', () => {
  const store = new Map();
  globalThis.localStorage = {
    getItem: key => store.get(key) || null,
    setItem: (key, value) => store.set(key, value)
  };

  const transport = createTransport({ url: 'ws://localhost:8765' });
  transport.setUrl('192.168.1.30:8765', { reconnect: false });

  assert.equal(transport.getUrl(), 'ws://192.168.1.30:8765');
  assert.equal(store.get('automixer_ws_url'), 'ws://192.168.1.30:8765');
});

test('main cockpit source does not call direct live write websocket methods', () => {
  const cockpitSources = [
    path.join(srcRoot, 'AutomixerPrototype.tsx'),
    path.join(srcRoot, 'App.js'),
    path.join(srcRoot, 'services/useOperatorBackend.ts')
  ];
  const forbiddenCalls = [
    '.setFader(',
    '.setGain(',
    '.setEQ(',
    '.setCompressor(',
    '.routeOutput(',
    '.loadSnap(',
    '.saveSnap('
  ];

  for (const sourcePath of cockpitSources) {
    const source = fs.readFileSync(sourcePath, 'utf8');
    for (const call of forbiddenCalls) {
      assert.equal(source.includes(call), false, `${path.basename(sourcePath)} must not call ${call}`);
    }
  }

  const operatorBackend = fs.readFileSync(path.join(srcRoot, 'services/useOperatorBackend.ts'), 'utf8');
  assert.match(operatorBackend, /applyOperatorProposal\(/);
});
