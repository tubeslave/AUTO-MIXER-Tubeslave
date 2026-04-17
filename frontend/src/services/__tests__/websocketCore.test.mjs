import test from 'node:test';
import assert from 'node:assert/strict';

import { BaseWebSocketTransport } from '../websocketCore.mjs';

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

test('send() serializes messages when socket is open', async () => {
  const transport = createTransport();
  const connectPromise = transport.connect();
  const socket = FakeWebSocket.instances[0];

  socket.open();
  await connectPromise;
  transport.send({ type: 'ping', id: 7 });

  assert.deepEqual(socket.sentMessages, ['{"type":"ping","id":7}']);
});
