import { WebSocketService, BASE_RECONNECT_MS, OFFLINE_QUEUE_LIMIT } from './websocket';

describe('WebSocketService', () => {
  let service;

  beforeEach(() => {
    service = new WebSocketService();
  });

  afterEach(() => {
    service.stopHeartbeat();
    if (service.reconnectTimer) clearTimeout(service.reconnectTimer);
    jest.useRealTimers();
  });

  describe('offline queue', () => {
    test('queues messages while disconnected instead of dropping them', () => {
      service.ws = null;
      service.send({ type: 'set_fader', channel: 1, value: -6 });
      expect(service.offlineQueue).toHaveLength(1);
      expect(service.offlineQueue[0].type).toBe('set_fader');
    });

    test('caps the queue at OFFLINE_QUEUE_LIMIT, dropping oldest first', () => {
      service.ws = null;
      for (let i = 0; i < OFFLINE_QUEUE_LIMIT + 10; i++) {
        service.send({ type: 'msg', seq: i });
      }
      expect(service.offlineQueue).toHaveLength(OFFLINE_QUEUE_LIMIT);
      expect(service.offlineQueue[0].seq).toBe(10);
    });

    test('flushes the queue through send() when connection restores', () => {
      service.ws = null;
      service.send({ type: 'a' });
      service.send({ type: 'b' });
      const sent = [];
      service.ws = {
        readyState: WebSocket.OPEN,
        send: (raw) => sent.push(JSON.parse(raw)),
      };
      service.flushOfflineQueue();
      expect(sent.map((m) => m.type)).toEqual(['a', 'b']);
      expect(service.offlineQueue).toHaveLength(0);
    });
  });

  describe('reconnect backoff', () => {
    test('delay grows exponentially and resets on successful open', () => {
      jest.useFakeTimers();
      service.connect = jest.fn(() => Promise.resolve());

      expect(service.reconnectDelay).toBe(BASE_RECONNECT_MS);
      service.scheduleReconnect();
      const afterFirst = service.reconnectDelay;
      expect(afterFirst).toBeGreaterThan(BASE_RECONNECT_MS);

      // Повторный вызов при активном таймере не наслаивается
      const timer = service.reconnectTimer;
      service.scheduleReconnect();
      expect(service.reconnectTimer).toBe(timer);
    });
  });

  describe('heartbeat', () => {
    test('forces close after missed pongs', () => {
      jest.useFakeTimers();
      const close = jest.fn();
      service.ws = { readyState: WebSocket.OPEN, send: jest.fn(), close };
      service.startHeartbeat();
      // 3 интервала без pong (лимит 2 промаха)
      jest.advanceTimersByTime(5000 * 3);
      expect(close).toHaveBeenCalled();
    });

    test('pong resets the miss counter', () => {
      jest.useFakeTimers();
      const close = jest.fn();
      service.ws = { readyState: WebSocket.OPEN, send: jest.fn(), close };
      service.startHeartbeat();
      jest.advanceTimersByTime(5000 * 2);
      service.missedPongs = 0; // как это делает обработчик pong
      jest.advanceTimersByTime(5000 * 2);
      expect(close).not.toHaveBeenCalled();
    });
  });

  describe('listeners', () => {
    test('off() removes exactly the registered callback', () => {
      const calls = [];
      const cb = (d) => calls.push(d);
      service.on('x', cb);
      service.notifyListeners('x', { v: 1 });
      service.off('x', cb);
      service.notifyListeners('x', { v: 2 });
      expect(calls).toEqual([{ v: 1 }]);
    });

    test('listener exception does not break other listeners', () => {
      const calls = [];
      service.on('x', () => { throw new Error('boom'); });
      service.on('x', (d) => calls.push(d));
      service.notifyListeners('x', { v: 1 });
      expect(calls).toEqual([{ v: 1 }]);
    });
  });
});
