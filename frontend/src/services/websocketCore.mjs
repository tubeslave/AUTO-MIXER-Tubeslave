export class BaseWebSocketTransport {
  constructor({
    url = 'ws://localhost:8765',
    reconnectInterval = 3000,
    WebSocketImpl = globalThis.WebSocket
  } = {}) {
    this.ws = null;
    this.url = url;
    this.listeners = new Map();
    this.reconnectInterval = reconnectInterval;
    this.reconnectTimer = null;
    this.connectPromise = null;
    this.shouldReconnect = true;
    this.WebSocketImpl = WebSocketImpl;
  }

  connect() {
    if (this.ws && this.ws.readyState === this.WebSocketImpl.OPEN) {
      return Promise.resolve();
    }

    if (this.connectPromise) {
      return this.connectPromise;
    }

    if (!this.WebSocketImpl) {
      return Promise.reject(new Error('WebSocket API is not available'));
    }

    this.shouldReconnect = true;
    this.connectPromise = new Promise((resolve, reject) => {
      let settled = false;

      try {
        this.ws = new this.WebSocketImpl(this.url);

        this.ws.onopen = () => {
          console.log('WebSocket connected');
          if (this.reconnectTimer) {
            clearTimeout(this.reconnectTimer);
            this.reconnectTimer = null;
          }
          this.connectPromise = null;
          settled = true;
          resolve();
        };

        this.ws.onmessage = (event) => {
          try {
            const data = JSON.parse(event.data);
            console.log('WebSocket message received:', data.type, data);
            this.notifyListeners(data.type, data);
          } catch (error) {
            console.error('Error parsing message:', error);
          }
        };

        this.ws.onerror = (error) => {
          console.error('WebSocket error:', error);
          if (!settled) {
            this.connectPromise = null;
            settled = true;
            reject(error);
          }
        };

        this.ws.onclose = () => {
          console.log('WebSocket disconnected');
          this.ws = null;
          this.connectPromise = null;
          this.notifyListeners('disconnected', {});
          if (this.shouldReconnect) {
            this.scheduleReconnect();
          }
        };
      } catch (error) {
        this.connectPromise = null;
        reject(error);
      }
    });

    return this.connectPromise;
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
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
  }

  send(message) {
    if (this.ws && this.ws.readyState === this.WebSocketImpl.OPEN) {
      console.log('Sending WebSocket message:', message);
      try {
        const jsonMessage = JSON.stringify(message);
        console.log('Sending JSON string:', jsonMessage);
        this.ws.send(jsonMessage);
      } catch (error) {
        console.error('Error sending WebSocket message:', error);
      }
    } else {
      const state = this.ws ? this.ws.readyState : 'null';
      console.error('WebSocket is not connected. ReadyState:', state);
      console.error('WebSocket object:', this.ws);
      console.error('Message that failed to send:', message);
      if (state === 0) {
        console.warn('WebSocket is still connecting...');
      } else if (state === 2 || state === 3) {
        console.warn('WebSocket is closed. Attempting to reconnect...');
        this.connect().catch(err => console.error('Reconnect failed:', err));
      }
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
