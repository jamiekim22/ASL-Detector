// WebSocket connection states
export const CONNECTION_STATES = {
  CONNECTING: 'connecting',
  CONNECTED: 'connected',
  DISCONNECTED: 'disconnected',
  ERROR: 'error',
  RECONNECTING: 'reconnecting'
};

class WebSocketManager {
  constructor(options = {}) {
    this.options = {
      maxReconnectAttempts: 5,
      reconnectInterval: 3000,
      ...options
    };
    
    this.ws = null;
    this.reconnectAttempts = 0;
    this.reconnectTimer = null;
    this.isManualClose = false;
    
    // Event handlers
    this.onOpen = options.onOpen || (() => {});
    this.onMessage = options.onMessage || (() => {});
    this.onClose = options.onClose || (() => {});
    this.onError = options.onError || (() => {});
    this.onStateChange = options.onStateChange || (() => {});
  }

  connect() {
    const protocol = window.location.protocol === 'https:' ? 'wss' : 'ws';
    const host = window.location.hostname;
    const port = window.location.hostname === 'localhost' ? ':8000' : '';
    const url = `${protocol}://${host}${port}/api/ws/predict`;
    
    console.log('Connecting to WebSocket:', url);
    this.onStateChange(CONNECTION_STATES.CONNECTING);
    
    this.ws = new WebSocket(url);
    
    this.ws.addEventListener('open', (event) => {
      console.log('WebSocket connection opened');
      this.reconnectAttempts = 0;
      this.onStateChange(CONNECTION_STATES.CONNECTED);
      this.onOpen(event);
    });

    this.ws.addEventListener('message', (event) => {
      let data;
      try {
        data = JSON.parse(event.data);
      } catch (e) {
        console.error('Invalid JSON from server:', event.data);
        return;
      }
      
      this.onMessage(data);
    });

    this.ws.addEventListener('close', (event) => {
      console.log('WebSocket connection closed:', event.code, event.reason);
      
      if (!this.isManualClose && this.shouldReconnect()) {
        this.scheduleReconnect();
      } else {
        this.onStateChange(CONNECTION_STATES.DISCONNECTED);
      }
      
      this.onClose(event);
    });

    this.ws.addEventListener('error', (event) => {
      console.error('WebSocket error:', event);
      this.onStateChange(CONNECTION_STATES.ERROR);
      this.onError(event);
    });
  }

  disconnect() {
    this.isManualClose = true;
    this.clearReconnectTimer();
    
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.close();
    }
  }

  send(data) {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      if (typeof data === 'object') {
        this.ws.send(JSON.stringify(data));
      } else {
        this.ws.send(data);
      }
      return true;
    }
    return false;
  }

  sendFrame(canvas) {
    if (!this.isConnected()) return false;
    
    try {
      const dataUrl = canvas.toDataURL('image/jpeg', 0.8);
      return this.send(dataUrl);
    } catch (error) {
      console.error('Error sending frame:', error);
      return false;
    }
  }

  isConnected() {
    return this.ws && this.ws.readyState === WebSocket.OPEN;
  }

  shouldReconnect() {
    return this.reconnectAttempts < this.options.maxReconnectAttempts;
  }

  scheduleReconnect() {
    if (this.reconnectTimer) return;
    
    this.reconnectAttempts++;
    this.onStateChange(CONNECTION_STATES.RECONNECTING);
    
    console.log(`Scheduling reconnect attempt ${this.reconnectAttempts}/${this.options.maxReconnectAttempts}`);
    
    this.reconnectTimer = setTimeout(() => {
      this.reconnectTimer = null;
      this.isManualClose = false;
      this.connect();
    }, this.options.reconnectInterval);
  }

  clearReconnectTimer() {
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }
  }
}

// Factory function for backwards compatibility
export function createWebSocket(options) {
  const manager = new WebSocketManager(options);
  manager.connect();
  return manager;
}