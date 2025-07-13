export function createWebSocket({ onOpen, onMessage, onClose, onError }) {
  const protocol = window.location.protocol === 'https:' ? 'wss' : 'ws';
  const url = `${protocol}://${window.location.host}/api/ws/predict`;
  const ws = new WebSocket(url);
  ws.addEventListener('open', (event) => {
    if (onOpen) onOpen(event);
  });
  ws.addEventListener('message', (event) => {
    let data;
    try {
      data = JSON.parse(event.data);
    } catch (e) {
      console.error('Invalid JSON from server:', event.data);
      return;
    }
    if (onMessage) onMessage(data);
  });
  ws.addEventListener('close', (event) => {
    if (onClose) onClose(event);
  });
  ws.addEventListener('error', (event) => {
    if (onError) onError(event);
  });
  return ws;
}