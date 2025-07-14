console.log('main.js loaded');

window.addEventListener('DOMContentLoaded', () => {
  console.log('main.js DOMContentLoaded - initializing prediction loop');
  const video = document.getElementById('camera-feed');
  const canvas = document.getElementById('overlay-canvas');
  // Toggle camera start/stop button
  const toggleBtn = document.getElementById('camera-toggle');
  const statusIndicator = document.getElementById('status-indicator');
  const statusText = document.getElementById('status-text');
  const currentPrediction = document.getElementById('current-prediction');
  const confidenceBar = document.getElementById('confidence-bar');
  const confidenceText = document.getElementById('confidence-text');
  const letterHistory = document.getElementById('letter-history');
  const totalPredictionsEl = document.getElementById('total-predictions');
  const avgConfidenceEl = document.getElementById('avg-confidence');
  const sessionTimeEl = document.getElementById('session-time');

  let ws;
  let captureInterval;
  let isStreaming = false;
  let predictionsCount = 0;
  let confidenceSum = 0;
  const sessionStart = Date.now();

  function connectWebSocket() {
    const scheme = window.location.protocol === 'https:' ? 'wss' : 'ws';
    
    // Connect to backend WebSocket on port 8000
    const host = window.location.hostname;
    const wsUrl = `${scheme}://${host}:8000/api/ws/predict`;
    console.log('Connecting to WebSocket:', wsUrl);
    ws = new WebSocket(wsUrl);

    ws.addEventListener('open', () => {
      console.log('WebSocket opened');
      if (window.updateConnectionStatus) {
        window.updateConnectionStatus('connected');
      }
    });

    ws.addEventListener('message', (event) => {
      console.log('WebSocket message:', event.data);
      const data = JSON.parse(event.data);
      if (data.error) {
        console.error('Prediction error:', data.error);
        return;
      }
      updatePrediction(data.sign, data.confidence);
    });

    ws.addEventListener('close', (event) => {
      console.log('WebSocket closed', event);
      if (window.updateConnectionStatus) {
        window.updateConnectionStatus('disconnected');
      }
    });
    ws.addEventListener('error', (event) => {
      console.error('WebSocket error', event);
    });
  }

  function sendFrame() {
    console.log('sendFrame called');
    if (!ws || ws.readyState !== WebSocket.OPEN) return;
    // Skip frame if video metadata not yet loaded
    if (video.videoWidth === 0 || video.videoHeight === 0) {
      return;
    }
    const ctx = canvas.getContext('2d');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    ctx.drawImage(video, 0, 0);
    const dataUrl = canvas.toDataURL('image/jpeg');
    ws.send(dataUrl);
    console.log('Frame sent');
  }

  function updatePrediction(sign, confidence) {
    currentPrediction.textContent = sign;
    const percent = Math.round(confidence * 100);
    confidenceBar.style.width = `${percent}%`;
    confidenceText.textContent = `Confidence: ${percent}%`;
    // update history
    const span = document.createElement('span');
    span.textContent = sign;
    span.classList.add('inline-flex', 'items-center', 'px-2', 'py-1', 'rounded-full', 'text-xs', 'font-medium', 'bg-blue-100', 'text-blue-800');
    letterHistory.appendChild(span);
    // update stats
    predictionsCount++;
    confidenceSum += confidence;
    totalPredictionsEl.textContent = predictionsCount;
    avgConfidenceEl.textContent = `${Math.round((confidenceSum / predictionsCount) * 100)}%`;
    // update session time
    const elapsed = Date.now() - sessionStart;
    const mins = String(Math.floor(elapsed / 60000)).padStart(2, '0');
    const secs = String(Math.floor((elapsed % 60000) / 1000)).padStart(2, '0');
    sessionTimeEl.textContent = `${mins}:${secs}`;
  }

  connectWebSocket();
  // Handle start/stop toggle
  toggleBtn.addEventListener('click', () => {
    if (!isStreaming) {
      // Start streaming
      captureInterval = setInterval(sendFrame, 200); // 5 fps
      toggleBtn.querySelector('.btn-label').textContent = 'Stop Camera';
      toggleBtn.classList.remove('btn-start');
      toggleBtn.classList.add('btn-stop');
      isStreaming = true;
    } else {
      // Stop streaming
      clearInterval(captureInterval);
      toggleBtn.querySelector('.btn-label').textContent = 'Start Camera';
      toggleBtn.classList.remove('btn-stop');
      toggleBtn.classList.add('btn-start');
      isStreaming = false;
    }
  });
});
