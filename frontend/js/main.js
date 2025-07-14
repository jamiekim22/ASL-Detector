import { createWebSocket, CONNECTION_STATES } from './websocket.js';

console.log('main.js loaded');

window.addEventListener('DOMContentLoaded', () => {
  console.log('main.js DOMContentLoaded - initializing ASL recognition');
  
  // DOM elements
  const video = document.getElementById('camera-feed');
  const canvas = document.getElementById('overlay-canvas');
  const toggleBtn = document.getElementById('camera-toggle');
  const thresholdSlider = document.getElementById('confidence-threshold');
  
  // Create a hidden canvas for frame capture (not visible to user)
  const hiddenCanvas = document.createElement('canvas');
  const hiddenCtx = hiddenCanvas.getContext('2d');
  
  // Application state
  let wsManager = null;
  let captureInterval = null;
  let isStreaming = false;
  let frameRate = 5; // fps
  let confidenceThreshold = 70;
  
  // Statistics tracking
  let stats = {
    total: 0,
    confidenceSum: 0,
    sessionStart: Date.now(),
    lastPrediction: null
  };

  // Initialize WebSocket connection
  function initializeWebSocket() {
    wsManager = createWebSocket({
      onStateChange: (state) => {
        console.log('WebSocket state changed:', state);
        window.updateConnectionStatus(state);
        
        // Handle different connection states
        switch (state) {
          case CONNECTION_STATES.CONNECTED:
            window.showUserFeedback('Connected to ASL recognition service', 'success');
            break;
          case CONNECTION_STATES.DISCONNECTED:
            window.showUserFeedback('Disconnected from service', 'error');
            if (isStreaming) {
              stopStreaming();
            }
            break;
          case CONNECTION_STATES.ERROR:
            window.showUserFeedback('Connection error occurred', 'error');
            break;
          case CONNECTION_STATES.RECONNECTING:
            window.showUserFeedback('Attempting to reconnect...', 'warning');
            break;
        }
      },
      
      onMessage: (data) => {
        console.log('Received prediction:', data);
        
        if (data.error) {
          console.error('Prediction error:', data.error);
          window.showUserFeedback(`Prediction error: ${data.error}`, 'error');
          return;
        }
        
        const { sign, confidence } = data;
        
        // Update prediction display
        const shouldAddToHistory = sign !== stats.lastPrediction;
        window.updatePrediction(sign, confidence, {
          confidenceThreshold: confidenceThreshold / 100,
          addToHistory: shouldAddToHistory
        });
        
        // Update statistics
        if (confidence >= confidenceThreshold / 100) {
          stats.total++;
          stats.confidenceSum += confidence;
          stats.lastPrediction = sign;
          
          updateSessionStats();
        }
      },
      
      onError: (error) => {
        console.error('WebSocket error:', error);
        window.showUserFeedback('Connection error occurred', 'error');
      }
    });
  }

  // Frame capture and transmission
  function sendFrame() {
    if (!wsManager || !wsManager.isConnected()) {
      console.warn('WebSocket not connected, skipping frame');
      return;
    }
    
    // Check if video is ready
    if (video.videoWidth === 0 || video.videoHeight === 0) {
      return;
    }
    
    // Use the hidden canvas to capture the frame for sending
    hiddenCanvas.width = video.videoWidth;
    hiddenCanvas.height = video.videoHeight;
    hiddenCtx.drawImage(video, 0, 0);
    
    // Also draw to the visible overlay canvas for user preview (smaller size)
    const previewSize = 150; // Small preview size
    canvas.width = previewSize;
    canvas.height = previewSize;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(video, 0, 0, previewSize, previewSize);
    
    // Add a small green border to indicate frame is being sent
    ctx.strokeStyle = '#10b981'; // green color
    ctx.lineWidth = 3;
    ctx.strokeRect(0, 0, previewSize, previewSize);
    
    // Send the full-size frame to backend
    const dataUrl = hiddenCanvas.toDataURL('image/jpeg', 0.8);
    const success = wsManager.send(dataUrl);
    if (!success) {
      console.warn('Failed to send frame');
    }
  }

  // Start streaming predictions
  function startStreaming() {
    if (isStreaming) return;
    
    if (!wsManager || !wsManager.isConnected()) {
      window.showUserFeedback('Please wait for connection to establish', 'warning');
      return;
    }
    
    console.log('Starting ASL recognition stream');
    captureInterval = setInterval(sendFrame, 1000 / frameRate);
    isStreaming = true;
    
    // Update UI
    updateToggleButton();
    window.showUserFeedback('ASL recognition started', 'success');
    
    // Reset session stats
    stats.sessionStart = Date.now();
    stats.total = 0;
    stats.confidenceSum = 0;
    stats.lastPrediction = null;
    updateSessionStats();
  }

  // Stop streaming predictions
  function stopStreaming() {
    if (!isStreaming) return;
    
    console.log('Stopping ASL recognition stream');
    if (captureInterval) {
      clearInterval(captureInterval);
      captureInterval = null;
    }
    isStreaming = false;
    
    // Clear the preview canvas
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    // Update UI
    updateToggleButton();
    window.showUserFeedback('ASL recognition stopped', 'info');
  }

  // Update toggle button appearance
  function updateToggleButton() {
    if (!toggleBtn) return;
    
    const label = toggleBtn.querySelector('.btn-label');
    if (isStreaming) {
      label.textContent = 'Stop Recognition';
      toggleBtn.classList.remove('btn-start');
      toggleBtn.classList.add('btn-stop');
    } else {
      label.textContent = 'Start Recognition';
      toggleBtn.classList.remove('btn-stop');
      toggleBtn.classList.add('btn-start');
    }
  }

  // Update session statistics
  function updateSessionStats() {
    const sessionTime = (Date.now() - stats.sessionStart) / 1000;
    const avgConfidence = stats.total > 0 ? stats.confidenceSum / stats.total : 0;
    
    window.updateStats({
      total: stats.total,
      avgConfidence: avgConfidence,
      sessionTime: sessionTime
    });
  }

  // Event listeners
  toggleBtn?.addEventListener('click', () => {
    if (isStreaming) {
      stopStreaming();
    } else {
      startStreaming();
    }
  });

  // Settings: confidence threshold
  thresholdSlider?.addEventListener('input', () => {
    confidenceThreshold = parseInt(thresholdSlider.value);
    console.log('Confidence threshold updated:', confidenceThreshold);
  });

  // Settings: frame rate control (you can add this to the settings modal)
  window.setFrameRate = (fps) => {
    frameRate = Math.max(1, Math.min(10, fps)); // Limit between 1-10 fps
    console.log('Frame rate updated:', frameRate);
    
    // Restart streaming with new frame rate if currently streaming
    if (isStreaming) {
      stopStreaming();
      setTimeout(startStreaming, 100);
    }
  };

  // Initialize periodic stats update
  setInterval(updateSessionStats, 1000);

  // Initialize the application
  console.log('Initializing WebSocket connection...');
  initializeWebSocket();
  
  // Initialize UI
  updateToggleButton();
  window.updateConnectionStatus('connecting');
});
