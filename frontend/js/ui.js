window.addEventListener('DOMContentLoaded', () => {
  const thresholdSlider = document.getElementById('confidence-threshold');
  const thresholdValue = document.getElementById('threshold-value');
  const frameRateSlider = document.getElementById('frame-rate');
  const frameRateValue = document.getElementById('frame-rate-value');
  
  // Confidence threshold slider
  if (thresholdSlider && thresholdValue) {
    thresholdValue.textContent = `${thresholdSlider.value}%`;
    thresholdSlider.addEventListener('input', () => {
      thresholdValue.textContent = `${thresholdSlider.value}%`;
    });
  }

  // Frame rate slider
  if (frameRateSlider && frameRateValue) {
    frameRateValue.textContent = `${frameRateSlider.value} FPS`;
    frameRateSlider.addEventListener('input', () => {
      const fps = frameRateSlider.value;
      frameRateValue.textContent = `${fps} FPS`;
      
      // Update frame rate in main app if function exists
      if (window.setFrameRate) {
        window.setFrameRate(parseInt(fps));
      }
    });
  }

  // Modal controls
  const settingsBtn = document.getElementById('settings-toggle');
  const settingsModal = document.getElementById('settingsModal');
  const closeModalBtn = document.getElementById('close-modal');
  const cancelBtn = document.getElementById('cancel-settings');

  if (settingsBtn && settingsModal) {
    settingsBtn.addEventListener('click', () => {
      settingsModal.style.display = 'flex';
    });
  }

  [closeModalBtn, cancelBtn].forEach(btn => {
    if (btn) {
      btn.addEventListener('click', () => {
        settingsModal.style.display = 'none';
      });
    }
  });

  // Close modal when clicking outside
  settingsModal?.addEventListener('click', (e) => {
    if (e.target === settingsModal) {
      settingsModal.style.display = 'none';
    }
  });

  // Clear history button
  const clearHistoryBtn = document.getElementById('clear-history');
  const letterHistory = document.getElementById('letter-history');
  if (clearHistoryBtn && letterHistory) {
    clearHistoryBtn.addEventListener('click', () => {
      letterHistory.innerHTML = '';
    });
  }

  // Status indicator updates based on connection
  const statusIndicator = document.getElementById('status-indicator');
  const statusText = document.getElementById('status-text');
  
  // Helper functions for other scripts to update status
  window.updateConnectionStatus = (status) => {
    if (!statusIndicator || !statusText) return;
    
    statusIndicator.classList.remove('bg-gray-400', 'bg-green-500', 'bg-red-500', 'bg-yellow-500');
    
    switch (status) {
      case 'connected':
        statusIndicator.classList.add('bg-green-500');
        statusText.textContent = 'Connected';
        break;
      case 'disconnected':
        statusIndicator.classList.add('bg-red-500');
        statusText.textContent = 'Disconnected';
        break;
      case 'error':
        statusIndicator.classList.add('bg-red-500');
        statusText.textContent = 'Connection Error';
        break;
      case 'reconnecting':
        statusIndicator.classList.add('bg-yellow-500');
        statusText.textContent = 'Reconnecting...';
        break;
      default:
        statusIndicator.classList.add('bg-gray-400');
        statusText.textContent = 'Connecting...';
    }
  };

  // Enhanced prediction display with animations and confidence threshold
  window.updatePrediction = (sign, confidence, options = {}) => {
    const currentPrediction = document.getElementById('current-prediction');
    const confidenceBar = document.getElementById('confidence-bar');
    const confidenceText = document.getElementById('confidence-text');
    const letterHistory = document.getElementById('letter-history');
    
    if (!currentPrediction || !confidenceBar || !confidenceText) return;

    const percent = Math.round(confidence * 100);
    const threshold = options.confidenceThreshold || 70;
    
    // Only update display if confidence meets threshold
    if (percent >= threshold) {
      // Animate prediction change
      currentPrediction.style.opacity = '0.5';
      setTimeout(() => {
        currentPrediction.textContent = sign;
        currentPrediction.style.opacity = '1';
      }, 100);
      
      // Update confidence bar with animation
      confidenceBar.style.width = `${percent}%`;
      confidenceBar.style.backgroundColor = getConfidenceColor(percent);
      confidenceText.textContent = `Confidence: ${percent}%`;
      
      // Add to history if it's a new prediction
      if (options.addToHistory && sign !== '-') {
        addToHistory(sign, confidence);
      }
    } else {
      // Show low confidence state
      currentPrediction.textContent = '-';
      confidenceBar.style.width = `${percent}%`;
      confidenceBar.style.backgroundColor = '#e5e7eb';
      confidenceText.textContent = `Confidence: ${percent}% (Too Low)`;
    }
  };

  // Enhanced stats update function
  window.updateStats = (stats) => {
    const totalPredictionsEl = document.getElementById('total-predictions');
    const avgConfidenceEl = document.getElementById('avg-confidence');
    const sessionTimeEl = document.getElementById('session-time');
    
    if (totalPredictionsEl) {
      totalPredictionsEl.textContent = stats.total || 0;
    }
    
    if (avgConfidenceEl) {
      const avgPercent = Math.round((stats.avgConfidence || 0) * 100);
      avgConfidenceEl.textContent = `${avgPercent}%`;
    }
    
    if (sessionTimeEl) {
      const totalSeconds = Math.floor(stats.sessionTime || 0);
      const mins = String(Math.floor(totalSeconds / 60)).padStart(2, '0');
      const secs = String(totalSeconds % 60).padStart(2, '0');
      sessionTimeEl.textContent = `${mins}:${secs}`;
    }
  };

  // Helper function to get confidence color
  function getConfidenceColor(percent) {
    if (percent >= 90) return '#10b981'; // green-500
    if (percent >= 75) return '#f59e0b'; // amber-500
    if (percent >= 60) return '#ef4444'; // red-500
    return '#6b7280'; // gray-500
  }

  // Helper function to add prediction to history
  function addToHistory(sign, confidence) {
    const letterHistory = document.getElementById('letter-history');
    if (!letterHistory) return;
    
    const span = document.createElement('span');
    span.textContent = sign;
    span.title = `Confidence: ${Math.round(confidence * 100)}%`;
    
    const confidenceClass = confidence >= 0.9 ? 'bg-green-100 text-green-800' :
                           confidence >= 0.75 ? 'bg-yellow-100 text-yellow-800' :
                           'bg-red-100 text-red-800';
    
    span.classList.add('inline-flex', 'items-center', 'px-2', 'py-1', 'rounded-full', 
                       'text-xs', 'font-medium', confidenceClass, 'cursor-pointer');
    
    // Add click to remove functionality
    span.addEventListener('click', () => {
      span.remove();
    });
    
    // Add with fade-in animation
    span.style.opacity = '0';
    letterHistory.appendChild(span);
    setTimeout(() => {
      span.style.opacity = '1';
      span.style.transition = 'opacity 0.3s ease-in';
    }, 10);
    
    // Limit history to last 20 items
    const children = letterHistory.children;
    if (children.length > 20) {
      children[0].remove();
    }
  }

  // Enhanced error handling and user feedback
  window.showUserFeedback = (message, type = 'info', duration = 3000) => {
    // Create feedback element if it doesn't exist
    let feedbackEl = document.getElementById('user-feedback');
    if (!feedbackEl) {
      feedbackEl = document.createElement('div');
      feedbackEl.id = 'user-feedback';
      feedbackEl.className = 'fixed top-4 right-4 z-50 max-w-sm';
      document.body.appendChild(feedbackEl);
    }
    
    const alertEl = document.createElement('div');
    const typeClasses = {
      success: 'bg-green-50 border-green-200 text-green-800',
      error: 'bg-red-50 border-red-200 text-red-800',
      warning: 'bg-yellow-50 border-yellow-200 text-yellow-800',
      info: 'bg-blue-50 border-blue-200 text-blue-800'
    };
    
    alertEl.className = `p-4 border rounded-lg shadow-lg ${typeClasses[type] || typeClasses.info}`;
    alertEl.textContent = message;
    
    feedbackEl.appendChild(alertEl);
    
    // Auto remove after duration
    setTimeout(() => {
      alertEl.remove();
    }, duration);
  };
});