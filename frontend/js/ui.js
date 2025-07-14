window.addEventListener('DOMContentLoaded', () => {
  const thresholdSlider = document.getElementById('confidence-threshold');
  const thresholdValue = document.getElementById('threshold-value');
  if (thresholdSlider && thresholdValue) {
    thresholdValue.textContent = `${thresholdSlider.value}%`;
    thresholdSlider.addEventListener('input', () => {
      thresholdValue.textContent = `${thresholdSlider.value}%`;
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
    
    statusIndicator.classList.remove('bg-gray-400', 'bg-green-500', 'bg-red-500');
    
    switch (status) {
      case 'connected':
        statusIndicator.classList.add('bg-green-500');
        statusText.textContent = 'Connected';
        break;
      case 'disconnected':
        statusIndicator.classList.add('bg-red-500');
        statusText.textContent = 'Disconnected';
        break;
      default:
        statusIndicator.classList.add('bg-gray-400');
        statusText.textContent = 'Connecting...';
    }
  };
});