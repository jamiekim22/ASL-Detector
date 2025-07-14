const video = document.getElementById('camera-feed');
const select = document.getElementById('camera-select');
const toggleBtn = document.getElementById('camera-toggle');
let currentStream = null;

// List available video input devices
async function listCameras() {
  console.log('Listing cameras...');
  try {
    const devices = await navigator.mediaDevices.enumerateDevices();
    const videoDevices = devices.filter((d) => d.kind === 'videoinput');
    console.log(`Found ${videoDevices.length} video input devices`);
    populateCameraSelect(videoDevices);
  } catch (err) {
    console.error('Error enumerating devices:', err);
  }
}

function populateCameraSelect(devices) {
  console.log('Populating camera select with devices:', devices);
  select.innerHTML = '';
  devices.forEach((device, idx) => {
    const option = document.createElement('option');
    option.value = device.deviceId;
    option.text = device.label || `Camera ${idx + 1}`;
    select.appendChild(option);
  });
}

// Start video stream from selected camera
async function startCamera() {
  console.log('Start camera clicked, selected device:', select.value);
  if (currentStream) {
    console.log('Camera already running, ignoring startCamera');
    return;
  }
  const deviceId = select.value;
  const constraints = deviceId
    ? { video: { deviceId: { exact: deviceId } } }
    : { video: true };
  try {
    currentStream = await navigator.mediaDevices.getUserMedia(constraints);
    video.srcObject = currentStream;
    // Update toggle button to Stop state
    toggleBtn.querySelector('.btn-label').textContent = 'Stop Camera';
    toggleBtn.classList.remove('btn-start');
    toggleBtn.classList.add('btn-stop');
  } catch (err) {
    console.error('Error accessing camera:', err);
  }
}

// Stop the video stream
function stopCamera() {
  console.log('Stop camera clicked');
  if (!currentStream) {
    console.log('No current stream to stop');
    return;
  }
  currentStream.getTracks().forEach((track) => track.stop());
  video.srcObject = null;
  currentStream = null;
  // Update toggle button to Start state
  toggleBtn.querySelector('.btn-label').textContent = 'Start Camera';
  toggleBtn.classList.remove('btn-stop');
  toggleBtn.classList.add('btn-start');
}

// Wire up event listeners on page load
window.addEventListener('DOMContentLoaded', () => {
  console.log('DOM loaded, initializing camera controls');
  if (!navigator.mediaDevices) {
    console.error('WebRTC not supported in this browser.');
    toggleBtn.disabled = true;
    return;
  }
  listCameras();
  // Toggle camera on button click
  toggleBtn.addEventListener('click', () => {
    if (!currentStream) {
      startCamera();
    } else {
      stopCamera();
    }
  });
});
