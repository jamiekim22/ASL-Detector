import glob
import base64
import os

import pytest
from fastapi.testclient import TestClient

from backend.app import app
from backend.config import settings

client = TestClient(app)

# fixture to bypass MediaPipe hand detection for tests (cuz pre-cropped)
@pytest.fixture(autouse=True)
def bypass_hand_detection(monkeypatch):
    monkeypatch.setattr('backend.api.routes.detect_hand_roi', lambda img: img)

# Collect all test images
TEST_DIR = os.path.join(os.path.dirname(__file__), '..', 'ml_development', 'data', 'asl_test')
IMAGE_FILES = [f for f in glob.glob(os.path.join(TEST_DIR, '*')) if f.lower().endswith(('.jpg', '.png'))]

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_model_info():
    response = client.get("/api/model/info")
    assert response.status_code == 200
    data = response.json()
    assert "model_path" in data
    assert isinstance(data["model_path"], str)

def test_predict_invalid_image():
    response = client.post("/api/predict", json={"image_base64": "not-base64"})
    assert response.status_code == 400
    assert "Invalid" in response.json().get("detail", "")

# Use websocket to send invalid image
def test_ws_predict_invalid(monkeypatch):
    with client.websocket_connect("/api/ws/predict") as websocket:
        websocket.send_text("not-base64")
        data = websocket.receive_json()
        assert "error" in data

@pytest.mark.parametrize('img_path', IMAGE_FILES)
def test_predict_all_letters(img_path):
    settings.model_path = os.path.join(os.path.dirname(__file__), '..', 'ml_development', 'asl_model.h5')

    with open(img_path, 'rb') as f:
        encoded = base64.b64encode(f.read()).decode('utf-8')
    prefix = 'data:image/jpeg;base64,' if img_path.lower().endswith('.jpg') else 'data:image/png;base64,'
    response = client.post('/api/predict', json={'image_base64': prefix + encoded})
    assert response.status_code == 200
    data = response.json()

    expected = os.path.splitext(os.path.basename(img_path))[0].split('_')[0]
    print(f"Tested {os.path.basename(img_path)} -> Predicted: {data.get('sign')} (Expected: {expected})")
    assert data.get('sign') == expected

def test_metrics_endpoint():
    response = client.get("/api/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "inference_count" in data and "average_latency_ms" in data
