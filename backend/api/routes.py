from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
from typing import List
from ..config import settings
import numpy as np
import time
from loguru import logger
from ..ml.model_loader import load_model
from ..ml.preprocessor import decode_base64_image, preprocess_image
from ..ml.mediapipe_handler import detect_hand_roi
from ..ml.model_loader import idx_to_label
from ..api.metrics_store import metrics_store
from ..app import LOG_CACHE

router = APIRouter(prefix="/api")

class PredictRequest(BaseModel):
    image_base64: str

class PredictResponse(BaseModel):
    sign: str
    confidence: float
    timestamp: float

class ModelInfoResponse(BaseModel):
    model_path: str

class MetricsResponse(BaseModel):
    inference_count: int = 0
    average_latency_ms: float = 0.0

class LogsResponse(BaseModel):
    logs: List[str]

class ConfigResponse(BaseModel):
    app_name: str
    debug: bool
    cors_origins: List[str]
    model_path: str

@router.get("/model/info", response_model=ModelInfoResponse, tags=["model"])
async def model_info():
    return ModelInfoResponse(model_path=settings.model_path)

@router.post("/predict", response_model=PredictResponse, tags=["prediction"])
async def predict(request: PredictRequest):
    """
    Predicts sign from base64-encoded image
    """
    start_time = time.time()
    logger.info("Predict endpoint called")
    try:
        img = decode_base64_image(request.image_base64)
        hand_img = detect_hand_roi(img)
        batch = preprocess_image(hand_img)
    except Exception as e:
        logger.error(f"Error processing image: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    model = load_model()
    preds = model.predict(batch)
    probs = np.squeeze(preds)
    idx = int(np.argmax(probs))
    confidence = float(probs[idx])
    
    sign = idx_to_label(idx) # map idx to actual sign label
    latency_ms = (time.time() - start_time) * 1000
    metrics_store.record(latency_ms)
    timestamp = start_time
    logger.info(f"Prediction: {sign} ({confidence:.2f}), latency={latency_ms:.1f}ms")
    return PredictResponse(sign=sign, confidence=confidence, timestamp=timestamp)

@router.get("/metrics", response_model=MetricsResponse, tags=["metrics"])
async def get_metrics():
    """Return inference count and average latency."""
    m = metrics_store.get_metrics()
    return MetricsResponse(inference_count=m["inference_count"], average_latency_ms=m["average_latency_ms"])

@router.get("/logs", response_model=LogsResponse, tags=["logs"])
async def get_logs():
    """
    Return the 100 most recent log messages captured in-memory.
    """
    return LogsResponse(logs=LOG_CACHE[-100:])

@router.get("/config", response_model=ConfigResponse, tags=["config"])
async def get_config():
    return ConfigResponse(
        app_name=settings.app_name,
        debug=settings.debug,
        cors_origins=settings.cors_origins,
        model_path=settings.model_path
    )

@router.websocket("/ws/predict")
async def ws_predict(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            try:
                img = decode_base64_image(data)
                batch = preprocess_image(img)
            except Exception as e:
                await websocket.send_json({"error": f"Invalid image data: {e}"})
                continue
            model = load_model()
            preds = model.predict(batch)
            probs = np.squeeze(preds)
            idx = int(np.argmax(probs))
            confidence = float(probs[idx])
            sign = idx_to_label(idx)
            await websocket.send_json({"sign": sign, "confidence": confidence})
    except WebSocketDisconnect:
        pass