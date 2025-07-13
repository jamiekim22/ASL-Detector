from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
from typing import List
from ..config import settings
import numpy as np
from ..ml.model_loader import load_model
from ..ml.preprocessor import decode_base64_image, preprocess_image

router = APIRouter(prefix="/api")

class PredictRequest(BaseModel):
    image_base64: str

class PredictResponse(BaseModel):
    sign: str
    confidence: float

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
    try:
        img = decode_base64_image(request.image_base64)
        batch = preprocess_image(img)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image data: {e}")
    # run inference
    model = load_model()
    preds = model.predict(batch)
    probs = np.squeeze(preds)
    idx = int(np.argmax(probs))
    confidence = float(probs[idx])
    # TODO: map idx to actual sign label
    sign = str(idx)
    return PredictResponse(sign=sign, confidence=confidence)

@router.get("/metrics", response_model=MetricsResponse, tags=["metrics"])
async def get_metrics():
    # TODO: metrics collection
    return MetricsResponse()

@router.get("/logs", response_model=LogsResponse, tags=["logs"])
async def get_logs():
    # TODO: fetch logs
    return LogsResponse(logs=[])

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
            sign = str(idx)
            await websocket.send_json({"sign": sign, "confidence": confidence})
    except WebSocketDisconnect:
        pass