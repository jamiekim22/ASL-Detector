from fastapi import FastAPI
from contextlib import asynccontextmanager
from starlette.middleware.cors import CORSMiddleware
from config import settings
from api.routes import router as api_router
from ml.model_loader import load_model
from loguru import logger

# In-memory log cache
LOG_CACHE: list[str] = []
# Capture INFO+ messages to cache
# Sink receives formatted message, so append msg directly
logger.add(lambda msg: LOG_CACHE.append(msg), level="INFO", format="{message}")

app = FastAPI(
    title=settings.app_name,
    version="0.1.0",
    description="ASL Detector backend API",
    debug=settings.debug
)

# expose in-memory log cache on app state for routes
app.state.LOG_CACHE = LOG_CACHE

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()  # Load the model at startup
    yield
    # clean up code below to clear resources

@app.get("/health")
async def health_check():
    return {"status": "ok"}

app.include_router(api_router)