from pydantic_settings import BaseSettings
from pathlib import Path

model_path: str = "models/asl_model.h5"

class Settings(BaseSettings):
    app_name: str = "ASL Detector API"
    debug: bool = False
    cors_origins: list[str] = ["*"]
    model_path: str = model_path

    class Config:
        env_file = ".env"
        case_sensitive = False

settings = Settings()